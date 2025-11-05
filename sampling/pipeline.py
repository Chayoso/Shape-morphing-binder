"""
SDF-based Point Cloud Upsampling Pipeline.

Pipeline:
  STAGE 1: Build SDF grid from volumetric points
  STAGE 2: Extract surface anchors
  STAGE 3: Importance sampling
  STAGE 4: SDF projection + tangent smoothing
  (Optional) Normal smoothing (disabled by default)
  STAGE 5: Covariance construction

Author: CHAYO
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from pathlib import Path

from .utils.config import default_cfg, validate_cfg
from .utils.utils import ensure_torch, as_numpy
from .utils.validation import check_config_and_warn
from .analysis.knn import HybridFAISSKNN, FAISS_AVAILABLE
# 🔥 Level Set pipeline
from .analysis.levelset import (
    LevelSetGrid,
    extract_surface_anchors,
)
from .core.sampler import sample_points
from .core.levelset_ops import tangent_smooth_then_project, project_zero, grad_normals, align_normals_local_majority, voxel_quota_drop, quick_diagnostics
from .geometry.deformation_covariance import build_deformation_covariance
from .geometry.curvature_covariance import create_curvature_based_covariance_star
from .geometry.learnable_covariance import LearnableCovariance


# ============================================================================
# 🔥 PCA 기반 Normal 계산
# ============================================================================
@torch.no_grad()
def compute_normals_pca(points: torch.Tensor, knn, k: int = 32, prefer_outward: bool = True) -> torch.Tensor:
    """
    PCA로 local surface normal 계산 + orientation consistency.
    
    Args:
        points: (N, 3) point positions
        knn: KNN 함수
        k: 이웃 개수
        prefer_outward: True면 바깥쪽 향하도록 (centroid 기준)
    
    Returns:
        normals: (N, 3) unit normals (바깥쪽 향함)
    """
    k = min(k, points.shape[0])
    
    # KNN search
    idx, _ = knn(points, points, k)
    neighbors = points[idx]  # (N, k, 3)
    
    # Center
    centroid = neighbors.mean(dim=1, keepdim=True)  # (N, 1, 3)
    centered = neighbors - centroid  # (N, k, 3)
    
    # Covariance matrix
    C = torch.einsum('nki,nkj->nij', centered, centered) / k  # (N, 3, 3)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(C)  # eigenvalues are sorted ascending
    
    # Normal = eigenvector with smallest eigenvalue (most planar direction)
    normals = eigenvectors[:, :, 0]  # (N, 3)
    
    # Normalize
    normals = F.normalize(normals, dim=-1, eps=1e-6)
    
    # 🔥 Orient outward: point away from global centroid
    if prefer_outward:
        global_centroid = points.mean(dim=0)  # (3,)
        to_point = points - global_centroid  # (N, 3) vector from center to point
        
        # If normal points inward (towards centroid), flip it
        dot = (normals * to_point).sum(dim=-1)  # (N,)
        flip = dot < 0
        normals[flip] = -normals[flip]
        
        # 🔥 Ensure all normals point upward (positive Z component)
        # This forces even bottom surfaces to point "outward/upward"
        pointing_down = normals[:, 2] < 0
        if pointing_down.any():
            normals[pointing_down] = -normals[pointing_down]
    
    return normals


# ============================================================================
# ⚡ NaN/Inf Health Check
# ============================================================================
def _check_finite(tag: str, *tensors):
    """⚡ Fast NaN/Inf health check at stage boundaries."""
    for t in tensors:
        if not torch.isfinite(t).all():
            raise FloatingPointError(f"[{tag}] non-finite detected")


# ============================================================================
# ⚡ Identity Shortcut α Schedule (morphing drift suppression)
# ============================================================================
def alpha_schedule(ep: int) -> float:
    """
    ⚡ α 스케줄: 초반 강한 gradient, 후반 0으로 수렴.
    
    - ep < 3:  0.05 (strong connection)
    - ep < 8:  0.03
    - ep < 15: 0.015
    - ep >= 15: 0.0 (no drift)
    """
    if ep < 3:
        return 0.05
    elif ep < 8:
        return 0.03
    elif ep < 15:
        return 0.015
    else:
        return 0.0


# ============================================================================
# SDF Visualization Helper (Import from stage_viz)
# ============================================================================
from .io.stage_viz import (
    _visualize_input_points,
    _visualize_sdf_grid,
    _visualize_anchors,
    _visualize_sampled_points,
    _visualize_projected_points,
    _visualize_normals,
    _visualize_curvature_data,
)


def upsample(
    x_low: torch.Tensor,
    F_low: torch.Tensor,
    cfg: Optional[Dict] = None,
    state: Optional[Dict] = None,
    seed: int = 1234,
    return_torch: bool = True,
    export_stages: bool = False,
    learnable_cov_module=None,  # Optional learnable covariance module
    current_episode: int = -1,  # Current episode number for per-episode visualization
    external_levelset=None,  # Pre-computed level set (ignored, for API compatibility)
    use_simple_pipeline: bool = True  # 🔥 NEW: Direct F→Cov without upsampling
) -> Dict:
    """
    Main differentiable point cloud upsampling pipeline.
    
    Transforms sparse point cloud with deformation gradients into dense
    point cloud with anisotropic covariances for high-quality rendering.
    
    Pipeline stages:
        1. Surface Detection: PCA-based planarity analysis
        2. Anchor-Density Map: Differentiable density estimation
        3. Importance Sampling: Gumbel-Softmax with tangent jitter
        4. Taubin Smoothing: Shrinkage-free Laplacian smoothing
        5. Normal Smoothing: Spatial Laplacian with adaptive bandwidth
        6. Covariance Construction: Deformation-aware via F-field
    
    All operations are fully differentiable, enabling:
        - Learning importance sampling distribution
        - Joint optimization with neural networks
        - End-to-end surface reconstruction
    
    Typical workflow:
        ```python
        # Coarse point cloud from simulation or reconstruction
        x_low = torch.randn(10000, 3, requires_grad=True)
        F_low = torch.eye(3).expand(10000, 3, 3).clone()
        
        # Upsample with default configuration
        result = upsample(x_low, F_low)
        
        # Extract outputs
        points = result['points']      # (100k, 3) dense positions
        normals = result['normals']    # (100k, 3) smooth normals
        cov = result['cov']            # (100k, 3, 3) anisotropic covariances
        
        # Use for rendering (e.g., 3D Gaussian Splatting)
        render_loss = gaussian_splatting(points, normals, cov, gt_image)
        render_loss.backward()  # Gradients flow to x_low, F_low
        ```
    
    Args:
        x_low: (N, 3) coarse anchor point positions
               - Typically from downsampling, simulation, or sparse reconstruction
               - Can be torch.Tensor or numpy array (will be converted)
               - Can have requires_grad=True for position optimization
               - Typical N: 5k-20k (upsamples to 50k-200k)
        
        F_low: (N, 3, 3) deformation gradient tensors at anchors
               - Describes local affine transformation around each point
               - Identity F=I means no deformation (isotropic)
               - Can be learned from neural network or physical simulation
               - Used to create anisotropic covariances in final output
        
        cfg: Configuration dictionary (optional)
             - If None, uses default_cfg() with sensible defaults
             - Override specific parameters as needed
             - Structure: nested dict with keys for each stage
             - Example:
               ```python
               cfg = {
                   'sampling': {'M': 100000, 'tau': 0.2},
                   'taubin': {'enabled': True, 'iters': 3}
               }
               ```
        
        state: State dictionary for EMA and caching (optional)
               - If None, initializes empty state
               - Updated in-place during pipeline execution
               - Used for temporal stability (e.g., surface detection EMA)
               - Pass same state across multiple frames for video
        
        seed: Random seed for reproducibility
              - Controls Gumbel-Softmax sampling and jittering
              - Use different seeds for data augmentation
              - Default 1234 is arbitrary but consistent
        
        return_torch: Return format (default: True)
                      - True: return torch.Tensor (GPU-friendly)
                      - False: return numpy.ndarray (CPU-friendly)
        
        learnable_cov_module: Optional learnable covariance module
                             - For morphing episodes with hybrid covariance
                             - If None, uses state-cached module or physics-only
    
    Returns:
        result: Dictionary containing:
            
            **Main outputs:**
            - points: (M, 3) upsampled point positions
                      * Dense point cloud after all processing
                      * Smoothed and surface-aligned
                      * Differentiable w.r.t. x_low
            
            - normals: (M, 3) surface normals
                       * Unit vectors perpendicular to local surface
                       * Spatially smoothed for consistency
                       * Differentiable w.r.t. x_low
            
            - cov: (M, 3, 3) anisotropic covariance matrices
                   * Defines ellipsoidal Gaussian shape per point
                   * Aligned with local deformation (from F_low)
                   * Symmetric positive definite
                   * For 3D Gaussian Splatting rendering
            
            - F_interp: (M, 3, 3) interpolated deformation gradients
                        * Smoothed and interpolated from F_low
                        * Used to construct covariances
                        * Differentiable w.r.t. F_low
            
            **Auxiliary outputs:**
            - anchors: (M, 3) anchor positions before jittering
                       * The "base" positions that were sampled
                       * Useful for debugging and visualization
                       * Shows importance sampling distribution
            
            - debug: Dictionary with diagnostic information
                     * Stage enable/disable flags
                     * Statistics (mean probabilities, counts)
                     * Useful for monitoring pipeline behavior
            
            - state: Updated state dictionary
                     * Contains EMA values and cached data
                     * Pass to next call for temporal stability
                     * Can be saved/loaded for checkpointing
    
    Complexity:
        - Time: O(M·N + M·k²) where M >> N (upsampling factor 5-20×)
        - Memory: O(M·N) for Gumbel-Softmax (batched to reduce peak)
        - GPU: ~5 seconds for N=10k → M=100k on RTX 3090
        - Bottleneck: Gumbel-Softmax sampling (Stage 3)
    
    Memory management:
        - Gumbel-Softmax batched to limit peak memory
        - FAISS cache cleared at end (configurable)
        - Intermediate tensors freed aggressively
        - Use smaller M or batch_size if OOM
    
    Configuration structure:
        ```python
        cfg = {
            'knn': {
                'use_faiss': True,      # Use FAISS for fast KNN
                'tau': 0.15,            # Attention temperature
                'nlist': 100,           # IVF cells
            },
            'surface_detection': {
                'enabled': True,        # Stage on/off
                'k': 48,                # PCA neighbors
                'ema_alpha': 0.3,       # Temporal smoothing
            },
            'anchor_density': {
                'enabled': True,
                'stage2_k': 16,         # KNN neighbors for density
                'anchor_density_beta': 0.7,  # Bias strength
                'spacing_bias_gamma': 0.6,   # Fallback bias
            },
            'sampling': {
                'M': 50000,             # Output points
                'tau': 0.2,             # Gumbel temperature
                'alpha': 0.35,          # Jitter scale
            },
            'taubin': {
                'enabled': True,
                'iters': 3,
                'lambda_smooth': 0.33,
                'lambda_inflate': -0.53,
            },
            'normal_smooth': {
                'enabled': True,
                'iters': 2,
                'k': 16,
                'lambda_smooth': 0.8,
            },
            'covariance': {
                'k_F': 32,
                'sigma0': 0.08,
                'use_polar_decomposition': True,
                'F_smooth': {
                    'num_nodes': 180,
                    'lambda_lap': 1e-2,
                }
            },
            'performance': {
                'clear_cache': True,    # Free memory at end
            },
            'debug': {
                'verbose': False,       # Print progress
            }
        }
        ```
    
    Example - Basic usage:
        >>> x_low = torch.randn(5000, 3)
        >>> F_low = torch.eye(3).expand(5000, 3, 3).clone()
        >>> result = upsample(x_low, F_low)
        >>> print(result['points'].shape)  # (50000, 3)
    
    Example - Custom configuration:
        >>> cfg = {
        ...     'sampling': {'M': 100000},  # 20× upsampling
        ...     'taubin': {'enabled': False},  # Skip smoothing
        ... }
        >>> result = upsample(x_low, F_low, cfg=cfg)
    
    Example - Differentiable learning:
        >>> x_low = torch.randn(5000, 3, requires_grad=True)
        >>> F_low = torch.eye(3).expand(5000, 3, 3).clone().requires_grad_(True)
        >>> result = upsample(x_low, F_low)
        >>> loss = some_loss_function(result['points'], result['cov'])
        >>> loss.backward()
        >>> # Gradients available: x_low.grad, F_low.grad
    
    Example - Temporal stability:
        >>> state = {}
        >>> for frame in video_frames:
        ...     x_low = process_frame(frame)
        ...     result = upsample(x_low, F_low, state=state)
        ...     # state is updated with EMA values for next frame
    
    Notes:
        - All stages can be disabled via cfg for ablation studies
        - Verbose mode (cfg['debug']['verbose']=True) prints progress
        - State dict enables temporal coherence across frames
        - FAISS cache cleared by default to free memory
        - Returns torch tensors by default (faster for GPU pipelines)
    
    See also:
        - default_cfg(): Default configuration values
        - validate_cfg(): Configuration validation
        - Each stage's module for detailed stage documentation
    """
    # ========================================================================
    # Setup & Validation
    # ========================================================================
    if cfg is None:
        cfg = default_cfg()
    
    validate_cfg(cfg)
    
    # Optional: Check and warn about configuration issues
    verbose = cfg.get("debug", {}).get("verbose", False)
    if verbose:
        check_config_and_warn(cfg, verbose=False)  # Only show errors, not all warnings
    
    if state is None:
        state = {}
    
    # Device detection
    device = x_low.device if torch.is_tensor(x_low) else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Ensure torch tensors on correct device
    x_low = ensure_torch(x_low, device=device)
    F_low = ensure_torch(F_low, device=device).reshape(-1, 3, 3)
    
    # 🔥 Extract upsample config (all pipeline configs are under it)
    upsample_cfg = cfg.get("upsample", {})
    
    # Setup KNN module
    knn_cfg = upsample_cfg.get("knn", {})
    knn = HybridFAISSKNN(
        use_faiss=knn_cfg.get("use_faiss", True) and FAISS_AVAILABLE,
        use_ivf=knn_cfg.get("use_ivf", True),
        tau=knn_cfg.get("tau", 0.15),
        nlist=knn_cfg.get("nlist", 100),
        nprobe=knn_cfg.get("nprobe", 10),
    )
    
    # Random generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Verbose mode
    verbose = cfg.get("debug", {}).get("verbose", False)
    
    # Initialize stage data collection
    stage_outputs = {} if export_stages else None
    
    # Performance config
    perf_cfg = upsample_cfg.get("performance", {})
    
    # ========================================================================
    # 🔥 SIMPLE PATH: Direct F → Covariance (No upsampling)
    # ========================================================================
    # Check config or parameter
    use_simple = use_simple_pipeline or upsample_cfg.get("use_simple_pipeline", False)
    
    if use_simple:
        return _upsample_simple_direct(
            x_low, F_low, cfg, state, return_torch, 
            learnable_cov_module, knn, device, verbose,
            current_episode
        )
    
    # ========================================================================
    # SDF PATH: Volumetric → Surface via SDF (Replaces PCA-based)
    # ========================================================================
    return _upsample_sdf_path(
        x_low, F_low, cfg, state, seed, return_torch, export_stages,
        learnable_cov_module, knn, device, generator, verbose, perf_cfg,
        current_episode,
        external_levelset
    )

def _upsample_simple_direct(
    x_low: torch.Tensor,
    F_low: torch.Tensor,
    cfg: Dict,
    state: Optional[Dict],
    return_torch: bool,
    learnable_cov_module,
    knn,
    device,
    verbose: bool,
    current_episode: int = -1
) -> Dict:
    """
    🔥 Direct F → Covariance pipeline (NO upsampling, NO SDF).
    
    Simplest possible path:
    1. Use x_low as-is (no upsampling)
    2. Compute normals from F (polar decomposition)
    3. Build covariance from F directly
    
    Ultra-fast for optimization when we don't need dense point clouds.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"🔥 SIMPLE DIRECT PIPELINE (F → Covariance)")
        print(f"{'='*70}")
        print(f"Input points: {x_low.shape[0]:,}")
    
    # ========================================================================
    # 1. Compute mesh geometry (PCA)
    # ========================================================================
    if verbose:
        print(f"\n[1/2] Computing mesh geometry...")
    
    from sampling.analysis.pca import batched_pca_surface_optimized
    
    # KNN for PCA
    k_pca = min(32, x_low.shape[0] - 1)
    idx_nn, w_nn = knn(x_low, x_low, k_pca)
    
    # Batched PCA analysis
    normals_pca, surfvar, spacing_pca, curvature, anisotropy, planarity, \
    principal_dir1, principal_dir2, principal_curv = batched_pca_surface_optimized(
        x=x_low,
        indices=idx_nn,
        weights=w_nn,
        return_principal_dirs=True
    )
    
    # Orient normals outward
    normals = compute_normals_pca(x_low, knn, k=k_pca, prefer_outward=True)
    
    if verbose:
        print(f"  PCA neighbors: k={k_pca}")
        print(f"  Planarity:   mean={planarity.mean():.4f}, range=[{planarity.min():.4f}, {planarity.max():.4f}]")
        print(f"  Anisotropy:  mean={anisotropy.mean():.4f}, range=[{anisotropy.min():.4f}, {anisotropy.max():.4f}]")
        
        normal_mag = normals.norm(dim=-1)
        print(f"  Normal magnitude: mean={normal_mag.mean():.4f}, range=[{normal_mag.min():.4f}, {normal_mag.max():.4f}]")
        
        global_centroid = x_low.mean(dim=0)
        to_point = x_low - global_centroid
        outward_dot = (normals * to_point).sum(dim=-1)
        num_outward = (outward_dot > 0).sum().item()
        print(f"  Outward orientation: {num_outward:,}/{normals.shape[0]:,} ({100*num_outward/normals.shape[0]:.1f}%)")
    
    # ========================================================================
    # 2. Build covariance
    # ========================================================================
    if verbose:
        print(f"\n[2/2] Building covariance...")
    
    upsample_cfg = cfg.get("upsample", {})
    cov_cfg = upsample_cfg.get("covariance", {})
    
    is_target = (current_episode <= 0)
    
    if is_target:
        # ---- TARGET: 곡률 기반 또는 Identity ----
        use_curvature_cov = cov_cfg.get("use_curvature_for_target", False)
        
        if use_curvature_cov:
            # 곡률 기반 공분산
            if verbose:
                print(f"  Mode: Target mesh → Curvature-based covariance")
            
            # Convert to numpy
            pts_np = x_low.detach().cpu().numpy()
            nrm_np = normals.detach().cpu().numpy()
            plan_np = planarity.detach().cpu().numpy()
            aniso_np = anisotropy.detach().cpu().numpy()
            curv_np = principal_curv.detach().cpu().numpy()
            dir1_np = principal_dir1.detach().cpu().numpy()
            dir2_np = principal_dir2.detach().cpu().numpy()
            dirs_raw = (dir1_np, dir2_np)
            
            # Sigma parameters
            sigma_params = dict(cov_cfg.get("curvature_sigma", {}))
            
            if verbose:
                print(f"  Planarity: mean={plan_np.mean():.4f}")
                print(f"  Anisotropy: mean={aniso_np.mean():.4f}")
                print(f"  Principal curvature: mean={curv_np.mean():.6f}")
            
            # Create curvature-based covariance
            cov_np = create_curvature_based_covariance_star(
                points=pts_np,
                normals=nrm_np,
                planarity=plan_np,
                anisotropy=aniso_np,
                sigma_params=sigma_params,
                principal_curv=curv_np,
                principal_dirs=dirs_raw
            )
            
            cov = torch.from_numpy(cov_np).to(device)
        else:
            # Identity 기반 등방성 공분산
            if verbose:
                print(f"  Mode: Target mesh → Isotropic covariance")
            
            # Simple isotropic covariance: σ²·I
            sigma_iso = float(cov_cfg.get("sigma_isotropic", 0.01))
            cov = torch.eye(3, device=device, dtype=x_low.dtype).unsqueeze(0).expand(x_low.shape[0], 3, 3) * (sigma_iso ** 2)
            
            if verbose:
                print(f"  Isotropic sigma: {sigma_iso:.6f}")
                print(f"  Covariance: {sigma_iso**2:.6e} * I")
        
        F_interp = torch.eye(3, device=device, dtype=x_low.dtype).unsqueeze(0).expand(x_low.shape[0], 3, 3)
    
    else:
        # ---- MORPH: F-field 기반 Σ ----
        if verbose:
            print(f"  Mode: Morphing (ep={current_episode}) → F-based covariance")
        
        cov, F_interp, _ = build_deformation_covariance(
            points=x_low,
            x_low=x_low,
            F_low=F_low,
            knn=knn,
            cfg=upsample_cfg,
            learnable_cov_module=learnable_cov_module,
            x_low_normals=normals,
            x_low_curvature=None
        )
    
    # ========================================================================
    # 3. Quality validation
    # ========================================================================
    if verbose:
        print(f"\n[3/3] Output quality...")
        
        # Normal validation
        final_mag = normals.norm(dim=-1)
        print(f"\n  Normals:")
        print(f"    Count: {normals.shape[0]:,}")
        print(f"    Magnitude: mean={final_mag.mean():.6f}, std={final_mag.std():.6f}")
        print(f"    Range: [{final_mag.min():.6f}, {final_mag.max():.6f}]")
        
        bad_normals = (final_mag < 0.99) | (final_mag > 1.01)
        if bad_normals.any():
            print(f"    ⚠️  {bad_normals.sum().item()} normals not unit length")
        else:
            print(f"    ✅ All normals are unit vectors")
        
        # Z-up consistency
        up_vec = torch.tensor([0.0, 0.0, 1.0], device=normals.device, dtype=normals.dtype)
        z_up_dot = normals @ up_vec
        num_up = (z_up_dot > 0).sum().item()
        print(f"    Z-up alignment: {num_up:,}/{normals.shape[0]:,} ({100*num_up/normals.shape[0]:.1f}%)")
        
        # Covariance validation
        print(f"\n  Covariance:")
        print(f"    Shape: {cov.shape}")
        cov_det = torch.det(cov)
        print(f"    det(Σ): mean={cov_det.mean():.6e}, range=[{cov_det.min():.6e}, {cov_det.max():.6e}]")
        
        eigenvals = torch.linalg.eigvalsh(cov)
        print(f"    Eigenvalues: min={eigenvals.min():.6e}, max={eigenvals.max():.6e}")
        
        bad_cov = (cov_det <= 0)
        if bad_cov.any():
            print(f"    ⚠️  {bad_cov.sum().item()} covariances are not positive definite!")
        else:
            print(f"    ✅ All covariances are positive definite")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ SIMPLE PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Output points: {x_low.shape[0]:,}")
        print(f"Normals: {normals.shape}")
        print(f"Covariance: {cov.shape}")
    
    # Format output
    result = {
        "points": x_low if return_torch else as_numpy(x_low),
        "normals": normals if return_torch else as_numpy(normals),
        "cov": cov if return_torch else as_numpy(cov),
        "F_interp": F_interp if return_torch else as_numpy(F_interp),
        "anchors": x_low if return_torch else as_numpy(x_low),
        "state": state,
        "debug": {
            "pipeline_mode": "simple_direct",
            "num_points": x_low.shape[0],
            "normal_valid": True,
            "cov_valid": True,
        }
    }
    
    return result


def _upsample_sdf_path(
    x_low: torch.Tensor,
    F_low: torch.Tensor,
    cfg: Dict,
    state: Optional[Dict],
    seed: int,
    return_torch: bool,
    export_stages: bool,
    learnable_cov_module,
    knn,
    device,
    generator,
    verbose: bool,
    perf_cfg: Dict,
    current_episode: int = -1,
    external_levelset = None
) -> Dict:
    """
    SDF-based upsampling pipeline (alternative to PCA-based).
    
    Pipeline:
        STAGE-1: Build differentiable SDF grid
        STAGE-2: Extract surface anchors + curvature
        STAGE-3: Importance sampling (reuse existing sampler)
        STAGE-4: Project to SDF=0 + optional Taubin
        STAGE-5: Recompute ∇SDF normals + smooth
        STAGE-6: Curvature-based Σ (target) or F-field Σ (morph)
    """
    # 🔥 Extract upsample config first (all pipeline configs are under it)
    upsample_cfg = cfg.get("upsample", {})
    
    # ========================================================================
    # STAGE 0: Input Point Cloud Visualization
    # ========================================================================
    debug_cfg = upsample_cfg.get("debug", {})
    
    # 🔥 Setup export directory (episode별로 구분)
    from pathlib import Path
    import numpy as np
    
    # 🔥 CRITICAL: output_dir은 debug config나 top-level config에서 읽기
    if "output_dir" in debug_cfg:
        output_dir = Path(debug_cfg["output_dir"])
    elif "output_dir" in cfg:
        output_dir = Path(cfg["output_dir"])
    else:
        output_dir = Path("output/default/")
    
    # 🔥 FIXED: Always use episode-aware or target-aware paths
    if debug_cfg.get("export_dir") is not None:
        export_dir = Path(debug_cfg.get("export_dir"))
    else:
        # Episode별 또는 target별 폴더 생성
        if current_episode >= 0:
            # Training episode: output_dir/ep{N}/stages/
            export_dir = Path(output_dir) / f"ep{current_episode:03d}" / "stages"
        else:
            # Target render: output_dir/target/stages/
            export_dir = Path(output_dir) / "target" / "stages"
    
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # 🔥 NaN/Inf 안전장치: F_low 검증 및 수정
    if torch.isnan(F_low).any() or torch.isinf(F_low).any():
        if verbose:
            print(f"  ⚠️  WARNING: F_low contains NaN/Inf! Clamping to safe range.")
            print(f"     NaN count: {torch.isnan(F_low).sum().item()}")
            print(f"     Inf count: {torch.isinf(F_low).sum().item()}")
        
        # NaN/Inf 제거 및 클램핑
        F_low = torch.nan_to_num(F_low, nan=1.0, posinf=3.0, neginf=0.1)
        F_low = F_low.clamp(min=0.01, max=3.0)  # 물리적으로 합리적인 범위
    
    if debug_cfg.get("export_stages", False):
        # Reduced verbose output
        if verbose:
            print(f"\n[Stages] Exporting to {export_dir}/")
        
        _visualize_input_points(
            save_path=str(export_dir / "stage0_input.png"),
            points=x_low,
            bbox_min=None,  # Auto-compute
            bbox_max=None,
            verbose=verbose
        )
    
    # ========================================================================
    # STAGE 1: Initialize Level Set φ₀ (OR use external)
    # ========================================================================
    sdf_cfg = upsample_cfg.get("sdf", {})
    
    # 🔥 항상 출력 (resolution 문제 디버깅용)
    print(f"\n[SDF Config] Loaded values:")
    print(f"  resolution: {sdf_cfg.get('resolution', 'NOT IN CONFIG')} (default: 128)")
    print(f"  padding: {sdf_cfg.get('padding', 'NOT IN CONFIG')} (default: 0.5)")
    print(f"  k_sdf: {sdf_cfg.get('k_sdf', 'NOT IN CONFIG')} (default: 200)")
    
    # ========================================================================
    # STAGE 1: SDF Grid Initialization
    # ========================================================================
    if True:  # (advect 제거 후 legacy wrapper)
        if verbose:
            print("\n" + "="*80)
            print("STAGE 1: Level Set φ₀ Initialization")
            print("="*80)
        
        # Config에서 SDF 파라미터 읽기
        resolution_val = int(sdf_cfg.get("resolution", 128))
        padding_val = float(sdf_cfg.get("padding", 0.5))
        
        if verbose:
            print(f"  📦 [SDF Config]:")
            print(f"     resolution: {resolution_val}")
            print(f"     padding: {padding_val}")
        
        # Initialize level set grid
        k_sdf = int(sdf_cfg.get("k_sdf", 150))
        
        levelset = LevelSetGrid(
            resolution=resolution_val,
            padding=padding_val,
        )
        
        # 🔥 항상 출력: 실제 생성된 LevelSetGrid 확인
        print(f"  ✓ LevelSetGrid created:")
        print(f"     resolution: {levelset.res} (grid will be {levelset.res}³)")
        print(f"     padding: {levelset.padding}")
        
        # Build initial φ from volumetric points
        downsample_to = sdf_cfg.get("downsample_points", None)
        
        if verbose:
            print(f"  📦 Other params: k_sdf={sdf_cfg.get('k_sdf', 200)}, chunk_size={sdf_cfg.get('chunk_size', 65536)}")
        
        levelset.init_from_points(
            x_vol=x_low,
            knn=knn,
            k_density=k_sdf,
            distance_scale=10.0,
            downsample_to=downsample_to,
            gaussian_sigma_vox=float(sdf_cfg.get("gaussian_sigma_vox", 0.4)),
            sigmoid_center=float(sdf_cfg.get("sigmoid_center", 0.48)),
            sigmoid_slope=float(sdf_cfg.get("sigmoid_slope", 8.0)),
            use_gaussian=bool(sdf_cfg.get("use_gaussian", True)),
            chunk_size=int(sdf_cfg.get("chunk_size", 131072)),
            verbose=verbose
        )
        
        # ⚡ Precompute gradient volume for fast path
        levelset.refresh_grad_cache()
        if verbose:
            print("  ⚡ Gradient volume cached for fast operations")
        
        if verbose:
            R = int(levelset.phi.shape[0])
            sdf_min, sdf_max = float(levelset.phi.min()), float(levelset.phi.max())
            surface_voxels = (levelset.phi.abs() < 0.05).sum().item()
            total_voxels = R ** 3
            N_orig = len(x_low)
            N_sdf = downsample_to if (downsample_to and N_orig > downsample_to) else N_orig
            
            print(f"  Method: ⚡⚡⚡ FAISS-accelerated (k=200 neighbors)")
            print(f"  Input points: {N_orig:,} → {N_sdf:,} (for SDF)")
            print(f"  Grid: {R}³ = {total_voxels:,} voxels")
            print(f"  SDF range: [{sdf_min:.4f}, {sdf_max:.4f}]")
            print(f"  Surface voxels (|SDF|<0.05): {surface_voxels:,} ({surface_voxels/total_voxels:.1%})")
            
            # Speedup estimate
            speedup = (total_voxels * N_sdf) / (total_voxels * 200)
            print(f"  Speedup vs naive: ~{speedup:.0f}× faster (FAISS k-NN)")
    
    # Get references for compatibility
    sdf_grid = levelset.phi
    bbox_min = levelset.bbox_min
    bbox_max = levelset.bbox_max
    
    # Visualize SDF grid after STAGE 1
    if debug_cfg.get("export_stages", False):
        # Convert to numpy
        sdf_np = sdf_grid.detach().cpu().numpy()
        bbox_min_np = bbox_min.detach().cpu().numpy()
        bbox_max_np = bbox_max.detach().cpu().numpy()
        
        # Visualize
        _visualize_sdf_grid(export_dir / "stage1_sdf_grid.png", sdf_np, bbox_min_np, bbox_max_np, verbose)
    
    # Visualization after STAGE 1
    # debug_cfg already defined at the top of function
    stage_outputs = {}
    if debug_cfg.get("export_stages", False):
        stage_outputs['stage1_levelset'] = levelset
    
    # ========================================================================
    # STAGE 1.5: Input Mesh Curvature Analysis (PCA-based)
    # ========================================================================
    curvature_data = None
    anc_cfg = upsample_cfg.get("surface_anchors", {})
    compute_curvature = bool(anc_cfg.get("compute_curvature", False))
    
    if compute_curvature and (knn is not None) and (x_low.shape[0] > 8):
        if verbose:
            print("\n" + "="*80)
            print("STAGE 1.5: Input Mesh Curvature Analysis (PCA-based)")
            print("="*80)
        
        from .analysis.pca import batched_pca_surface_optimized
        
        k = min(32, x_low.shape[0] - 1)
        idx_nn, w_nn = knn(x_low, x_low, k)
        
        # PCA 기반 곡률 계산 (입력 메시에서)
        try:
            normals_pca, surfvar, spacing_pca, curvature, anisotropy, planarity, \
            principal_dir1, principal_dir2, principal_curv = batched_pca_surface_optimized(
                x=x_low,
                indices=idx_nn,
                weights=w_nn,
                return_principal_dirs=True
            )
            
            # 곡률 데이터 저장 (Stage 2에서 앵커로 보간)
            curvature_data = {
                "points": x_low,                      # (N, 3) 원본 포인트
                "planarity": planarity,               # (N,)
                "anisotropy": anisotropy,             # (N,)
                "principal_curv": principal_curv,     # (N, 2)
                "principal_dirs": (principal_dir1, principal_dir2),  # tuple of (N,3)
                "curvature": curvature,               # (N,)
                "surfvar": surfvar,                   # (N,)
            }
            
            if verbose:
                print(f"  Method: PCA eigenvalue analysis (k={k})")
                print(f"  Input points: {x_low.shape[0]:,}")
                print(f"  Planarity:   mean={planarity.mean():.4f}, std={planarity.std():.4f}, range=[{planarity.min():.4f}, {planarity.max():.4f}]")
                print(f"  Anisotropy:  mean={anisotropy.mean():.4f}, std={anisotropy.std():.4f}, range=[{anisotropy.min():.4f}, {anisotropy.max():.4f}]")
                print(f"  Curvature:   mean={curvature.mean():.6f}, max={curvature.max():.6f}")
                k1, k2 = principal_curv.unbind(-1)
                print(f"  Principal k1: mean={k1.mean():.6f}, range=[{k1.min():.6f}, {k1.max():.6f}]")
                print(f"  Principal k2: mean={k2.mean():.6f}, range=[{k2.min():.6f}, {k2.max():.6f}]")
                print(f"  Surface var: mean={surfvar.mean():.4f}")
                print(f"  ✓ Geometric features computed from input mesh")
            
            # Visualization after Stage 1.5
            if debug_cfg.get("export_stages", False):
                _visualize_curvature_data(
                    save_path=str(export_dir / "stage1.5_curvature.png"),
                    points=x_low,
                    planarity=planarity,
                    anisotropy=anisotropy,
                    principal_curv=principal_curv,
                    bbox_min=None,
                    bbox_max=None,
                    verbose=verbose
                )
        except Exception as e:
            if verbose:
                print(f"  ✗ Curvature computation failed: {e}")
                print(f"  → Continuing without curvature data")
            curvature_data = None
    
    # ========================================================================
    # STAGE 2: Surface Anchor Extraction
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 2: Surface Anchor Extraction (Level Set)")
        print("="*80)
    
    # 🔥 Auto-scale num_anchors based on resolution
    num_anchors_cfg = int(anc_cfg.get("num_anchors", 50_000))
    if anc_cfg.get("auto_scale_anchors", True):
        vox = (levelset.bbox_max - levelset.bbox_min).max().item() / (levelset.res - 1)
        L_diag = (levelset.bbox_max - levelset.bbox_min).norm().item()
        surface_area_est = 4.0 * 3.14159 * (L_diag * 0.25)**2
        target_density = 1.5
        num_anchors_auto = int(target_density * surface_area_est / (vox * vox))
        num_anchors_auto = max(10_000, min(200_000, num_anchors_auto))
        
        if verbose:
            print(f"  🔧 [Auto-scaling] Anchors: {num_anchors_cfg:,} → {num_anchors_auto:,} (R={levelset.res}, dx={vox:.4f})")
        num_anchors = num_anchors_auto
    else:
        num_anchors = num_anchors_cfg
    
    anchors_out = extract_surface_anchors(
        levelset=levelset,
        num_anchors=num_anchors,
        sigma_mult=float(anc_cfg.get("sigma_mult", 1.4)),
        project_iters=int(anc_cfg.get("project_iters", 3)),
        compute_curvature=False,
        knn=knn,
        curvature_eps_mult=float(anc_cfg.get("curvature_eps_mult", 1.25)),
        max_sdf_after_proj=anc_cfg.get("max_sdf_after_proj", None),
        verbose=verbose
    )
    
    anchors_x = anchors_out["anchors"]
    anchors_n = anchors_out["normals"]
    anchors_spacing = anchors_out["spacing"]
    p_surf_raw = anchors_out["p_surf_raw"]
    
    if verbose:
        M_a = anchors_x.shape[0]
        print(f"  Anchors: {M_a:,}")
        print(f"  Spacing: mean={anchors_spacing.mean():.4f}")
    
    # Visualization after STAGE 2
    if debug_cfg.get("export_stages", False):
        stage_outputs['stage2_anchors'] = anchors_x
        # 🔥 Visualize anchors
        _visualize_anchors(
            save_path=str(export_dir / "stage2_anchors.png"),
            anchors=anchors_x,
            normals=anchors_n,
            p_surf_raw=p_surf_raw,
            bbox_min=levelset.bbox_min,
            bbox_max=levelset.bbox_max,
            verbose=verbose
        )
    
    # ========================================================================
    # STAGE 3: Importance Sampling (Reuse Existing Sampler)
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 3: Soft Continuous Upsampling (SDF path)")
        print("="*80)
    
    # 🔥 Determine episode mode early (needed for gradient control)
    cov_cfg = upsample_cfg.get("covariance", {})
    episode = int(cov_cfg.get("episode", -1))
    is_morph = (episode >= 0)  # Morphing mode enables gradients
    
    # Setup sampler config (YAML 우선: 기본값은 '없는 키'에만 주입)
    samp_cfg = dict(upsample_cfg.get("sampling", {}))  # Shallow copy
    _defaults = {
        "uniform_pi": True,
        "u_local": 0.75,
        "center_tau": 12.0,
        "voxel_scale": 1.20,
        "local_S": 12,
        "micro_jitter_j0": 0.18,
        "use_soft_nms": True,
        "suppress_lambda": 0.25,
        "keep_ratio": 0.35,
        "beta_soft": 0.80,
        "bias_target": 1.5,
        "use_dynamic_cap": True
    }
    for k, v in _defaults.items():
        samp_cfg.setdefault(k, v)
    
    # 밀도 가중치 (완만하게만)
    w_density = (1.0 / (anchors_spacing + 1e-6))
    w_density = w_density / (w_density.mean() + 1e-12)
    
    samp_cfg["p_surf_raw"] = p_surf_raw
    samp_cfg["w_density"] = w_density
    # 🔥 Auto-scale M based on num_anchors
    M_cfg = int(samp_cfg.get("M", 500_000))
    if samp_cfg.get("auto_scale_M", True):
        upsampling_ratio = 10  # 앵커 대비 10배
        M_auto = upsampling_ratio * len(anchors_x)
        M_auto = max(100_000, min(2_000_000, M_auto))  # 범위 제한
        
        if verbose:
            print(f"  🔧 [Auto-scaling] M: {M_cfg:,} → {M_auto:,} ({upsampling_ratio}× anchors)")
        samp_cfg["M"] = M_auto
    else:
        samp_cfg["M"] = M_cfg
    
    # Sample points (probs=None → use cfg paths)
    points, normals_up, anchors_sel, anchor_selection_count = sample_points(
        anchors_x, anchors_n, anchors_spacing, probs=None, cfg=samp_cfg, generator=generator
    )
    
    if verbose:
        print(f"  Anchors {anchors_x.shape[0]:,} → Dense {points.shape[0]:,} "
              f"({points.shape[0]/max(1,anchors_x.shape[0]):.1f}×)")
    
    # 🔥 CRITICAL: Add identity shortcut to preserve gradient connection to x_low
    # Sampling is discrete (non-differentiable), but we can add weak gradient via shortcut
    if is_morph:
        if verbose:
            print(f"  [3-GRAD] Adding identity shortcut to x_low (α=0.02)...")
        
        # Find nearest x_low neighbors (use k=4 for smooth interpolation)
        k_interp = min(4, x_low.shape[0])
        idx_nn, w_nn = knn(points, x_low, k=k_interp)
        
        # Weighted average (differentiable!)
        nearest_x = (w_nn.unsqueeze(-1) * x_low[idx_nn]).sum(dim=1)  # (M, 3)
        
        # ⚡ Mix with scheduled alpha (후반 drift 억제)
        alpha_sampling = alpha_schedule(episode)
        if verbose:
            print(f"  [3-GRAD] α={alpha_sampling:.3f} (ep={episode})")
        points = (1.0 - alpha_sampling) * points.detach() + alpha_sampling * nearest_x
        
        if verbose:
            print(f"  [3-GRAD] Gradient connection established: points → x_low (k={k_interp})")
    
    # 🔥 CRITICAL: Recompute normals and SAVE separately for diagnostics
    # The sampler uses weighted-average normals from anchors, which don't match
    # the jittered positions. Replace with accurate ∇φ at actual sampled positions.
    if verbose:
        print(f"  [3-POST] Recomputing normals from ∇φ (fixing jitter mismatch)...")
    
    with torch.no_grad():
        normals_up_anchor_avg = normals_up  # Keep original for comparison
        normals_stage3 = levelset.grad_fast(points)  # ⚡ Fast gradient path
        
        # 🔥 NEW: Local majority alignment (replaces global reference)
        normals_stage3 = align_normals_local_majority(
            normals_stage3, points, knn, k=16, verbose=verbose
        )
        
        # Report normal adjustment from anchor average to ∇φ
        cos_angle = (normals_stage3 * normals_up_anchor_avg).sum(dim=-1).clamp(-1, 1)
        angle_deg = torch.acos(cos_angle) * (180.0 / 3.14159)
        if verbose:
            print(f"  [3-POST] Normal correction (anchor avg → ∇φ): "
                  f"mean={float(angle_deg.mean()):.1f}°, "
                  f"p95={float(angle_deg.kthvalue(int(0.95*angle_deg.numel())).values):.1f}°")
        
        # For subsequent stages (will be overwritten in Stage 4-A, but that's OK)
        normals_up = normals_stage3
    
    # Visualization after STAGE 3
    if debug_cfg.get("export_stages", False):
        stage_outputs['stage3_sampled'] = points
        # 🔥 Visualize sampled points
        _visualize_sampled_points(
            save_path=str(export_dir / "stage3_sampled.png"),
            points=points,
            normals=normals_up,
            anchors=anchors_x,
            bbox_min=levelset.bbox_min,
            bbox_max=levelset.bbox_max,
            verbose=verbose
        )
    
    # ========================================================================
    # STAGE 3.5: (옵션) 과밀 억제 Post-pass
    # ========================================================================
    voxel_drop_cfg = upsample_cfg.get("voxel_drop", {})
    if voxel_drop_cfg.get("enabled", False):
        if verbose:
            print("\n" + "="*80)
            print("STAGE 3.5: Voxel Quota Drop (clustering prevention)")
            print("="*80)
        
        # 평균 NN 기반 복셀 크기 추정
        voxel_size = float(anchors_spacing.mean().item()) * 0.90
        
        # 🔥 Auto-scale max_per_voxel based on local density
        max_per_voxel_cfg = int(voxel_drop_cfg.get("max_per_voxel", 8))
        if voxel_drop_cfg.get("auto_scale_quota", True):
            # 로컬 밀도 기반 자동 계산
            vox_idx = (points / max(voxel_size, 1e-6)).floor().long()
            key = (vox_idx[:,0]*73856093 ^ vox_idx[:,1]*19349663 ^ vox_idx[:,2]*83492791) & 0x7fffffff
            _, cnt = key.unique(return_counts=True)
            avg_cnt = cnt.float().mean().item()
            target_ratio = 10  # 평균의 10배까지 허용
            max_per_voxel_auto = int(max(4, min(16, target_ratio * avg_cnt * 1.1)))
            
            if verbose:
                print(f"  🔧 [Auto-scaling] max_per_voxel: {max_per_voxel_cfg} → {max_per_voxel_auto} (avg={avg_cnt:.1f})")
            max_per_voxel = max_per_voxel_auto
        else:
            max_per_voxel = max_per_voxel_cfg
        
        keep_mask = voxel_quota_drop(points, max_per_voxel=max_per_voxel, voxel_size=voxel_size)
        
        # 드롭 적용
        points_pruned = points[keep_mask]
        normals_pruned = normals_up[keep_mask]
        normals_stage3_pruned = normals_stage3[keep_mask]
        
        num_dropped = points.shape[0] - points_pruned.shape[0]
        
        # 부족분 보충 (앵커에서 균일 샘플 + 소지터 → 재투영)
        need = points.shape[0] - points_pruned.shape[0]
        if need > 0:
            idx = torch.randint(0, anchors_x.shape[0], (need,), device=points.device)
            refill = anchors_x[idx] + 0.15 * voxel_size * torch.randn_like(anchors_x[idx])
            refill = levelset.project_zero_fast(refill, iters=2, lr=0.8)
            refill_normals = levelset.grad_fast(refill)
            
            # 🔥 NEW: Local majority alignment
            refill_normals = align_normals_local_majority(
                refill_normals, refill, knn, k=8, verbose=verbose  # k=8 (fewer points)
            )
            
            points = torch.cat([points_pruned, refill], dim=0)
            normals_up = torch.cat([normals_pruned, refill_normals], dim=0)
            normals_stage3 = torch.cat([normals_stage3_pruned, refill_normals], dim=0)
            
            if verbose:
                print(f"  Dropped: {num_dropped:,}, Refilled: {need:,}")
        else:
            points = points_pruned
            normals_up = normals_pruned
            normals_stage3 = normals_stage3_pruned
            
            if verbose:
                print(f"  Dropped: {num_dropped:,}, No refill needed")
        
        if verbose:
            print(f"  Final: {points.shape[0]:,} points (uniform coverage)")
    
    # ========================================================================
    # STAGE 4: Level Set Projection + Tangent Smoothing
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 4: Level Set φ=0 Projection + Tangent Smoothing")
        print("="*80)
        mode = "Morphing (gradient ON)" if is_morph else "Target (gradient OFF)"
        print(f"  Mode: {mode} (episode={episode})")
    
    # 4-A) 1차 투영 (가볍게)
    ref_cfg = upsample_cfg.get("sdf_refinement", {})
    ref_points = project_zero(
        levelset,
        points,
        iters=int(ref_cfg.get("proj_iters", 2)),  # 2~3
        lr=float(ref_cfg.get("lr", 0.8)),
        require_grad_points=is_morph,   # 🔥 morphing일 때만 좌표 미분 켠다
        detach_phi=True,                # φ는 항상 detach
        clamp_step_mult=float(ref_cfg.get("clamp_step_mult", 0.8)),
        tol=float(ref_cfg.get("tol", 0.0))
    )
    
    if verbose:
        drift = (points - ref_points).norm(dim=-1).detach()
        print(f"  [4-A] 1차 투영 |Δx|: mean={float(drift.mean()):.6f}, "
              f"p95={float(drift.kthvalue(int(0.95*drift.numel())).values):.6f}")
    
    # ∇φ 기반 노멀 재계산 및 부호 정렬
    normals_ref = levelset.grad_fast(ref_points)
    
    # 🔥 NEW: Local majority alignment
    normals_ref = align_normals_local_majority(
        normals_ref, ref_points, knn, k=16, verbose=verbose
    )
    
    # 이후 파이프라인에서 사용할 최종 노멀로 교체
    normals_up = normals_ref
    
    # Visualization after initial projection
    if debug_cfg.get("export_stages", False):
        stage_outputs['stage4_projected'] = ref_points
        # 🔥 Visualize projected points
        _visualize_projected_points(
            save_path=str(export_dir / "stage4_projected.png"),
            points=ref_points,
            normals=normals_up,
            bbox_min=levelset.bbox_min,
            bbox_max=levelset.bbox_max,
            verbose=verbose
        )
    
    # ========================================================================
    # STAGE 4-B: (Optional) Tangent Smoothing - DISABLED in config
    # ========================================================================
    # NOTE: Config에서 비활성화됨 (F-smoothing + Σ 비등방으로 대체)
    ref_normals = normals_up
    
    # 🔥 CRITICAL: Define normals_final BEFORE diagnostics
    normals_final = ref_normals
    
    if verbose:
        print("  [4-B] ⊘ Tangent smoothing SKIPPED (disabled in config)")
    
    # ========================================================================
    # STAGE 4 DIAGNOSTICS (품질 검증)
    # ========================================================================
    # 간단 요약 (quick_diagnostics)
    if verbose:
        diag = quick_diagnostics(levelset, ref_points, normals_final, anchors_spacing)
        dx = float(((levelset.bbox_max - levelset.bbox_min) / (levelset.phi.shape[0] - 1)).mean())
        
        print("="*80)
        print("Stage 4 Quick Diagnostics Summary")
        print("="*80)
        print(f"Voxel size (Δx): {dx:.6f}")
        print(f"|φ| mean: {diag['phi_mean']:.6e} ({diag['phi_mean']/dx:.3f}·Δx)")
        print(f"|φ| p95 : {diag['phi_p95']:.6e} ({diag['phi_p95']/dx:.3f}·Δx)")
        print(f"|φ| max : {diag['phi_max']:.6e} ({diag['phi_max']/dx:.3f}·Δx)")
        print(f"‖n‖ mean: {diag['n_mag_mean']:.5f}, min: {diag['n_mag_min']:.5f}")
        print(f"Voxel dup mean: {diag['voxel_count_mean']:.2f}, p95: {diag['voxel_count_p95']:.2f}, max: {diag['voxel_count_max']}")
        print("="*80 + "\n")
        
        # 🔥 Normal Quality Diagnostics
        print(f"\n{'='*60}")
        print("Normal Quality Diagnostics")
        print(f"{'='*60}")
        
        # 1) Local consistency (이웃과 얼마나 일치하는가)
        k_test = min(8, ref_points.shape[0] - 1)
        idx_nn, _ = knn(ref_points, ref_points, k_test)
        normals_nn = ref_normals[idx_nn]
        dot_prods = (ref_normals.unsqueeze(1) * normals_nn).sum(dim=-1)
        consistency = (dot_prods > 0).float().mean()
        
        print(f"  Local consistency (k={k_test}):")
        print(f"    Agreement rate: {consistency:.3f} (목표 > 0.95)")
        print(f"    Mean dot: {dot_prods.mean():.3f} (목표 > 0.8)")
        
        # 2) Outward-facing (중심에서 멀어지는 방향)
        center = ref_points.mean(dim=0)
        outward = ref_points - center
        outward_norm = F.normalize(outward, dim=-1, eps=1e-9)
        facing_dot = (ref_normals * outward_norm).sum(dim=-1)
        facing_out_ratio = (facing_dot > 0).float().mean()
        
        print(f"  Outward-facing check:")
        print(f"    Ratio: {facing_out_ratio:.3f} (목표 > 0.85)")
        print(f"    Mean dot: {facing_dot.mean():.3f}")
        
        # 3) Z-up bias (토끼는 위를 향해야 함)
        z_up = torch.tensor([0, 0, 1], device=ref_normals.device, dtype=ref_normals.dtype)
        z_dot = (ref_normals * z_up).sum(dim=-1)
        z_up_ratio = (z_dot > 0).float().mean()
        
        print(f"  Z-up bias:")
        print(f"    Up-facing ratio: {z_up_ratio:.3f} (토끼: ~0.7-0.9)")
        print(f"{'='*60}\n")
        
        # 🔥 귀 보존 품질 게이트 (사용자 패치)
        print(f"{'='*60}")
        print("🐰 Detail Preservation Quality Gate (Ear/Peak Detection)")
        print(f"{'='*60}")
        
        # 1) 고지대(상위 Z) 비율 체크
        z_vals = ref_points[:, 2]
        z_thr = torch.quantile(z_vals, 0.85)
        peak_ratio = (z_vals > z_thr).float().mean().item()
        
        print(f"  Peak region (Z > p85):")
        print(f"    Ratio: {peak_ratio:.3f} (토끼 권장: 0.12~0.20)")
        
        if peak_ratio < 0.10:
            print(f"    ⚠️  WARNING: Peak points too few! (귀 손실 가능)")
            print(f"    → 권장: SDF resolution↑ or num_anchors↑")
        elif peak_ratio > 0.10 and peak_ratio < 0.20:
            print(f"    ✅ PASS: Peak region preserved")
        else:
            print(f"    ℹ️  Peak ratio high (OK if correct)")
        
        # 2) φ 잔차 재확인 (프로젝션 품질)
        phi_p95_dx = diag['phi_p95'] / dx
        print(f"\n  Projection quality:")
        print(f"    |φ| p95: {phi_p95_dx:.3f}·Δx (권장: < 0.5)")
        
        if phi_p95_dx > 0.5:
            print(f"    ⚠️  WARNING: Projection quality low!")
            print(f"    → 권장: project_iters↑ or tol↓")
        else:
            print(f"    ✅ PASS: Well-aligned to φ=0")
        
        print(f"{'='*60}\n")

    # Visualization after STAGE 4
    if debug_cfg.get("export_stages", False):
        stage_outputs['stage4_projected'] = ref_points
        stage_outputs['stage4_normals'] = ref_normals
        # 🔥 Visualize projected points
        _visualize_projected_points(
            save_path=str(export_dir / "stage4_projected.png"),
            points=ref_points,
            normals=ref_normals,
            bbox_min=levelset.bbox_min,
            bbox_max=levelset.bbox_max,
            verbose=verbose
        )
    
    
    # ========================================================================
    # STAGE 5: Covariance Construction (SDF path)
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 5: Covariance Construction (SDF path)")
        print("="*80)
    
    # episode와 is_morph는 STAGE 3에서 이미 정의됨
    is_target = (episode < 0)
    
    if is_target:
        # ---- TARGET: 곡률 기반 Σ★ (NumPy) ----
        # Convert to numpy
        pts_np = ref_points.detach().cpu().numpy()      # (M,3)
        nrm_np = normals_final.detach().cpu().numpy()   # (M,3)
        
        # 🔥 Stage 1.5에서 온 곡률 데이터를 직접 보간 (입력 메시 기반)
        if curvature_data is not None:
            src_points = curvature_data["points"]  # (N_in, 3) on GPU
            k_interp = int(cov_cfg.get("k_interp", 8))
            k_interp = min(k_interp, src_points.shape[0] - 1)
            
            # KNN: ref_points → input mesh
            idx_curv, w_curv = knn(ref_points, src_points, k=k_interp)
            
            # planarity 보간
            plan_src = curvature_data["planarity"]  # (N_in,) on GPU
            plan_nn = plan_src[idx_curv]  # (M, k)
            plan_pts = (w_curv * plan_nn).sum(dim=1)  # (M,)
            plan_np = plan_pts.detach().cpu().numpy()
            
            # anisotropy 보간
            aniso_src = curvature_data["anisotropy"]
            aniso_nn = aniso_src[idx_curv]
            aniso_pts = (w_curv * aniso_nn).sum(dim=1)
            aniso_np = aniso_pts.detach().cpu().numpy()
            
            # principal_curv 보간
            curv_src = curvature_data["principal_curv"]  # (N_in, 2)
            curv_nn = curv_src[idx_curv]  # (M, k, 2)
            curv_pts = (w_curv.unsqueeze(-1) * curv_nn).sum(dim=1)  # (M, 2)
            curv_np = curv_pts.detach().cpu().numpy()
            
            # principal_dirs 보간 (방향 벡터 → 재정규화 필요)
            dir1_src, dir2_src = curvature_data["principal_dirs"]
            dir1_nn = dir1_src[idx_curv]  # (M, k, 3)
            dir2_nn = dir2_src[idx_curv]
            dir1_pts = (w_curv.unsqueeze(-1) * dir1_nn).sum(dim=1)  # (M, 3)
            dir2_pts = (w_curv.unsqueeze(-1) * dir2_nn).sum(dim=1)
            dir1_pts = F.normalize(dir1_pts, dim=-1)
            dir2_pts = F.normalize(dir2_pts, dim=-1)
            
            # NumPy 변환
            dir1_np = dir1_pts.detach().cpu().numpy()
            dir2_np = dir2_pts.detach().cpu().numpy()
            dirs_raw = (dir1_np, dir2_np)  # tuple of (M, 3)
        else:
            # 기본값 설정
            plan_np = np.full((pts_np.shape[0],), 0.5, dtype=np.float32)
            aniso_np = np.full((pts_np.shape[0],), 0.5, dtype=np.float32)
            curv_np = None
            dirs_raw = None
        
        # Verbose logging
        if verbose:
            print(f"  [Episode {episode}] Target mesh - curvature-based Σ★")
            if curvature_data is not None:
                print(f"  [Curvature Data] From Stage 1.5 (input mesh):")
                print(f"    Source: {src_points.shape[0]:,} input points")
                print(f"    Target: {ref_points.shape[0]:,} dense points")
                print(f"    → Interpolated: {src_points.shape[0]:,} input points → {ref_points.shape[0]:,} dense points (k={k_interp})")
            else:
                print(f"  [Curvature Data] Not available (Stage 1.5 skipped)")
                print(f"    → Using default: planarity=0.5, anisotropy=0.5 for all {pts_np.shape[0]:,} points")
        
        # sigma 파라미터
        sigma_params = dict(cov_cfg.get("curvature_sigma", {}))
        
        if verbose:
            print(f"  [Final Curvature Stats for Σ★]")
            print(f"    Planarity:  mean={plan_np.mean():.4f}, std={plan_np.std():.4f}, range=[{plan_np.min():.4f}, {plan_np.max():.4f}]")
            print(f"    Anisotropy: mean={aniso_np.mean():.4f}, std={aniso_np.std():.4f}, range=[{aniso_np.min():.4f}, {aniso_np.max():.4f}]")
            if curv_np is not None:
                print(f"    Principal curv: k1∈[{curv_np[:,0].min():.6f}, {curv_np[:,0].max():.6f}], k2∈[{curv_np[:,1].min():.6f}, {curv_np[:,1].max():.6f}]")
        
        # create_curvature_based_covariance_star 호출
        cov_np = create_curvature_based_covariance_star(
            points=pts_np,
            normals=nrm_np,
            planarity=plan_np,
            anisotropy=aniso_np,
            sigma_params=sigma_params,
            principal_curv=curv_np,         # 없으면 None
            principal_dirs=dirs_raw         # 없으면 None
        )
        
        # Back to torch
        cov = torch.from_numpy(cov_np).to(ref_points.device, ref_points.dtype)
        F_interp = torch.eye(3, device=ref_points.device, dtype=ref_points.dtype)\
                     .unsqueeze(0).expand(ref_points.shape[0], 3, 3)
        
        cov_key = "cov_target"
        
        if verbose:
            sigma_n0 = sigma_params.get("sigma_n0", 0.04)
            sigma_t0 = sigma_params.get("sigma_t0", 0.05)
            a = sigma_params.get("a", 3.0)
            b = sigma_params.get("b", 0.6)
            print(f"  ✓ Curvature-based Σ★: σ_n0={sigma_n0:.3f}, σ_t0={sigma_t0:.3f}, a={a:.1f}, b={b:.1f}")
            print(f"    NumPy → Torch conversion complete")
    
    else:
        # ---- MORPH: F-field 기반 Σ ----
        if verbose:
            print(f"  [Episode {episode}] Morph - deformation-based Σ")
        
        # 🔥 에피소드별 스케줄 적용 (F-smoothing)
        cov_cfg_ep = dict(cov_cfg)  # 복사
        if 'F_smooth' in cov_cfg_ep and current_episode >= 0:
            F_smooth_cfg = dict(cov_cfg_ep['F_smooth'])
            schedule_cfg = F_smooth_cfg.get('schedule', None)
            
            if schedule_cfg is not None:
                try:
                    from utils.schedule_utils import apply_schedule_to_cfg
                    F_smooth_cfg = apply_schedule_to_cfg(
                        F_smooth_cfg, schedule_cfg, current_episode, verbose=verbose
                    )
                    cov_cfg_ep['F_smooth'] = F_smooth_cfg
                except Exception as e:
                    if verbose:
                        print(f"  [WARN] F-smooth schedule failed: {e}")
        
        # 🔥 곡률 정보 추출 (엣지-어웨어 smoothing용)
        x_low_curvature = None
        if curvature_data is not None:
            principal_curv = curvature_data.get("principal_curv")
            if principal_curv is not None and len(principal_curv) == x_low.shape[0]:
                # 평균 곡률 사용 (무단위)
                k1, k2 = principal_curv[:, 0], principal_curv[:, 1]
                # 🔥 FIX: torch operations + NaN/Inf 방지
                x_low_curvature = (k1.abs() + k2.abs()) / 2.0
                
                # NaN/Inf 체크 및 클램핑
                x_low_curvature = torch.nan_to_num(x_low_curvature, nan=0.0, posinf=1.0, neginf=0.0)
                x_low_curvature = x_low_curvature.clamp(min=0.0, max=10.0)  # 합리적 범위로 제한
                
                if not isinstance(x_low_curvature, torch.Tensor):
                    x_low_curvature = torch.from_numpy(x_low_curvature)
                x_low_curvature = x_low_curvature.to(x_low.device, x_low.dtype)
                
                if verbose:
                    print(f"  [Curvature] mean={x_low_curvature.mean():.6f}, max={x_low_curvature.max():.6f}, has_nan={torch.isnan(x_low_curvature).any().item()}")
        
        # 🔥 s_min을 voxel Δx 기반으로 동적 설정
        dx = ((levelset.bbox_max - levelset.bbox_min) / (levelset.res - 1)).max().item()
        
        # floor_k 스케줄 적용
        floor_k_schedule = cov_cfg.get("floor_k_schedule", None)
        if floor_k_schedule is not None:
            # 에피소드별 floor_k 찾기
            floor_k = None
            for ep_min, ep_max, k_val in floor_k_schedule:
                if ep_min <= current_episode <= ep_max:
                    floor_k = k_val
                    break
            if floor_k is None:
                floor_k = 0.8  # 기본값
            
            if verbose:
                print(f"  [floor_k Schedule] ep={current_episode}: floor_k={floor_k:.2f}")
        else:
            floor_k = float(cov_cfg.get("floor_k", 0.8))
        
        # s_min = floor_k * Δx (voxel 기반)
        s_min_dynamic = floor_k * dx
        cov_cfg_ep['sv_min'] = s_min_dynamic
        
        if verbose:
            print(f"  [SV Config] Δx={dx:.6f}, floor_k={floor_k:.2f}")
            print(f"             → sv_min={s_min_dynamic:.6f} (dynamic)")
        
        # 🔥 NaN 체크: 입력 검증
        if verbose:
            print(f"\n[NaN Check - Before Covariance]")
            print(f"  ref_points: has_nan={torch.isnan(ref_points).any().item()}")
            print(f"  x_low: has_nan={torch.isnan(x_low).any().item()}")
            print(f"  F_low: has_nan={torch.isnan(F_low).any().item()}, has_inf={torch.isinf(F_low).any().item()}")
        
        cov_phys, F_interp, _ = build_deformation_covariance(
            ref_points, x_low, F_low, knn, cov_cfg_ep, learnable_cov_module=None,
            x_low_normals=None,  # TODO: x_low의 노멀 계산 (현재는 비활성화)
            x_low_curvature=x_low_curvature  # 🔥 NEW: 곡률 기반 엣지 검출
        )
        
        # 🔥 NaN 체크: 출력 검증
        if verbose:
            print(f"[NaN Check - After Covariance]")
            print(f"  cov_phys: has_nan={torch.isnan(cov_phys).any().item()}, has_inf={torch.isinf(cov_phys).any().item()}")
            print(f"  F_interp: has_nan={torch.isnan(F_interp).any().item()}")
        
        # (옵션) Learnable 혼합
        learn_cfg = cov_cfg.get("learnable", {})
        if bool(learn_cfg.get("enabled", False)):
            alpha = float(learn_cfg.get("alpha", 0.3))  # 0.3 권장
            
            # 🔥 FIX: 파라미터 우선, 없으면 state 캐싱
            if learnable_cov_module is not None:
                # 외부에서 전달받은 모듈 사용 (학습 루프)
                cov = learnable_cov_module(cov_physics=cov_phys, alpha=alpha)
                if verbose:
                    print(f"  ✓ Learnable hybrid (external module): α={alpha:.2f}")
            else:
                # State 캐싱 사용 (단일 호출)
                if (state is None) or ("learn_cov" not in state) or (state["learn_cov"].M != cov_phys.shape[0]):
                    if state is None:
                        state = {}
                    state["learn_cov"] = LearnableCovariance(
                        M=cov_phys.shape[0], 
                        device=cov_phys.device,
                        dtype=cov_phys.dtype, 
                        init_scale=float(cov_cfg.get("sigma0", 0.08))
                    )
                cov = state["learn_cov"](cov_physics=cov_phys, alpha=alpha)
                if verbose:
                    print(f"  ✓ Learnable hybrid (state-cached): α={alpha:.2f}")
        else:
            cov = cov_phys
            if verbose:
                print(f"  ✓ Physics-only Σ")
        
        cov_key = "cov_pred"
    
    # ========================================================================
    # Prepare Output
    # ========================================================================
    debug_info = {
        "N_input": len(x_low),
        "M_output": len(ref_points),
        "upsampling_factor": len(ref_points) / len(x_low),
        "device": str(device),
        "seed": seed,
        "path": "SDF",
    }
    
    if verbose:
        print("\n" + "="*80)
        print("SDF Pipeline Complete!")
        print(f"  Input:  {debug_info['N_input']:,} volumetric points")
        print(f"  Output: {debug_info['M_output']:,} surface points")
        print(f"  Factor: {debug_info['upsampling_factor']:.1f}×")
        print("="*80 + "\n")
    
    # Visualization after STAGE 5
    if debug_cfg.get("export_stages", False):
        stage_outputs['stage5_points'] = ref_points
        stage_outputs['stage5_cov'] = cov
    
    # Cleanup
    if perf_cfg.get("clear_cache", True):
        if hasattr(knn, "clear_cache"):
            knn.clear_cache()
            if verbose:
                print("Cleanup: FAISS cache cleared\n")
        elif verbose:
            print("Cleanup: No cache to clear\n")
    
    # Build result
    # 🔥 CRITICAL: Detach tensors before numpy conversion (morphing mode has gradients)
    result = {
        "points": ref_points if return_torch else as_numpy(ref_points.detach() if hasattr(ref_points, 'detach') else ref_points),
        "normals": normals_final if return_torch else as_numpy(normals_final.detach() if hasattr(normals_final, 'detach') else normals_final),
        "F_interp": F_interp if return_torch else as_numpy(F_interp.detach() if hasattr(F_interp, 'detach') else F_interp),
        "anchors": anchors_sel if return_torch else as_numpy(anchors_sel.detach() if hasattr(anchors_sel, 'detach') else anchors_sel),
        "debug": debug_info,
        "state": state,
        "cov": cov if return_torch else as_numpy(cov.detach() if hasattr(cov, 'detach') else cov),
        "levelset": levelset,
    }
    result[cov_key] = result["cov"]  # Naming compatibility
    
    # Optional: Export φ grid
    if cfg.get("sdf", {}).get("export_grid", False):
        result["sdf_grid"] = levelset.phi if return_torch else as_numpy(levelset.phi)
        result["bbox_min"] = levelset.bbox_min if return_torch else as_numpy(levelset.bbox_min)
        result["bbox_max"] = levelset.bbox_max if return_torch else as_numpy(levelset.bbox_max)
    
    return result


__all__ = [
    "upsample",
]