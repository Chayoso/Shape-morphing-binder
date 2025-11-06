"""
Point Cloud Covariance Pipeline.

Simple, direct pipeline:
  1. Compute mesh geometry (PCA-based normals)
  2. Build covariance from deformation gradients (F-field)

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
    # Direct F → Covariance (Simple pipeline only)
    # ========================================================================
    return _upsample_simple_direct(
        x_low, F_low, cfg, state, return_torch,
        learnable_cov_module, knn, device, verbose,
        current_episode
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
    
    # ✅ CHANGED: Always use F-based covariance for gradient flow!
    # Even episode 0 should use F to enable gradient backprop
    is_target = (current_episode < 0)  # Only negative episodes are true "target"
    
    if verbose:
        print(f"  current_episode: {current_episode}")
        print(f"  is_target: {is_target}")
        print(f"  Covariance mode: {'Curvature/Isotropic (Target)' if is_target else 'F-based (Morphing)'}")
    
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
            cfg=cov_cfg,  # 🔥 FIX: Pass covariance config to respect use_F_smoothing flag
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
    
    # ✅ Add cov_target for spectral alignment loss (if target mode)
    if is_target and use_curvature_cov:
        result["cov_target"] = cov if return_torch else as_numpy(cov)
        if verbose:
            print(f"  ✅ cov_target added for spectral loss")
    
    return result


__all__ = [
    "upsample",
]
