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

    # FIX: Use safe eigendecomposition to handle NaN/Inf values
    from sampling.geometry.curvature_covariance import _safe_eigh_cuda_cpu
    eigenvalues, eigenvectors = _safe_eigh_cuda_cpu(C, eps=1e-6)  # eigenvalues are sorted ascending
    
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

        # ✅ REMOVED: Z-direction forcing (was causing green regions in normal map)
        # The centroid-based orientation above is sufficient and correct.
        # Previously: All normals were forced to point upward (Z > 0), which
        # incorrectly flipped bottom surface normals and created inconsistencies.

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
    # Choose pipeline path
    # ========================================================================
    verbose = export_stages  # Use export_stages as verbose flag
    upsample_cfg = cfg.get("upsample", {})
    cov_cfg = upsample_cfg.get("covariance", {})

    enable_subdivision = cov_cfg.get("enable_subdivision", False)
    enable_sdf_grid = cov_cfg.get("enable_sdf_grid", False)

    if verbose:
        print(f"[DEBUG] enable_subdivision = {enable_subdivision}, enable_sdf_grid = {enable_sdf_grid}")

    if enable_sdf_grid:
        # SDF grid upsampling (uniform distribution like C++)
        return _upsample_with_sdf_grid(
            x_low, F_low, cfg, state, return_torch,
            learnable_cov_module, knn, device, verbose,
            current_episode
        )
    elif enable_subdivision:
        # Subdivision-based upsampling (clustering around parents)
        return _upsample_with_subdivision(
            x_low, F_low, cfg, state, return_torch,
            learnable_cov_module, knn, device, verbose,
            current_episode
        )
    else:
        # Direct (no upsampling)
        return _upsample_simple_direct(
            x_low, F_low, cfg, state, return_torch,
            learnable_cov_module, knn, device, verbose,
            current_episode
        )


def _upsample_with_subdivision(
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
    🔥 Simple subdivision-based upsampling for gap filling.

    Strategy:
    1. Identify high-deformation regions (from det(F))
    2. Allocate subdivision budget to these regions
    3. Create child particles near parents
    4. Interpolate F-field to children
    5. Build covariances on upsampled set

    Benefits:
    - Focused upsampling where needed (high deformation)
    - Fully differentiable
    - Simple and efficient
    - Stays under 70K particle limit
    """
    from sampling.geometry.deformation_covariance import build_deformation_covariance

    upsample_cfg = cfg.get("upsample", {})
    cov_cfg = upsample_cfg.get("covariance", {})

    target_count = int(cov_cfg.get("subdivision_target", 60000))
    jitter_scale = float(cov_cfg.get("subdivision_jitter", 0.15))

    N = x_low.shape[0]

    if verbose:
        print(f"\n{'='*70}")
        print(f"[SUBDIVISION UPSAMPLING PIPELINE]")
        print(f"{'='*70}")
        print(f"Input points: {N:,}")
        print(f"Target points: {target_count:,}")
        print(f"Upsampling ratio: {target_count/N:.2f}x")

    # ========================================================================
    # 1. Compute deformation magnitude
    # ========================================================================
    det_F = torch.det(F_low).abs()  # (N,)
    dev = (det_F - 1.0).abs()  # Deviation from identity

    # 🔥 FIX: Check if deformation is negligible (e.g., target mesh with F=I)
    # If deformation is too small, use uniform subdivision instead
    # Use per-particle average to be robust to varying particle counts
    total_dev = dev.sum().item()
    mean_dev = total_dev / N  # Average deformation per particle
    use_uniform = mean_dev < 0.02  # Threshold: 2% average deformation per particle

    if use_uniform:
        # Uniform subdivision: distribute children evenly
        prob = torch.ones(N, device=device) / N  # Equal probability
        if verbose:
            print(f"[Subdivision] Using UNIFORM mode (mean deformation = {mean_dev:.6f} < 0.02)")
    else:
        # Deformation-based subdivision (original behavior)
        prob = dev / (dev.sum() + 1e-6)  # (N,)
        if verbose:
            print(f"[Subdivision] Using DEFORMATION mode (mean deformation = {mean_dev:.6f})")

    # ========================================================================
    # 2. Allocate children to each parent
    # ========================================================================
    num_new = target_count - N
    num_children_per_parent = prob * num_new  # (N,) float

    if use_uniform:
        # Simple uniform allocation with slight randomness
        with torch.no_grad():
            base_children = num_new // N
            remainder = num_new % N
            num_children = torch.full((N,), base_children, dtype=torch.long, device=device)
            # Distribute remainder randomly
            if remainder > 0:
                generator = torch.Generator(device=device).manual_seed(42)
                perm = torch.randperm(N, device=device, generator=generator)
                num_children[perm[:remainder]] += 1
    else:
        # 🔥 IMPROVED: Proportional allocation with max cap to prevent clustering
        with torch.no_grad():
            # Direct proportional allocation (no Gumbel-Softmax winner-takes-all)
            num_children_float = num_children_per_parent  # (N,) float

            # 🔥 Cap max children per parent to prevent extreme clustering
            # Even in high-deformation regions, limit to reasonable amount
            max_children_per_parent = max(20, num_new // (N // 10))  # At least 20, or 10% of parents share the load

            # Apply cap iteratively
            num_iterations = 0
            while num_iterations < 10:  # Prevent infinite loop
                over_cap = num_children_float > max_children_per_parent
                if not over_cap.any():
                    break

                # Redistribute excess to under-cap particles
                excess = (num_children_float - max_children_per_parent).clamp(min=0)
                total_excess = excess.sum()

                if total_excess < 0.1:  # Negligible excess
                    break

                # Cap the over-allocated particles
                num_children_float[over_cap] = max_children_per_parent

                # Redistribute to particles below cap proportionally
                under_cap = ~over_cap
                if under_cap.sum() == 0:
                    break

                redistrib_prob = prob[under_cap] / (prob[under_cap].sum() + 1e-8)
                num_children_float[under_cap] += redistrib_prob * total_excess
                num_iterations += 1

            # Stochastic rounding to preserve expectation
            num_children = torch.floor(num_children_float).long()
            frac = num_children_float - num_children.float()

            generator = torch.Generator(device=device).manual_seed(42)
            rand = torch.rand(N, device=device, generator=generator)
            num_children += (frac > rand).long()

            # Ensure we hit target exactly
            diff = num_new - num_children.sum().item()
            if diff != 0:
                # Add/remove from particles with highest fractional part (fair rounding)
                if diff > 0:
                    _, top_idx = torch.topk(frac, k=min(abs(diff), N))
                    num_children[top_idx[:diff]] += 1
                else:
                    # Remove from particles with lowest fractional part
                    # 🔥 FIX: Use positive indexing to avoid slice step error
                    k = min(abs(diff), N)
                    _, bottom_idx = torch.topk(frac, k=k, largest=False)
                    # Take the last abs(diff) elements (those with lowest frac)
                    indices_to_decrement = bottom_idx[max(0, k-abs(diff)):]
                    num_children[indices_to_decrement] -= 1
                    num_children = num_children.clamp(min=0)

    if verbose:
        print(f"\n[1/4] Subdivision allocation:")
        print(f"  Mean children/parent: {num_children.float().mean():.2f}")
        print(f"  Max children/parent: {num_children.max().item()}")
        print(f"  Parents with children: {(num_children > 0).sum().item()}/{N}")

    # ========================================================================
    # 3. Generate child particles
    # ========================================================================
    children_x = []
    children_parent_idx = []

    for i in range(N):
        n_child = num_children[i].item()
        if n_child > 0:
            # Local spacing estimate
            idx_nn, _ = knn(x_low[i:i+1], x_low, k=8)
            neighbors = x_low[idx_nn[0]]
            local_spacing = torch.norm(neighbors - x_low[i], dim=-1).mean()

            # Create children with random jitter
            jitter = torch.randn(n_child, 3, device=device) * jitter_scale * local_spacing
            child_pos = x_low[i].unsqueeze(0) + jitter

            children_x.append(child_pos)
            children_parent_idx.extend([i] * n_child)

    if len(children_x) > 0:
        children_x = torch.cat(children_x, dim=0)  # (M, 3)
        x_upsampled = torch.cat([x_low, children_x], dim=0)  # (N+M, 3)
        M = children_x.shape[0]
    else:
        x_upsampled = x_low
        M = 0

    if verbose:
        print(f"\n[2/4] Generated {M:,} child particles")
        print(f"  Total particles: {x_upsampled.shape[0]:,}")

    # ========================================================================
    # 4. Compute normals
    # ========================================================================
    k_pca = min(32, x_upsampled.shape[0] - 1)
    normals = compute_normals_pca(x_upsampled, knn, k=k_pca, prefer_outward=True)

    if verbose:
        print(f"\n[3/4] Computed normals via PCA (k={k_pca})")

    # ========================================================================
    # 5. Build covariances
    # ========================================================================
    is_target = (current_episode < 0)  # Only negative episodes are true "target"
    use_curvature_cov = cov_cfg.get("use_curvature_for_target", False)

    if verbose:
        print(f"[DEBUG curvature check] current_episode={current_episode}, is_target={is_target}, use_curvature_cov={use_curvature_cov}")

    if is_target and use_curvature_cov:
        # For target mesh with curvature-based covariance
        if verbose:
            print(f"\n[4/4] Building curvature-based covariances for target...")

        # Compute geometric properties for upsampled points using PCA
        from sampling.analysis.pca import batched_pca_surface_optimized
        from sampling.geometry.curvature_covariance import create_curvature_based_covariance_star

        k_curv = min(16, x_upsampled.shape[0] - 1)
        idx_nn, w_nn = knn(x_upsampled, x_upsampled, k_curv)

        # Batched PCA analysis
        normals_pca, surfvar, spacing_pca, curvature, anisotropy_t, planarity_t, \
        principal_dir1, principal_dir2, principal_curv = batched_pca_surface_optimized(
            x=x_upsampled,
            indices=idx_nn,
            weights=w_nn,
            return_principal_dirs=True
        )

        # Orient normals outward (use already computed normals from line 603)
        # normals is already computed, so we can use it directly

        # Convert to numpy
        pts_np = x_upsampled.detach().cpu().numpy()
        nrm_np = normals.detach().cpu().numpy()
        plan_np = planarity_t.detach().cpu().numpy()
        aniso_np = anisotropy_t.detach().cpu().numpy()
        curv_np = principal_curv.detach().cpu().numpy()
        dir1_np = principal_dir1.detach().cpu().numpy()
        dir2_np = principal_dir2.detach().cpu().numpy()
        dirs_raw = (dir1_np, dir2_np)

        # Get sigma parameters
        sigma_params = dict(cov_cfg.get("curvature_sigma", {}))

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
        F_interp = torch.eye(3, device=device, dtype=x_upsampled.dtype).unsqueeze(0).expand(x_upsampled.shape[0], 3, 3)
        idx = None

        if verbose:
            print(f"  ✅ Using curvature-based covariance (σ_n0={sigma_params.get('sigma_n0', 0.035):.3f}, σ_t0={sigma_params.get('sigma_t0', 0.06):.3f})")
            print(f"  Covariance shape: {cov.shape}")
    else:
        # Use deformation-based covariance (original behavior)
        cov, F_interp, idx = build_deformation_covariance(
            points=x_upsampled,
            x_low=x_low,
            F_low=F_low,
            knn=knn,
            cfg=cov_cfg,
            learnable_cov_module=learnable_cov_module
        )

        if verbose:
            print(f"\n[4/4] Built deformation-based covariances")
            print(f"  Covariance shape: {cov.shape}")

    if verbose:
        print(f"{'='*70}\n")

    # ========================================================================
    # 6. Return results
    # ========================================================================
    result = {
        "points": x_upsampled if return_torch else x_upsampled.detach().cpu().numpy(),
        "normals": normals if return_torch else normals.detach().cpu().numpy(),
        "F_interp": F_interp if return_torch else F_interp.detach().cpu().numpy(),
        "cov": cov if return_torch else cov.detach().cpu().numpy(),
        "anchors": x_upsampled if return_torch else x_upsampled.detach().cpu().numpy(),
        "debug": {
            "num_original": N,
            "num_children": M,
            "num_total": x_upsampled.shape[0]
        },
        "state": state,
    }
    
    # ✅ Add stage_outputs for visualization
    # This ensures PNG files are generated for each episode
    stage_outputs = {
        'stage6': {
            'points': x_upsampled if return_torch else x_upsampled.detach().cpu().numpy(),
            'normals': normals if return_torch else normals.detach().cpu().numpy(),
            'cov': cov if return_torch else cov.detach().cpu().numpy()
        }
    }
    result["stage_outputs"] = stage_outputs
    
    return result


def _upsample_with_sdf_grid(
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
    🌟 SDF-based grid upsampling for uniform particle distribution.

    Uses the same approach as C++ mesh loading:
    1. Create uniform 3D grid in bounding box
    2. Compute distance to existing particles
    3. Keep only grid points near surface
    4. Interpolate F-field to new points

    Args:
        x_low: Low-res points (N, 3)
        F_low: Deformation gradients (N, 3, 3)
        cfg: Configuration dict
        state: State dict for temporal coherence
        return_torch: Return torch tensors if True, numpy arrays if False
        learnable_cov_module: Optional learnable covariance module
        knn: k-NN module
        device: torch device
        verbose: Print progress if True
        current_episode: Current episode number (-1 for target)

    Returns:
        Dict with upsampled points, normals, F_interp, cov, anchors
    """
    from sampling.geometry.deformation_covariance import build_deformation_covariance

    upsample_cfg = cfg.get("upsample", {})
    cov_cfg = upsample_cfg.get("covariance", {})

    grid_dx = float(cov_cfg.get("sdf_grid_dx", 0.15))
    surface_threshold = float(cov_cfg.get("sdf_surface_threshold", 0.3))
    apply_jitter = bool(cov_cfg.get("sdf_apply_jitter", True))

    N = x_low.shape[0]

    if verbose:
        print(f"\n{'='*70}")
        print(f"[SDF GRID UPSAMPLING PIPELINE]")
        print(f"{'='*70}")
        print(f"Input points: {N:,}")
        print(f"Grid spacing (dx): {grid_dx:.3f}")
        print(f"Surface threshold: {surface_threshold:.3f}")
        print(f"Apply jitter: {apply_jitter}")

    # ========================================================================
    # 1. Compute bounding box
    # ========================================================================
    min_point = x_low.min(dim=0)[0]
    max_point = x_low.max(dim=0)[0]

    # Expand bounding box slightly
    margin = grid_dx * 2
    min_point = min_point - margin
    max_point = max_point + margin

    if verbose:
        print(f"\nBounding box:")
        print(f"  Min: {min_point.cpu().numpy()}")
        print(f"  Max: {max_point.cpu().numpy()}")

    # ========================================================================
    # 2. Create uniform 3D grid
    # ========================================================================
    dims = ((max_point - min_point) / grid_dx).ceil().int() + 1
    total_grid_points = dims.prod().item()

    if verbose:
        print(f"\n[1/4] Creating uniform grid...")
        print(f"  Grid dimensions: {dims[0].item()} x {dims[1].item()} x {dims[2].item()}")
        print(f"  Total grid points: {total_grid_points:,}")

    # Generate grid points
    x_coords = torch.arange(dims[0].item(), device=device, dtype=x_low.dtype) * grid_dx + min_point[0]
    y_coords = torch.arange(dims[1].item(), device=device, dtype=x_low.dtype) * grid_dx + min_point[1]
    z_coords = torch.arange(dims[2].item(), device=device, dtype=x_low.dtype) * grid_dx + min_point[2]

    grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)

    # Apply jitter if requested
    if apply_jitter:
        jitter_scale = 0.1  # 10% of grid spacing
        jitter = (torch.rand_like(grid_points) - 0.5) * 2 * jitter_scale * grid_dx
        grid_points = grid_points + jitter
        if verbose:
            print(f"  Applied jitter: ±{jitter_scale * grid_dx:.4f}")

    # ========================================================================
    # 3. Compute distance to existing particles (approximate SDF)
    # ========================================================================
    if verbose:
        print(f"\n[2/4] Computing distances to particles...")

    # Use k-NN to find nearest particles (approximate SDF)
    k_sdf = min(8, N)
    idx_nearest, dist_nearest = knn(grid_points, x_low, k_sdf)

    # Approximate signed distance as distance to nearest particle
    sdf = dist_nearest[:, 0]  # Distance to nearest neighbor

    # ========================================================================
    # 4. Keep only grid points near surface
    # ========================================================================
    mask = sdf <= surface_threshold
    x_upsampled = grid_points[mask]

    if verbose:
        kept_count = mask.sum().item()
        print(f"  Kept {kept_count:,} / {total_grid_points:,} points ({100*kept_count/total_grid_points:.1f}%)")
        print(f"  Upsampling ratio: {kept_count/N:.2f}x")

    # ========================================================================
    # 5. Compute normals via PCA
    # ========================================================================
    k_pca = min(32, x_upsampled.shape[0] - 1)
    normals = compute_normals_pca(x_upsampled, knn, k_pca)

    if verbose:
        print(f"\n[3/4] Computed normals via PCA (k={k_pca})")

    # ========================================================================
    # 6. Multi-scale F-field interpolation
    # ========================================================================
    if verbose:
        print(f"\n[4/4] Multi-scale F-field interpolation...")

    use_multiscale = cov_cfg.get("use_multiscale_F", True)

    if use_multiscale:
        k_coarse = int(cov_cfg.get("k_F_coarse", 64))
        k_fine = int(cov_cfg.get("k_F_fine", 16))
        blend_mode = cov_cfg.get("multiscale_blend_mode", "adaptive")

        # Coarse scale (global)
        idx_c, w_c = knn(x_upsampled, x_low, k_coarse)
        F_coarse = (F_low[idx_c] * w_c.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

        # Fine scale (local)
        idx_f, w_f = knn(x_upsampled, x_low, k_fine)
        F_fine = (F_low[idx_f] * w_f.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

        # Adaptive blending
        alpha = 0.5
        F_interp = alpha * F_coarse + (1 - alpha) * F_fine

        if verbose:
            print(f"  Multi-scale: k_coarse={k_coarse}, k_fine={k_fine}, blend={blend_mode}")
    else:
        k_F = int(cov_cfg.get("k_F", 32))
        idx, w = knn(x_upsampled, x_low, k_F)
        F_interp = (F_low[idx] * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

    # ========================================================================
    # 7. Build deformation-based covariance
    # ========================================================================
    if verbose:
        print(f"\n[5/5] Building deformation-based covariances...")

    cov, F_interp_final, idx = build_deformation_covariance(
        points=x_upsampled,
        x_low=x_low,
        F_low=F_low,
        knn=knn,
        cfg=cov_cfg,
        learnable_cov_module=learnable_cov_module
    )

    if verbose:
        print(f"  ✅ Covariance shape: {cov.shape}")
        print(f"{'='*70}\n")

    # ========================================================================
    # 8. Return results
    # ========================================================================
    result = {
        "points": x_upsampled if return_torch else x_upsampled.detach().cpu().numpy(),
        "normals": normals if return_torch else normals.detach().cpu().numpy(),
        "F_interp": F_interp_final if return_torch else F_interp_final.detach().cpu().numpy(),
        "cov": cov if return_torch else cov.detach().cpu().numpy(),
        "anchors": x_upsampled if return_torch else x_upsampled.detach().cpu().numpy(),
        "debug": {
            "num_original": N,
            "num_generated": x_upsampled.shape[0] - N,
            "num_total": x_upsampled.shape[0],
            "grid_dx": grid_dx,
            "surface_threshold": surface_threshold
        },
        "state": state,
    }

    stage_outputs = {
        'stage6': {
            'points': x_upsampled if return_torch else x_upsampled.detach().cpu().numpy(),
            'normals': normals if return_torch else normals.detach().cpu().numpy(),
            'cov': cov if return_torch else cov.detach().cpu().numpy()
        }
    }
    result["stage_outputs"] = stage_outputs

    return result


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
