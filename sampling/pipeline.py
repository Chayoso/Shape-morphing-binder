"""
Main Differentiable Point Cloud Upsampling Pipeline.

This module implements a complete 6-stage pipeline for high-quality point cloud
upsampling with deformation-aware covariance estimation. All stages are fully
differentiable for end-to-end learning.

═══════════════════════════════════════════════════════════════════════════════
PIPELINE OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

Input:  N sparse anchors with deformation gradients {xᵢ, Fᵢ}
Output: M dense points with anisotropic covariances {pⱼ, Σⱼ}  (M >> N)

         ┌────────────────────────────────────────────────────┐
         │  INPUT: Sparse Point Cloud + Deformation Field     │
         │  • x_low: (N, 3) anchor positions                  │
         │  • F_low: (N, 3, 3) deformation gradients          │
         └─────────────────┬──────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────────────┐
         │  STAGE 1: Surface Detection (PCA Analysis)          │
         │  ─────────────────────────────────────────          │
         │  • Weighted PCA on local neighborhoods              │
         │  • Extract: normals, surface variance, spacing      │
         │  • Compute: surf_prob = f(planarity, EMA)           │
         │                                                     │
         │  Output: {normals, surf_prob, spacing}              │
         └─────────────────┬───────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────────────┐
         │  STAGE 2: Anchor-Density Map (Differentiable)       │
         │  ────────────────────────────────────────           │
         │  • Build soft k-NN kernel density: ρᵢ               │
         │  • Normalize to stable range [0.25, 4.0]            │
         │  • Prepare sampler cfg with density bias            │
         │                                                     │
         │  Output: {rho_anchor, cfg_out, state}               │
         └─────────────────┬───────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────────────┐
         │  STAGE 3: Importance Sampling (Gumbel-Softmax)      │
         │  ──────────────────────────────────────────         │
         │  • Sample M indices: Y ~ GumbelSoftmax(probs, τ)    │
         │  • Interpolate: anchors = Y @ x_low                 │
         │  • Build tangent frame {t₁, t₂, n}                  │
         │  • Jitter: p = anchor + α·h·(U·t₁ + V·t₂) + ε       │
         │                                                     │
         │  Output: {points (M,3), normals_up, anchors}        │
         └─────────────────┬───────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────────────┐
         │  STAGE 4: Taubin Smoothing (Shrinkage-Free)         │
         │  ────────────────────────────────────────           │
         │  • Laplacian pass:  p' = p + λ·L·p                  │
         │  • Inflation pass:  p" = p' + μ·L·p'                │ 
         │  • Constraint: tangent motion only (preserve n)     │
         │                                                     │
         │  Output: {smoothed_points}                          │
         └─────────────────┬───────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────────────┐
         │  STAGE 5: Normal Smoothing (Spatial Laplacian)      │
         │  ───────────────────────────────────────────        │
         │  • Adaptive bandwidth: h = soft_median(distances)   │
         │  • Spatial weights: w = exp(-d²/h²)                 │ 
         │  • Smooth: n' = normalize(Σ wᵢ·nᵢ)                  │
         │  • EMA blend: n ← λ·n' + (1-λ)·n                    │
         │                                                     │
         │  Output: {smoothed_normals}                         │
         └─────────────────┬───────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────────────┐
         │  STAGE 6: Covariance Construction (F-field)         │
         │  ────────────────────────────────────────           │
         │  • Smooth F-field: Graph Laplacian on anchors       │
         │  • Interpolate F to upsampled points via KNN        │
         │  • Polar decomposition: F = R·S                     │
         │  • Build covariance: Σ = R·S·Σ₀·S·Rᵀ                │
         │                                                     │
         │  Output: {cov (M,3,3), F_interp}                    │
         └─────────────────┬───────────────────────────────────┘
                           │
                           ▼
         ┌────────────────────────────────────────────────────┐
         │  OUTPUT: Dense Point Cloud with Covariances        │
         │  • points: (M, 3) upsampled positions              │
         │  • normals: (M, 3) smoothed normals                │
         │  • cov: (M, 3, 3) anisotropic covariances          │
         │  • F_interp: (M, 3, 3) interpolated F-field        │
         └────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
KEY INNOVATIONS
═══════════════════════════════════════════════════════════════════════════════

1. **Differentiable Importance Sampling**
   - Gumbel-Softmax: differentiable categorical sampling
   - Learn where to upsample via gradient descent
   - Focuses computation on complex/uncertain regions

2. **Surface-Aligned Jittering**
   - Tangent space perturbations preserve surface structure
   - Adaptive scaling based on local point density
   - Multi-scale jitter (tangent + normal + micro)

3. **Shrinkage-Free Smoothing**
   - Taubin's λ-μ scheme prevents volume loss
   - Tangent-only motion respects surface constraints
   - Differentiable for end-to-end learning

4. **Deformation-Aware Covariances**
   - F-field smoothing via graph Laplacian
   - Polar decomposition: separates rotation & stretch
   - Anisotropic Gaussians aligned with deformation

5. **End-to-End Differentiability**
   - All operations support gradient flow
   - Enables learning: sampling distribution, smoothing params, etc.
   - Joint optimization with neural networks

═══════════════════════════════════════════════════════════════════════════════
MATHEMATICAL FORMULATION
═══════════════════════════════════════════════════════════════════════════════

Stage 1: Surface Detection
──────────────────────────
    C = (1/Σwᵢ) · Σᵢ wᵢ·(xᵢ - c)(xᵢ - c)ᵀ      [Weighted covariance]
    λ₀ ≤ λ₁ ≤ λ₂ = eig(C)                      [Eigenvalues]
    surfvar = λ₀ / (λ₀ + λ₁ + λ₂)              [Planarity metric]
    surf_prob = ema(1 - surfvar)               [Importance weight]

Stage 2: Anchor-Density Map
────────────────────────────
    ρᵢ = Σⱼ exp(-(dᵢⱼ/hᵢ)²) · αⱼ              [Soft kernel density]
    ρ' = ρ / mean(ρ)                           [Normalize]
    ρ_anchor = clamp(ρ', 0.25, 4.0)            [Stable range]

Stage 3: Importance Sampling
─────────────────────────────
    Y = GumbelSoftmax(probs, τ)                [Sample M from N]
    p = Y @ x                                  [Interpolate]
    p_jitter = p + α·h·(U·t₁ + V·t₂) + ε       [Surface jitter]

Stage 4: Taubin Smoothing
──────────────────────────
    L = D - W                                  [Laplacian matrix]
    p⁽¹⁾ = p + λ·L·p                           [Smooth pass]
    p⁽²⁾ = p⁽¹⁾ + μ·L·p⁽¹⁾                     [Inflate pass]
    p_tangent = p⁽²⁾ - (p⁽²⁾·n)·n              [Tangent constraint]

Stage 5: Normal Smoothing
──────────────────────────
    h = soft_median({‖xᵢ - xⱼ‖}ⱼ)              [Adaptive bandwidth]
    wᵢⱼ = exp(-‖xᵢ - xⱼ‖²/h²)                  [Spatial weights]
    n' = normalize(Σⱼ wᵢⱼ·nⱼ)                  [Smooth]
    n ← λ·n' + (1-λ)·n                         [EMA blend]

Stage 6: Covariance Construction
─────────────────────────────────
    F_smooth = (WᵀW + λ·L)⁻¹·Wᵀ·F              [Graph Laplacian]
    F_interp = Σⱼ wⱼ·F_j                       [KNN interpolation]
    R, S = polar_decomp(F_interp)              [Rotation & stretch]
    Σ = R·S·Σ₀·S·Rᵀ                            [Anisotropic covariance]

═══════════════════════════════════════════════════════════════════════════════
TYPICAL PARAMETERS
═══════════════════════════════════════════════════════════════════════════════

Surface Detection:
    k: 48                  # Neighbors for PCA
    ema_alpha: 0.3         # EMA decay for temporal stability

Anchor-Density Map:
    stage2_k: 16           # Neighbors for density
    anchor_density_beta: 0.7   # Bias strength (sparse preference)
    spacing_bias_gamma: 0.6    # Fallback bias exponent

Importance Sampling:
    M: 50000               # Upsampling factor (5-10× of N)
    tau: 0.2               # Gumbel-Softmax temperature
    alpha: 0.35            # Jitter magnitude
    thickness: 0.0         # Normal offset (0 = 2D surface)

Taubin Smoothing:
    iters: 3               # Number of λ-μ passes
    lambda_smooth: 0.33    # Smoothing weight (positive)
    lambda_inflate: -0.53  # Inflation weight (negative)
    k: 24                  # Neighbors for Laplacian

Normal Smoothing:
    iters: 2               # Smoothing iterations
    k: 16                  # Spatial neighbors
    lambda_smooth: 0.8     # EMA blend factor

Covariance Construction:
    k_F: 32                # Neighbors for F interpolation
    sigma0: 0.08           # Base Gaussian scale
    use_polar: True        # Polar decomposition (recommended)
    F_smooth:
        num_nodes: 180     # Graph nodes for Laplacian
        node_knn: 8        # Graph connectivity
        lambda_lap: 1e-2   # Laplacian weight

═══════════════════════════════════════════════════════════════════════════════
COMPLEXITY ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

Stage               Time Complexity        Memory          Bottleneck
───────────────────────────────────────────────────────────────────────────────
Surface Detection   O(N·k·3)              O(N·k)          PCA (lightweight)
Anchor-Density Map  O(N·k)                O(N·k)          KNN + kernel (fast)
Importance Sampling O(M·N)                O(M·N)          Gumbel-Softmax ⚠️
Taubin Smoothing    O(M·k·iters)          O(M·k)          Laplacian (medium)
Normal Smoothing    O(M·k²·iters)         O(M·k²)         Soft median ⚠️
Covariance          O(N·K + M·k)          O(M·k)          F interpolation
───────────────────────────────────────────────────────────────────────────────
Total               O(M·N + M·k²)         O(M·N)          Sampling dominant

Typical: N=10k, M=100k, k=32 → ~10GB memory, ~5 seconds on GPU

Memory optimization:
- Batch Gumbel-Softmax: O(batch_size·N) instead of O(M·N)
- Sparse KNN results: store only k neighbors per point
- Clear FAISS cache after each stage

═══════════════════════════════════════════════════════════════════════════════

Author: CHAYO
Version: 2.1.0 (Fully Documented)
"""

import torch
from typing import Dict, Optional
from pathlib import Path

from .utils.config import default_cfg, validate_cfg
from .utils.utils import ensure_torch, as_numpy
from .utils.validation import check_config_and_warn
from .analysis.knn import HybridFAISSKNN, FAISS_AVAILABLE
from .core.surface_detect import detect_surface
from .core.density_map import run_stage2_anchor_density
from .core.sampler import sample_points
from .core.taubin_smooth import taubin_smooth
from .core.normal_smooth import smooth_normals
from .geometry.covariance import build_covariance


def upsample(
    x_low: torch.Tensor,
    F_low: torch.Tensor,
    cfg: Optional[Dict] = None,
    state: Optional[Dict] = None,
    seed: int = 1234,
    return_torch: bool = True,
    export_stages: bool = False,
    learnable_cov_module=None  # 🔥 NEW: Optional learnable covariance module
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
    
    # Setup KNN module
    knn_cfg = cfg.get("knn", {})
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
    
    # ========================================================================
    # STAGE 1: Surface Detection (PCA-based Planarity)
    # ========================================================================
    # Get config
    surf_cfg = cfg.get("surface_detection", {})
    use_anisotropic_jitter = cfg.get("sampling", {}).get("use_anisotropic_jitter", False)
    
    # 🔥 IMPORTANT: Always compute surface detection!
    # x and F change every Pass, so cached surface would be invalid.
    # Caching only makes sense if x/F are identical across multiple upsample() calls.
    
    if verbose:
        print("\n" + "="*80)
        print("STAGE 1/6: Surface Detection (PCA-based Planarity)")
        print("="*80)
    
    if surf_cfg.get("enabled", True):
        result = detect_surface(
            x_low, knn, surf_cfg, state, return_curvature_dirs=use_anisotropic_jitter
        )
    
        if use_anisotropic_jitter:
            surf_prob, normals, spacing, state, principal_dir1, principal_dir2, principal_curv = result
        else:
            surf_prob, normals, spacing, state = result
            principal_dir1 = principal_dir2 = principal_curv = None
    else:
        # Uniform probability fallback
        N = x_low.shape[0]
        surf_prob = torch.full((N,), 1.0 / N, device=device)
        
        # Still need normals for downstream stages
        from .analysis.pca import batched_pca_surface_optimized
        k = surf_cfg.get("k", 48)
        idx, w = knn(x_low, x_low, k)
        normals, _, spacing = batched_pca_surface_optimized(x_low, idx, w)
        principal_dir1 = principal_dir2 = principal_curv = None
    
    # ========================================================================
    # Filter to Effective Surface Anchors (Integrate into Stage 1)
    # ========================================================================
    # 🔥 Filter to surface anchors (always execute, cache contains unfiltered data)
    prob_threshold = 1e-12  # Keep only anchors with non-zero probability
    gate_eval = (surf_prob > prob_threshold)
    
    n_total = x_low.shape[0]
    n_surface = int(gate_eval.sum().item())
    
    # Filter to surface anchors only
    x_low = x_low[gate_eval]
    normals = normals[gate_eval]
    spacing = spacing[gate_eval]
    F_low = F_low[gate_eval]
    
    # Renormalize probability over surface anchors
    surf_prob = surf_prob[gate_eval]
    surf_prob = surf_prob / (surf_prob.sum() + 1e-8)
    
    # Filter principal curvature data if needed (use_anisotropic_jitter already defined above)
    if use_anisotropic_jitter and principal_curv is not None:
        principal_dir1 = principal_dir1[gate_eval]
        principal_dir2 = principal_dir2[gate_eval]
        principal_curv = principal_curv[gate_eval]
    
    N = x_low.shape[0]  # Update N
    
    if verbose:
        print(f"✓ Surface detection: {n_total:,} → {n_surface:,} anchors ({100*n_surface/n_total:.1f}%)")
        print(f"  Mean prob: {surf_prob.mean():.6f}")
        
        if use_anisotropic_jitter and principal_curv is not None:
            print(f"  [Anisotropic Jitter] ✓ ENABLED")
            print(f"    - Principal curvatures: k1=[{principal_curv[:,0].min():.3f}, {principal_curv[:,0].max():.3f}], "
                  f"k2=[{principal_curv[:,1].min():.3f}, {principal_curv[:,1].max():.3f}]")
        else:
            print(f"  [Anisotropic Jitter] ⊘ DISABLED (isotropic jitter)")
    
    # Export Stage 1 data
    if export_stages:
        surface_mask = torch.ones_like(surf_prob, dtype=torch.bool)  # All are surface now
        
        stage_out = {
            'points': x_low.detach().clone() if return_torch else as_numpy(x_low),
            'surf_prob': surf_prob.detach().clone() if return_torch else as_numpy(surf_prob),
            'normals': normals.detach().clone() if return_torch else as_numpy(normals),
            'spacing': spacing.detach().clone() if return_torch else as_numpy(spacing),
            'surface_mask': surface_mask.detach().clone() if return_torch else as_numpy(surface_mask),
        }
        
        if use_anisotropic_jitter and principal_curv is not None:
            stage_out['principal_dir1'] = principal_dir1.detach().clone() if return_torch else as_numpy(principal_dir1)
            stage_out['principal_dir2'] = principal_dir2.detach().clone() if return_torch else as_numpy(principal_dir2)
            stage_out['principal_curv'] = principal_curv.detach().clone() if return_torch else as_numpy(principal_curv)
        
        stage_outputs['stage1'] = stage_out
    
    # ========================================================================
    # STAGE 2: Anchor-Density Map (Differentiable Density Estimation)
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 2/6: Anchor-Density Map (Differentiable Density Estimation)")
        print("="*80)
    
    # Get anchor_density config section (handle potential tensor collision)
    density_cfg_raw = cfg.get("anchor_density", {})
    if isinstance(density_cfg_raw, dict):
        density_cfg = density_cfg_raw
    else:
        # If anchor_density was overwritten by tensor, look in original cfg
        density_cfg = {}
    
    if density_cfg.get("enabled", True):
        # 🔥 NEW: Pass surf_prob to compute surface-weighted density
        rho_anchor, cfg_out, state = run_stage2_anchor_density(
            x_low, spacing, knn, density_cfg, state, surf_prob=surf_prob
        )
        
        # Merge density config into cfg for Stage 3
        cfg.update(cfg_out)
        
        if verbose:
            print(f"✓ Built anchor-density map for {len(x_low)} points")
            
            # 🔥 NEW: Surface weighting info
            if state.get('surf_weighted', False):
                print(f"  [Surface-Weighted Density] ✓ ENABLED")
                print(f"    - surf_prob: [{state.get('surf_prob_min', 0.0):.3f}, {state.get('surf_prob_max', 0.0):.3f}] "
                      f"(mean: {state.get('surf_prob_mean', 0.0):.3f})")
                print(f"    - Neighbors weighted by surface probability")
            else:
                print(f"  [Surface-Weighted Density] ⊘ DISABLED (all anchors used)")
            
            print(f"  Density stats:")
            print(f"    - mean: {state.get('rho_anchor_mean', 0.0):.3f}")
            print(f"    - min:  {state.get('rho_anchor_min', 0.0):.3f}")
            print(f"    - max:  {state.get('rho_anchor_max', 0.0):.3f}")
            
            # Correlation validation
            corr = state.get('spacing_density_corr', None)
            validation = state.get('density_validation', 'N/A')
            surf_weighted = state.get('surf_weighted', False)
            if corr is not None:
                print(f"  Correlation check:")
                print(f"    - corr(spacing, 1/ρ): {corr:.3f} [{validation}]")
                if "PASS" in validation:
                    if surf_weighted:
                        print(f"    ✓ Surface regions have high density (expected for surface-weighted)")
                    else:
                        print(f"    ✓ Sparse regions have low density (expected)")
                elif "WEAK" in validation:
                    print(f"    ⚠ Weak correlation (check k or bandwidth)")
                elif "FAIL" in validation:
                    if surf_weighted:
                        print(f"    ✗ Unexpected positive correlation for surface-weighted density")
                    else:
                        print(f"    ✗ Unexpected negative correlation")
            
            beta = cfg.get("anchor_density_beta", 0.7)
            gamma = cfg.get("spacing_bias_gamma", 0.6)
            print(f"  Sampler bias: β={beta:.2f}, γ={gamma:.2f}")
    else:
        rho_anchor = None
        if verbose:
            print("⊘ Anchor-density map disabled (uniform sampling)")
    
    # Export Stage 2 data
    # 🔥 FIXED: x_low is already filtered to surface anchors in Stage 1.5
    # No need to apply mask again - just export all (they're all surface anchors)
    if export_stages:
        x_surface = x_low.detach().clone() if return_torch else as_numpy(x_low)
        rho_surface = rho_anchor.detach().clone() if (rho_anchor is not None and return_torch) else (as_numpy(rho_anchor) if rho_anchor is not None else None)
        
        if verbose:
            print(f"  [Export] Exporting {len(x_low):,} surface anchors (pre-filtered in Stage 1.5)")
        
        stage_outputs['stage2'] = {
            'points': x_surface,
            'rho_anchor': rho_surface,
        }
    
    # ========================================================================
    # STAGE 3: Importance Sampling (Gumbel-Softmax + Tangent Jitter)
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 3/6: Importance Sampling (Gumbel-Softmax + Tangent Jitter)")
        print("="*80)

    # Get sampling config (handle nested structure: sampling.sampling)
    samp_cfg_outer = cfg.get("sampling", {})
    samp_cfg = samp_cfg_outer.get("sampling", samp_cfg_outer)  # Try nested first, fallback to outer
    
    # Merge anchor-density settings into samp_cfg (from STAGE 2)
    # 🔥 FIXED: Check if use_anchor_density is set in samp_cfg (user preference)
    user_wants_density = samp_cfg.get("use_anchor_density", cfg.get("use_anchor_density", False))
    
    if user_wants_density and cfg.get("use_anchor_density", False):
        samp_cfg["use_anchor_density"] = True
        samp_cfg["anchor_density_values"] = cfg.get("anchor_density_values")
        samp_cfg["anchor_density_beta"] = cfg.get("anchor_density_beta")
        samp_cfg["spacing_bias_gamma"] = cfg.get("spacing_bias_gamma")
    else:
        samp_cfg["use_anchor_density"] = False
    
    # 🔥 NEW: Pass principal curvature directions for anisotropic jitter
    if use_anisotropic_jitter and principal_dir1 is not None:
        samp_cfg["use_anisotropic_jitter"] = True
        samp_cfg["principal_dir1"] = principal_dir1
        samp_cfg["principal_dir2"] = principal_dir2
        samp_cfg["principal_curv"] = principal_curv
    
    N = len(x_low)  # number of anchors before upsampling
    
    # Extract key parameters for logging (all are actually used in sampler)
    M = int(samp_cfg.get("M", 50000))
    tau = float(samp_cfg.get("tau", 0.2))
    alpha = float(samp_cfg.get("alpha", 0.35))
    thickness = float(samp_cfg.get("thickness", 0.0))
    gs_batch = int(samp_cfg.get("gs_batch", 2048))
    ensure_cover = bool(samp_cfg.get("ensure_anchor_coverage", True))
    micro_jitter_scale = float(samp_cfg.get("micro_jitter_scale", 0.2))
    tangent_micro_only = bool(samp_cfg.get("tangent_micro_only", True))
    
    # Hole-fix patches
    prob_floor = float(samp_cfg.get("prob_floor", 1e-8))
    uniform_mix = float(samp_cfg.get("uniform_mix", 0.02))
    plane_snap = bool(samp_cfg.get("plane_snap", True))
    plane_snap_beta = float(samp_cfg.get("plane_snap_beta", 0.5))
    topk_pool = int(samp_cfg.get("topk_pool", 8))
    thickness_gamma = float(samp_cfg.get("thickness_gamma", 0.15))
    
    # Surface-constrained sampling
    prob_floor_mode = str(samp_cfg.get("prob_floor_mode", "density"))
    uniform_mix_surface_only = bool(samp_cfg.get("uniform_mix_surface_only", True))
    coverage_only_surface = bool(samp_cfg.get("coverage_only_surface", True))
    mask_topk_with_surface = bool(samp_cfg.get("mask_topk_with_surface", True))
    
    # Density-based floor
    density_floor_tau = float(samp_cfg.get("density_floor_tau", 1.0))
    density_floor_gamma = float(samp_cfg.get("density_floor_gamma", 2.0))
    
    # One-sided thickness
    thickness_one_sided = bool(samp_cfg.get("thickness_one_sided", True))
    inside_barrier_lambda = float(samp_cfg.get("inside_barrier_lambda", 1.0))

    # Rough peak per batch for logits/softmax (float32)
    est_mb_per_batch = (gs_batch * max(N, 1) * 4) / (1024**2)

    if verbose:
        print(f"- Total anchors (N): {N:,}")
        print(f"  └─ (All N anchors are already filtered surface anchors)")
        print(f"- Target samples (M): {M:,}  (upsampling {M/max(N,1):.1f}×)")
        
        # Check if density bias is active
        use_density = samp_cfg.get("use_anchor_density", False)
        has_rho = samp_cfg.get("anchor_density_values") is not None
        if use_density:
            if has_rho:
                print(f"\n  [Anchor-Density Bias] ✓ ACTIVE")
                print(f"  • rho_anchor shape: {samp_cfg['anchor_density_values'].shape}")
                print(f"  • beta (sparse bias): {samp_cfg.get('anchor_density_beta', 0.7):.2f}")
            else:
                print(f"\n  [Anchor-Density Bias] ⚠ FALLBACK to spacing")
                print(f"  • gamma (spacing bias): {samp_cfg.get('spacing_bias_gamma', 0.6):.2f}")
        
        print(f"\n  [Core Sampling]")
        print(f"  • Gumbel tau: {tau:.3f} | alpha: {alpha:.3f} | thickness: {thickness:.3f}")
        print(f"  • gs_batch: {gs_batch} (≈ {est_mb_per_batch:.1f} MB/batch)")
        print(f"  • micro_jitter: {micro_jitter_scale:.3f} (tangent_only={tangent_micro_only})")
        print(f"\n  [Hole-Fix Patches]")
        print(f"  • prob_floor: {prob_floor:.1e} | uniform_mix: {uniform_mix:.3f}")
        print(f"  • plane_snap: {plane_snap} (beta={plane_snap_beta:.2f})")
        print(f"  • topk_pool: {topk_pool} | thickness_gamma: {thickness_gamma:.3f}")
        print(f"\n  [Surface-Constrained]")
        print(f"  • prob_floor_mode: '{prob_floor_mode}'")
        print(f"  • uniform_mix_surface_only: {uniform_mix_surface_only}")
        print(f"  • coverage_only_surface: {coverage_only_surface}")
        print(f"  • mask_topk_with_surface: {mask_topk_with_surface}")
        print(f"\n  [Density-Based Floor]")
        print(f"  • density_floor_tau: {density_floor_tau:.2f}")
        print(f"  • density_floor_gamma: {density_floor_gamma:.2f}")
        print(f"\n  [One-Sided Thickness]")
        print(f"  • thickness_one_sided: {thickness_one_sided}")
        print(f"  • inside_barrier_lambda: {inside_barrier_lambda:.2f}")
        print(f"\n  [Coverage]")
        print(f"  • ensure_anchor_coverage: {ensure_cover}")
        
        # Warnings
        if micro_jitter_scale > 0.25 and not tangent_micro_only:
            print(f"\n  [WARN] High micro_jitter_scale with isotropic micro may increase interior leakage")
        if thickness > 0.0 and not thickness_one_sided:
            print(f"\n  [WARN] Symmetric thickness can create interior points")

    # Call sampler
    points, normals_up, anchors, anchor_selection_count = sample_points(
        x_low, normals, spacing, surf_prob, samp_cfg, generator
    )

    # Proactive cleanup (helps control transient peaks between stages)
    del surf_prob, spacing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Result summary
    if verbose:
        up_factor = len(points) / max(N, 1)
        print(f"\n✓ Sampled {len(points):,} points from {N:,} anchors ({up_factor:.1f}×)")
        if len(points) != M:
            print(f"  [INFO] Expected {M:,} but got {len(points):,} points")
        
        # Heat map stats
        total_selections = anchor_selection_count.sum().item()
        max_selections = anchor_selection_count.max().item()
        min_selections = anchor_selection_count.min().item()
        avg_selections = total_selections / N
        print(f"  [Heat Map] Selections per anchor: min={min_selections}, max={max_selections}, avg={avg_selections:.1f}")
    
    # Export Stage 3 data
    if export_stages:
        stage_outputs['stage3'] = {
            'points': points.detach().clone() if return_torch else as_numpy(points),
            'anchors': anchors.detach().clone() if return_torch else as_numpy(anchors),
            'normals': normals_up.detach().clone() if return_torch else as_numpy(normals_up),
            'anchor_positions': x_low.detach().clone() if return_torch else as_numpy(x_low),
            'anchor_selection_count': anchor_selection_count.detach().clone() if return_torch else as_numpy(anchor_selection_count),
        }


    
    # ========================================================================
    # STAGE 4: Taubin Smoothing (Shrinkage-Free Laplacian)
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 4/6: Taubin Smoothing (Shrinkage-Free Laplacian)")
        print("="*80)
    
    taubin_cfg = cfg.get("taubin", {})
    if taubin_cfg.get("enabled", True):
        points = taubin_smooth(points, normals_up, knn, taubin_cfg)
        
        if verbose:
            n_iters = taubin_cfg.get('iters', 3)
            lam = taubin_cfg.get('lambda_smooth', 0.33)
            mu = taubin_cfg.get('lambda_inflate', -0.53)
            print(f"✓ Applied {n_iters} iterations of Taubin smoothing")
            print(f"  λ (smooth): {lam:+.3f}")
            print(f"  μ (inflate): {mu:+.3f}")
    else:
        if verbose:
            print("⊘ Taubin smoothing disabled")
    
    # Export Stage 4 data
    if export_stages:
        stage_outputs['stage4'] = {
            'points': points.detach().clone() if return_torch else as_numpy(points),
            'normals': normals_up.detach().clone() if return_torch else as_numpy(normals_up),
        }
    
    # ========================================================================
    # STAGE 5: Normal Smoothing (Spatial Laplacian)
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 5/6: Normal Smoothing (Spatial Laplacian)")
        print("="*80)
    
    norm_cfg = cfg.get("normal_smooth", {})
    if norm_cfg.get("enabled", True):
        normals_up = smooth_normals(normals_up, points, knn, norm_cfg)
        
        if verbose:
            n_iters = norm_cfg.get('iters', 2)
            lam = norm_cfg.get('lambda_smooth', 0.8)
            k = norm_cfg.get('k', 16)
            print(f"✓ Applied {n_iters} iterations of normal smoothing")
            print(f"  λ (blend): {lam:.3f}")
            print(f"  k (neighbors): {k}")
    else:
        if verbose:
            print("⊘ Normal smoothing disabled")
    
    # Export Stage 5 data
    if export_stages:
        stage_outputs['stage5'] = {
            'points': points.detach().clone() if return_torch else as_numpy(points),
            'normals': normals_up.detach().clone() if return_torch else as_numpy(normals_up),
        }
    
    # ========================================================================
    # STAGE 6: Covariance Construction (F-field Interpolation)
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 6/6: Covariance Construction (F-field Interpolation)")
        print("="*80)
    
    cov_cfg = cfg.get("covariance", {})
    
    # Pass rho_anchor to covariance for density-based scale adjustment
    if rho_anchor is not None:
        density_cfg_for_cov = cov_cfg.get("density", {})
        density_cfg_for_cov["rho_anchor"] = rho_anchor  # This is fine - nested dict
        cov_cfg["density"] = density_cfg_for_cov
        
        if verbose:
            if density_cfg_for_cov.get("use_scale_prior", False):
                print(f"  ✓ Density-based scale adjustment ENABLED:")
                print(f"    - rho_anchor passed: {rho_anchor.shape}")
                print(f"    - scale_kappa: {density_cfg_for_cov.get('scale_kappa', 0.15):.3f}")
                print(f"    - max_scale_up: {density_cfg_for_cov.get('scale_max_up', 0.12):.3f}")
            else:
                print(f"  ⊘ Density-based scale adjustment disabled (rho available but not used)")
    else:
        if verbose:
            print(f"  ⊘ No rho_anchor available (density-based scale skipped)")
    
    # 🔥 Prepare curvature data for target mesh (F≈Identity)
    # For target mesh, use curvature-based anisotropic covariance instead of isotropic
    curvature_cov_cfg = cov_cfg.get("curvature_cov", {})
    if curvature_cov_cfg.get("enabled", False):
        if verbose:
            print(f"  [Curvature-Based Cov] Computing curvature on upsampled points...")
        
        # Compute curvature on upsampled points directly
        from .analysis.pca import batched_pca_surface_optimized
        k_curv = int(curvature_cov_cfg.get("k_neighbors", 16))
        idx_curv, w_curv = knn(points, points, k_curv)
        
        normals_curv, _, _, _, dir1, dir2, curv_vals = batched_pca_surface_optimized(
            points, idx_curv, w_curv, return_principal_dirs=True
        )
        
        # Package curvature data
        curvature_data = {
            'kappa1': curv_vals[:, 0],           # (M,) max curvature
            'kappa2': curv_vals[:, 1],           # (M,) min curvature
            'principal_dirs': torch.stack([dir1, dir2], dim=-1),  # (M, 3, 2)
            'normals': normals_curv,             # (M, 3)
        }
        
        cov_cfg["curvature_data"] = curvature_data
        
        if verbose:
            print(f"    ✓ Curvature computed: k1=[{curv_vals[:,0].min():.3f}, {curv_vals[:,0].max():.3f}], "
                  f"k2=[{curv_vals[:,1].min():.3f}, {curv_vals[:,1].max():.3f}]")
    
    cov, F_interp, _ = build_covariance(points, x_low, F_low, knn, cov_cfg, learnable_cov_module)
    
    if verbose:
        use_polar = cov_cfg.get("use_polar_decomposition", True)
        sigma0 = cov_cfg.get("sigma0", 0.08)
        k_F = cov_cfg.get("k_F", 32)
        print(f"✓ Built {len(cov):,} covariance matrices")
        print(f"  Method: {'Polar decomposition' if use_polar else 'Direct (FF^T)'}")
        print(f"  σ₀ (base scale): {sigma0:.4f}")
        print(f"  k_F (neighbors): {k_F}")
    
    # Export Stage 6 data
    if export_stages:
        stage_outputs['stage6'] = {
            'points': points.detach().clone() if return_torch else as_numpy(points),
            'normals': normals_up.detach().clone() if return_torch else as_numpy(normals_up),
            'cov': cov.detach().clone() if return_torch else as_numpy(cov),
        }
    
    # ========================================================================
    # Prepare Output
    # ========================================================================
    debug_info = {
        "N_input": len(x_low),
        "M_output": len(points),
        "upsampling_factor": len(points) / len(x_low),
        "surface_detection": surf_cfg.get("enabled", True),
        "anchor_density": density_cfg.get("enabled", True),
        "taubin_smoothing": taubin_cfg.get("enabled", True),
        "normal_smoothing": norm_cfg.get("enabled", True),
        "rho_anchor_mean": state.get("rho_anchor_mean", 0.0),
        "device": str(device),
        "seed": seed,
    }
    
    if verbose:
        print("="*80)
        print("Pipeline Complete!")
        print(f"  Input:  {debug_info['N_input']:,} points")
        print(f"  Output: {debug_info['M_output']:,} points")
        print(f"  Factor: {debug_info['upsampling_factor']:.1f}×")
        print("="*80 + "\n")
        
    # ========================================================================
    # Cleanup
    # ========================================================================
    del normals
    perf_cfg = cfg.get("performance", {})
    if perf_cfg.get("clear_cache", True):
        knn.clear_cache()
        
        if verbose:
            print("\n" + "="*80)
            print("Cleanup: FAISS cache cleared")
    
  
    # Convert to numpy if requested
    result = {
        "points": points if return_torch else as_numpy(points),
        "normals": normals_up if return_torch else as_numpy(normals_up),
        "cov": cov if return_torch else as_numpy(cov),
        "F_interp": F_interp if return_torch else as_numpy(F_interp),
        "anchors": anchors if return_torch else as_numpy(anchors),
        "debug": debug_info,
        "state": state,
    }
    
    # Add stage outputs if requested
    if export_stages:
        result["stage_outputs"] = stage_outputs
    
    return result

__all__ = [
    "upsample",
]