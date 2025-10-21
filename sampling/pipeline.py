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
         │  STAGE 2: Volume Filtering (Soft Selection)         │
         │  ────────────────────────────────────────           │
         │  • Compute orientation consensus (normal alignment) │
         │  • Apply sigmoid gating: w = σ(consensus - θ)       │
         │  • Update: filtered_prob = surf_prob · w            │
         │                                                     │
         │  Output: {filtered_prob, volume_weight}             │
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

Stage 2: Volume Filtering
──────────────────────────
    consensus = (1/k)·Σⱼ |nᵢ · nⱼ|             [Normal alignment]
    w = sigmoid(α·(consensus - θ))             [Soft gate]
    filtered_prob = surf_prob · w              [Gated probability]

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

Volume Filtering:
    k: 16                  # Neighbors for consensus
    alpha: 10.0            # Sigmoid sharpness
    theta: 0.85            # Consensus threshold

Importance Sampling:
    M: 50000               # Upsampling factor (5-10× of N)
    tau: 0.2               # Gumbel-Softmax temperature
    alpha: 0.35            # Jitter magnitude
    thickness: 0.0         # Normal offset (0 = 2D surface)

Taubin Smoothing:
    iters: 3               # Number of λ-μ passes
    lambda_smooth: 0.33    # Smoothing weight (positive)
    mu_inflate: -0.34      # Inflation weight (negative)
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
Volume Filtering    O(N·k)                O(N)            Consensus (fast)
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
from .analysis.knn import HybridFAISSKNN, FAISS_AVAILABLE
from .core.surface_detect import detect_surface
from .core.volume_filter import apply_volume_filter
from .core.sampler import sample_points
from .core.taubin_smooth import taubin_smooth
from .core.normal_smooth import smooth_normals
from .geometry.covariance import build_covariance
from .io.export import save_comparison_png


def upsample(
    x_low: torch.Tensor,
    F_low: torch.Tensor,
    cfg: Optional[Dict] = None,
    state: Optional[Dict] = None,
    seed: int = 1234,
    return_torch: bool = True
) -> Dict:
    """
    Main differentiable point cloud upsampling pipeline.
    
    Transforms sparse point cloud with deformation gradients into dense
    point cloud with anisotropic covariances for high-quality rendering.
    
    Pipeline stages:
        1. Surface Detection: PCA-based planarity analysis
        2. Volume Filtering: Soft geometric consistency check
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
            'volume_filter': {
                'enabled': True,
                'k': 16,
                'alpha': 10.0,          # Sigmoid sharpness
                'theta': 0.85,          # Consensus threshold
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
                'mu_inflate': -0.34,
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
    
    # ========================================================================
    # STAGE 1: Surface Detection (PCA-based Planarity)
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 1/6: Surface Detection (PCA-based Planarity)")
        print("="*80)
    
    surf_cfg = cfg.get("surface_detection", {})
    if surf_cfg.get("enabled", True):
        surf_prob, normals, spacing, state = detect_surface(
            x_low, knn, surf_cfg, state
        )
    else:
        # Uniform probability fallback
        N = x_low.shape[0]
        surf_prob = torch.full((N,), 1.0 / N, device=device)
        
        # Still need normals for downstream stages
        from .analysis.pca import batched_pca_surface_optimized
        k = surf_cfg.get("k", 48)
        idx, w = knn(x_low, x_low, k)
        normals, _, spacing = batched_pca_surface_optimized(x_low, idx, w)
    
    if verbose:
        print(f"✓ Computed surface probabilities for {len(x_low)} points")
        print(f"  Mean prob: {surf_prob.mean():.6f}")
        print(f"  Max prob:  {surf_prob.max():.6f}")
        print(f"  Min prob:  {surf_prob.min():.6f}")
    
    # ========================================================================
    # STAGE 2: Volume Filtering (Soft Geometric Consistency)
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 2/6: Volume Filtering (Soft Geometric Consistency)")
        print("="*80)
    
    vol_cfg = cfg.get("volume_filter", {})
    if vol_cfg.get("enabled", True):
        filtered_prob, volume_weight = apply_volume_filter(
            surf_prob, normals, x_low, knn, vol_cfg
        )
        
        if verbose:
            mask = volume_weight > 0.5
            n_surface = mask.sum().item()
            n_total = len(x_low)
            print(f"✓ Identified {n_surface}/{n_total} surface points "
                  f"({100*n_surface/n_total:.1f}%)")
            print(f"  Volume weight: min={volume_weight.min():.3f}, "
                  f"max={volume_weight.max():.3f}, "
                  f"mean={volume_weight.mean():.3f}")
    else:
        filtered_prob = surf_prob
        volume_weight = torch.ones_like(surf_prob)
        
        if verbose:
            print("⊘ Volume filtering disabled (using raw surface probabilities)")
    
    # ========================================================================
    # STAGE 3: Importance Sampling (Gumbel-Softmax + Tangent Jitter)
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 3/6: Importance Sampling (Gumbel-Softmax + Tangent Jitter)")
        print("="*80)

    samp_cfg = cfg.get("sampling", {})
    M  = int(samp_cfg.get("M", 50000))
    tau = float(samp_cfg.get("tau", 0.2))
    alpha = float(samp_cfg.get("alpha", 0.35))
    thickness = float(samp_cfg.get("thickness", 0.0))

    # Memory-safe sampler knobs
    gs_batch = int(samp_cfg.get("gs_batch", 2048))
    ensure_cover = bool(samp_cfg.get("ensure_anchor_coverage", True))
    micro_jitter_scale = float(samp_cfg.get("micro_jitter_scale", 0.2))

    # Inside-suppression / count-preserving knobs (sampler will ignore if unsupported)
    tangent_micro_only    = bool(samp_cfg.get("tangent_micro_only", True))
    plane_snap            = bool(samp_cfg.get("plane_snap", True))
    min_anchor_prob       = float(samp_cfg.get("min_anchor_prob", 1e-4))
    thickness_one_sided   = bool(samp_cfg.get("thickness_one_sided", True))
    keep_count            = bool(samp_cfg.get("keep_count", True))
    topup_max_tries       = int(samp_cfg.get("topup_max_tries", 3))

    N = len(x_low)  # number of anchors before upsampling

    # Rough peak per batch for logits/softmax (float32), excluding other tensors
    est_mb_per_batch = (gs_batch * max(N, 1) * 4) / (1024**2)

    if verbose:
        print(f"- Anchors (N): {N:,}")
        print(f"- Target samples (M): {M:,}  (upsampling {M/max(N,1):.1f}×)")
        print(f"- Gumbel tau: {tau:.3f} | Jitter alpha: {alpha:.3f} | Thickness: {thickness:.3f}")
        print(f"- Streaming gs_batch: {gs_batch}  (≈ {est_mb_per_batch:.1f} MB logits/batch)")
        print(f"- Ensure anchor coverage: {ensure_cover} | min_anchor_prob: {min_anchor_prob:.1e}")
        print(f"- Micro jitter scale: {micro_jitter_scale:.3f} | tangent_micro_only: {tangent_micro_only}")
        print(f"- Plane snap: {plane_snap} | thickness_one_sided: {thickness_one_sided}")
        print(f"- keep_count: {keep_count} (topup_max_tries={topup_max_tries})")

        # Heads-up messages mirroring sampler behavior
        if ensure_cover and M < N:
            print("  [note] ensure_anchor_coverage=True but M < N; "
                "sampler will fall back to top-M anchors by probability within valid anchors.")
        if ensure_cover and M >= N:
            print("  [note] coverage: each valid anchor (prob ≥ min_anchor_prob) appears at least once; "
                "remaining slots are filled stochastically.")
        if micro_jitter_scale > 0.25 and not tangent_micro_only:
            print("  [warn] High micro_jitter_scale with isotropic micro may increase interior leakage; "
                "consider tangent_micro_only=True or lower micro_jitter_scale.")
        if thickness > 0.0 and not thickness_one_sided:
            print("  [warn] Symmetric thickness can create interior points; "
                "set thickness_one_sided=True to bias outward.")

    # Call sampler (memory-safe ST version keeps the same signature)
    points, normals_up, anchors = sample_points(
        x_low, normals, spacing, filtered_prob, samp_cfg, generator
    )

    # Proactive cleanup (helps control transient peaks between stages)
    del filtered_prob, spacing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Invariants & quick sanity prints
    if verbose:
        up_factor = len(points)/max(N,1)
        print(f"✓ Sampled {len(points):,} points from {N:,} anchors ({up_factor:.1f}×)")
        print(f"  Jitter alpha: {alpha:.3f} | Gumbel tau: {tau:.3f}")
        print(f"  gs_batch:     {gs_batch} (≈ {est_mb_per_batch:.1f} MB logits/batch)")
        print(f"  coverage:     {'on' if ensure_cover else 'off'}; "
            f"micro_scale={micro_jitter_scale:.3f}; plane_snap={plane_snap}")

        # Count-preservation invariant
        if len(points) != M:
            print(f"  [warn] sampler returned {len(points)} points but M={M}; "
                "enable keep_count=True to top-up or check sampler configuration.")


    
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
            mu = taubin_cfg.get('mu_inflate', -0.34)
            print(f"✓ Applied {n_iters} iterations of Taubin smoothing")
            print(f"  λ (smooth): {lam:+.3f}")
            print(f"  μ (inflate): {mu:+.3f}")
    else:
        if verbose:
            print("⊘ Taubin smoothing disabled")
    
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
    
    # ========================================================================
    # STAGE 6: Covariance Construction (F-field Interpolation)
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 6/6: Covariance Construction (F-field Interpolation)")
        print("="*80)
    
    cov_cfg = cfg.get("covariance", {})
    cov, F_interp, _ = build_covariance(points, x_low, F_low, knn, cov_cfg)
    
    if verbose:
        use_polar = cov_cfg.get("use_polar_decomposition", True)
        sigma0 = cov_cfg.get("sigma0", 0.08)
        k_F = cov_cfg.get("k_F", 32)
        print(f"✓ Built {len(cov):,} covariance matrices")
        print(f"  Method: {'Polar decomposition' if use_polar else 'Direct (FF^T)'}")
        print(f"  σ₀ (base scale): {sigma0:.4f}")
        print(f"  k_F (neighbors): {k_F}")
    
    # ========================================================================
    # Prepare Output
    # ========================================================================
    debug_info = {
        "N_input": len(x_low),
        "M_output": len(points),
        "upsampling_factor": len(points) / len(x_low),
        "surface_detection": surf_cfg.get("enabled", True),
        "volume_filtering": vol_cfg.get("enabled", True),
        "taubin_smoothing": taubin_cfg.get("enabled", True),
        "normal_smoothing": norm_cfg.get("enabled", True),
        "mean_surf_prob": float(surf_prob.mean().detach().item()),
        "mean_volume_weight": float(volume_weight.mean().detach().item()),
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
    del normals, surf_prob, volume_weight
    perf_cfg = cfg.get("performance", {})
    if perf_cfg.get("clear_cache", True):
        knn.clear_cache()
        
        if verbose:
            print("\n" + "="*80)
            print("Cleanup: FAISS cache cleared")
    
  
    # Convert to numpy if requested
    if return_torch:
        return {
            "points": points,
            "normals": normals_up,
            "cov": cov,
            "F_interp": F_interp,
            "anchors": anchors,
            "debug": debug_info,
            "state": state,
        }
    else:
        return {
            "points": as_numpy(points),
            "normals": as_numpy(normals_up),
            "cov": as_numpy(cov),
            "F_interp": as_numpy(F_interp),
            "anchors": as_numpy(anchors),
            "debug": debug_info,
            "state": state,
        }

__all__ = [
    "upsample",
]