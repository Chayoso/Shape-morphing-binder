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
import numpy as np
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
from .geometry.deformation_covariance import build_deformation_covariance
from .geometry.curvature_covariance import create_curvature_based_covariance_star
from .geometry.learnable_covariance import LearnableCovariance


def upsample(
    x_low: torch.Tensor,
    F_low: torch.Tensor,
    cfg: Optional[Dict] = None,
    state: Optional[Dict] = None,
    seed: int = 1234,
    return_torch: bool = True,
    export_stages: bool = False,
    learnable_cov_module=None  # Optional learnable covariance module
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
    
    # Always compute surface detection for robust sampling
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
            # Returns: p_surf_raw, normals, spacing, state, planarity, anisotropy, z_s, rho_gate, principal_dirs...
            p_surf_raw, normals, spacing, state, planarity_all, anisotropy_all, z_s, rho_gate, principal_dir1, principal_dir2, principal_curv = result
        else:
            # Returns: p_surf_raw, normals, spacing, state, planarity, anisotropy, z_s, rho_gate
            p_surf_raw, normals, spacing, state, planarity_all, anisotropy_all, z_s, rho_gate = result
            principal_dir1 = principal_dir2 = principal_curv = None
    else:
        # Uniform probability fallback
        N = x_low.shape[0]
        p_surf_raw = torch.ones(N, device=device)
        z_s = torch.zeros(N, device=device)
        rho_gate = torch.zeros(N, device=device)
        
        # Still need normals for downstream stages
        from .analysis.pca import batched_pca_surface_optimized
        k = surf_cfg.get("k", 48)
        idx, w = knn(x_low, x_low, k)
        normals, _, spacing, _, planarity_all, anisotropy_all = batched_pca_surface_optimized(x_low, idx, w)
        principal_dir1 = principal_dir2 = principal_curv = None
    
    # ========================================================================
    # STAGE1: Summary
    # ========================================================================
    N = x_low.shape[0]
    
    if verbose:
        print(f"✓ Surface detection: {N:,} anchors")
        print(f"  p_surf_raw: mean={p_surf_raw.mean():.4f}, >0.5={(p_surf_raw > 0.5).float().mean():.1%}")
        
        if use_anisotropic_jitter and principal_curv is not None:
            print(f"  [Anisotropic Jitter] ✓ ENABLED")
            print(f"    - Principal curvatures: k1=[{principal_curv[:,0].min():.3f}, {principal_curv[:,0].max():.3f}], "
                  f"k2=[{principal_curv[:,1].min():.3f}, {principal_curv[:,1].max():.3f}]")
        else:
            print(f"  [Anisotropic Jitter] ⊘ DISABLED (isotropic jitter)")
    
    # ========================================================================
    # Export Stage 1 data (PCA metrics + z-score + boost mask)
    # ========================================================================
    if export_stages:
        stage_out = {
            'points': x_low.detach().clone() if return_torch else as_numpy(x_low),
            'normals': normals.detach().clone() if return_torch else as_numpy(normals),
            'spacing': spacing.detach().clone() if return_torch else as_numpy(spacing),
            'planarity': planarity_all.detach().clone() if return_torch else as_numpy(planarity_all),
            'anisotropy': anisotropy_all.detach().clone() if return_torch else as_numpy(anisotropy_all),
            'z_s': z_s.detach().clone() if return_torch else as_numpy(z_s),
            'rho_gate': rho_gate.detach().clone() if return_torch else as_numpy(rho_gate),
            'p_surf_raw': p_surf_raw.detach().clone() if return_torch else as_numpy(p_surf_raw),
        }
        
        if use_anisotropic_jitter and principal_curv is not None:
            stage_out['principal_dir1'] = principal_dir1.detach().clone() if return_torch else as_numpy(principal_dir1)
            stage_out['principal_dir2'] = principal_dir2.detach().clone() if return_torch else as_numpy(principal_dir2)
            stage_out['principal_curv'] = principal_curv.detach().clone() if return_torch else as_numpy(principal_curv)
        
        stage_outputs['stage1'] = stage_out
    
    # ========================================================================
    # STAGE 2: Anchor Density (Multi-scale harmonic mean)
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 2/6: Multi-Scale Anchor Density (Harmonic Mean)")
        print("="*80)
    
    # Multi-scale density: harmonic mean of k=8 and k=32
    from .core.density_map import build_multiscale_density
    w_density = build_multiscale_density(
        x_low, knn,
        k_small=8,
        k_large=32,
        surf_prob=None,
        state=state
    )
    
    # π calculation: Simple multiplicative (density-based)
    alpha = 2.2  # Surface concentration
    beta = 0.7   # Density bias
    eps = 1e-8
    
    pi = (p_surf_raw.clamp(min=eps)**alpha) * (w_density**beta)
    pi = pi / (pi.sum() + eps)  # ∑π=1
    
    # Correlation diagnostics
    def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
        a_n = (a - a.mean()) / (a.std() + eps)
        b_n = (b - b.mean()) / (b.std() + eps)
        return float((a_n * b_n).mean().item())
    
    corr_spacing_density = _corr(spacing, w_density)
    corr_spacing_pi = _corr(spacing, pi)
    
    if state is not None:
        state['stage2_corr_spacing_density'] = corr_spacing_density
        state['stage2_corr_spacing_pi'] = corr_spacing_pi
    
    if verbose:
        print(f"✓ Computed multi-scale density")
        print(f"  w_density: mean={w_density.mean():.4f}, range=[{w_density.min():.3f}, {w_density.max():.3f}]")
        print(f"  corr(spacing, density): {corr_spacing_density:+.3f}")
        print(f"  ")
        print(f"✓ Computed sampling distribution π")
        print(f"  Formula: normalize(p_surf_raw^{alpha:.1f} × w_density^{beta:.1f})")
        print(f"  π: mean={pi.mean():.6f}, range=[{pi.min():.4e}, {pi.max():.4e}]")
        print(f"  corr(spacing, π): {corr_spacing_pi:+.3f}")
    
    # Use pi for sampling
    surf_prob = pi.clone()

    # rho_anchor for covariance prior
    rho_anchor = w_density
    
    # Export Stage 2 data
    if export_stages:
        stage_outputs['stage2'] = {
            'points': x_low.detach().clone() if return_torch else as_numpy(x_low),
            'w_den': w_density.detach().clone() if return_torch else as_numpy(w_density),
            'pi': pi.detach().clone() if return_torch else as_numpy(pi),
            'spacing': spacing.detach().clone() if return_torch else as_numpy(spacing),
        }
    
    # ========================================================================
    # STAGE 3: Soft Continuous Upsampling (Gumbel-Softmax)
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 3/6: Soft Continuous Upsampling (Gumbel-Softmax)")
        print("="*80)

    # Get sampling config
    samp_cfg_outer = cfg.get("sampling", {})
    samp_cfg = samp_cfg_outer.get("sampling", samp_cfg_outer)
    
    N = len(x_low)
    
    # Setup sampler config
    samp_cfg["p_surf_raw"] = p_surf_raw
    samp_cfg["w_density"] = w_density
    samp_cfg["M"] = int(samp_cfg.get("M", 50000))
    samp_cfg["alpha"] = float(samp_cfg.get("alpha", alpha))
    samp_cfg["beta"] = float(samp_cfg.get("beta", beta))
    samp_cfg["p_target"] = float(samp_cfg.get("p_target", 0.85))
    samp_cfg["uniform_mix"] = float(samp_cfg.get("uniform_mix", 0.15))
    samp_cfg["gate_k"] = float(samp_cfg.get("gate_k", 120.0))
    samp_cfg["keep_ratio"] = float(samp_cfg.get("keep_ratio", 0.70))
    samp_cfg["beta_soft"] = float(samp_cfg.get("beta_soft", 1.15))
    samp_cfg["eps_mix"] = float(samp_cfg.get("eps_mix", 0.002))
    samp_cfg["center_tau"] = float(samp_cfg.get("center_tau", 4.5))
    samp_cfg["tau_local"] = float(samp_cfg.get("tau_local", 1.15))
    samp_cfg["u_local"] = float(samp_cfg.get("u_local", 0.68))
    samp_cfg["w_cap"] = float(samp_cfg.get("w_cap", 0.13))
    samp_cfg["bias_target"] = float(samp_cfg.get("bias_target", 2.5))
    samp_cfg["local_S"] = int(samp_cfg.get("local_S", 20))
    samp_cfg["voxel_scale"] = float(samp_cfg.get("voxel_scale", 0.58))
    samp_cfg["gamma_flat"] = float(samp_cfg.get("gamma_flat", 0.70))
    samp_cfg["cap_mult"] = float(samp_cfg.get("cap_mult", 18.0))
    samp_cfg["use_dynamic_cap"] = bool(samp_cfg.get("use_dynamic_cap", True))
    
    samp_cfg["local_batch"] = int(samp_cfg.get("local_batch", 98304))
    samp_cfg["micro_jitter_j0"] = float(samp_cfg.get("micro_jitter_j0", 0.22))
    samp_cfg["use_amp"] = bool(samp_cfg.get("use_amp", True))
    samp_cfg["use_soft_nms"] = bool(samp_cfg.get("use_soft_nms", False))
    samp_cfg["suppress_lambda"] = float(samp_cfg.get("suppress_lambda", 0.15))
    samp_cfg["use_st"] = bool(samp_cfg.get("use_st", False))
    samp_cfg["use_voxel_diversity"] = bool(samp_cfg.get("use_voxel_diversity", True))
    
    samp_cfg["debug"] = {
        "pi_breakdown": bool(samp_cfg.get("debug", {}).get("pi_breakdown", True)),
        "verbose": bool(samp_cfg.get("debug", {}).get("verbose", False))
    }
    
    M = samp_cfg["M"]
    S = samp_cfg["local_S"]
    tau_local = samp_cfg["tau_local"]
    local_batch = samp_cfg["local_batch"]
    
    if verbose:
        print(f"- Total anchors (N): {N:,}")
        print(f"- Target samples (M): {M:,} (upsampling {M/max(N,1):.1f}×)")
        print(f"- Distribution: α={alpha:.1f}, β={beta:.1f}, uniform_mix={samp_cfg['uniform_mix']:.2f}")
        print(f"- Soft-cut: keep_ratio={samp_cfg['keep_ratio']:.2f}, beta_soft={samp_cfg['beta_soft']:.2f}")
        print(f"- Flattening: center_tau={samp_cfg['center_tau']:.1f}, tau_local={samp_cfg['tau_local']:.2f}")

    # Call memory-safe soft continuous sampler
    from .core.sampler import sample_points
    
    # probs=None → sampler internally creates pi_cache
    points, normals_up, anchors, anchor_selection_count = sample_points(
        x_low, normals, spacing, None, samp_cfg, generator
    )
    
    # Get results from cfg
    p_up = samp_cfg.get("p_up")
    diag = samp_cfg.get("diagnostics", {})

    # Result summary with diagnostics
    if verbose:
        print(f"\n✓ Local-Soft ST upsampling: {len(points):,} points")
        print(f"  Method: {'Soft' if not samp_cfg['use_st'] else 'ST'} + Voxel-Hash + Micro-Jitter")
        print(f"  Memory: O(B·S) = O({local_batch:,} × {S}) per batch")
        
        # p_up stats
        if p_up is not None:
            print(f"  p_up: mean={p_up.mean():.4f}, range=[{p_up.min():.3f}, {p_up.max():.3f}]")
        
        # Show correlations only
        if state is not None:
            print(f"\n  [Density Correlations]")
            if 'stage2_corr_spacing_density' in state:
                corr_den = state['stage2_corr_spacing_density']
                print(f"    corr(spacing, density): {corr_den:+.3f}")
            if 'stage2_corr_spacing_pi' in state:
                corr_pi = state['stage2_corr_spacing_pi']
                print(f"    corr(spacing, π):       {corr_pi:+.3f}")
    
    # Export Stage 3 data
    if export_stages:
        stage_outputs['stage3'] = {
            'points': points.detach().clone() if return_torch else as_numpy(points),
            'anchors': anchors.detach().clone() if return_torch else as_numpy(anchors),
            'normals': normals_up.detach().clone() if return_torch else as_numpy(normals_up),
            'p_up': p_up.detach().clone() if return_torch else as_numpy(p_up),
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
    
    # CRITICAL: Different covariance construction for target vs predicted
    # Episode convention:
    #   -1: Initial target render (before training starts)
    #    0: First episode, still uses target mesh for initialization
    #   >0: Training episodes with predicted/deformed meshes
    #
    # Covariance strategy:
    #   episode ≤ 0 → Target: curvature-based (geometry/curvature_covariance.py)
    #   episode > 0 → Predicted: deformation-based (geometry/deformation_covariance.py)
    
    episode = cov_cfg.get("episode", -1)
    is_target_mesh = (episode <= 0)  # True for episode -1 and 0
    
    if is_target_mesh:
        # Target mesh: Use curvature-based covariance
        if verbose:
            print(f"  [Episode {episode}] Target mesh - using curvature-based covariance")
        
        # Reuse STAGE 1 curvature data (no need to recompute)
        if export_stages and 'stage1' in stage_outputs:
            stage1_data = stage_outputs['stage1']
            planarity_stage1 = stage1_data['planarity']  # (N_anchors,)
            anisotropy_stage1 = stage1_data['anisotropy']  # (N_anchors,)
            
            # Map to upsampled points via nearest anchor
            # Use stage1 points (anchors) for mapping
            stage1_points = stage1_data['points']
            
            # Ensure tensors for KNN
            if isinstance(stage1_points, np.ndarray):
                stage1_points = torch.from_numpy(stage1_points).to(device)
            
            idx_map, _ = knn(points, stage1_points, 1)
            
            # Map planarity and anisotropy to upsampled points
            planarity_pts = planarity_stage1[idx_map.squeeze()]
            anisotropy_pts = anisotropy_stage1[idx_map.squeeze()]
            
            # Convert to numpy for curvature_covariance function
            def to_numpy(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy()
                return x
            
            planarity_pts_np = to_numpy(planarity_pts)
            anisotropy_pts_np = to_numpy(anisotropy_pts)
            points_np = to_numpy(points)
            normals_np = to_numpy(normals_up)
            
            if verbose:
                print(f"    ✓ Reused STAGE 1 curvature data (planarity, anisotropy)")
            
            # Get sigma params from config
            # Note: run.py copies curvature_sigma from optimization.loss to upsample.covariance
            sigma_params = cov_cfg.get('curvature_sigma', {
                'sigma_n0': 0.020,
                'sigma_t0': 0.030,
                'a': 3.0,
                'b': 0.5,
                'u': 0.4
            })
            
            if verbose:
                print(f"    ✓ Using curvature_sigma params:")
                print(f"      sigma_n0={sigma_params.get('sigma_n0', 0.020):.4f}, sigma_t0={sigma_params.get('sigma_t0', 0.030):.4f}")
                print(f"      a={sigma_params.get('a', 3.0):.2f}, b={sigma_params.get('b', 0.5):.2f}, u={sigma_params.get('u', 0.4):.2f}")
            
            # Create curvature-based target covariance Σ★ (returns numpy)
            cov_np = create_curvature_based_covariance_star(
                points_np,
                normals_np,
                planarity_pts_np,
                anisotropy_pts_np,
                sigma_params
            )
            cov_target = torch.from_numpy(cov_np).to(device)
            F_interp = torch.eye(3, device=device).unsqueeze(0).expand(len(points), 3, 3)  # Identity for target
            cov = cov_target 
            
            if verbose:
                print(f"    ✓ Curvature-based Σ★: trace range=[{torch.diagonal(cov_target, dim1=-2, dim2=-1).sum(-1).min():.6f}, "
                      f"{torch.diagonal(cov_target, dim1=-2, dim2=-1).sum(-1).max():.6f}]")
        else:
            print(f"[WARN] Stage 1 data not available, falling back to deformation-based")
            cov_target, F_interp, _ = build_deformation_covariance(points, x_low, F_low, knn, cov_cfg, learnable_cov_module)
            cov = cov_target
    else:
        # Predicted mesh: Use deformation-based covariance from F-field
        if verbose:
            print(f"  [Episode {episode}] Predicted mesh - using deformation-based covariance")
        
        # Build physics-based covariance (no learnable yet)
        cov_phys, F_interp, _ = build_deformation_covariance(
            points, x_low, F_low, knn, cov_cfg, learnable_cov_module=None
        )
        
        # Optional: Apply learnable covariance refinement
        learnable_cfg = cov_cfg.get("learnable", {})
        if learnable_cfg.get("enabled", False):
            if verbose:
                print(f"  [Learnable] Applying learnable covariance refinement...")
            
            # Initialize or reuse learnable module
            M = cov_phys.shape[0]
            device = cov_phys.device
            dtype = cov_phys.dtype
            
            # Check if module exists and has correct size
            if learnable_cov_module is None or learnable_cov_module.M != M:
                if verbose:
                    print(f"    Creating new LearnableCovariance module (M={M})")
                sigma0 = cov_cfg.get("sigma0", 0.08)
                learnable_cov_module = LearnableCovariance(
                    M=M, device=device, dtype=dtype, init_scale=sigma0
                )
            
            # Apply hybrid mixing
            alpha = float(learnable_cfg.get("alpha", 0.3))
            cov_pred = learnable_cov_module(cov_physics=cov_phys, alpha=alpha)
            
            if verbose:
                print(f"    ✓ Hybrid covariance: α={alpha:.2f} (physics {alpha*100:.0f}% + learnable {(1-alpha)*100:.0f}%)")
        else:
            # Pure physics
            cov_pred = cov_phys
        
        cov = cov_pred  # Backward compatibility
    
    if verbose:
        sigma0 = cov_cfg.get("sigma0", 0.08)
        print(f"✓ Built {len(cov):,} covariance matrices")
        
        if is_target_mesh:
            # Target: Curvature-based
            print(f"  Method: Curvature-based (planarity/anisotropy from STAGE 1)")
            sigma_params = cov_cfg.get('curvature_sigma', {})
            print(f"  σ_n0: {sigma_params.get('sigma_n0', 0.020):.3f}, σ_t0: {sigma_params.get('sigma_t0', 0.030):.3f}")
            print(f"  Sensitivity: a={sigma_params.get('a', 3.0):.1f}, b={sigma_params.get('b', 0.5):.1f}, u={sigma_params.get('u', 0.4):.1f}")
        else:
            # Predicted: Deformation-based
            use_polar = cov_cfg.get("use_polar_decomposition", True)
            k_F = cov_cfg.get("k_F", 32)
            print(f"  Method: {'Polar decomposition (F-field)' if use_polar else 'Direct (FF^T)'}")
            print(f"  σ₀ (base scale): {sigma0:.4f}")
            print(f"  k_F (neighbors): {k_F}")
    
    # Export Stage 6 data
    if export_stages:
        stage_outputs['stage6'] = {
            'points': points.detach().clone() if return_torch else as_numpy(points),
            'normals': normals_up.detach().clone() if return_torch else as_numpy(normals_up),
            'cov': cov.detach().clone() if return_torch else as_numpy(cov),
        }
    
    # Store with clear naming
    cov_key = 'cov_target' if is_target_mesh else 'cov_pred'
    
    # ========================================================================
    # Prepare Output
    # ========================================================================
    debug_info = {
        "N_input": len(x_low),
        "M_output": len(points),
        "upsampling_factor": len(points) / len(x_low),
        "surface_detection": surf_cfg.get("enabled", True),
        "anchor_density": True,
        "taubin_smoothing": taubin_cfg.get("enabled", True),
        "normal_smoothing": norm_cfg.get("enabled", True),
        "rho_anchor_mean": float(rho_anchor.mean().item()) if isinstance(rho_anchor, torch.Tensor) else 0.0,
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
    
  
    # Convert to numpy if requested and add clear naming
    result = {
        "points": points if return_torch else as_numpy(points),
        "normals": normals_up if return_torch else as_numpy(normals_up),
        "F_interp": F_interp if return_torch else as_numpy(F_interp),
        "anchors": anchors if return_torch else as_numpy(anchors),
        "debug": debug_info,
        "state": state,
    }
    
    # Add covariance with clear naming (avoid key overwrite confusion)
    cov_final = cov if return_torch else as_numpy(cov)
    
    if is_target_mesh:
        # Target: Primary key is 'cov_target' (Σ★)
        result["cov_target"] = cov_final
        result["cov"] = cov_final  # Backward compatibility (same reference)
    else:
        # Predicted: Primary key is 'cov_pred' (Σ)
        result["cov_pred"] = cov_final  
        result["cov"] = cov_final  # Backward compatibility (same reference)
    
    # Add stage outputs if requested
    if export_stages:
        result["stage_outputs"] = stage_outputs
        
        # Also add stage_data for backward compatibility
        # Use FINAL filtered values (after snap + blue-noise)
        if 'stage1' in stage_outputs:
            if 'stage_data' not in result:
                result['stage_data'] = {}
            
            # Get final filtered data from stage_out
            stage1_out = stage_outputs['stage1']
            result['stage_data']['stage1'] = {
                'points': stage1_out['points'],
                'planarity': stage1_out['planarity'],
                'anisotropy': stage1_out['anisotropy'],
                'normals': stage1_out['normals'],
                'p_surf_raw': stage1_out['p_surf_raw']
            }
            
            # Also add STAGE2 data if available (for curvature_covariance.py compatibility)
            if 'stage2' in stage_outputs:
                stage2_out = stage_outputs['stage2']
                result['stage_data']['stage2'] = {
                    'w_density': stage2_out['w_den'],
                    'pi': stage2_out['pi']
                }
    
    return result

__all__ = [
    "upsample",
]