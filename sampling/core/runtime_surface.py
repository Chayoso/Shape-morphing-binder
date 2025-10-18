"""
runtime_surface.py - Runtime Surface Upsampling with Legacy Compatibility
===========================================================================

This module provides a complete differentiable pipeline for upsampling coarse
point clouds into dense surface representations suitable for 3D Gaussian Splatting.

UPSAMPLING PROCESS OVERVIEW
============================

The upsampling process transforms a coarse point cloud (N points) into a dense
surface representation (M >> N points, typically M = 50,000) through the following
pipeline:

┌─────────────────────────────────────────────────────────────────────────────┐
│ INPUT: Coarse Point Cloud                                                   │
│  - x_low: (N, 3) positions                                                  │
│  - F_low: (N, 3, 3) deformation gradients                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Surface Detection (PCA-based)                                       │
│  - Compute k-NN for each point (k=36)                                       │
│  - Perform weighted PCA to extract:                                         │
│    * Surface normals (minimum eigenvector)                                  │
│    * Surface variance (planarity metric)                                    │
│    * Local spacing (adaptive jitter scale)                                  │
│  - Classify points as "surface-like" using soft thresholding                │
│  - Output: surf_prob (N,) - probability distribution for sampling           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Importance Sampling (Gumbel-Softmax)                                │
│  - Compute importance weights:                                              │
│    w_import = (surf_prob^1.5) / (spacing^density_gamma)                     │
│  - Sample M anchor points using Gumbel-Softmax (differentiable):            │
│    * Generate Gumbel noise: g = -log(-log(U))                               │
│    * Soft one-hot: Y = softmax((log(w_import) + g) / tau)                   │
│    * Straight-through estimator: Y_hard = onehot(argmax(Y)) - Y.detach() + Y│
│  - Interpolate surface properties:                                          │
│    * mu_anchors = Y @ x_low                                                 │
│    * normals = normalize(Y @ normals_low)                                   │
│    * spacing = Y @ spacing_low                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Tangent Space Jittering                                             │
│  - Build orthonormal tangent frame [t1, t2, n] at each point                │
│  - Generate adaptive random offsets:                                        │
│    * Tangent: α_adapt * h * (U_rot * t1 + V_rot * t2)                       │
│    * Normal: thickness * Z * n                                              │
│    * Micro: 0.2 * α * h * ε (high-frequency noise)                          │
│  - Apply jitter: mu = mu_anchors + tangent + normal + micro                 │
│  Output: Dense point cloud (M, 3)                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Density Equalization (Optional)                                     │
│  - Iteratively adjust point positions to equalize local density:            │
│    * Compute local density: ρ = Σ_k w_k * exp(-||p_i - p_k||^2 / h^2)       │
│    * Displacement: Δp = -α * tanh((ρ - ρ_mean) / ρ_mean) * Σ w_k (p_k - p)  │
│  - Re-project to MLS surface every N iterations                             │
│  - Prevent clustering and ensure uniform coverage                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Surface Smoothing (Optional)                                        │
│  - Apply tangent Laplacian smoothing:                                       │
│    * Compute Laplacian: L(p) = Σ w_k (p_k - p)                              │
│    * Split into tangent/normal: L = L_t + L_n                               │
│    * Update: p' = p + α * (L_t + λ * L_n)                                   │
│  - Reduces "voxel bumps" while preserving shape                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Deformation Gradient Smoothing                                      │
│  - Smooth F fields using graph Laplacian:                                   │
│    * Select K nodes via random sampling                                     │
│    * Build sparse graph Laplacian L_graph                                   │
│    * Build interpolation weights W: points → nodes                          │
│    * Solve: (W^T W + λL) Y = W^T F                                          │
│    * Interpolate back: F_smooth = W @ Y                                     │
│  - Prevents high-frequency oscillations in F                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: Covariance Construction                                             │
│  - Interpolate smoothed F to upsampled points:                              │
│    * Find k-NN of each upsampled point in coarse cloud                      │
│    * F_local = Σ w_k * F_smooth[k]                                          │
│  - Construct anisotropic covariance:                                        │
│    * Σ = σ₀² * F_local @ F_local^T                                          │
│  - Result: (M, 3, 3) covariance matrices for 3D Gaussians                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ OUTPUT: Dense Surface Representation                                        │
│  - points: (M, 3) upsampled positions                                       │
│  - normals: (M, 3) surface normals                                          │
│  - cov: (M, 3, 3) covariance matrices                                       │
│  - F_smooth: (M, 3, 3) interpolated deformation gradients                   │
└─────────────────────────────────────────────────────────────────────────────┘

KEY FEATURES
============

1. **Hybrid FAISS KNN**: Fast approximate k-NN search with differentiable reweighting
   - Forward: FAISS gives indices (no gradients, fast)
   - Backward: Recompute distances and softmax weights (differentiable)

2. **Differentiable Sampling**: Gumbel-Softmax for continuous relaxation
   - Allows gradients to flow through discrete sampling decisions
   - Straight-through estimator for crisp one-hot vectors

3. **Adaptive Jittering**: Spacing-aware noise injection
   - Larger jitter in sparse regions, smaller in dense regions
   - Respects local surface geometry (tangent plane)

4. **MLS Surface Projection**: Oriented-point implicit surface
   - Iteratively moves points to minimize signed distance
   - Preserves sharp features while smoothing noise

5. **Density-Aware Equalization**: Repulsion-based redistribution
   - Prevents clustering and voids
   - Maintains attachment to underlying surface

USAGE EXAMPLES
==============

Basic usage:
    >>> from sampling.runtime_surface import default_cfg, synthesize_runtime_surface
    >>> cfg = default_cfg()
    >>> result = synthesize_runtime_surface(x_low, F_low, cfg)
    >>> points, cov = result["points"], result["cov"]

Custom configuration:
    >>> cfg = default_cfg()
    >>> cfg["M"] = 100_000  # More points
    >>> cfg["surf_jitter_alpha"] = 0.8  # More jitter
    >>> cfg["post_equalize"]["enabled"] = True
    >>> result = synthesize_runtime_surface(x_low, F_low, cfg)

Fast path (reuse positions, recompute covariances only):
    >>> cfg["fast_cov_only"] = True
    >>> result = synthesize_runtime_surface(x_low, F_low_updated, cfg, ema_state)

MATHEMATICAL DETAILS
====================

Surface Detection (PCA):
    Given neighbors Q_1, ..., Q_k with weights w_1, ..., w_k:
    
    Centroid: c = Σ w_i Q_i / Σ w_i
    Centered: Q'_i = Q_i - c
    Covariance: C = Σ w_i Q'_i Q'_i^T / Σ w_i
    
    Eigen-decomposition: C = V Λ V^T
    Normal: n = V[:, 0] (eigenvector of smallest eigenvalue)
    Surface variance: λ_0 / (λ_0 + λ_1 + λ_2)

Gumbel-Softmax Sampling:
    Standard Gumbel: G ~ -log(-log(U)), U ~ Uniform(0,1)
    Logits: z_i = log(π_i) + G_i
    Soft sample: y_i = exp(z_i / τ) / Σ_j exp(z_j / τ)
    Hard sample: y_i = 1[i = argmax_j z_j]
    Straight-through: y_hard - y_soft.detach() + y_soft

Tangent Frame Construction (Gram-Schmidt):
    Given normal n:
    1. Choose initial vector a = [1,0,0] (or [0,1,0] if parallel to n)
    2. Project out normal: t1 = normalize(a - (a·n)n)
    3. Cross product: t2 = normalize(n × t1)
    Result: Orthonormal basis [t1, t2, n]

MLS Projection:
    Implicit surface: f(p) = Σ w_i (p - q_i)·n_i
    Gradient: ∇f(p) = Σ w_i n_i =: n̄
    Newton step: p' = p - f(p) * n̄
    Iterate until convergence

Density Equalization:
    Local density: ρ(p) = Σ w_k exp(-||p - p_k||²/h²)
    Deviation: s = tanh((ρ - ρ*) / ρ*)
    Displacement: Δp = -α s Σ w_k (p_k - p) / ρ
    
Covariance from Deformation Gradient:
    Material space: dX (reference configuration)
    World space: dx (current configuration)
    Relationship: dx = F dX
    
    Gaussian in material space: G(X) = exp(-||X||²/(2σ₀²))
    Pullback to world: G(x) = exp(-||F⁻¹x||²/(2σ₀²))
    Covariance: Σ = σ₀² F F^T

PERFORMANCE NOTES
=================

Typical timings (N=1000, M=50000, GPU):
    - Surface detection: ~10ms
    - Sampling: ~5ms
    - Density equalization (8 iters): ~40ms
    - F smoothing: ~15ms
    - Covariance construction: ~5ms
    Total: ~75ms per frame

Memory usage:
    - Peak: ~2GB for M=50000
    - Dominated by k-NN search and density equalization

Optimization tips:
    - Use FAISS IVF for N > 10000
    - Enable AMP for mixed precision (1.5-2x speedup)
    - Adjust M based on scene complexity
    - Disable post_equalize for real-time applications

REFERENCES
==========

[1] Alexa et al. "Point set surfaces." IEEE VIS 2001.
    (MLS surface reconstruction)

[2] Jang et al. "Categorical Reparameterization with Gumbel-Softmax." ICLR 2017.
    (Differentiable sampling)

[3] Qi et al. "PointNet++: Deep Hierarchical Feature Learning." NeurIPS 2017.
    (Adaptive sampling strategies)

[4] Zwicker et al. "EWA Splatting." IEEE TVCG 2002.
    (Gaussian splatting fundamentals)

LEGACY COMPATIBILITY
====================

This module re-exports all functions from the modularized structure to maintain
backward compatibility with existing code that imports from `sampling.runtime_surface`.

New code should import directly from `sampling`:
    from sampling import default_cfg, synthesize_runtime_surface
    
Legacy code continues to work unchanged:
    from sampling.runtime_surface import default_cfg, synthesize_runtime_surface

Author: CHAYO (Hybrid FAISS + Differentiable Pipeline)
Date: 2025-01-18
Version: 1.0.0
"""

# ============================================================================
# Re-exports from modularized structure
# ============================================================================

# Configuration
from ..utils.config import (
    default_cfg,
    EPS_NORMALIZE,
    EPS_SAFE,
    EPS_PCA,
    MIN_PROB,
    TANH_SCALE,
    CLAMP_GUMBEL,
    CLAMP_RANDN,
    CLAMP_SPACING,
    CLAMP_KERNEL_EXP,
    DEFAULT_CONFIG,
    ED_CONFIG,
    POST_EQUALIZE_CONFIG,
    SMOOTHER_CONFIG,
)

# Utilities
from ..utils.utils import (
    ensure_torch as _ensure_torch,
    normalize as _normalize,
    validate_positive_definite as _validate_positive_definite,
    as_numpy as _as_np,
)

# KNN - Hybrid FAISS with differentiable reweighting
from ..analysis.knn import HybridFAISSKNN, FAISS_AVAILABLE

# PCA - Surface normal and variance computation
from ..analysis.pca import (
    batched_pca_surface_optimized,
    compute_weighted_centroid,
    compute_weighted_covariance,
    extract_normal_from_pca,
    compute_local_spacing,
)

# Detection - Surface probability computation
from ..analysis.detection import (
    compute_surface_mask_diff,
    soft_quantile,
    compute_surface_threshold,
    compute_surface_probability,
)

# Gumbel sampling - Differentiable categorical sampling
from ..core.gumbel_sampling import (
    gumbel_softmax_onehot,
    generate_gumbel_noise,
)

# Point sampling - Surface point generation with adaptive jittering
from ..core.point_sampling import (
    sample_surface_points_diff,
    compute_importance_weights,
    build_tangent_frame,
    generate_tangent_jitter,
    compute_adaptive_jitter_scale,
)

# MLS - Moving Least Squares surface projection
from ..geometry.mls import (
    project_to_mls_surface_diff,
    compute_mls_signed_distance,
    compute_mls_normal,
)

# Density - Density equalization with repulsion
from ..analysis.density import (
    density_equalize_diff,
    compute_local_density,
    mask_self_neighbors,
    compute_density_displacement,
)

# Smoother - Tangent Laplacian smoothing
from ..processing.smoother import (
    surface_smoother_diff,
    interpolate_normals_at_points,
    compute_laplacian_displacement,
    split_tangent_normal_components,
    smooth_normals_diff,
)

# Deformation - F field smoothing via graph Laplacian
from ..processing.deformation import (
    smooth_F_diff_optimized,
    select_graph_nodes,
    build_graph_laplacian,
    build_interpolation_weights,
)

# Main synthesis - Complete upsampling pipeline
from ..core.synthesis import (
    synthesize_runtime_surface,
    extract_config_params,
    run_fast_cov_only_path,
    prepare_inputs,
)

# Export - File I/O utilities
from ..io.export import (
    save_comparison_png,
    save_axis_hist_png,
    save_ply_xyz,
    save_gaussians_npz,
    setup_matplotlib,
    compute_plot_bounds,
    set_axis_limits,
)

# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Main entry points
    'default_cfg',
    'synthesize_runtime_surface',
    
    # Configuration constants
    'EPS_NORMALIZE',
    'EPS_SAFE',
    'EPS_PCA',
    'MIN_PROB',
    'TANH_SCALE',
    'CLAMP_GUMBEL',
    'CLAMP_RANDN',
    'CLAMP_SPACING',
    'CLAMP_KERNEL_EXP',
    'DEFAULT_CONFIG',
    'ED_CONFIG',
    'POST_EQUALIZE_CONFIG',
    'SMOOTHER_CONFIG',
    
    # KNN
    'HybridFAISSKNN',
    'FAISS_AVAILABLE',
    
    # PCA
    'batched_pca_surface_optimized',
    'compute_weighted_centroid',
    'compute_weighted_covariance',
    'extract_normal_from_pca',
    'compute_local_spacing',
    
    # Detection
    'compute_surface_mask_diff',
    'soft_quantile',
    'compute_surface_threshold',
    'compute_surface_probability',
    
    # Sampling
    'gumbel_softmax_onehot',
    'generate_gumbel_noise',
    'sample_surface_points_diff',
    'compute_importance_weights',
    'build_tangent_frame',
    'generate_tangent_jitter',
    'compute_adaptive_jitter_scale',
    
    # MLS & Density
    'project_to_mls_surface_diff',
    'compute_mls_signed_distance',
    'compute_mls_normal',
    'density_equalize_diff',
    'compute_local_density',
    'mask_self_neighbors',
    'compute_density_displacement',
    
    # Smoothing
    'surface_smoother_diff',
    'interpolate_normals_at_points',
    'compute_laplacian_displacement',
    'split_tangent_normal_components',
    'smooth_normals_diff',
    
    # Deformation
    'smooth_F_diff_optimized',
    'select_graph_nodes',
    'build_graph_laplacian',
    'build_interpolation_weights',
    
    # Synthesis helpers
    'extract_config_params',
    'run_fast_cov_only_path',
    'prepare_inputs',
    
    # Export utilities
    'save_comparison_png',
    'save_axis_hist_png',
    'save_ply_xyz',
    'save_gaussians_npz',
    'setup_matplotlib',
    'compute_plot_bounds',
    'set_axis_limits',
]

__version__ = '1.0.0'
__author__ = 'CHAYO'
__date__ = '2025-01-18'


# ============================================================================
# Quick Start Examples
# ============================================================================

def example_basic_usage():
    """
    Basic upsampling example.
    
    This demonstrates the minimal code needed to upsample a point cloud.
    """
    import torch
    
    # Create synthetic coarse point cloud
    N = 1000
    x_low = torch.randn(N, 3, device='cuda')  # Positions
    F_low = torch.eye(3, device='cuda').unsqueeze(0).expand(N, 3, 3)  # Identity deformation
    
    # Get default configuration
    cfg = default_cfg()
    cfg["M"] = 50_000  # Target number of points
    
    # Run upsampling
    result = synthesize_runtime_surface(
        x_low, F_low, cfg,
        seed=42,
        differentiable=True,
        return_torch=True
    )
    
    # Extract results
    points = result["points"]  # (M, 3)
    normals = result["normals"]  # (M, 3)
    cov = result["cov"]  # (M, 3, 3)
    
    print(f"Upsampled {N} → {len(points)} points")
    return result


def example_custom_config():
    """
    Advanced configuration example.
    
    Shows how to customize various aspects of the upsampling pipeline.
    """
    import torch
    
    N = 500
    x_low = torch.randn(N, 3, device='cuda')
    F_low = torch.eye(3, device='cuda').unsqueeze(0).expand(N, 3, 3)
    
    # Customize configuration
    cfg = default_cfg()
    
    # Increase point count and jitter
    cfg["M"] = 100_000
    cfg["surf_jitter_alpha"] = 0.8
    
    # Enable advanced features
    cfg["post_equalize"]["enabled"] = True
    cfg["post_equalize"]["iters"] = 12
    cfg["smoother"]["enabled"] = True
    
    # Adjust surface detection
    cfg["k_surface"] = 48
    cfg["surface_power"] = 6.0
    
    result = synthesize_runtime_surface(x_low, F_low, cfg)
    return result


def example_fast_path():
    """
    Fast path example for animation.
    
    When positions don't change but F (deformation) does, we can reuse
    the upsampled positions and only recompute covariances.
    """
    import torch
    
    N = 1000
    x_low = torch.randn(N, 3, device='cuda')
    F_low = torch.eye(3, device='cuda').unsqueeze(0).expand(N, 3, 3)
    
    cfg = default_cfg()
    cfg["M"] = 50_000
    
    # First frame: full upsampling
    result1 = synthesize_runtime_surface(x_low, F_low, cfg)
    ema_state = result1["state"]
    
    # Subsequent frames: fast path (positions cached)
    F_low_updated = F_low * 1.1  # Deformation changed
    cfg["fast_cov_only"] = True
    
    result2 = synthesize_runtime_surface(
        x_low, F_low_updated, cfg,
        ema_state=ema_state  # Pass previous state
    )
    
    # Positions are reused, only covariances recomputed (10-20x faster)
    print(f"Same positions: {torch.allclose(result1['points'], result2['points'])}")
    print(f"Different covariances: {not torch.allclose(result1['cov'], result2['cov'])}")
    
    return result2


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("Running basic example...")
    print("="*80)
    example_basic_usage()