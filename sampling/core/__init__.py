"""
Core sampling operations.

Includes:
- Surface detection (PCA-based)
- Anchor-density map (differentiable)
- Importance sampling (Gumbel-Softmax)
- Taubin smoothing (differentiable)
- Normal smoothing (Laplacian)
"""

from .surface_detect import (
    detect_surface,
)

from .density_map import (
    build_anchor_density,
    prepare_sampling_cfg,
    run_stage2_anchor_density,
)

from .sampler import (
    sample_points,
    sample_points_fast,
    precompute_knn_indices,
    precompute_tangent_bases,
    build_pi_base,
    build_pi_complete,
    soft_nms_pi,
)

from .taubin_smooth import (
    taubin_smooth,
    compute_laplacian,
    split_tangent_normal,
)

from .normal_smooth import (
    smooth_normals,
)


__all__ = [
    # Surface detection
    "detect_surface",
    
    # Anchor-density map
    "build_anchor_density",
    "prepare_sampling_cfg",
    "run_stage2_anchor_density",
    
    # Sampling
    "sample_points",
    "sample_points_fast",
    "precompute_knn_indices",
    "precompute_tangent_bases",
    "build_pi_base",
    "build_pi_complete",
    "soft_nms_pi",
    
    # Taubin smoothing
    "taubin_smooth",
    "compute_laplacian",
    "split_tangent_normal",
    
    # Normal smoothing
    "smooth_normals",
]