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
    soft_quantile,
)

from .density_map import (
    build_anchor_density,
    prepare_sampling_cfg,
    run_stage2_anchor_density,
)

from .sampler import (
    sample_points,
    gumbel_softmax_sample,
    build_tangent_frame,
    generate_tangent_jitter,
    compute_adaptive_jitter_scale,
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
    "soft_quantile",
    
    # Anchor-density map
    "build_anchor_density",
    "prepare_sampling_cfg",
    "run_stage2_anchor_density",
    
    # Sampling
    "sample_points",
    "gumbel_softmax_sample",
    "build_tangent_frame",
    "generate_tangent_jitter",
    "compute_adaptive_jitter_scale",
    
    # Taubin smoothing
    "taubin_smooth",
    "compute_laplacian",
    "split_tangent_normal",
    
    # Normal smoothing
    "smooth_normals",
]