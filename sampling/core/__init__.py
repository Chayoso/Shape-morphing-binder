"""
Core sampling operations.

Includes:
- Surface detection (PCA-based)
- Volume filtering (soft)
- Importance sampling (Gumbel-Softmax)
- Taubin smoothing (differentiable)
- Normal smoothing (Laplacian)
"""

from .surface_detect import (
    detect_surface,
    soft_quantile,
    compute_surface_threshold,
    compute_surface_probability,
)

from .volume_filter import (
    apply_volume_filter,
    compute_normal_consistency,
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
    "compute_surface_threshold",
    "compute_surface_probability",
    
    # Volume filtering
    "apply_volume_filter",
    "compute_normal_consistency",
    
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