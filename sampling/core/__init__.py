"""
Core sampling operations.

Includes:
- Importance sampling (Gumbel-Softmax)
- Normal smoothing (Laplacian)

Note: 
- Taubin smoothing: 미사용 (F-smoothing + Σ 비등방으로 대체)
- Surface detection: SDF-based pipeline (analysis/levelset.py)
"""

from .sampler import (
    sample_points,
    sample_points_fast,
    precompute_knn_indices,
    precompute_tangent_bases,
    build_pi_base,
    build_pi_complete,
    soft_nms_pi,
)

# from .taubin_smooth import (  # 🔥 미사용 (F-smoothing + Σ로 대체)
#     taubin_smooth,
#     compute_laplacian,
#     split_tangent_normal,
# )

from .normal_smooth import (
    smooth_normals,
)


__all__ = [
    # Sampling
    "sample_points",
    "sample_points_fast",
    "precompute_knn_indices",
    "precompute_tangent_bases",
    "build_pi_base",
    "build_pi_complete",
    "soft_nms_pi",
    
    # Normal smoothing
    "smooth_normals",
]