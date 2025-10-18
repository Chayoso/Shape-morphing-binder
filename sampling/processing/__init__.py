"""
Surface processing operations.

Includes:
- Surface deformation (F field smoothing)
- Surface smoothing (tangent Laplacian)
- Normal smoothing
"""

# ============================================================================
# Deformation
# ============================================================================
from .deformation import (
    select_graph_nodes,
    build_graph_laplacian,
    build_interpolation_weights,
    smooth_F_diff_optimized,
)

# ============================================================================
# Smoothing
# ============================================================================
from .smoother import (
    interpolate_normals_at_points,
    compute_laplacian_displacement,
    split_tangent_normal_components,
    surface_smoother_diff,
    smooth_normals_diff,
)


__all__ = [
    # Deformation
    "select_graph_nodes",
    "build_graph_laplacian",
    "build_interpolation_weights",
    "smooth_F_diff_optimized",
    
    # Smoothing
    "interpolate_normals_at_points",
    "compute_laplacian_displacement",
    "split_tangent_normal_components",
    "surface_smoother_diff",
    "smooth_normals_diff",
]