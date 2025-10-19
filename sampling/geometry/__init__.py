"""
Geometric operations.

Includes:
- Covariance construction (F-field interpolation)
- F-field smoothing (graph Laplacian)
"""

from .covariance import (
    build_covariance,
    smooth_F_field,
    select_graph_nodes,
    build_graph_laplacian,
    build_interpolation_weights,
)


__all__ = [
    "build_covariance",
    "smooth_F_field",
    "select_graph_nodes",
    "build_graph_laplacian",
    "build_interpolation_weights",
]