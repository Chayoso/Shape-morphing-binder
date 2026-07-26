"""
Geometric operations for covariance construction.

This module contains all covariance-related functionality:
- Deformation-based: F-field interpolation for predicted meshes (Episode > 0)
- Curvature-based: Planarity/anisotropy for target meshes (Episode -1, 0)
- Learnable: Neural network refinement (optional)
"""

from .deformation_covariance import (
    build_deformation_covariance,
    smooth_F_field,
    polar_decomposition,
    select_graph_nodes,
    build_graph_laplacian,
    build_interpolation_weights,
    soft_top_k,
    gumbel_select_k,
)

from .curvature_covariance import (
    create_curvature_based_covariance_star,
    # apply_target_covariance_patch,  # DEPRECATED - removed
)

from .learnable_covariance import (
    LearnableCovariance,
)


__all__ = [
    # Deformation-based (F-field)
    "build_deformation_covariance",
    "smooth_F_field",
    "polar_decomposition",
    "select_graph_nodes",
    "build_graph_laplacian",
    "build_interpolation_weights",
    "soft_top_k",
    "gumbel_select_k",
    
    # Curvature-based (Target Σ★)
    "create_curvature_based_covariance_star",
    # "apply_target_covariance_patch",  # DEPRECATED - removed
    
    # Learnable (Optional)
    "LearnableCovariance",
]
