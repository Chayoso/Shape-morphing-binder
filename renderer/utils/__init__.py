"""Common utilities for rendering."""

from .conversion import (
    to_torch_tensor,
    to_numpy_array,
)
from .covariance import (
    pack_covariance_3x3_to_6d,
    unpack_covariance_6d_to_3x3,
    pack_covariance_torch,
    decompose_covariance_to_scale_rotation,
    rotation_matrix_to_quaternion,
)
from .projection_2d import project_points_to_screen
from .validation import (
    validate_render_inputs,
    validate_shading_inputs,
)
from .debug import (
    is_debug_enabled,
    debug_print,
    get_tensor_stats,
)

__all__ = [
    # Conversion
    "to_torch_tensor",
    "to_numpy_array",
    
    # Covariance
    "pack_covariance_3x3_to_6d",
    "unpack_covariance_6d_to_3x3",
    "pack_covariance_torch",
    "decompose_covariance_to_scale_rotation",
    "rotation_matrix_to_quaternion",
    
    # Projection
    "project_points_to_screen",
    
    # Validation
    "validate_render_inputs",
    "validate_shading_inputs",
    
    # Debug
    "is_debug_enabled",
    "debug_print",
    "get_tensor_stats",
]