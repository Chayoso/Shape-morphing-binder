"""
renderer - 3D Gaussian Splatting Rendering System

A modular rendering pipeline for differentiable 3D Gaussian splatting.

Components:
    - Core: GSRenderer3DGS renderer
    - Camera: Projection and view transforms
    - Shading: Lighting and shading models
    - Composite: Image compositing and blending
    - Utils: Common utilities

Example:
    >>> from renderer import GSRenderer3DGS, make_matrices_from_yaml, compute_shading
    >>> 
    >>> # Setup camera
    >>> W, H, tfovx, tfovy, view, proj, campos = make_matrices_from_yaml(cam_cfg)
    >>> 
    >>> # Create renderer
    >>> renderer = GSRenderer3DGS(W, H, tfovx, tfovy, view, proj, campos)
    >>> 
    >>> # Render
    >>> output = renderer.render(xyz, cov, rgb, return_torch=True)
"""

__version__ = "2.0.0"

# Core
from .core import GSRenderer3DGS, RenderConfig

# Camera
from .camera import (
    make_matrices_from_yaml,
    build_lookat_camera_pose,
    compute_tan_half_fov,
    build_gl_projection_matrix,
    ensure_4x4_matrix,
    invert_transform,
)

# Shading
from .shading import (
    compute_shading,
    LightConfig,
    compute_lambert_shading,
    compute_phong_shading,
    orient_normals,
)

# Composite
from .composite import (
    composite_with_background,
    alpha_blend_composite,
    depth_test_composite,
    normalize_to_float32,
    resize_image,
)

# Utils
from .utils import (
    to_torch_tensor,
    to_numpy_array,
    pack_covariance_3x3_to_6d,
    decompose_covariance_to_scale_rotation,
    debug_print,
    is_debug_enabled,
)

__all__ = [
    "__version__",
    
    # Core
    "GSRenderer3DGS",
    "RenderConfig",
    
    # Camera
    "make_matrices_from_yaml",
    "build_lookat_camera_pose",
    "compute_tan_half_fov",
    "build_gl_projection_matrix",
    "ensure_4x4_matrix",
    "invert_transform",
    
    # Shading
    "compute_shading",
    "LightConfig",
    "compute_lambert_shading",
    "compute_phong_shading",
    "orient_normals",
    
    # Composite
    "composite_with_background",
    "alpha_blend_composite",
    "depth_test_composite",
    "normalize_to_float32",
    "resize_image",
    
    # Utils
    "to_torch_tensor",
    "to_numpy_array",
    "pack_covariance_3x3_to_6d",
    "decompose_covariance_to_scale_rotation",
    "debug_print",
    "is_debug_enabled",
]