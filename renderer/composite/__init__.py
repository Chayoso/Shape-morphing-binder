"""Compositing system for image blending."""

from .utils import (
    normalize_to_float32,
    resize_image,
    ensure_rgb_format,
    normalize_alpha,
)
from .blend import alpha_blend_composite
from .depth import depth_test_composite
from .main import composite_with_background

__all__ = [
    # Utils
    "normalize_to_float32",
    "resize_image",
    "ensure_rgb_format",
    "normalize_alpha",
    
    # Compositing
    "alpha_blend_composite",
    "depth_test_composite",
    "composite_with_background",
]