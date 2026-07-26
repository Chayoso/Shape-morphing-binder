"""Main compositing interface."""

from __future__ import annotations
from typing import Optional
import numpy as np

from .utils import (
    normalize_to_float32,
    resize_image,
    ensure_rgb_format,
    DEFAULT_DEPTH_EPSILON
)
from .blend import alpha_blend_composite
from .depth import depth_test_composite


def composite_with_background(
    fg_rgb: np.ndarray,
    fg_alpha: Optional[np.ndarray] = None,
    fg_depth: Optional[np.ndarray] = None,
    bg_rgb: Optional[np.ndarray] = None,
    bg_depth: Optional[np.ndarray] = None,
    depth_epsilon: float = DEFAULT_DEPTH_EPSILON
) -> np.ndarray:
    """
    Composite foreground rendering with background.
    
    Automatically selects compositing mode:
    - Z-test compositing if both fg_depth and bg_depth provided
    - Alpha blending otherwise
    
    Args:
        fg_rgb: Foreground RGB (H, W, 3) float32 in [0, 1]
        fg_alpha: Foreground alpha (H, W) float32 in [0, 1], optional
        fg_depth: Foreground depth (H, W) float32, z > 0 for valid, optional
        bg_rgb: Background RGB (H, W, 3) float32 in [0, 1], optional
        bg_depth: Background depth (H, W) float32, z > 0 for valid, optional
        depth_epsilon: Epsilon for depth comparison (default: 1e-6)
    
    Returns:
        Composited RGB image (H, W, 3) in [0, 1]
    
    Raises:
        ValueError: If input shapes are invalid
    
    Examples:
        >>> # Alpha blending
        >>> result = composite_with_background(
        ...     fg_rgb=rendered_image,
        ...     fg_alpha=rendered_alpha,
        ...     bg_rgb=background_image
        ... )
        
        >>> # Z-test compositing
        >>> result = composite_with_background(
        ...     fg_rgb=rendered_image,
        ...     fg_depth=rendered_depth,
        ...     bg_rgb=background_image,
        ...     bg_depth=background_depth
        ... )
    """
    # Validate input
    if fg_rgb.ndim != 3 or fg_rgb.shape[2] != 3:
        raise ValueError(f"fg_rgb must be (H, W, 3), got {fg_rgb.shape}")
    
    H, W, _ = fg_rgb.shape
    
    # If no background, return foreground
    if bg_rgb is None:
        return fg_rgb
    
    # Prepare background
    bg_rgb_processed = normalize_to_float32(bg_rgb)
    
    # Resize background if needed
    if bg_rgb_processed.shape[:2] != (H, W):
        bg_rgb_processed = resize_image(bg_rgb_processed, H, W)
    
    # Ensure RGB format
    bg_rgb_processed = ensure_rgb_format(bg_rgb_processed)
    
    # Select compositing mode
    if fg_depth is not None and bg_depth is not None:
        # Depth-based compositing
        return depth_test_composite(
            fg_rgb=fg_rgb,
            fg_depth=fg_depth,
            bg_rgb=bg_rgb_processed,
            bg_depth=bg_depth,
            fg_alpha=fg_alpha,
            depth_epsilon=depth_epsilon
        )
    else:
        # Alpha blending
        return alpha_blend_composite(
            fg_rgb=fg_rgb,
            fg_alpha=fg_alpha,
            bg_rgb=bg_rgb_processed
        )