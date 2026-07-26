"""Depth-test compositing."""

from __future__ import annotations
from typing import Optional
import numpy as np

from .utils import normalize_alpha, DEFAULT_DEPTH_EPSILON, MIN_VALID_DEPTH


def depth_test_composite(
    fg_rgb: np.ndarray,
    fg_depth: np.ndarray,
    bg_rgb: np.ndarray,
    bg_depth: np.ndarray,
    fg_alpha: Optional[np.ndarray] = None,
    depth_epsilon: float = DEFAULT_DEPTH_EPSILON
) -> np.ndarray:
    """
    Depth-aware Z-test compositing.
    
    Determines visibility based on depth values:
    - If fg_depth < bg_depth: show foreground
    - If bg_depth < fg_depth: show background
    - If either invalid: use alpha blending or default
    
    Args:
        fg_rgb: Foreground RGB (H, W, 3) in [0, 1]
        fg_depth: Foreground depth (H, W), z > 0 for valid
        bg_rgb: Background RGB (H, W, 3) in [0, 1]
        bg_depth: Background depth (H, W), z > 0 for valid
        fg_alpha: Optional alpha for invalid regions
        depth_epsilon: Threshold for depth comparison
    
    Returns:
        Composited RGB image (H, W, 3) in [0, 1]
    
    Notes:
        - Depth values should be in camera space (z-forward)
        - 0 or negative depth indicates invalid/empty pixels
        - Epsilon prevents z-fighting artifacts
    """
    # Convert to float32
    depth_fg = fg_depth.astype(np.float32)
    depth_bg = bg_depth.astype(np.float32)
    
    # Determine valid pixels
    valid_fg = depth_fg > MIN_VALID_DEPTH
    valid_bg = depth_bg > MIN_VALID_DEPTH
    
    # Initialize output with background
    out = bg_rgb.copy()
    
    # Case 1: Foreground is in front (or background invalid)
    fg_in_front = valid_fg & (~valid_bg | (depth_fg <= depth_bg - depth_epsilon))
    out[fg_in_front] = fg_rgb[fg_in_front]
    
    # Case 2: Background is in front -> already in 'out' (default)
    
    # Case 3: Both invalid -> use alpha blending if available
    both_invalid = (~valid_fg) & (~valid_bg)
    
    if both_invalid.any() and fg_alpha is not None:
        alpha = normalize_alpha(fg_alpha)
        alpha_expanded = alpha[..., None]
        
        out[both_invalid] = (
            fg_rgb[both_invalid] * alpha_expanded[both_invalid] +
            bg_rgb[both_invalid] * (1.0 - alpha_expanded[both_invalid])
        )
    
    return np.clip(out, 0.0, 1.0)