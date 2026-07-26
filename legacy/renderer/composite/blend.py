"""Alpha blending compositing."""

from __future__ import annotations
from typing import Optional
import numpy as np

from .utils import normalize_alpha


def alpha_blend_composite(
    fg_rgb: np.ndarray,
    fg_alpha: Optional[np.ndarray],
    bg_rgb: np.ndarray
) -> np.ndarray:
    """
    Standard alpha-over compositing.
    
    Formula: out = fg * α + bg * (1 - α)
    
    Args:
        fg_rgb: Foreground RGB (H, W, 3) in [0, 1]
        fg_alpha: Foreground alpha (H, W) in [0, 1], or None for opaque
        bg_rgb: Background RGB (H, W, 3) in [0, 1]
    
    Returns:
        Composited RGB image (H, W, 3) in [0, 1]
    """
    if fg_alpha is None:
        # Opaque foreground
        return fg_rgb
    
    # Normalize alpha
    alpha = normalize_alpha(fg_alpha)
    
    # Alpha blending
    out = fg_rgb * alpha[..., None] + bg_rgb * (1.0 - alpha[..., None])
    
    return np.clip(out, 0.0, 1.0)