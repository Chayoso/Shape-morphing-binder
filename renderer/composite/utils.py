"""Image processing utilities for compositing."""

from __future__ import annotations
from typing import Tuple
import numpy as np


DEFAULT_DEPTH_EPSILON = 1e-6
MIN_VALID_DEPTH = 1e-6


def normalize_to_float32(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to float32 in range [0, 1].
    
    Args:
        img: Input image (any dtype)
    
    Returns:
        Normalized float32 image in [0, 1]
    """
    img = np.asarray(img)
    
    if img.dtype == np.float32:
        return np.clip(img, 0.0, 1.0)
    
    # Convert uint8/uint16 to float32
    if img.dtype == np.uint8:
        scale = 255.0
    elif img.dtype == np.uint16:
        scale = 65535.0
    else:
        # Generic normalization for other types
        scale = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 1.0
    
    img_float = img.astype(np.float32) / scale
    return np.clip(img_float, 0.0, 1.0)


def resize_image(
    img: np.ndarray, 
    target_height: int, 
    target_width: int,
    use_bilinear: bool = True
) -> np.ndarray:
    """
    Resize image to target dimensions.
    
    Uses PIL for high-quality resampling if available,
    otherwise falls back to numpy-based nearest neighbor.
    
    Args:
        img: Input image (H, W) or (H, W, C)
        target_height: Target height
        target_width: Target width
        use_bilinear: Use bilinear interpolation (PIL only)
    
    Returns:
        Resized image with same dtype as input
    
    Notes:
        - Preserves channel dimension
        - Converts to/from uint8 for PIL processing
        - Falls back to nearest neighbor if PIL unavailable
    """
    try:
        from PIL import Image
        
        # Determine mode
        if img.ndim == 3 and img.shape[2] == 3:
            mode = 'RGB'
        elif img.ndim == 3 and img.shape[2] == 4:
            mode = 'RGBA'
        else:
            mode = 'L'  # Grayscale
        
        # Convert to uint8 for PIL
        if img.dtype == np.float32:
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img_uint8 = img
        
        # Create PIL image
        pil_img = Image.fromarray(img_uint8, mode=mode)
        
        # Resize
        resample_mode = Image.BILINEAR if (use_bilinear and mode in ['RGB', 'RGBA']) else Image.NEAREST
        pil_resized = pil_img.resize((target_width, target_height), resample=resample_mode)
        
        # Convert back to numpy
        resized = np.asarray(pil_resized)
        
        # Restore shape for grayscale
        if resized.ndim == 2 and img.ndim == 3:
            resized = resized[..., None]
        
        # Convert back to float32 if needed
        if img.dtype == np.float32:
            resized = resized.astype(np.float32) / 255.0
        
        return resized
    
    except ImportError:
        # Fallback: nearest neighbor using numpy
        return _resize_nearest_neighbor(img, target_height, target_width)


def _resize_nearest_neighbor(
    img: np.ndarray, 
    target_height: int, 
    target_width: int
) -> np.ndarray:
    """
    Fallback resize using nearest neighbor interpolation.
    
    Args:
        img: Input image
        target_height: Target height
        target_width: Target width
    
    Returns:
        Resized image
    """
    src_h, src_w = img.shape[:2]
    
    # Compute sampling indices
    y_indices = np.linspace(0, src_h - 1, target_height).astype(np.int32)
    x_indices = np.linspace(0, src_w - 1, target_width).astype(np.int32)
    
    # Sample using advanced indexing
    if img.ndim == 2:
        return img[np.ix_(y_indices, x_indices)]
    else:
        return img[np.ix_(y_indices, x_indices)]


def ensure_rgb_format(img: np.ndarray) -> np.ndarray:
    """
    Convert image to RGB format (H, W, 3).
    
    Args:
        img: Input image (H, W) or (H, W, C)
    
    Returns:
        RGB image (H, W, 3)
    """
    if img.ndim == 2:
        # Grayscale -> RGB by stacking
        return np.stack([img, img, img], axis=-1)
    elif img.ndim == 3:
        if img.shape[2] == 1:
            # Single channel -> RGB
            return np.concatenate([img, img, img], axis=-1)
        elif img.shape[2] == 3:
            # Already RGB
            return img
        elif img.shape[2] == 4:
            # RGBA -> RGB (drop alpha)
            return img[..., :3]
        else:
            raise ValueError(f"Unsupported number of channels: {img.shape[2]}")
    else:
        raise ValueError(f"Unsupported image dimensions: {img.ndim}")


def normalize_alpha(alpha: np.ndarray) -> np.ndarray:
    """
    Normalize alpha channel to 2D array [0, 1].
    
    Args:
        alpha: Alpha channel (H, W) or (H, W, 1) or (H, W, C)
    
    Returns:
        2D alpha array (H, W) in [0, 1]
    """
    alpha = np.asarray(alpha, dtype=np.float32)
    
    # Squeeze channel dimension if present
    if alpha.ndim == 3:
        alpha = alpha[..., 0]
    
    return np.clip(alpha, 0.0, 1.0)