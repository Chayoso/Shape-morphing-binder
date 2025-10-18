"""Camera matrix utilities."""

from __future__ import annotations
from typing import Tuple
import numpy as np


def ensure_4x4_matrix(m) -> np.ndarray:
    """
    Convert input to 4x4 numpy array.
    
    Args:
        m: Input matrix (4x4 array or flat list of 16 floats)
    
    Returns:
        4x4 float32 numpy array
    
    Raises:
        ValueError: If input cannot be reshaped to 4x4
    """
    M = np.asarray(m, dtype=np.float32)
    
    if M.shape == (16,):
        M = M.reshape(4, 4)
    
    if M.shape != (4, 4):
        raise ValueError(
            f"Expected 4x4 matrix or flat length-16 array, got shape {M.shape}"
        )
    
    return M


def invert_transform(m: np.ndarray) -> np.ndarray:
    """
    Compute inverse of 4x4 transformation matrix.
    
    Args:
        m: 4x4 transformation matrix
    
    Returns:
        Inverted 4x4 matrix (float32)
    """
    return np.linalg.inv(m).astype(np.float32)


def compute_tan_half_fov(
    fx: float, 
    fy: float, 
    width: int, 
    height: int
) -> Tuple[float, float]:
    """
    Compute tangent of half FOV from pinhole camera intrinsics.
    
    For a pinhole camera:
        tan(FOVx/2) = width / (2 * fx)
        tan(FOVy/2) = height / (2 * fy)
    
    Args:
        fx: Focal length in x (pixels)
        fy: Focal length in y (pixels)
        width: Image width (pixels)
        height: Image height (pixels)
    
    Returns:
        (tanfovx, tanfovy): Tangent of half horizontal and vertical FOV
    """
    tanfovx = float(width) / (2.0 * float(fx))
    tanfovy = float(height) / (2.0 * float(fy))
    return tanfovx, tanfovy