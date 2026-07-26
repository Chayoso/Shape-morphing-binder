"""Projection matrix construction."""

from __future__ import annotations
import numpy as np


def build_gl_projection_matrix(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    znear: float,
    zfar: float
) -> np.ndarray:
    """
    Build OpenGL-style perspective projection matrix from pixel intrinsics.
    
    This matrix maps camera space coordinates to homogeneous clip space.
    NDC (Normalized Device Coordinates) are obtained by perspective division.
    
    The matrix uses OpenCV camera convention (+Z forward) and produces
    clip.w = z_cam > 0 for points in front of the camera.
    
    Args:
        fx, fy: Focal lengths (pixels)
        cx, cy: Principal point (pixels)
        width, height: Image dimensions (pixels)
        znear, zfar: Near and far clipping planes
    
    Returns:
        4x4 projection matrix (row-major, float32)
    
    Notes:
        - Output is row-major; transpose before passing to rasterizer
        - This is the standard intrinsics->OpenGL mapping
        - Depth encoding: z_ndc = (z_cam * zfar) / (z_cam * (zfar - znear) + znear * zfar)
    """
    P = np.zeros((4, 4), dtype=np.float32)
    
    # Scale factors to map from pixels to NDC [-1, 1]
    P[0, 0] = 2.0 * fx / float(width)
    P[1, 1] = 2.0 * fy / float(height)
    
    # Principal point offset
    P[0, 2] = 1.0 - 2.0 * cx / float(width)
    P[1, 2] = 2.0 * cy / float(height) - 1.0
    
    # Depth encoding (OpenGL-like with +Z forward)
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = (-znear * zfar) / (zfar - znear)
    
    # Perspective division: clip.w = z_cam
    P[3, 2] = 1.0
    
    return P