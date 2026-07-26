"""2D projection utilities."""

from __future__ import annotations
from typing import Tuple
import numpy as np


def project_points_to_screen(
    xyz: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D screen coordinates.
    
    Args:
        xyz: (N, 3) world-space positions
        proj_matrix: (4, 4) projection matrix (transposed)
        width: Image width
        height: Image height
    
    Returns:
        means2D: (N, 2) screen coordinates (u, v)
        valid: (N,) boolean mask for valid projections
    
    Notes:
        - Handles points behind camera (w < 0)
        - Invalid points set to (-1e6, -1e6)
        - Uses perspective division (clip space → NDC → screen)
    """
    N = xyz.shape[0]
    
    # Homogeneous coordinates
    xyz_homogeneous = np.concatenate([
        xyz.astype(np.float32),
        np.ones((N, 1), dtype=np.float32)
    ], axis=1)
    
    # Apply projection
    proj = proj_matrix.T.astype(np.float32)
    clip = xyz_homogeneous @ proj.T
    
    # Perspective division
    w = clip[:, 3:4]
    
    # Validate: finite and in front of camera
    valid = np.isfinite(clip).all(axis=1) & (w[:, 0] > 0)
    
    # NDC coordinates
    ndc = np.full((N, 2), np.nan, dtype=np.float32)
    ndc[valid, 0] = clip[valid, 0] / w[valid, 0]
    ndc[valid, 1] = clip[valid, 1] / w[valid, 0]
    
    # Screen coordinates
    u = (ndc[:, 0] * 0.5 + 0.5) * float(width)
    v = (-ndc[:, 1] * 0.5 + 0.5) * float(height)
    
    means2D = np.stack([u, v], axis=1).astype(np.float32)
    
    # Mark invalid points
    invalid_mask = ~np.isfinite(means2D).all(axis=1)
    means2D[invalid_mask] = np.array([-1e6, -1e6], dtype=np.float32)
    
    return means2D, valid