"""Input validation utilities."""

from __future__ import annotations
import numpy as np


def validate_render_inputs(xyz: np.ndarray, cov):
    """
    Validate renderer inputs.
    
    Args:
        xyz: Point positions
        cov: Covariances
    
    Raises:
        ValueError: If inputs are invalid
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N, 3), got {xyz.shape}")
    
    if not np.isfinite(xyz).all():
        raise ValueError("xyz contains NaN or Inf")
    
    # Validate covariance shape
    if isinstance(cov, list):
        if len(cov) != xyz.shape[0]:
            raise ValueError(f"cov list length {len(cov)} != xyz length {xyz.shape[0]}")
    else:
        cov_array = np.asarray(cov)
        if cov_array.ndim == 2 and cov_array.shape[1] != 6:
            raise ValueError(f"Packed cov must be (N, 6), got {cov_array.shape}")
        elif cov_array.ndim == 3 and cov_array.shape[1:] != (3, 3):
            raise ValueError(f"Full cov must be (N, 3, 3), got {cov_array.shape}")


def validate_shading_inputs(
    xyz: np.ndarray,
    normals: np.ndarray,
    camera_pos: np.ndarray
):
    """
    Validate shading computation inputs.
    
    Args:
        xyz: Point positions
        normals: Surface normals
        camera_pos: Camera position
    
    Raises:
        ValueError: If inputs are invalid
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N, 3), got {xyz.shape}")
    
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError(f"normals must be (N, 3), got {normals.shape}")
    
    if xyz.shape[0] != normals.shape[0]:
        raise ValueError(f"xyz and normals must have same length")
    
    if camera_pos.shape != (3,):
        raise ValueError(f"camera_pos must be (3,), got {camera_pos.shape}")
    
    if not np.isfinite(xyz).all():
        raise ValueError("xyz contains NaN or Inf")
    
    if not np.isfinite(normals).all():
        raise ValueError("normals contains NaN or Inf")