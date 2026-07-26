"""Look-at camera transform."""

from __future__ import annotations
from typing import Optional
import numpy as np


def build_lookat_camera_pose(
    eye: np.ndarray,
    target: np.ndarray,
    up: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Build camera-to-world (c2w) matrix from look-at parameters.
    
    Constructs an OpenCV-style camera coordinate system:
        +Z = forward (eye -> target)
        +X = right
        +Y = down
    
    The output matrix is column-major, where columns represent:
        [right, down, forward, position]
    
    Args:
        eye: (3,) Camera position in world coordinates
        target: (3,) Look-at point in world coordinates
        up: (3,) World 'up' direction hint (default: Y-up)
            Note: This is NOT the camera's up direction, but a world-space
            hint to define the camera's horizontal plane orientation.
    
    Returns:
        c2w: (4,4) camera-to-world transformation matrix (float32)
    
    Notes:
        - Right-handed coordinate system
        - If 'forward' and 'up' are parallel, an alternative up vector is chosen
        - Output matrix can be inverted to get world-to-camera (w2c)
    """
    if up is None:
        up = np.array([0, 1, 0], dtype=np.float32)
    
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    
    # Step 1: Compute forward direction (camera Z-axis)
    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    
    if forward_norm < 1e-12:
        raise ValueError("Eye and target positions are too close (degenerate camera)")
    
    forward = forward / forward_norm
    
    # Step 2: Compute right direction (camera X-axis)
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    
    # Handle degenerate case: forward and up are parallel
    if right_norm < 1e-6:
        # Choose alternative up vector
        if abs(np.dot(forward, [0, 0, 1])) < 0.999:
            alt_up = np.array([0, 0, 1], dtype=np.float32)
        else:
            alt_up = np.array([1, 0, 0], dtype=np.float32)
        
        right = np.cross(forward, alt_up)
        right_norm = np.linalg.norm(right)
    
    right = right / right_norm
    
    # Step 3: Compute down direction (camera Y-axis)
    # For OpenCV (+Y = down), use: down = forward × right
    # This ensures right-handed system: right × down = forward
    down = np.cross(forward, right)
    # No normalization needed (forward and right are orthonormal)
    
    # Step 4: Assemble camera-to-world matrix
    # Rotation part: columns are [right, down, forward]
    R = np.stack([right, down, forward], axis=1)
    
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = eye
    
    return c2w