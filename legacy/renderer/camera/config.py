"""Camera configuration parser."""

from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np

from .utils import ensure_4x4_matrix, invert_transform, compute_tan_half_fov
from .projection import build_gl_projection_matrix
from .lookat import build_lookat_camera_pose


def make_matrices_from_yaml(
    camera_cfg: Dict[str, Any]
) -> Tuple[int, int, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build camera matrices from configuration dictionary.
    
    This is the main entry point for creating rendering matrices from
    camera parameters, typically loaded from a YAML config file.
    
    Args:
        camera_cfg: Camera configuration dictionary with keys:
            Required:
                - width, height: Image dimensions (int)
                - fx, fy: Focal lengths in pixels (float)
                - cx, cy: Principal point in pixels (float)
            Optional:
                - znear, zfar: Clipping planes (float, default: 0.01, 100.0)
                - lookat: Look-at parameters (dict)
                    - eye: Camera position [x, y, z]
                    - target: Look-at point [x, y, z]
                    - up: Up direction hint [x, y, z] (default: [0, 1, 0])
                - c2w: Camera-to-world matrix (4x4 or flat 16 values)
                    Note: If 'lookat' is provided, 'c2w' is ignored
    
    Returns:
        Tuple of:
            - width (int): Image width
            - height (int): Image height
            - tanfovx (float): tan(FOVx/2)
            - tanfovy (float): tan(FOVy/2)
            - view_matrix (4x4 float32): World-to-view transform (transposed)
            - proj_matrix (4x4 float32): Full projection (transposed)
            - camera_position (3, float32): Camera position in world space
    
    Example:
        >>> config = {
        ...     "width": 1280, "height": 720,
        ...     "fx": 1152.0, "fy": 1152.0,
        ...     "cx": 640.0, "cy": 360.0,
        ...     "lookat": {
        ...         "eye": [0, 2, -5],
        ...         "target": [0, 0, 0],
        ...         "up": [0, 1, 0]
        ...     }
        ... }
        >>> W, H, tfovx, tfovy, view, proj, campos = make_matrices_from_yaml(config)
    """
    # Extract required parameters
    width = int(camera_cfg.get("width", 1280))
    height = int(camera_cfg.get("height", 720))
    
    # Intrinsics (default to reasonable values if not provided)
    fx = float(camera_cfg.get("fx", width * 0.9))
    fy = float(camera_cfg.get("fy", height * 0.9))
    cx = float(camera_cfg.get("cx", width / 2.0))
    cy = float(camera_cfg.get("cy", height / 2.0))
    
    # Clipping planes
    znear = float(camera_cfg.get("znear", 0.01))
    zfar = float(camera_cfg.get("zfar", 100.0))
    
    # Determine camera pose
    lookat_cfg = camera_cfg.get("lookat")
    
    if lookat_cfg is not None:
        # Build pose from look-at parameters
        eye = lookat_cfg.get("eye", [0.0, 0.0, -5.0])
        target = lookat_cfg.get("target", [0.0, 0.0, 0.0])
        up = lookat_cfg.get("up", [0.0, 1.0, 0.0])
        
        c2w = build_lookat_camera_pose(
            np.array(eye, dtype=np.float32),
            np.array(target, dtype=np.float32),
            np.array(up, dtype=np.float32)
        )
    else:
        # Use explicit c2w matrix or identity
        c2w = camera_cfg.get("c2w", np.eye(4, dtype=np.float32))
        c2w = ensure_4x4_matrix(c2w)
    
    # Compute derived matrices
    w2c = invert_transform(c2w)  # World-to-camera
    tanfovx, tanfovy = compute_tan_half_fov(fx, fy, width, height)
    
    # Build projection matrix
    proj = build_gl_projection_matrix(fx, fy, cx, cy, width, height, znear, zfar)
    
    # Full transform: world -> clip space
    full_transform = proj @ w2c
    
    # Transpose for rasterizer API (expects column-major)
    view_matrix_transposed = w2c.T.copy()
    proj_matrix_transposed = full_transform.T.copy()
    
    # Extract camera position (translation component of c2w)
    camera_position = c2w[:3, 3].astype(np.float32)
    
    return (
        width,
        height,
        tanfovx,
        tanfovy,
        view_matrix_transposed.astype(np.float32),
        proj_matrix_transposed.astype(np.float32),
        camera_position
    )