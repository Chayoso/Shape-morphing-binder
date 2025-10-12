# renderer/camera_utils.py
# Build view/projection matrices from OpenCV intrinsics (fx,fy,cx,cy) and c2w.
# Output matches what diff-gaussian-rasterization expects:
#  - tanfovx/y scalars
#  - viewmatrix = world->view (w2c)   (TRANSPOSED for the API)
#  - projmatrix = (P @ w2c)           (TRANSPOSED for the API)
# Conventions:
#  * We keep an OpenCV-like camera (+Z forward) and build a GL-style P that
#    produces clip.w = z_cam > 0 for visible points.
from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np

def _ensure_4x4(m) -> np.ndarray:
    M = np.asarray(m, dtype=np.float32)
    if M.shape == (16,):
        M = M.reshape(4, 4)
    if M.shape != (4, 4):
        raise ValueError("Expected 4x4 matrix or flat length-16 list for c2w (camera-to-world).")
    return M

def _inv4x4(m: np.ndarray) -> np.ndarray:
    return np.linalg.inv(m).astype(np.float32)

def _tanfov_from_fx_fy(fx: float, fy: float, W: int, H: int) -> Tuple[float, float]:
    # pinhole: tan(FOVx/2) = W/(2*fx), tan(FOVy/2) = H/(2*fy)
    return (float(W) / (2.0 * float(fx))), (float(H) / (2.0 * float(fy)))

def _gl_perspective_from_intrinsics(fx: float, fy: float, cx: float, cy: float,
                                    W: int, H: int, znear: float, zfar: float) -> np.ndarray:
    """OpenGL-style perspective matrix from pixel intrinsics.
    Maps camera space -> homogeneous clip. NDC is obtained by division.
    Row-major result; we will pass its TRANSPOSE to the rasterizer.
    The form below is the well-known intrinsics->GL mapping where clip.w = z_cam.
    """
    P = np.zeros((4, 4), dtype=np.float32)
    # Scale to NDC [-1,1]
    P[0, 0] = 2.0 * fx / float(W)
    P[1, 1] = 2.0 * fy / float(H)
    P[0, 2] = 1.0 - 2.0 * cx / float(W)
    P[1, 2] = 2.0 * cy / float(H) - 1.0
    # Depth (OpenGL-like but with +Z forward camera so w=z_cam>0)
    P[2, 2] = (zfar) / (zfar - znear)
    P[2, 3] = (-znear * zfar) / (zfar - znear)
    P[3, 2] = 1.0
    return P

import numpy as np

def c2w_from_lookat(eye, target, up=(0, 1, 0)):
    """
    Build camera-to-world (column-major) for an OpenCV-style camera:
    +Z = forward, +X = right, +Y = down.
    Columns of the returned matrix are [right, down, fwd, position].

    Args:
        eye    : (3,) camera position in world
        target : (3,) look-at point in world
        up     : (3,) world 'up' hint (not camera up). Default Y-up.

    Returns:
        c2w: (4,4) float32 camera-to-world matrix (column-major)
    """
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    # 1. Forward vector (Z_cam)
    fwd = target - eye
    fwd /= (np.linalg.norm(fwd) + 1e-12)

    # 2. Right vector (X_cam)
    # The world 'up' is a hint to define the orientation of the camera's horizontal plane.
    # The true camera 'up' will be orthogonal to fwd and right.
    right = np.cross(fwd, up)

    # Handle case where fwd and up are parallel
    if np.linalg.norm(right) < 1e-6:
        # If fwd is aligned with world up, choose a different 'up' hint
        # For example, if fwd is (0,1,0), a hint of (0,0,1) will work.
        alt_up = np.array([0, 0, 1], dtype=np.float32)
        if abs(np.dot(fwd, alt_up)) > 0.999: # if fwd is also parallel to Z
            alt_up = np.array([1, 0, 0], dtype=np.float32)
        right = np.cross(fwd, alt_up)

    right /= (np.linalg.norm(right) + 1e-12)

    # 3. Down vector (Y_cam)
    # For OpenCV (+Y=down), the cross product must be fwd x right.
    # This ensures a right-handed system: right(X) x down(Y) = fwd(Z).
    # [FIX] Changed cross product from (right, fwd) to (fwd, right)
    down = np.cross(fwd, right)
    # No need to normalize 'down' as 'fwd' and 'right' are already unit-length and orthogonal.
    
    # 4. Assemble matrix
    # [FIX] Updated docstring to reflect column-major output.
    # The implementation was already creating a column-major matrix.
    R = np.stack([right, down, fwd], axis=1) # The columns are the basis vectors
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = eye
    return c2w

def make_matrices_from_yaml(camera_cfg: Dict[str, Any]) -> Tuple[int, int, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Build (W,H,tanfovx,tanfovy,view,proj,campos) from a YAML-like dict.
    Required keys:
      - width, height           (ints)
      - fx, fy, cx, cy          (floats)  # pixels
    Optional:
      - c2w: list[16] or (4,4) row-major camera-to-world, default identity
      - znear, zfar: floats
    Returns np.float32 arrays. 'view' is world->view, 'proj' is (P @ V)^T per common 3DGS wrappers.
    """
    W = int(camera_cfg.get("width", 1280))
    H = int(camera_cfg.get("height", 720))
    fx = float(camera_cfg.get("fx", W * 0.9))
    fy = float(camera_cfg.get("fy", H * 0.9))
    cx = float(camera_cfg.get("cx", W / 2.0))
    cy = float(camera_cfg.get("cy", H / 2.0))
    znear = float(camera_cfg.get("znear", 0.01))
    zfar  = float(camera_cfg.get("zfar", 100.0))

    # c2w = camera_cfg.get("c2w", np.eye(4, dtype=np.float32))
    # c2w = _ensure_4x4(c2w)
    # w2c = _inv4x4(c2w)
    
    lookat = camera_cfg.get("lookat", None)
    if lookat is not None:
        eye = lookat.get("eye", [0.0, 0.0, -5.0])
        tar = lookat.get("target", [0.0, 0.0, 0.0])
        upv = lookat.get("up", [0.0, 1.0, 0.0])  # world Y-up default
        c2w = c2w_from_lookat(eye, tar, upv)
    else:
        c2w = camera_cfg.get("c2w", np.eye(4, dtype=np.float32))
        c2w = _ensure_4x4(c2w)

    w2c = _inv4x4(c2w)

    # tan(FOV/2) used by rasterizer for tile sizes etc.
    tanfovx, tanfovy = _tanfov_from_fx_fy(fx, fy, W, H)

    # GL-style projection and full transform
    P = _gl_perspective_from_intrinsics(fx, fy, cx, cy, W, H, znear, zfar)
    full = (P @ w2c)  # world -> clip
    view_T = w2c.T.copy()
    proj_T = full.T.copy()

    campos = c2w[:3, 3].astype(np.float32)
    return W, H, tanfovx, tanfovy, view_T.astype(np.float32), proj_T.astype(np.float32), campos