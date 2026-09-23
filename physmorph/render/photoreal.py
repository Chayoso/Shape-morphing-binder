"""Photo-realistic 3DGS rendering via diff_gauss (graphdeco rasterizer).

Renders the MPM-deformed Gaussians — center x, covariance Σ=σ0²FFᵀ (so the splat
anisotropy RIDES the deformation F, the SH-under-F centerpiece), per-Gaussian color,
opacity — into an RGB image from a look-at camera. Used for the flagship "texture
rides F" figures. Requires a CUDA build of diff_gauss (present in diffmpm_v2.3.0).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .covariance import cov_from_F, decompose_cov, sigma0_from_nn


def _world2view(cam_pos, target=(0, 0, 0), up=(0, 1, 0)):
    """World->view matrix, graphdeco/3DGS convention (camera looks down +z)."""
    cam_pos = np.asarray(cam_pos, np.float64); target = np.asarray(target, np.float64)
    fwd = target - cam_pos; fwd /= np.linalg.norm(fwd) + 1e-9        # +z into scene
    right = np.cross(np.asarray(up, np.float64), fwd); right /= np.linalg.norm(right) + 1e-9
    up2 = np.cross(fwd, right)
    R = np.stack([right, up2, fwd], 0)                               # world->cam (rows = view axes)
    W = np.eye(4)
    W[:3, :3] = R
    W[:3, 3] = -R @ cam_pos
    return W.astype(np.float32)


def _proj_graphdeco(fovx, fovy, znear=0.01, zfar=100.0):
    tx = math.tan(fovx / 2); ty = math.tan(fovy / 2)
    top = ty * znear; bottom = -top; right = tx * znear; left = -right
    P = np.zeros((4, 4), np.float32)
    P[0, 0] = 2 * znear / (right - left)
    P[1, 1] = 2 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P.astype(np.float32)


def _graphdeco_rasterizer():
    """Return (Settings, Rasterizer) from whichever graphdeco-style build is importable, else None."""
    try:
        from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
        return GaussianRasterizationSettings, GaussianRasterizer
    except Exception:
        pass
    try:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        return GaussianRasterizationSettings, GaussianRasterizer
    except Exception:
        return None


def _have_gsplat():
    try:
        import gsplat  # noqa: F401
        return True
    except Exception:
        return False


def render_3dgs(x, color, F=None, sigma0=None, opacity=0.92, cov=None,
                azimuth=0.6, elevation=0.3, dist=None, res=800, fovy_deg=45.0,
                bg=(1, 1, 1), center=None, device="cuda"):
    """Render deformed Gaussians to an (H,W,3) float image in [0,1].

    x: (N,3) centers. color: (N,3) linear RGB. F: (N,3,3) deformation (Σ=σ0²FFᵀ);
    or pass `cov` (N,3,3) directly. Camera orbits the centroid at (azimuth,elevation).

    Backend: graphdeco diff_gauss / diff_gaussian_rasterization if available (cluster), else gsplat
    (local). Both consume the same (means, quats wxyz, scales, opacity, linear-RGB) and a +z-forward
    look-at camera, so the call sites are backend-agnostic.
    """
    x = np.ascontiguousarray(x, np.float32); N = len(x)
    ctr = x.mean(0) if center is None else np.asarray(center, np.float32)
    rad = float(np.linalg.norm(x - ctr, axis=1).max())
    if dist is None:
        dist = rad * 3.0
    if sigma0 is None:
        sigma0 = sigma0_from_nn(x, 0.7)
    if cov is None:
        F = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1)) if F is None else F
        cov = cov_from_F(F, sigma0)
    scales, quats = decompose_cov(cov)               # (N,3),(N,4) wxyz

    cam = ctr + dist * np.array([math.cos(elevation) * math.sin(azimuth),
                                 math.sin(elevation),
                                 math.cos(elevation) * math.cos(azimuth)], np.float32)
    W = _world2view(cam, ctr)                         # world->view (+z forward)
    fovy = math.radians(fovy_deg); fovx = fovy        # square image
    op_arr = (np.full(N, float(opacity), np.float32) if np.isscalar(opacity)
              else np.ascontiguousarray(opacity, np.float32).reshape(N))

    gd = _graphdeco_rasterizer()
    if gd is not None:
        Settings, Rasterizer = gd
        P = _proj_graphdeco(fovx, fovy)
        settings = Settings(
            image_height=res, image_width=res, tanfovx=math.tan(fovx / 2), tanfovy=math.tan(fovy / 2),
            bg=torch.tensor(bg, dtype=torch.float32, device=device), scale_modifier=1.0,
            viewmatrix=torch.tensor(W.T, device=device), projmatrix=torch.tensor((P @ W).T, device=device),
            sh_degree=0, campos=torch.tensor(cam, device=device), prefiltered=False, debug=False)
        raster = Rasterizer(settings)
        t = lambda a: torch.tensor(np.ascontiguousarray(a, np.float32), device=device)
        means3D = t(x); means2D = torch.zeros_like(means3D, requires_grad=False)
        out = raster(means3D, means2D, t(op_arr).reshape(N, 1), shs=None, colors_precomp=t(color),
                     scales=t(scales), rotations=t(quats))
        img = out[0] if isinstance(out, (tuple, list)) else out
        img = img.clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
        return img[::-1].copy()                      # graphdeco expects a y-down (COLMAP) camera; our
        # _world2view is y-up, so rows come out flipped — verified by a +y-red/−y-blue probe.

    if _have_gsplat():
        import gsplat
        fy = (res / 2) / math.tan(fovy / 2); fx = (res / 2) / math.tan(fovx / 2)
        K = torch.tensor([[[fx, 0, res / 2], [0, fy, res / 2], [0, 0, 1]]], dtype=torch.float32, device=device)
        viewmat = torch.tensor(W, dtype=torch.float32, device=device)[None]   # (1,4,4) world->cam, +z fwd
        tt = lambda a: torch.tensor(np.ascontiguousarray(a, np.float32), device=device)
        rc, _, _ = gsplat.rasterization(
            tt(x), tt(quats), tt(scales), tt(op_arr), tt(color), viewmat, K, res, res,
            near_plane=0.01, far_plane=1e3, render_mode="RGB",
            backgrounds=torch.tensor([bg], dtype=torch.float32, device=device))
        return rc[0].clamp(0, 1).detach().cpu().numpy()[::-1].copy()   # same y-up/y-down fix as above

    raise RuntimeError("no 3DGS rasteriser available (diff_gauss / diff_gaussian_rasterization / gsplat)")
