"""Real 3DGS render loss — the viewer's own forward model as the objective.

Why: every CIC/grid loss is blind to sub-cell arrangement, and the soft-silhouette
saturates on sparse mass — which is exactly why floaters that are OBVIOUS in the
Gaussian-splat viewer never generated gradients (dossier: docs/floaters.md). This
loss rasterizes the body with the SAME differentiable Gaussian rasterizer family the
viewer/deliverables use (diff_gauss, cov3Ds_precomp path), compares against
pre-rendered target images per view, and backpropagates through means3D — so a
Gaussian that LOOKS wrong IS wrong to the optimizer. Slots into the λ-balanced
render channel in place of the CIC soft-silhouette (cfg.use_gauss_loss).

Cameras: the same azimuth×elevation rings as make_views, at a radius framing the
target; white background; per-particle isotropic covariance sigma0 (positions-only
gradients in v1 — F-coupled covariance is the pre-registered v2)."""
from __future__ import annotations

import math

import numpy as np
import torch


def _gs():
    """Dual-API import: the hyde06 fork (diff_gauss, wants norm3Ds_precomp) or the
    INRIA original (diff_gaussian_rasterization) — whichever this machine has."""
    try:
        import diff_gauss as m
        return m, True
    except ImportError:
        import diff_gaussian_rasterization as m
        return m, False


def _camera(az: float, el: float, radius: float, res: int, fov: float, dev):
    GaussianRasterizationSettings = _gs()[0].GaussianRasterizationSettings
    campos = radius * np.array([math.sin(az) * math.cos(el), math.sin(el),
                                math.cos(az) * math.cos(el)], np.float32)
    fwd = -campos / np.linalg.norm(campos)
    right = np.cross(fwd, [0.0, 1.0, 0.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    A = np.stack([right, -up, fwd], 0).astype(np.float32)     # COLMAP y-down
    W = np.eye(4, dtype=np.float32)
    W[:3, :3] = A
    W[:3, 3] = -A @ campos
    tx = math.tan(fov / 2)
    P = np.zeros((4, 4), np.float32)
    P[0, 0] = 1 / tx; P[1, 1] = 1 / tx
    P[2, 2] = 100.0 / (100.0 - 0.01); P[2, 3] = -100.0 * 0.01 / (100.0 - 0.01)
    P[3, 2] = 1.0
    return GaussianRasterizationSettings(
        image_height=res, image_width=res, tanfovx=tx, tanfovy=tx,
        bg=torch.ones(3, device=dev), scale_modifier=1.0,
        viewmatrix=torch.tensor(W.T, device=dev),
        projmatrix=torch.tensor((P @ W).T, device=dev), sh_degree=0,
        campos=torch.tensor(campos, device=dev), prefiltered=False, debug=False)


class GaussViews:
    """Per-run bundle: cameras + fixed per-particle covariance + target images."""

    def __init__(self, views, extent: float, sigma0: float, res: int, dev):
        self.res = res
        self.dev = dev
        radius = extent * 2.6
        fov = 0.7
        self.cams = [_camera(th, phi, radius, res, fov, dev) for th, phi in views]
        s2 = sigma0 * sigma0
        self.cov6_1 = torch.tensor([s2, 0, 0, s2, 0, s2], device=dev)
        self.targets = None                       # filled by bake_targets

    def _render(self, x: torch.Tensor, cam):
        mod, has_norm = _gs()
        n = len(x)
        rast = mod.GaussianRasterizer(raster_settings=cam)
        kw = dict(means3D=x, means2D=torch.zeros_like(x),
                  opacities=torch.full((n, 1), 0.9, device=self.dev),
                  colors_precomp=torch.full((n, 3), 0.35, device=self.dev))
        if has_norm:
            out = rast(**kw, cov3Ds_precomp=self.cov6_1.expand(n, 6).contiguous(),
                       norm3Ds_precomp=torch.zeros(n, 3, device=self.dev))
        else:
            out = rast(**kw, cov3D_precomp=self.cov6_1.expand(n, 6).contiguous(),
                       shs=None, scales=None, rotations=None)
        return out[0] if isinstance(out, tuple) else out

    def bake_targets(self, tgt_x: torch.Tensor):
        with torch.no_grad():
            self.targets = [self._render(tgt_x, c).detach() for c in self.cams]

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Mean multi-view L1 against the baked target renders (robust; L2 washes the
        sparse-floater signal back out)."""
        L = x.new_zeros(())
        for cam, timg in zip(self.cams, self.targets):
            L = L + (self._render(x.contiguous(), cam) - timg).abs().mean()
        return L / len(self.cams)
