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
target; white background.  When F is supplied, the rasterizer receives the viewer's
covariance Sigma=sigma0^2 F F^T and gradients flow through that covariance to F.
F=None keeps the original isotropic positions-only path for compatibility."""
from __future__ import annotations

import math

import numpy as np
import torch

from ..render.children import expand_children_torch, tangent_child_offsets


def gaussian_covariance(F: torch.Tensor, sigma0: float, jitter: float = 1e-8
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(Sigma, Sigma6)`` for the differentiable precomputed-covariance path.

    ``Sigma6`` follows graphdeco's upper-triangle order
    ``(xx, xy, xz, yy, yz, zz)``.  No decomposition or detach is used, so a
    rasterizer gradient with respect to its covariance input reaches ``F``.
    """
    Fm = F.reshape(-1, 3, 3)
    cov = (float(sigma0) ** 2) * (Fm @ Fm.transpose(1, 2))
    if jitter > 0:
        cov = cov + float(jitter) * torch.eye(3, dtype=cov.dtype,
                                               device=cov.device).unsqueeze(0)
    cov6 = torch.stack((cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
                        cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]), dim=1)
    return cov, cov6.contiguous()


def gaussian_shape_diagnostics(F: torch.Tensor, sigma0: float,
                               reference_spacing: float | None = None,
                               radius_sigma: float = 3.0) -> dict[str, float]:
    """Raw-state diagnostics for oversized or ill-conditioned Gaussian ellipsoids.

    The values use the singular values of F, hence do not consume the renderer.
    ``radius_*`` is the selected sigma support radius along the longest axis.
    """
    with torch.no_grad():
        sv = torch.linalg.svdvals(torch.as_tensor(F).reshape(-1, 3, 3)).abs()
        scales = float(sigma0) * sv
        major = scales.max(dim=1).values
        minor = scales.min(dim=1).values
        cond = major / minor.clamp_min(torch.finfo(scales.dtype).eps)
        radius = float(radius_sigma) * major

        def q95(v):
            return float(torch.quantile(v.float(), 0.95))

        out = {
            "gauss_scale_min": float(minor.min()),
            "gauss_scale_max": float(major.max()),
            "gauss_scale_p95": q95(major),
            "gauss_condition_p95": q95(cond),
            "gauss_condition_max": float(cond.max()),
            "gauss_radius_p95": q95(radius),
            "gauss_radius_max": float(radius.max()),
        }
        if reference_spacing is not None:
            spacing = max(float(reference_spacing), 1e-12)
            out["gauss_radius_over_spacing_p95"] = q95(radius / spacing)
            out["gauss_radius_over_spacing_max"] = float((radius / spacing).max())
        return out


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
    """Per-run cameras, a rest Gaussian scale, and baked target images."""

    def __init__(self, views, extent: float, sigma0: float, res: int, dev,
                 child_count: int = 1, child_sigma_scale: float = 0.55,
                 child_offset_scale: float = 0.35, child_k: int = 16):
        self.res = res
        self.dev = dev
        radius = extent * 2.6
        fov = 0.7
        self.cams = [_camera(th, phi, radius, res, fov, dev) for th, phi in views]
        self.sigma0 = float(sigma0)          # isotropic: scales+identity-quaternion
        self.child_count = int(child_count)
        if self.child_count < 1 or self.child_count > 4:
            raise ValueError("child_count must be in [1,4]")
        self.child_sigma_scale = float(child_sigma_scale)
        self.child_offset_scale = float(child_offset_scale)
        self.child_k = int(child_k)
        if not 0 < self.child_sigma_scale <= 1:
            raise ValueError("child_sigma_scale must be in (0,1]")
        self.source_offsets = None
        self.target_offsets = None
        # parametrization (works across forks; the local fork rejects scales=None
        # even on the cov3Ds path)
        self.targets = None                       # filled by bake_targets

    @property
    def primitive_sigma(self) -> float:
        return self.sigma0 * (self.child_sigma_scale if self.child_count > 1 else 1.0)

    def configure_source(self, rest_x: np.ndarray, mask: np.ndarray | None = None):
        """Freeze source-material offsets; no child state is added to MPM."""
        offsets = tangent_child_offsets(rest_x, mask, self.sigma0, self.child_count,
                                        self.child_offset_scale, self.child_k)
        self.source_offsets = torch.as_tensor(offsets, device=self.dev)

    def _render(self, x: torch.Tensor, cam, F: torch.Tensor | None = None,
                mask: torch.Tensor | None = None,
                offsets: torch.Tensor | None = None):
        mod, has_norm = _gs()
        render_sigma = self.sigma0
        if offsets is not None:
            x, F = expand_children_torch(x, F, offsets, mask)
            render_sigma *= self.child_sigma_scale
        elif mask is not None:
            mask = mask.to(device=x.device, dtype=torch.bool)
            x = x[mask]
            if F is not None:
                F = F[mask]
        n = len(x)
        rast = mod.GaussianRasterizer(raster_settings=cam)
        kw = dict(means3D=x, means2D=torch.zeros_like(x),
                  opacities=torch.full((n, 1), 0.9, dtype=x.dtype, device=x.device),
                  colors_precomp=torch.full((n, 3), 0.35, dtype=x.dtype,
                                             device=x.device))
        if F is None:
            # Legacy positions-only path. Graphdeco requires exactly one of
            # (scales, rotations) and a precomputed covariance.
            scales = torch.full((n, 3), render_sigma, dtype=x.dtype, device=x.device)
            rots = torch.zeros(n, 4, dtype=x.dtype, device=x.device)
            rots[:, 0] = 1.0
            kw.update(scales=scales, rotations=rots)
        else:
            _, cov6 = gaussian_covariance(F.to(dtype=x.dtype, device=x.device),
                                          render_sigma)
            if has_norm:
                # hyde06 diff_gauss fork: plural cov3Ds and an explicit normal input.
                # Precomputed colors do not consume normals, but the fork requires the
                # correctly-shaped slot on this path.
                kw.update(cov3Ds_precomp=cov6, norm3Ds_precomp=torch.zeros_like(x))
            else:
                # Original graphdeco diff_gaussian_rasterization API.
                kw.update(cov3D_precomp=cov6)
        out = rast(**kw)
        return out[0] if isinstance(out, tuple) else out

    def bake_targets(self, tgt_x: torch.Tensor, F: torch.Tensor | None = None,
                     mask: torch.Tensor | None = None):
        if self.child_count > 1:
            mask_np = (None if mask is None else mask.detach().cpu().numpy().astype(bool))
            offsets = tangent_child_offsets(tgt_x.detach().cpu().numpy(), mask_np,
                                            self.sigma0, self.child_count,
                                            self.child_offset_scale, self.child_k)
            self.target_offsets = torch.as_tensor(offsets, device=tgt_x.device)
        with torch.no_grad():
            self.targets = [self._render(tgt_x, c, F, mask, self.target_offsets).detach()
                            for c in self.cams]

    def loss(self, x: torch.Tensor, F: torch.Tensor | None = None,
             mask: torch.Tensor | None = None) -> torch.Tensor:
        """Mean multi-view L1 against the baked target renders (robust; L2 washes the
        sparse-floater signal back out)."""
        if self.child_count > 1 and self.source_offsets is None:
            raise RuntimeError("configure_source must be called before child-render loss")
        L = x.new_zeros(())
        for cam, timg in zip(self.cams, self.targets):
            L = L + (self._render(x.contiguous(), cam, F, mask,
                                  self.source_offsets) - timg).abs().mean()
        return L / len(self.cams)
