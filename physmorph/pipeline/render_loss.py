"""D_render — multi-view (azimuth x elevation) asymmetric soft-silhouette loss + λ balancing.

docs/pipeline_v2.md §3.4, §3.6. Builds on losses/silhouette.py CIC splat primitives.

Asymmetry: deficit INSIDE the target silhouette (w_hole) = holes / missing extremities;
excess OUTSIDE (w_spray) = ejecta — the render objective itself pulls strays back, replacing
the v1 clamps. Since relu(a_t-a)^2 + relu(a-a_t)^2 == (a-a_t)^2, w_hole=w_spray=1 recovers
the plain MSE of v1's d_img.
"""
from __future__ import annotations

import numpy as np
import torch

from ..losses.silhouette import _project, soft_silhouette
from ..losses.volumetric import rasterize_mass


def make_views(n_azim: int, elevs=(0.0, 0.5, -0.5)) -> list[tuple[float, float]]:
    """(theta, phi) view set: an azimuth ring per elevation. Each ring gets a DISTINCT
    azimuth offset (i/n_rings of one step) so no two rings stack in azimuth — the previous
    alternating offset re-stacked rings 0 and 2 for odd ring counts."""
    views = []
    n_rings = max(len(tuple(elevs)), 1)
    for i, phi in enumerate(elevs):
        off = (i / n_rings) * (2 * np.pi / max(n_azim, 1))
        for t in np.linspace(0, 2 * np.pi, n_azim, endpoint=False):
            views.append((float(t + off), float(phi)))
    return views


def target_silhouettes(target_x: torch.Tensor, views, res: int, extent: float, k=1.5):
    with torch.no_grad():
        return [soft_silhouette(target_x, th, res, extent, k, phi).detach() for th, phi in views]


def d_render(x: torch.Tensor, target_alphas, views, res: int, extent: float,
             k: float = 1.5, w_hole: float = 2.0, w_spray: float = 1.0) -> torch.Tensor:
    """Mean over views of the asymmetric per-pixel silhouette penalty."""
    loss = x.new_zeros(())
    for a_t, (th, phi) in zip(target_alphas, views):
        a = soft_silhouette(x, th, res, extent, k, phi)
        deficit = torch.clamp(a_t - a, min=0.0)     # hole / missing coverage
        excess = torch.clamp(a - a_t, min=0.0)      # spray / ejecta
        loss = loss + (w_hole * deficit.pow(2) + w_spray * excess.pow(2)).mean()
    return loss / len(views)


def field_normals(x: torch.Tensor, grid_min, dx: float, dims):
    """Per-particle outward surface normals from the density field: n = -∇ρ/|∇ρ|.

    Differentiable end-to-end (CIC rasterise -> central differences -> trilinear gather),
    the SDFDiff recipe applied to the mass field. Interior particles get |∇ρ|≈0 and thus
    near-zero shading gradients — the channel is surface-dominant by construction."""
    nx, ny, nz = dims
    rho = rasterize_mass(x, torch.ones(len(x), device=x.device), grid_min, dx, dims)
    r = rho.reshape(nx, ny, nz)
    g = torch.zeros(nx, ny, nz, 3, device=x.device, dtype=x.dtype)
    g[1:-1, :, :, 0] = (r[2:] - r[:-2])
    g[:, 1:-1, :, 1] = (r[:, 2:] - r[:, :-2])
    g[:, :, 1:-1, 2] = (r[:, :, 2:] - r[:, :, :-2])
    n_grid = -g / (2 * dx)
    # trilinear gather of the normal field at particle positions
    rel = (x - grid_min) / dx
    base = torch.floor(rel).long()
    frac = rel - base.to(x.dtype)
    n_p = torch.zeros_like(x)
    flat = n_grid.reshape(-1, 3)
    for ox in (0, 1):
        wx = frac[:, 0] if ox else 1 - frac[:, 0]
        for oy in (0, 1):
            wy = frac[:, 1] if oy else 1 - frac[:, 1]
            for oz in (0, 1):
                wz = frac[:, 2] if oz else 1 - frac[:, 2]
                ii = (base[:, 0] + ox).clamp(0, nx - 1)
                jj = (base[:, 1] + oy).clamp(0, ny - 1)
                kk = (base[:, 2] + oz).clamp(0, nz - 1)
                w = (wx * wy * wz).unsqueeze(1)
                n_p = n_p + w * flat[(ii * ny + jj) * nz + kk]
    return n_p / n_p.norm(dim=1, keepdim=True).clamp_min(1e-6)


def shaded_view(x: torch.Tensor, theta: float, phi: float, res: int, extent: float,
                n_p: torch.Tensor, k: float = 1.5, ambient: float = 0.25):
    """Headlight-Lambertian shaded image (res,res): per-particle brightness
    b = ambient + (1-ambient)·max(n·l, 0) with l = the view direction, splatted with CIC
    weights and normalised by coverage (a weighted-average image, density-decoupled)."""
    right = x.new_tensor([np.cos(theta), 0.0, -np.sin(theta)])
    up = x.new_tensor([-np.sin(phi) * np.sin(theta), np.cos(phi),
                       -np.sin(phi) * np.cos(theta)])
    l_dir = -torch.linalg.cross(right, up)          # toward the camera (headlight)
    b = ambient + (1 - ambient) * (n_p @ l_dir).clamp(min=0)
    p = _project(x, theta, phi)
    rel = (p + extent) / (2 * extent) * res
    base = torch.floor(rel).long()
    frac = rel - base.to(x.dtype)
    num = x.new_zeros(res * res)
    den = x.new_zeros(res * res)
    for ox in (0, 1):
        wx = frac[:, 0] if ox else 1 - frac[:, 0]
        for oy in (0, 1):
            wy = frac[:, 1] if oy else 1 - frac[:, 1]
            ii, jj = base[:, 0] + ox, base[:, 1] + oy
            valid = (ii >= 0) & (ii < res) & (jj >= 0) & (jj < res)
            idx = (ii * res + jj).clamp(0, res * res - 1)
            w = torch.where(valid, wx * wy, torch.zeros_like(wx))
            num = num.index_add(0, idx, w * b)
            den = den.index_add(0, idx, w)
    alpha = 1.0 - torch.exp(-k * den)
    shade = num / den.clamp_min(1e-6)
    return (shade * alpha).reshape(res, res), alpha.reshape(res, res)


def shade_targets(target_x: torch.Tensor, views, res: int, extent: float,
                  grid_min, dx: float, dims, k=1.5):
    with torch.no_grad():
        n_t = field_normals(target_x, grid_min, dx, dims)
        return [shaded_view(target_x, th, phi, res, extent, n_t, k)[0].detach()
                for th, phi in views]


def d_pbr(x: torch.Tensor, shade_tgts, views, res: int, extent: float,
          grid_min, dx: float, dims, k: float = 1.5) -> torch.Tensor:
    """Mean multi-view shaded-image L2 — curvature-sensitive feedback the silhouette
    cannot see (flat vs domed regions shade differently at equal coverage)."""
    n_p = field_normals(x, grid_min, dx, dims)
    loss = x.new_zeros(())
    for s_t, (th, phi) in zip(shade_tgts, views):
        s, _ = shaded_view(x, th, phi, res, extent, n_p, k)
        loss = loss + (s - s_t).pow(2).mean()
    return loss / len(views)


class LambdaBalancer:
    """λ_R = α_λ · ||∇_phys|| / ||∇_render||, EMA-smoothed so the balancer cannot oscillate.

    The C++ get_control_layer_grad_norm rule ("lambda = alpha * phys_norm / render_norm"),
    plus an EMA because the raw ratio is itself a noisy per-iteration quantity and feeding it
    straight back is one of the v1 oscillation sources. α_λ=0 disables the render channel."""

    def __init__(self, alpha_lam: float, ema: float = 0.3, cap: float | None = None):
        self.alpha_lam = float(alpha_lam)
        self.ema = float(ema)
        self.cap = cap
        self.lam = None

    @property
    def active(self) -> bool:
        return self.alpha_lam > 0.0

    def update(self, phys_norm: float, render_norm: float) -> float:
        target = self.alpha_lam * phys_norm / max(render_norm, 1e-12)
        # CAP: once D_render saturates its gradient vanishes and the raw ratio diverges —
        # observed live at full scale: λ 1.1e3 → 1.77e5 with a mid-window inversion in tow.
        # A converged render term should FADE, not take over the objective.
        if self.cap is not None:
            target = min(target, self.cap)
        self.lam = target if self.lam is None else (1 - self.ema) * self.lam + self.ema * target
        return self.lam
