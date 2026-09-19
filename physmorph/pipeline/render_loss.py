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
    """Mean over views of the asymmetric per-pixel silhouette penalty (all views rasterised
    in one pass — 2026-09-16 speed; per-view maths unchanged)."""
    from ..losses.silhouette import soft_silhouette_multi
    a = soft_silhouette_multi(x, views, res, extent, k)                # (V,res,res)
    a_t = torch.stack(list(target_alphas), 0)
    deficit = torch.clamp(a_t - a, min=0.0)         # hole / missing coverage
    excess = torch.clamp(a - a_t, min=0.0)          # spray / ejecta
    return (w_hole * deficit.pow(2) + w_spray * excess.pow(2)).mean(dim=(1, 2)).mean()


def field_normals(x: torch.Tensor, grid_min, dx: float, dims, blur_cells: float = 0.0):
    """Per-particle outward normals + surface weight from the density field.

    Returns (n_hat, sw): n = -∇ρ/|∇ρ| (SDFDiff recipe, differentiable end-to-end) and
    sw = |∇ρ|/max|∇ρ| ∈ [0,1]. Adversarial finding (v4 round 1): normalising the gradient
    REMOVES the surface/interior discriminator on solid clouds (shell 39% of particles
    carried only 53% of the gradient) — so the magnitude is returned and used to WEIGHT
    each particle's shading contribution instead of being silently discarded.
    blur_cells > 0: separable Gaussian blur of the CIC density (in cells) before the
    gradient — on a render-pixel grid this is the renderer's own density (G1,
    docs/surface_gradient.md §4), so the shading normals are those of the drawn surface."""
    nx, ny, nz = dims
    rho = rasterize_mass(x, torch.ones(len(x), device=x.device), grid_min, dx, dims)
    r = rho.reshape(nx, ny, nz)
    if blur_cells > 0:
        import torch.nn.functional as Fn
        rad = int(3 * blur_cells)
        k = torch.exp(-torch.arange(-rad, rad + 1, device=x.device, dtype=r.dtype) ** 2 / (2 * blur_cells ** 2))
        k = k / k.sum()
        r5 = r.reshape(1, 1, nx, ny, nz)
        r5 = Fn.conv3d(r5, k.view(1, 1, 1, 1, -1), padding=(0, 0, rad))
        r5 = Fn.conv3d(r5, k.view(1, 1, 1, -1, 1), padding=(0, rad, 0))
        r5 = Fn.conv3d(r5, k.view(1, 1, -1, 1, 1), padding=(rad, 0, 0))
        r = r5.reshape(nx, ny, nz)
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
    mag = n_p.norm(dim=1, keepdim=True)
    sw = (mag / mag.max().clamp_min(1e-12)).squeeze(1)
    return n_p / mag.clamp_min(1e-6), sw


def shaded_view(x: torch.Tensor, theta: float, phi: float, res: int, extent: float,
                n_p: torch.Tensor, sw: torch.Tensor, k: float = 1.5,
                ambient: float = 0.25, beta: float = 3.0):
    """Headlight-Lambertian, alpha-masked shaded image (res,res).

    b = ambient + (1-ambient)·max(n·l, 0), l = +view direction (TOWARD the camera —
    adversarial round measured the previous sign as exactly backlit, l·d = −1).
    Per-particle splat weight = CIC · surface weight sw · soft front-bias exp(β(z̃−1))
    (an approximate visibility term; without it a depth-mirrored cloud shaded identically).
    The weighted-average shade is then alpha-masked, so the loss compares shading only
    where coverage exists; the coverage signal itself lives in d_render."""
    right = x.new_tensor([np.cos(theta), 0.0, -np.sin(theta)])
    up = x.new_tensor([-np.sin(phi) * np.sin(theta), np.cos(phi),
                       -np.sin(phi) * np.cos(theta)])
    l_dir = torch.linalg.cross(right, up)           # right × up = +d, toward the camera
    b = ambient + (1 - ambient) * (n_p @ l_dir).clamp(min=0)
    z = x @ l_dir                                   # depth toward camera (bigger = nearer)
    zr = (z - z.min()) / (z.max() - z.min()).clamp_min(1e-6)
    vis = torch.exp(beta * (zr - 1.0))              # soft front-bias in (e^-beta, 1]
    pw = (sw + 0.05) * vis                          # surface-weighted, visibility-biased
    p = _project(x, theta, phi)
    rel = (p + extent) / (2 * extent) * res
    base = torch.floor(rel).long()
    frac = rel - base.to(x.dtype)
    num = x.new_zeros(res * res)
    den = x.new_zeros(res * res)
    cov = x.new_zeros(res * res)
    for ox in (0, 1):
        wx = frac[:, 0] if ox else 1 - frac[:, 0]
        for oy in (0, 1):
            wy = frac[:, 1] if oy else 1 - frac[:, 1]
            ii, jj = base[:, 0] + ox, base[:, 1] + oy
            valid = (ii >= 0) & (ii < res) & (jj >= 0) & (jj < res)
            idx = (ii * res + jj).clamp(0, res * res - 1)
            w = torch.where(valid, wx * wy, torch.zeros_like(wx))
            num = num.index_add(0, idx, w * pw * b)
            den = den.index_add(0, idx, w * pw)
            cov = cov.index_add(0, idx, w)
    alpha = 1.0 - torch.exp(-k * cov)
    shade = num / den.clamp_min(1e-6)
    return (shade * alpha).reshape(res, res), alpha.reshape(res, res)


def shaded_views_multi(x: torch.Tensor, views, res: int, extent: float,
                       n_p: torch.Tensor, sw: torch.Tensor, k: float = 1.5,
                       ambient: float = 0.25, beta: float = 3.0):
    """All views of shaded_view at once: (V,res,res) shade*alpha and alpha."""
    from ..losses.silhouette import _view_basis
    right, up = _view_basis(x, views)
    V, N = right.shape[0], x.shape[0]
    l_dir = torch.linalg.cross(right, up)                         # (V,3) toward the camera
    b = ambient + (1 - ambient) * (n_p @ l_dir.T).clamp(min=0)   # (N,V)
    z = x @ l_dir.T                                               # (N,V)
    zr = (z - z.min(0).values) / (z.max(0).values - z.min(0).values).clamp_min(1e-6)
    vis = torch.exp(beta * (zr - 1.0))
    pw = (sw[:, None] + 0.05) * vis                               # (N,V)
    p = torch.stack([x @ right.T, x @ up.T], -1)                  # (N,V,2)
    rel = (p + extent) / (2 * extent) * res
    base = torch.floor(rel).long()
    frac = rel - base.to(x.dtype)
    voff = (torch.arange(V, device=x.device) * (res * res)).view(1, V)
    num = x.new_zeros(V * res * res)
    den = x.new_zeros(V * res * res)
    cov = x.new_zeros(V * res * res)
    for ox in (0, 1):
        wx = frac[..., 0] if ox else 1 - frac[..., 0]
        for oy in (0, 1):
            wy = frac[..., 1] if oy else 1 - frac[..., 1]
            ii, jj = base[..., 0] + ox, base[..., 1] + oy
            valid = (ii >= 0) & (ii < res) & (jj >= 0) & (jj < res)
            idx = (voff + ii * res + jj).clamp(0, V * res * res - 1).reshape(-1)
            w = torch.where(valid, wx * wy, torch.zeros_like(wx))
            num = num.index_add(0, idx, (w * pw * b).reshape(-1))
            den = den.index_add(0, idx, (w * pw).reshape(-1))
            cov = cov.index_add(0, idx, w.reshape(-1))
    alpha = 1.0 - torch.exp(-k * cov)
    shade = num / den.clamp_min(1e-6)
    return (shade * alpha).reshape(V, res, res), alpha.reshape(V, res, res)


def shade_targets(target_x: torch.Tensor, views, res: int, extent: float,
                  grid_min, dx: float, dims, k=1.5, ambient=0.25, normals=None, blur_cells: float = 0.0):
    """Target shaded images. normals=(n_t, sw_t) overrides the density-gradient normals
    with precomputed ones — G1: the normals of the target's reconstructed surface
    (surface_recon.target_surface_normals), so the reference carries no shot noise."""
    with torch.no_grad():
        if normals is not None:
            n_t, sw_t = normals
        else:
            n_t, sw_t = field_normals(target_x, grid_min, dx, dims, blur_cells)
        return [shaded_view(target_x, th, phi, res, extent, n_t, sw_t, k, ambient)[0]
                .detach() for th, phi in views]


def d_pbr(x: torch.Tensor, shade_tgts, views, res: int, extent: float,
          grid_min, dx: float, dims, k: float = 1.5, ambient: float = 0.25,
          blur_cells: float = 0.0) -> torch.Tensor:
    """Mean multi-view shaded-image L2 — orientation/curvature feedback beyond coverage
    (~70% of its gradient direction is orthogonal to the pure-silhouette term, measured).
    Visibility is only the soft front-bias approximation from shaded_view."""
    n_p, sw = field_normals(x, grid_min, dx, dims, blur_cells)
    s, _ = shaded_views_multi(x, views, res, extent, n_p, sw, k, ambient)   # (V,res,res)
    s_t = torch.stack(list(shade_tgts), 0)
    return (s - s_t).pow(2).mean(dim=(1, 2)).mean()


class LambdaBalancer:
    """λ_R = α_λ · ||∇_phys|| / ||∇_render||, EMA-smoothed so the balancer cannot oscillate.

    The C++ get_control_layer_grad_norm rule ("lambda = alpha * phys_norm / render_norm"),
    plus an EMA because the raw ratio is itself a noisy per-iteration quantity and feeding it
    straight back is one of the v1 oscillation sources. α_λ=0 disables the render channel."""

    def __init__(self, alpha_lam: float, ema: float = 0.3, cap: float | None = None,
                 cap_rel: float | None = None):
        self.alpha_lam = float(alpha_lam)
        self.ema = float(ema)
        self.cap = cap
        # RELATIVE cap (density units): fixed at cap_rel x the FIRST raw target, so the
        # divergence guard has the same meaning in any loss unit (the absolute 5e3 was a
        # legacy-unit number that bound at measured raw ratios of 6.5e3-1.1e4)
        self.cap_rel = cap_rel
        self.lam = None
        self.capped = False          # telemetry: the last update hit the cap (a binding
                                     # cap silently under-weights the render channel —
                                     # measured raw ratios of 6.5e3-1.1e4 vs cap 5e3)

    @property
    def active(self) -> bool:
        return self.alpha_lam > 0.0

    def update(self, phys_norm: float, render_norm: float) -> float:
        target = self.alpha_lam * phys_norm / max(render_norm, 1e-12)
        if self.cap_rel is not None and self.cap is None and self.lam is None:
            self.cap = float(self.cap_rel) * max(target, 1e-12)
        # CAP: once D_render saturates its gradient vanishes and the raw ratio diverges —
        # observed live at full scale: λ 1.1e3 → 1.77e5 with a mid-window inversion in tow.
        # A converged render term should FADE, not take over the objective.
        self.capped = self.cap is not None and target > self.cap
        if self.cap is not None:
            target = min(target, self.cap)
        self.lam = target if self.lam is None else (1 - self.ema) * self.lam + self.ema * target
        return self.lam
