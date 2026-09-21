"""Lightweight differentiable multi-view silhouette loss D_img (§3.5, eq 14).

Orthographic ring cameras + soft 2D CIC splat -> alpha image. Self-contained
(no diff_gauss); gives projection-consistency supervision. The honest weak
render signal of the diagnosis; pairs with D_vol in the single graph.
"""
from __future__ import annotations

import numpy as np
import torch


def ring_thetas(V: int) -> np.ndarray:
    return np.linspace(0, 2 * np.pi, V, endpoint=False).astype(np.float32)


def _project(x: torch.Tensor, theta: float, phi: float = 0.0) -> torch.Tensor:
    """Orthographic projection onto camera at azimuth theta, elevation phi (up ~ +y).

    View direction d = (cosφ sinθ, sinφ, cosφ cosθ); right = (cosθ, 0, −sinθ);
    up = d × right = (−sinφ sinθ, cosφ, −sinφ cosθ). phi=0 reproduces the v1 ring."""
    right = x.new_tensor([np.cos(theta), 0.0, -np.sin(theta)])
    up = x.new_tensor([-np.sin(phi) * np.sin(theta), np.cos(phi), -np.sin(phi) * np.cos(theta)])
    return torch.stack([x @ right, x @ up], dim=1)


_KERNEL = {"name": "cic"}


def set_kernel(name: str):
    """Splat kernel of every rasteriser in this module and render_loss: 'cic' (bilinear, 2x2
    footprint) or 'quad' (quadratic B-spline, 3x3, the lowest order whose DERIVATIVE is
    continuous — under CIC a particle's gradient is the derivative of its own footprint and
    flips sign across pixel edges, so neighbours at different sub-pixel offsets pull in
    different directions: half of the render covector's energy at two spacings was this and
    the target images' shot noise, docs/surface_gradient.md §7). Set once per run (runner)."""
    if name not in ("cic", "quad"):
        raise ValueError(f"unknown splat kernel {name!r}")
    _KERNEL["name"] = name


def splat_terms(rel: torch.Tensor):
    """Per-particle splat footprint for pixel-lattice coordinates rel (..., 2): a list of
    (ii, jj, w) with integer node indices and partition-of-unity weights (sum over the list = 1)."""
    if _KERNEL["name"] == "cic":
        base = torch.floor(rel).long()
        frac = rel - base.to(rel.dtype)
        out = []
        for ox in (0, 1):
            wx = frac[..., 0] if ox else 1 - frac[..., 0]
            for oy in (0, 1):
                wy = frac[..., 1] if oy else 1 - frac[..., 1]
                out.append((base[..., 0] + ox, base[..., 1] + oy, wx * wy))
        return out
    c = torch.round(rel).long()
    d = rel - c.to(rel.dtype)                                   # in [-0.5, 0.5]
    w1 = [0.5 * (0.5 - d[..., 0]) ** 2, 0.75 - d[..., 0] ** 2, 0.5 * (0.5 + d[..., 0]) ** 2]
    w2 = [0.5 * (0.5 - d[..., 1]) ** 2, 0.75 - d[..., 1] ** 2, 0.5 * (0.5 + d[..., 1]) ** 2]
    out = []
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            out.append((c[..., 0] + ox, c[..., 1] + oy, w1[ox + 1] * w2[oy + 1]))
    return out


def soft_silhouette(x: torch.Tensor, theta: float, res: int, extent: float,
                    k: float = 1.5, phi: float = 0.0) -> torch.Tensor:
    """Differentiable alpha image (res,res) via a 2D coverage splat (set_kernel)."""
    p = _project(x, theta, phi)
    rel = (p + extent) / (2 * extent) * res
    img = x.new_zeros(res * res)
    for ii, jj, w in splat_terms(rel):
        valid = (ii >= 0) & (ii < res) & (jj >= 0) & (jj < res)
        idx = (ii * res + jj).clamp(0, res * res - 1)
        img = img.index_add(0, idx, torch.where(valid, w, torch.zeros_like(w)))
    return (1.0 - torch.exp(-k * img)).reshape(res, res)


def target_silhouettes(target_x: torch.Tensor, thetas, res: int, extent: float, k=1.5):
    with torch.no_grad():
        return [soft_silhouette(target_x, float(t), res, extent, k).detach() for t in thetas]


def d_img(x: torch.Tensor, target_alphas, thetas, res: int, extent: float, k=1.5) -> torch.Tensor:
    """Mean multi-view silhouette MSE — eq (14) (DT/IoU omitted for compactness)."""
    loss = x.new_zeros(())
    for a_t, th in zip(target_alphas, thetas):
        a = soft_silhouette(x, float(th), res, extent, k)
        loss = loss + ((a - a_t) ** 2).mean()
    return loss / len(thetas)


def _view_basis(x: torch.Tensor, views):
    """(V,3) right and up vectors for a list of (theta, phi) views."""
    th = x.new_tensor([float(t) for t, _ in views])
    ph = x.new_tensor([float(p) for _, p in views])
    right = torch.stack([torch.cos(th), torch.zeros_like(th), -torch.sin(th)], 1)
    up = torch.stack([-torch.sin(ph) * torch.sin(th), torch.cos(ph), -torch.sin(ph) * torch.cos(th)], 1)
    return right, up


def soft_silhouette_multi(x: torch.Tensor, views, res: int, extent: float,
                          k: float = 1.5) -> torch.Tensor:
    """All views at once: (V,res,res) alpha images, one index_add per CIC corner."""
    right, up = _view_basis(x, views)
    V, N = right.shape[0], x.shape[0]
    p = torch.stack([x @ right.T, x @ up.T], -1)                 # (N,V,2)
    rel = (p + extent) / (2 * extent) * res
    voff = (torch.arange(V, device=x.device) * (res * res)).view(1, V)
    img = x.new_zeros(V * res * res)
    for ii, jj, w in splat_terms(rel):
        valid = (ii >= 0) & (ii < res) & (jj >= 0) & (jj < res)
        idx = (voff + ii * res + jj).clamp(0, V * res * res - 1)
        img = img.index_add(0, idx.reshape(-1), torch.where(valid, w, torch.zeros_like(w)).reshape(-1))
    return (1.0 - torch.exp(-k * img)).reshape(V, res, res)
