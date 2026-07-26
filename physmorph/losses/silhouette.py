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


def _project(x: torch.Tensor, theta: float) -> torch.Tensor:
    """Orthographic projection onto camera at azimuth theta (up = +y)."""
    right = x.new_tensor([np.cos(theta), 0.0, -np.sin(theta)])
    up = x.new_tensor([0.0, 1.0, 0.0])
    return torch.stack([x @ right, x @ up], dim=1)


def soft_silhouette(x: torch.Tensor, theta: float, res: int, extent: float,
                    k: float = 1.5) -> torch.Tensor:
    """Differentiable alpha image (res,res) via 2D CIC coverage splat."""
    p = _project(x, theta)
    rel = (p + extent) / (2 * extent) * res
    base = torch.floor(rel).long()
    frac = rel - base.float()
    img = x.new_zeros(res * res)
    for ox in (0, 1):
        wx = frac[:, 0] if ox else 1 - frac[:, 0]
        for oy in (0, 1):
            wy = frac[:, 1] if oy else 1 - frac[:, 1]
            ii, jj = base[:, 0] + ox, base[:, 1] + oy
            valid = (ii >= 0) & (ii < res) & (jj >= 0) & (jj < res)
            idx = (ii * res + jj).clamp(0, res * res - 1)
            img = img.index_add(0, idx, torch.where(valid, wx * wy, torch.zeros_like(wx)))
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
