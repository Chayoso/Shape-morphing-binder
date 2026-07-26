"""Differentiable density-based surface weight + outlier mask. §3.9, eq (20)(21).

Replaces the legacy marching-cubes mask: GPU, per-frame, differentiable, soft.
Boundary particles (lower local density) get high weight; interior gets low.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from ..losses.volumetric import rasterize_mass


def particle_density(x: torch.Tensor, grid_min, dx: float, dims, m=None) -> torch.Tensor:
    """Normalized local density rho_hat in [0,1] at each particle (eq 20)."""
    N = x.shape[0]
    m = torch.ones(N, device=x.device) if m is None else m
    grid = rasterize_mass(x, m, grid_min, dx, dims)
    nx, ny, nz = dims
    rel = (x - grid_min) / dx
    base = torch.floor(rel).long()
    frac = rel - base.float()
    dens = torch.zeros(N, device=x.device)
    for ox in (0, 1):
        wx = frac[:, 0] if ox else 1 - frac[:, 0]
        for oy in (0, 1):
            wy = frac[:, 1] if oy else 1 - frac[:, 1]
            for oz in (0, 1):
                wz = frac[:, 2] if oz else 1 - frac[:, 2]
                ii, jj, kk = base[:, 0] + ox, base[:, 1] + oy, base[:, 2] + oz
                valid = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny) & (kk >= 0) & (kk < nz)
                idx = ((ii * ny + jj) * nz + kk).clamp(0, nx * ny * nz - 1)
                dens = dens + torch.where(valid, wx * wy * wz * grid[idx], torch.zeros_like(dens))
    return dens / dens.max().clamp_min(1e-8)


def surface_weight(x: torch.Tensor, grid_min, dx: float, dims,
                   beta: float = 8.0, tau: float = 0.5, m=None) -> torch.Tensor:
    """Soft surface membership w = sigmoid(beta (tau - rho_hat)). eq (21)."""
    rho = particle_density(x, grid_min, dx, dims, m=m)
    return torch.sigmoid(beta * (tau - rho))


def outlier_mask(x: np.ndarray, k: int = 8, factor: float = 3.0) -> np.ndarray:
    """Boolean mask of isolated outliers: kNN distance > factor * median."""
    x = np.ascontiguousarray(x, np.float32)
    d, _ = cKDTree(x).query(x, k=k + 1)
    dk = d[:, -1]
    return dk > factor * np.median(dk)
