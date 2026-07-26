"""Volumetric mass-matching loss D_vol (torch, differentiable) — eq (13).

Cloud-in-cell (trilinear) mass rasterization is differentiable w.r.t. particle
positions, so gradients pull particles toward target-occupied cells.
"""
from __future__ import annotations

import torch


def rasterize_mass(x: torch.Tensor, m: torch.Tensor,
                   grid_min: torch.Tensor, dx: float, dims: tuple[int, int, int]) -> torch.Tensor:
    """Trilinear (CIC) scatter of particle mass onto a flat (nx*ny*nz,) grid."""
    nx, ny, nz = dims
    rel = (x - grid_min) / dx
    base = torch.floor(rel).long()
    frac = rel - base.float()
    grid = x.new_zeros(nx * ny * nz)
    for ox in (0, 1):
        wx = frac[:, 0] if ox else 1.0 - frac[:, 0]
        for oy in (0, 1):
            wy = frac[:, 1] if oy else 1.0 - frac[:, 1]
            for oz in (0, 1):
                wz = frac[:, 2] if oz else 1.0 - frac[:, 2]
                w = wx * wy * wz
                ii, jj, kk = base[:, 0] + ox, base[:, 1] + oy, base[:, 2] + oz
                valid = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny) & (kk >= 0) & (kk < nz)
                idx = ((ii * ny + jj) * nz + kk).clamp(0, nx * ny * nz - 1)
                contrib = torch.where(valid, w * m, torch.zeros_like(w))
                grid = grid.index_add(0, idx, contrib)
    return grid


def target_mass_grid(target_x: torch.Tensor, m: torch.Tensor,
                     grid_min: torch.Tensor, dx: float, dims) -> torch.Tensor:
    """Constant target grid from target particles (detached)."""
    with torch.no_grad():
        return rasterize_mass(target_x, m, grid_min, dx, dims).detach()


def d_vol(x: torch.Tensor, m: torch.Tensor, target_grid: torch.Tensor,
          grid_min: torch.Tensor, dx: float, dims,
          min_mass: float = 0.0, penalty: float = 0.0) -> torch.Tensor:
    """Log-mass-ratio divergence — eq (13)."""
    cur = rasterize_mass(x, m, grid_min, dx, dims)
    diff = torch.log(cur + 1.0) - torch.log(target_grid + 1.0)
    loss = 0.5 * (diff * diff).sum()
    if penalty > 0:
        loss = loss + penalty * torch.clamp(min_mass - cur, min=0.0).pow(2).sum()
    return loss
