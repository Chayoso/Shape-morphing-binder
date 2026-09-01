"""Sobolev / grid-GS preconditioning of a per-particle vector field (docs/method.md §6).

The raw render pull is a Jacobi-style signal: every particle sees the same pixel residual
at once, gradients vanish in coverage pockets, and high-frequency components dominate.
Smoothing the field through the GRID — scatter (CIC) → screened-diffusion red-black GS
sweeps → gather — replaces the L² gradient with a Sobolev-metric gradient (the classic
preconditioning of Repulsive Curves / Preconditioned Deformation Grids, PG 2025), which:
  * propagates surface information into zero-gradient pockets (holes, occluded bands),
  * damps the Jacobi overshoot (neighbouring particles stop double-counting a pixel),
  * costs O(iters · grid) torch ops, no extra rollouts.

Used as a SEARCH-DIRECTION transform only: the smoothed field is pulled back to control
space by seeding the existing MPM adjoint (optimizer.py), so the physics stays exact and
nothing here needs to be differentiable.
"""
from __future__ import annotations

import torch


def smooth_particle_field(x, g, grid_min, dx: float, dims, iters: int = 20,
                          kappa: float = 4.0):
    """Screened-diffusion smooth of per-particle vectors g (N,3) at positions x (N,3).

    Solves (I + kappa*L) u = g_hat on the node grid (g_hat = CIC-averaged g) with red-black
    Gauss-Seidel sweeps, then gathers back to the particles. iters=0 returns g unchanged.
    Empty nodes start at zero and are FILLED by the sweeps — that is the propagation."""
    if iters <= 0:
        return g
    nx, ny, nz = dims
    dev = g.device
    rel = (x - grid_min) / dx
    base = torch.floor(rel).long()
    frac = rel - base.float()

    num = torch.zeros(nx * ny * nz, 3, device=dev, dtype=g.dtype)
    den = torch.zeros(nx * ny * nz, device=dev, dtype=g.dtype)
    corners = []
    for ox in (0, 1):
        wx = frac[:, 0] if ox else 1 - frac[:, 0]
        for oy in (0, 1):
            wy = frac[:, 1] if oy else 1 - frac[:, 1]
            for oz in (0, 1):
                wz = frac[:, 2] if oz else 1 - frac[:, 2]
                w = wx * wy * wz
                ii = (base[:, 0] + ox).clamp(0, nx - 1)
                jj = (base[:, 1] + oy).clamp(0, ny - 1)
                kk = (base[:, 2] + oz).clamp(0, nz - 1)
                idx = (ii * ny + jj) * nz + kk
                num.index_add_(0, idx, g * w.unsqueeze(1))
                den.index_add_(0, idx, w)
                corners.append((idx, w))
    u = (num / den.clamp_min(1e-12).unsqueeze(1)).reshape(nx, ny, nz, 3)
    u = torch.where(den.reshape(nx, ny, nz, 1) > 1e-12, u, torch.zeros_like(u))

    # red-black screened-diffusion sweeps: u = (g_hat + kappa * avg6(u)) / (1 + kappa),
    # with g_hat = 0 on empty nodes (pure diffusion there -> in-fill).
    ghat = u.clone()
    ii, jj, kk = torch.meshgrid(torch.arange(nx, device=dev), torch.arange(ny, device=dev),
                                torch.arange(nz, device=dev), indexing="ij")
    red = ((ii + jj + kk) % 2 == 0).unsqueeze(-1)

    def avg6(f):
        s = torch.zeros_like(f)
        s[1:] += f[:-1]; s[:-1] += f[1:]
        s[:, 1:] += f[:, :-1]; s[:, :-1] += f[:, 1:]
        s[:, :, 1:] += f[:, :, :-1]; s[:, :, :-1] += f[:, :, 1:]
        return s / 6.0
    for _ in range(iters):
        for mask in (red, ~red):
            u = torch.where(mask, (ghat + kappa * avg6(u)) / (1.0 + kappa), u)

    uf = u.reshape(-1, 3)
    out = torch.zeros_like(g)
    for idx, w in corners:
        out += uf[idx] * w.unsqueeze(1)

    # preserve the raw field's global magnitude (a preconditioner reshapes, it must not
    # silently rescale the step — the line search calibrates step size on norms)
    nr = g.norm()
    ns = out.norm().clamp_min(1e-30)
    return out * (nr / ns)
