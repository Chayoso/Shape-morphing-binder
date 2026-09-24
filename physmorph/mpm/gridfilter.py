"""The null-space projection at the window commit (docs/method.md 10.20; config.commit_pic).

The MPM grid represents a particle field only through its transfer: P2G scatters with the cubic
B-spline weights w_gp, G2P gathers them back. The composition P = G2P ∘ P2G is a projection onto
the grid-representable subspace of particle fields, and I − P is the grid's null space — the
sub-cell modes the dynamics cannot act on and the cell-sum objective cannot see, which is where
the measured sub-cell disorder and the tail's breathing live (docs/experiments.md 2026-09-23
night; the audit). Gritton and Berzins (Comput. Particle Mech. 2017) remove that null space per
cell by an SVD of the P2G operator; XPIC(m) (Hammerquist and Nairn, CMAME 2017) removes it by
alternating transfers, exactly as m → ∞. Here it is applied ONCE per window, to the window's
displacement:

    x_end  <-  x_start + P(x_end - x_start),   P(d)_p = Σ_g w_gp (Σ_q w_gq m_q d_q) / (Σ_q w_gq m_q)

with the simulation's own stencil (base node floor(x/dx) - 1, 4³ cubic B-spline weights) evaluated
at the window-START positions, mass-weighted. No constant. Positions only; velocities, C and F are
untouched (the projected displacement is a fraction of a spacing).
"""
from __future__ import annotations

import numpy as np


def _cubic_bspline(t):
    """The cubic B-spline N(t) on |t| < 2 (t in cells)."""
    import torch
    a = t.abs()
    return torch.where(a < 1.0, 0.5 * a ** 3 - a ** 2 + 2.0 / 3.0,
                       torch.where(a < 2.0, (2.0 - a) ** 3 / 6.0, torch.zeros_like(a)))


def grid_project(d: np.ndarray, x: np.ndarray, dx: float, grid_min, dims, m: np.ndarray | None = None,
                 device: str = "cuda", order: int = 5):
    """XPIC(order) of the field d (N,3) at particle positions x (N,3) on the MPM grid (cell dx,
    origin grid_min, dims (nx, ny, nz)) with the cubic B-spline weights: with P = G2P ∘ P2G (one
    mass-weighted transfer), the filter is I − (I − P)^order — order 1 is the plain PIC transfer
    (a smoothing: on a random cloud P is not an exact projection, a linear field comes back with
    a first-order sampling error), and the order → ∞ limit is the exact projection onto the
    grid-representable subspace (Hammerquist & Nairn 2017). Returns (filtered d (N,3) float32,
    stats): stats = the RMS of the removed part over the RMS of d (the null-space share) and the
    medians in wu."""
    import torch
    x_t = torch.as_tensor(np.ascontiguousarray(x, np.float32), device=device)
    d_t = torch.as_tensor(np.ascontiguousarray(d, np.float32), device=device)
    N = x_t.shape[0]
    m_t = torch.ones(N, device=device) if m is None else torch.as_tensor(np.asarray(m, np.float32), device=device)
    if m_t.dim() == 0 or m_t.numel() == 1:
        m_t = torch.ones(N, device=device) * float(m_t.reshape(-1)[0])
    gmin = torch.as_tensor(np.asarray(grid_min, np.float32), device=device)
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    inv_dx = 1.0 / float(dx)
    p = (x_t - gmin) * inv_dx                                   # in cells
    base = torch.floor(p).long() - 1                            # the kernel's base node
    G = nx * ny * nz
    with torch.no_grad():
        # the stencil: 4^3 nodes, weights and ids computed once (positions are fixed for the filter)
        ws, gids = [], []
        for oi in range(4):
            for oj in range(4):
                for ok in range(4):
                    node = base + torch.tensor([oi, oj, ok], device=device)
                    t = node.float() - p                            # node - particle, in cells
                    w = _cubic_bspline(t[:, 0]) * _cubic_bspline(t[:, 1]) * _cubic_bspline(t[:, 2])
                    ok_ = ((node[:, 0] >= 0) & (node[:, 0] < nx) & (node[:, 1] >= 0) & (node[:, 1] < ny)
                           & (node[:, 2] >= 0) & (node[:, 2] < nz))
                    ws.append(w * ok_)
                    gids.append(((node[:, 0] * ny + node[:, 1]) * nz + node[:, 2]).clamp(0, G - 1))
        wsum = torch.stack(ws, 0).sum(0).clamp_min(1e-12)
        grid_m = torch.zeros(G, device=device)
        for w, g in zip(ws, gids):
            grid_m.index_add_(0, g, w * m_t)
        grid_m = grid_m.clamp_min(1e-12)

        def P(f):
            gv = torch.zeros(G, 3, device=device)
            for w, g in zip(ws, gids):
                gv.index_add_(0, g, (w * m_t)[:, None] * f)
            gd = gv / grid_m[:, None]
            o = torch.zeros_like(f)
            for w, g in zip(ws, gids):
                o += w[:, None] * gd[g]
            return o / wsum[:, None]                              # partition of unity away from the walls

        # XPIC(order): d_filtered = d - (I - P)^order d
        r = d_t.clone()
        for _ in range(max(int(order), 1)):
            r = r - P(r)
        out = d_t - r
        rem = r
        stats = {"null_share": float(rem.norm(dim=1).pow(2).mean().sqrt() / max(float(d_t.norm(dim=1).pow(2).mean().sqrt()), 1e-12)),
                 "removed_median": float(rem.norm(dim=1).median()),
                 "d_median": float(d_t.norm(dim=1).median())}
    return out.float().cpu().numpy(), stats
