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


def target_dt_grid(target_grid: torch.Tensor, dx: float, dims,
                   clamp: float | None = None) -> torch.Tensor:
    """Unsigned Euclidean distance transform (world units) OUTSIDE target-occupied
    cells, flat zero inside, optionally clamped (far field stays the box leash's job).

    Support = ANY target CIC mass on THIS grid, so thin features always count as
    support — the ear-erosion falsifier guard. The grid must be FINE and target-fitted
    (Opus finding 2: on the coarse loss grid, CIC dilation + the flat trilinear cell
    left a ~1-world-unit dead radius where 72-90% of the production fringe felt zero
    gradient — DT *value* visibility had been conflated with *gradient* visibility);
    the runner builds it at cfg.dt_res on a target-fitted cube so dilation + flat band
    shrink to ~2 fine cells. 3D on purpose: the 2D multi-view variant was falsified by
    forensics (visual hull hides interior concavities)."""
    from scipy.ndimage import distance_transform_edt
    import numpy as np
    nx, ny, nz = dims
    occ = (target_grid > 1e-6).reshape(nx, ny, nz).cpu().numpy()
    assert occ.any(), "empty support: EDT would measure distance to the array border"
    dt = distance_transform_edt(~occ) * dx
    if clamp is not None:
        dt = np.minimum(dt, clamp)
    return torch.as_tensor(dt, dtype=target_grid.dtype,
                           device=target_grid.device).reshape(-1)


def d_w1(x: torch.Tensor, m: torch.Tensor, dt_grid: torch.Tensor,
         grid_min: torch.Tensor, dx: float, dims) -> torch.Tensor:
    """One-sided-W1 cleanup term: SUM_p m_p * DT(x_p), trilinear-sampled.

    SUM, not mean (Opus finding 1: the mean form gave each particle authority w_dt/N —
    measured 300-3270x below the other terms at the shipped weight, an accidental
    no-op, and non-transferable between particle counts). In sum form the per-particle
    pull is exactly w_dt * grad-DT: N-invariant, bounded by ~sqrt(3)*w_dt (trilinear
    EDT interpolant), pointing at target support, INDEPENDENT of local density — the
    complement d_vol cannot supply (its log-ratio gradient fades with the stray's own
    cell mass). Lineage: DRWR flat-inside/linear-outside asymmetry, PhysMorph-GS v1
    L_DT, 3DGS-MCMC L1-opacity (also a sum over primitives), Sinkhorn/W1
    isolated-point gradients (rationale §7)."""
    nx, ny, nz = dims
    rel = (x - grid_min) / dx
    base = torch.floor(rel).long()
    frac = rel - base.float()
    val = x.new_zeros(len(x))
    for ox in (0, 1):
        wx = frac[:, 0] if ox else 1.0 - frac[:, 0]
        for oy in (0, 1):
            wy = frac[:, 1] if oy else 1.0 - frac[:, 1]
            for oz in (0, 1):
                wz = frac[:, 2] if oz else 1.0 - frac[:, 2]
                ii = (base[:, 0] + ox).clamp(0, nx - 1)
                jj = (base[:, 1] + oy).clamp(0, ny - 1)
                kk = (base[:, 2] + oz).clamp(0, nz - 1)
                val = val + wx * wy * wz * dt_grid[(ii * ny + jj) * nz + kk]
    return (m * val).sum()


def gather_cic(field: torch.Tensor, x: torch.Tensor,
               grid_min: torch.Tensor, dx: float, dims) -> torch.Tensor:
    """Trilinear gather of a flat cell field at particle positions (per-particle)."""
    nx, ny, nz = dims
    rel = (x - grid_min) / dx
    base = torch.floor(rel).long()
    frac = rel - base.float()
    val = x.new_zeros(len(x))
    for ox in (0, 1):
        wx = frac[:, 0] if ox else 1.0 - frac[:, 0]
        for oy in (0, 1):
            wy = frac[:, 1] if oy else 1.0 - frac[:, 1]
            for oz in (0, 1):
                wz = frac[:, 2] if oz else 1.0 - frac[:, 2]
                ii = (base[:, 0] + ox).clamp(0, nx - 1)
                jj = (base[:, 1] + oy).clamp(0, ny - 1)
                kk = (base[:, 2] + oz).clamp(0, nz - 1)
                val = val + wx * wy * wz * field[(ii * ny + jj) * nz + kk]
    return val


def w1_budget(x: torch.Tensor, dt_grid: torch.Tensor, grid_min: torch.Tensor,
              dx: float, dims, budget_frac: float) -> float:
    """Scalar transport-budget factor for the W1 term: min(1, budget·N / n_out).

    Third and final gate design. Per-particle gates were falsified twice — grid-CIC
    density (silenced 100% of the out-of-support mass: fringe shares coarse cells with
    thin features) and fixed-k kNN isolation (blind to 3-10-particle clumps, LOF-class
    scores are at chance on clustered outliers per DROD; larger k reaches the nearby
    feature mass and closes further, measured 43%->13%). The budget form has NO
    per-particle classification to get wrong: every out-of-support particle keeps the
    full pull direction, and one scalar caps the TOTAL pull mass at budget_frac·N
    full-pull equivalents. Early windows (a third of the body outside support) scale
    down ~30x — the dose-response catastrophe cannot form; late windows (a few hundred
    floaters) run at 1.0. Partial/unbalanced-OT reading: a hard bound on transported
    mass per window (Séjourné et al. 2023), self-annealing, N-invariant."""
    with torch.no_grad():
        val = gather_cic(dt_grid, x, grid_min, dx, dims)
        n_out = int((val > 1e-9).sum())
        return min(1.0, budget_frac * len(x) / max(n_out, 1))


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
