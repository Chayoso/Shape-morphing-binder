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


def isolation_gate(x: torch.Tensor, lo: float = 1.2, hi: float = 1.8,
                   k: int = 8) -> torch.Tensor:
    """Per-particle COMPLEMENTARITY gate for the W1 term, at kNN SCALE, detached.

    The W1 term exists for the mass the density losses cannot see (sparse fringe);
    dense outside mass is d_vol/d_render's job, and giving it the constant W1 pull
    double-drives bulk transport — measured as a dose-response catastrophe (stray
    0.8/4.0/7.8% at w_dt 0.05/0.2/1.0, inversions, first3 100%). The first gate
    (loss-grid CIC density) was falsified by autopsy: fringe between thin features
    shares coarse cells with the features, so the gate read median 0.0000 on 100% of
    the out-of-support mass — silenced exactly where needed. kNN scale separates them
    (measured on hero5: out-of-support kNN ratio median 1.69 vs bulk 0.99), and a
    false positive inside support costs nothing (DT=0 there). gate = ramp of
    d_kNN/median(d_kNN) from lo to hi; frozen per window (detached). Anchor: Floaters
    No More — recondition the gradient by a region property, not the objective; the
    ratio is the same isolation definition as the stray metric itself."""
    from scipy.spatial import cKDTree
    with torch.no_grad():
        xn = x.detach().cpu().numpy()
        dk = cKDTree(xn).query(xn, k=k + 1, workers=-1)[0][:, -1]
        import numpy as np
        ratio = dk / max(float(np.median(dk)), 1e-12)
        gate = np.clip((ratio - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return torch.as_tensor(gate, dtype=x.dtype, device=x.device)


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
