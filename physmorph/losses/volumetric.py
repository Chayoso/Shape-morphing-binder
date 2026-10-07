"""Mass rasterisation and the grid/cleanup terms of the settled objective (torch, on the device).

rasterize_mass is the cloud-in-cell (trilinear) scatter every grid term shares; it is
differentiable in the particle positions. d_vol / d_vol_density are the cell-sum mass
losses (the density form is dimensionless); in settled transport they calibrate the unit
ratio and the transport scale and are logged, the objective itself is the Sinkhorn
divergence (grid_ot.py). d_w1 pulls isolated particles down the target's distance
transform; d_nn_band pulls near-band strays to their nearest target point.
"""
from __future__ import annotations

import torch

from .. import gpu


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
    """Unsigned Euclidean distance transform (world units) OUTSIDE target-occupied cells,
    flat zero inside, optionally clamped. Support = any target CIC mass on this (fine,
    target-fitted) grid, so thin features always count as support."""
    nx, ny, nz = dims
    occ = (target_grid > 1e-6).reshape(nx, ny, nz)
    if not bool(occ.any()):
        raise ValueError("empty support: the EDT would measure distance to the array border")
    dt = gpu.edt(~occ).double() * dx
    if clamp is not None:
        dt = torch.clamp(dt, max=clamp)
    return dt.to(target_grid.dtype).reshape(-1)


def d_w1(x: torch.Tensor, m: torch.Tensor, dt_grid: torch.Tensor,
         grid_min: torch.Tensor, dx: float, dims) -> torch.Tensor:
    """One-sided W1 cleanup: SUM_p m_p DT(x_p), trilinear-sampled. A sum, so the pull on a
    particle is w_dt grad DT whatever N is, pointing at the target support and independent
    of the local density."""
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


def isolation_gate(x: torch.Tensor, lo: float = 1.2, hi: float = 1.8,
                   k: int = 8, local: torch.Tensor | None = None) -> torch.Tensor:
    """Per-particle kNN-isolation gate of the W1 term, detached and frozen per window:
    a ramp of d_kNN / median(d_kNN) from lo to hi, so only true singletons feel the
    full pull and dense rim mass none. local (N,) (D122, a sample of two pitches): each
    particle's spacing over the base; its distance is read in its own spacing, so a coarser
    interior particle is not an isolated one."""
    from ..render.knn_gpu import knn_self_torch
    with torch.no_grad():
        d_t, _ = knn_self_torch(x, k + 1)
        dk_t = d_t[:, -1]
        if local is not None:
            dk_t = dk_t / local.to(dk_t.dtype)
        ratio_t = dk_t / torch.clamp(dk_t.median(), min=1e-12)
        return torch.clamp((ratio_t - lo) / max(hi - lo, 1e-6), 0.0, 1.0).to(x.dtype)


def nn_band_assign(x0: torch.Tensor, target_knn: gpu.KNN, spacing: float,
                   berth_k: float, far_k: float):
    """Frozen per-window assignment of the near-band cleanup: each particle's nearest
    target point, and eligibility for particles between berth_k and far_k target
    spacings from it (inside the berth is legitimate rim)."""
    with torch.no_grad():
        dist, idx = target_knn.query(x0, 1)
        dist, idx = dist[:, 0], idx[:, 0]
        elig = (dist > berth_k * spacing) & (dist < far_k * spacing)
        return idx, elig.float()


def d_nn_band(x: torch.Tensor, m: torch.Tensor, tgt_pts: torch.Tensor,
              assigned: torch.Tensor, elig: torch.Tensor, berth: float) -> torch.Tensor:
    """SUM_p m_p elig_p relu(|x_p - tgt[assigned_p]| - berth): a constant pull toward the
    assigned target point beyond the berth, zero inside."""
    d = (x - tgt_pts[assigned]).norm(dim=1)
    return (m * elig * torch.clamp(d - berth, min=0.0)).sum()


def d_nn_band_current(x: torch.Tensor, m: torch.Tensor, tgt_pts: torch.Tensor,
                      elig: torch.Tensor, berth: float, target_knn: gpu.KNN,
                      far: float | None = None) -> torch.Tensor:
    """d_nn_band against each particle's CURRENT nearest target point (queried at every
    evaluation, so value and gradient agree; continuous across nearest-point switches).
    far: the band's outer edge, a length; particles at or beyond it are not counted (the same
    band as the frozen assignment's, read at the current state)."""
    if not bool(torch.isfinite(x).all()):
        # A bad trial is rejected by the finite-state and merit checks; keep a zero derivative.
        return (torch.nan_to_num(x) * 0).sum() + x.new_tensor(float('inf'))
    dist, idx = target_knn.query(x.detach(), 1)
    if far is not None:
        elig = elig * (dist[:, 0] < far).to(elig.dtype)
    return d_nn_band(x, m, tgt_pts, idx[:, 0], elig, berth)


def d_vol(x: torch.Tensor, m: torch.Tensor, target_grid: torch.Tensor,
          grid_min: torch.Tensor, dx: float, dims) -> torch.Tensor:
    """Log-mass-ratio cell sum (the legacy unit of every fixed weight)."""
    cur = rasterize_mass(x, m, grid_min, dx, dims)
    diff = torch.log(cur + 1.0) - torch.log(target_grid + 1.0)
    return 0.5 * (diff * diff).sum()


def density_units(target_grid: torch.Tensor) -> tuple[float, int]:
    """(m_ref, n_support): the mean target mass of an occupied cell and the number of
    occupied cells, the constants of the dimensionless d_vol_density."""
    occ = target_grid > 1e-6
    n = int(occ.sum())
    m_ref = float(target_grid[occ].mean()) if n else 1.0
    return max(m_ref, 1e-12), max(n, 1)


def d_vol_density(x: torch.Tensor, m: torch.Tensor, target_grid: torch.Tensor,
                  grid_min: torch.Tensor, dx: float, dims, m_ref: float,
                  n_support: int) -> torch.Tensor:
    """Dimensionless mass matching:
    D = 1/2 (1/n_support) sum_cells [log(1 + m/m_ref) - log(1 + m_t/m_ref)]^2."""
    cur = rasterize_mass(x, m, grid_min, dx, dims)
    diff = torch.log1p(cur / m_ref) - torch.log1p(target_grid / m_ref)
    return 0.5 * (diff * diff).sum() / float(n_support)


# ---- Xu et al.'s EndLayerMassLoss, reproduced for the baseline (legacy/DiffMPMLib3D/CompGraph.cpp) ----

def _cubic(r: torch.Tensor) -> torch.Tensor:
    """The MPM's cubic B-spline of a distance in cells (constitutive.bspline_w)."""
    ar = r.abs()
    return torch.where(ar < 1.0, 0.5 * ar ** 3 - ar ** 2 + 2.0 / 3.0,
                       torch.where(ar < 2.0, (2.0 - ar).clamp_min(0.0) ** 3 / 6.0, torch.zeros_like(ar)))


def rasterize_mass_cubic(x: torch.Tensor, m: torch.Tensor, grid_min: torch.Tensor, dx: float, dims) -> torch.Tensor:
    """P2G of particle mass with the MPM's cubic B-spline (the 4^3 stencil of kernels.base_node) onto a flat grid:
    the C++ oracle's G_Reset + P2G before EndLayerMassLoss."""
    nx, ny, nz = dims
    X = (x - grid_min) / dx
    base = torch.floor(X).long() - 1
    grid = x.new_zeros(nx * ny * nz)
    w = [[_cubic(base[:, a] + o - X[:, a]) for o in range(4)] for a in range(3)]
    for ox in range(4):
        for oy in range(4):
            wxy = w[0][ox] * w[1][oy]
            for oz in range(4):
                ii, jj, kk = base[:, 0] + ox, base[:, 1] + oy, base[:, 2] + oz
                valid = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny) & (kk >= 0) & (kk < nz)
                idx = ((ii * ny + jj) * nz + kk).clamp(0, nx * ny * nz - 1)
                grid = grid.index_add(0, idx, torch.where(valid, wxy * w[2][oz] * m, torch.zeros_like(m)))
    return grid


def d_vol_xu(x: torch.Tensor, m: torch.Tensor, target_grid: torch.Tensor, grid_min: torch.Tensor, dx: float, dims,
             out_of_target: float = 5.0, eps: float = 1e-4, min_mass: float = 1e-3,
             penalty_weight: float = 1.0) -> torch.Tensor:
    """Xu et al.'s EndLayerMassLoss as the C++ oracle computes it: on the simulation grid (cubic B-spline P2G), the
    value 1/2 sum (log(c + 1 + eps) - log(t + 1 + eps))^2 + penalty_weight sum_{c < min_mass} (min_mass - c)^2, and
    the gradient the oracle back-propagates: each node's dL/dm times out_of_target (5) where the target node is
    empty, so a particle outside the target is pulled back five times as hard (the value is unchanged)."""
    cur = rasterize_mass_cubic(x, m, grid_min, dx, dims)
    log_diff = torch.log(cur + 1.0 + eps) - torch.log(target_grid + 1.0 + eps)
    low = (cur < min_mass).to(cur.dtype)
    value = 0.5 * log_diff.pow(2).sum() + penalty_weight * ((min_mass - cur).pow(2) * low).sum()
    dLdm = log_diff / (cur + 1.0 + eps) - 2.0 * penalty_weight * (min_mass - cur) * low
    pen = torch.where(target_grid > 1e-12, torch.ones_like(target_grid), torch.full_like(target_grid, out_of_target))
    surrogate = ((pen * dLdm).detach() * cur).sum()
    return value.detach() + surrogate - surrogate.detach()
