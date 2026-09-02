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
    """Per-particle kNN-isolation gate for the W1 term, detached — RESTORED after the
    budget experiment: on the honest metric the kNN-gated arm is the clear winner
    (hero8_dtiso out_dt 0.477% / ear 1 / silIoU 0.9601 vs budget 0.905% / 8 / 0.9444
    = a measured no-op). Lesson: selectivity in WHO gets pulled beats scheduling of
    how much — the kNN gate runs full pull on true singletons ALL run long and never
    touches dense rim mass. Its inversion side effect is owned by the trajectory-det
    acceptance guard. KNOWN UNRESOLVED (stack-review f7): compact 3-10-particle clumps
    sit below the lo ramp and get zero pull — fill v1 did NOT absorb them (measured
    ineffective, ear coverage 23.1->23.5%) and the flagship runs without fill; treat
    clumps as an open defect with no owner. gate = ramp of d_kNN/median from lo to hi,
    frozen per window."""
    from scipy.spatial import cKDTree
    import numpy as np
    with torch.no_grad():
        xn = x.detach().cpu().numpy()
        dk = cKDTree(xn).query(xn, k=k + 1, workers=-1)[0][:, -1]
        import numpy as np
        ratio = dk / max(float(np.median(dk)), 1e-12)
        gate = np.clip((ratio - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return torch.as_tensor(gate, dtype=x.dtype, device=x.device)


def w1_budget(x: torch.Tensor, dt_grid: torch.Tensor, grid_min: torch.Tensor,
              dx: float, dims, budget_frac: float) -> float:
    """Scalar transport-budget factor for the W1 term: min(1, budget·N / n_out).

    FALSIFIED as the flagship gate (§7.6: measured no-op at budget 0.01 — mid-run
    partial pull on every out particle cleaned nothing and damaged the rim; kNN
    selectivity won on the honest metric). Kept only for A/B. Original design notes:
    per-particle gates were falsified twice before it — grid-CIC
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


def deficit_field(x: torch.Tensor, m: torch.Tensor, tmass_fine: torch.Tensor,
                  grid_min: torch.Tensor, dx: float, dims, thresh: float = 0.6,
                  sigma: float = 2.0, clamp: float | None = None):
    """Per-window HOLE-side W1 field: EDT (world units) to under-covered target cells,
    restricted to TRUE target support. Returns (dt_flat, deficit_mass) or None.

    Fattal's gathering-term construction (Target-Driven Smoke, TOG 04) on grids:
    blur body and target mass (sigma cells), mark cells where blurred body < thresh x
    blurred target — a one-signed relative-ratio residual (the AbsGS lesson) — AND the
    unblurred target actually occupies the cell (v2: without the support-AND the mask
    was 95-100% Gaussian tail OUTSIDE the target = an outward fringe factory, Opus
    stack-review F1). thresh 0.6 (was 0.3: the fixed point was a 30%-filled feature,
    F13; hysteresis still absent — documented residual, bang-bang bounded by the
    budget). deficit_mass = summed shortfall, the budget's denominator (F2: budgeting
    on the in-range PARTICLE count attenuated the useful mode 22-78x while the harmful
    mode ran at full weight). Frozen per window, detached."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    import numpy as np
    nx, ny, nz = dims
    with torch.no_grad():
        b = rasterize_mass(x, m, grid_min, dx, dims).reshape(nx, ny, nz).cpu().numpy()
        t = tmass_fine.reshape(nx, ny, nz).cpu().numpy()
        bb = gaussian_filter(b, sigma)
        tb = gaussian_filter(t, sigma)
        # v2 (Opus stack-review F1, CRITICAL): the ratio test on the blurred field
        # alone put 95-100% of the mask OUTSIDE target support (Gaussian tails), so
        # v1 was an outward surface-dilation force — a fringe FACTORY. AND with the
        # unblurred target occupancy: measured 1140->0 spurious cells with 99.3% of
        # one-cell ear target cells still marked.
        deficit = (t > 1e-6) & (tb > 1e-4) & (bb < thresh * tb)
        if not deficit.any():
            return None
        dt = distance_transform_edt(~deficit) * dx
        if clamp is not None:
            dt = np.minimum(dt, clamp)
        deficit_mass = float(np.maximum(thresh * tb - bb, 0.0)[deficit].sum())
        return (torch.as_tensor(dt, dtype=x.dtype, device=x.device).reshape(-1),
                deficit_mass)


def nn_band_assign(x0: torch.Tensor, tgt_pts: torch.Tensor, spacing: float,
                   berth_k: float = 1.5, far_k: float = 4.5,
                   tail_frac: float = 0.0):
    """Frozen per-window assignment for the GRID-FREE near-band cleanup (fork-halo
    forensic 2026-09-02): the visible floaters live 0.05-0.10 wu off support — INSIDE
    the fine-DT grid's CIC-dilation dead band, where the W1 pull is zero regardless of
    any gate. This term uses the target's own geometry (nearest target particle), so
    there is no dilation and no dead band by construction. Eligibility is the NEAR
    band only (berth_k..far_k target-NN spacings): inside the berth is legitimate rim
    (zero force there via the relu in d_nn_band); beyond far_k stays the DT-W1's
    kNN-gated job by default.  ``tail_frac>0`` additionally selects only the
    farthest bounded fraction beyond far_k; this catches coherent floater clusters
    without turning the term into an all-particle Chamfer servo.  Returns
    (assigned_idx, eligible_mask), both frozen/detached."""
    from scipy.spatial import cKDTree
    import numpy as np
    with torch.no_grad():
        xn = x0.detach().cpu().numpy()
        tn = tgt_pts.detach().cpu().numpy()
        dist, idx = cKDTree(tn).query(xn, workers=-1)
        elig = (dist > berth_k * spacing) & (dist < far_k * spacing)
        tail_frac = float(np.clip(tail_frac, 0.0, 1.0))
        budget = min(len(dist), int(np.ceil(tail_frac * len(dist))))
        far = np.flatnonzero(dist >= far_k * spacing)
        if budget > 0 and len(far):
            if len(far) > budget:
                chosen = far[np.argpartition(dist[far], -budget)[-budget:]]
            else:
                chosen = far
            elig[chosen] = True
        return (torch.as_tensor(np.ascontiguousarray(idx), device=x0.device),
                torch.as_tensor(elig.astype(np.float32), device=x0.device))


def d_nn_band(x: torch.Tensor, m: torch.Tensor, tgt_pts: torch.Tensor,
              assigned: torch.Tensor, elig: torch.Tensor, berth: float) -> torch.Tensor:
    """SUM_p m_p·elig_p·relu(|x_p - tgt[assigned_p]| - berth): constant per-particle
    pull toward the assigned target point beyond the berth, zero inside (rim safe)."""
    dvec = x - tgt_pts[assigned]
    d = dvec.norm(dim=1)
    return (m * elig * torch.clamp(d - berth, min=0.0)).sum()


def growth_demand(x: torch.Tensor, m: torch.Tensor, tmass_fine: torch.Tensor,
                  grid_min: torch.Tensor, dx: float, dims,
                  sigma: float = 2.0) -> "np.ndarray":
    """Per-particle coverage demand in [0,1] for the GROWTH channel: blurred relative
    shortfall relu(tb-bb)/tb on TRUE support, trilinearly gathered at the particle.
    Zero wherever coverage is met — growth stops by construction (demand-driven)."""
    from scipy.ndimage import gaussian_filter
    import numpy as np
    nx, ny, nz = dims
    with torch.no_grad():
        b = rasterize_mass(x, m, grid_min, dx, dims).reshape(nx, ny, nz).cpu().numpy()
        t = tmass_fine.reshape(nx, ny, nz).cpu().numpy()
        bb = gaussian_filter(b, sigma)
        tb = gaussian_filter(t, sigma)
        dem = np.where(t > 1e-6, np.maximum(tb - bb, 0.0) / np.maximum(tb, 1e-9), 0.0)
        field = torch.as_tensor(dem, dtype=x.dtype, device=x.device).reshape(-1)
        return gather_cic(field, x, grid_min, dx, dims).clamp(0.0, 1.0).cpu().numpy()


def deficit_assign(x0: torch.Tensor, m: torch.Tensor, tmass_fine: torch.Tensor,
                   grid_min: torch.Tensor, dx: float, dims, thresh: float = 0.6,
                   sigma: float = 2.0, cap_frac: float = 0.02,
                   range_wu: float = 1e9):
    """Fill v4 — TARGET-HARD assignment (the partial-OT reading: demand marginal
    hard, source soft; the symmetric relaxation discards instead of filling, Bai et
    al. warning). Growth v1's autopsy: demand gathered AT particles is a universal
    surface signal (99% of high demand landed outside the ears) because THE DEFICIT
    IS WHERE PARTICLES ARE NOT. So anchor on the deficit CELLS: each under-covered
    true-support cell requests donors — its nearest body particles, capacity scaled
    with its shortfall mass — and only the MATCHED pairs feel any pull. The pair
    count is capacity-bounded (<= cap_frac*N), so dominance is bounded by the
    matching itself, not by a weight scalar that then dies with the physics gradient
    (the fill-v3 failure). Frozen per window. Returns (particle_idx, centers) or
    None."""
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
    import numpy as np
    nx, ny, nz = dims
    with torch.no_grad():
        b = rasterize_mass(x0, m, grid_min, dx, dims).reshape(nx, ny, nz).cpu().numpy()
        t = tmass_fine.reshape(nx, ny, nz).cpu().numpy()
        bb = gaussian_filter(b, sigma)
        tb = gaussian_filter(t, sigma)
        deficit = (t > 1e-6) & (tb > 1e-4) & (bb < thresh * tb)
        if not deficit.any():
            return None
        ii, jj, kk = np.nonzero(deficit)
        gm = grid_min.cpu().numpy()
        centers = np.stack([ii, jj, kk], 1).astype(np.float32) * dx + gm + 0.5 * dx
        short = np.maximum(thresh * tb - bb, 0.0)[deficit]
        order = np.argsort(-short)               # worst-covered cells claim donors first
        centers, short = centers[order], short[order]
        n_cap = max(1, int(cap_frac * len(x0)))
        xn = x0.detach().cpu().numpy()
        tree = cKDTree(xn)
        used = np.zeros(len(xn), bool)
        pi, cc = [], []
        kq = 8
        dists_all, idxs_all = tree.query(centers, k=kq, workers=-1)
        for row in range(len(centers)):
            k_need = int(np.clip(np.ceil(short[row]), 1, 4))
            took = 0
            for dd, idx in zip(dists_all[row], idxs_all[row]):
                if used[idx] or dd > range_wu:
                    continue
                used[idx] = True
                pi.append(int(idx)); cc.append(centers[row])
                took += 1
                if took >= k_need:
                    break
            if len(pi) >= n_cap:
                break
        if not pi:
            return None
        return (torch.as_tensor(np.asarray(pi), device=x0.device),
                torch.as_tensor(np.asarray(cc, np.float32), device=x0.device))


def d_fill_pairs(x: torch.Tensor, pidx: torch.Tensor, centers: torch.Tensor,
                 berth: float) -> torch.Tensor:
    """SUM over matched pairs of relu(|x_p - cell_center| - berth): constant pull of
    each matched donor into its assigned deficit cell; zero once inside the berth."""
    d = (x[pidx] - centers).norm(dim=1)
    return torch.clamp(d - berth, min=0.0).sum()


def coverage_shortfall(x: torch.Tensor, m: torch.Tensor, tmass_fine: torch.Tensor,
                       grid_min: torch.Tensor, dx: float, dims,
                       sigma: float = 2.0) -> float:
    """Stable fill-progress statistic (stack-review f12/F9): normalized continuous
    target shortfall sum(relu(t_blur - b_blur))/sum(t_blur) over TRUE support — no
    mask, no threshold, no gate, so it is comparable across windows (per-window binary
    EDT energies are not: their masks change)."""
    from scipy.ndimage import gaussian_filter
    import numpy as np
    nx, ny, nz = dims
    with torch.no_grad():
        b = rasterize_mass(x, m, grid_min, dx, dims).reshape(nx, ny, nz).cpu().numpy()
        t = tmass_fine.reshape(nx, ny, nz).cpu().numpy()
        bb = gaussian_filter(b, sigma)
        tb = gaussian_filter(t, sigma)
        sup = t > 1e-6
        denom = float(tb[sup].sum())
        return float(np.maximum(tb - bb, 0.0)[sup].sum()) / max(denom, 1e-9)


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
