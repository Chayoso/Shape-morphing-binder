"""Entropic optimal-transport (Sinkhorn) coverage loss — hypothesis H3 (2026-09-16).

The cell-sum density loss rewards a lone particle in an empty target cell with the largest
marginal gain (that is the pull that walks surface leaders away from the body). A transport
loss moves mass as a FLOW: every particle is matched to target mass under a global plan, so
the gradient on a particle is its displacement to its matched target location and nothing
rewards leaving the body. This is the PRT-EMD choice of PlasticineLab (Huang et al. 2021)
computed with entropic regularisation (Cuturi 2013; Feydy et al. 2019 for the debiased form).

Implementation: log-domain Sinkhorn, cost |x - y|^2, epsilon = (particle spacing)^2 (the
resolution of the cloud itself), between uniform measures. The pipeline path
(entropic_map / debiased_map_displacement) solves the dual on a fixed uniform SUBSAMPLE of
the particles against M target samples (every sweep O(n_sub M), independent of N) with
geometric epsilon-scaling from the squared target diameter, stopping at a row-marginal
error tolerance, and evaluates the out-of-sample entropic map (row-normalised barycentric
projection with the subsample's potentials; Pooladian & Niles-Weed 2021) for all N
particles in one pass. The per-window loss is the L2 distance of each particle to its
(debiased) map image; the optimiser rescales it once by gradient-norm parity with D_vol at
the first window (the same one-shot calibration the h1 / jdens terms use), so no new
weight is introduced. __call__ (the full-plan envelope loss) and barycentric_targets (the
full-plan projection) are kept for tests and small clouds.
"""
from __future__ import annotations

import os

import torch
import warp as wp


class SinkhornPull:
    """Stateful entropic OT between a moving cloud x (N,3) and a fixed target sample y (M,3).

    call(x) -> loss (scalar, differentiable in x through the transport cost only)
    The potentials f (N,), g (M,) persist and warm-start the next call.
    """

    def __init__(self, y: torch.Tensor, eps: float, iters: int = 10, chunk: int = 16384,
                 tol: float = 1e-2):
        self.y = y.detach()
        self.M = y.shape[0]
        self.eps = float(eps)
        self.iters = int(iters)           # sweep budget per solve (a cap, not a count)
        self.tol = float(tol)             # stop when the row-marginal error max_i |N sum_j pi_ij - 1| < tol
        self.last_err = float("nan")      # marginal error of the last solve (diagnostic)
        self.last_sweeps = 0
        self.warm_levels = -1             # eps halvings re-annealed on a warm solve; -1: the full anneal
        self.coarse_sweeps = 0            # >0: fixed sweeps per coarse eps level (only the target level
                                          # iterates to tol); 0: every level iterates to tol
        self.err_norm = "mean"            # marginal error: "mean" = the L1 mass error |pi 1 - a|_1 (the
                                          # standard Sinkhorn criterion; the max over rows stalls on a
                                          # few outliers: real bunny 152 sweeps vs 84, identical map)
        self.last_err_max = float("nan")
        self._self_pull = None            # persistent self-transport solver (debiasing), warm-started
        # rows per chunk so that one (rows x M) cost block stays at <= 2^27 floats (512 MB)
        self.chunk = max(1024, min(int(chunk), (1 << 27) // max(self.M, 1)))
        self.f = None                     # (N,) dual on the particles
        self.f_cold = True                # first solve: epsilon-scaled sweeps
        self.g = torch.zeros(self.M, device=y.device)
        self.log_b = -torch.log(torch.tensor(float(self.M), device=y.device))

    def _cost_rows(self, x, s, e):
        # squared distances (rows s:e of x) -> (e-s, M)
        xs = x[s:e]
        return (xs * xs).sum(1, keepdim=True) - 2.0 * xs @ self.y.T + (self.y * self.y).sum(1)[None, :]

    @torch.no_grad()
    def _solve(self, x):
        N = x.shape[0]
        eps = self.eps
        log_a = -torch.log(torch.tensor(float(N), device=x.device))
        if self.f is None or self.f.shape[0] != N:
            self.f = torch.zeros(N, device=x.device)
        f, g = self.f, self.g
        # epsilon-scaling (Feydy 2019): the first sweeps run at a larger epsilon and anneal
        # to the target so that a small epsilon converges within the sweep budget
        # epsilon-scaling (Schmitzer 2019; Feydy et al. 2019): a COLD solve anneals eps
        # geometrically from the squared diameter of the target down to the target eps,
        # halving per level, each level iterated to the tolerance (warm-started from the
        # previous level). The number of levels is set by the geometry, log2(diam^2/eps);
        # a WARM solve (later windows) runs at the target eps only.
        # Every solve anneals (warm potentials only shorten the levels): at the target eps
        # the fixed-point iteration is too slow to absorb even a sub-spacing move (40k:
        # 570 sweeps warm at the target eps vs 95-108 through the full anneal).
        if self.f_cold or self.warm_levels < 0:
            diam2 = float((self.y.max(0).values - self.y.min(0).values).pow(2).sum())
            n_lev = max(1, int(torch.tensor(diam2 / self.eps).log2().ceil()))
            scales = [2.0 ** k for k in range(n_lev, -1, -1)]
        else:
            scales = [2.0 ** k for k in range(self.warm_levels, -1, -1)]
        self.f_cold = False
        err = float("inf")
        n_sw = 0
        lev = 0
        # the cost block is reused across sweeps when it fits one chunk (the subsample solves)
        C_all = self._cost_rows(x, 0, N) if N <= self.chunk else None
        cost = (lambda s, e: C_all if C_all is not None else self._cost_rows(x, s, e))
        # symmetric case (self-transport of a point set onto itself): f = g, one averaged
        # update per sweep (Feydy et al. 2019, symmetric Sinkhorn), same marginal criterion
        sym = (self.y.shape[0] == N and self.y.data_ptr() == x.data_ptr()
               and C_all is not None)               # the symmetric update needs the cached block
        lev_sw = 0                        # sweeps spent at the current level
        for it in range(self.iters):
            eps = self.eps * scales[lev]
            lev_sw += 1
            coarse_done = (self.coarse_sweeps > 0 and lev < len(scales) - 1
                           and lev_sw >= self.coarse_sweeps)
            if sym:
                lse = torch.logsumexp((f[None, :] - C_all) / eps + self.log_b, dim=1)
                f_new = -eps * lse
                dev_t = (torch.exp((f - f_new) / eps) - 1.0).abs()
                err_t, err_m = dev_t.max(), dev_t.mean()
                f = 0.5 * (f + f_new)
                g = f
                n_sw = it + 1
                if coarse_done:
                    lev += 1; lev_sw = 0
                elif (it % 4 == 3) or it == self.iters - 1:
                    err_max = float(err_t)
                    err = float(err_m) if self.err_norm == "mean" else err_max
                    if err < self.tol:
                        if lev == len(scales) - 1:
                            break
                        lev += 1; lev_sw = 0
                continue
            # g_j = -eps * logsumexp_i( (f_i - C_ij)/eps + log a_i )   (accumulated over row chunks)
            acc = None
            for s in range(0, N, self.chunk):
                e = min(N, s + self.chunk)
                C = cost(s, e)
                lse = torch.logsumexp((f[s:e, None] - C) / eps + log_a, dim=0)
                acc = lse if acc is None else torch.logaddexp(acc, lse)
            g = -eps * acc
            # f_i = -eps * logsumexp_j( (g_j - C_ij)/eps + log b_j ). The row marginal of the
            # plan BEFORE this f-sweep (i.e. after the g-sweep) measures convergence: at the
            # fixed point both sweeps are identities and the marginal error vanishes.
            err_t = None
            err_s = None
            for s in range(0, N, self.chunk):
                e = min(N, s + self.chunk)
                C = cost(s, e)
                lse = torch.logsumexp((g[None, :] - C) / eps + self.log_b, dim=1)
                dev_t = (torch.exp(f[s:e] / eps + lse) - 1.0).abs()
                em, es = dev_t.max(), dev_t.sum()
                err_t = em if err_t is None else torch.maximum(err_t, em)
                err_s = es if err_s is None else err_s + es
                f[s:e] = -eps * lse
            n_sw = it + 1
            if coarse_done:
                lev += 1; lev_sw = 0      # fixed budget spent at a coarse level: next epsilon
                continue
            # the convergence test forces a device sync: check it every 4 sweeps
            if (it % 4 == 3) or it == self.iters - 1:
                err_max = float(err_t)
                err = float(err_s) / N if self.err_norm == "mean" else err_max
                if err < self.tol:
                    if lev == len(scales) - 1:
                        break
                    lev += 1; lev_sw = 0  # next (smaller) epsilon, warm-started
        self.f, self.g = f, g
        self.last_err, self.last_sweeps = err, n_sw
        self.last_err_max = locals().get("err_max", float("nan"))
        return f, g, log_a

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Value: the entropic OT cost OT_eps = <a, f> + <b, g> (dual at the optimum).
        Gradient: sum_j pi_ij dC_ij/dx_i under the DETACHED plan — the envelope-theorem
        derivative of OT_eps, so value and gradient are consistent."""
        f, g, log_a = self._solve(x.detach())
        N = x.shape[0]
        eps = self.eps
        value = (f.mean() + g.mean()).detach()               # uniform a, b
        prim = x.new_zeros(())
        for s in range(0, N, self.chunk):
            e = min(N, s + self.chunk)
            with torch.no_grad():
                Cd = self._cost_rows(x.detach(), s, e)
                logpi = (f[s:e, None] + g[None, :] - Cd) / eps                        # log plan
                pi = torch.softmax(logpi, dim=1) / float(N)                            # rows sum to a_i
            C = self._cost_rows(x, s, e)                                              # differentiable
            prim = prim + (pi * C).sum()
        return value + prim - prim.detach()


    @torch.no_grad()
    def debiased_displacement(self, x: torch.Tensor, n_self: int = 8192, seed: int = 0) -> torch.Tensor:
        """Per-particle displacement of the DEBIASED Sinkhorn divergence
        S_eps = OT_eps(a,b) - 1/2 OT_eps(a,a) - 1/2 OT_eps(b,b): (T_i - x_i) - (T_i^self - x_i)
        = T_i - T_i^self, where T^self is the barycentric map of the cloud onto a subsample of
        ITSELF. The self-term cancels the entropic shrinkage toward the interior (Feydy et al.
        2019), which is what left holes in thin features with the plain barycentric map."""
        T = self.barycentric_targets(x)
        sp = self._self_pull
        if sp is None or sp.M != min(n_self, x.shape[0]):
            g = torch.Generator(device="cpu").manual_seed(seed)
            idx = torch.randperm(x.shape[0], generator=g)[: min(n_self, x.shape[0])].to(x.device)
            sp = self._self_pull = SinkhornPull(x[idx], eps=self.eps, iters=self.iters, tol=self.tol)
            sp.warm_levels = self.warm_levels
            sp._idx = idx
        else:
            sp.y = x[sp._idx].detach()        # the same subsample, moved with the cloud: warm start
        T_self = sp.barycentric_targets(x)
        return T - T_self

    @torch.no_grad()
    def barycentric_targets(self, x: torch.Tensor) -> torch.Tensor:
        """T_i = sum_j pi_ij y_j / sum_j pi_ij — where the plan sends particle i (the OT
        displacement field), the row-NORMALISED barycentric projection. Dividing by a_i
        instead assumed converged row marginals; with the potentials far from converged in
        the first windows that scaled T outside the convex hull of the target (a synthetic
        ball-to-spike test reached 1.7x the spike length). Solved once per window from the
        window start positions; inside the window the loss is the cheap per-particle L2 to
        these targets (the PlasticineLab PRT-EMD matching relaxed to one plan per window)."""
        f, g, log_a = self._solve(x)
        N = x.shape[0]
        eps = self.eps
        T = torch.zeros_like(x)
        for s in range(0, N, self.chunk):
            e = min(N, s + self.chunk)
            C = self._cost_rows(x, s, e)
            pi = torch.softmax((f[s:e, None] + g[None, :] - C) / eps, dim=1)   # row-normalised
            T[s:e] = pi @ self.y                                   # convex combination of targets
        return T

    @torch.no_grad()
    def entropic_map(self, x: torch.Tensor, n_sub: int = 8192, seed: int = 0) -> torch.Tensor:
        """Out-of-sample entropic map (Pooladian & Niles-Weed 2021): the dual potential g on
        the target side is solved on a fixed uniform SUBSAMPLE of the particles (n_sub x M,
        every sweep O(n_sub M) instead of O(N M)), then the row-normalised barycentric
        projection T(x_i) = softmax_j((g_j - |x_i - y_j|^2)/eps) @ y is evaluated for ALL N
        particles in one chunked pass — the same estimator as barycentric_targets, with the
        potentials from the subsample. The subsample is kept across calls (it moves with the
        cloud) and its potentials warm-start the next solve, which still anneals through all
        eps levels (see _solve)."""
        N = x.shape[0]
        n_sub = min(int(n_sub), N)
        if getattr(self, "_sub_idx", None) is None or self._sub_idx.shape[0] != n_sub:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            self._sub_idx = torch.randperm(N, generator=gen)[:n_sub].to(x.device)
            self.f = None
            self.f_cold = True
        xs = x[self._sub_idx]
        f, g, log_a = self._solve(xs)
        T = torch.zeros_like(x)
        for s in range(0, N, self.chunk):
            e = min(N, s + self.chunk)
            C = self._cost_rows(x, s, e)
            T[s:e] = torch.softmax((g[None, :] - C) / self.eps, dim=1) @ self.y
        return T

    @torch.no_grad()
    def debiased_map_displacement(self, x: torch.Tensor, n_sub: int = 8192, seed: int = 0) -> torch.Tensor:
        """Debiased displacement with the subsampled potentials: T(x) - T_self(x), where the
        self map transports the cloud onto its own subsample (the entropic shrinkage cancels,
        Feydy et al. 2019). Cost per call: two n_sub x n_sub solves + two N x n_sub passes."""
        T = self.entropic_map(x, n_sub, seed)
        xs = x[self._sub_idx]
        sp = self._self_pull
        xs = xs.detach()
        if sp is None or sp.M != xs.shape[0]:
            sp = self._self_pull = SinkhornPull(xs, eps=self.eps, iters=self.iters, tol=self.tol)
        else:
            sp.y = xs
        sp.coarse_sweeps = self.coarse_sweeps
        # the self plan is symmetric (a = b = the subsample): solve it on the subsample itself
        # (the same tensor as sp.y triggers the symmetric update in _solve)
        T_self = sp.entropic_map_from(x, sp.y)
        return T - T_self

    @torch.no_grad()
    def _c_transform(self, f_sub: torch.Tensor, xs: torch.Tensor, y_full: torch.Tensor, log_w: float) -> torch.Tensor:
        """Exact dual potential at ANY point of the other side from the subsample potentials:
        g(y) = -eps logsumexp_i((f_i - |x_i - y|^2)/eps + log a_i)  (the c-transform)."""
        M = y_full.shape[0]
        rows = max(1024, (1 << 27) // max(xs.shape[0], 1))
        g = torch.empty(M, device=y_full.device)
        for s in range(0, M, rows):
            e = min(M, s + rows)
            ys = y_full[s:e]
            C = (ys * ys).sum(1, keepdim=True) - 2.0 * ys @ xs.T + (xs * xs).sum(1)[None, :]
            g[s:e] = -self.eps * torch.logsumexp((f_sub[None, :] - C) / self.eps + log_w, dim=1)
        return g

    @torch.no_grad()
    def _map_over(self, x: torch.Tensor, g_full: torch.Tensor, y_full: torch.Tensor) -> torch.Tensor:
        """Row-normalised barycentric map of every x onto the FULL point set y_full with its
        potentials g_full: T(x_i) = softmax_j((g_j - |x_i - y_j|^2)/eps) . y_j."""
        N = x.shape[0]
        rows = max(256, (1 << 27) // max(y_full.shape[0], 1))
        T = torch.zeros_like(x)
        yy = (y_full * y_full).sum(1)[None, :]
        for s in range(0, N, rows):
            e = min(N, s + rows)
            xs = x[s:e]
            C = (xs * xs).sum(1, keepdim=True) - 2.0 * xs @ y_full.T + yy
            T[s:e] = torch.softmax((g_full[None, :] - C) / self.eps, dim=1) @ y_full
        return T

    @torch.no_grad()
    def debiased_map_full(self, x: torch.Tensor, y_full: torch.Tensor, n_sub: int = 8192,
                          seed: int = 0) -> torch.Tensor:
        """Debiased displacement evaluated on the FULL point sets: the dual is solved on the
        subsample (n_sub particles vs the M target samples, as entropic_map), then the
        potentials are c-transformed to every target point and every particle, and the
        barycentric maps are taken over ALL target points (T) and ALL particles (T_self).
        This removes the sample-scale noise of the 8192-point maps (at 150k the sample
        spacing is 2.6 particle spacings; the noise of the subsampled map was the blur radius
        itself, so a converged cloud still showed |d| ~ blur for 3/4 of the particles).
        Cost: one N x M_full and one N x N pass per call (chunked), independent of sweeps."""
        N = x.shape[0]
        n_sub = min(int(n_sub), N)
        if getattr(self, "_sub_idx", None) is None or self._sub_idx.shape[0] != n_sub:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            self._sub_idx = torch.randperm(N, generator=gen)[:n_sub].to(x.device)
            self.f = None
            self.f_cold = True
        xs = x[self._sub_idx]
        f, g, log_a = self._solve(xs)                       # dual on (subsample, target samples)
        g_full = self._c_transform(f, xs, y_full, log_a)    # potential at every target point
        T = self._map_over(x, g_full, y_full)
        # self plan (subsample onto itself, symmetric): potential at every particle
        sp = self._self_pull
        if sp is None or sp.M != xs.shape[0]:
            sp = self._self_pull = SinkhornPull(xs, eps=self.eps, iters=self.iters, tol=self.tol)
        else:
            sp.y = xs.detach()
        sp.coarse_sweeps = self.coarse_sweeps
        fs, gs, log_as = sp._solve(sp.y)
        f_self_full = sp._c_transform(fs, sp.y, x, log_as)
        T_self = sp._map_over(x, f_self_full, x)
        return T - T_self

    @torch.no_grad()
    def entropic_map_from(self, x: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """entropic_map with an explicit subsample xs (the self-transport case)."""
        f, g, log_a = self._solve(xs)
        N = x.shape[0]
        T = torch.zeros_like(x)
        for s in range(0, N, self.chunk):
            e = min(N, s + self.chunk)
            C = self._cost_rows(x, s, e)
            T[s:e] = torch.softmax((g[None, :] - C) / self.eps, dim=1) @ self.y
        return T

    @torch.no_grad()
    def divergence(self, x: torch.Tensor, n_sub: int = 8192, seed: int = 0) -> float:
        """Debiased Sinkhorn divergence to the target sample, up to the constant OT(b,b):
        S = OT_eps(a, b) - 1/2 OT_eps(a, a), both on the fixed n_sub-particle subsample.
        A merit component for the transport recipes: it is what they descend, it is
        defined on the FIXED target, and it decreases monotonically along a transport path
        where the cell sum plateaus (the slow tangential redistribution that formed C's
        arms read as 'no progress' to the fixed-scale merit). Cold-solved each call (the
        potentials of the pacing solver are left untouched)."""
        N = x.shape[0]
        n_sub = min(int(n_sub), N)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(N, generator=gen)[:n_sub].to(x.device)
        xs = x[idx].detach()
        # 2026-09-23: the two solves are WARM across calls (the subsample is the same fixed draw every
        # call, and the cloud moves less than a spacing per window): the same fixed point to the same
        # tolerance, through the warm anneal levels instead of the full cold ladder. Measured at 150k:
        # the cold pair was 0.9 s of a 6 s window. PHYSMORPH_OTDIV_COLD=1 restores the cold solves.
        cold = os.environ.get("PHYSMORPH_OTDIV_COLD", "") == "1"
        ab = getattr(self, "_div_ab", None)
        if cold or ab is None:
            ab = SinkhornPull(self.y, eps=self.eps, iters=self.iters, tol=self.tol)
            ab.f_cold = True
            self._div_ab = ab
        v_ab = float(ab(xs))
        aa = getattr(self, "_div_aa", None)
        if cold or aa is None or aa.M != xs.shape[0]:
            aa = SinkhornPull(xs, eps=self.eps, iters=self.iters, tol=self.tol)
            self._div_aa = aa
        else:
            aa.y = xs
        v_aa = float(aa(xs))
        return v_ab - 0.5 * v_aa

    @torch.no_grad()
    def argmax_targets(self, x: torch.Tensor) -> torch.Tensor:
        """Rounded (Monge) map: the target sample carrying the largest plan mass for each
        particle. No entropic averaging, so a thin feature is reached to its tip (the
        barycentre of a thin feature's samples sits inside it); discrete at the sample scale."""
        f, g, log_a = self._solve(x)
        N = x.shape[0]
        T = torch.zeros_like(x)
        for s in range(0, N, self.chunk):
            e = min(N, s + self.chunk)
            C = self._cost_rows(x, s, e)
            T[s:e] = self.y[((f[s:e, None] + g[None, :] - C) / self.eps).argmax(1)]
        return T


def _grid_measure(grid, grid_min, dx, dims):
    ids = torch.nonzero(grid > 0, as_tuple=False).flatten()
    if not ids.numel():
        raise ValueError('grid transport requires nonempty measures')
    nx, ny, nz = dims
    nodes = torch.stack((ids // (ny * nz), (ids // nz) % ny, ids % nz), 1)
    return grid_min + dx * nodes.to(grid.dtype), grid[ids] / grid[ids].sum()


@torch.no_grad()
def _grid_potentials(xa, a, y, b, eps, iters, tol):
    if not eps > 0 or not tol > 0 or iters < 1:
        raise ValueError('grid transport requires positive blur, tolerance and sweep budget')
    if len(xa) * len(y) > (1 << 27):
        raise ValueError('grid transport support exceeds the dense solve memory budget')
    C = ((xa * xa).sum(1)[:, None] - 2. * xa @ y.T
         + (y * y).sum(1)[None, :]).clamp_min(0.)
    la, lb = a.log(), b.log()
    f, g = torch.zeros_like(a), torch.zeros_like(b)
    span = torch.maximum(xa.max(0).values, y.max(0).values) - torch.minimum(xa.min(0).values, y.min(0).values)
    level = max(0, int(torch.ceil(torch.log2(span.square().sum().clamp_min(eps) / eps))))
    for iteration in range(iters):
        temperature = eps * 2. ** level
        # Parallel half-updates preserve f == g exactly for identical measures.
        fn = -temperature * torch.logsumexp((g[None, :] - C) / temperature + lb[None, :], 1)
        gn = -temperature * torch.logsumexp((f[:, None] - C) / temperature + la[:, None], 0)
        check = iteration % 4 == 3 or iteration == iters - 1
        if check:
            ea = (a * torch.expm1((f - fn) / temperature).abs()).sum()
            eb = (b * torch.expm1((g - gn) / temperature).abs()).sum()
            error = float(torch.maximum(ea, eb))
        f, g = .5 * (f + fn), .5 * (g + gn)
        if check and error < tol:
            if level == 0:
                break
            level -= 1
    if level != 0 or temperature != eps:
        raise ValueError('grid transport did not reach the requested blur within the sweep budget')
    if not error <= tol:
        raise ValueError(f'grid transport did not converge: marginal error {error:g} > {tol:g}')
    return f, g


class _GridTransportNotConverged(ValueError):
    pass


@wp.kernel(enable_backward=False)
def _grid_logsumexp_axis(field: wp.array(dtype=float), cost: wp.array2d(dtype=float),
                        temperature: wp.array(dtype=float), width: int, stride: int,
                        result: wp.array(dtype=float)):
    i = wp.tid()
    coordinate = (i // stride) % width
    start = i - coordinate * stride
    maximum = float(-wp.inf)
    for j in range(width):
        value = field[start + j * stride] - cost[coordinate, j] / temperature[0]
        if wp.isnan(value):
            result[i] = value
            return
        maximum = wp.max(maximum, value)
    if not wp.isfinite(maximum):
        result[i] = maximum  # Includes an entirely empty (-inf) grid line.
        return
    total = float(0.)
    for j in range(width):
        value = field[start + j * stride] - cost[coordinate, j] / temperature[0]
        total += wp.exp(value - maximum)
    result[i] = maximum + wp.log(total)


class GridSinkhornLoss:
    """Fixed-target Sinkhorn divergence with a separable full-grid envelope."""

    def __init__(self, target, grid_min, dx, dims, *, eps, iters=400, tol=1e-3,
                 mass_total=None, cuda_blocks=False, support=None):
        if not eps > 0 or not tol > 0 or iters < 1 or not float(target.sum()) > 0:
            raise ValueError('grid transport requires positive mass, blur, tolerance and sweep budget')
        self.dims, self.eps, self.iters, self.tol = dims, eps, iters, tol
        self.grid_min, self.dx = grid_min, dx
        self.mass_total = mass_total
        self.support = support
        self.b = target.detach() / target.sum()
        self.costs = [((torch.arange(n, device=target.device, dtype=target.dtype)[:, None]
                       - torch.arange(n, device=target.device, dtype=target.dtype)[None, :]) * dx).square()
                      for n in dims]
        self.diameter2 = dx ** 2 * sum((n - 1) ** 2 for n in dims)
        self.cuda_blocks = False
        self._solve_graphs = {}
        self.target_potential, _ = self.solve(self.b, self.b)
        self.cuda_blocks = cuda_blocks

    def transform(self, dual, log_weights, temperature):
        # Squared Euclidean cost is additive by axis. Log-domain separability
        # is exact and avoids both dense 3D costs and underflow of exp(-C/eps).
        if (self.cuda_blocks and dual.is_cuda and not torch.is_grad_enabled()
                and dual.dtype == log_weights.dtype == self.b.dtype == torch.float32):
            # Same stable log-domain reduction, without grid-by-axis temporaries.
            # Use Torch's CURRENT stream, including the non-default capture stream.
            temp = torch.as_tensor(temperature, device=dual.device, dtype=dual.dtype).reshape(1)
            field = (dual / temp + log_weights).reshape(self.dims).reshape(-1)
            stream = wp.stream_from_torch(dual.device)
            stride = field.numel()
            for width, cost in zip(self.dims, self.costs):
                stride //= width
                result = torch.empty_like(field)
                wp.launch(_grid_logsumexp_axis, dim=field.numel(),
                          inputs=[wp.from_torch(field), wp.from_torch(cost),
                                  wp.from_torch(temp), width, stride],
                          outputs=[wp.from_torch(result)], device=stream.device, stream=stream)
                field = result
            return -temp * field
        field = (dual / temperature + log_weights).reshape(self.dims)
        for axis, cost in enumerate(self.costs):
            field = field.movedim(axis, -1)
            field = torch.logsumexp(field.unsqueeze(-2) - cost / temperature, -1)
            field = field.movedim(-1, axis)
        return -temperature * field.reshape(-1)

    @torch.no_grad()
    def solve(self, a, b):
        if (self.cuda_blocks and a.is_cuda and self.iters % 4 == 0
                and a.device == b.device == self.b.device
                and a.dtype == b.dtype == self.b.dtype
                and a.shape == b.shape == self.b.shape):
            with torch.cuda.device(a.device):
                return self._solve_cuda_blocks(a, b)
        la, lb = a.log(), b.log()
        f, g = torch.zeros_like(a), torch.zeros_like(b)
        level = max(0, int(torch.ceil(torch.log2(a.new_tensor(max(self.diameter2, self.eps) / self.eps)))))
        for iteration in range(self.iters):
            temperature = self.eps * 2. ** level
            fn = self.transform(g, lb, temperature)
            # Identical measures keep f == g exactly under the parallel update.
            gn = fn if a is b else self.transform(f, la, temperature)
            check = iteration % 4 == 3 or iteration == self.iters - 1
            if check:
                ea = (torch.exp(la + (f - fn) / temperature) - a).abs().sum()
                eb = (torch.exp(lb + (g - gn) / temperature) - b).abs().sum()
                error = float(torch.maximum(ea, eb))
            f, g = .5 * (f + fn), .5 * (g + gn)
            if check and error < self.tol:
                if level == 0:
                    break
                level -= 1
        if level != 0 or temperature != self.eps or not error <= self.tol:
            raise _GridTransportNotConverged(f'grid transport did not converge: marginal error {error:g}')
        return f, g

    def _solve_cuda_blocks(self, a, b):
        """Capture four unchanged sweeps; keep the original residual/blur schedule.

        Every call starts from zero duals. Graphs own their buffers, not a temporal
        warm start: line-search values must not depend on previous trials.
        Like the owning optimizer, a loss instance is used serially.
        """
        same = a is b
        if same not in self._solve_graphs:
            la, lb = a.log().clone(), b.log().clone()
            f, g = torch.zeros_like(a), torch.zeros_like(b)
            temp, error = a.new_tensor(self.eps), a.new_zeros(())
            abuf, bbuf = a.clone(), b.clone()

            def block():
                for i in range(4):
                    fn = self.transform(g, lb, temp)
                    gn = fn if same else self.transform(f, la, temp)
                    if i == 3:
                        ea = (torch.exp(la + (f - fn) / temp) - abuf).abs().sum()
                        eb = (torch.exp(lb + (g - gn) / temp) - bbuf).abs().sum()
                        error.copy_(torch.maximum(ea, eb))
                    f.copy_(.5 * (f + fn))
                    g.copy_(.5 * (g + gn))

            stream = torch.cuda.Stream(device=a.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                block()
                block()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                block()
            self._solve_graphs[same] = (graph, la, lb, f, g, temp, error, abuf, bbuf)
        graph, la, lb, f, g, temp, error, abuf, bbuf = self._solve_graphs[same]
        la.copy_(a.log())
        lb.copy_(b.log())
        abuf.copy_(a)
        bbuf.copy_(b)
        f.zero_()
        g.zero_()
        level = max(0, int(torch.ceil(torch.log2(a.new_tensor(max(self.diameter2, self.eps) / self.eps)))))
        for iteration in range(0, self.iters, 4):
            temperature = self.eps * 2. ** level
            temp.fill_(temperature)
            graph.replay()
            residual = float(error)
            if residual < self.tol:
                if level == 0:
                    break
                level -= 1
        if level != 0 or temperature != self.eps or not residual <= self.tol:
            raise _GridTransportNotConverged(f'grid transport did not converge: marginal error {residual:g}')
        # The next cross/self solve rewrites the capture buffers.
        return f.clone(), g.clone()

    def __call__(self, current):
        total = current.sum()
        if self.mass_total is not None and abs(float(total.detach()) - self.mass_total) > 1e-6 * self.mass_total:
            # Never reward disappearance beyond the raster boundary.
            return total * 0. + float('inf')
        if not float(total.detach()) > 0:
            raise ValueError('grid transport requires nonempty measures')
        a = current / total
        needs_gradient = torch.is_grad_enabled() and current.requires_grad
        try:
            f, g = self.solve(a, self.b)
            fs, gs = self.solve(a, a)
        except _GridTransportNotConverged:
            if needs_gradient:
                raise
            # An unsolved trial is inadmissible, never an approximate descent step.
            return total * 0. + float('inf')
        # Self solves are symmetric (f == g). Subtract potentials before reducing
        # so tiny divergences do not lose precision against O(1) self-energies.
        value = ((a.double() * (f.double() - fs.double())).sum()
                 + (self.b.double() * (g.double() - self.target_potential.double())).sum()).to(current.dtype)
        if not needs_gradient:
            return value
        with torch.no_grad():
            # Includes zero-mass nodes, so entering an empty CIC node has the
            # correct one-sided derivative even at grid-aligned particles.
            phi = (self.transform(g, self.b.log(), self.eps)
                   - self.transform(gs, a.log(), self.eps))
        envelope = (a * phi).sum()
        return value.detach() + (envelope - envelope.detach())

    def state_energy(self, x, mass, velocity, horizon):
        """Transport plus squared residual displacement, both in length-squared units.

        The physically simulated release tail supplies the actual endpoint.
        The horizon sets the position/velocity conversion without another weight.
        This is a terminal cost, not an edit to the simulated positions/velocities.
        """
        from .volumetric import rasterize_mass
        if self.mass_total is not None:
            actual = rasterize_mass(x, mass, self.grid_min, self.dx, self.dims)
            if abs(float(actual.sum().detach()) - self.mass_total) > 1e-6 * self.mass_total:
                # Inward velocity must not conceal particles already outside.
                return actual.sum() * 0. + velocity.sum() * 0. + float('inf')
        drift = horizon * velocity
        current = rasterize_mass(x, mass, self.grid_min, self.dx, self.dims)
        value = self(current) + (mass * drift.square().sum(1)).sum() / mass.sum()
        return value if self.support is None else self.support(value, x)

@torch.no_grad()
def grid_transport_displacement(x, mass, target_grid, grid_min, dx, dims,
                                *, eps, iters=400, tol=1e-2):
    """Label-independent transport map on the same spatial mass quadrature."""
    from .volumetric import rasterize_mass
    current = rasterize_mass(x, mass, grid_min, dx, dims)
    xa, a = _grid_measure(current, grid_min, dx, dims)
    yb, b = _grid_measure(target_grid, grid_min, dx, dims)

    def mapped(y, weights):
        _, g = _grid_potentials(xa, a, y, weights, eps, iters, tol)
        lb = weights.log()
        result = torch.empty_like(x)
        chunk = max(256, (1 << 26) // len(y))
        yy = (y * y).sum(1)[None, :]
        for start in range(0, len(x), chunk):
            q = x[start:start + chunk]
            cost = (q * q).sum(1)[:, None] - 2. * q @ y.T + yy
            result[start:start + chunk] = torch.softmax((g[None, :] - cost) / eps + lb[None, :], 1) @ y
        return result

    return mapped(yb, b) - mapped(xa, a)


def target_samples(tgt: torch.Tensor, m: int, seed: int = 0) -> torch.Tensor:
    """m uniform samples of the target cloud (the target is a filled sampling already)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(tgt.shape[0], generator=g)[: min(m, tgt.shape[0])]
    return tgt[idx.to(tgt.device)]
