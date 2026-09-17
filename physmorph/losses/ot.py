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

import torch


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
        sym = self.y.shape[0] == N and self.y.data_ptr() == x.data_ptr()
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


def target_samples(tgt: torch.Tensor, m: int, seed: int = 0) -> torch.Tensor:
    """m uniform samples of the target cloud (the target is a filled sampling already)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(tgt.shape[0], generator=g)[: min(m, tgt.shape[0])]
    return tgt[idx.to(tgt.device)]

