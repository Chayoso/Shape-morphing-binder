"""Entropic optimal-transport (Sinkhorn) coverage loss — hypothesis H3 (2026-09-16).

The cell-sum density loss rewards a lone particle in an empty target cell with the largest
marginal gain (that is the pull that walks surface leaders away from the body). A transport
loss moves mass as a FLOW: every particle is matched to target mass under a global plan, so
the gradient on a particle is its displacement to its matched target location and nothing
rewards leaving the body. This is the PRT-EMD choice of PlasticineLab (Huang et al. 2021)
computed with entropic regularisation (Cuturi 2013; Feydy et al. 2019 for the debiased form).

Implementation: log-domain Sinkhorn between the particle cloud (uniform weights, all N
particles, chunked over rows) and M target samples (uniform), cost |x - y|^2, epsilon =
(eps_cells * dx)^2 — the loss cell is the resolution the plan can distinguish. The dual
potentials are solved WITHOUT autograd (warm-started across calls) and the loss value that is
differentiated is the primal transport cost under the DETACHED plan; by the envelope theorem
its gradient with respect to x equals the exact Sinkhorn gradient at the optimum. Units: the
optimiser rescales it once by gradient-norm parity with D_vol at the first window (the same
one-shot calibration the h1 / jdens terms use), so no new weight is introduced.
"""
from __future__ import annotations

import torch


class SinkhornPull:
    """Stateful entropic OT between a moving cloud x (N,3) and a fixed target sample y (M,3).

    call(x) -> loss (scalar, differentiable in x through the transport cost only)
    The potentials f (N,), g (M,) persist and warm-start the next call.
    """

    def __init__(self, y: torch.Tensor, eps: float, iters: int = 10, chunk: int = 16384):
        self.y = y.detach()
        self.M = y.shape[0]
        self.eps = float(eps)
        self.iters = int(iters)
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
        scales = [4.0, 2.0, 1.0] if self.f_cold else [1.0]
        self.f_cold = False
        for it in range(self.iters):
            eps = self.eps * scales[min(len(scales) - 1, (it * len(scales)) // max(self.iters, 1))]
            # f_i = -eps * logsumexp_j( (g_j - C_ij)/eps + log b_j )
            for s in range(0, N, self.chunk):
                e = min(N, s + self.chunk)
                C = self._cost_rows(x, s, e)
                f[s:e] = -eps * torch.logsumexp((g[None, :] - C) / eps + self.log_b, dim=1)
            # g_j = -eps * logsumexp_i( (f_i - C_ij)/eps + log a_i )   (accumulated over row chunks)
            acc = None
            for s in range(0, N, self.chunk):
                e = min(N, s + self.chunk)
                C = self._cost_rows(x, s, e)
                lse = torch.logsumexp((f[s:e, None] - C) / eps + log_a, dim=0)
                acc = lse if acc is None else torch.logaddexp(acc, lse)
            g = -eps * acc
        self.f, self.g = f, g
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
                logpi = (f[s:e, None] + g[None, :] - Cd) / eps + log_a + self.log_b   # log plan
                pi = torch.exp(logpi)                                                  # (rows, M)
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
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(x.shape[0], generator=g)[: min(n_self, x.shape[0])].to(x.device)
        self_pull = SinkhornPull(x[idx], eps=self.eps, iters=self.iters)
        T_self = self_pull.barycentric_targets(x)
        return T - T_self

    @torch.no_grad()
    def barycentric_targets(self, x: torch.Tensor) -> torch.Tensor:
        """T_i = sum_j pi_ij y_j / a_i — where the plan sends particle i (the OT displacement
        field). Solved once per window from the window's start positions; inside the window
        the loss is the cheap per-particle L2 to these targets (a per-window EMD matching,
        the PlasticineLab PRT-EMD choice relaxed to one matching per window)."""
        f, g, log_a = self._solve(x)
        N = x.shape[0]
        eps = self.eps
        T = torch.zeros_like(x)
        for s in range(0, N, self.chunk):
            e = min(N, s + self.chunk)
            C = self._cost_rows(x, s, e)
            logpi = (f[s:e, None] + g[None, :] - C) / eps + log_a + self.log_b
            pi = torch.exp(logpi)                                  # rows sum to a_i = 1/N
            T[s:e] = (pi @ self.y) * float(N)
        return T


def target_samples(tgt: torch.Tensor, m: int, seed: int = 0) -> torch.Tensor:
    """m uniform samples of the target cloud (the target is a filled sampling already)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(tgt.shape[0], generator=g)[: min(m, tgt.shape[0])]
    return tgt[idx.to(tgt.device)]

