"""Fickian particle shifting of the sub-cell arrangement (docs/method.md 10.18; Lind, Xu, Stansby,
Rogers 2012, via Lind, Rogers, Stansby 2020 §10; docs/related_work.md 2026-09-23 night).

The quadrature of the continuum — the particles below the grid cell — is a null space of the
cell sum, the transport plan and the render, and no term of the objective orders it (the pair
c300 / d300, docs/experiments.md 2026-09-23 night: stretched tips, clumps, cavities, a bumpy
surface, and an outer loop that chases sub-cell residuals window after window). Shifting is the
SPH remedy: the particle concentration C_i = Σ_j (m_j / ρ_j) W_ij is 1 in a uniform arrangement,
and each particle diffuses down its gradient,

    Δx_i = −λ h² ∇C_i,   ∇C_i = Σ_j (m_j / ρ_j) [1 + R (W_ij / W(Δp))^n] ∇_i W_ij,   λ = ½, R = 0.2, n = 4,

λ = ½ being the stability limit of the explicit diffusion step (Lind 2012 / Skillen 2013 eq.
10.3–10.4 of the 2020 review) and [1 + R (W_ij / W(Δp))^n] Monaghan's (2000) anti-pairing factor
that Lind 2012 carries in the shifting gradient (R = 0.2, n = 4, Δp the particle spacing): without
it the kernel gradient vanishes for close pairs and the step DIS-orders a cloud (measured,
`/scratch/shift_probe2.py`: the 1-NN spacing CV of a jittered lattice 0.16 → 0.36 in ten
steps; with it 0.16 → 0.08, a uniform random cloud 0.38 → 0.10). With uniform
masses m_j / ρ_j = 1 / n_j (n_j the kernel number density) and a Gaussian W of width h the
normalisation cancels. Positions only: masses, velocities, C, F are untouched (the shift is a
fraction of a spacing; Lind: the velocity correction "is often not necessary"). The free-surface
rule: where the neighbourhood is one-sided (the centroid offset, the same measure as the outer
layer of surface_recon.layer_by_asymmetry) the shift keeps its tangential part only, with a
weight rising to 1 at half a spacing of offset, so the surface is ordered along itself and never
pushed in or out. The explicit step never moves a particle past its neighbours: |Δx| ≤ ½ spacing.
h = h_sp spacings of the NATIVE quadrature, default 1: the Gaussian exp(−r²/h²) at h = Δp has the
width of the cubic-spline kernel at the standard SPH smoothing length h = 1.3 Δp (σ 0.71 against
0.78 Δp) — the literature's ratio, and the one that orders (h = 1.3 Δp of the Gaussian is weaker, 2 Δp
dis-orders: probe above). k = 40 neighbours: the Gaussian is below 2 % beyond 2 h, and a ball of
radius 2 Δp holds (4π/3)·8 ≈ 34 particles."""
from __future__ import annotations

import numpy as np


def native_spacing(x: np.ndarray, n_sub: int = 20000, seed: int = 0) -> float:
    """The cloud's median 8-NN distance, on a subsample scaled to the full density (the same
    estimator as the optimiser's sp0 and the renderer's spacing)."""
    from scipy.spatial import cKDTree
    x = np.ascontiguousarray(x, np.float32)
    n = min(len(x), n_sub)
    sub = x[np.random.default_rng(seed).choice(len(x), n, replace=False)] if n < len(x) else x
    d = cKDTree(sub).query(sub, k=9, workers=-1)[0][:, -1]
    return float(np.median(d)) * (n / len(x)) ** (1.0 / 3.0)


def fickian_shift(x: np.ndarray, spacing: float, h_sp: float = 1.0, k: int = 40, lam: float = 0.5,
                  cap_sp: float = 0.5, pair_R: float = 0.2, pair_n: int = 4):
    """One explicit Fickian shifting step of the cloud x (N,3) at particle spacing `spacing`.
    Returns (dx (N,3) float32, stats): stats = median / p99 / max shift in spacings, the number
    of particles under the free-surface rule, and the concentration-gradient RMS before the step
    (|∇C| h, dimensionless: the disorder measure)."""
    import torch
    from ..render.knn_gpu import gpu_available, knn_self, knn_self_torch
    x = np.ascontiguousarray(x, np.float32)
    N = len(x)
    h = float(h_sp * spacing)
    k = int(min(k, N - 1))
    if k < 2:
        return np.zeros_like(x), {"median_sp": 0.0, "p99_sp": 0.0, "max_sp": 0.0, "n_surface": 0, "disorder": 0.0}
    if gpu_available() and N >= 4096:
        xt = torch.as_tensor(x, device="cuda")
        d, nb = knn_self_torch(xt, k + 1)                      # self at column 0
        d = d[:, 1:].float(); nb = nb[:, 1:].long()
    else:
        xt = torch.as_tensor(x)
        d_np, nb_np = knn_self(x, k + 1)                       # scipy rows, self at column 0
        d = torch.as_tensor(np.ascontiguousarray(d_np[:, 1:], np.float32))
        nb = torch.as_tensor(np.ascontiguousarray(nb_np[:, 1:]).astype(np.int64))
    with torch.no_grad():
        w = torch.exp(-(d / h) ** 2)                            # W_ij, Gaussian of width h
        n = 1.0 + w.sum(1)                                      # kernel number density (self included)
        r = xt[:, None, :] - xt[nb]                             # x_i − x_j  (N,k,3)
        # ∇_i W_ij = −(2 r_ij / h²) W_ij  ⇒  Δx_i = −λ h² Σ_j f_ij ∇_i W_ij / n_j = 2λ Σ_j f_ij r_ij W_ij / n_j
        # with f_ij = 1 + R (W_ij / W(Δp))^n the anti-pairing factor (Monaghan 2000; Lind 2012)
        f = 1.0 + float(pair_R) * (w / float(np.exp(-(float(spacing) / h) ** 2))) ** int(pair_n)
        g = (r * (f * w / n[nb])[..., None]).sum(1)             # Σ_j f_ij r_ij W_ij / n_j  (= −h²/2 ∇C_i)
        dx = 2.0 * float(lam) * g
        disorder = float((2.0 * g.norm(dim=1) / h).pow(2).mean().sqrt())   # |∇C| h, RMS
        # the free-surface rule: one-sided neighbourhood → tangential shift only
        off = xt - xt[nb].mean(1)
        om = off.norm(dim=1)
        nrm = off / (om[:, None] + 1e-12)
        ws = torch.clamp(om / (0.5 * float(spacing)), 0.0, 1.0)
        dx = dx - ws[:, None] * (dx * nrm).sum(1, keepdim=True) * nrm
        # never past a neighbour: |Δx| ≤ cap_sp spacings
        m = dx.norm(dim=1)
        dx = dx * torch.clamp(float(cap_sp * spacing) / (m + 1e-12), max=1.0)[:, None]
        m = dx.norm(dim=1)
        stats = {"median_sp": float(m.median()) / float(spacing),
                 "p99_sp": float(torch.quantile(m, 0.99)) / float(spacing),
                 "max_sp": float(m.max()) / float(spacing),
                 "n_surface": int((ws >= 1.0).sum()),
                 "disorder": disorder}
    return dx.float().cpu().numpy(), stats
