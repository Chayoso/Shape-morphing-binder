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

2026-09-14 (docs/render_controls_physics.md §6): the field may now carry K channels
(3 for dL/dx, 9 for dL/dF — the audit found the smoothing branch DROPPED the covariance
covector), and the stationary red-black sweeps can be accelerated with the Chebyshev
semi-iterative scheme of Wang 2015 (“A Chebyshev semi-iterative approach for
accelerating projective and position-based dynamics”), the same acceleration VBD
inherits from PD: with the sweep's spectral radius rho the k-th iterate is
    u_{k+1} = omega_{k+1} (gamma (S(u_k) - u_k) + u_k - u_{k-1}) + u_{k-1},
    omega_1 = 1, omega_2 = 2/(2 - rho^2), omega_{k+1} = 4 / (4 - rho^2 omega_k),
which turns the O(rho^k) decay of GS into O((rho / (1 + sqrt(1 - rho^2)))^k).
For screened diffusion (I + kappa L) u = g the Jacobi radius is kappa/(1+kappa) and
red-black GS has rho = (kappa/(1+kappa))^2; that is the default estimate.
"""
from __future__ import annotations

import torch


def chebyshev_rho(kappa: float) -> float:
    """Spectral radius of the red-black GS sweep for (I + kappa L) with the 6-point
    Laplacian average (Jacobi radius kappa/(1+kappa), GS = Jacobi squared). This is the
    periodic/Neumann bound; avg6 divides by 6 at the grid boundary (an implicit zero
    extension), so the true radius is slightly SMALLER — the estimate is conservative
    for convergence but over-extrapolates a little (REFUTE F6)."""
    r = float(kappa) / (1.0 + float(kappa))
    return r * r


def smooth_particle_field(x, g, grid_min, dx: float, dims, iters: int = 20,
                          kappa: float = 4.0, cheb_rho: float = 0.0,
                          cheb_gamma: float = 0.9, cheb_delay: int = 2,
                          rescale: bool = True):
    """Screened-diffusion smooth of per-particle fields g (N,K) at positions x (N,3).

    Solves (I + kappa*L) u = g_hat on the node grid (g_hat = CIC-averaged g) with red-black
    Gauss-Seidel sweeps, then gathers back to the particles. iters=0 returns g unchanged.
    Empty nodes start at zero and are FILLED by the sweeps — that is the propagation.
    cheb_rho > 0 enables Chebyshev acceleration of the sweep (see module docstring);
    cheb_delay sweeps run plain first so the extrapolation starts from a smooth iterate."""
    if iters <= 0:
        return g
    nx, ny, nz = dims
    dev = g.device
    K = g.shape[1]
    rel = (x - grid_min) / dx
    base = torch.floor(rel).long()
    frac = rel - base.float()

    num = torch.zeros(nx * ny * nz, K, device=dev, dtype=g.dtype)
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
    u = (num / den.clamp_min(1e-12).unsqueeze(1)).reshape(nx, ny, nz, K)
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

    def sweep(f):
        for mask in (red, ~red):
            f = torch.where(mask, (ghat + kappa * avg6(f)) / (1.0 + kappa), f)
        return f

    # REFUTE F6 (2026-09-15): below ~8 sweeps the accelerated iterate is WORSE than the
    # plain one (warm-up + extrapolation transient: kappa 20, 4 sweeps 0.87 vs 0.48);
    # acceleration is therefore applied only when iters >= cheb_min_iters
    cheb_min_iters = 8
    if cheb_rho <= 0.0 or iters < cheb_min_iters:
        for _ in range(iters):
            u = sweep(u)
    else:
        rho2 = float(cheb_rho) ** 2
        u_prev = u
        omega = 1.0
        for k in range(iters):
            u_hat = sweep(u)
            if k < cheb_delay:                       # warm-up: plain sweeps
                u_prev, u = u, u_hat
                continue
            # Wang 2015 recursion restarted at the first accelerated step:
            # omega_1 = 1 (a plain step), omega_2 = 2/(2-rho^2), omega_{k+1} = 4/(4-rho^2 omega_k)
            if k == cheb_delay:
                omega = 1.0
            elif k == cheb_delay + 1:
                omega = 2.0 / (2.0 - rho2)
            else:
                omega = 4.0 / (4.0 - rho2 * omega)
            u_next = omega * (cheb_gamma * (u_hat - u) + u - u_prev) + u_prev
            u_prev, u = u, u_next

    uf = u.reshape(-1, K)
    out = torch.zeros_like(g)
    for idx, w in corners:
        out += uf[idx] * w.unsqueeze(1)

    if not rescale:
        return out
    # preserve the raw field's global magnitude (a preconditioner reshapes, it must not
    # silently rescale the step — the line search calibrates step size on norms)
    nr = g.norm()
    ns = out.norm().clamp_min(1e-30)
    return out * (nr / ns)


def grid_residual(x, g, u_particles, grid_min, dx: float, dims, kappa: float = 4.0):
    """Diagnostic: ||(I + kappa L) u - g_hat|| on the node grid for a particle field u
    re-scattered from the particles. NOTE: the gather->scatter round trip is lossy, so
    this floors at the interpolation error after ~2 sweeps (measured 17.4 for a 6k/16^3
    white field at kappa 4) and cannot rank iterations; compare against a converged
    solution instead (tests/test_chebyshev_smooth.py)."""
    nx, ny, nz = dims
    K = g.shape[1]

    def scatter(f):
        rel = (x - grid_min) / dx
        base = torch.floor(rel).long()
        frac = rel - base.float()
        num = torch.zeros(nx * ny * nz, K, device=g.device, dtype=g.dtype)
        den = torch.zeros(nx * ny * nz, device=g.device, dtype=g.dtype)
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
                    num.index_add_(0, idx, f * w.unsqueeze(1))
                    den.index_add_(0, idx, w)
        v = (num / den.clamp_min(1e-12).unsqueeze(1)).reshape(nx, ny, nz, K)
        return torch.where(den.reshape(nx, ny, nz, 1) > 1e-12, v, torch.zeros_like(v))

    ghat, u = scatter(g), scatter(u_particles)
    s = torch.zeros_like(u)
    s[1:] += u[:-1]; s[:-1] += u[1:]
    s[:, 1:] += u[:, :-1]; s[:, :-1] += u[:, 1:]
    s[:, :, 1:] += u[:, :, :-1]; s[:, :, :-1] += u[:, :, 1:]
    lap = u - s / 6.0
    return float((u + kappa * lap - ghat).norm())
