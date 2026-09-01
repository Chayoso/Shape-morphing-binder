"""Quasi-static grid-level block descent — the VBD idea on MPM's grid (docs/method.md §7).

Each commit solves, for a grid displacement field u on the ACTIVE nodes,

  u* = argmin_u  E(u) = sum_p V_p psi_SNH(F_e(u)) + D_vol(x(u)) + lam_R D_render(x(u))
                        + w_box leash(x(u))

with  x_p(u) = x_p + sum_g w_gp u_g            (CIC interpolation, stencil FROZEN at
      F_e(u) = (I + grad_u(x_p)) F_e0          commit start = total-Lagrangian per commit,
      grad_u(x_p) = sum_g u_g (grad w_gp)^T     rebinned between commits, like MPM)

psi_SNH = stable Neo-Hookean (Smith et al. 2018 — the energy VBD itself uses): SVD-free,
defined for all J, autograd-friendly. F_e0 = F·Fp^-1 at commit start (plasticity outside).

Solver = VBD transplanted from mesh vertices to grid nodes: 2-color (parity) blocks,
per-node diagonal stiffness preconditioning, damped steps with per-color backtracking on
the TOTAL energy, stop at |grad E| <= tol·|grad E_0|. The rotation of the incremental map
costs nothing (quasi-static: no momentum, rest state by construction — the w_kin term of
the dynamic path has no analogue here because there is nothing to bring to rest).

Differentiability: the morph itself needs none (the solve IS the optimisation); the
material/system-ID channel would use the IFT adjoint validated in
scripts/probe_gs_differentiability.py (unrolled 0.1% / IFT 0.04% vs FD at convergence).
"""
from __future__ import annotations

import numpy as np
import torch


def psi_snh(Fe, lam, mu):
    """Stable Neo-Hookean (Smith 2018, simplified): mu/2 (I_C - 3) - mu (J - 1)
    + lam/2 (J - 1)^2. Defined for all J (no log), SPD-friendly around identity."""
    Ic = Fe.pow(2).sum((-2, -1))
    J = torch.det(Fe)
    return 0.5 * mu * (Ic - 3.0) - mu * (J - 1.0) + 0.5 * lam * (J - 1.0) ** 2


class QuasiStaticGrid:
    """One commit's frozen-stencil grid problem. Rebuild (rebind) each commit."""

    def __init__(self, x, grid_min, dx: float, dims, young, poisson, device="cuda",
                 w_min=1e-2, node_cap=0.5):
        nx, ny, nz = dims
        self.dims, self.dx, self.dev = dims, float(dx), device
        x = torch.as_tensor(np.ascontiguousarray(x, np.float32), device=device)
        self.x0 = x
        N = len(x)
        gm = torch.as_tensor(np.asarray(grid_min, np.float32), device=device)
        rel = (x - gm) / dx
        base = torch.floor(rel).long()
        frac = rel - base.float()
        idxs, ws, gws = [], [], []
        for ox in (0, 1):
            sx = 1.0 if ox else -1.0
            wx = frac[:, 0] if ox else 1 - frac[:, 0]
            for oy in (0, 1):
                sy = 1.0 if oy else -1.0
                wy = frac[:, 1] if oy else 1 - frac[:, 1]
                for oz in (0, 1):
                    sz = 1.0 if oz else -1.0
                    wz = frac[:, 2] if oz else 1 - frac[:, 2]
                    ii = (base[:, 0] + ox).clamp(0, nx - 1)
                    jj = (base[:, 1] + oy).clamp(0, ny - 1)
                    kk = (base[:, 2] + oz).clamp(0, nz - 1)
                    idxs.append((ii * ny + jj) * nz + kk)
                    ws.append(wx * wy * wz)
                    # grad of the trilinear weight w.r.t. particle position
                    gws.append(torch.stack([sx * wy * wz, wx * sy * wz, wx * wy * sz],
                                           1) / dx)
        self.idx = torch.stack(idxs, 1)              # (N,8)
        self.w = torch.stack(ws, 1)                  # (N,8)
        self.gw = torch.stack(gws, 1)                # (N,8,3)

        # active nodes: require ACCUMULATED interpolation weight >= w_min — a node touched
        # only by near-zero corner weights has a near-zero diagonal, and one such node's
        # d = g/diag poisons its whole color's candidate (the classic MPM small-node-mass
        # pathology, Steffen et al. 2008; measured here: total solver freeze). Excluded
        # nodes keep u = 0; their kinematic contribution is O(w_min).
        wsum = torch.zeros(nx * ny * nz, device=device)
        wsum.index_add_(0, self.idx.reshape(-1), self.w.reshape(-1))
        act = wsum >= w_min
        act[self.idx.reshape(-1)[self.w.reshape(-1) >= 0.5]] = True   # keep dominant nodes
        self.active = act.nonzero().squeeze(1)       # (A,)
        remap = torch.full((nx * ny * nz,), -1, dtype=torch.long, device=device)
        remap[self.active] = torch.arange(len(self.active), device=device)
        cidx = remap[self.idx]                       # (N,8); -1 = pinned (u=0)
        self.pinned = cidx < 0
        self.cidx = cidx.clamp_min(0)
        self.A = len(self.active)
        self.node_cap = float(node_cap) * dx         # per-node trust radius

        # 8-color (per-axis parity) blocks: two nodes couple iff some particle's 2^3 CIC
        # stencil touches both, i.e. iff they differ by <=1 cell per axis — same-parity
        # nodes therefore NEVER couple, and each color's block update is a true decoupled
        # Gauss-Seidel step. (Round-2 lesson: (i+j+k)%2 parity does NOT decouple this
        # stencil — intra-color conflicts forced constant backtracking and the step never
        # grew; sweeps hit the cap every commit with move ~ 3e-3.)
        lin = self.active
        i = lin // (ny * nz); j = (lin // nz) % ny; k = lin % nz
        self.color_id = (i % 2) * 4 + (j % 2) * 2 + (k % 2)
        lam_l, mu_l = (young * poisson / ((1 + poisson) * (1 - 2 * poisson)),
                       young / (2 * (1 + poisson)))
        self.lam_l, self.mu_l = float(lam_l), float(mu_l)
        Vp = (dx ** 3) / max(1.0, N / max(1.0, float(act.sum())))   # ~uniform ppc estimate
        self.Vp = float(Vp)
        diag = torch.zeros(self.A, device=device)
        contrib = (self.Vp * (2 * self.mu_l + self.lam_l) * self.gw.pow(2).sum(-1))
        contrib = torch.where(self.pinned, torch.zeros_like(contrib), contrib)
        diag.index_add_(0, self.cidx.reshape(-1), contrib.reshape(-1))
        # RELATIVE floor: an absolute epsilon floor let fringe nodes take 1e8-scale steps
        self.diag = diag.clamp_min(1e-3 * diag.median()).unsqueeze(1)   # (A,1)

    def kinematics(self, u):
        """u (A,3) -> particle displacement (N,3) and incremental map A_p = I + grad u (N,3,3)."""
        un = u[self.cidx]                                            # (N,8,3)
        un = torch.where(self.pinned.unsqueeze(-1), torch.zeros_like(un), un)
        disp = (un * self.w.unsqueeze(-1)).sum(1)
        gradu = torch.einsum("nkd,nkg->ndg", un, self.gw)            # du_d/dx_g
        Ap = torch.eye(3, device=u.device).expand(len(disp), 3, 3) + gradu
        return disp, Ap

    def elastic(self, u, Fe0):
        _, Ap = self.kinematics(u)
        Fe = Ap @ Fe0
        return self.Vp * psi_snh(Fe, self.lam_l, self.mu_l).sum(), Fe

    def solve(self, energy_fn, sweeps=60, tol=5e-3, step=0.9, ls=4):
        """Colored damped block descent on E(u). energy_fn(u) -> scalar.

        The diagonal preconditioner captures the ELASTIC curvature; the data terms'
        scale is unknown a priori, so the global step is trust-region-like: a clean
        sweep (every color accepted at first try) GROWS it 1.5x, backtracking shrinks
        locally — the step self-calibrates across arbitrary stiffness/data ratios while
        the per-color energy check keeps descent monotone.
        Returns (u*, info) with the energy trace and the gradient-gate verdict."""
        u = torch.zeros(self.A, 3, device=self.dev, requires_grad=True)
        masks = [(self.color_id == c).unsqueeze(1) for c in range(8)]
        masks = [m for m in masks if bool(m.any())]
        g0 = None
        with torch.no_grad():
            E_prev = float(energy_fn(u))
        trace, gn = [E_prev], float("inf")
        a = step
        for s_i in range(sweeps):
            clean = True
            for m in masks:
                g = torch.autograd.grad(energy_fn(u), u)[0]
                d = (g / self.diag) * m
                # per-node trust radius: one runaway node must not poison the whole
                # color's candidate (acceptance is per color, not per node)
                dn = d.norm(dim=1, keepdim=True)
                d = d * (self.node_cap / dn.clamp_min(1e-30)).clamp(max=1.0)
                at = a
                for _ in range(ls + 1):
                    with torch.no_grad():
                        u_try = (u - at * d).detach()
                        E_new = float(energy_fn(u_try))
                    if np.isfinite(E_new) and E_new <= E_prev:
                        u = u_try.requires_grad_(True)
                        E_prev = E_new
                        break
                    at *= 0.5
                    clean = False
                else:
                    clean = False                      # no acceptable step for this color
            if clean:
                a = min(a * 1.5, step * 1e7)
            gn = float(torch.autograd.grad(energy_fn(u), u)[0].norm())
            if g0 is None:
                g0 = max(gn, 1e-30)
            trace.append(E_prev)
            if gn <= tol * g0:
                break
        return u.detach(), {"sweeps": s_i + 1, "gnorm": gn, "gnorm0": g0, "step_end": a,
                            "converged": bool(gn <= tol * g0), "energy": trace}
