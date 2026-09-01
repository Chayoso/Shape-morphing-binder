"""Local-global solver: global MPM window owns bulk transport, a SURFACE-BAND VBD-style
Gauss-Seidel pass owns the render residual (docs/rationale.md §5, user-proposed).

Why: the render pull is 8-16x surface-concentrated (measured), and the late-run
render-vs-physics conflict (cos = -0.74) is a RIM disagreement. PCGrad projects that
conflict away — its known weakness under strong conflict. Here the rim gets its own
LOCAL propagation instead: band-limited colored Gauss-Seidel (the retired VBD arm's
solver, with all its hardening lessons), anchored by the global state.

Band = occupied grid cells with an empty 6-neighbour (one-cell surface shell); DOFs are
their corner nodes with accumulated CIC weight >= w_min (small-node-mass lesson, R12);
EVERYTHING ELSE IS PINNED — the interior is a Dirichlet anchor supplied by the global
MPM solution, so the local pass cannot undo bulk transport, only re-shape the shell.
Energy = stable-NH elastic (coherence regulariser) + lambda_R * D_render. Quasi-static:
velocities untouched, rest state preserved; det(F) changes flow into the same guards.
"""
from __future__ import annotations

import numpy as np
import torch

from .render_loss import d_render


def _psi_snh(Fe, lam, mu):
    Ic = Fe.pow(2).sum((-2, -1))
    J = torch.det(Fe)
    return 0.5 * mu * (Ic - 3.0) - mu * (J - 1.0) + 0.5 * lam * (J - 1.0) ** 2


class SurfaceLocal:
    """One commit's frozen-stencil surface-band problem (rebind each use)."""

    def __init__(self, x, grid_min, dx: float, dims, young, poisson, device="cuda",
                 w_min=1e-2, node_cap=0.5):
        nx, ny, nz = dims
        self.dims, self.dx, self.dev = dims, float(dx), device
        xt = torch.as_tensor(np.ascontiguousarray(x, np.float32), device=device)
        self.x0 = xt
        N = len(xt)
        gm = torch.as_tensor(np.asarray(grid_min, np.float32), device=device)
        rel = (xt - gm) / dx
        base = torch.floor(rel).long()
        frac = rel - base.float()
        idxs, ws, gws = [], [], []
        cell = (base[:, 0].clamp(0, nx - 1) * ny + base[:, 1].clamp(0, ny - 1)) * nz \
            + base[:, 2].clamp(0, nz - 1)
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
                    gws.append(torch.stack([sx * wy * wz, wx * sy * wz, wx * wy * sz],
                                           1) / dx)
        self.idx = torch.stack(idxs, 1)
        self.w = torch.stack(ws, 1)
        self.gw = torch.stack(gws, 1)

        # ---- surface band: occupied cells with an empty 6-neighbour ----
        ncell = nx * ny * nz
        occ_cnt = torch.zeros(ncell, device=device)
        occ_cnt.index_add_(0, cell, torch.ones_like(cell, dtype=occ_cnt.dtype))
        occ = (occ_cnt > 0).reshape(nx, ny, nz)
        n_occ = torch.zeros(nx, ny, nz, dtype=torch.float32, device=device)
        o = occ.float()
        n_occ[1:] += o[:-1]; n_occ[:-1] += o[1:]
        n_occ[:, 1:] += o[:, :-1]; n_occ[:, :-1] += o[:, 1:]
        n_occ[:, :, 1:] += o[:, :, :-1]; n_occ[:, :, :-1] += o[:, :, 1:]
        surf_cell = occ & (n_occ < 6)                      # boundary incl. domain edge
        # nodes of surface cells (8 corners), weight-thresholded (R12)
        sc = surf_cell.reshape(-1).nonzero().squeeze(1)
        ci = sc // (ny * nz); cj = (sc // nz) % ny; ck = sc % nz
        cand = torch.zeros(ncell, dtype=torch.bool, device=device)
        for ox in (0, 1):
            for oy in (0, 1):
                for oz in (0, 1):
                    cand[((ci + ox).clamp(0, nx - 1) * ny + (cj + oy).clamp(0, ny - 1))
                         * nz + (ck + oz).clamp(0, nz - 1)] = True
        wsum = torch.zeros(ncell, device=device)
        wsum.index_add_(0, self.idx.reshape(-1), self.w.reshape(-1))
        act = cand & (wsum >= w_min)
        self.active = act.nonzero().squeeze(1)
        remap = torch.full((ncell,), -1, dtype=torch.long, device=device)
        remap[self.active] = torch.arange(len(self.active), device=device)
        cidx = remap[self.idx]
        self.pinned = cidx < 0                              # interior = Dirichlet anchor
        self.cidx = cidx.clamp_min(0)
        self.A = len(self.active)
        self.node_cap = float(node_cap) * dx
        # particles with any active corner: the only ones the band can move
        self.band_p = (~self.pinned).any(1)

        lam_l = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        mu_l = young / (2 * (1 + poisson))
        self.lam_l, self.mu_l = float(lam_l), float(mu_l)
        Vp = (dx ** 3) / max(1.0, N / max(1.0, float((occ_cnt > 0).sum())))
        self.Vp = float(Vp)
        if self.A:
            diag = torch.zeros(self.A, device=device)
            contrib = self.Vp * (2 * mu_l + lam_l) * self.gw.pow(2).sum(-1)
            contrib = torch.where(self.pinned, torch.zeros_like(contrib), contrib)
            diag.index_add_(0, self.cidx.reshape(-1), contrib.reshape(-1))
            self.diag = diag.clamp_min(1e-3 * diag.median()).unsqueeze(1)
            lin = self.active
            i = lin // (ny * nz); j = (lin // nz) % ny; k = lin % nz
            self.color_id = (i % 2) * 4 + (j % 2) * 2 + (k % 2)

    def kinematics(self, u):
        un = u[self.cidx]
        un = torch.where(self.pinned.unsqueeze(-1), torch.zeros_like(un), un)
        disp = (un * self.w.unsqueeze(-1)).sum(1)
        gradu = torch.einsum("nkd,nkg->ndg", un, self.gw)
        Ap = torch.eye(3, device=u.device).expand(len(disp), 3, 3) + gradu
        return disp, Ap

    def solve(self, energy_fn, sweeps=10, step=0.9):
        """8-color exact-line-search block descent (the retired VBD solver's recipe)."""
        u = torch.zeros(self.A, 3, device=self.dev, requires_grad=True)
        masks = [(self.color_id == c).unsqueeze(1) for c in range(8)]
        masks = [m for m in masks if bool(m.any())]
        with torch.no_grad():
            E_prev = float(energy_fn(u))
        a, E0 = step, E_prev
        for _ in range(sweeps):
            for m in masks:
                g = torch.autograd.grad(energy_fn(u), u)[0]
                d = (g / self.diag) * m
                dn = d.norm(dim=1, keepdim=True)
                d = d * (self.node_cap / dn.clamp_min(1e-30)).clamp(max=1.0)
                gd = float((g * d).sum())
                if gd <= 1e-30:
                    continue
                with torch.no_grad():
                    E1 = float(energy_fn((u - a * d).detach()))
                cands = [(E1, a)] if np.isfinite(E1) else []
                c = (E1 - E_prev + a * gd) / (a * a)
                if np.isfinite(c) and c > 1e-30:
                    t = min(gd / (2 * c), 20 * a)
                    if abs(t - a) > 0.05 * a:
                        with torch.no_grad():
                            Et = float(energy_fn((u - t * d).detach()))
                        if np.isfinite(Et):
                            cands.append((Et, t))
                best = min(cands) if cands else None
                if best is not None and best[0] <= E_prev:
                    E_prev = best[0]
                    u = (u - best[1] * d).detach().requires_grad_(True)
                    a = min(max(0.5 * a + 0.5 * best[1], 1e-3 * step), step * 1e7)
                else:
                    a *= 0.5
        return u.detach(), {"E0": E0, "E1": E_prev}


def surface_local_pass(x, F, Fp, tgt, cfg, lam_r, prm):
    """Run one band-limited surface correction. Returns (x', F', tele) or None if the
    band is empty / lambda inactive. Elastic uses F_e0 = F Fp^-1 frozen at entry."""
    if lam_r <= 0 or cfg.lg_sweeps <= 0:
        return None
    dev = cfg.device
    sl = SurfaceLocal(x, prm.grid_min, prm.dx, (prm.nx, prm.ny, prm.nz),
                      cfg.lg_young, cfg.poisson, dev)
    if sl.A == 0:
        return None
    Fe0 = torch.as_tensor(
        np.einsum("nij,njk->nik", F, np.linalg.inv(Fp)).astype(np.float32), device=dev)

    def energy(u):
        disp, Ap = sl.kinematics(u)
        xn = sl.x0 + disp
        E_el = sl.Vp * _psi_snh(Ap @ Fe0, sl.lam_l, sl.mu_l)[sl.band_p].sum()
        lr = d_render(xn, tgt.sils, tgt.views, cfg.render_res, tgt.extent,
                      cfg.sil_k, cfg.w_hole, cfg.w_spray)
        return E_el + lam_r * lr

    u, info = sl.solve(energy, sweeps=cfg.lg_sweeps)
    with torch.no_grad():
        disp, Ap = sl.kinematics(u)
        x_new = (sl.x0 + disp).cpu().numpy().astype(np.float32)
        F_new = Ap.cpu().numpy().astype(np.float32) @ F
    tele = {"lg_E0": info["E0"], "lg_E1": info["E1"], "lg_nodes": int(sl.A),
            "lg_move": float(np.linalg.norm(x_new - x, axis=1).mean())}
    return x_new, F_new, tele
