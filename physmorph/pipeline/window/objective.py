"""The window objective, evaluated on the released end state of a rollout.

Physics: the transport energy (grid Sinkhorn divergence to the fixed target + the surface
proximity + the residual drift of the released end), scaled once to D_vol's gradient norm at
the source; the stability term (the released motion, (T dt)^2 mean |v|^2 over the released
steps, unscaled); control magnitude and smoothness; the (J-1) log J volume prior. The domain
box is a validity constraint of the rollout, not a term.
Render: silhouette + matched shading, weighted by lambda outside this module.
Cleanup (fixed weights, outside the render balance): the isolation-gated W1 pull and the
near-band pull (particles between the sampling berth and one loss cell from the target). Frozen per window: the transport gate of u, the isolation gate, the
near-band assignment and the control-smoothness neighbours.
"""
from __future__ import annotations

import time

import torch

from ... import gpu
from ...losses.grid_ot import GridSinkhornLoss, grid_transport_displacement
from ...losses.volumetric import (d_nn_band, d_nn_band_current, d_vol_density, d_w1,
                                  isolation_gate, nn_band_assign)
from ..render_loss import d_pbr, d_render
from .setup import Window


def velocity_variance(V: torch.Tensor, split: int) -> torch.Tensor:
    """A record (2026-09-30: no longer in the objective). Driven phase: fluctuation about its mean;
    released phase: all motion."""
    return ((V[:split] - V[:split].mean(0)).square().sum()
            + V[split:].square().sum()) / (V.shape[0] * V.shape[1])


def released_motion(V: torch.Tensor, split: int, horizon: float) -> torch.Tensor:
    """The stability term: horizon^2 x the mean over the released steps (from `split`) and the particles of |v|^2,
    in length^2. Invariant under v -> 2v with horizon -> horizon / 2."""
    return horizon ** 2 * V[split:].square().sum(2).mean()


def end_drift(vT: torch.Tensor, horizon: float) -> torch.Tensor:
    """The residual drift of the released end: horizon^2 x the mean over the particles of |v_T|^2, the squared
    displacement the end velocity would add over one more horizon. Part of the geometry energy (alone it does not
    keep a run sound, R11b; without it the released end stays faster, R11d-s)."""
    return horizon ** 2 * vT.square().sum(1).mean()


class Objective:
    def __init__(self, win: Window):
        cfg, tgt, prm = win.cfg, win.tgt, win.prm
        self.win, self.cfg, self.tgt = win, cfg, tgt
        x0 = win.x0
        eps = float(tgt.ldx) ** 2                     # blur: one loss cell
        # the transport gate of u: u acts only on layer particles whose remaining transport
        # (to the label-free transport image of the window start) is within
        # layer_gate_ot_cells MPM cells; farther off, u's per-particle step rides a moving
        # surface. Recomputed at every window start.
        t0 = time.perf_counter()
        g_grid, g_dx, g_dims = tgt.gate if tgt.gate is not None else (tgt.grid, tgt.ldx, tgt.ldims)
        disp = grid_transport_displacement(x0, tgt.m, g_grid, tgt.lgmin, g_dx, g_dims,
                                           eps=float(g_dx) ** 2, iters=cfg.ot_iters, tol=cfg.ot_tol)
        dn = disp.norm(dim=1)
        print(f"[win] fixed grid transport: mean |d|={float(dn.mean()):.3g} wu", flush=True)
        gate = (dn <= float(cfg.layer_gate_ot_cells) * float(prm.dx)).float() * (win.lmask > 0.5).float()
        win.set_u_gate(gate)
        on_layer = win.lmask > 0.5
        self.u_gate_frac = float(gate[on_layer].mean()) if bool(on_layer.any()) else 0.0
        print(f"[layer] u transport gate: {self.u_gate_frac * 100:.1f} % of the layer within "
              f"{cfg.layer_gate_ot_cells:g} cell(s) of its OT image", flush=True)
        torch.cuda.synchronize()
        print(f"[win] OT spatial quadrature: {time.perf_counter() - t0:.2f}s", flush=True)
        if tgt.grid_ot is None:
            tgt.grid_ot = GridSinkhornLoss(tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims, eps=eps,
                                           iters=cfg.ot_iters, tol=min(cfg.ot_tol, 1e-3),
                                           mass_total=float(tgt.m.sum()), cuda_blocks=True,
                                           support=tgt.support)
        self.horizon = cfg.T * prm.dt
        if tgt.ot_scale is None:
            # one scale for the run: the transport term's gradient norm equals D_vol's at the source
            xg = x0.detach().clone().requires_grad_(True)
            gd = torch.autograd.grad(self.dvol_density(xg), xg)[0].norm()
            gt = torch.autograd.grad(self.transport(xg), xg)[0].norm()
            tgt.ot_scale = float(gd / gt.clamp_min(1e-30))
        # frozen per window
        self.knn_creg = gpu.knn(x0, cfg.creg_k + 1)[1][:, 1:]
        m_dt = tgt.m * isolation_gate(x0, cfg.dt_iso_lo, cfg.dt_iso_hi)
        self.dt_idx = torch.nonzero(m_dt > 0).squeeze(1)
        self.m_dt = m_dt
        # the near band: between the sampling berth and one loss cell from the target (the transport's blur length;
        # farther out the transport owns the particle)
        self.nn_idx, self.nn_elig = nn_band_assign(x0, tgt.knn, tgt.nn_spacing, cfg.nn_berth_k,
                                                   float(tgt.ldx) / float(tgt.nn_spacing))
        self.berth = cfg.nn_berth_k * tgt.nn_spacing
        if not self.berth < float(tgt.ldx):
            raise ValueError("the near band is empty: the sampling berth reaches the loss cell")

    # ---- terms ----
    def dvol_density(self, xT):
        t = self.tgt
        return d_vol_density(xT, t.m, t.grid, t.lgmin, t.ldx, t.ldims, t.m_ref, t.n_support)

    def transport(self, xT):
        """The geometry energy of the released end state: the transport divergence plus the fine term."""
        return self.tgt.grid_ot.state_energy(xT, self.tgt.m)

    def stability(self, V):
        """The stability term: the released motion (T dt)^2 x the mean over the released steps and particles of
        |v|^2, length^2, outside the transport's ot_scale. Zero for a body at rest after the release; a release
        that oscillates and comes to rest only at its end pays as a constant one of the same speed. With the
        residual drift of the released end (inside the geometry energy, see losses) it replaces the end kinetic
        energy w_kin |v_T|^2 (5) and the velocity variance w_kin_var (200). What each piece does was measured:
        without the released motion six gallery meshes stop at 8-11 windows still moving (R11b; R11c: neither the
        end kinetic energy nor the driven fluctuation prevents it); without the drift the runs are sound but the
        released end is 1.4x faster at 40k and 1.6-2.9x at 300k and small meshes lose thin coverage (R11d, R11d-s,
        R11e). Magnitude: inside ot_scale (R11) the released motion was 3-6x weaker than the legacy 100 wu and runs
        lost; outside, (T dt)^2 = 6.96e-3 (dt = 0.00417 at every N) equals the legacy 100 wu = 6.5e-3 to 7.0e-3 at
        40k (unit_ratio 1.43e4 to 1.55e4) without a constant, and is 2.2x it at 300k (unit_ratio 3.13e4)."""
        return released_motion(V, self.cfg.T, self.horizon)

    def losses(self, xT, FT, vT, V):
        """(lv, lk, lr, lpbr, d_sil, lstab): the scaled geometry of the released end (the transport energy plus
        its residual drift, (T dt)^2 mean |v_T|^2: the squared displacement the end velocity would add over one
        more horizon, length^2 like the transport), the end kinetic energy (a record), render (silhouette + w_pbr
        shading), shading, the silhouette alone (a tensor, for the record) and the stability term (added in
        phys_core)."""
        cfg, t = self.cfg, self.tgt
        lstab = self.stability(V)
        lv = t.ot_scale * (self.transport(xT) + end_drift(vT, self.horizon))
        lk = vT.pow(2).sum(1).mean()
        lsil = d_render(xT, t.sils, t.views, cfg.render_res, t.extent, cfg.sil_k, cfg.w_hole,
                        cfg.w_spray)
        lpbr = d_pbr(xT, t.shade, t.views, cfg.render_res, t.extent, t.pgmin, t.pdx, t.pdims,
                     cfg.sil_k, cfg.pbr_ambient, t.pblur)
        return lv, lk, lsil + cfg.w_pbr * lpbr, lpbr, lsil.detach(), lstab

    def phys_core(self, e):
        """The physics objective without the cleanup terms, from an Eval: lambda's reference and the
        direction PCGrad protects. Only the driven half of the control is costed."""
        cfg, wu, N = self.cfg, self.win.wu, self.win.N
        dfc, xT, FT = e.dfc[:cfg.T], e.xT, e.FT
        L = e.lv + e.lstab + wu * cfg.w_ctrl * dfc.pow(2).sum() / (cfg.T * N)
        L = L + wu * cfg.w_creg * (dfc - dfc[:, self.knn_creg].mean(2)).pow(2).mean()
        J = torch.linalg.det(FT.view(-1, 3, 3))
        L = L + wu * cfg.w_jvol * ((J - 1.0) * torch.log(J.clamp_min(1e-6))).mean()
        return L

    def cleanup(self, xT, common_geometry=False):
        """Fixed-weight one-signed cleanup (not lambda-scaled, not in phys_core): the spray cleanup (the W1 pull of
        isolated particles down the target's distance field, summed on the isolation gate's support) and the near
        band. common_geometry: the form the selection merit reads, a function of the committed state alone: the
        isolation gate and the near band's nearest target points and band are taken at the state itself, not from
        the window's start. Until R13 this form summed the distance over EVERY particle, a dense body-to-target
        distance that made up two fifths of the merit (R12f); it is a different quantity from the spray cleanup
        and is no longer part of the merit (recorded as `merit_w1_gap`)."""
        cfg, t, wu = self.cfg, self.tgt, self.win.wu
        if common_geometry:
            m_cur = t.m * isolation_gate(xT, cfg.dt_iso_lo, cfg.dt_iso_hi)
            L = wu * cfg.w_dt * d_w1(xT, m_cur, t.dt3, t.dtgmin, t.dtdx, t.dtdims)
            return L + wu * cfg.w_nn * d_nn_band_current(xT, t.m, t.pts, torch.ones_like(t.m),
                                                         self.berth, t.knn, far=float(t.ldx))
        if self.dt_idx.numel() > 0:
            L = wu * cfg.w_dt * d_w1(xT.index_select(0, self.dt_idx),
                                     self.m_dt.index_select(0, self.dt_idx),
                                     t.dt3, t.dtgmin, t.dtdx, t.dtdims)
        else:
            L = xT.sum() * 0.0                          # an empty gate: zero, still on the graph
        return L + wu * cfg.w_nn * d_nn_band(xT, t.m, t.pts, self.nn_idx, self.nn_elig, self.berth)

    def near_band_far(self, xT) -> float:
        """A record: the near band's value beyond the loss cell at the current state, in the merit's units (what a
        ruler without the band's outer edge would add to the selection merit)."""
        cfg, t, wu = self.cfg, self.tgt, self.win.wu
        ones = torch.ones_like(t.m)
        with torch.no_grad():
            whole = d_nn_band_current(xT, t.m, t.pts, ones, self.berth, t.knn)
            band = d_nn_band_current(xT, t.m, t.pts, ones, self.berth, t.knn, far=float(t.ldx))
        return float(wu * cfg.w_nn * (whole - band))

    def w1_merit_gap(self, xT) -> float:
        """A record: the dense body-to-target distance (the distance field summed over every particle) minus the
        spray cleanup read at the current state, in the merit's units: what the selection merit carried until R13."""
        cfg, t, wu = self.cfg, self.tgt, self.win.wu
        with torch.no_grad():
            whole = d_w1(xT, t.m, t.dt3, t.dtgmin, t.dtdx, t.dtdims)
            gated = d_w1(xT, t.m * isolation_gate(xT, cfg.dt_iso_lo, cfg.dt_iso_hi), t.dt3, t.dtgmin, t.dtdx, t.dtdims)
        return float(wu * cfg.w_dt * (whole - gated))

    def scale_record(self, xT) -> dict:
        """A record: the position-space gradient norms of the two local terms at a committed state, in the
        objective's units, with the number of particles each acts on (the transport's and the surface term's
        norms come from support_record; ot_scale converts them)."""
        cfg, t, wu = self.cfg, self.tgt, self.win.wu
        x = xT.detach().clone().requires_grad_(True)
        out = {"ot_scale": float(t.ot_scale), "n_spray": int(self.dt_idx.numel()), "n_near": int(self.nn_elig.sum())}
        if self.dt_idx.numel() > 0:
            spray = wu * cfg.w_dt * d_w1(x.index_select(0, self.dt_idx), self.m_dt.index_select(0, self.dt_idx),
                                         t.dt3, t.dtgmin, t.dtdx, t.dtdims)
            out["g_spray"] = float(torch.autograd.grad(spray, x)[0].norm())
        else:
            out["g_spray"] = 0.0
        d = (x.detach() - t.pts[self.nn_idx]).norm(dim=1)
        out["n_near_active"] = int(((d > self.berth) & (self.nn_elig > 0)).sum())
        near = wu * cfg.w_nn * d_nn_band(x, t.m, t.pts, self.nn_idx, self.nn_elig, self.berth)
        g = torch.autograd.grad(near, x, allow_unused=True)[0]
        out["g_near"] = 0.0 if g is None else float(g.norm())
        return out

    def scalar(self, e, lam_r) -> float:
        """The full objective as a float: phys_core + cleanup + lambda render."""
        with torch.no_grad():
            L = float(self.phys_core(e) + self.cleanup(e.xT.detach()))
        return L + lam_r * float(e.lr.detach())
