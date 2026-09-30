"""The window objective, evaluated on the released end state of a rollout.

Physics: the transport energy (grid Sinkhorn divergence to the fixed target + residual
drift, bounded local support), scaled once to D_vol's gradient norm at the source; end
kinetic energy; velocity variance (driven phase) and all motion (released phase); control
magnitude and smoothness; the far-field box leash; the (J-1) log J volume prior.
Render: silhouette + matched shading, weighted by lambda outside this module.
Cleanup (fixed weights, outside the render balance): the isolation-gated W1 pull and the
near-band pull. Frozen per window: the transport gate of u, the isolation gate, the
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
    """The stability term: horizon^2 x the mean over the released steps (from `split`) and the particles of
    |v|^2, in length^2. Invariant under v -> 2v with horizon -> horizon / 2, as the residual drift it replaces."""
    return horizon ** 2 * V[split:].square().sum(2).mean()


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
        self.nn_idx, self.nn_elig = nn_band_assign(x0, tgt.knn, tgt.nn_spacing, cfg.nn_berth_k,
                                                   cfg.nn_far_k)
        self.berth = cfg.nn_berth_k * tgt.nn_spacing

    # ---- terms ----
    def dvol_density(self, xT):
        t = self.tgt
        return d_vol_density(xT, t.m, t.grid, t.lgmin, t.ldx, t.ldims, t.m_ref, t.n_support)

    def transport(self, xT):
        """The geometry energy of the released end state: the transport divergence plus the fine term."""
        return self.tgt.grid_ot.state_energy(xT, self.tgt.m)

    def stability(self, V):
        """The released motion: (T dt)^2 mean over the released steps and particles of |v|^2, length^2 like the
        geometry energy. Zero for a body at rest after the release, the residual drift's value for a constant
        released velocity, and larger for a release that oscillates and comes to rest only at its end. It
        replaces three terms (2026-09-30): the residual drift |T dt v_T|^2 inside the transport energy, the end
        kinetic energy w_kin |v_T|^2 and w_kin_var (the driven fluctuation and the released motion)."""
        return released_motion(V, self.cfg.T, self.horizon)

    def losses(self, xT, FT, vT, V):
        """(lv, lk, lr, lpbr, d_sil, lstab): the scaled geometry with the stability term, the end kinetic
        energy (a record), render (silhouette + w_pbr shading), shading, the silhouette alone (a tensor, for
        the record) and the stability term alone."""
        cfg, t = self.cfg, self.tgt
        lstab = self.stability(V)
        lv = t.ot_scale * (self.transport(xT) + lstab)
        lk = vT.pow(2).sum(1).mean()
        lsil = d_render(xT, t.sils, t.views, cfg.render_res, t.extent, cfg.sil_k, cfg.w_hole,
                        cfg.w_spray)
        lpbr = d_pbr(xT, t.shade, t.views, cfg.render_res, t.extent, t.pgmin, t.pdx, t.pdims,
                     cfg.sil_k, cfg.pbr_ambient, t.pblur)
        return lv, lk, lsil + cfg.w_pbr * lpbr, lpbr, lsil.detach(), lstab

    def phys_core(self, lv, dfc, xT, FT):
        """The physics objective without the cleanup terms: lambda's reference and the
        direction PCGrad protects. dfc: the expanded control; only the driven half is costed."""
        cfg, wu, N = self.cfg, self.win.wu, self.win.N
        dfc = dfc[:cfg.T]
        L = lv + wu * cfg.w_ctrl * dfc.pow(2).sum() / (cfg.T * N)
        L = L + wu * cfg.w_box * torch.clamp(xT.abs() - self.tgt.extent, min=0).pow(2).sum(1).mean()
        L = L + wu * cfg.w_creg * (dfc - dfc[:, self.knn_creg].mean(2)).pow(2).mean()
        J = torch.linalg.det(FT.view(-1, 3, 3))
        L = L + wu * cfg.w_jvol * ((J - 1.0) * torch.log(J.clamp_min(1e-6))).mean()
        return L

    def cleanup(self, xT, common_geometry=False):
        """Fixed-weight one-signed cleanup (not lambda-scaled, not in phys_core). The W1 sum
        runs on the isolation gate's support (the same sum and gradient). common_geometry:
        the ungated W1 and the near band against the CURRENT nearest target points, the
        form every window's selection merit is compared in."""
        cfg, t, wu = self.cfg, self.tgt, self.win.wu
        if common_geometry:
            L = wu * cfg.w_dt * d_w1(xT, t.m, t.dt3, t.dtgmin, t.dtdx, t.dtdims)
            return L + wu * cfg.w_nn * d_nn_band_current(xT, t.m, t.pts, torch.ones_like(t.m),
                                                         self.berth, t.knn)
        if self.dt_idx.numel() > 0:
            L = wu * cfg.w_dt * d_w1(xT.index_select(0, self.dt_idx),
                                     self.m_dt.index_select(0, self.dt_idx),
                                     t.dt3, t.dtgmin, t.dtdx, t.dtdims)
        else:
            L = xT.sum() * 0.0                          # an empty gate: zero, still on the graph
        return L + wu * cfg.w_nn * d_nn_band(xT, t.m, t.pts, self.nn_idx, self.nn_elig, self.berth)

    def scalar(self, lv, lr, lam_r, dfc, xT, FT) -> float:
        """The full objective as a float: phys_core + cleanup + lambda render."""
        with torch.no_grad():
            L = float(self.phys_core(lv, dfc.detach(), xT.detach(), FT.detach()) + self.cleanup(xT.detach()))
        return L + lam_r * float(lr.detach())
