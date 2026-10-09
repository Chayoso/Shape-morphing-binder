"""The window objective, evaluated on the released end state of a rollout.

Physics: the transport energy (grid Sinkhorn divergence to the fixed target + the surface
proximity + the residual drift of the released end), scaled once to D_vol's gradient norm at
the source; the stability term (the released motion, (T dt)^2 mean |v|^2 over the released
steps, unscaled). No regulariser of the control or of the volume: the increment clip, the
released phase and the domain box (a validity constraint of the rollout) bound them.
Render: silhouette + matched shading, weighted by lambda outside this module.
Cleanup (fixed weights, outside the render balance): the spray cleanup (the isolation-gated
W1 pull) and the near-band pull (particles between the sampling berth and one loss cell from
the target). Frozen per window: the transport gate of u, the isolation gate and the
near-band assignment.
"""
from __future__ import annotations

import time

import torch

from ... import gpu
from ...prof import timed
from ...losses.grid_ot import GridSinkhornLoss, grid_transport_displacement
from ...losses.volumetric import (d_nn_band, d_nn_band_current, d_vol_density, d_vol_xu, d_w1,
                                  nn_band_assign, spray_gate)
from ...render.exterior import Tracked, ZhuBridson
from ..render_loss import d_exterior, d_pbr, d_render
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
        self.discs, self.ext_builds = None, 0         # the exterior's discs and how often this window looked for them
        self.ext_apart = 0                            # D127: the discs apart from the body dropped at the last search
        x0 = win.x0
        eps = float(tgt.ldx) ** 2                     # blur: one loss cell
        self.xu = cfg.baseline in ("xu", "xu_spray")
        self.spray_only = cfg.baseline == "xu_spray"     # the baseline with the spray cleanup alone (the ejection guard)
        if self.xu:                                   # the baseline: none of the transport's machinery, no u
            win.set_u_gate(torch.zeros_like(win.lmask))
            self.u_gate_frac, self.horizon = 0.0, cfg.T * prm.dt
            self.knn_ctrl = gpu.knn(x0, 9)[1][:, 1:]
            if tgt.ot_scale is None:                  # the scale of Xu's loss against D_vol's at the source (as ours
                xg = x0.detach().clone().requires_grad_(True)       # transport's): the spray cleanup is divided by it
                gd = torch.autograd.grad(self.dvol_density(xg), xg)[0].norm()
                gx = torch.autograd.grad(d_vol_xu(xg, tgt.m * cfg.xu_mass, *tgt.xu, **cfg.xu_kw()), xg)[0].norm()
                tgt.ot_scale = float(gd / gx.clamp_min(1e-30))
            # the step control's constants in the baseline loss's units (they are D_vol's, converted to the density
            # units ours is scaled to; the baseline's gradient is 1/ot_scale of that): the same step for the same
            # relative gradient as ours takes, as the spray cleanup is divided (D112)
            win.eps_eff /= tgt.ot_scale
            win.target_norm_eff /= tgt.ot_scale
            if self.spray_only:                       # the spray cleanup's isolation gate, frozen per window as ours
                self.m_dt = tgt.m * self.spray_gate(x0)
                self.dt_idx = torch.nonzero(self.m_dt > 0).squeeze(1)
            return
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
        cell = float(cfg.cell_shape) if cfg.cell_shape > 0 else float(prm.dx)    # D137: the shape's cell
        gate = (dn <= float(cfg.layer_gate_ot_cells) * cell).float() * (win.lmask > 0.5).float()
        if cfg.u_off:                                 # D124 ablation: u never acts (its gate zero on every particle)
            gate = torch.zeros_like(gate)
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
        self.knn_ctrl = gpu.knn(x0, 9)[1][:, 1:]      # eight neighbours, for the control-roughness record only
        m_dt = tgt.m * self.spray_gate(x0)
        self.dt_idx = torch.nonzero(m_dt > 0).squeeze(1)
        self.m_dt = m_dt
        # the near band: between the sampling berth and one loss cell from the target (the transport's blur length;
        # farther out the transport owns the particle)
        self.nn_idx, self.nn_elig = nn_band_assign(x0, tgt.knn, tgt.nn_spacing, cfg.nn_berth_k,
                                                   float(tgt.ldx) / float(tgt.nn_spacing))
        self.berth = cfg.nn_berth_k * tgt.nn_spacing
        if not self.berth < float(tgt.ldx):
            raise ValueError("the near band is empty: the sampling berth reaches the loss cell")

    def spray_gate(self, x):
        """The spray cleanup's isolation gate at a state (frozen at the window start for the term, read at the state
        for the selection merit's form): cfg.spray_gate "knn", the ramp dt_iso_lo..hi of the 8th-neighbour distance
        over its median in the particle's own spacing (the code as it was); "grid" (D126), the MPM's own decoupling
        test (losses/volumetric.grid_isolation_gate), no constant."""
        cfg, t = self.cfg, self.tgt
        return spray_gate(x, cfg.spray_gate, cfg.dt_iso_lo, cfg.dt_iso_hi, local=t.body_local, prm=self.win.prm)

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
        lk = vT.pow(2).sum(1).mean()
        if self.xu:                                   # the baseline: Xu et al.'s mass loss alone, on the released end
            zero = xT.sum() * 0.0
            return d_vol_xu(xT, t.m * cfg.xu_mass, *t.xu, **cfg.xu_kw()), lk, zero, zero, zero.detach(), zero
        lstab = self.stability(V)
        with timed("geom"):
            lv = t.ot_scale * (self.transport(xT) + end_drift(vT, self.horizon))
        with timed("render"):
            lsil, lpbr = self.render_terms(xT)
        return lv, lk, lsil + cfg.w_pbr * lpbr, lpbr, lsil.detach(), lstab

    def render_terms(self, xT):
        """(silhouette, shading) of a state: on the particle cloud, or (cfg.render_exterior, D62) on the exterior.
        There the discs follow the particles from the state they were found at (render/exterior.py, Tracked), by
        one Newton step of the field; once a tenth of them has moved more than half a lattice pitch they have left
        their cells and are found again at the state being read (the lattice is fixed in space, so the same surface
        gives the same discs). Found once per window only, they stay behind a surface that the control moves more
        than the field's range, about 1.8 pitches (D62's first run: stopped at 9 windows). `ext_builds` counts the
        searches of the window (a record)."""
        cfg, t = self.cfg, self.tgt
        if t.ext is None:
            return (d_render(xT, t.sils, t.views, cfg.render_res, t.extent, cfg.sil_k, cfg.w_hole, cfg.w_spray),
                    d_pbr(xT, t.shade, t.views, cfg.render_res, t.extent, t.pgmin, t.pdx, t.pdims,
                          cfg.sil_k, cfg.pbr_ambient, t.pblur))
        e = t.ext
        if self.discs is not None:
            p, n, move = self.discs.read(xT)
            if float(torch.quantile(move.detach().abs()[::max(1, len(move) // 50000)], .9)) > .5 * e.h:
                self.discs = None
        if self.discs is None:
            with torch.no_grad(), timed("exterior"):
                self.discs = Tracked(ZhuBridson(xT.detach(), e.pitch, radius=e.radius, offset=e.offset), e.lattice, e.h, e.skin)
                if cfg.render_body_only:          # D127: the body's largest connected disc set alone (the display's rule)
                    self.ext_apart = self.discs.body_only(e.h)
            self.ext_builds += 1
            p, n, move = self.discs.read(xT)
        return d_exterior(p, n, e.sils, e.shade, t.views, cfg.render_res, t.extent, cfg.sil_k, cfg.w_hole,
                          cfg.w_spray, cfg.pbr_ambient)

    def phys_core(self, e):
        """The physics objective without the cleanup terms, from an Eval: lambda's reference and the direction
        PCGrad protects: the scaled geometry of the released end (transport, surface proximity, residual drift)
        and the released motion. The three legacy regularisers are gone (R14, R14b): the control magnitude and
        the control smoothness were 1e-10 and 1e-8 of the merit, and without the volume prior det F stays where
        it was (minimum 0.90, quantiles 0.987 to 1.015)."""
        return e.lv + e.lstab

    def cleanup(self, xT, common_geometry=False):
        """Fixed-weight one-signed cleanup (not lambda-scaled, not in phys_core): the spray cleanup (the W1 pull of
        isolated particles down the target's distance field, summed on the isolation gate's support) and the near
        band. common_geometry: the form the selection merit reads, a function of the committed state alone: the
        isolation gate and the near band's nearest target points and band are taken at the state itself, not from
        the window's start. Until R13 this form summed the distance over EVERY particle, a dense body-to-target
        distance that made up two fifths of the merit (R12f); it is a different quantity from the spray cleanup
        and is no longer part of the merit (recorded as `merit_w1_gap`)."""
        cfg, t, wu = self.cfg, self.tgt, self.win.wu
        if self.xu:                                   # the baseline: no cleanup, or the spray cleanup alone
            if not self.spray_only:
                return xT.sum() * 0.0
            if common_geometry:
                m_cur = t.m * self.spray_gate(xT)
                return wu * cfg.w_dt * d_w1(xT, m_cur, t.dt3, t.dtgmin, t.dtdx, t.dtdims) / t.ot_scale
            if self.dt_idx.numel() == 0:
                return xT.sum() * 0.0
            return wu * cfg.w_dt / t.ot_scale * d_w1(xT.index_select(0, self.dt_idx),        # in Xu's units
                                                     self.m_dt.index_select(0, self.dt_idx),
                                                     t.dt3, t.dtgmin, t.dtdx, t.dtdims)
        if common_geometry:
            m_cur = t.m * self.spray_gate(xT)
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
        if self.xu:
            return 0.0
        cfg, t, wu = self.cfg, self.tgt, self.win.wu
        ones = torch.ones_like(t.m)
        with torch.no_grad():
            whole = d_nn_band_current(xT, t.m, t.pts, ones, self.berth, t.knn)
            band = d_nn_band_current(xT, t.m, t.pts, ones, self.berth, t.knn, far=float(t.ldx))
        return float(wu * cfg.w_nn * (whole - band))

    def w1_merit_gap(self, xT) -> float:
        """A record: the dense body-to-target distance (the distance field summed over every particle) minus the
        spray cleanup read at the current state, in the merit's units: what the selection merit carried until R13."""
        if self.xu:
            return 0.0
        cfg, t, wu = self.cfg, self.tgt, self.win.wu
        with torch.no_grad():
            whole = d_w1(xT, t.m, t.dt3, t.dtgmin, t.dtdx, t.dtdims)
            gated = d_w1(xT, t.m * self.spray_gate(xT), t.dt3, t.dtgmin, t.dtdx, t.dtdims)
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

    def active_set_record(self, xT, lam_r) -> dict:
        """A record: on the particles each local term acts on at a committed state (the near band's: eligible
        and beyond the berth; the spray cleanup's: a non-zero gradient), the rms position gradient of that term,
        of the scaled transport, of the surface term, of the weighted render term, of the other local term and
        of the sum of the three non-local ones; the cosines between the local pull and each of them over the set;
        the share of the set's particles on which the local pull opposes that sum. Which term sets the direction
        of those particles, and whether the terms agree, at every N (D9b)."""
        cfg, t, wu = self.cfg, self.tgt, self.win.wu
        x = xT.detach().clone().requires_grad_(True)

        def grad(L):
            g = torch.autograd.grad(L, x, allow_unused=True)[0]
            return torch.zeros_like(x) if g is None else g.detach()

        ot, sup = t.grid_ot, t.support
        saved, ot.support = ot.support, None
        try:
            g_ot = float(t.ot_scale) * grad(ot.state_energy(x, t.m))
        finally:
            ot.support = saved
        g_surf = float(t.ot_scale) * grad(sup.penalty(x)) if sup is not None else torch.zeros_like(x)
        g_near = grad(wu * cfg.w_nn * d_nn_band(x, t.m, t.pts, self.nn_idx, self.nn_elig, self.berth))
        if self.dt_idx.numel() > 0:
            g_spray = grad(wu * cfg.w_dt * d_w1(x.index_select(0, self.dt_idx), self.m_dt.index_select(0, self.dt_idx),
                                                t.dt3, t.dtgmin, t.dtdx, t.dtdims))
        else:
            g_spray = torch.zeros_like(x)
        lsil, lpbr = self.render_terms(x)
        g_rend = float(lam_r) * grad(lsil + cfg.w_pbr * lpbr)
        others = g_ot + g_surf + g_rend
        out, N = {}, x.shape[0]

        def cos(a, b):
            na, nb = float(a.norm()), float(b.norm())
            return float((a * b).sum()) / (na * nb) if na > 0 and nb > 0 else None

        for name, loc, xloc in (("near", g_near, g_spray), ("spray", g_spray, g_near)):
            S = loc.norm(dim=1) > 0
            n = int(S.sum())
            rec = {"n": n, "frac": n / N}
            if n > 0:
                rms = lambda g: float(g[S].pow(2).sum(1).mean().sqrt())
                rec.update(local=rms(loc), ot=rms(g_ot), surf=rms(g_surf), rend=rms(g_rend), xlocal=rms(xloc), others=rms(others),
                           cos_ot=cos(loc[S], g_ot[S]), cos_surf=cos(loc[S], g_surf[S]), cos_rend=cos(loc[S], g_rend[S]),
                           cos_xlocal=cos(loc[S], xloc[S]), cos_others=cos(loc[S], others[S]),
                           opp=float(((loc[S] * others[S]).sum(1) < 0).float().mean()))
            out[name] = rec
        if cfg.term_dump:                   # the same gradients per particle, for the runner's dump
            return {"active_set": out, "term_grads": {"ot": g_ot, "surf": g_surf, "near": g_near,
                                                      "spray": g_spray, "rend": g_rend}}
        return {"active_set": out}

    def control_record(self, dfc) -> dict:
        """A record: the size and the roughness of an expanded control: the mean squared increment of the driven
        half and its mean squared difference from the mean of the eight nearest neighbours (what the removed
        control regularisers charged)."""
        cfg, N = self.cfg, self.win.N
        with torch.no_grad():
            d = dfc[:cfg.T]
            return {"ctrl_mag": float(d.pow(2).sum() / (cfg.T * N)),
                    "ctrl_rough": float((d - d[:, self.knn_ctrl].mean(2)).pow(2).mean())}

    def scalar(self, e, lam_r) -> float:
        """The full objective as a float: phys_core + cleanup + lambda render."""
        with torch.no_grad(), timed("cleanup"):
            L = float(self.phys_core(e) + self.cleanup(e.xT.detach()))
        return L + lam_r * float(e.lr.detach())
