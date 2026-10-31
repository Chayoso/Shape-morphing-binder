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
from ...losses.grid_ot import GridSinkhornLoss, grid_transport_displacement, grid_transport_displacement_separable
from ...losses.volumetric import (d_nn_band, d_nn_band_current, d_vol_density, d_vol_xu, d_w1,
                                  isolation_gate, nn_band_assign)
from ...render.exterior import make_field, tracked
from ..render_loss import d_count, d_exterior, d_exterior_content, d_exterior_masked, d_pbr, d_render
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
        x0 = win.x0
        eps = float(tgt.ldx) ** 2                     # blur: one loss cell
        self.xu = cfg.baseline in ("xu", "xu_spray")
        self.spray_only = cfg.baseline == "xu_spray"     # the baseline with the spray cleanup alone (the ejection guard)
        if self.xu and cfg.gap_split != "off":
            raise ValueError("gap_split reads the transport image, which the Xu baseline does not compute")
        if self.xu:                                   # the baseline: none of the transport's machinery, no u
            win.set_u_gate(torch.zeros_like(win.lmask))
            self.u_gate_frac, self.horizon = 0.0, cfg.T * prm.dt
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
        # D174 (exact): the gate is read on the layer only (zero elsewhere), so only the layer's rows of the transport map
        # are computed; D148's gap sides read every particle's image
        rows = None if cfg.gap_split != "off" else torch.nonzero(win.lmask > 0.5).squeeze(1)
        with timed("gate_ot"):
            disp = grid_transport_displacement(x0, tgt.m, g_grid, tgt.lgmin, g_dx, g_dims,
                                               eps=float(g_dx) ** 2, iters=cfg.ot_iters, tol=cfg.ot_tol, rows=rows)
        dn = disp.norm(dim=1)
        print(f"[win] fixed grid transport: mean |d|={float(dn.mean()):.3g} wu"
              + ("" if rows is None else " (the layer's particles)"), flush=True)
        cell = float(cfg.cell_shape) if cfg.cell_shape > 0 else float(prm.dx)    # D137: the shape's cell
        if rows is None:
            gate = (dn <= float(cfg.layer_gate_ot_cells) * cell).float() * (win.lmask > 0.5).float()
        else:
            gate = torch.zeros_like(win.lmask)
            gate[rows] = (dn <= float(cfg.layer_gate_ot_cells) * cell).float()
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
        if cfg.gap_split != "off":                    # D148: each particle's side point, its debiased transport image:
            if cfg.gap_split == "fine":               # D148b: the loss grid's plan (blur one loss cell)
                with timed("gap_sides"):
                    disp = grid_transport_displacement_separable(x0, tgt.m, tgt.grid_ot)
            q = (x0 + disp).detach().float().contiguous()   # else the u gate's (the gate grid's)
            rep = None
            if tgt.gap_walls is not None:             # D148c: the representative cells, persistent in the gap
                from ...mpm.gap_walls import rep_cells
                wl = tgt.gap_walls
                prev = tgt.gap_rep if tgt.gap_rep is not None else torch.full((len(q),), -1, dtype=torch.long,
                                                                               device=q.device)
                rep = rep_cells(q, prev, wl["lo"], wl["cell"], wl["nc"], wl["occ"], wl["nearest"])
                tgt.gap_rep = rep
            win.set_gap_sides(q, rep)
        self.horizon = cfg.T * prm.dt
        if tgt.ot_scale is None:
            # one scale for the run: the transport term's gradient norm equals D_vol's at the source
            xg = x0.detach().clone().requires_grad_(True)
            gd = torch.autograd.grad(self.dvol_density(xg), xg)[0].norm()
            gt = torch.autograd.grad(self.transport(xg), xg)[0].norm()
            tgt.ot_scale = float(gd / gt.clamp_min(1e-30))
        # frozen per window
        with timed("spray_gate"):
            m_dt = tgt.m * self.spray_gate(x0)
            self.dt_idx = torch.nonzero(m_dt > 0).squeeze(1)
        self.m_dt = m_dt
        # the near band: between the sampling berth and one loss cell from the target (the transport's blur length;
        # farther out the transport owns the particle)
        if getattr(cfg, "placement_at_cell", False):   # D162 (2): the berth is the loss cell, so the band is empty
            self.berth = float(tgt.ldx)
            self.nn_idx = torch.zeros(len(x0), dtype=torch.long, device=x0.device)
            self.nn_elig = torch.zeros(len(x0), device=x0.device)
            return
        with timed("nn_band"):
            self.nn_idx, self.nn_elig = nn_band_assign(x0, tgt.knn, tgt.nn_spacing, cfg.nn_berth_k,
                                                       float(tgt.ldx) / float(tgt.nn_spacing))
        self.berth = cfg.nn_berth_k * tgt.nn_spacing
        if not self.berth < float(tgt.ldx):
            raise ValueError("the near band is empty: the sampling berth reaches the loss cell")

    def spray_gate(self, x):
        """The spray cleanup's isolation gate at a state (frozen at the window start for the term, read at the state
        for the selection merit's form): the ramp dt_iso_lo..hi of the 8th-neighbour distance over its median."""
        return isolation_gate(x, self.cfg.dt_iso_lo, self.cfg.dt_iso_hi,
                              scale=float(self.tgt.ldx) if getattr(self.cfg, "placement_at_cell", False) else None)   # D162 (2)

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
            if getattr(t, "geometry", True):
                lv = t.ot_scale * (self.transport(xT) + end_drift(vT, self.horizon))
            else:                                     # D175: a keyframe's segment, the end drift alone (target-free)
                lv = t.ot_scale * end_drift(vT, self.horizon)
        if not getattr(t, "soundness", True):         # D175c: the segment's merit is the keyframe's alone (on the graph)
            lv, lstab = 0.0 * lv, 0.0 * lstab
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
        self.lr_m = None                              # D178: set by _exterior on the masked rule's path only
        if t.ext is None:
            return (d_render(xT, t.sils, t.views, cfg.render_res, t.extent, cfg.sil_k, cfg.w_hole, cfg.w_spray),
                    d_pbr(xT, t.shade, t.views, cfg.render_res, t.extent, t.pgmin, t.pdx, t.pdims,
                          cfg.sil_k, cfg.pbr_ambient, t.pblur))
        e = t.ext
        if getattr(e, "counts", None) is not None:    # D175b: a keyframe compared on projected counts (the particles)
            if (e.mask_sil is not None and getattr(t, "keyframe", None) is None
                    and getattr(cfg, "render_split", "") in ("masked", "content")):
                # D178 (render_compare column + the masked rule): the count term and its part inside the content mask
                from ..render_loss import count_pictures
                c_k = torch.stack(list(e.counts), 0)
                r = (count_pictures(xT, t.views, cfg.render_res, t.extent) - c_k).pow(2)
                den = c_k.pow(2).mean()
                self.lr_m = (r * e.mask_sil).mean() / den
                return r.mean() / den, xT.sum() * 0.0
            return d_count(xT, e.counts, t.views, cfg.render_res, t.extent), xT.sum() * 0.0
        if e.albedo is not None:                      # D160: the discs' colours from the particles' fixed albedos
            return self._render_terms_albedo(xT, e)
        if self.discs is not None:
            p, n, move = self.discs.read(xT)
            if float(torch.quantile(move.detach().abs()[::max(1, len(move) // 50000)], .9)) > .5 * e.h:
                self.discs = None
        if self.discs is None:
            with torch.no_grad(), timed("exterior"):
                self.discs = tracked(make_field(e.field, xT.detach(), e.pitch, radius=e.radius, offset=e.offset,
                                                device_field=True), e.lattice, e.h, e.skin, e.colour, e.colour_tau)
                if getattr(self.cfg, "render_normal_radius", 0.0):   # D183: the discs' normals at the narrower kernel
                    self.discs.normal_radius = float(self.cfg.render_normal_radius) * e.pitch
            self.ext_builds += 1
            p, n, move = self.discs.read(xT)
        return self._exterior(p, n, e)

    def _exterior(self, p, n, e, c=None):
        """d_exterior's (silhouette, shading) of the discs (c: their albedos). D178 (render_split "masked"): where the
        target carries a content mask and no keyframe segment is open, the same rendering also gives the masked render
        term, self.lr_m = masked silhouette + w_pbr masked shading (the dFc block's extra pull, solve.gradient); else
        self.lr_m is None and the path is d_exterior's."""
        cfg, t = self.cfg, self.tgt
        self.lr_m = None
        split = getattr(cfg, "render_split", "")
        if split in ("content", "appearance") and c is not None and getattr(t, "keyframe", None) is None:
            # D178 (M2): with an appearance target the held term is the appearance part everywhere and the geometry part
            # inside the geometry content mask (render_loss.d_exterior_content); without one M2 is the masked rule below.
            # D178 (M3, "appearance"): the appearance part alone, no geometry term (the geometry the 3D target fixes is
            # written below the cell through u, as Hu); without an appearance target M3 is Hu
            geo = split == "content"
            sil, shd, self.lr_m = d_exterior_content(p, n, e.sils, e.shade, e.mask_sil if geo else None,
                                                     getattr(e, "mask_geo", None) if geo else None,
                                                     t.views, cfg.render_res, t.extent, cfg.sil_k, cfg.w_hole, cfg.w_spray,
                                                     cfg.pbr_ambient, albedo=c, w_pbr=cfg.w_pbr)
            return sil, shd
        if ((e.mask_sil is None and e.mask_shade is None) or getattr(t, "keyframe", None) is not None
                or split not in ("masked", "content")):
            return d_exterior(p, n, e.sils, e.shade, t.views, cfg.render_res, t.extent, cfg.sil_k, cfg.w_hole,
                              cfg.w_spray, cfg.pbr_ambient, albedo=c)
        sil, shd, sil_m, shd_m = d_exterior_masked(p, n, e.sils, e.shade, e.mask_sil, e.mask_shade, t.views,
                                                   cfg.render_res, t.extent, cfg.sil_k, cfg.w_hole, cfg.w_spray,
                                                   cfg.pbr_ambient, albedo=c)
        self.lr_m = sil_m + cfg.w_pbr * shd_m
        return sil, shd

    def _render_terms_albedo(self, xT, e):
        """render_terms with the particles' albedos (D160, appearance): the same discs, found and followed as above;
        each disc's albedo is its particles' under its own kernel weights (Tracked.read), and the shading term
        compares colour pictures. The physics terms never read the albedos."""
        cfg, t = self.cfg, self.tgt
        if self.discs is not None:
            p, n, move, c = self.discs.read(xT, e.albedo, e.albedo_ref)
            if float(torch.quantile(move.detach().abs()[::max(1, len(move) // 50000)], .9)) > .5 * e.h:
                self.discs = None
        if self.discs is None:
            with torch.no_grad(), timed("exterior"):
                self.discs = tracked(make_field(e.field, xT.detach(), e.pitch, radius=e.radius, offset=e.offset,
                                                device_field=True), e.lattice, e.h, e.skin, e.colour, e.colour_tau)
                if getattr(self.cfg, "render_normal_radius", 0.0):   # D183: the discs' normals at the narrower kernel
                    self.discs.normal_radius = float(self.cfg.render_normal_radius) * e.pitch
            self.ext_builds += 1
            p, n, move, c = self.discs.read(xT, e.albedo, e.albedo_ref)
        return self._exterior(p, n, e, c)

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
        distance that made up two fifths of the merit (R12f); it is a different quantity from the spray cleanup."""
        cfg, t, wu = self.cfg, self.tgt, self.win.wu
        if not getattr(t, "geometry", True):          # D175: a keyframe's segment, no pull to the final target
            return xT.sum() * 0.0
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
            knn = gpu.knn(self.win.x0, 9)[1][:, 1:]   # eight neighbours at the window start (only this record reads them)
            return {"ctrl_mag": float(d.pow(2).sum() / (cfg.T * N)),
                    "ctrl_rough": float((d - d[:, knn].mean(2)).pow(2).mean())}

    def scalar(self, e, lam_r) -> float:
        """The full objective as a float: phys_core + cleanup + lambda render."""
        with torch.no_grad(), timed("cleanup"):
            L = float(self.phys_core(e) + self.cleanup(e.xT.detach()))
        return L + lam_r * float(e.lr.detach())
