"""D155 (option A; an A/B switch, off by default: cfg.morph_time 0 is the frozen path): the window's target is the point
of the geodesic between the source and the target at the window's end time, for the transport and the render pictures
alike.

D155's measurement on the tag: the physics and the render fall together, but both in the first 0.6-1.6 s, because every
window minimises the distance of its end to the FINAL target (a greedy descent that moves as fast as the step allows),
followed by a long sub-pitch refit. Here the target itself moves: at time t it is the displacement interpolation of the
transport between the source and the target (McCann 1997; the constant-speed geodesic of Benamou and Brenier), read
from the target's side as D147's scene schedule reads it (wip/d147, physmorph/pipeline/scene/schedule.py):
Z_s = Y + (1 - s) D_Y(Y), with Y a sample of the target and D_Y the pipeline's own transport map from the target to the
source (the u gate's operator, losses/grid_ot.grid_transport_displacement: the debiased barycentric map on the MPM
cell's grid, blur one cell), and s = 3u^2 - 2u^3 of u = t / morph_time. The run script sets morph_time to the body's
elastic period, 2 x the source's diagonal / the P-wave speed, which the discretisation already computes (prepare's
[disc] line): no new constant. At s = 1 the target is the run's own pack, point for point (no switch of the points;
the relief and the further draws are its own from there).

Every pack on the way is built by target.build_target from Z_s as the run's target is built from Y: its transport grid,
proximity, cleanup grids and gate, and its render pictures as the mean over the same samples (the run's sample and its
draws, each moved by the map read at its own points), so the render reads the in-between shape as it reads the target
now. The packs carry the run's calibrations (the unit ratios and the transport scale, measured once at the source on
the final target, as the objective measures them at the first window) and the run's state on the target (the settled
step, u's carried moments). Their layer relaxation has no relief (the mesh's relief is the final shape's, D147). Costs
are compared within one target only (run/selection.new_epoch, as for a change of the render pictures): each new pack
opens a selection epoch, so the ramp's windows are never rejected for a merit read against another target, and the
run's stop and delivery read the final target's epoch.

D155b (morph_u_final, --morph_u_gate final): u's transport gate reads the body's arrival at the FINAL target (the tag's
gate: a layer particle within layer_gate_ot_cells MPM cells of its transport image on the final target), not at the
moving one. D155's 40k gate found u's gate open on 93-100 % of the layer from the first window on the moving target
(the body is always within a cell of it), and with it 2-6x the tag's apart particles in windows 3-12, nearly all bumps
of u's relief on the in-between surfaces (u off removed them). The packs on the way also skip the particle cloud's
pictures the run never reads with its render terms on the exterior (SKIP_UNUSED_PICTURES, exact)."""
from __future__ import annotations

import time

import torch

from ... import gpu
from ...losses.grid_ot import GridSinkhornLoss, grid_transport_displacement
from ...losses.volumetric import d_vol_density, target_mass_grid
from ...render.exterior import make_field, tracked
from ..render_loss import d_exterior, d_pbr, d_render
from ..target import build_target

CALIBRATED = ("unit_ratio", "unit_grad_ratio", "ot_scale")        # measured once at the source on the final target
# D155b (exact): the packs on the way do not draw the particle cloud's pictures, which a run with its render terms on the
# exterior never reads (window/objective.render_terms reads tgt.sils / tgt.shade only when tgt.ext is None); a proof
# sets it False to compare the runs bit for bit
SKIP_UNUSED_PICTURES = True
RUN_STATE = ("settled_step", "u_adam", "force_ratio", "gap_rep")    # the run's state kept on the target pack


def ramp(t: float, t_m: float) -> float:
    """D147's smoothstep 3u^2 - 2u^3 of u = t / t_m in [0, 1]: the target's speed is zero at the start and the end."""
    u = min(max(t / t_m, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def transport_ruler(pack, cfg) -> GridSinkhornLoss:
    """A transport loss on the pack's grid, built as window/objective.Objective builds the run's at its first window."""
    return GridSinkhornLoss(pack.grid, pack.lgmin, pack.ldx, pack.ldims, eps=float(pack.ldx) ** 2,
                            iters=cfg.ot_iters, tol=min(cfg.ot_tol, 1e-3), mass_total=float(pack.m.sum()),
                            cuda_blocks=True, support=pack.support)


def transport_scale(pack, ot: GridSinkhornLoss, x0: torch.Tensor) -> float:
    """window/objective.Objective's one scale for the run (the transport term's gradient norm equal to D_vol's) at the
    source on the final target: the first window's target is the source itself here, where both gradients vanish."""
    xg = x0.detach().clone().requires_grad_(True)
    gd = torch.autograd.grad(d_vol_density(xg, pack.m, pack.grid, pack.lgmin, pack.ldx, pack.ldims, pack.m_ref,
                                           pack.n_support), xg)[0].norm()
    xg = x0.detach().clone().requires_grad_(True)
    gt = torch.autograd.grad(ot.state_energy(xg, pack.m), xg)[0].norm()
    return float(gd / gt.clamp_min(1e-30))


class MorphTarget:
    """final: the run's built and calibrated TargetPack (Y = final.pts); source: the source sample (CUDA tensor);
    draws: the further samples of the target its render pictures average (prepare's tgt_draws, or None)."""

    def __init__(self, final, source: torch.Tensor, draws, prm, cfg, log=print):
        self.final, self.prm, self.cfg, self.log = final, prm, cfg, log
        self.t_m = float(cfg.morph_time)
        self.ruler = transport_ruler(final, cfg)      # the final target's transport, for the records on the way
        if final.ot_scale is None:
            final.ot_scale = transport_scale(final, self.ruler, source)
        g_grid, g_dx, g_dims = final.gate if final.gate is not None else (final.grid, final.ldx, final.ldims)
        t0 = time.perf_counter()
        with torch.no_grad():
            src = target_mass_grid(source, torch.ones(len(source), device=source.device), final.lgmin, g_dx, g_dims)
            self.samples = [final.pts] + [gpu.tensor(d) for d in (draws or [])]
            self.D = [grid_transport_displacement(Y, torch.ones(len(Y), device=Y.device), src, final.lgmin, g_dx,
                                                  g_dims, eps=float(g_dx) ** 2, iters=cfg.ot_iters, tol=cfg.ot_tol)
                      for Y in self.samples]
        torch.cuda.synchronize()
        dn = self.D[0].norm(dim=1)
        log(f"[morph] D155: the target moves along the geodesic over {self.t_m:.4f} s (the elastic period); transport "
            f"from the target to the source |D_Y| mean {float(dn.mean()):.4f}, max {float(dn.max()):.4f} wu, at "
            f"{len(self.samples)} sample(s) in {time.perf_counter() - t0:.2f} s; transport scale {final.ot_scale:.6g}")
        self.cur, self.s, self.build_s = final, None, 0.0

    def at(self, t: float):
        """(the pack of the window ending at time t, its s, whether the pack changed)."""
        s = ramp(t, self.t_m)
        if s == self.s:
            return self.cur, s, False
        t0 = time.perf_counter()
        if s >= 1.0:
            new = self.final
        else:
            with torch.no_grad():
                Z = [(Y + (1.0 - s) * D).contiguous() for Y, D in zip(self.samples, self.D)]
                new = build_target(Z[0], self.prm, self.cfg, draws=Z[1:] or None,
                                   particle_pictures=not (SKIP_UNUSED_PICTURES and self.cfg.render_exterior))
            for k in CALIBRATED:
                setattr(new, k, getattr(self.final, k))
            if self.cfg.morph_u_final:                 # D155b: u reads its arrival at the final target, as on the tag
                f = self.final
                new.gate_ref = f.gate if f.gate is not None else (f.grid, f.ldx, f.ldims)
        for k in RUN_STATE:                            # the run's state on the target goes with it
            setattr(new, k, getattr(self.cur, k))
        torch.cuda.synchronize()
        self.build_s = time.perf_counter() - t0
        self.cur, self.s = new, s
        return new, s, True

    @torch.no_grad()
    def final_terms(self, x: torch.Tensor) -> dict:
        """A record: the committed state's transport energy and render terms against the FINAL target (on the way the
        window's own record reads its in-between target), the render on discs found afresh at the state."""
        f, cfg = self.final, self.cfg
        out = {"final_TE": float(self.ruler.state_energy(x, f.m))}
        if f.ext is None:
            sil = d_render(x, f.sils, f.views, cfg.render_res, f.extent, cfg.sil_k, cfg.w_hole, cfg.w_spray)
            pbr = d_pbr(x, f.shade, f.views, cfg.render_res, f.extent, f.pgmin, f.pdx, f.pdims, cfg.sil_k,
                        cfg.pbr_ambient, f.pblur)
        else:
            e = f.ext
            discs = tracked(make_field(e.field, x, e.pitch, radius=e.radius, offset=e.offset, device_field=True),
                            e.lattice, e.h, e.skin)
            p, n, _ = discs.read(x)
            sil, pbr = d_exterior(p, n, e.sils, e.shade, f.views, cfg.render_res, f.extent, cfg.sil_k, cfg.w_hole,
                                  cfg.w_spray, cfg.pbr_ambient)
        out.update(final_sil=float(sil), final_pbr=float(pbr))
        return out
