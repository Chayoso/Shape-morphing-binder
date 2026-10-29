"""The morph's progress in the scene (D147; the user's redesign, 2026-10-14: the bodies meet mid-morph, not as finished
bunnies). Each body's target at time t is the displacement interpolation of its transport from the source to the target
(McCann 1997), Z_s = X + s D(X), with D the pipeline's own transport map at the source (the u gate's operator:
losses/grid_ot.grid_transport_displacement, the debiased barycentric map on the MPM cell's grid, blur one cell) and s a
smoothstep ramp from 0 to 1 over the morph time. The objective is the frozen one on that target: no new term, no
weight; the body is shaped toward where the transport would have carried it by then. From s = 1 on, the run's own
target (its relief, its draws). The packs on the way carry the final target's units (unit_ratio, its transport scale),
so the terms keep their weights through the ramp; their relaxation has no relief (a mesh's, which Z_s has not)."""
from __future__ import annotations

import torch

from ...losses.grid_ot import GridSinkhornLoss, grid_transport_displacement
from ...losses.volumetric import d_vol_density
from ..target import build_target


def ramp(t: float, t_m: float) -> float:
    """The smoothstep 3u^2 - 2u^3 of u = t / t_m in [0, 1]: the target's speed starts and ends at zero."""
    u = min(max(t / t_m, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def transport_scale(pack, cfg, X: torch.Tensor) -> float:
    """The run's one transport scale at the source, as window/objective.Objective computes it at the first window (the
    transport term's gradient norm equal to D_vol's); built here on the final target, since the first window's target
    is then the source itself, where both gradients vanish."""
    if pack.grid_ot is None:
        pack.grid_ot = GridSinkhornLoss(pack.grid, pack.lgmin, pack.ldx, pack.ldims, eps=float(pack.ldx) ** 2,
                                        iters=cfg.ot_iters, tol=min(cfg.ot_tol, 1e-3), mass_total=float(pack.m.sum()),
                                        cuda_blocks=True, support=pack.support)
    xg = X.detach().clone().requires_grad_(True)
    gd = torch.autograd.grad(d_vol_density(xg, pack.m, pack.grid, pack.lgmin, pack.ldx, pack.ldims, pack.m_ref,
                                           pack.n_support), xg)[0].norm()
    xg = X.detach().clone().requires_grad_(True)
    gt = torch.autograd.grad(pack.grid_ot.state_energy(xg, pack.m), xg)[0].norm()
    return float(gd / gt.clamp_min(1e-30))


class MorphSchedule:
    """bodies: scene/window.Body with their final, calibrated targets; prms: each target's build grid (the single-body
    preparation's MPM parameters, as scene_run builds the targets); morph_time: the ramp's length, seconds."""

    def __init__(self, bodies: list, prms: list, morph_time: float):
        self.t_m, self.prms = float(morph_time), prms
        self.final = [bd.tgt for bd in bodies]
        self.D = []
        for bd, pack in zip(bodies, self.final):
            cfg = bd.cfg
            if pack.ot_scale is None:
                pack.ot_scale = transport_scale(pack, cfg, bd.Xq)
            g_grid, g_dx, g_dims = pack.gate if pack.gate is not None else (pack.grid, pack.ldx, pack.ldims)
            with torch.no_grad():
                self.D.append(grid_transport_displacement(bd.Xq, pack.m, g_grid, pack.lgmin, g_dx, g_dims,
                                                          eps=float(g_dx) ** 2, iters=cfg.ot_iters, tol=cfg.ot_tol))
            dn = self.D[-1].norm(dim=1)
            print(f"[morph] body {len(self.D) - 1}: transport at the source |D| mean {float(dn.mean()):.4f}, max "
                  f"{float(dn.max()):.4f} wu; the ramp over {self.t_m:g} s", flush=True)

    def apply(self, bodies: list, t: float) -> float:
        """Each body's target for the window ending at time t (the objective reads its released end); returns s."""
        s = ramp(t, self.t_m)
        for b, bd in enumerate(bodies):
            if s >= 1.0:
                bd.tgt = self.final[b]
                continue
            f = self.final[b]
            pack = build_target((bd.Xq + s * self.D[b]).detach().contiguous(), self.prms[b], bd.cfg)
            pack.unit_ratio, pack.unit_grad_ratio, pack.ot_scale = f.unit_ratio, f.unit_grad_ratio, f.ot_scale
            bd.tgt = pack
        return s
