"""The morph's progress in the scene (D147; the user's redesign, 2026-10-14: the bodies meet mid-morph, not as finished
bunnies). Each body's target at time t is the displacement interpolation of its transport between the source and the
target (McCann 1997), read from the target's side: Z_s = Y + (1 - s) D_Y(Y), with Y the run's own target sample and
D_Y the pipeline's own transport map from the target to the source (the u gate's operator,
losses/grid_ot.grid_transport_displacement: the debiased barycentric map on the MPM cell's grid, blur one cell), s a
smoothstep ramp from 0 to 1 over the morph time. At s = 1 the target is the run's own sample point for point, so the
morph ends without a switch (read from the source's side, X + s D(X), it ended on the transport's one-cell blur of
the target, which lacks the ears' and the face's detail, and the switch to the true target at s = 1 made the last
15-20 % of the visible morph happen in half a second: the late snap the user saw); the blur sits at the start, on the
sphere, where it is harmless. The objective is the frozen one on that target: no new term, no weight. The packs on the
way carry the final target's units (unit_ratio, its transport scale), so the terms keep their weights through the
ramp; their relaxation has no relief and their pictures no further draws (the mesh's and the extra samples', which
the run's own target adds from s = 1)."""
from __future__ import annotations

import torch

from ...losses.grid_ot import GridSinkhornLoss, grid_transport_displacement
from ...losses.volumetric import d_vol_density, target_mass_grid
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
        self.D = []                                     # D_Y at each body's target sample
        for bd, pack in zip(bodies, self.final):
            cfg = bd.cfg
            if pack.ot_scale is None:
                pack.ot_scale = transport_scale(pack, cfg, bd.Xq)
            _, g_dx, g_dims = pack.gate if pack.gate is not None else (pack.grid, pack.ldx, pack.ldims)
            Y = pack.pts
            with torch.no_grad():
                src = target_mass_grid(bd.Xq, torch.ones(len(bd.Xq), device=Y.device), pack.lgmin, g_dx, g_dims)
                self.D.append(grid_transport_displacement(Y, torch.ones(len(Y), device=Y.device), src, pack.lgmin,
                                                          g_dx, g_dims, eps=float(g_dx) ** 2, iters=cfg.ot_iters,
                                                          tol=cfg.ot_tol))
            dn = self.D[-1].norm(dim=1)
            print(f"[morph] body {len(self.D) - 1}: transport from the target to the source |D_Y| mean "
                  f"{float(dn.mean()):.4f}, max {float(dn.max()):.4f} wu; the ramp over {self.t_m:g} s", flush=True)

    def apply(self, bodies: list, t: float) -> float:
        """Each body's target for the window ending at time t (the objective reads its released end); returns s."""
        s = ramp(t, self.t_m)
        for b, bd in enumerate(bodies):
            if s >= 1.0:
                bd.tgt = self.final[b]
                continue
            f = self.final[b]
            pack = build_target((f.pts + (1.0 - s) * self.D[b]).detach().contiguous(), self.prms[b], bd.cfg)
            pack.unit_ratio, pack.unit_grad_ratio, pack.ot_scale = f.unit_ratio, f.unit_grad_ratio, f.ot_scale
            bd.tgt = pack
        return s
