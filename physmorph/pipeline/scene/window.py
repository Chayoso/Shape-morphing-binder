"""What a window of the scene freezes at its start (D147): each body's slice of the state, its frame, its layer, bonds and
minimum spacing (every neighbour list local to the body), its rollout spec on the world grid, and the pair's
persistent no-grad rollout (mpm/bodies.Bodies) that every line-search candidate and the commit are evaluated on.

Each body's objective is the frozen single-body Objective, handed a BodyWindow: the window as that body's objective sees
it, in the body's own frame (its start positions y0 = R0^T (x0 - c0), its layer mask, its target and units).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import warp as wp

from ... import gpu
from ...mpm.bodies import Bodies, BodiesAdjoint, body_trajectory
from ...mpm.constitutive import lame
from ...mpm.function import RolloutSpec
from ...prof import timed
from ..window.layer import layer_relax_data, layer_spacing
from ..window.setup import StartState, positions_in_domain
from .frame import body_frame


@dataclass
class Body:
    """A body of the scene: its rows of the particle arrays, its source sample about its centroid in its target's
    orientation (the frame's reference), its target, the sample's volume per particle and its dynamics mass."""
    start: int
    stop: int
    Xq: torch.Tensor
    tgt: object
    vp: float
    mass: float
    cfg: object = None                  # the body's own run config (its loss grid and berth from its prepare); None = the scene's

    @property
    def rows(self):
        return slice(self.start, self.stop)

    @property
    def n(self) -> int:
        return self.stop - self.start


class BodyWindow:
    """The window as one body's Objective reads it (window/objective.py): the body's start positions in its own frame,
    its layer mask, its target and its units; set_u_gate writes the body's rows of the scene's gate."""

    def __init__(self, scene: "SceneWindow", b: int, y0: torch.Tensor, lmask: torch.Tensor):
        body = scene.bodies[b]
        self.scene, self.b = scene, b
        self.cfg, self.prm, self.tgt = body.cfg or scene.cfg, scene.prm, body.tgt
        self.x0, self.lmask, self.N = y0, lmask, body.n
        self.wu = 1.0 / body.tgt.unit_ratio
        self.eps_eff = scene.cfg.eps / body.tgt.unit_grad_ratio
        self.target_norm_eff = scene.cfg.target_norm / body.tgt.unit_grad_ratio
        self.loss_floor_eff = 1.0 / body.tgt.unit_ratio

    def set_u_gate(self, gate: torch.Tensor) -> None:
        self.scene.set_u_gate(self.b, gate)


class SceneWindow:
    def __init__(self, start: StartState, prm, cfg, bodies: list, vol0: torch.Tensor, bonds: list, mu: float):
        self.cfg, self.prm, self.bodies, self.mu = cfg, prm, bodies, float(mu)
        self.scene = True
        self.x0 = start.x
        self.N = N = int(start.x.shape[0])
        self.Tc, self.T = cfg.T, 2 * cfg.T
        T = self.T
        lam0, mu0 = lame(cfg.young, cfg.poisson)
        self.frames0, self.views, self.specs = [], [], []
        self.lmask = torch.zeros(N, device=start.x.device)
        self.lnrm = torch.zeros(N, 3, device=start.x.device)
        sp = []
        vx = cfg.volume_exact
        for b, body in enumerate(bodies):
            r = body.rows
            x0 = start.x[r]
            y0, c0, R0 = (t.detach() for t in body_frame(x0, body.Xq))
            self.frames0.append((c0, R0))
            with timed("layer_data"):
                sp_b = layer_spacing(x0)
                lmask, lnrm, lnbr, lw = layer_relax_data(x0, sp_b, k=cfg.layer_k, h_sp=cfg.layer_h_sp)
            sp.append(sp_b)
            self.lmask[r], self.lnrm[r] = lmask, lnrm
            with timed("layer_relief"):          # the target's relief, read in the body's frame
                ref = None if body.tgt.relief is None else body.tgt.relief.at(y0, lmask, lnrm @ R0, lnbr, lw)
            layer = (lmask, lnrm, lnbr, lw, 1.0 / float(cfg.T), None, ref)
            spacing = None
            if cfg.min_spacing > 0:
                rr = cfg.min_spacing * float(torch.as_tensor(vol0[r]).mean()) ** (1.0 / 3.0)
                with timed("spacing_knn"):
                    spacing = (gpu.knn(x0, 17)[1][:, 1:], rr)
            nbr, rest, frag = bonds[b]
            sub = lambda a: None if a is None else a[r]          # noqa: E731
            self.specs.append(RolloutSpec(x0=x0, m=body.mass, lam=lam0, mu=mu0, prm=prm, T=T, F0=sub(start.F),
                                          Fp=start.Fp[r], v0=sub(start.v), C0=sub(start.C), device=cfg.device,
                                          vol0=vol0[r], bond_nbr=nbr, bond_rest=rest, bond_frag=frag, layer=layer,
                                          bond_history=True, control_steps=cfg.T, polar_adjoint=True,
                                          spacing=spacing, volume_exact=vx, J0=sub(start.J) if vx != "off" else None,
                                          track_geom=False))
            self.views.append(BodyWindow(self, b, y0, lmask))
        self.sp0 = min(sp)                       # u's per-window bound (one spacing); the bodies' samples are alike
        # the persistent no-grad rollout of the pair: the control and u are single buffers whose rows the bodies view
        self.dc_buf = torch.zeros(T, N, 3, 3, device=cfg.device)
        self.u_buf = torch.zeros(N, device=cfg.device)
        trajs = []
        with timed("tr_build"):
            for body, spec in zip(bodies, self.specs):
                r = body.rows
                seq = [wp.from_torch(self.dc_buf[t, r], dtype=wp.mat33) for t in range(T)]
                uw = wp.from_torch(self.u_buf[r], dtype=wp.float32)
                trajs.append(body_trajectory(spec, seq, uw, requires_grad=False, persistent=True))
            self.tr = Bodies(trajs, [bd.vp for bd in bodies], self.mu)
            self.tr.capture()
        self._adj = None
        b0 = bodies[0].tgt                       # the step control's units: the bodies' targets are alike
        self.wu = 1.0 / b0.unit_ratio
        self.eps_eff = cfg.eps / b0.unit_grad_ratio
        self.target_norm_eff = cfg.target_norm / b0.unit_grad_ratio
        self.loss_floor_eff = 1.0 / b0.unit_ratio

    def expand(self, leaf: torch.Tensor) -> torch.Tensor:
        return torch.cat((leaf, torch.zeros_like(leaf)), dim=0)

    def adjoint(self) -> BodiesAdjoint:
        if self._adj is None:
            with timed("adj_build"):
                self._adj = BodiesAdjoint(self.specs, [b.vp for b in self.bodies], self.mu)
        return self._adj

    def set_u_gate(self, b: int, gate: torch.Tensor) -> None:
        spec = self.specs[b]
        spec.layer = spec.layer[:5] + (gate,) + spec.layer[6:]
        wp.to_torch(self.tr.trajs[b].layer_ug).copy_(gate)

    def u_gate(self) -> torch.Tensor:
        return torch.cat([wp.to_torch(tr.layer_ug) for tr in self.tr.trajs])

    def load(self, leaf: torch.Tensor, u: torch.Tensor | None) -> torch.Tensor:
        dc = self.expand(leaf.detach()).detach().contiguous()
        self.dc_buf.copy_(dc.view(self.T, self.N, 3, 3))
        self.u_buf.copy_(u.detach() if u is not None else torch.zeros_like(self.u_buf))
        return dc

    def positions_in_domain(self) -> bool:
        x = torch.stack([self.tr.cat("x", t) for t in range(self.T + 1)])
        return positions_in_domain(x, self.prm)
