"""What a window of the scene freezes at its start (D147): each body's slice of the state, its frame, its layer, bonds and
minimum spacing (every neighbour list local to the body), its rollout spec on the world grid, and the pair's
persistent no-grad rollout (mpm/bodies.Bodies) that every line-search candidate and the commit are evaluated on.

Each body's objective is the frozen single-body Objective, handed a BodyWindow: the window as that body's objective sees
it, in the body's own frame (its start positions y0 = R0^T (x0 - c0), its layer mask, its target and units).
"""
from __future__ import annotations

import dataclasses
import weakref
from dataclasses import dataclass, field

import torch
import warp as wp

from ... import gpu
from ...mpm.bodies import Bodies, BodiesAdjoint, body_trajectory
from ...mpm.constitutive import lame
from ...mpm.function import RolloutSpec
from ...prof import timed
from ..window.layer import layer_relax_data, layer_spacing
from ..window.setup import StartState, positions_in_domain
from .frame import body_frame, deformation_velocity


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
    cfg: object = None                  # the body's own run config (its prepare's loss grid, berth); None: the scene's
    fixed: tuple | None = None          # V0 only: (c, R) of a frame fixed at launch (the world's), velocities as they are
    raster: object = None               # scene/raster.BodyRaster (fix 1): the loss raster grows to cover the body; None:
                                        #   the target's raster as built
    base: object = None                 # the window's target before the raster's growth (set by scene/raster.cover)
    packs: dict = field(default_factory=dict)       # the base target on grown rasters, by step count

    @property
    def rows(self):
        return slice(self.start, self.stop)

    @property
    def n(self) -> int:
        return self.stop - self.start

    def frame(self, x: torch.Tensor):
        """(y, c, R) of the body's positions x (its rows): the Buckens frame, or (V0) the frame fixed at launch."""
        if self.fixed is not None:
            c, R = self.fixed
            return (x - c) @ R, c, R
        return body_frame(x, self.Xq)

    def deformation(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """The velocity the body's settling terms read: less its rigid velocity, or (V0) as it is."""
        return v if self.fixed is not None else deformation_velocity(x, v)


class BodyWindow:
    """The window as one body's Objective reads it (window/objective.py): the body's start positions in its own frame,
    its layer mask, its target and its units; set_u_gate writes the body's rows of the scene's gate."""

    def __init__(self, scene: "SceneWindow", b: int, y0: torch.Tensor, lmask: torch.Tensor):
        body = scene.bodies[b]
        # a weak reference: the scene holds its views, and a strong one back would make a cycle that only the garbage
        # collector frees, keeping a window's trajectories (gigabytes at 300k) alive across windows (D148's finding)
        self._scene, self.b = weakref.ref(scene), b
        self.cfg, self.prm, self.tgt = body.cfg or scene.cfg, scene.prm, body.tgt
        self.x0, self.lmask, self.N = y0, lmask, body.n
        self.wu = 1.0 / body.tgt.unit_ratio
        self.eps_eff = scene.cfg.eps / body.tgt.unit_grad_ratio
        self.target_norm_eff = scene.cfg.target_norm / body.tgt.unit_grad_ratio
        self.loss_floor_eff = 1.0 / body.tgt.unit_ratio

    def set_u_gate(self, gate: torch.Tensor) -> None:
        self._scene().set_u_gate(self.b, gate)


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
            y0, c0, R0 = (t.detach() for t in body.frame(x0))
            self.frames0.append((c0, R0))
            with timed("layer_data"):
                sp_b = layer_spacing(x0)
                lmask, lnrm, lnbr, lw = layer_relax_data(x0, sp_b, k=cfg.layer_k, h_sp=cfg.layer_h_sp)
            sp.append(sp_b)
            self.lmask[r], self.lnrm[r] = lmask, lnrm
            with timed("layer_relief"):          # the target's relief, read in the body's frame
                ref = None if body.tgt.relief is None else body.tgt.relief.at(y0, lmask, lnrm @ R0, lnbr, lw)
            layer = (lmask, lnrm, lnbr, lw, 1.0 / float(cfg.T), None, ref,     # as window/setup.Window's, D146 too
                     cfg.T if cfg.relax_driven else None)
            spacing = None
            if cfg.min_spacing > 0:
                rr = cfg.min_spacing * float(torch.as_tensor(vol0[r]).mean()) ** (1.0 / 3.0)
                with timed("spacing_knn"):
                    snbr = gpu.knn(x0, 17)[1][:, 1:]
                    spacing = (gpu.symmetric_rows(snbr) if cfg.spacing_symmetric else snbr, rr)   # D149 as setup.py
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
                trajs.append(body_trajectory(spec, seq, uw, requires_grad=False, persistent=True, vp=body.vp))
            self.tr = Bodies(trajs, [bd.vp for bd in bodies], self.mu)
            self.tr.capture()
        self._adj = None
        self.zero_rolled = False                 # the persistent buffers hold the zero-control rollout (raster.cover)
        self.raster_k = self.raster_extent = None
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

    def free_end_F(self) -> torch.Tensor:
        """Each body's end F under the window's committed controls (the ones last loaded: the commit's) without the
        loads: alone, no gravity, no floor (D147, the coordinator's go on assimilating the control's share). The
        scene's plastic assimilation reads it in place of the loaded end F, so the elastic strain a collision, the
        body's weight or the floor make stays elastic and only the controls' deformation becomes rest shape. For a body
        without loads it is the same rollout, so the frozen rule (to the atomics' rounding)."""
        prm = dataclasses.replace(self.prm, f_ext=(0.0, 0.0, 0.0), floor_y=-1.0e9)
        out = []
        for body, spec in zip(self.bodies, self.specs):
            r = body.rows
            seq = [wp.from_torch(self.dc_buf[t, r], dtype=wp.mat33) for t in range(self.T)]
            uw = wp.from_torch(self.u_buf[r], dtype=wp.float32)
            tr = body_trajectory(dataclasses.replace(spec, prm=prm), seq, uw, requires_grad=False, persistent=False)
            Bodies([tr], [body.vp], self.mu).rollout()
            out.append(wp.to_torch(tr.F[self.T]).reshape(-1, 3, 3).clone())
        return torch.cat(out)
