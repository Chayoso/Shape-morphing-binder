"""What a window freezes at its start: masses, the outer layer, the rollout spec, the
persistent no-grad trajectory every line-search candidate is evaluated on, and the unit
constants of the objective.

A window simulates 2T steps: T driven steps with the controls, then T released steps with
the controls at zero while the physics keeps running. The control leaf is the driven half,
(T, N, 3, 3) per particle and step; `expand` appends the zero released half.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import warp as wp

from ... import gpu
from ...mpm.constitutive import lame
from ...mpm.function import PersistentAdjoint, RolloutSpec
from ...mpm.state import MPMParams
from ...mpm.traj import Trajectory
from ..config import PipelineConfig
from ..target import TargetPack
from .basis import GridBasis
from .layer import layer_relax_data, layer_spacing


@dataclass
class StartState:
    """The promoted state a window starts from (CUDA tensors; None = rest / identity)."""
    x: torch.Tensor
    Fp: torch.Tensor
    F: torch.Tensor | None = None
    v: torch.Tensor | None = None
    C: torch.Tensor | None = None


class Window:
    def __init__(self, start: StartState, prm: MPMParams, cfg: PipelineConfig, tgt: TargetPack,
                 vol0: torch.Tensor, bonds: tuple):
        self.cfg, self.prm, self.tgt = cfg, prm, tgt
        self.x0 = start.x
        self.N = N = int(start.x.shape[0])
        self.Tc = cfg.T                               # driven steps
        self.T = T = 2 * cfg.T                        # driven + released steps
        lam0, mu0 = lame(cfg.young, cfg.poisson)
        # the dynamics mass of the discretisation: the body's mass does not depend on N, so a
        # unit control moves the 300k body as it moves the 40k one (loss-side masses are unit)
        m = float(cfg.mass_ref_n) / N if cfg.mass_ref_n > 0 and N != cfg.mass_ref_n else 1.0
        # the stress control's grid nodes and each particle's transfer weights at the window's start (D109)
        self.basis = GridBasis(start.x, prm.dx, prm.grid_min, (prm.nx, prm.ny, prm.nz))
        # the outer layer: relaxed toward its neighbours' plane over one window (fraction
        # 1/T per driven step) and carrying the u control
        self.sp0 = layer_spacing(start.x)
        self.lmask, self.lnrm, lnbr, lw = layer_relax_data(start.x, self.sp0, k=cfg.layer_k,
                                                           h_sp=cfg.layer_h_sp)
        # the relaxation's reference: its own operator on the layer's feet on the target's surface (target.relief)
        ref = None if tgt.relief is None else tgt.relief.at(start.x, self.lmask, self.lnrm, lnbr, lw)
        layer = (self.lmask, self.lnrm, lnbr, lw, 1.0 / float(cfg.T), None, 0.0, None, ref)
        # the minimum spacing (kernels.k_update, D70): no two particles nearer than cfg.min_spacing of the pitch their
        # rest volume gives, among each particle's 16 nearest at the window's start
        spacing = None
        if cfg.min_spacing > 0:
            spacing = (gpu.knn(start.x, 17)[1][:, 1:], cfg.min_spacing * float(torch.as_tensor(vol0).mean()) ** (1.0 / 3.0))
        nbr, rest, frag = bonds
        self.spec = RolloutSpec(x0=start.x, m=m, lam=lam0, mu=mu0, prm=prm, T=T, F0=start.F,
                                Fp=start.Fp, v0=start.v, C0=start.C, device=cfg.device, vol0=vol0,
                                bond_nbr=nbr, bond_rest=rest, bond_frag=frag, layer=layer,
                                bond_history=True, control_steps=cfg.T, polar_adjoint=True, spacing=spacing)
        # the persistent no-grad trajectory: allocated once, rolled out as a CUDA graph for
        # every candidate, the warm-start comparison and the commit rollout; the control is
        # copied into dc_buf, which its dFc sequence views
        self.dc_buf = torch.zeros(T, N, 3, 3, device=cfg.device)
        seq = [wp.from_torch(self.dc_buf[t], dtype=wp.mat33) for t in range(T)]
        self.tr = Trajectory(start.x, m, lam0, mu0, prm, T, F0=start.F, Fp=start.Fp, v0=start.v,
                             C0=start.C, dFc=seq, device=cfg.device, requires_grad=False, vol0=vol0,
                             persistent=True, bonds=bonds, layer=layer, bond_history=True,
                             control_steps=cfg.T, polar_adjoint=True, spacing=spacing)
        self.tr.capture()
        self._adj = None
        # unit constants: every fixed weight and gradient-magnitude constant is a legacy-unit
        # number, converted by the ratios measured at the source
        self.wu = 1.0 / tgt.unit_ratio
        self.eps_eff = cfg.eps / tgt.unit_grad_ratio
        self.target_norm_eff = cfg.target_norm / tgt.unit_grad_ratio
        self.loss_floor_eff = 1.0 / tgt.unit_ratio

    def expand(self, leaf: torch.Tensor) -> torch.Tensor:
        """(T, N, 3, 3) driven controls -> (2T, N, 3, 3) with the released half at zero."""
        return torch.cat((leaf, torch.zeros_like(leaf)), dim=0)

    def adjoint(self) -> PersistentAdjoint:
        """The tape trajectory (forward and adjoint as CUDA graphs), built at the first
        gradient; it reads the spec as it is then (the u gate included)."""
        if self._adj is None:
            self._adj = PersistentAdjoint(self.spec)
        return self._adj

    def set_u_gate(self, gate: torch.Tensor) -> None:
        """Per-particle gate of u (1 where u may act), in the spec and the eval trajectory."""
        self.spec.layer = self.spec.layer[:5] + (None, 0.0, gate) + self.spec.layer[8:]
        wp.to_torch(self.tr.layer_ug).copy_(gate)

    def load(self, leaf: torch.Tensor, u: torch.Tensor | None) -> torch.Tensor:
        """Copy a candidate into the eval trajectory's buffers; returns the expanded control."""
        dc = self.expand(leaf.detach()).detach().contiguous()
        self.dc_buf.copy_(dc.view(self.T, self.N, 3, 3))
        if u is not None:
            wp.to_torch(self.tr.layer_u).copy_(u.detach())
        return dc

    def positions_in_domain(self) -> bool:
        """Every frame of the last eval rollout inside the runner's two-cell safety margin."""
        x = torch.stack([wp.to_torch(p) for p in self.tr.x])
        return positions_in_domain(x, self.prm)


def domain_bounds(prm: MPMParams, device="cuda"):
    """(lo, hi): the MPM domain shrunk by two cells, the runner's clip box (float32)."""
    import numpy as np
    origin = np.asarray(prm.grid_min, np.float32)
    far = origin + prm.dx * np.array([prm.nx, prm.ny, prm.nz], np.float32)
    lo = torch.as_tensor(origin + 2 * prm.dx, device=device)
    hi = torch.as_tensor(far - 2 * prm.dx, device=device)
    return lo, hi


def positions_in_domain(x: torch.Tensor, prm: MPMParams) -> bool:
    lo, hi = domain_bounds(prm, x.device)
    return bool((torch.isfinite(x) & (x >= lo) & (x <= hi)).all())
