"""torch.autograd.Function bridging Warp MPM rollout into the torch graph.

dFc (torch leaf) -> WarpMPM -> (x_T, F_T) torch tensors with autograd.
See docs/SPEC.md §4.2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import warp as wp

from .state import MPMParams
from .traj import Trajectory


@dataclass
class RolloutSpec:
    x0: np.ndarray
    m: object
    lam: object
    mu: object
    prm: MPMParams
    T: int
    Fp: np.ndarray | None = None
    v0: np.ndarray | None = None
    F0: np.ndarray | None = None
    C0: np.ndarray | None = None
    device: str = "cuda"


class _WarpMPM(torch.autograd.Function):
    """dFc_t is either (N,3,3) — one control shared by every step — or (T,N,3,3), a control
    SEQUENCE with an independent field per step, which is the formulation the C++ CompGraph
    optimises. The sequence case is the reason autodiff beats the C++ here: ONE backward pass
    yields dL/ddFc[t] for every t at once, where the C++ does a forward+backward per layer."""

    @staticmethod
    def forward(ctx, dFc_t: torch.Tensor, spec: RolloutSpec):
        N, T = spec.x0.shape[0], spec.T
        seq = dFc_t.dim() == 4
        if seq:
            assert dFc_t.shape[0] == T, f"dFc sequence must be (T={T},N,3,3), got {tuple(dFc_t.shape)}"
            dc = dFc_t.contiguous()
            # each slice of a contiguous (T,N,3,3) tensor is itself contiguous, so from_torch
            # SHARES memory with the leaf and .grad maps straight back — no copies.
            dFc_wp = [wp.from_torch(dc[t].view(N, 3, 3), dtype=wp.mat33,
                                    requires_grad=dFc_t.requires_grad) for t in range(T)]
        else:
            dFc_wp = wp.from_torch(dFc_t.contiguous().view(N, 3, 3), dtype=wp.mat33,
                                   requires_grad=dFc_t.requires_grad)
        traj = Trajectory(spec.x0, spec.m, spec.lam, spec.mu, spec.prm, T,
                          Fp=spec.Fp, v0=spec.v0, F0=spec.F0, C0=spec.C0, dFc=dFc_wp,
                          device=spec.device, requires_grad=True)
        ctx.tape = wp.Tape()
        with ctx.tape:
            xT, FT = traj.rollout()
        ctx.traj, ctx.dFc_wp, ctx.seq = traj, dFc_wp, seq
        return wp.to_torch(xT).clone(), wp.to_torch(FT).reshape(N, 9).clone()

    @staticmethod
    def backward(ctx, gx: torch.Tensor, gF: torch.Tensor):
        traj = ctx.traj
        N, T = traj.N, traj.T
        gx_wp = wp.from_torch(gx.contiguous(), dtype=wp.vec3)
        gF_wp = wp.from_torch(gF.contiguous().view(N, 3, 3), dtype=wp.mat33)
        ctx.tape.backward(grads={traj.x[T]: gx_wp, traj.F[T]: gF_wp})
        if ctx.seq:
            g = torch.stack([wp.to_torch(d.grad).reshape(N, 3, 3).clone() for d in ctx.dFc_wp])
        else:
            g = wp.to_torch(ctx.dFc_wp.grad).reshape(N, 3, 3).clone()
        ctx.tape.zero()
        return g, None


def warp_mpm(dFc_t: torch.Tensor, spec: RolloutSpec):
    """Differentiable rollout. dFc_t is (N,3,3) [shared control] or (T,N,3,3) [control sequence].
    Returns (x_T [N,3], F_T [N,9]) torch tensors."""
    return _WarpMPM.apply(dFc_t, spec)
