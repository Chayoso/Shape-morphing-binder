"""torch.autograd.Function bridging Warp MPM rollout into the torch graph.

Leaves -> WarpMPM -> terminal state, with autograd. See docs/SPEC.md §4.2 and
docs/pipeline_v2.md §3.

Two entry points:
  warp_mpm(dFc, spec)                 -> (x_T, F_T)          # v1-compatible
  warp_mpm_full(dFc, spec, lam, mu)   -> (x_T, F_T, v_T)     # v2: material leaves + velocity

dFc is (N,3,3) — one control shared by every step — or (T,N,3,3), a control SEQUENCE with an
independent field per step (the C++ CompGraph formulation). The sequence case is the reason
autodiff beats the C++ here: ONE backward pass yields dL/ddFc[t] for every t at once, where the
C++ does a forward+backward per layer.

lam/mu are optional per-particle (N,) torch tensors; when they require grad the same tape
backward also yields dL/dλ_i, dL/dμ_i — the render feedback's *material* channel (§3.2 ch.2).
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
    lam: object            # scalar/np default; ignored when a lam tensor leaf is passed
    mu: object
    prm: MPMParams
    T: int
    Fp: np.ndarray | None = None
    v0: np.ndarray | None = None
    F0: np.ndarray | None = None
    C0: np.ndarray | None = None
    device: str = "cuda"
    vol0: np.ndarray | None = None  # one-time source-rest Vp; reused across all windows


def _leaf_f32(t: torch.Tensor):
    """(N,) float leaf -> warp array sharing memory, grads mapped back."""
    return wp.from_torch(t.contiguous(), dtype=wp.float32, requires_grad=t.requires_grad)


class _WarpMPM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, dFc_t: torch.Tensor, lam_t, mu_t, spec: RolloutSpec):
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
        lam_wp = _leaf_f32(lam_t) if lam_t is not None else spec.lam
        mu_wp = _leaf_f32(mu_t) if mu_t is not None else spec.mu
        traj = Trajectory(spec.x0, spec.m, lam_wp, mu_wp, spec.prm, T,
                          Fp=spec.Fp, v0=spec.v0, F0=spec.F0, C0=spec.C0, dFc=dFc_wp,
                          device=spec.device, requires_grad=True, vol0=spec.vol0)
        ctx.tape = wp.Tape()
        with ctx.tape:
            xT, FT = traj.rollout()
        ctx.traj, ctx.dFc_wp, ctx.seq = traj, dFc_wp, seq
        ctx.dFc_req = dFc_t.requires_grad          # material-only optimisation is legal:
        ctx.lam_wp = lam_wp if (lam_t is not None and lam_t.requires_grad) else None
        ctx.mu_wp = mu_wp if (mu_t is not None and mu_t.requires_grad) else None
        return (wp.to_torch(xT).clone(), wp.to_torch(FT).reshape(N, 9).clone(),
                wp.to_torch(traj.v[T]).clone())

    @staticmethod
    def backward(ctx, gx: torch.Tensor, gF: torch.Tensor, gv: torch.Tensor):
        traj = ctx.traj
        N, T = traj.N, traj.T
        grads = {traj.x[T]: wp.from_torch(gx.contiguous(), dtype=wp.vec3),
                 traj.F[T]: wp.from_torch(gF.contiguous().view(N, 3, 3), dtype=wp.mat33),
                 traj.v[T]: wp.from_torch(gv.contiguous(), dtype=wp.vec3)}
        ctx.tape.backward(grads=grads)
        # each leaf's grad is read ONLY if that input required grad (a warp array made
        # from a no-grad tensor has grad=None -> to_torch(None) crashes; caught by G1b)
        if not ctx.dFc_req:
            g = None
        elif ctx.seq:
            g = torch.stack([wp.to_torch(d.grad).reshape(N, 3, 3).clone() for d in ctx.dFc_wp])
        else:
            g = wp.to_torch(ctx.dFc_wp.grad).reshape(N, 3, 3).clone()
        g_lam = wp.to_torch(ctx.lam_wp.grad).clone() if ctx.lam_wp is not None else None
        g_mu = wp.to_torch(ctx.mu_wp.grad).clone() if ctx.mu_wp is not None else None
        ctx.tape.zero()
        return g, g_lam, g_mu, None


def warp_mpm_full(dFc_t: torch.Tensor, spec: RolloutSpec, lam_t=None, mu_t=None):
    """Differentiable rollout with material leaves. Returns (x_T [N,3], F_T [N,9], v_T [N,3])."""
    return _WarpMPM.apply(dFc_t, lam_t, mu_t, spec)


def warp_mpm(dFc_t: torch.Tensor, spec: RolloutSpec):
    """v1-compatible entry: constant material from spec. Returns (x_T, F_T)."""
    xT, FT, _ = _WarpMPM.apply(dFc_t, None, None, spec)
    return xT, FT
