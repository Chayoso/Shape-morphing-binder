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
    Fg0: np.ndarray | None = None   # geometric (render) deformation at window start
    bond_nbr: np.ndarray | None = None   # (N,K) frozen material neighbours (material bonds)
    bond_rest: np.ndarray | None = None  # (N,K) rest lengths (runner state)
    bond_frag: np.ndarray | None = None  # (N,) 1.0 where the particle is in a fragment
    layer: tuple | None = None           # (mask, nrm, nbr, w, frac): outer-layer relaxation (traj.Trajectory)


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
                          device=spec.device, requires_grad=True, vol0=spec.vol0, layer=spec.layer)
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


def _dfc_to_warp(dFc_t: torch.Tensor, N: int, T: int):
    """(T,N,3,3) or (N,3,3) torch control -> warp array(s) sharing memory."""
    if dFc_t.dim() == 4:
        assert dFc_t.shape[0] == T, f"dFc sequence must be (T={T},N,3,3), got {tuple(dFc_t.shape)}"
        dc = dFc_t.contiguous()
        return [wp.from_torch(dc[t].view(N, 3, 3), dtype=wp.mat33,
                              requires_grad=dFc_t.requires_grad) for t in range(T)], True
    return wp.from_torch(dFc_t.contiguous().view(N, 3, 3), dtype=wp.mat33,
                         requires_grad=dFc_t.requires_grad), False


class _WarpMPMExt(torch.autograd.Function):
    """Extended bridge (docs/render_controls_physics.md §3, §5): the same rollout, but
    the terminal GEOMETRIC deformation Fg_T and EVERY step velocity are exposed, so
      * the render covariance can ride Fg (no direct control route into the image), and
      * a RUNNING kinetic term sum_t |v_t|^2 can penalise in-window ringing instead of
        only the endpoint (the stop-and-go driver, docs/oscillation_triage.md).
    Outputs: x_T (N,3), F_T (N,9), v_T (N,3), Fg_T (N,9), V (T,N,3) with V[t-1] = v_t."""

    @staticmethod
    def forward(ctx, dFc_t: torch.Tensor, lam_t, mu_t, spec: RolloutSpec):
        N, T = spec.x0.shape[0], spec.T
        dFc_wp, seq = _dfc_to_warp(dFc_t, N, T)
        lam_wp = _leaf_f32(lam_t) if lam_t is not None else spec.lam
        mu_wp = _leaf_f32(mu_t) if mu_t is not None else spec.mu
        traj = Trajectory(spec.x0, spec.m, lam_wp, mu_wp, spec.prm, T,
                          Fp=spec.Fp, v0=spec.v0, F0=spec.F0, C0=spec.C0, dFc=dFc_wp,
                          device=spec.device, requires_grad=True, vol0=spec.vol0,
                          Fg0=spec.Fg0, track_geom=True,
                          bonds=((spec.bond_nbr, spec.bond_rest, spec.bond_frag) if spec.bond_nbr is not None else None),
                          layer=spec.layer)
        ctx.tape = wp.Tape()
        with ctx.tape:
            xT, FT = traj.rollout()
        ctx.traj, ctx.dFc_wp, ctx.seq = traj, dFc_wp, seq
        ctx.dFc_req = dFc_t.requires_grad
        ctx.lam_wp = lam_wp if (lam_t is not None and lam_t.requires_grad) else None
        ctx.mu_wp = mu_wp if (mu_t is not None and mu_t.requires_grad) else None
        V = torch.stack([wp.to_torch(traj.v[t]).clone() for t in range(1, T + 1)])
        return (wp.to_torch(xT).clone(), wp.to_torch(FT).reshape(N, 9).clone(),
                wp.to_torch(traj.v[T]).clone(),
                wp.to_torch(traj.Fg[T]).reshape(N, 9).clone(), V)

    @staticmethod
    def backward(ctx, gx, gF, gv, gFg, gV):
        traj = ctx.traj
        N, T = traj.N, traj.T
        dev = traj.device

        def z(shape):
            return torch.zeros(shape, device=gx.device if gx is not None else dev)
        gx = z((N, 3)) if gx is None else gx
        gF = z((N, 9)) if gF is None else gF
        gv = z((N, 3)) if gv is None else gv
        gFg = z((N, 9)) if gFg is None else gFg
        gV = z((T, N, 3)) if gV is None else gV
        gvT = (gv + gV[T - 1]).contiguous()          # v_T is also V[T-1]: one seed
        grads = {traj.x[T]: wp.from_torch(gx.contiguous(), dtype=wp.vec3),
                 traj.F[T]: wp.from_torch(gF.contiguous().view(N, 3, 3), dtype=wp.mat33),
                 traj.v[T]: wp.from_torch(gvT, dtype=wp.vec3),
                 traj.Fg[T]: wp.from_torch(gFg.contiguous().view(N, 3, 3), dtype=wp.mat33)}
        gVc = gV.contiguous()
        for t in range(1, T):
            grads[traj.v[t]] = wp.from_torch(gVc[t - 1], dtype=wp.vec3)
        ctx.tape.backward(grads=grads)
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


def warp_mpm_ext(dFc_t: torch.Tensor, spec: RolloutSpec, lam_t=None, mu_t=None):
    """Extended differentiable rollout. Returns (x_T [N,3], F_T [N,9], v_T [N,3],
    Fg_T [N,9], V [T,N,3])."""
    return _WarpMPMExt.apply(dFc_t, lam_t, mu_t, spec)


# ---- persistent tape trajectory: forward and adjoint as CUDA graphs (2026-09-16) ----------------
_ADJ_WARMED: set = set()


class PersistentAdjoint:
    """One tape trajectory whose forward rollout and adjoint are captured CUDA graphs.
    Inputs/outputs go through persistent buffers: `dc` (T,N,3,3) is the control the dFc
    sequence views; `sx, sF, sv, sFg, sV` are the adjoint seeds the tape assigns from;
    the control gradient is read from the dFc grads. Replaying the forward graph rolls out
    the CURRENT contents of the trajectory's initial state (fixed for a window) with the
    current `dc`; replaying the adjoint graph differentiates the LAST forward with the
    current seeds — so several seeds per forward (PCGrad's per-term gradients) cost one
    graph launch each. Non-CUDA devices record a fresh tape per forward (tests)."""

    def __init__(self, spec: RolloutSpec):
        N, T, dev = spec.x0.shape[0], spec.T, spec.device
        self.N, self.T, self.dev = N, T, dev
        self.cuda = str(dev).startswith("cuda")
        self.dc = torch.zeros(T, N, 3, 3, device=dev)
        self.dc_wp = [wp.from_torch(self.dc[t], dtype=wp.mat33, requires_grad=True) for t in range(T)]
        bonds = ((spec.bond_nbr, spec.bond_rest, spec.bond_frag) if spec.bond_nbr is not None else None)
        self.traj = Trajectory(spec.x0, spec.m, spec.lam, spec.mu, spec.prm, T,
                               Fp=spec.Fp, v0=spec.v0, F0=spec.F0, C0=spec.C0, dFc=self.dc_wp,
                               device=dev, requires_grad=True, vol0=spec.vol0,
                               Fg0=spec.Fg0, track_geom=True, bonds=bonds, persistent=True,
                               layer=spec.layer)
        tr = self.traj
        self.sx = torch.zeros(N, 3, device=dev)
        self.sF = torch.zeros(N, 3, 3, device=dev)
        self.sv = torch.zeros(N, 3, device=dev)
        self.sFg = torch.zeros(N, 3, 3, device=dev)
        self.sV = torch.zeros(max(T - 1, 1), N, 3, device=dev)
        self.seeds = {tr.x[T]: wp.from_torch(self.sx, dtype=wp.vec3),
                      tr.F[T]: wp.from_torch(self.sF, dtype=wp.mat33),
                      tr.v[T]: wp.from_torch(self.sv, dtype=wp.vec3),
                      tr.Fg[T]: wp.from_torch(self.sFg, dtype=wp.mat33)}
        for t in range(1, T):
            self.seeds[tr.v[t]] = wp.from_torch(self.sV[t - 1], dtype=wp.vec3)
        # every gradient buffer the adjoint accumulates into (tape.zero() only knows the
        # arrays of a tape that has already run backward — a fresh tape zeroes nothing)
        self.grad_arrays = []
        for val in list(vars(tr).values()) + [self.dc_wp]:
            items = val if isinstance(val, (list, tuple)) else [val]
            for a in items:
                if isinstance(a, wp.array) and a.grad is not None and a.grad not in self.grad_arrays:
                    self.grad_arrays.append(a.grad)
        self.tape = None
        self.g_fwd = self.g_bwd = None
        if self.cuda:
            key = str(dev)
            if key not in _ADJ_WARMED:              # module load outside any capture
                self._record_forward()
                self._zero_grads()
                self.tape.backward(grads=self.seeds)
                wp.synchronize_device(dev)
                _ADJ_WARMED.add(key)
            with wp.ScopedCapture(device=dev) as cap:
                self._record_forward()
            self.g_fwd = cap.graph
            with wp.ScopedCapture(device=dev) as cap:
                self._zero_grads()
                self.tape.backward(grads=self.seeds)
            self.g_bwd = cap.graph

    def _zero_grads(self):
        for g in self.grad_arrays:
            g.zero_()

    def _record_forward(self):
        self.tape = wp.Tape()
        with self.tape:
            self.traj.rollout()

    def forward(self):
        if self.g_fwd is not None:
            wp.capture_launch(self.g_fwd)
        else:
            self._record_forward()

    def backward(self):
        if self.g_bwd is not None:
            wp.capture_launch(self.g_bwd)
        else:
            self._zero_grads()
            self.tape.backward(grads=self.seeds)

    def apply(self, dFc_t: torch.Tensor):
        return _WarpMPMPersistent.apply(dFc_t, self)


class _WarpMPMPersistent(torch.autograd.Function):
    """The _WarpMPMExt bridge on a PersistentAdjoint (same outputs, same seeds)."""

    @staticmethod
    def forward(ctx, dFc_t: torch.Tensor, adj: PersistentAdjoint):
        N, T = adj.N, adj.T
        adj.dc.copy_(dFc_t.detach().reshape(T, N, 3, 3))
        adj.forward()
        ctx.adj = adj
        tr = adj.traj
        V = torch.stack([wp.to_torch(tr.v[t]).clone() for t in range(1, T + 1)])
        return (wp.to_torch(tr.x[T]).clone(), wp.to_torch(tr.F[T]).reshape(N, 9).clone(),
                wp.to_torch(tr.v[T]).clone(),
                wp.to_torch(tr.Fg[T]).reshape(N, 9).clone(), V)

    @staticmethod
    def backward(ctx, gx, gF, gv, gFg, gV):
        adj = ctx.adj
        N, T = adj.N, adj.T
        with torch.no_grad():
            adj.sx.copy_(gx) if gx is not None else adj.sx.zero_()
            adj.sF.copy_(gF.reshape(N, 3, 3)) if gF is not None else adj.sF.zero_()
            adj.sFg.copy_(gFg.reshape(N, 3, 3)) if gFg is not None else adj.sFg.zero_()
            gvT = (gv if gv is not None else 0.0) + (gV[T - 1] if gV is not None else 0.0)
            if isinstance(gvT, torch.Tensor):
                adj.sv.copy_(gvT)
            else:
                adj.sv.zero_()
            if gV is not None and T > 1:
                adj.sV.copy_(gV[:T - 1])
            else:
                adj.sV.zero_()
            adj.backward()
            g = torch.stack([wp.to_torch(d.grad).reshape(N, 3, 3).clone() for d in adj.dc_wp])
        return g.reshape(T, N, 3, 3), None
