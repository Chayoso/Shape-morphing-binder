"""Rollouts of a window's controls: the differentiable one (gradients), the no-grad one
(line-search candidates) and the commit rollout (what the runner promotes).

A candidate is admissible only with a FINITE, ORIENTATION-PRESERVING whole trajectory that
stays inside the domain: NaN particles vanish from the splats and det F <= 0 is invisible
to the data terms, so either could fake a lower loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import warp as wp

from .objective import Objective, velocity_variance
from .setup import Window


@dataclass
class Eval:
    xT: torch.Tensor
    FT: torch.Tensor                  # (N, 9)
    vT: torch.Tensor
    lv: torch.Tensor
    lk: torch.Tensor
    lr: torch.Tensor
    lpbr: torch.Tensor
    d_sil: torch.Tensor
    dfc: torch.Tensor                 # the expanded control (2T, N, 3, 3)
    V: torch.Tensor                   # velocities of steps 1..2T
    lk_run: torch.Tensor
    lk_var: torch.Tensor
    jt: float | None = None           # whole-trajectory min det (no-grad path)
    in_domain: bool = True

    def state(self):
        return (self.xT, self.FT, self.vT)


def state_ok(e: Eval) -> bool:
    """Finite and orientation-preserving over the whole trajectory, inside the domain."""
    if not all(bool(torch.isfinite(t).all()) for t in (e.xT, e.FT, e.vT)):
        return False
    if not e.in_domain:
        return False
    if not bool((torch.linalg.det(e.FT.view(-1, 3, 3)) > 0).all()):
        return False
    if e.jt is not None:
        return e.jt > 1e-4          # margin above the float32 det noise floor
    return True


def _evaluate(obj: Objective, xT, FT, vT, dfc, V, **kw) -> Eval:
    lv, lk, lr, lpbr, d_sil = obj.losses(xT, FT, vT)
    return Eval(xT, FT, vT, lv, lk, lr, lpbr, d_sil, dfc, V, V.pow(2).sum(2).mean(),
                velocity_variance(V, obj.cfg.T), **kw)


def graph_terms(win: Window, obj: Objective, leaf: torch.Tensor, u: torch.Tensor) -> Eval:
    """Differentiable rollout (the persistent tape; forward and adjoint as CUDA graphs)."""
    dfc = win.expand(leaf)
    xT, FT, vT, _, V = win.adjoint().apply(dfc, u)
    return _evaluate(obj, xT, FT, vT, dfc, V)


def _trajectory_min_det(win: Window, dc: torch.Tensor) -> float:
    """min over the window of det F after each step and det(F + dFc) before it: the stored F
    is smoothed, so an inversion of the EFFECTIVE deformation could hide in it."""
    tr, T, N = win.tr, win.T, win.N
    F_post = torch.stack([wp.to_torch(tr.F[t]).reshape(N, 3, 3) for t in range(1, T + 1)])
    F_pre = torch.stack([wp.to_torch(tr.F[t]).reshape(N, 3, 3) for t in range(T)])
    j_eff = torch.linalg.det(F_pre + dc.view(T, N, 3, 3)).min()
    return float(torch.minimum(torch.linalg.det(F_post).min(), j_eff))


def eval_terms(win: Window, obj: Objective, leaf: torch.Tensor, u: torch.Tensor) -> Eval:
    """No-grad rollout of a candidate on the persistent trajectory (no tape, no adjoint
    buffers). The outputs are copies: the next candidate rewrites the buffers."""
    with torch.no_grad():
        dc = win.load(leaf, u)
        tr, T, N = win.tr, win.T, win.N
        tr.run()
        xT = wp.to_torch(tr.x[T]).clone()
        FT = wp.to_torch(tr.F[T]).reshape(N, 9).clone()
        vT = wp.to_torch(tr.v[T]).clone()
        V = torch.stack([wp.to_torch(tr.v[t]) for t in range(1, T + 1)])
        e = _evaluate(obj, xT, FT, vT, dc, V)
        e.jt = _trajectory_min_det(win, dc)
        e.in_domain = win.positions_in_domain()
        return e


@dataclass
class Commit:
    """The committed rollout. x/F are views of the eval trajectory's per-step buffers,
    valid until the window is released; the end state is copied."""
    x: list                           # 2T+1 (N,3) tensors
    F: list                           # 2T+1 (N,3,3) tensors
    end_F: torch.Tensor
    end_v: torch.Tensor
    end_C: torch.Tensor
    n_inv_steps: int                  # particles with det F <= 0 at any step
    jmin_traj: float
    E_final: float
    jt_final: float
    valid: bool                       # finite, oriented, inside the domain
    owner: object = None              # the window whose buffers x and F view (kept alive)


def commit_rollout(win: Window, obj: Objective, leaf, u, lam_r: float) -> Commit:
    """Roll out the accepted controls once more and validate the rollout itself (CUDA
    atomics make a replay differ from the accepted candidate by noise)."""
    with torch.no_grad():
        e = eval_terms(win, obj, leaf, u)
        E_final = obj.scalar(e.lv, e.lk, e.lr, lam_r, e.dfc, e.xT, e.FT, e.lk_var)
        tr, T, N = win.tr, win.T, win.N
        inv_any, jmin = None, float("inf")
        for t in range(1, T + 1):
            det_t = torch.linalg.det(wp.to_torch(tr.F[t]).reshape(-1, 3, 3).float())
            bad = det_t <= 0.0                            # NaN rows compare False
            inv_any = bad if inv_any is None else (inv_any | bad)
            jmin = min(jmin, float(det_t.min()))
        return Commit(x=[wp.to_torch(tr.x[t]) for t in range(T + 1)],
                      F=[wp.to_torch(tr.F[t]).reshape(N, 3, 3) for t in range(T + 1)],
                      end_F=wp.to_torch(tr.F[T]).reshape(N, 3, 3).clone(),
                      end_v=wp.to_torch(tr.v[T]).clone(), end_C=wp.to_torch(tr.C[T]).reshape(N, 3, 3).clone(),
                      n_inv_steps=int(inv_any.sum()), jmin_traj=jmin, E_final=E_final,
                      jt_final=e.jt, valid=state_ok(e), owner=win)
