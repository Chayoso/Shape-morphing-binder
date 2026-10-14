"""The scene's rollouts (D147), as window/rollout.py's for one body: the differentiable one, the no-grad one every
line-search candidate is scored on, and the commit rollout; the pair's states are concatenated in body order, and the
per-step positions reach the objective (each step's rigid velocity is fitted at its own positions)."""
from __future__ import annotations

import torch
import warp as wp

from ...prof import timed
from ..window.rollout import Commit, Eval, state_ok, state_reason


def graph_terms(win, obj, leaf: torch.Tensor, u: torch.Tensor) -> Eval:
    dfc = win.expand(leaf)
    with timed("mpm_tape"):
        xT, FT, vT, V, X = win.adjoint().apply(dfc, u)
    return obj.evaluate(xT, FT, vT, dfc, V, X)


def _min_det(win, dc: torch.Tensor) -> float:
    """window/rollout._trajectory_min_det over every body: det(F + dFc) before each step, det F and the tracked J after."""
    T = win.T
    mins = []
    for body, tr in zip(win.bodies, win.tr.trajs):
        dcs = dc.view(T, win.N, 3, 3)[:, body.rows]
        for t in range(T):
            mins.append(torch.linalg.det(wp.to_torch(tr.F[t]).reshape(-1, 3, 3) + dcs[t]).min())
            mins.append(torch.linalg.det(wp.to_torch(tr.F[t + 1]).reshape(-1, 3, 3)).min())
            if tr.volume_exact:
                mins.append(wp.to_torch(tr.J[t + 1]).min())
    return float(torch.stack(mins).min())


def _read(win):
    tr, T = win.tr, win.T
    X = torch.stack([tr.cat("x", t) for t in range(1, T + 1)])
    V = torch.stack([tr.cat("v", t) for t in range(1, T + 1)])
    FT = torch.cat([wp.to_torch(b.F[T]).reshape(-1, 9) for b in tr.trajs])
    return X, V, FT


def eval_terms(win, obj, leaf: torch.Tensor, u: torch.Tensor) -> Eval:
    with torch.no_grad():
        dc = win.load(leaf, u)
        with timed("mpm_eval"):
            win.tr.run()
        X, V, FT = _read(win)
        with timed("eval_losses"):
            e = obj.evaluate(X[-1].clone(), FT, V[-1].clone(), dc, V, X)
        with timed("eval_checks"):
            e.jt = _min_det(win, dc)
            e.in_domain = win.positions_in_domain()
        return e


def commit_rollout(win, obj, leaf, u, lam_r: float) -> Commit:
    """The accepted controls rolled out once more (the contact's impulse record starts at zero here); the per-step
    states are copies concatenated in body order."""
    with torch.no_grad():
        win.tr.impulse.zero_()
        e = eval_terms(win, obj, leaf, u)
        E_final = obj.scalar(e, lam_r)
        tr, T = win.tr, win.T
        F = [torch.cat([wp.to_torch(b.F[t]).reshape(-1, 3, 3) for b in tr.trajs]) for t in range(T + 1)]
        inv_any, jmin = None, float("inf")
        for t in range(1, T + 1):
            det_t = torch.linalg.det(F[t].float())
            bad = det_t <= 0.0
            if tr.trajs[0].volume_exact:
                bad = bad | (tr.cat("J", t) <= 0.0)
            inv_any = bad if inv_any is None else (inv_any | bad)
            jmin = min(jmin, float(det_t.min()))
        vx = tr.trajs[0].volume_exact
        J = [tr.cat("J", t) for t in range(T + 1)] if vx else None
        return Commit(x=[tr.cat("x", t) for t in range(T + 1)], F=F, end_F=F[T].clone(),
                      end_v=tr.cat("v", T), end_C=tr.cat("C", T).reshape(-1, 3, 3),
                      n_inv_steps=int(inv_any.sum()), jmin_traj=jmin, E_final=E_final, jt_final=e.jt,
                      valid=state_ok(e), owner=win, reason=state_reason(e), J=J, end_J=J[T].clone() if vx else None)
