"""One window of the scene (D147): the window optimiser with the scene's window and objective, u free of each body's six
rigid modes, and a commit at every window.

Time always advances in the scene: the physics must run the scene's whole length (a collision or a landing raises the
merit whatever the control does, so the single-body selection would reject and restart those windows). The line search
already makes each window's control no worse than its start control, zero or the warm start (kept only if it beats
zero), on the window's own objective, i.e. no worse than the uncontrolled continuation. A window with no accepted step,
or whose replay failed, commits the zero control's rollout.
"""
from __future__ import annotations

import torch
import warp as wp

from ..window.solve import WindowOptimizer, WindowResult
from .objective import SceneObjective
from .window import SceneWindow


class _FanOut:
    """One attribute set on every body's transport (repeat, warm, the solved potentials): the window optimiser reads and
    writes tgt.grid_ot as one object."""

    def __init__(self, items):
        object.__setattr__(self, "_items", items)

    def __setattr__(self, k, v):
        if k == "_solved":
            for it, s in zip(self._items, v if v is not None else (None,) * len(self._items)):
                it._solved = s
        else:
            for it in self._items:
                setattr(it, k, v)

    def __getattr__(self, k):
        return tuple(getattr(it, k) for it in self._items)


class SceneTarget:
    """What the window optimiser reads as its target: the bodies (each with its own TargetPack), the contact's friction,
    the carried step length, and the bodies' transports as one."""

    def __init__(self, bodies: list, mu: float):
        self.bodies, self.mu, self.settled_step = bodies, float(mu), None

    @property
    def grid_ot(self):
        ots = [b.tgt.grid_ot for b in self.bodies]
        return None if any(o is None for o in ots) else _FanOut(ots)


def _window(start, prm, cfg, tgt: SceneTarget, vol0, bonds):
    return SceneWindow(start, prm, cfg, tgt.bodies, vol0, bonds, tgt.mu)


class SceneOptimizer(WindowOptimizer):
    window_cls, objective_cls = staticmethod(_window), SceneObjective

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # u's displacement keeps no part along any body's six rigid modes (D94, per body: no body is moved or turned)
        with torch.no_grad():
            win = self.win
            g = wp.to_torch(win.tr.layer_ug) * win.lmask
            cols = []
            for body in win.bodies:
                r = torch.zeros_like(win.x0)
                r[body.rows] = win.x0[body.rows] - win.x0[body.rows].mean(0)
                on = torch.zeros_like(g)
                on[body.rows] = g[body.rows]
                cols += [on[:, None] * win.lnrm, on[:, None] * torch.linalg.cross(r, win.lnrm, dim=1)]
            A = torch.cat(cols, 1).double()
            self.rigid = (A, torch.linalg.pinv(A.T @ A))

    def commit(self, hist, replay_rel, grad_converged, ls_exhausted, L_start, leaf0) -> WindowResult:
        res = super().commit(hist, replay_rel, grad_converged, ls_exhausted, L_start, leaf0)
        if res.commit is not None:
            return res
        from .rollout import commit_rollout             # the uncontrolled continuation: time advances
        with torch.no_grad():
            self.dFc.zero_()
            self.u.zero_()
        c = commit_rollout(self.win, self.obj, self.dFc, self.u, self.lam_r)
        res.stats.update(zero_commit=1, dfc=self.dFc.detach()[:self.cfg.T].clone(), commit_reason=c.reason,
                         commit_E_final=float(c.E_final), selection_merit=float(c.E_final))
        res.commit = c if c.valid else None
        return res
