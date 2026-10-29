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
    def unit_ratio(self) -> float:
        """The line search's noise floor reads the loss unit: the bodies' targets are alike, body 0's."""
        return self.bodies[0].tgt.unit_ratio

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
        # D164 amendment 1 is single-body: here each body's terms call the transport in turn, so the potentials and lists
        # left after an evaluation are the last body's alone (the first body would read the second's). The scene's commit
        # solves and searches at its own rollout, as the tag's; its check reads D164's measure
        self.commit_reads_accepted = False
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
            with torch.no_grad():
                res.stats["L_split"] = _split_record(self.win, self.u.detach())
            return res
        from .rollout import commit_rollout             # the uncontrolled continuation: time advances
        with torch.no_grad():
            self.dFc.zero_()
            self.u.zero_()
        c = commit_rollout(self.win, self.obj, self.dFc, self.u, self.lam_r)
        res.stats.update(zero_commit=1, dfc=self.dFc.detach()[:self.cfg.T].clone(), commit_reason=c.reason,
                         commit_E_final=float(c.E_final), selection_merit=float(c.E_final))
        res.commit = c if c.valid else None
        if res.commit is not None:
            with torch.no_grad():
                res.stats["L_split"] = _split_record(self.win, self.u.detach())
        return res


def momentum_split(win, u: torch.Tensor) -> list:
    """A record: each body's change of angular momentum over the committed window (per unit mass, APIC's affine part
    included; window/telemetry.below_grid_record on the body's own trajectory), split into the grid's steps (the stress
    with the control, the contact, gravity, the floor), u's displacement, the minimum spacing's push and the rest of the
    updates below the grid (the relaxation, the bonds, the floor's position constraint); about the body's own start
    centre of mass and about the pair's start centroid (there the bodies' grid shares add up to what the contact and the
    stress keep). The point moves only the grid's share and the start: L about c = L about c_b + (c_b - c) x P (the
    affine part and the updates below the grid, (x' - x) x v, do not depend on it). Gravity's torque,
    sum_t dt sum_p (x_p - c) x g, is given apart for both points."""
    from types import SimpleNamespace
    from ..window.telemetry import below_grid_record
    du = (wp.to_torch(win.tr.layer_ug) * u * win.lmask)[:, None] * win.lnrm
    g = torch.tensor(win.prm.f_ext, dtype=torch.float64, device=u.device)
    dt = float(win.prm.dt)
    keys = ("L_start", "L_grid", "L_u", "L_space", "L_relax")
    c_pair = win.x0.mean(0).double()
    out = []
    for body, tr in zip(win.bodies, win.tr.trajs):
        r = body.rows
        b = below_grid_record(SimpleNamespace(tr=tr, T=win.T, N=body.n, x0=win.x0[r]), du[r])
        c_own = win.x0[r].mean(0).double()
        P0, PT = (wp.to_torch(tr.v[t]).double().sum(0) for t in (0, win.T))
        rec = {}
        for name, c in (("own", c_own), ("pair", c_pair)):
            s = c_own - c
            grav = sum(torch.linalg.cross(wp.to_torch(tr.x[t]).double() - c, g.expand(body.n, 3), dim=1).sum(0)
                       for t in range(win.T)) * dt
            shift = {"L_start": torch.linalg.cross(s, P0, dim=0), "L_grid": torch.linalg.cross(s, PT - P0, dim=0)}
            rec[name] = {k: (torch.tensor(b[k], dtype=torch.float64, device=u.device) + shift[k]).tolist()
                         if k in shift else b[k] for k in keys}
            rec[name]["L_gravity"] = grav.tolist()
        out.append(rec)
    return out


def _split_record(win, u):
    """momentum_split, which a record's failure must not cost the run (as pipeline_run's measurements)."""
    try:
        return momentum_split(win, u)
    except Exception as e:                              # noqa: BLE001
        print(f"[scene] momentum_split failed: {e!r}", flush=True)
        return None
