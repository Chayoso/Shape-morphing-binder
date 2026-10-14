"""The scene's objective (D147): the sum over the bodies of the frozen single-body objective, each read in its body's own
frame. Positions enter as y = R^T (x - c) (the Buckens frame, frame.body_frame, differentiated exactly), velocities as
the deformation velocity v - (v_c + w x r) (frame.deformation_velocity, every step from its own positions), so gravity,
contact and spin move a body without a charge, and the controls shape it. The render terms of a body are drawn at y: the
eighteen views and the target pictures ride the body, and each body is drawn alone.
"""
from __future__ import annotations

import torch

from ..window.objective import Objective
from ..window.rollout import Eval
from .window import SceneWindow


class _Discs(tuple):
    """The bodies' disc sets (one entry per body, None until found); p0 counts them all (the optimiser's record)."""

    @property
    def p0(self):
        return range(sum(len(d.p0) for d in self if d is not None))


class SceneObjective:
    def __init__(self, win: SceneWindow):
        self.win, self.cfg = win, win.cfg
        self.parts = [Objective(v) for v in win.views]
        self.horizon = self.parts[0].horizon
        self.u_gate_frac = sum(p.u_gate_frac for p in self.parts) / len(self.parts)

    # the exterior's discs of each body (the optimiser keeps the set the accepted candidate was scored with)
    @property
    def discs(self):
        return _Discs(p.discs for p in self.parts)

    @discs.setter
    def discs(self, value):
        for p, d in zip(self.parts, value if value is not None else (None,) * len(self.parts)):
            p.discs = d

    @property
    def ext_builds(self) -> int:
        return sum(p.ext_builds for p in self.parts)

    def frame(self, b: int, x: torch.Tensor) -> torch.Tensor:
        """Body b's positions in its own frame (x: every body's rows)."""
        body = self.win.bodies[b]
        return body.frame(x[body.rows])[0]

    def evaluate(self, xT, FT, vT, dfc, V, X, **kw) -> Eval:
        """The window's terms summed over the bodies, as window/rollout._evaluate (records over every particle)."""
        T = self.cfg.T
        lv = lk = lr = lpbr = d_sil = lstab = 0.0
        for b, (part, body) in enumerate(zip(self.parts, self.win.bodies)):
            r = body.rows
            Vb = torch.stack([body.deformation(X[t, r], V[t, r]) for t in range(V.shape[0])])
            out = part.losses(self.frame(b, xT), FT[r], body.deformation(xT[r], vT[r]), Vb)
            lv, lk, lr, lpbr = lv + out[0], lk + out[1], lr + out[2], lpbr + out[3]
            d_sil, lstab = d_sil + out[4], lstab + out[5]
        norm = V.shape[0] * V.shape[1]
        lk_drv = (V[:T] - V[:T].mean(0)).square().sum() / norm
        lk_rel = V[T:].square().sum() / norm
        return Eval(xT, FT, vT, lv, lk, lr, lpbr, d_sil, dfc, V, V.pow(2).sum(2).mean(), lk_drv + lk_rel,
                    lstab=lstab, lk_drv=lk_drv, lk_rel=lk_rel, **kw)

    def phys_core(self, e):
        return e.lv + e.lstab

    def cleanup(self, xT, common_geometry=False):
        return sum(p.cleanup(self.frame(b, xT), common_geometry) for b, p in enumerate(self.parts))

    def near_band_far(self, xT) -> float:
        return sum(p.near_band_far(self.frame(b, xT)) for b, p in enumerate(self.parts))

    def scalar(self, e, lam_r) -> float:
        with torch.no_grad():
            L = float(self.phys_core(e) + self.cleanup(e.xT.detach()))
        return L + lam_r * float(e.lr.detach())

    def body_terms(self, xT) -> list:
        """A record: each body's terms at a committed state, in its frame (transport energy, silhouette, shading)."""
        out = []
        with torch.no_grad():
            for b, p in enumerate(self.parts):
                y = self.frame(b, xT)
                sil, pbr = p.render_terms(y)
                out.append({"transport": float(p.transport(y)), "d_sil": float(sil), "d_pbr": float(pbr)})
        return out
