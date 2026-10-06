"""Acceptance of committed windows and the stopping rule.

Progress, acceptance and the final delivery all read one number, the window's selection
merit (the full objective at the committed end state, cleanup in common geometry). A
window that raises it by more than 5 % is rejected and the state restored; after a
sustained plateau the gate latches and also rejects low-gain windows and low-gain
reversals. The run stops at the best window after `reject_stop` consecutive rejections,
after `patience` windows without a merit improvement, or at the window budget. The render
weight is calibrated at every window; the merit is linear in it, so the references are
rescored with the judged window's weight and every comparison is made under one weight.
"""
from __future__ import annotations

import numpy as np

from ..config import PipelineConfig


class Selection:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.epoch = 0
        self.lam = None                  # the render weight the references are scored with
        self._render = 0.0               # the judged window's render term
        self.stale = 0
        self.anneal = 1.0                # the next window's step scale
        self.reject_streak = 0
        self.last_reject_score = None
        self.new_epoch(count=False)

    def new_epoch(self, count=True):
        """Costs are comparable only within one set of render images: reset the references."""
        if count:
            self.epoch += 1
        self.best, self.last = float("inf"), None
        self.scales = self.prev = self.prev_phys = self.prev_disp = None
        self.best_render = self.prev_render = 0.0
        self.latched = False

    def plateau(self, a: int) -> bool:
        return self.last is not None and a - self.last > self.cfg.patience

    def check_lambda(self, rec: dict, lam: float, render: float = 0.0):
        """The merit is linear in the render weight: when the weight moves, the references (the last accepted
        window's merit and the best one) are rescored with it from their own render terms, and the window is
        judged against them as before. `render` is this window's render term (silhouette plus shading)."""
        if np.isfinite(rec["selection_merit"]):
            if self.lam is not None and lam != self.lam:
                if self.prev is not None:
                    self.prev += (lam - self.lam) * self.prev_render
                if np.isfinite(self.best):
                    self.best += (lam - self.lam) * self.best_render
            self.lam, self._render = lam, render
        rec["selection_epoch"] = self.epoch

    def judge(self, rec: dict, components: dict, disp) -> tuple[bool, bool, bool]:
        """(outer_reject, brake_reject, improved) for a committed window; writes the gate's
        record fields."""
        cfg, q = self.cfg, rec["selection_merit"]
        reversal_cos = None
        if self.prev_disp is not None:
            reversal_cos = float((disp * self.prev_disp).sum()
                                 / max(float(disp.norm()) * float(self.prev_disp.norm()), 1e-12))
        rec["reversal_cos"] = reversal_cos
        improved = bool(np.isfinite(q) and (self.last is None or q < self.best - cfg.tol * abs(self.best)))
        invalid = not np.isfinite(rec["transport_energy"]) or not np.isfinite(q)
        outer_reject = brake_reject = invalid
        outer_gain = -float("inf") if invalid else None
        if self.scales is None and not invalid:
            self.scales = {k: max(abs(v), 1e-8) for k, v in components.items()}
        score = float("inf") if invalid else float(q)
        score_phys = float("inf") if invalid else float(components["phys"] / self.scales["phys"])
        phys_gain = -float("inf") if invalid else None
        if self.prev is not None and not invalid:
            outer_gain = (self.prev - score) / max(abs(self.prev), 1e-8)
            phys_gain = ((self.prev_phys - score_phys) / max(abs(self.prev_phys), 1e-8)
                         if self.prev_phys is not None else outer_gain)
            # the latch needs a SUSTAINED plateau (a third consecutive non-improving
            # window); a real improvement releases it
            self.latched = self.latched or ((not improved) and self.stale >= 2)
            if improved:
                self.latched = False
            outer_reject = self.latched and outer_gain < cfg.outer_merit_tol
            brake_reject = outer_gain < -0.05                # a runaway, never a trade
            outer_reject = outer_reject or brake_reject
            if (self.latched and reversal_cos is not None and reversal_cos < cfg.outer_reversal_cos
                    and outer_gain < cfg.outer_reversal_gain):
                outer_reject = True
        rec.update({"outer_merit": score, "outer_gain": outer_gain, "outer_merit_phys": score_phys,
                    "phys_gain": phys_gain, "outer_gate_latched": int(self.latched),
                    "outer_accepted": 0 if outer_reject else 1})
        self._score, self._score_phys = score, score_phys
        return outer_reject, brake_reject, improved

    def rejected(self, rec: dict, brake_reject: bool, replay_rel: float) -> bool:
        """Book a rejected window; True when the run should stop. A brake reject is one
        insane candidate (free), unless it REPLAYS the previous rejected merit."""
        cfg, score = self.cfg, self._score
        self.reject_streak += 1
        tol = max(cfg.outer_merit_tol, 10.0 * float(replay_rel or 0.0))
        replay = (self.last_reject_score is not None
                  and abs(score - self.last_reject_score) <= tol * max(abs(self.last_reject_score), 1e-8))
        self.last_reject_score = score
        rec.update({"reject_streak": self.reject_streak, "replay": int(replay)})
        if not brake_reject or replay:
            self.stale += 1
        self.anneal = max(0.05, self.anneal * cfg.anneal_stale)
        return self.stale >= cfg.patience or (cfg.reject_stop > 0 and self.reject_streak >= cfg.reject_stop)

    def accepted(self, rec: dict, a: int, disp, improved: bool) -> bool:
        """Book an accepted window; True when the run has converged (patience spent)."""
        cfg, q = self.cfg, rec["selection_merit"]
        self.prev, self.prev_phys, self.prev_disp = self._score, self._score_phys, disp.clone()
        self.prev_render = self._render
        if self.last is None or q < self.best - cfg.tol * abs(self.best):
            self.best, self.last, self.best_render = q, a, self._render
        self.reject_streak, self.last_reject_score = 0, None
        self.stale = 0 if improved else self.stale + 1
        self.anneal = min(1.0, self.anneal * 1.15) if improved else max(0.05, self.anneal * cfg.anneal_stale)
        rec.update({"improved": int(improved), "stale": self.stale, "anneal": self.anneal, "pace_bound": 0})
        return self.stale >= cfg.patience

    def null(self) -> bool:
        """A window with no accepted step (no simulated time); True when patience is spent."""
        self.stale += 1
        self.reject_streak, self.last_reject_score = 0, None
        self.anneal = max(0.05, self.anneal * self.cfg.anneal_stale)
        return self.stale >= self.cfg.patience


def best_window(hist: list, n_frames: int, tol: float, w_pbr: float = 1.0):
    """The delivered slice: up to the best-merit accepted window of the last epoch, every window's merit
    scored with the render weight of the last one (the merit is linear in it). At the epoch's largest weight
    instead (D103) the 300k runs end a few windows earlier with a slightly lower silhouette term of their own,
    but no better display against an independent sample and less at rest (angular momentum at the end above
    the twin's): not kept. Returns (deliver_n, truncation record or None)."""
    acc = [r for r in hist if r.get("frame_end") and not r.get("null_commit")
           and r.get("d_vol") is not None and np.isfinite(r.get("selection_merit", float("nan")))]
    if not acc:
        return n_frames, None
    epoch = max(r["selection_epoch"] for r in acc)
    acc = [r for r in acc if r["selection_epoch"] == epoch]
    lam = acc[-1].get("lambda") or 0.0
    merit = lambda r: (r["selection_merit"] + (lam - (r.get("lambda") or 0.0))  # noqa: E731
                       * ((r.get("d_render") or 0.0) + w_pbr * (r.get("d_pbr") or 0.0)))
    best = min(acc, key=merit)
    if best["frame_end"] < n_frames or (best is not acc[-1] and merit(acc[-1]) > merit(best) * (1 + tol)):
        n = int(best["frame_end"])
        return n, {"best_animation": int(best["animation"]) + 1, "frames_kept": n,
                   "frames_dropped": n_frames - n}
    return n_frames, None
