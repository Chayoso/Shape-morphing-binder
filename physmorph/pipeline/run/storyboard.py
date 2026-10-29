"""D175: picture keyframes, a render-only control of the trajectory (docs/design/keyframe_render_control.md; off by
default: cfg.keyframe_time () is the frozen path).

A keyframe k = (t_k, P_k): the pictures P_k in the run's views, drawn by the final pictures' exterior operators (the
final pack's pitch, kernel, lattice and camera box) from the keyframe's samples (its sample and its draws); the run reads
the samples only through the pictures. The storyboard is the keyframes plus the final target. Keyframe k's segment is
every window whose end frame lies in (frame(t_k-1), frame(t_k)], frame(t) = round(t / (2T dt)) 2T (Treuille et al. 2003's
segments); later windows read the final target.

One objective per window (McNamara et al. 2004's W_t): in a segment the render reads P_k, and the final target's geometry
(transport, surface proximity) and the cleanup that pulls toward its points and distance field are absent; only the
target-free soundness terms stay (the released motion, the end drift). The gradient, PCGrad's protected direction, the
Armijo scalar and the merit all read that objective (TargetPack.geometry False). The render weight is the run's unit: the
held weight (lambda_ema 0), calibrated once by a pre-pass at the source on the final target (calibrate), on both control
blocks inside a segment (solve._block_lams); after the last keyframe the run's own split (render_split u = Hu: held on u,
the rule's per-window value on dFc). The relaxation has no relief in a segment, u's gate reads the final target (its own
objective, unchanged). The storyboard owns the clock: a segment window with no accepted step commits its zero-control
rollout as a settle window does (run.runner); the stops and the delivery read the final target's epoch only; each new
entry opens a selection epoch. The state the run carries on the target pack (settled step, u's moments, the transport's
loss) stays on the one pack: the storyboard swaps the pictures, the relief and the two flags in place.

keyframe_geometry "final" (the ablation, arm N): only the pictures are swapped; the final geometry, cleanup and relief
are kept (the clock and the held weight as in the method)."""
from __future__ import annotations

import dataclasses

import numpy as np
import torch

from ..target import keyframe_pictures


def frame_of(t: float, T: int, dt: float) -> int:
    """The window boundary nearest to time t: round(t / (2T dt)) 2T (a commit adds 2T frames)."""
    return int(round(float(t) / (2 * T * float(dt)))) * 2 * T


def entry_of(end_frame: int, frames: list) -> int | None:
    """The storyboard entry of the window ending at end_frame: the first keyframe k with end_frame <= frame_k, else
    None (the final target)."""
    for k, f in enumerate(frames):
        if end_frame <= f:
            return k
    return None


class Storyboard:
    """final: the run's built TargetPack; keyframes: per cfg.keyframe_time, the list of the keyframe's samples ((N_k, 3)
    arrays in the target's frame); source: the source sample (CUDA tensor)."""

    def __init__(self, final, keyframes, source: torch.Tensor, prm, cfg, log=print):
        if final.ext is None:
            raise ValueError("keyframes are drawn on the exterior (render_exterior)")
        if keyframes is None or len(keyframes) != len(cfg.keyframe_time):
            raise ValueError("one keyframe (its samples) per keyframe_time")
        self.final_ext, self.final_relief = final.ext, final.relief
        self.own = cfg.keyframe_geometry == "own"
        self.frames = [frame_of(t, cfg.T, prm.dt) for t in cfg.keyframe_time]
        if self.frames[0] < 2 * cfg.T or any(b <= a for a, b in zip(self.frames, self.frames[1:])):
            raise ValueError(f"keyframe frames {self.frames}: each at least one window after the last")
        c0 = source.mean(0)
        self.exts = []
        for k, (t, f, smp) in enumerate(zip(cfg.keyframe_time, self.frames, keyframes)):
            # admissible: the centre of mass kept (dFc is an internal stress and u is rigid-free: neither moves it)
            ck = torch.stack([torch.as_tensor(np.asarray(s, np.float32)).mean(0) for s in smp]).mean(0).to(c0.device)
            off = float((ck - c0).norm())
            if off > 0.25 * float(final.ldx):
                raise ValueError(f"keyframe {k}: its centre is {off:.4g} wu from the source's (> a quarter loss cell); "
                                 "a keyframe must keep the centre of mass")
            sils, shade = keyframe_pictures(final, smp, cfg)
            self.exts.append(dataclasses.replace(final.ext, sils=sils, shade=shade))
            cov = float(torch.stack([(a > 0.5).float().mean() for a in sils]).mean())
            log(f"[keyframe] D175: keyframe {k} at t = {t:g} s = frame {f}, {len(smp)} sample(s) of {len(smp[0])} "
                f"points, centre {off:.2e} wu from the source's, mean silhouette coverage {cov:.4f}"
                + ("" if self.own else "; ablation: the final geometry kept (keyframe_geometry final)"))
        self.cur = "unset"

    def apply(self, tgt, end_frame: int):
        """Set the pack to the entry of the window ending at end_frame. Returns (the keyframe index or None, changed)."""
        k = entry_of(end_frame, self.frames)
        if k == self.cur:
            return k, False
        if k is None:
            tgt.ext, tgt.relief, tgt.geometry, tgt.keyframe = self.final_ext, self.final_relief, True, None
        else:
            tgt.ext, tgt.keyframe = self.exts[k], k
            tgt.relief = None if self.own else self.final_relief
            tgt.geometry = not self.own
        self.cur = k
        return k, True

    @staticmethod
    def calibrate(start, prm, cfg, tgt, balancer, vol0, bonds, log=print) -> None:
        """The render weight's pre-pass: one gradient at the source on the final target (the first window's
        calibration of a run without keyframes), so the held weight is the run's unit from the first segment on."""
        from ..window.rollout import graph_terms
        from ..window.solve import WindowOptimizer
        opt = WindowOptimizer(start, prm, cfg, tgt, balancer, vol0, bonds, 1.0, None, lambda *_: None)
        e = graph_terms(opt.win, opt.obj, opt.leaves[0], opt.u)
        opt.gradient(e, 0)
        if tgt.grid_ot is not None:
            tgt.grid_ot.repeat = False
        log(f"[keyframe] D175: the render weight from the pre-pass at the source on the final target: lambda = "
            f"{balancer.lam:.6g} (held), g_share {opt.tele.get('g_share'):.4f}, g_raw_cos {opt.tele.get('g_raw_cos'):.4f}")
        opt.win._adj = None
        del opt, e
