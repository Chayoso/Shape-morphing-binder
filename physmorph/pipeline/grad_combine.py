"""Combining the physics and render control gradients (docs/render_controls_physics.md §5).

All functions take lists of per-leaf tensors (the joint gradient over leaves) and
return a list of the same shapes. They are search-DIRECTION rules only: the composite
is still line-searched on the fixed window objective, which is what guarantees descent
(a projection alone does not, for arbitrary lambda — adversarial finding 2026-09-01).

Modes (cfg.grad_project_mode):
  "render"   one-sided PCGrad (Yu et al. 2020, arXiv:2001.06782): strip from the RENDER
             gradient its component along the physics gradient when they conflict —
             physics descent is preserved (legacy behaviour).
  "phys"     the mirror image: strip from the PHYSICS gradient its conflicting component.
             Render-first: the image decides the direction inside the conflict cone.
  "cagrad"   two-task CAGrad (Liu et al. 2021, arXiv:2110.14048): d = g0 + (c·||g0||)·gw/||gw||
             where g0 = mean gradient and gw = argmin over the simplex of the worst-case
             improvement, solved exactly on the 1-D simplex by golden-section search.
  "blend"    magnitude anchored to the physics gradient, render only steers direction
             (the physics-domain evidence of arXiv:2607.25060: surgery rules inflate the
             primary metric; anchoring the step to the primary and letting the auxiliary
             steer does not): d = ||g_p|| · (ĝ_p + β ĝ_r) / ||ĝ_p + β ĝ_r|| with a FIXED
             β (blend_beta). REFUTE F3 (2026-09-15): with β = λ||g_r||/||g_p|| this is
             direction-identical to the plain sum; the mode is therefore defined with the
             fixed β, i.e. it is the norm-balanced composite WITHOUT the balancer's EMA and
             cap and with the step magnitude pinned to ||g_p||.
Note (arXiv:2609.01558): Adam's moments re-introduce conflict after any gradient-level
projection; the optimizer therefore also reports the cosine of the ACCEPTED step with
each channel (render_cos / phys_cos) so the effective conflict is measured, not assumed.
"""
from __future__ import annotations

import torch


def _dot(a, b) -> torch.Tensor:
    return sum((x * y).sum() for x, y in zip(a, b))


def _norm2(a) -> torch.Tensor:
    return sum(x.pow(2).sum() for x in a)


def pcgrad(g_keep, g_strip):
    """Strip from g_strip its component along g_keep when they conflict.
    Returns (g_strip', conflicted)."""
    dot = _dot(g_keep, g_strip)
    if float(dot) >= 0:
        return list(g_strip), False
    k2 = _norm2(g_keep).clamp_min(1e-30)
    return [b - (dot / k2) * a for a, b in zip(g_keep, g_strip)], True


def cagrad_two(g_a, g_b, c: float = 0.5, iters: int = 40):
    """Two-task CAGrad. g_a, g_b are the two task gradients (lists); returns the
    conflict-averse direction d (list) such that the update -d improves the average
    loss while guarding the worst task, with c in [0,1) the conservatism."""
    g0 = [0.5 * (a + b) for a, b in zip(g_a, g_b)]
    n0 = _norm2(g0).sqrt()
    if float(n0) <= 0:
        return list(g0)
    aa, bb, ab = _norm2(g_a), _norm2(g_b), _dot(g_a, g_b)
    ga0, gb0 = _dot(g_a, g0), _dot(g_b, g0)
    phi = (c * float(n0)) ** 2

    def objective(w):        # CAGrad's inner minimisation over the simplex weight w
        gw2 = float(w * w * aa + (1 - w) * (1 - w) * bb + 2 * w * (1 - w) * ab)
        gw_g0 = float(w * ga0 + (1 - w) * gb0)
        return gw_g0 + (phi * gw2) ** 0.5
    lo, hi = 0.0, 1.0                 # golden-section search: the objective is convex in w
    r = 0.6180339887
    x1, x2 = hi - r * (hi - lo), lo + r * (hi - lo)
    f1, f2 = objective(x1), objective(x2)
    for _ in range(iters):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - r * (hi - lo)
            f1 = objective(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + r * (hi - lo)
            f2 = objective(x2)
    w = 0.5 * (lo + hi)
    gw = [w * a + (1 - w) * b for a, b in zip(g_a, g_b)]
    ngw = _norm2(gw).sqrt().clamp_min(1e-30)
    lam = (phi ** 0.5) / ngw
    return [g + lam * h for g, h in zip(g0, gw)]


def blend_anchored(g_p, g_r, beta: float):
    """Physics-anchored blend: direction of (ĝ_p + beta ĝ_r), magnitude ||g_p||."""
    np_ = _norm2(g_p).sqrt().clamp_min(1e-30)
    nr_ = _norm2(g_r).sqrt().clamp_min(1e-30)
    d = [a / np_ + float(beta) * b / nr_ for a, b in zip(g_p, g_r)]
    nd = _norm2(d).sqrt().clamp_min(1e-30)
    return [x * (np_ / nd) for x in d]


def combine(mode: str, g_p, g_r, lam_r: float, cagrad_c: float = 0.5,
            blend_beta: float = 0.5):
    """Composite control direction for the window: returns (g, info).
    g_p: physics-core gradient, g_r: render gradient (already smoothed/masked),
    lam_r: the balanced render weight. Modes as in the module docstring; "off" or an
    unknown-free 'none' returns the plain weighted sum."""
    info = {"conflicted": False, "mode": mode}
    if mode == "render":
        g_r2, conf = pcgrad(g_p, g_r)
        info["conflicted"] = conf
        return [a + lam_r * b for a, b in zip(g_p, g_r2)], info
    if mode == "phys":
        g_p2, conf = pcgrad([lam_r * b for b in g_r], g_p)
        info["conflicted"] = conf
        return [a + lam_r * b for a, b in zip(g_p2, g_r)], info
    if mode == "cagrad":
        info["conflicted"] = float(_dot(g_p, g_r)) < 0
        return cagrad_two(g_p, [lam_r * b for b in g_r], cagrad_c), info
    if mode == "blend":
        info["conflicted"] = float(_dot(g_p, g_r)) < 0
        info["beta"] = float(blend_beta)
        return blend_anchored(g_p, g_r, float(blend_beta)), info
    if mode in ("off", "none", "sum"):
        return [a + lam_r * b for a, b in zip(g_p, g_r)], info
    raise ValueError(f"unknown grad_project_mode {mode!r}")
