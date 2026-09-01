"""D_render — multi-view (azimuth x elevation) asymmetric soft-silhouette loss + λ balancing.

docs/pipeline_v2.md §3.4, §3.6. Builds on losses/silhouette.py CIC splat primitives.

Asymmetry: deficit INSIDE the target silhouette (w_hole) = holes / missing extremities;
excess OUTSIDE (w_spray) = ejecta — the render objective itself pulls strays back, replacing
the v1 clamps. Since relu(a_t-a)^2 + relu(a-a_t)^2 == (a-a_t)^2, w_hole=w_spray=1 recovers
the plain MSE of v1's d_img.
"""
from __future__ import annotations

import numpy as np
import torch

from ..losses.silhouette import soft_silhouette


def make_views(n_azim: int, elevs=(0.0, 0.5, -0.5)) -> list[tuple[float, float]]:
    """(theta, phi) view set: an azimuth ring per elevation. Each ring gets a DISTINCT
    azimuth offset (i/n_rings of one step) so no two rings stack in azimuth — the previous
    alternating offset re-stacked rings 0 and 2 for odd ring counts."""
    views = []
    n_rings = max(len(tuple(elevs)), 1)
    for i, phi in enumerate(elevs):
        off = (i / n_rings) * (2 * np.pi / max(n_azim, 1))
        for t in np.linspace(0, 2 * np.pi, n_azim, endpoint=False):
            views.append((float(t + off), float(phi)))
    return views


def target_silhouettes(target_x: torch.Tensor, views, res: int, extent: float, k=1.5):
    with torch.no_grad():
        return [soft_silhouette(target_x, th, res, extent, k, phi).detach() for th, phi in views]


def d_render(x: torch.Tensor, target_alphas, views, res: int, extent: float,
             k: float = 1.5, w_hole: float = 2.0, w_spray: float = 1.0) -> torch.Tensor:
    """Mean over views of the asymmetric per-pixel silhouette penalty."""
    loss = x.new_zeros(())
    for a_t, (th, phi) in zip(target_alphas, views):
        a = soft_silhouette(x, th, res, extent, k, phi)
        deficit = torch.clamp(a_t - a, min=0.0)     # hole / missing coverage
        excess = torch.clamp(a - a_t, min=0.0)      # spray / ejecta
        loss = loss + (w_hole * deficit.pow(2) + w_spray * excess.pow(2)).mean()
    return loss / len(views)


class LambdaBalancer:
    """λ_R = α_λ · ||∇_phys|| / ||∇_render||, EMA-smoothed so the balancer cannot oscillate.

    The C++ get_control_layer_grad_norm rule ("lambda = alpha * phys_norm / render_norm"),
    plus an EMA because the raw ratio is itself a noisy per-iteration quantity and feeding it
    straight back is one of the v1 oscillation sources. α_λ=0 disables the render channel."""

    def __init__(self, alpha_lam: float, ema: float = 0.3, cap: float | None = None):
        self.alpha_lam = float(alpha_lam)
        self.ema = float(ema)
        self.cap = cap
        self.lam = None

    @property
    def active(self) -> bool:
        return self.alpha_lam > 0.0

    def update(self, phys_norm: float, render_norm: float) -> float:
        target = self.alpha_lam * phys_norm / max(render_norm, 1e-12)
        # CAP: once D_render saturates its gradient vanishes and the raw ratio diverges —
        # observed live at full scale: λ 1.1e3 → 1.77e5 with a mid-window inversion in tow.
        # A converged render term should FADE, not take over the objective.
        if self.cap is not None:
            target = min(target, self.cap)
        self.lam = target if self.lam is None else (1 - self.ema) * self.lam + self.ema * target
        return self.lam
