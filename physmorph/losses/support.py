"""Local particle support bounded by the remaining transport error.

Uniform support pressure can shrink legitimate thin target features. Coupling
its cost B to transport E as E + E B / (E + B) keeps the extra cost below E
and removes that pressure as E tends to zero. This is a terminal objective,
not a whole-trajectory guarantee or a post-simulation particle correction.

The floor is half the target's kernel density: its median (one global floor), or with
target_ref the target's own leave-one-out density continued to the particle's position,
f(x) = 1/2 (t(x) - K(0)), t the target's kernel sum at x over its k + 1 nearest target points
(Kelsall & Diggle 1995: both densities read at the same point with the same kernel, so a free
surface lowers both alike; Monaghan 2005: the kernel sum continues the leave-one-out sum with
an exponentially small error). At every target point it equals half the point's own
leave-one-out density, so the target itself never pays; it is continuous in x, where the
earlier form (the floor of the nearest target point) stepped by about 13 % for a typical
particle at a Voronoi boundary; and it falls to zero, no floor and no penalty, about 1.4
spacings outside the target, where the earlier form carried the nearest point's floor outward
and charged spray that the W1 cleanup handles. The global floor asks thin target features to
be denser than the target samples them.

The per-particle penalty of a log-density deficit t = log f - log s is the log form
relu(t)^2, or the ratio form relu(1 - s/f)^2: the missing fraction of the floor's local mass,
which is at most 1. The two agree as s -> f. The log form grows as the square of the distance
to the neighbours for an isolated particle, so one stray particle can carry B; under the ratio
form a particle adds at most radius^2 / N, and isolated particles are left to the W1 cleanup.
"""
from __future__ import annotations

import math

import torch

from .. import gpu

FORMS = ("log", "ratio")


def deficit_penalty(t: torch.Tensor, form: str) -> torch.Tensor:
    """Penalty of the log-density deficit t = log f - log s (zero above the floor)."""
    if form == "log":
        return torch.relu(t).square()
    return torch.relu(-torch.expm1(-t)).square()           # 1 - s/f = -expm1(-t)


class TransportSupport:
    def __init__(self, target: torch.Tensor, weight: float, target_ref: bool = False,
                 form: str = "log"):
        target = gpu.tensor(target, torch.float64)          # scalar geometry in float64, like cKDTree
        if target.ndim != 2 or target.shape[1] != 3 or len(target) < 2 or not bool(torch.isfinite(target).all()):
            raise ValueError('support target must be finite (N,3), N >= 2')
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('support weight must be finite and nonnegative')
        if form not in FORMS:
            raise ValueError(f"support form must be one of {FORMS}")
        self.weight = float(weight)
        self.target_ref = bool(target_ref)
        self.form = form
        k = min(32, len(target) - 1)
        tree = gpu.KNN(target)
        d = tree.query(target, k + 1)[0][:, 1:]                   # float64, like cKDTree
        self.radius = gpu.median(d[:, min(7, k - 1)])
        if not self.radius > 0:
            raise ValueError('support target must have positive neighbor spacing')
        self.h = .5 * self.radius
        rho = torch.exp(-d * d / (2 * self.h * self.h)).sum(1)
        self.log_floor = float(math.log(.5 * gpu.median(rho)))
        if self.target_ref:
            self.tree, self.k, self.target_pts = tree, k, target
            self.log_floor_pt = math.log(.5) + torch.log(rho)     # per target point, float64 (the floor there)

    def floor(self, x):
        """The log-density floor of every particle: a scalar, or with target_ref (N,) the log of
        1/2 (t(x) - K(0)), t the target's kernel sum at x over its k + 1 nearest target points
        (neighbour indices detached, the kernel differentiable in x). Where t(x) <= K(0) the
        floor is at or below zero: no floor, no penalty, and no gradient."""
        if not self.target_ref:
            return self.log_floor
        idx = self.tree.query(x.detach(), self.k + 1)[1].to(x.device)
        y = self.target_pts.to(device=x.device, dtype=x.dtype)
        t = torch.exp(-(x[:, None] - y[idx]).square().sum(2) / (2 * self.h * self.h)).sum(1)
        return torch.log((.5 * (t - 1.0)).clamp_min(torch.finfo(x.dtype).tiny))

    def log_density(self, x):
        from ..render.knn_gpu import knn_self_torch
        k = min(32, len(x) - 1)
        _, indices = knn_self_torch(x, k + 1)
        # Exclude identity, not column zero: duplicate positions can tie with self.
        is_self = indices == torch.arange(len(x), device=x.device)[:, None]
        order = torch.argsort(is_self.to(torch.int8), dim=1, stable=True)
        neighbors = indices.gather(1, order)[:, :k]
        return torch.logsumexp(
            -(x[neighbors] - x[:, None]).square().sum(2) / (2 * self.h * self.h), dim=1)

    def penalty_per_point(self, x):
        return self.radius ** 2 * deficit_penalty(self.floor(x) - self.log_density(x), self.form)

    def penalty(self, x):
        if x.ndim != 2 or x.shape[1] != 3 or len(x) < 2:
            raise ValueError('support positions must have shape (N,3), N >= 2')
        return self.radius ** 2 * deficit_penalty(self.floor(x) - self.log_density(x), self.form).mean()

    def __call__(self, energy, x):
        # Keep containment/failed-solve sentinels intact and avoid inf/inf.
        if self.weight == 0 or not bool(torch.isfinite(energy)) or float(energy.detach()) < 0:
            return energy
        # Only scalar arithmetic needs double precision. Divide before
        # multiplying, and divide through by large weights to avoid overflow.
        penalty, e = self.penalty(x).double(), energy.double()
        eps = 1e-12 * self.radius ** 2
        if self.weight >= 1:
            numerator = penalty
            denominator = penalty + e / self.weight + max(
                eps / self.weight, torch.finfo(torch.float64).tiny)
        else:
            numerator = self.weight * penalty
            denominator = e + numerator + eps
        return energy + (e * (numerator / denominator)).to(energy.dtype)
