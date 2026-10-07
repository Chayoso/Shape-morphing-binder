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

FORMS = ("log", "ratio")                          # TransportSupport; "proximity" selects SurfaceProximity


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

    def target_sum(self, x):
        """t(x): the target's kernel sum at every particle over its k + 1 nearest target points
        (neighbour indices detached, the kernel differentiable in x)."""
        idx = self.tree.query(x.detach(), self.k + 1)[1].to(x.device)
        y = self.target_pts.to(device=x.device, dtype=x.dtype)
        return torch.exp(-(x[:, None] - y[idx]).square().sum(2) / (2 * self.h * self.h)).sum(1)

    def floor(self, x):
        """The log-density floor of every particle: a scalar, or with target_ref (N,) the log of
        1/2 (t(x) - K(0)); where t(x) <= K(0) the floor is at or below zero (no floor), and its log
        is that of the smallest positive number."""
        if not self.target_ref:
            return self.log_floor
        return torch.log((.5 * (self.target_sum(x) - 1.0)).clamp_min(torch.finfo(x.dtype).tiny))

    def deficit(self, x):
        """The log-density deficit log f - log s of every particle. With target_ref the floor f
        may be at or below zero, and wherever it is at or below the body density s the deficit
        is exactly zero, on safe inputs: the penalty is zero there and its gradient finite (the
        penalty of a deficit near -100 is zero too, but its ratio form's gradient overflows)."""
        log_s = self.log_density(x)
        if not self.target_ref:
            return self.log_floor - log_s
        f = .5 * (self.target_sum(x) - 1.0)
        above = f > log_s.exp()
        return torch.where(above, torch.log(torch.where(above, f, torch.ones_like(f))) - log_s, torch.zeros_like(f))

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
        return self.radius ** 2 * deficit_penalty(self.deficit(x), self.form)

    def penalty(self, x):
        if x.ndim != 2 or x.shape[1] != 3 or len(x) < 2:
            raise ValueError('support positions must have shape (N,3), N >= 2')
        return self.radius ** 2 * deficit_penalty(self.deficit(x), self.form).mean()

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


class SurfaceProximity:
    """The fine part of the geometry objective read from the TARGET SURFACE, in place of the current-particle
    support: at every outer target point y (the census outer set of the target's volume sample, a shell about one
    spacing thick, area-uniform up to sampling noise), the kernel of the body's NEAREST particle K(d_min(y)) against
    half the kernel at one sampling pitch, K(sp)/2, with the support's kernel and h (h = r8/2) and sp the target's
    median nearest-neighbour spacing. Penalty radius^2 mean_y relu(1 - K(d_min) / (K(sp)/2))^2: length^2 like the
    transport, at most radius^2, zero where a body particle sits within about 1.5 spacings of y (the threshold
    sqrt(sp^2 + 2 h^2 ln 2) = 1.53 sp at h = sp, the hard metric's 1.5 spacings without a constant of its own) and
    largest at a gap several spacings wide. It asks the metric's question, "is there body material at this target
    sample", made differentiable: d_min is continuous in x (the nearest particle may change, its distance does not).
    A density ratio against the target's own density (D8) cannot see the 1.6-2 spacing gaps of thin features,
    because the target's density there is a fraction of a unit and the tails of body material 2-3 spacings away
    exceed half of it. The sampling pitch does not collapse at thin features. Mass supply is the transport's part;
    this term places the surface. No bound and no weight."""

    def __init__(self, target: torch.Tensor, local: torch.Tensor | None = None):
        """local (N,) (D122, a sample of two pitches): each target point's spacing over the base; the medians are then
        the base sample's, the outer set is taken at each point's own spacing, and the kernel at an outer point y is
        the base kernel at y's own scale (h_y = h local_y; the floor K(sp_y) / 2 is the same number at every scale), so
        the question "is there body material at this target sample" is asked at the sample's own pitch there."""
        from ..thin import outer_mask
        target = gpu.tensor(target, torch.float64)
        if target.ndim != 2 or target.shape[1] != 3 or len(target) < 2 or not bool(torch.isfinite(target).all()):
            raise ValueError('proximity target must be finite (N,3), N >= 2')
        self.k = k = min(32, len(target) - 1)
        d = gpu.KNN(target).query(target, k + 1)[0][:, 1:]
        if local is not None:
            d = d / local.to(d.dtype)[:, None]
        self.radius = gpu.median(d[:, min(7, k - 1)])
        if not self.radius > 0:
            raise ValueError('proximity target must have positive neighbor spacing')
        self.h = .5 * self.radius
        self.spacing = gpu.median(d[:, 0])
        self.floor = .5 * math.exp(-self.spacing ** 2 / (2 * self.h * self.h))   # K(sp) / 2
        outer = outer_mask(target.float(), self.spacing, local)
        self.y = target[outer]                                                   # (M, 3) float64
        self.h_y = None if local is None else (self.h * local.to(target.dtype)[outer]).contiguous()
        self.weight = None                                                       # no weight, no bound

    def nearest_kernel(self, x):
        """K(d_min(y)) at every outer target point, over its k nearest body particles (indices detached, the
        distance differentiable in x)."""
        k = min(self.k, len(x))
        idx = gpu.KNN(x.detach()).query(self.y, k)[1].to(x.device)
        y = self.y.to(device=x.device, dtype=x.dtype)
        d2 = (x[idx] - y[:, None]).square().sum(2).min(1).values
        if self.h_y is not None:
            return torch.exp(-d2 / (2 * self.h_y.to(device=x.device, dtype=x.dtype).square()))
        return torch.exp(-d2 / (2 * self.h * self.h))

    def penalty_per_point(self, x):
        return self.radius ** 2 * torch.relu(1.0 - self.nearest_kernel(x) / self.floor).square()

    def penalty(self, x):
        if x.ndim != 2 or x.shape[1] != 3 or len(x) < 2:
            raise ValueError('proximity positions must have shape (N,3), N >= 2')
        return self.penalty_per_point(x).mean()

    def __call__(self, energy, x):
        if not bool(torch.isfinite(energy)) or float(energy.detach()) < 0:
            return energy                                    # containment / failed-solve sentinels stay intact
        return energy + self.penalty(x).to(energy.dtype)
