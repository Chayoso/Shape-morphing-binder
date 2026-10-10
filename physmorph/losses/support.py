"""The surface proximity: the fine part of the geometry objective, read from the target's surface.

(The current-particle support with its log and ratio deficit forms, the R1-R4 terms it replaced, was removed in the
cleanup after freeze-2026-10-09b; the tag keeps it.)
"""
from __future__ import annotations

import math

import torch

from .. import gpu


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

    def __init__(self, target: torch.Tensor):
        from ..thin import outer_mask
        target = gpu.tensor(target, torch.float64)
        if target.ndim != 2 or target.shape[1] != 3 or len(target) < 2 or not bool(torch.isfinite(target).all()):
            raise ValueError('proximity target must be finite (N,3), N >= 2')
        self.k = k = min(32, len(target) - 1)
        d = gpu.KNN(target).query(target, k + 1)[0][:, 1:]
        self.radius = gpu.median(d[:, min(7, k - 1)])
        if not self.radius > 0:
            raise ValueError('proximity target must have positive neighbor spacing')
        self.h = .5 * self.radius
        self.spacing = gpu.median(d[:, 0])
        self.floor = .5 * math.exp(-self.spacing ** 2 / (2 * self.h * self.h))   # K(sp) / 2
        outer = outer_mask(target.float(), self.spacing)
        self.y = target[outer]                                                   # (M, 3) float64

    def nearest_kernel(self, x):
        """K(d_min(y)) at every outer target point, over its k nearest body particles (indices detached, the
        distance differentiable in x)."""
        k = min(self.k, len(x))
        idx = gpu.KNN(x.detach()).query(self.y, k)[1].to(x.device)
        y = self.y.to(device=x.device, dtype=x.dtype)
        d2 = (x[idx] - y[:, None]).square().sum(2).min(1).values
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
