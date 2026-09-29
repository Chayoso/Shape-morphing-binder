"""Local particle support bounded by the remaining transport error.

Uniform support pressure can shrink legitimate thin target features. Coupling
its cost B to transport E as E + E B / (E + B) keeps the extra cost below E
and removes that pressure as E tends to zero. This is a terminal objective,
not a whole-trajectory guarantee or a post-simulation particle correction.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from scipy.spatial import cKDTree


class TransportSupport:
    def __init__(self, target, weight: float):
        target = np.asarray(target)
        if target.ndim != 2 or target.shape[1] != 3 or len(target) < 2 or not np.isfinite(target).all():
            raise ValueError('support target must be finite (N,3), N >= 2')
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('support weight must be finite and nonnegative')
        self.weight = float(weight)
        k = min(32, len(target) - 1)
        d = cKDTree(target).query(target, k=k + 1, workers=4)[0][:, 1:]
        self.radius = float(np.median(d[:, min(7, k - 1)]))
        if not self.radius > 0:
            raise ValueError('support target must have positive neighbor spacing')
        self.h = .5 * self.radius
        rho = np.exp(-d * d / (2 * self.h * self.h)).sum(1)
        self.log_floor = float(np.log(.5 * np.median(rho)))

    def penalty(self, x):
        if x.ndim != 2 or x.shape[1] != 3 or len(x) < 2:
            raise ValueError('support positions must have shape (N,3), N >= 2')
        k = min(32, len(x) - 1)
        if x.is_cuda and x.dtype == torch.float32:
            from ..render.knn_gpu import knn_self_torch
            _, indices = knn_self_torch(x, k + 1)
        else:
            indices = torch.as_tensor(cKDTree(x.detach().cpu().numpy()).query(
                x.detach().cpu().numpy(), k=k + 1, workers=4)[1], device=x.device)
        # Exclude identity, not column zero: duplicate positions can tie with self.
        is_self = indices == torch.arange(len(x), device=x.device)[:, None]
        order = torch.argsort(is_self.to(torch.int8), dim=1, stable=True)
        neighbors = indices.gather(1, order)[:, :k]
        log_density = torch.logsumexp(
            -(x[neighbors] - x[:, None]).square().sum(2) / (2 * self.h * self.h), dim=1)
        return self.radius ** 2 * torch.relu(self.log_floor - log_density).square().mean()

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
