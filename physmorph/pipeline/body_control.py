"""Cell-scale external force, parameterised by nominal window displacement.

The basis is frozen at window start. The pulse has zero impulse and unit
displacement for a free particle under semi-implicit Euler; elasticity, contact
and damping determine the actual motion.
"""
from __future__ import annotations

import numpy as np
import torch


def rest_to_rest_pulse(T: int, dt: float) -> np.ndarray:
    if T < 2 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("body control needs T >= 2 and positive finite dt")
    q = T - 1 - 2 * np.arange(T, dtype=np.float64)
    return (q / (dt * dt * np.dot(T - np.arange(T), q))).astype(np.float32)


class BodyControlBasis:
    """Trilinear vector coefficients on occupied nodes of the MPM lattice."""

    def __init__(self, x: np.ndarray, origin, dx: float, device="cpu"):
        x = np.asarray(x, np.float32)
        if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all() or dx <= 0:
            raise ValueError("finite (N,3) positions and positive dx required")
        r = (x - np.asarray(origin, np.float32)) / float(dx)
        base = np.floor(r).astype(np.int64)
        f = r - base
        offsets = np.asarray([(i, j, k) for i in (0, 1) for j in (0, 1)
                              for k in (0, 1)], np.int64)
        nodes, inv = np.unique((base[:, None] + offsets).reshape(-1, 3),
                               axis=0, return_inverse=True)
        weights = np.prod(np.where(offsets[None] != 0, f[:, None], 1 - f[:, None]), axis=2)
        self.idx = torch.as_tensor(inv.reshape(len(x), 8), device=device)
        self.weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
        self.n_nodes, self.dx, self.device = len(nodes), float(dx), device

    def zeros(self):
        return torch.zeros(self.n_nodes, 3, device=self.device, requires_grad=True)

    def expand(self, coefficients):
        return self.dx * (coefficients[self.idx] * self.weights[..., None]).sum(1)
