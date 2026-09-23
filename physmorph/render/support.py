"""Target-free opacity validity for material-bound render primitives.

Physics particles remain authoritative.  These weights only say whether one of those
particles still has enough local support to represent a continuum Gaussian.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _smoothstep(x, lo: float, hi: float):
    t = np.clip((x - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass(frozen=True)
class MaterialSupport:
    """Frozen source-material neighbours for render-only support checks."""

    neighbor: np.ndarray
    rest_length: np.ndarray

    @classmethod
    def from_rest(cls, rest_x: np.ndarray, k: int = 8) -> "MaterialSupport":
        from scipy.spatial import cKDTree

        rest = np.ascontiguousarray(rest_x, np.float32)
        if rest.ndim != 2 or rest.shape[1] != 3 or len(rest) < 2:
            raise ValueError("rest_x must have shape (N,3) with N >= 2")
        degree = min(max(int(k), 1), len(rest) - 1)
        dist, idx = cKDTree(rest).query(rest, k=degree + 1, workers=-1)
        return cls(np.ascontiguousarray(idx[:, 1:], np.int64),
                   np.ascontiguousarray(dist[:, 1:], np.float32))

    def opacity(self, x: np.ndarray, isolation_on: float = 4.0,
                isolation_off: float = 5.0, stretch_on: float = 1.75,
                stretch_off: float = 2.0, support_full: float = 2.0) -> np.ndarray:
        """Return ``[0,1]`` opacity multipliers without consulting the target.

        A primitive fades only when it is both an extreme current-body NN outlier and
        fewer than ``support_full`` frozen material bonds remain below the permitted
        stretch ramp.  The result is intentionally non-differentiable and must
        not become an optimisation reward for tearing material apart.
        """
        from scipy.spatial import cKDTree

        cur = np.ascontiguousarray(x, np.float32)
        n, degree = self.neighbor.shape
        if cur.shape != (n, 3):
            raise ValueError("x must have shape (N,3) matching the material graph")
        if isolation_off <= isolation_on:
            raise ValueError("isolation_off must be greater than isolation_on")
        if stretch_off <= stretch_on:
            raise ValueError("stretch_off must be greater than stretch_on")
        keep = min(max(float(support_full), 1.0), float(degree))
        dist = cKDTree(cur).query(cur, k=2, workers=-1)[0]
        nn_ratio = dist[:, 1] / max(float(np.median(dist[:, 1])), 1e-12)
        edge = cur[self.neighbor] - cur[:, None, :]
        stretch = np.linalg.norm(edge, axis=2) / np.maximum(self.rest_length, 1e-8)
        retained = (1.0 - _smoothstep(stretch, float(stretch_on),
                                      float(stretch_off))).sum(1)
        isolated = _smoothstep(nn_ratio, float(isolation_on), float(isolation_off))
        unsupported = 1.0 - _smoothstep(retained, 0.0, keep)
        return np.ascontiguousarray(1.0 - isolated * unsupported, np.float32)
