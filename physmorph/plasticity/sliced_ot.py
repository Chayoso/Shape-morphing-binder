"""Sliced optimal transport (balanced assignment) displacement.

Unlike the Sinkhorn *barycentric* map (which averages target anchors -> particle
CLUSTERING -> uneven density), sliced-OT is a balanced, measure-preserving
transport: it matches source to target by rank along many random 1-D directions,
so a uniform source stays uniform (uniform final density). Pure numpy, O(K N logN).

Bonneel et al. 2015, "Sliced and Radon Wasserstein Barycenters".
"""
from __future__ import annotations

import numpy as np


def sliced_ot_displacement(x, y, n_dirs=48, seed=0, step=1.0) -> np.ndarray:
    """Per-particle displacement that transports source x toward target y, balanced.

    x: (N,3) source, y: (M,3) target. If M != N, y is resampled to N (balanced
    matching needs equal counts). Returns d (N,3) = step * mean over directions of
    the rank-matched 1-D transport.
    """
    x = np.ascontiguousarray(x, np.float32)
    y = np.ascontiguousarray(y, np.float32)
    rng = np.random.default_rng(seed)
    N = x.shape[0]
    if y.shape[0] != N:
        y = y[rng.integers(0, y.shape[0], N)]
    d = np.zeros((N, 3), np.float64)
    for _ in range(n_dirs):
        th = rng.standard_normal(3)
        th /= np.linalg.norm(th) + 1e-12
        xp = x @ th                                   # (N,) source projections
        yp = y @ th                                   # (N,) target projections
        xs = np.argsort(xp, kind="stable")            # source ranks
        ys = np.sort(yp)                              # target sorted by projection
        matched = np.empty(N, np.float64)
        matched[xs] = ys                              # rank r source <- rank r target
        d += np.outer(matched - xp, th)               # move along th by rank gap
    return (step * d / n_dirs).astype(np.float32)
