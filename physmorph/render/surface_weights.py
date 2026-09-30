"""Persistent soft surface score of a cloud (viewer and renderer tooling)."""
from __future__ import annotations

import numpy as np


def surface_weights(x: np.ndarray, k: int, fraction: float, floor: float) -> np.ndarray:
    """Soft surface score from one-sided local neighbour directions: a volume-interior
    particle sees approximately cancelling unit directions, a boundary particle does not.
    Computed once in source/material coordinates, so the active set cannot flicker."""
    from scipy.spatial import cKDTree
    kk = min(len(x), max(2, 3 * int(k) + 1))  # tolerate duplicate voxel samples
    _, idx = cKDTree(x).query(x, k=kk, workers=-1)
    d = x[idx[:, 1:]] - x[:, None, :]
    dn = np.linalg.norm(d, axis=2, keepdims=True)
    valid = dn[..., 0] > 1e-8
    unit = d / np.maximum(dn, 1e-8)
    score = np.linalg.norm((unit * valid[..., None]).sum(1) /
                           np.maximum(valid.sum(1, keepdims=True), 1), axis=1)
    threshold = float(np.quantile(score, 1.0 - float(fraction)))
    width = max(float(np.std(score)) * 0.15, 1e-4)
    soft = 1.0 / (1.0 + np.exp(np.clip(-(score - threshold) / width, -40.0, 40.0)))
    return np.ascontiguousarray(floor + (1.0 - floor) * soft, np.float32)
