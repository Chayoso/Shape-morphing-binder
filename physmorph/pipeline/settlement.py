"""Arrival evidence is evaluated on the accepted state, independently of pacing."""
import numpy as np


def accepted_arrivals(x, images, radius, fallback=None):
    x = np.asarray(x)
    if images is None or radius is None:
        return np.zeros(len(x), bool) if fallback is None else np.asarray(fallback, bool).copy()
    images = np.asarray(images)
    if images.shape != x.shape or not np.isfinite(radius) or radius <= 0:
        raise ValueError("arrival images must match positions and radius must be positive")
    return np.isfinite(x).all(1) & np.isfinite(images).all(1) & (np.linalg.norm(x - images, axis=1) <= radius)
