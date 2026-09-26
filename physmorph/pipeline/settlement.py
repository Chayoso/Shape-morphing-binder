"""Arrival evidence is evaluated on the accepted state, independently of pacing."""
import numpy as np


def accepted_arrivals(x, images, radius, fallback=None, *, require_start=False):
    x = np.asarray(x)
    if require_start and (images is None or radius is None or fallback is None
                          or np.shape(fallback) != (len(x),)):
        raise ValueError("confirmed arrival requires full plan images and a window-start mask")
    if images is None or radius is None:
        return np.zeros(len(x), bool) if fallback is None else np.asarray(fallback, bool).copy()
    images = np.asarray(images)
    if images.shape != x.shape or not np.isfinite(radius) or radius <= 0:
        raise ValueError("arrival images must match positions and radius must be positive")
    arrived = np.isfinite(x).all(1) & np.isfinite(images).all(1) & (np.linalg.norm(x - images, axis=1) <= radius)
    return arrived & np.asarray(fallback, bool) if require_start else arrived
