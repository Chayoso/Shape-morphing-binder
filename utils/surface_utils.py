"""
Surface-aware utilities for volumetric morphing.

This module provides a minimal fixed-surface extraction pipeline:
- reconstruct a proxy surface from the source particle cloud on a voxel grid
- map particles near that reconstructed surface to a persistent surface mask

The mask is then reused for surface-only rendering and correction while the
full particle set remains the volumetric state for MPM.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


def reconstruct_surface_from_particles(
    x: np.ndarray,
    resolution: int = 64,
    padding: float = 2.0,
    sigma: float = 1.5,
    level_ratio: float = 0.02,
):
    """
    Reconstruct a proxy surface from a particle cloud on a regular voxel grid.

    Returns:
        dict with keys:
          - verts: (V, 3) reconstructed surface vertices or None
          - faces: (F, 3) faces or None
          - origin: (3,)
          - spacing: float
          - level: float
          - density_max: float
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 3 or x.shape[0] == 0:
        raise ValueError("x must be (N,3) and non-empty")

    mins = x.min(axis=0) - padding
    maxs = x.max(axis=0) + padding
    spacing = float((maxs - mins).max() / max(int(resolution), 2))
    spacing = max(spacing, 1e-6)

    idx = np.clip(((x - mins) / spacing).astype(np.int32), 0, resolution - 1)
    density = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    for i in range(x.shape[0]):
        ix, iy, iz = idx[i]
        density[ix, iy, iz] += 1.0

    if sigma > 0:
        density = gaussian_filter(density, sigma=float(sigma))

    density_max = float(density.max())
    level = float(density_max * level_ratio)
    if density_max <= 1e-8 or level <= 0.0:
        return {
            'verts': None,
            'faces': None,
            'origin': mins.astype(np.float32),
            'spacing': spacing,
            'level': level,
            'density_max': density_max,
        }

    try:
        verts, faces, _, _ = marching_cubes(density, level=level)
        verts = verts * spacing + mins
    except Exception:
        verts, faces = None, None

    return {
        'verts': None if verts is None else verts.astype(np.float32),
        'faces': None if faces is None else faces.astype(np.int32),
        'origin': mins.astype(np.float32),
        'spacing': spacing,
        'level': level,
        'density_max': density_max,
    }


def build_fixed_surface_mask(
    x: np.ndarray,
    resolution: int = 64,
    padding: float = 2.0,
    sigma: float = 1.5,
    level_ratio: float = 0.02,
    threshold_mult: float = 1.75,
    min_surface_frac: float = 0.08,
    max_surface_frac: float = 0.40,
):
    """
    Build a persistent surface mask from a volumetric particle cloud.

    Steps:
      1. reconstruct a surface proxy on a 3D grid
      2. assign each particle a distance to the reconstructed surface
      3. keep particles close to that surface
    """
    x = np.asarray(x, dtype=np.float32)
    recon = reconstruct_surface_from_particles(
        x,
        resolution=resolution,
        padding=padding,
        sigma=sigma,
        level_ratio=level_ratio,
    )
    verts = recon['verts']
    spacing = float(recon['spacing'])
    N = x.shape[0]

    if verts is None or len(verts) == 0:
        mask = np.ones((N,), dtype=bool)
        recon.update({
            'surface_mask': mask,
            'surface_distance_threshold': 0.0,
            'surface_fraction': 1.0,
            'surface_distance_mean': 0.0,
        })
        return recon

    dist, _ = cKDTree(verts).query(x, k=1)
    dist = np.asarray(dist, dtype=np.float32)

    base_thr = float(threshold_mult * spacing)
    thr = base_thr
    mask = dist <= thr
    frac = float(mask.mean())

    if frac < min_surface_frac:
        thr = max(thr, float(np.quantile(dist, min_surface_frac)))
    if frac > max_surface_frac:
        thr = min(thr, float(np.quantile(dist, max_surface_frac)))

    mask = dist <= thr
    frac = float(mask.mean())

    recon.update({
        'surface_mask': mask.astype(bool),
        'surface_distance_threshold': float(thr),
        'surface_fraction': frac,
        'surface_distance_mean': float(dist.mean()),
    })
    return recon


__all__ = [
    'reconstruct_surface_from_particles',
    'build_fixed_surface_mask',
]
