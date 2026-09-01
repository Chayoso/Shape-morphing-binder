"""Mesh loading + volumetric particle sampling (trimesh voxel-fill)."""
from __future__ import annotations

import numpy as np
import trimesh


def load_mesh(path: str) -> trimesh.Trimesh:
    m = trimesh.load(path, process=False, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    return m


def sample_volume(mesh: trimesh.Trimesh, n: int, seed: int = 0,
                  vox_res: int = 110) -> np.ndarray:
    """Uniform-ish volume sampling via voxel-fill + jittered centers.

    Fast and robust for non-watertight meshes (unlike ray-cast `contains`).
    Refines resolution once if too few interior voxels for n.
    """
    rng = np.random.default_rng(seed)
    ext = float(mesh.extents.max())
    centers = _fill_centers(mesh, ext / vox_res)
    if len(centers) < n and len(centers) > 0:
        centers = _fill_centers(mesh, ext / (vox_res * 2))  # finer
    if len(centers) == 0:
        centers = mesh.sample(n).astype(np.float32)          # last resort: surface
    pitch = ext / vox_res
    idx = rng.integers(0, len(centers), n)
    jitter = (rng.uniform(-0.5, 0.5, (n, 3)) * pitch).astype(np.float32)
    return (centers[idx] + jitter).astype(np.float32)


def load_normalized(path: str, n: int, seed: int = 1, size: float = 8.0) -> np.ndarray:
    """Sample n particles from a mesh, centred at the origin and scaled so the bbox
    diagonal is `size` — the normalisation every runner script used to duplicate."""
    x = sample_volume(load_mesh(path), n, seed=seed).astype(np.float32)
    x -= x.mean(0)
    return (x * (size / (np.linalg.norm(x.max(0) - x.min(0)) + 1e-9))).astype(np.float32)


def _fill_centers(mesh: trimesh.Trimesh, pitch: float) -> np.ndarray:
    try:
        return mesh.voxelized(pitch=pitch).fill().points.astype(np.float32)
    except Exception:
        return np.empty((0, 3), np.float32)
