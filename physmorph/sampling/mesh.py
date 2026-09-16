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

    2026-09-03 forensic: trimesh's default `fill()` (hole-based flood) adds ZERO
    interior voxels for a non-watertight mesh (bunny.obj: Euler -3), so this
    function silently returned a SURFACE SHELL and every bunny run morphed a
    solid sphere into a hollow target. `_fill_centers` now uses the axis-based
    fills ('base', then 'orthographic') and VERIFIES that the fill added interior
    voxels; it raises instead of falling back to a surface sample.
    """
    rng = np.random.default_rng(seed)
    ext = float(mesh.extents.max())
    centers = _fill_centers(mesh, ext / vox_res)
    if len(centers) < n and len(centers) > 0:
        centers = _fill_centers(mesh, ext / (vox_res * 2))  # finer
    if len(centers) == 0:
        raise ValueError("volumetric fill produced no voxels - refusing to sample a "
                         "surface shell as a 'volume'")
    pitch = ext / vox_res
    idx = rng.integers(0, len(centers), n)
    jitter = (rng.uniform(-0.5, 0.5, (n, 3)) * pitch).astype(np.float32)
    return (centers[idx] + jitter).astype(np.float32)


def filled_volume(mesh: trimesh.Trimesh, vox_res: int = 110) -> float:
    """Volume of the filled voxelization (mesh units^3) - the SAME fill the sampler
    uses, so source/target volume matching is consistent with the particles."""
    ext = float(mesh.extents.max())
    pitch = ext / vox_res
    return float(len(_fill_centers(mesh, pitch))) * pitch ** 3


def load_normalized(path: str, n: int, seed: int = 1, size: float = 8.0,
                    match_volume: float | None = None,
                    return_volume: bool = False):
    """Sample n particles from a mesh, centred at the origin and scaled so the bbox
    diagonal is `size` — the normalisation every runner script used to duplicate.

    match_volume: if given, the cloud is RESCALED (about the origin) so its filled
    volume equals this value (the source's): isochoric MPM particles cannot change
    the body's total volume, so a target of a different volume is unreachable.
    return_volume: also return the cloud's filled volume in world units^3."""
    mesh = load_mesh(path)
    x = sample_volume(mesh, n, seed=seed).astype(np.float32)
    x -= x.mean(0)
    s = size / (np.linalg.norm(x.max(0) - x.min(0)) + 1e-9)
    x = (x * s).astype(np.float32)
    vol = filled_volume(mesh) * float(s) ** 3
    if match_volume is not None and vol > 0:
        k = float((match_volume / vol) ** (1.0 / 3.0))
        x = (x * k).astype(np.float32)
        vol = vol * k ** 3
    return (x, vol) if return_volume else x


STREAK_REPORT = {"stripped": 0, "method": None}   # last fill's streak count (tests, logs)


def _strip_streaks(M: np.ndarray, surf: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove interior voxels with <= 2 of 6 filled neighbours (1-voxel columns that an
    axis fill draws between unrelated surface voxels of a non-watertight mesh)."""
    from scipy import ndimage
    k = np.zeros((3, 3, 3), int)
    k[1, 1, 0] = k[1, 1, 2] = k[1, 0, 1] = k[1, 2, 1] = k[0, 1, 1] = k[2, 1, 1] = 1
    nb = ndimage.convolve(M.astype(int), k, mode="constant")
    streak = M & ~surf & (nb <= 2)
    return M & ~streak, int(streak.sum())


def _fill_centers(mesh: trimesh.Trimesh, pitch: float) -> np.ndarray:
    """Interior+surface voxel centres. Axis-based fills work on non-watertight
    meshes; a fill is accepted only if it added interior voxels (>= 30% of the
    surface count), so a silent shell can never pass as a volume again.

    2026-09-16 (user forensic on the 40k target): trimesh's 'base' fill leaves 1-voxel
    STREAKS on the non-watertight bunny (485 interior voxels in columns of 28-67 voxels
    along one index axis, 5 clusters); at 20k they sample as scattered points, at 40k as a
    dotted line above the ear. 'orthographic' (a voxel is filled only if it is enclosed in
    all three axis projections) has none, so it is tried first, and any line-like interior
    voxel that survives is stripped and counted in STREAK_REPORT."""
    try:
        vg = mesh.voxelized(pitch=pitch)
        n_surf = int(vg.filled_count)
        surf = vg.matrix.copy()
        for method in ("orthographic", "base", "holes"):
            try:
                f = vg.copy().fill(method=method)
            except Exception:
                continue
            if int(f.filled_count) - n_surf >= 0.3 * n_surf:
                M, n_streak = _strip_streaks(f.matrix.copy(), surf)
                STREAK_REPORT["stripped"], STREAK_REPORT["method"] = n_streak, method
                if n_streak:
                    print(f"[sampling] fill '{method}': stripped {n_streak} streak voxels", flush=True)
                idx = np.argwhere(M)
                return f.indices_to_points(idx).astype(np.float32)
        return np.zeros((0, 3), np.float32)
    except Exception:
        return np.zeros((0, 3), np.float32)
