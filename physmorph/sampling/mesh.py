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


def sample_volume_shell(mesh: trimesh.Trimesh, n: int, shell_thickness: float, ratio: float = 6.0,
                        seed: int = 0, vox_res: int = 110):
    """SHELL-BIASED volume sampling (the C++ oracle's LoadShellBiasedMPMPointCloudFromObj):
    a surface shell of the given thickness (mesh units) is sampled at spacing h_s and the
    interior at spacing h_i = ratio * h_s, with n = V_shell / h_s^3 + V_int / h_i^3 solved
    for h_s. Returns (points, weights): weights are the particles' relative rest volumes
    (mean 1) — shell particles carry h_s^3, interior particles (ratio h_s)^3 — so the
    rasterised density stays uniform and the MPM rest-volume pass (eq 10) is consistent.
    With the MPM cell dx the shell then holds (dx / h_s)^3 particles per cell: the ratio the
    decoupling gap is measured in (docs/method.md 10.9)."""
    from scipy.ndimage import binary_erosion
    rng = np.random.default_rng(seed)
    ext = float(mesh.extents.max())
    pitch = ext / vox_res
    vg, M = _fill_grid(mesh, pitch)
    if M is None or int(M.sum()) == 0:
        raise ValueError("volumetric fill produced no voxels")
    k = max(1, int(round(shell_thickness / pitch)))
    interior = binary_erosion(M, iterations=k)
    shell = M & ~interior
    v_s = float(shell.sum()) * pitch ** 3
    v_i = float(interior.sum()) * pitch ** 3
    h_s = ((v_s + v_i / ratio ** 3) / float(n)) ** (1.0 / 3.0)
    n_s = int(round(v_s / h_s ** 3))
    n_s = min(max(n_s, 1), n - (1 if v_i > 0 else 0))
    n_i = n - n_s
    cs = vg.indices_to_points(np.argwhere(shell)).astype(np.float32)
    ci = vg.indices_to_points(np.argwhere(interior)).astype(np.float32) if n_i > 0 else np.zeros((0, 3), np.float32)
    def draw(centers, count):
        if count <= 0 or len(centers) == 0:
            return np.zeros((0, 3), np.float32)
        idx = rng.integers(0, len(centers), count)
        jit = (rng.uniform(-0.5, 0.5, (count, 3)) * pitch).astype(np.float32)
        return (centers[idx] + jit).astype(np.float32)
    xs, xi = draw(cs, n_s), draw(ci, n_i)
    x = np.concatenate([xs, xi]).astype(np.float32)
    w = np.concatenate([np.full(len(xs), v_s / max(n_s, 1), np.float32),
                        np.full(len(xi), v_i / max(n_i, 1), np.float32)])
    w = (w / w.mean()).astype(np.float32)
    print(f"[sampling] shell-biased: shell {n_s} pts (h_s={h_s:.4g}), interior {n_i} pts "
          f"(h_i={ratio * h_s:.4g}), shell volume {v_s / (v_s + v_i) * 100:.0f}% of the body", flush=True)
    return x, w


def _fill_grid(mesh: trimesh.Trimesh, pitch: float):
    """The filled voxel grid behind _fill_centers: (VoxelGrid, boolean matrix)."""
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
                M, _ = _strip_streaks(f.matrix.copy(), surf)
                return f, M
        return None, None
    except Exception:
        return None, None


def filled_volume(mesh: trimesh.Trimesh, vox_res: int = 110) -> float:
    """Volume of the filled voxelization (mesh units^3) - the SAME fill the sampler
    uses, so source/target volume matching is consistent with the particles."""
    ext = float(mesh.extents.max())
    pitch = ext / vox_res
    return float(len(_fill_centers(mesh, pitch))) * pitch ** 3


def load_normalized(path: str, n: int, seed: int = 1, size: float = 8.0,
                    match_volume: float | None = None,
                    return_volume: bool = False,
                    shell: tuple[float, float] | None = None):
    """Sample n particles from a mesh, centred at the origin and scaled so the bbox
    diagonal is `size` — the normalisation every runner script used to duplicate.

    match_volume: if given, the cloud is RESCALED (about the origin) so its filled
    volume equals this value (the source's): isochoric MPM particles cannot change
    the body's total volume, so a target of a different volume is unreachable.
    return_volume: also return the cloud's filled volume in world units^3.
    shell: (ratio, thickness_wu) — shell-biased sampling (sample_volume_shell) with the
    shell thickness given in WORLD units; the return then carries the per-particle
    relative rest volumes as a third value (x, vol, w) / (x, w)."""
    mesh = load_mesh(path)
    # per-asset up-axis (physmorph/sampling/orientation.json): the collection mixes z-up and y-up meshes
    from .orientation import orient_name, rotation
    _o = orient_name(path)
    if _o != "id":
        mesh.vertices = np.asarray(mesh.vertices, np.float64) @ rotation(_o).T
    if shell is not None:
        ratio, thick_wu = shell
        # the world scale before sampling (bbox diagonal -> size, then the volume match)
        ext = np.asarray(mesh.extents, np.float64)
        s0 = size / (float(np.linalg.norm(ext)) + 1e-9)
        vol0 = filled_volume(mesh) * float(s0) ** 3
        k0 = float((match_volume / vol0) ** (1.0 / 3.0)) if (match_volume is not None and vol0 > 0) else 1.0
        x, w = sample_volume_shell(mesh, n, thick_wu / (s0 * k0), ratio, seed=seed)
        x = x.astype(np.float32)
    else:
        x = sample_volume(mesh, n, seed=seed).astype(np.float32)
        w = None
    x -= x.mean(0)
    s = size / (np.linalg.norm(x.max(0) - x.min(0)) + 1e-9)
    x = (x * s).astype(np.float32)
    vol = filled_volume(mesh) * float(s) ** 3
    if match_volume is not None and vol > 0:
        k = float((match_volume / vol) ** (1.0 / 3.0))
        x = (x * k).astype(np.float32)
        vol = vol * k ** 3
    if shell is not None:
        return (x, vol, w) if return_volume else (x, w)
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
