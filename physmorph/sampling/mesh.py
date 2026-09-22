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


def sample_volume_stratified(mesh: trimesh.Trimesh, n: int, seed: int = 0) -> np.ndarray:
    """G5 (docs/surface_gradient.md §4): ONE jittered particle per fill voxel, no drawing with
    replacement. The fill resolution is chosen (bisection) so that the fill holds at least n
    voxels; if it holds more, the surplus is dropped uniformly WITHOUT replacement. The cloud
    is a jittered lattice: the relative shot noise of the blurred density falls from
    1/sqrt(particles per blur volume) to the lattice's own (bunny 40k: layer plane-residual
    RMS 0.35 -> 0.29 spacings, NN-distance CV 0.37 -> 0.29; sampling_test 2026-09-19)."""
    rng = np.random.default_rng(seed)
    ext = float(mesh.extents.max())
    lo, hi = 20, 400
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if len(_fill_centers(mesh, ext / mid)) < n:
            lo = mid
        else:
            hi = mid
    centers = _fill_centers(mesh, ext / hi)
    if len(centers) < n:
        raise ValueError(f"stratified fill at {hi}^3 holds {len(centers)} < n = {n} voxels")
    pitch = ext / hi
    keep = rng.choice(len(centers), n, replace=False) if len(centers) > n else np.arange(n)
    jitter = (rng.uniform(-0.5, 0.5, (n, 3)) * pitch).astype(np.float32)
    print(f"[sampling] stratified: fill {hi}^3 = {len(centers)} voxels for n = {n}, pitch {pitch:.4g}", flush=True)
    return (centers[keep] + jitter).astype(np.float32)


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
        vgr, Mr, _, _ = _fill_reliable(mesh, pitch)
        if vgr is not None:
            return vgr, Mr
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
                M, _, _ = _fill_pockets(M)
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
                    shell: tuple[float, float] | None = None,
                    sample: str = "replacement"):
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
        x = (sample_volume_stratified(mesh, n, seed=seed) if sample == "stratified"
             else sample_volume(mesh, n, seed=seed)).astype(np.float32)
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
POCKET_REPORT = {"filled": 0, "iters": 0}         # last fill: sub-voxel pockets filled by _fill_pockets


def _fill_pockets(M: np.ndarray, max_iters: int = 20) -> tuple[np.ndarray, int, int]:
    """Fill the sub-voxel pockets an axis fill leaves in a NON-WATERTIGHT mesh (2026-09-22, docs/
    surface_gradient.md 14): an empty voxel with a MAJORITY (>= 4 of 6) of filled face-neighbours
    is interior, filled, and the rule is iterated to convergence. It closes 1-voxel pockets and
    1-voxel tunnels only (a 2-voxel slot has at most 1 filled neighbour per voxel) — features below
    the sampler resolution, which the fill cannot represent anyway. On the 40k stratified pitch:
    bunny 287 -> 2 pockets, dragon 159 -> 0, beast 74 -> 0, armadillo 81 -> 0 (+0.3-1 % voxels);
    watertight meshes are untouched (bob 6). Returns (matrix, filled count, iterations)."""
    from scipy import ndimage
    k = np.zeros((3, 3, 3), int)
    k[1, 1, 0] = k[1, 1, 2] = k[1, 0, 1] = k[1, 2, 1] = k[0, 1, 1] = k[2, 1, 1] = 1
    M = M.copy()
    total = 0
    it = 0
    for it in range(1, max_iters + 1):
        nb = ndimage.convolve(M.astype(int), k, mode="constant")
        add = (~M) & (nb >= 4)
        n_add = int(add.sum())
        if n_add == 0:
            it -= 1
            break
        M |= add
        total += n_add
    return M, total, it


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
        vgr, Mr, n_streak, n_pocket = _fill_reliable(mesh, pitch)      # 2026-09-22: holes -> reliable axes
        if vgr is not None:
            STREAK_REPORT["stripped"], STREAK_REPORT["method"] = n_streak, "ortho_reliable"
            POCKET_REPORT["filled"], POCKET_REPORT["iters"] = n_pocket, 0
            if n_streak or n_pocket:
                print(f"[sampling] fill 'ortho_reliable': stripped {n_streak} streaks, filled {n_pocket} sub-voxel pockets", flush=True)
            return vgr.indices_to_points(np.argwhere(Mr)).astype(np.float32)
        for method in ("orthographic", "base", "holes"):
            try:
                f = vg.copy().fill(method=method)
            except Exception:
                continue
            if int(f.filled_count) - n_surf >= 0.3 * n_surf:
                M, n_streak = _strip_streaks(f.matrix.copy(), surf)
                STREAK_REPORT["stripped"], STREAK_REPORT["method"] = n_streak, method
                M, n_pocket, n_it = _fill_pockets(M)
                POCKET_REPORT["filled"], POCKET_REPORT["iters"] = n_pocket, n_it
                if n_pocket:
                    print(f"[sampling] fill '{method}': filled {n_pocket} sub-voxel pockets ({n_it} iterations)", flush=True)
                if n_streak:
                    print(f"[sampling] fill '{method}': stripped {n_streak} streak voxels", flush=True)
                idx = np.argwhere(M)
                return f.indices_to_points(idx).astype(np.float32)
        return np.zeros((0, 3), np.float32)
    except Exception:
        return np.zeros((0, 3), np.float32)


def _hole_footprints(mesh: trimesh.Trimesh, vg) -> list:
    """Per axis, the 2-D footprint (in voxel indices of the two other axes) of the mesh's BOUNDARY
    LOOPS — the holes of a non-watertight mesh — rasterised and filled: the columns along that axis
    whose enclosure test is unreliable because a hole, not a surface, closes them. Returns a list of
    three boolean 2-D arrays (or None for an axis without holes), dilated by one voxel."""
    from scipy import ndimage
    shape = tuple(int(s) for s in vg.shape)
    edges = mesh.edges_sorted
    grp = trimesh.grouping.group_rows(edges, require_count=1)        # edges of exactly one face
    if len(grp) == 0:
        return [None, None, None]
    be = edges[grp]
    V = np.asarray(mesh.vertices, np.float64)
    pitch = float(np.max(vg.pitch)) if np.ndim(vg.pitch) else float(vg.pitch)
    a, b = V[be[:, 0]], V[be[:, 1]]
    seg = np.linalg.norm(b - a, axis=1)
    nseg = np.maximum(2, np.ceil(seg / (0.5 * pitch)).astype(int))
    pts = np.concatenate([a[i] + (b[i] - a[i]) * np.linspace(0.0, 1.0, nseg[i])[:, None] for i in range(len(be))])
    idx = np.asarray(vg.points_to_indices(pts), np.int64)
    out = []
    for ax in range(3):
        o1, o2 = [k for k in range(3) if k != ax]
        M = np.zeros((shape[o1], shape[o2]), bool)
        i1 = np.clip(idx[:, o1], 0, shape[o1] - 1); i2 = np.clip(idx[:, o2], 0, shape[o2] - 1)
        M[i1, i2] = True
        F = ndimage.binary_fill_holes(M)
        F = ndimage.binary_dilation(F, iterations=1)
        out.append(F if F.any() else None)
    return out


def _fill_ortho_reliable(surf: np.ndarray, footprints: list) -> np.ndarray:
    """The orthographic fill with RELIABLE axes only (2026-09-22; docs/surface_gradient.md 14): along
    each axis a voxel is enclosed if a surface voxel lies before and after it on its line; a column
    that runs through a hole's footprint (the mesh's boundary loops projected along that axis) has
    no surface to close it and is not asked — the voxel is filled if enclosed along every
    reliable axis. A watertight mesh has no footprints and this is the plain intersection (a torus
    hole stays open: two axes enclose it, the third does not). For a base hole (bunny, maxplanck)
    the columns above it are closed by the two side projections instead of being left as empty
    shafts — the 40 %-density comb the plain intersection produced."""
    enc = []
    for ax in range(3):
        fwd = np.maximum.accumulate(surf, axis=ax)
        bwd = np.flip(np.maximum.accumulate(np.flip(surf, axis=ax), axis=ax), axis=ax)
        enc.append(fwd & bwd)
    filled = np.ones_like(surf)
    n_rel = np.zeros(surf.shape, np.int8)
    for ax in range(3):
        fp = footprints[ax]
        if fp is None:
            rel = np.ones(surf.shape, bool)
        else:
            o1, o2 = [k for k in range(3) if k != ax]
            rel = np.broadcast_to(np.expand_dims(~fp, ax), surf.shape)   # the (o1, o2) footprint lifted along ax
        filled &= (enc[ax] | ~rel)
        n_rel += rel.astype(np.int8)
    # a voxel with no reliable axis at all: fall back to the majority of the three enclosures
    maj = (enc[0].astype(np.int8) + enc[1].astype(np.int8) + enc[2].astype(np.int8)) >= 2
    filled = np.where(n_rel == 0, maj, filled)
    return filled | surf


def _fill_reliable(mesh: trimesh.Trimesh, pitch: float):
    """The fill used first by _fill_centers / _fill_grid: voxelise, the reliable-axis orthographic
    fill, strip streaks, fill sub-voxel pockets. Returns (VoxelGrid, matrix, n_streak, n_pocket) or
    (None, ...) when the fill added fewer than 30 % of the surface count (a shell)."""
    vg = mesh.voxelized(pitch=pitch)
    surf = vg.matrix.copy()
    fps = _hole_footprints(mesh, vg)
    M = _fill_ortho_reliable(surf, fps)
    if int(M.sum()) - int(surf.sum()) < 0.3 * int(surf.sum()):
        return None, None, 0, 0
    M, n_streak = _strip_streaks(M, surf)
    M, n_pocket, _ = _fill_pockets(M)
    return vg, M, n_streak, n_pocket
