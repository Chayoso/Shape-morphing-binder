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
    return stratified_draws(mesh, n, [seed])[0]


PITCH_PER_SP8 = 0.708      # the pipeline's constant: a jittered lattice's pitch = 0.708 x its median 8th-neighbour distance


def _stratified_fill(mesh: trimesh.Trimesh, n: int):
    """(resolution, voxel centres, pitch) of the stratified sampler's fill: the coarsest fill (bisection on the
    resolution of the mesh's largest extent) that holds at least n voxels."""
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
    return hi, centers, ext / hi


def stratified_fill_volume(mesh: trimesh.Trimesh, n: int) -> float:
    """The volume the stratified sample of n particles represents, in mesh units: its fill's voxels times the pitch
    cubed. The sample has one jittered particle in each of n of these voxels (the surplus dropped uniformly), so its
    number density is n / this volume everywhere inside (D132)."""
    _, centers, pitch = _stratified_fill(mesh, n)
    return float(len(centers)) * pitch ** 3


def stratified_draws(mesh: trimesh.Trimesh, n: int, seeds, surface_density: float = 1.0, band_sp: float = 0.0,
                     rest: dict | None = None) -> list:
    """sample_volume_stratified once per seed on one fill (the fill is found once): independent draws of the
    same sampler.

    surface_density F > 1 (D122, `--surface_density`): the same sampler with its density F times higher within the
    outer band than in the interior, at the same n. The band is the pipeline's own (band_sp spacings deep, the
    relaxation's neighbour width layer_h_sp; a spacing = the base lattice's pitch / 0.708, the pipeline's 8th-neighbour
    constant), measured from the surface of the FILL (the distance of a voxel's centre to the fill's boundary, an
    Euclidean distance transform of the filled voxels), with the base pitch that F = 1 gives at this n. The fill is
    refined (the same bisection) until its band voxels plus 1 / F of its interior voxels hold n: the band keeps every
    voxel it can (n_b = n F V_b / (F V_b + V_i), one jittered particle per voxel, the surplus dropped uniformly), the
    interior the rest, so the two densities are in the ratio F exactly; the order is shuffled. F = 1 is the code above,
    untouched: the same calls on the same generator, the same sample bit for bit. `rest` (F > 1): receives `w`, each
    particle's rest volume relative to the mean (band V_b / n_b, interior V_i / n_i, divided by V / n; mean 1), `base`
    (the F = 1 draw of the first seed, whose frame the caller keeps so that the sample stands where the base sample
    stood), and `report` (the band's depth in mesh units, its voxel and particle shares, the two pitches over the base
    pitch)."""
    hi, centers, pitch = _stratified_fill(mesh, n)
    print(f"[sampling] stratified: fill {hi}^3 = {len(centers)} voxels for n = {n}, pitch {pitch:.4g}", flush=True)
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(centers), n, replace=False) if len(centers) > n else np.arange(n)
        jitter = (rng.uniform(-0.5, 0.5, (n, 3)) * pitch).astype(np.float32)
        out.append((centers[keep] + jitter).astype(np.float32))
    if not surface_density > 1.0:
        return out
    return _surface_dense_draws(mesh, n, seeds, float(surface_density), float(band_sp), hi, pitch, out[0], rest)


def _band_centers(mesh: trimesh.Trimesh, pitch: float, depth: float):
    """(band centres, interior centres) of the fill at `pitch`: a voxel is in the band when its centre lies within
    `depth` (mesh units) of the fill's boundary (the Euclidean distance transform of the filled voxels, the centre of a
    voxel on the boundary half a pitch in)."""
    from scipy.ndimage import distance_transform_edt
    vg, M = _fill_grid(mesh, pitch)
    if vg is None or M is None or int(M.sum()) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    P = np.pad(M, 1)
    d = (distance_transform_edt(P)[1:-1, 1:-1, 1:-1] - 0.5) * pitch
    band = M & (d <= depth)
    cb = vg.indices_to_points(np.argwhere(band)).astype(np.float32)
    ci = vg.indices_to_points(np.argwhere(M & ~band)).astype(np.float32)
    return cb, ci


def _surface_dense_draws(mesh, n, seeds, F, band_sp, hi_base, pitch_base, base, rest):
    if not band_sp > 0:
        raise ValueError("surface_density > 1 needs a band (band_sp spacings) to be denser in")
    depth = band_sp * pitch_base / PITCH_PER_SP8
    ext = float(mesh.extents.max())
    enough = lambda res: (lambda cb, ci: len(cb) + len(ci) / F >= n)(*_band_centers(mesh, ext / res, depth))  # noqa: E731
    # the base fill refined by F^(1/3) holds n even with the whole body counted as interior, so the search ends there
    lo, hi = hi_base, min(400, int(np.ceil(hi_base * F ** (1.0 / 3.0))) + 1)
    if not enough(hi):
        raise ValueError(f"surface_density {F:g}: no fill up to {hi}^3 holds n = {n} with the band {F:g} x denser")
    if enough(lo):
        hi = lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if enough(mid):
            hi = mid
        else:
            lo = mid
    pitch = ext / hi
    cb, ci = _band_centers(mesh, pitch, depth)
    n_bv, n_iv = len(cb), len(ci)
    n_b = min(int(round(n * F * n_bv / (F * n_bv + n_iv))), n_bv)
    n_i = n - n_b
    if n_i > n_iv or n_i < 0:
        raise ValueError(f"surface_density {F:g}: the fill at {hi}^3 cannot hold {n_b} band + {n_i} interior particles")
    V = (n_bv + n_iv) * pitch ** 3
    w_b = (n_bv * pitch ** 3 / max(n_b, 1)) / (V / n)
    w_i = (n_iv * pitch ** 3 / max(n_i, 1)) / (V / n)
    print(f"[sampling] surface density {F:g}: band {depth:.4g} deep ({band_sp:g} spacings of the base pitch {pitch_base:.4g}); "
          f"fill {hi}^3 (pitch {pitch:.4g} = {pitch / pitch_base:.3f} base): band {n_bv} voxels ({n_bv / (n_bv + n_iv) * 100:.1f} % "
          f"of the volume) -> {n_b} particles ({n_b / n * 100:.1f} %), interior {n_iv} -> {n_i}; pitches band "
          f"{w_b ** (1 / 3):.3f}, interior {w_i ** (1 / 3):.3f} of the base", flush=True)
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        kb = rng.choice(n_bv, n_b, replace=False) if n_bv > n_b else np.arange(n_b)
        ki = rng.choice(n_iv, n_i, replace=False) if n_iv > n_i else np.arange(n_i)
        x = np.concatenate([cb[kb], ci[ki]]) + (rng.uniform(-0.5, 0.5, (n, 3)) * pitch).astype(np.float32)
        w = np.concatenate([np.full(n_b, w_b, np.float32), np.full(n_i, w_i, np.float32)])
        order = rng.permutation(n)                          # no strided subsample reads the band alone
        out.append(x[order].astype(np.float32))
        if rest is not None and seed == seeds[0]:
            rest["w"] = w[order].astype(np.float32)
    if rest is not None:
        rest["base"] = base
        rest["report"] = dict(surface_density=F, band_sp=band_sp, band_depth_mesh=depth, pitch_base_mesh=pitch_base,
                              fill_res=hi, fill_res_base=hi_base, pitch_over_base=pitch / pitch_base,
                              band_volume_share=n_bv / (n_bv + n_iv), band_particle_share=n_b / n,
                              band_pitch_over_base=w_b ** (1 / 3), interior_pitch_over_base=w_i ** (1 / 3))
    return out


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
        vgr, Mr, _, _ = _fill_reliable(mesh, pitch) if FILL_MODE == "reliable" else (None, None, 0, 0)
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
                if FILL_MODE == "reliable":
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


def _fill_volume_cached(mesh: trimesh.Trimesh, path: str, n: int, orient: str) -> float:
    """stratified_fill_volume of the (oriented) mesh, in mesh units, cached beside the samples (it does not depend on
    the seed; the same key fields as the sample cache)."""
    import hashlib
    import os
    cache_on = os.environ.get("PHYSMORPH_SAMPLE_CACHE", "1") != "0"
    key = hashlib.sha1((f"{os.path.abspath(path)}|{os.path.getmtime(path)}|{os.path.getsize(path)}|{n}|"
                        f"{FILL_MODE}|{orient}|fill-v1").encode()).hexdigest()[:16]
    cdir = os.environ.get("PHYSMORPH_CACHE",
                          os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                       "output", "cache"))
    cpath = os.path.join(cdir, f"fill_{os.path.splitext(os.path.basename(path))[0]}_{n}_{key}.npz")
    if cache_on and os.path.exists(cpath):
        try:
            return float(np.load(cpath)["vol_fill"])
        except Exception:
            pass
    v = stratified_fill_volume(mesh, n)
    if cache_on:
        try:
            os.makedirs(cdir, exist_ok=True)
            np.savez(cpath, vol_fill=v)
        except Exception:
            pass
    return v


def load_normalized(path: str, n: int, seed: int = 1, size: float = 8.0,
                    match_volume: float | None = None,
                    return_volume: bool = False,
                    shell: tuple[float, float] | None = None,
                    sample: str = "replacement", frame: dict | None = None,
                    surface_density: float = 1.0, band_sp: float = 0.0, rest: dict | None = None,
                    fill: dict | None = None):
    """Sample n particles from a mesh, centred at the origin and scaled so the bbox
    diagonal is `size` — the normalisation every runner script used to duplicate.

    surface_density F > 1 (D122, stratified only): the sample F times denser in the outer band of band_sp spacings
    (stratified_draws); it is centred and scaled by the BASE sample's (F = 1, same seed) mean and bounding box, so it
    stands in the frame the F = 1 sample has; `rest` receives `w` (each particle's rest volume relative to the mean)
    and `report` (the band's numbers; `band_depth` and `pitch_base` in world units once the frame is known). F = 1:
    nothing changes.

    match_volume: if given, the cloud is RESCALED (about the origin) so its filled
    volume equals this value (the source's): isochoric MPM particles cannot change
    the body's total volume, so a target of a different volume is unreachable.
    return_volume: also return the cloud's filled volume in world units^3.
    shell: (ratio, thickness_wu) — shell-biased sampling (sample_volume_shell) with the
    shell thickness given in WORLD units; the return then carries the per-particle
    relative rest volumes as a third value (x, vol, w) / (x, w).
    fill (D132; the uniform stratified sampler only): receives `volume`, the volume the sample represents in world
    units (stratified_fill_volume, scaled as the cloud): its number density is n / this volume. The filled volume
    above (and match_volume) is the mesh's at a 110^3 fill; the sampler's own fill is coarser (one voxel per particle),
    and the two differ by a surface term that depends on the shape."""
    if fill is not None and (shell is not None or sample != "stratified" or surface_density > 1.0):
        raise ValueError("fill: the volume of the uniform stratified sampler's fill only")
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
        # 2026-09-23 (speed): the stratified sample and the filled volume are deterministic in
        # (mesh file, n, seed, sampler, fill mode, orientation) and cost 90 s at 300k (trimesh
        # voxelise / subdivide); cached under output/cache (PHYSMORPH_CACHE overrides the
        # directory, PHYSMORPH_SAMPLE_CACHE=0 disables). The fill reports are restored on a hit.
        import hashlib
        import os
        cache_on = os.environ.get("PHYSMORPH_SAMPLE_CACHE", "1") != "0"
        dense = sample == "stratified" and surface_density > 1.0
        key = hashlib.sha1((f"{os.path.abspath(path)}|{os.path.getmtime(path)}|{os.path.getsize(path)}|{n}|{seed}|"
                            f"{sample}|{FILL_MODE}|{_o}|v1"
                            + (f"|sd{surface_density:g}|band{band_sp:g}" if dense else "")).encode()).hexdigest()[:16]
        cdir = os.environ.get("PHYSMORPH_CACHE",
                              os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                           "output", "cache"))
        cpath = os.path.join(cdir, f"sample_{os.path.splitext(os.path.basename(path))[0]}_{n}_{key}.npz")
        hit = None
        if cache_on and os.path.exists(cpath):
            try:
                z = np.load(cpath, allow_pickle=True)
                hit = (np.asarray(z["x"], np.float32), float(z["vol_mesh"]),
                       {"stripped": int(z["streak_stripped"]), "method": (str(z["streak_method"]) if str(z["streak_method"]) != "None" else None)},
                       {"filled": int(z["pocket_filled"]), "iters": int(z["pocket_iters"])})
                if dense:
                    hit += ({"w": np.asarray(z["w"], np.float32), "base": np.asarray(z["base"], np.float32),
                             "report": z["report"].item()},)
            except Exception:
                hit = None
        dense_rest = {} if dense else None
        if hit is not None:
            x, vol_mesh, sr, pr = hit[:4]
            STREAK_REPORT.update(sr); POCKET_REPORT.update(pr)
            if dense:
                dense_rest = hit[4]
        else:
            x = (stratified_draws(mesh, n, [seed], surface_density, band_sp, dense_rest)[0] if sample == "stratified"
                 else sample_volume(mesh, n, seed=seed)).astype(np.float32)
            vol_mesh = float(filled_volume(mesh))
            if cache_on:
                try:
                    os.makedirs(cdir, exist_ok=True)
                    np.savez(cpath, x=x, vol_mesh=vol_mesh, streak_stripped=STREAK_REPORT.get("stripped", 0),
                             streak_method=str(STREAK_REPORT.get("method")), pocket_filled=POCKET_REPORT.get("filled", 0),
                             pocket_iters=POCKET_REPORT.get("iters", 0),
                             **({"w": dense_rest["w"], "base": dense_rest["base"],
                                 "report": np.array(dense_rest["report"], dtype=object)} if dense else {}))
                except Exception:
                    pass
        w = None
    if shell is None and sample == "stratified" and surface_density > 1.0:
        # the frame of the base sample (F = 1, the same seed): its mean and bounding box, not this sample's, whose
        # centroid sits nearer the surface and whose extremes lie on a finer lattice
        base = dense_rest["base"]
        offset = base.mean(0).astype(np.float64)
        x = x - base.mean(0)
        s = size / (np.linalg.norm(base.max(0) - base.min(0)) + 1e-9)
        if rest is not None:
            rest["w"] = np.asarray(dense_rest["w"], np.float32)
            rest["report"] = dict(dense_rest["report"])
    else:
        offset = x.mean(0).astype(np.float64)
        x -= x.mean(0)
        s = size / (np.linalg.norm(x.max(0) - x.min(0)) + 1e-9)
    x = (x * s).astype(np.float32)
    vol = (vol_mesh if w is None and shell is None else filled_volume(mesh)) * float(s) ** 3
    k = 1.0
    if match_volume is not None and vol > 0:
        k = float((match_volume / vol) ** (1.0 / 3.0))
        x = (x * k).astype(np.float32)
        vol = vol * k ** 3
    if frame is not None:                    # where the cloud was put: world = (mesh - offset) * scale
        frame.update(offset=offset, scale=float(s) * k)
    if fill is not None:
        fill["volume"] = _fill_volume_cached(mesh, path, n, _o) * (float(s) * k) ** 3
    if rest is not None and rest.get("report") is not None:
        rest["report"]["band_depth"] = rest["report"]["band_depth_mesh"] * float(s) * k     # world units
        rest["report"]["pitch_base"] = rest["report"]["pitch_base_mesh"] * float(s) * k
        # the base sample itself in the same frame: what the F = 1 run has as its sample (the probes read their
        # pitch and the mesh's fit from it, so a surface-dense run is measured as the uniform one is)
        base = dense_rest["base"]
        rest["base"] = (((base - base.mean(0)) * s) * k).astype(np.float32)
    if shell is not None:
        return (x, vol, w) if return_volume else (x, w)
    return (x, vol) if return_volume else x


def _oriented(path: str) -> trimesh.Trimesh:
    mesh = load_mesh(path)
    from .orientation import orient_name, rotation
    o = orient_name(path)
    if o != "id":
        mesh.vertices = np.asarray(mesh.vertices, np.float64) @ rotation(o).T
    return mesh


def draws_in_frame(path: str, frame: dict, n: int, seeds, surface_density: float = 1.0, band_sp: float = 0.0) -> list:
    """Independent stratified draws of the mesh's volume (one per seed), each in the frame load_normalized put the
    mesh's cloud in (`frame`, as it filled it): further samples of the same target, for an expectation over them.
    surface_density, band_sp: the target's own sampler (stratified_draws), so the draws are samples of the same kind."""
    return [((x.astype(np.float64) - frame["offset"]) * frame["scale"]).astype(np.float32)
            for x in stratified_draws(_oriented(path), n, seeds, surface_density, band_sp)]


def surface_in_frame(path: str, frame: dict, n: int, seed: int = 0):
    """(points (n,3), normals (n,3)): n points of the mesh's surface and their faces' normals, in the frame
    load_normalized put the mesh's cloud in (`frame`, as it filled it)."""
    mesh = _oriented(path)
    pts, face = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return (((np.asarray(pts, np.float64) - frame["offset"]) * frame["scale"]).astype(np.float32),
            np.asarray(mesh.face_normals[face], np.float32))


STREAK_REPORT = {"stripped": 0, "method": None}   # last fill's streak count (tests, logs)
POCKET_REPORT = {"filled": 0, "iters": 0}         # last fill: sub-voxel pockets filled by _fill_pockets
FILL_MODE = "reliable"     # "reliable" (2026-09-22: hole-aware axes + pocket fill) | "legacy" (plain orthographic + streak
                           #   strip, the fill of every archive before 2026-09-22; surface_gt retries with it)


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
        vgr, Mr, n_streak, n_pocket = _fill_reliable(mesh, pitch) if FILL_MODE == "reliable" else (None, None, 0, 0)
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
                # the pocket fill belongs to the 2026-09-22 fill only: "legacy" must reproduce the old archives
                M, n_pocket, n_it = _fill_pockets(M) if FILL_MODE == "reliable" else (M, 0, 0)
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
    # CONSERVATIVE: the reliable-axis rule needs at least two reliable axes (one axis alone draws
    # 1-voxel streaks between unrelated surfaces, as the 'base' fill did — beast: 121 streaks,
    # 6 % low-density interior); with fewer, the plain three-axis intersection stands
    plain = enc[0] & enc[1] & enc[2]
    filled = np.where(n_rel >= 2, filled, plain)
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
