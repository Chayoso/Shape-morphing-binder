"""Reconstructed surfaces vs the target's TRUE mesh and a scale-defined roughness
(docs/experiments.md 2026-09-19, surface-smoothness acid test).

usage: surface_gt.py <run_json> <recon.ply> [<recon.ply> ...]

The run json's provenance gives src/tgt/n/seed; the target cloud is reproduced through
load_normalized (the same voxel fill + jitter, same seed) and the asset mesh is mapped into
the cloud frame by the same (centre, scale). Meshes come from render_photoreal.py --still
--save_mesh (a .json sidecar next to each .ply). TARGET meshes (frame -2) get the ground-
truth columns; morph-frame meshes get the roughness columns only.

  d_abs   mean |distance| from the recon vertices to the true surface (spacings)
  d_sgn   mean signed distance (+ outside): the bias of the level
  d_95    95th percentile of |distance|
  compl   mean distance from 100k true-surface samples to the recon surface (spacings)
  n_dev   mean unsigned angle between the recon vertex normal and the true face normal at
          the closest point (deg; the winding is ignored, `flip` = fraction facing inward)
  rough   roughness at the scale of two spacings: mean angle (deg) between a face normal and
          the area-weighted mean normal of the faces within 2 spacings of its centroid.
          Resolution-independent (a finer mesh of the same surface gives the same number);
          0 for a plane or a sphere much larger than 2 spacings, large for bumps of that size.
          The true mesh's own value is the floor: its features at that scale.
  bump    mean |dihedral| as rendered (the old H1 measure; depends on the triangle size)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from physmorph.sampling.mesh import load_mesh, sample_volume, filled_volume, load_normalized  # noqa: E402
from physmorph.sampling.orientation import orient_name, rotation, orient_archive  # noqa: E402

import open3d as o3d  # noqa: E402
import trimesh  # noqa: E402


def dihedral_mean(m):
    if m is None or len(m.triangles) == 0:
        return float("nan")
    m.compute_triangle_normals()
    f = np.asarray(m.triangles); nrm = np.asarray(m.triangle_normals)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0), axis=1)
    key = e[:, 0].astype(np.int64) * (f.max() + 1) + e[:, 1]
    tri = np.tile(np.arange(len(f)), 3)
    order = np.argsort(key, kind="stable"); key = key[order]; tri = tri[order]
    same = key[1:] == key[:-1]
    a_, b_ = tri[:-1][same], tri[1:][same]
    cosd = np.clip((nrm[a_] * nrm[b_]).sum(1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosd)).mean()) if len(cosd) else float("nan")


def roughness(m, r):
    """Mean angle (deg) between each face normal and the area-weighted mean normal of the faces
    whose centroids lie within r of its centroid (the face itself included)."""
    if m is None or len(m.triangles) == 0:
        return float("nan")
    v = np.asarray(m.vertices); f = np.asarray(m.triangles)
    e1 = v[f[:, 1]] - v[f[:, 0]]; e2 = v[f[:, 2]] - v[f[:, 0]]
    cr = np.cross(e1, e2); area = 0.5 * np.linalg.norm(cr, axis=1) + 1e-15
    n = cr / (2 * area[:, None])
    c = v[f].mean(1)
    kd = cKDTree(c)
    pairs = kd.query_pairs(r, output_type="ndarray")
    acc = n * area[:, None]
    np.add.at(acc, pairs[:, 0], (n * area[:, None])[pairs[:, 1]])
    np.add.at(acc, pairs[:, 1], (n * area[:, None])[pairs[:, 0]])
    mean_n = acc / (np.linalg.norm(acc, axis=1, keepdims=True) + 1e-15)
    cos = np.clip((n * mean_n).sum(1), -1, 1)
    return float(np.degrees(np.arccos(cos)).mean())


def detail_analysis(q, nq, cp, pid, gt_fn, spacing, r_sp=2.0, n_sub=50000, seed=0):
    """Structure or noise? Two band-limited measures at the 2-spacing scale on a random subset
    of the recon vertices (the neighbourhoods are taken over ALL vertices):
      hp_res  RMS of the high-passed signed distance to the true surface (s_i minus its
              2-spacing neighbourhood mean), in spacings — bumps that FOLLOW the true surface
              leave it unchanged, bumps that do not raise it
      dcorr   correlation of the high-passed recon normal field with the high-passed TRUE
              normal field sampled at the closest points — detail that is the target's is
              positively correlated, noise is not."""
    rng = np.random.default_rng(seed)
    n = len(q)
    sub = rng.choice(n, min(n_sub, n), replace=False)
    kd = cKDTree(q)
    nb = kd.query_ball_point(q[sub], r_sp * spacing, workers=-1)
    dvec = q - cp
    s = np.sign((dvec * gt_fn[pid]).sum(1)) * np.linalg.norm(dvec, axis=1)
    N = gt_fn[pid]
    hp_s = np.empty(len(sub)); hp_n = np.empty((len(sub), 3)); hp_N = np.empty((len(sub), 3))
    for k, (i, idx) in enumerate(zip(sub, nb)):
        idx = np.asarray(idx)
        hp_s[k] = s[i] - s[idx].mean()
        hp_n[k] = nq[i] - nq[idx].mean(0)
        hp_N[k] = N[i] - N[idx].mean(0)
    hp_res = float(np.sqrt((hp_s ** 2).mean()) / spacing)
    dcorr = float((hp_n * hp_N).sum() / (np.sqrt((hp_n ** 2).sum() * (hp_N ** 2).sum()) + 1e-30))
    return hp_res, dcorr


def scene_of(m):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    return sc


def closest(sc, q):
    r = sc.compute_closest_points(o3d.core.Tensor(np.asarray(q, np.float32)))
    return r["points"].numpy(), r["primitive_ids"].numpy()


def main():
    gt_all = "--gt_all" in sys.argv          # GT columns for every mesh (an END frame vs the true target
    args = [a for a in sys.argv[1:] if a != "--gt_all"]   # surface: morph error + roughness, not a floor)
    run_json, plys = args[0], args[1:]
    prov = json.load(open(run_json))["provenance"]
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    src_p, tgt_p, n, seed = prov["src"], prov["tgt"], int(prov["n"]), int(prov["seed"])
    src_p = src_p if os.path.exists(src_p) else os.path.join(repo, src_p)
    tgt_p = tgt_p if os.path.exists(tgt_p) else os.path.join(repo, tgt_p)
    metas = {p: json.load(open(os.path.splitext(p)[0] + ".json")) for p in plys}
    meta0 = metas[plys[0]]
    spacing = float(meta0["spacing"]); vox = float(meta0["vox"])
    r_rough = 2.0 * spacing
    # the cloud frame: reproduce load_normalized(tgt, n, seed + 1, match_volume=v_src)
    _, v_src = load_normalized(src_p, n, seed, return_volume=True,
                               sample=prov.get("sampler", "replacement"))   # the SOURCE's sampler sets v_src
    mesh = load_mesh(tgt_p)
    o = orient_name(tgt_p)
    V = np.asarray(mesh.vertices, np.float64)
    if o != "id":
        V = V @ rotation(o).T
    mesh = trimesh.Trimesh(vertices=V, faces=np.asarray(mesh.faces), process=False)
    if prov.get("sampler", "replacement") == "stratified":
        from physmorph.sampling.mesh import sample_volume_stratified
        raw = sample_volume_stratified(mesh, n, seed=seed + 1).astype(np.float64)
    else:
        raw = sample_volume(mesh, n, seed=seed + 1).astype(np.float64)
    mu = raw.mean(0)
    s = 8.0 / (np.linalg.norm(raw.max(0) - raw.min(0)) + 1e-9)
    vol = filled_volume(mesh) * s ** 3
    k = (v_src / vol) ** (1.0 / 3.0) if vol > 0 else 1.0
    c = s * k
    cloud = (raw - mu) * c
    z = np.load(meta0["npz"])
    _, tgt_arch, _, _ = orient_archive(z, meta0["npz"])
    resid = float(np.abs(np.asarray(tgt_arch, np.float64) - cloud).max())
    print(f"[surface_gt] {os.path.basename(tgt_p)}: cloud reproduced, max residual vs archive {resid:.2e} wu "
          f"({resid / spacing:.1e} spacings); spacing {spacing:.4f} wu, render vox {vox:.4f} wu ({vox / spacing:.2f} sp)")
    if resid > 1e-3 * spacing:
        print("[surface_gt] the archive's target is not this cloud (a different seed/normalisation?) — abort")
        return
    gt_v = (V - mu) * c
    gt = trimesh.Trimesh(vertices=gt_v, faces=np.asarray(mesh.faces), process=False)
    gt_o3 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(gt_v), o3d.utility.Vector3iVector(np.asarray(gt.faces, np.int32)))
    gt_sc = scene_of(gt_o3)
    gt_fn = np.asarray(gt.face_normals)
    gt_samp, _ = trimesh.sample.sample_surface(gt, 100000, seed=0)
    print(f"[surface_gt] true mesh: {len(gt.faces)} faces, rough(2sp) {roughness(gt_o3, r_rough):.2f} deg, "
          f"bump {dihedral_mean(gt_o3):.2f} deg")
    hdr = (f"{'candidate':<30} {'d_abs':>6} {'d_sgn':>6} {'d_95':>6} {'compl':>6} {'n_dev':>6} {'flip':>5} "
           f"{'rough':>6} {'bump':>6} {'tris':>7} {'comp':>4}")
    print("TARGET (units: spacings, degrees)"); print(hdr)
    rows_frames = []
    for ply in plys:
        meta = metas[ply]
        name = f"{meta['kernel']}/{meta['surface']}/{meta['post']}" + (f" pull{meta['pull']}" if meta.get("pull", 1) != 1 else "") + (f" {meta['layer']}" if meta.get("layer", "grad") != "grad" else "") + (f" cell{meta['poisson_cell']:.2f}" if meta.get("poisson_cell", 1.0) != 1.0 else "")
        m = o3d.io.read_triangle_mesh(ply)
        base = f"{name:<30}"
        if len(m.triangles) == 0:
            row = base + "  EMPTY (every component removed)"
            (print(row) if int(meta["frame"]) == -2 else rows_frames.append(row)); continue
        m.compute_vertex_normals()
        rough = roughness(m, r_rough); bump = dihedral_mean(m)
        if int(meta["frame"]) != -2 and not gt_all:
            rows_frames.append(f"{base} {'':>6} {'':>6} {'':>6} {'':>6} {'':>6} {'':>5} {rough:6.2f} {bump:6.2f} "
                               f"{len(m.triangles):7d} {meta['components']:4d}  frame {meta['frame']}")
            continue
        if int(meta["frame"]) != -2:
            base = f"{(name + ' f' + str(meta['frame'])):<30}"
        q = np.asarray(m.vertices); nq = np.asarray(m.vertex_normals)
        cp, pid = closest(gt_sc, q)
        dvec = q - cp
        d = np.linalg.norm(dvec, axis=1)
        sgn = np.sign((dvec * gt_fn[pid]).sum(1))
        cos = np.clip((nq * gt_fn[pid]).sum(1), -1, 1)
        ang = np.degrees(np.arccos(np.abs(cos)))
        cp2, _ = closest(scene_of(m), gt_samp)
        compl = np.linalg.norm(gt_samp - cp2, axis=1)
        hp_res, dcorr = detail_analysis(q, nq, cp, pid, gt_fn, spacing)
        print(f"{base} {d.mean() / spacing:6.2f} {(sgn * d).mean() / spacing:6.2f} {np.quantile(d, 0.95) / spacing:6.2f} "
              f"{compl.mean() / spacing:6.2f} {ang.mean():6.2f} {(cos < 0).mean():5.2f} {rough:6.2f} {bump:6.2f} "
              f"{len(m.triangles):7d} {meta['components']:4d}   hp_res {hp_res:.3f} sp  dcorr {dcorr:+.3f}")
    if rows_frames:
        print("MORPH FRAME"); print(hdr)
        for r in rows_frames:
            print(r)


if __name__ == "__main__":
    main()
