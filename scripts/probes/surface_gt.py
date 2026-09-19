"""Reconstructed surface of the TARGET cloud vs the target's TRUE mesh (docs/experiments.md
2026-09-19, surface-smoothness acid test).

usage: surface_gt.py <run_json> <recon.ply> [<recon.ply> ...]

The run json's provenance gives src/tgt/n/seed; the target cloud is reproduced through
load_normalized (the same voxel fill + jitter, same seed) and the asset mesh is mapped into
the cloud frame by the same (centre, scale). For each reconstructed mesh (written by
render_photoreal.py --still -2 --save_mesh) it prints

  d_abs   mean |distance| from the recon vertices to the true surface (spacings; wu)
  d_sgn   mean signed distance (+ outside): the bias of the level
  d_95    95th percentile of |distance|
  compl   mean distance from 100k true-surface samples to the recon surface (spacings)
  n_dev   mean angle between the recon vertex normal and the true face normal at the
          closest point (deg); frac>30 = fraction of vertices deviating by more than 30 deg
  bump    mean |dihedral| (deg) as rendered, and at a COMMON resolution (vertex clustering
          at the render voxel), against the true mesh through the same clustering

The true mesh through the same clustering is the floor of every column: a reconstruction
cannot be smoother than the true surface at that resolution and be right.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from physmorph.sampling.mesh import load_mesh, sample_volume, filled_volume, load_normalized  # noqa: E402
from physmorph.sampling.orientation import orient_name, rotation  # noqa: E402

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


def clustered(m, vox):
    return m.simplify_vertex_clustering(voxel_size=vox, contraction=o3d.geometry.SimplificationContraction.Average)


def scene_of(m):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    return sc


def closest(sc, q):
    r = sc.compute_closest_points(o3d.core.Tensor(np.asarray(q, np.float32)))
    return r["points"].numpy(), r["primitive_ids"].numpy()


def main():
    run_json, plys = sys.argv[1], sys.argv[2:]
    prov = json.load(open(run_json))["provenance"]
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    src_p, tgt_p, n, seed = prov["src"], prov["tgt"], int(prov["n"]), int(prov["seed"])
    src_p = src_p if os.path.exists(src_p) else os.path.join(repo, src_p)
    tgt_p = tgt_p if os.path.exists(tgt_p) else os.path.join(repo, tgt_p)
    # the cloud frame: reproduce load_normalized(tgt, n, seed + 1, match_volume=v_src)
    _, v_src = load_normalized(src_p, n, seed, return_volume=True)
    mesh = load_mesh(tgt_p)
    o = orient_name(tgt_p)
    V = np.asarray(mesh.vertices, np.float64)
    if o != "id":
        V = V @ rotation(o).T
    mesh = trimesh.Trimesh(vertices=V, faces=np.asarray(mesh.faces), process=False)
    raw = sample_volume(mesh, n, seed=seed + 1).astype(np.float64)
    mu = raw.mean(0)
    s = 8.0 / (np.linalg.norm(raw.max(0) - raw.min(0)) + 1e-9)
    vol = filled_volume(mesh) * s ** 3
    k = (v_src / vol) ** (1.0 / 3.0) if vol > 0 else 1.0
    c = s * k
    cloud = (raw - mu) * c
    # check against the archive of the first ply's json
    meta0 = json.load(open(os.path.splitext(plys[0])[0] + ".json"))
    z = np.load(meta0["npz"])
    from physmorph.sampling.orientation import orient_archive
    _, tgt_arch, _, _ = orient_archive(z, meta0["npz"])
    resid = float(np.abs(np.asarray(tgt_arch, np.float64) - cloud).max())
    spacing = float(meta0["spacing"]); vox = float(meta0["vox"])
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
    gt_bump = dihedral_mean(gt_o3); gt_bump_c = dihedral_mean(clustered(gt_o3, vox))
    print(f"[surface_gt] true mesh: {len(gt.faces)} faces, bump {gt_bump:.2f} deg as is, {gt_bump_c:.2f} deg clustered at the voxel")
    print(f"{'candidate':<28} {'d_abs(sp)':>9} {'d_sgn(sp)':>9} {'d_95(sp)':>8} {'compl(sp)':>9} {'n_dev':>6} {'frac>30':>7} "
          f"{'bump':>6} {'bump_c':>6} {'tris':>8} {'comp':>4}")
    for ply in plys:
        meta = json.load(open(os.path.splitext(ply)[0] + ".json"))
        m = o3d.io.read_triangle_mesh(ply)
        if len(m.triangles) == 0:
            print(f"{os.path.basename(ply):<28} empty"); continue
        m.compute_vertex_normals()
        q = np.asarray(m.vertices); nq = np.asarray(m.vertex_normals)
        cp, pid = closest(gt_sc, q)
        dvec = q - cp
        d = np.linalg.norm(dvec, axis=1)
        sgn = np.sign((dvec * gt_fn[pid]).sum(1))
        cos = np.clip((nq * gt_fn[pid]).sum(1), -1, 1)
        ang = np.degrees(np.arccos(cos))
        rc_sc = scene_of(m)
        cp2, _ = closest(rc_sc, gt_samp)
        compl = np.linalg.norm(gt_samp - cp2, axis=1)
        name = f"{meta['kernel']}/{meta['surface']}/{meta['post']}"
        print(f"{name:<28} {d.mean() / spacing:9.3f} {(sgn * d).mean() / spacing:9.3f} {np.quantile(d, 0.95) / spacing:8.3f} "
              f"{compl.mean() / spacing:9.3f} {ang.mean():6.2f} {(ang > 30).mean():7.3f} "
              f"{dihedral_mean(m):6.2f} {dihedral_mean(clustered(m, vox)):6.2f} {len(m.triangles):8d} {meta['components']:4d}")


if __name__ == "__main__":
    main()
