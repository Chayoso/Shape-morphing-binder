"""Acceptance check of the volume fill (docs/final_plan.md §6 step 1): for each asset, sample the
stratified 40k cloud as the pipeline does and measure the share of the DEEP interior (probe points
farther than 2 spacings inside the true surface) whose 1.5-spacing count is below 60 % of the bulk —
the interior pockets a non-watertight fill leaves (docs/surface_gradient.md §14: bunny 4.5 % before
the pocket fill). Also reports the fill's own pocket / streak counts. Pass: share < 0.5 %.

Usage: fill_check.py <asset> [<asset> ...]   (e.g. fill_check.py bunny dragon beast armadilo bob)"""
import sys

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
from physmorph.sampling import mesh as sm  # noqa: E402
from physmorph.sampling.orientation import orient_name, rotation  # noqa: E402

N = 40000
for asset in sys.argv[1:]:
    path = f"assets/{asset}.obj"
    m = sm.load_mesh(path)
    o = orient_name(path)
    V = np.asarray(m.vertices, np.float64)
    if o != "id":
        V = V @ rotation(o).T
    mesh = trimesh.Trimesh(vertices=V, faces=np.asarray(m.faces), process=False)
    pts = sm.sample_volume_stratified(mesh, N, seed=2).astype(np.float64)
    pocket, streak = sm.POCKET_REPORT["filled"], sm.STREAK_REPORT["stripped"]
    sub = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 20000), replace=False)]
    sp = float(np.median(cKDTree(sub).query(sub, k=9, workers=-1)[0][:, -1])) * (min(len(pts), 20000) / len(pts)) ** (1 / 3)
    m3 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(np.asarray(mesh.faces, np.int32)))
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m3))
    lo, hi = pts.min(0) - sp, pts.max(0) + sp
    step = 1.1 * sp
    g = np.stack(np.meshgrid(*[np.arange(l, h, step) for l, h in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
    dsurf = sc.compute_distance(o3d.core.Tensor(g.astype(np.float32))).numpy() / sp
    cnt = np.array([len(l) for l in cKDTree(pts).query_ball_point(g, 1.5 * sp)])
    deep = dsurf > 2.0
    bulk = float(np.median(cnt[deep & (cnt > 0)])) if (deep & (cnt > 0)).any() else 0.0
    inside = deep & (cnt > 0.15 * bulk)
    low = inside & (cnt < 0.6 * bulk)
    share = 100.0 * low.sum() / max(inside.sum(), 1)
    where = ""
    if low.any():
        L = g[low]
        be = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
        bv = V[np.unique(mesh.edges_sorted[be])] if len(be) else np.zeros((0, 3))
        dloop = cKDTree(bv).query(L, k=1, workers=-1)[0] / sp if len(bv) else np.full(len(L), np.inf)
        where = (f" | low bbox {np.round(L.min(0), 2)}..{np.round(L.max(0), 2)}, median distance to a boundary loop "
                 f"{np.median(dloop):.1f} sp, share within 3 sp of a loop {100 * (dloop < 3).mean():.0f} %")
    print(f"[fill_check] {asset:10s} watertight={mesh.is_watertight!s:5s} pockets filled {pocket:4d} streaks {streak:3d} | "
          f"deep interior probes {inside.sum():6d} bulk {bulk:.0f} low-density share {share:.2f} % -> {'PASS' if share < 0.5 else 'FAIL'}{where}",
          flush=True)
