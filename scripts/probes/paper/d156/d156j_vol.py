"""d156j_vol.py OUT_JSONL RUN_DIR KIND MESH [...] -- D156j, after its reading: where the carve removed volume, measured on the
solids themselves, free of the sampler (whose surface-voxel jitter puts 13-26 % of the mesh's own sample outside the mesh,
so the stage-1 over-carve, the mesh sample's share outside the solid, rises as a solid approaches the mesh). KIND i or g
(RUN_DIR/MESH_sfsi018 / sfsi18 or sfsgg018 / sfsgg18 _r96.obj, the d = 0 mesh and the solid). 400k uniform points of the
base's box inside the base, by open3d occupancy, split by the true mesh (fan-closed) and by their depth below the base:
- excess (inside the base, outside the mesh): the share the solid removed, shallow (< 1 pixel below the base) and deep;
- inside the mesh: the share the solid removed (the carve into the object);
- the mesh's surface: 100k points, the share more than a quarter pixel outside the solid. CPU."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import physmorph  # noqa: E402,F401
import numpy as np  # noqa: E402
import d157_lib as D7  # noqa: E402
import trimesh  # noqa: E402
from physmorph.sampling.mesh import _oriented, load_normalized  # noqa: E402

N, SEED, RES = 40000, 97, 96
out, run_dir, kind = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
names = {"i": ("sfsi0", "sfsi"), "g": ("sfsgg0", "sfsgg")}[kind]
for mesh in sys.argv[4:]:
    path = f"assets/{mesh}.obj"
    _, v_src = load_normalized("assets/isosphere.obj", N, SEED, return_volume=True, sample="stratified")
    frame = {}
    tgt = load_normalized(path, N, SEED + 1, match_volume=v_src, sample="stratified", frame=frame)
    box = float(np.abs(tgt).max()) * 1.25
    px = 2.0 * box / RES

    def surf(obj, close=False):
        m = _oriented(str(obj))
        V = (np.asarray(m.vertices, np.float64) - frame["offset"]) * frame["scale"]
        F = np.asarray(m.faces, np.int64)
        if close:
            t, _ = D7.close_holes(V, F)
            V, F = np.asarray(t.vertices), np.asarray(t.faces)
        return D7.Surf(V, F)

    true = surf(path, True)
    base = surf(run_dir / f"{mesh}_{names[0]}18_r{RES}.obj")
    sol = surf(run_dir / f"{mesh}_{names[1]}18_r{RES}.obj")
    rng = np.random.default_rng(0)
    lo, hi = base.V.min(0), base.V.max(0)
    P = lo + (hi - lo) * rng.random((400000, 3))
    P = P[base.inside(P)]
    in_mesh, in_sol = true.inside(P), sol.inside(P)
    depth = -base.signed(P) / px
    exc = ~in_mesh
    sp, _ = trimesh.sample.sample_surface(trimesh.Trimesh(true.V, true.F, process=False), 100000, seed=0)
    so = sol.signed(np.asarray(sp)) / px
    sb = base.signed(np.asarray(sp)) / px
    row = dict(mesh=mesh, kind=kind, points=int(len(P)), excess_share_of_base=float(exc.mean()),
               excess_removed=float((exc & ~in_sol).sum() / max(exc.sum(), 1)),
               excess_removed_shallow=float((exc & ~in_sol & (depth < 1)).sum() / max((exc & (depth < 1)).sum(), 1)),
               excess_removed_deep=float((exc & ~in_sol & (depth >= 1)).sum() / max((exc & (depth >= 1)).sum(), 1)),
               excess_deep_share=float((exc & (depth >= 1)).sum() / max(exc.sum(), 1)),
               mesh_cut=float((in_mesh & ~in_sol).sum() / max(in_mesh.sum(), 1)),
               surface_outside_solid_quarter_px=float((so > 0.25).mean()),
               surface_outside_base_quarter_px=float((sb > 0.25).mean()))
    print(json.dumps(row), flush=True)
    with open(out, "a") as fo:
        fo.write(json.dumps(row) + "\n")
