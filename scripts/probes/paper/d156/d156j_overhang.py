"""d156j_overhang.py OUT_JSONL MESH [...] -- D156j check 1, after its reading: the share of the mesh's own 40k sample
(seed 98, the run's target sample) outside the mesh itself (open3d occupancy on the fan-closed mesh in the run frame), and
outside the conservative and D156's hulls by the same occupancy. The stratified sampler keeps every surface voxel and
jitters within it, so part of its sample lies outside the mesh; that part is no test of a hull that hugs the mesh. CPU."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import physmorph  # noqa: E402,F401
import numpy as np  # noqa: E402
import d157_lib as D7  # noqa: E402
import open3d as o3d  # noqa: E402
from physmorph.sampling.mesh import _oriented, load_normalized  # noqa: E402

N, SEED = 40000, 97
H = Path("/data/relcfd/chayo/physmorph_v2/output/gpu/d156j/assets_hull")
H0 = Path("/data/relcfd/chayo/physmorph_v2/output/gpu/d156/assets_hull")
for mesh in sys.argv[2:]:
    path = f"assets/{mesh}.obj"
    _, v_src = load_normalized("assets/isosphere.obj", N, SEED, return_volume=True, sample="stratified")
    frame = {}
    tgt = load_normalized(path, N, SEED + 1, match_volume=v_src, sample="stratified", frame=frame)

    def occ_out(obj, close):
        m = _oriented(str(obj))
        V = (np.asarray(m.vertices, np.float64) - frame["offset"]) * frame["scale"]
        F = np.asarray(m.faces, np.int64)
        if close:
            t, _ = D7.close_holes(V, F)
            V, F = np.asarray(t.vertices), np.asarray(t.faces)
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
        return float((sc.compute_occupancy(o3d.core.Tensor(tgt), nthreads=D7.NT, nsamples=3).numpy() < 0.5).mean())

    row = dict(mesh=mesh, sample_outside_mesh=occ_out(path, True),
               sample_outside_conservative_hull=occ_out(H / f"{mesh}_hull18_r96.obj", False),
               sample_outside_d156_hull=occ_out(H0 / f"{mesh}_hull18_r96.obj", False))
    print(json.dumps(row), flush=True)
    with open(sys.argv[1], "a") as fo:
        fo.write(json.dumps(row) + "\n")
