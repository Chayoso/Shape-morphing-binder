"""d156f_extract2.py MESH -- D156f, a second extraction of the carved solid (an amendment after the pre-registered
reading: on the dragon the pre-registered extraction, the moved particles' exterior taken back by the mean offset c, scored
its own delta = 0 baseline at 23.6e-3 against the hull file's 14.7e-3, so it distorts thin parts and does not cancel).

The solid: the visual hull minus the layer the surface particles left, {x : depth below the hull's surface > delta of the
nearest surface particle (within one pitch of the hull's surface)}. At delta = 0 it is the hull itself (up to the same
half-pitch remesh, which is the baseline). Resampled uniformly by the run's sampler in mesh coordinates at the hull
sample's density; every score the mean over the eight seeds 98 + 100 k; the floor the mesh's own samples at 99 + 100 k
(as D156f). The same stage-1 numbers: the share of the way, the excess removed, the over-carve."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("D156_DEV", "cpu")
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import d156_sfs_f as F_  # noqa: E402
from d156_sfs_f import D7, o3d, trimesh, marching_cubes, cKDTree, nnf  # noqa: E402
from physmorph.pipeline.render_loss import d_exterior, exterior_targets  # noqa: E402
from physmorph.render.exterior import Lattice, make_field  # noqa: E402
from physmorph.sampling.mesh import _oriented, draws_in_frame, fill_volume_in_frame, load_normalized  # noqa: E402
from physmorph.sampling.orientation import orient_name  # noqa: E402

torch.set_grad_enabled(False)
cfg, AMB, views, NT = F_.cfg, F_.AMB, F_.views, F_.NT
mesh, N, RES = sys.argv[1], 40000, 96
A = Path("/data/relcfd/chayo/physmorph_v2/output/gpu/d156/assets_hull")
path, hpath = f"assets/{mesh}.obj", str(A / f"{mesh}_hull18_r96.obj")
z = np.load(A / f"{mesh}_sfsf18_r96_delta.npz")
X0, delta = z["x0"].astype(np.float64), z["delta"].astype(np.float64)
_, v_src = load_normalized("assets/isosphere.obj", N, 97, return_volume=True, sample="stratified")
frame = {}
tgt = load_normalized(path, N, 98, match_volume=v_src, sample="stratified", frame=frame)
box = float(np.abs(tgt).max()) * 1.25
pitch = 0.708 * D7.median_kth_spacing(tgt, 8, 20000)
T0 = torch.as_tensor(tgt)
center = T0.mean(0)
lattice = Lattice(center, 2.8 * float((T0 - center).norm(dim=1).max()))
h = min(box / RES, 0.92 * pitch)


def fresh(x):
    p, g, _, _ = lattice.discs(make_field("zb", torch.as_tensor(np.asarray(x, np.float32)), pitch), h, refine=False)
    return p, nnf.normalize(g, dim=1)


per = [exterior_targets(*fresh(s), views, RES, box, cfg.sil_k, AMB)
       for s in [tgt] + draws_in_frame(path, frame, N, [98 + 100 * k for k in range(1, 8)])]
sils = [torch.stack(v).mean(0) for v in zip(*[e[0] for e in per])]
shade = [torch.stack(v).mean(0) for v in zip(*[e[1] for e in per])]


def mean_loss(xs):
    v = []
    for x in xs:
        p, n = fresh(x)
        s, b = d_exterior(p, n, sils, shade, views, RES, box, cfg.sil_k, cfg.w_hole, cfg.w_spray, AMB)
        v.append(float(s + cfg.w_pbr * b))
    return float(np.mean(v))


m = _oriented(path)
Vw = (np.asarray(m.vertices, np.float64) - frame["offset"]) * frame["scale"]
tc, _ = D7.close_holes(Vw, np.asarray(m.faces, np.int64))
if tc.volume < 0:
    tc.invert()
hm = _oriented(hpath)
Hw = (np.asarray(hm.vertices, np.float64) - frame["offset"]) * frame["scale"]
hullt = trimesh.Trimesh(Hw, np.asarray(hm.faces), process=True)
if hullt.volume < 0:
    hullt.invert()
hull = D7.Surf(np.asarray(hullt.vertices), np.asarray(hullt.faces))
surf = np.abs(hull.signed(X0)) < pitch
kd = cKDTree(X0[surf])
hh = 0.5 * pitch
lo = Hw.min(0) - 3 * pitch
nc = np.ceil((Hw.max(0) + 3 * pitch - lo) / hh).astype(int) + 1
G = np.stack(np.meshgrid(*[lo[i] + hh * np.arange(nc[i]) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
depth = -hull.signed(G)
dnn = delta[surf][kd.query(G, workers=NT)[1]]
n_vh = int(round(N * fill_volume_in_frame(hpath, frame, N) / fill_volume_in_frame(path, frame, N)))
seeds = [98 + 100 * k for k in range(8)]
row = dict(mesh=mesh, surface_particles=int(surf.sum()))
for name, Fv in (("sfsg0", depth), ("sfsg", depth - dnn)):
    v, f, _, _ = marching_cubes(Fv.reshape(*nc), level=0.0, spacing=(hh, hh, hh), allow_degenerate=False)
    S = trimesh.Trimesh(v + lo, f, process=True)
    S.update_faces(S.nondegenerate_faces())
    S.merge_vertices()
    if not S.is_watertight:
        trimesh.repair.fill_holes(S)
    if S.volume < 0:
        S.invert()
    o = A / f"{mesh}_{name}18_r{RES}.obj"
    trimesh.Trimesh(S.vertices / frame["scale"] + frame["offset"], S.faces, process=False).export(str(o))
    assert orient_name(str(o)) == "id", o
    n_s = int(round(n_vh * fill_volume_in_frame(str(o), frame, N) / fill_volume_in_frame(hpath, frame, N)))
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(S.vertices.astype(np.float32)), o3d.core.Tensor(np.asarray(S.faces, np.uint32)))
    row[name] = dict(volume=float(S.volume), loss=mean_loss(draws_in_frame(str(o), frame, n_s, seeds)),
                     mesh_outside=float((sc.compute_occupancy(o3d.core.Tensor(tgt), nthreads=NT, nsamples=3).numpy()
                                         < 0.5).mean()), watertight=bool(S.is_watertight))
row["floor"] = mean_loss(draws_in_frame(path, frame, N, [99 + 100 * k for k in range(8)]))
row["vol_mesh"] = float(tc.volume)
b, s = row["sfsg0"], row["sfsg"]
row["way"] = (b["loss"] - s["loss"]) / (b["loss"] - row["floor"])
row["excess_removed"] = (b["volume"] - s["volume"]) / (b["volume"] - row["vol_mesh"])
row["passes"] = bool(row["way"] >= 0.5 and row["excess_removed"] >= 0.25 and s["mesh_outside"] <= b["mesh_outside"] + 0.01)
print(json.dumps(row), flush=True)
with open(A / "sfsg_r96.jsonl", "a") as fo:
    fo.write(json.dumps(row) + "\n")
