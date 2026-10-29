"""d156_port_check.py MESH -- D156c: the torch port (d156_rcarve_gpu.py, run on the CPU) against the numpy functions of
d156_photohull.py on the visual hull's surface voxels of MESH: surface, visibility, observation, the statistic T.
Prints the largest differences (CPU only, a check of the port, no carve)."""
import os
import sys
from pathlib import Path
os.environ["D156_DEV"] = "cpu"
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import d156_rcarve_gpu as G  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
PH = G.PH
from physmorph.render.exterior import Lattice, ZhuBridson  # noqa: E402
from physmorph.pipeline.render_loss import exterior_targets  # noqa: E402
from physmorph.sampling.mesh import _oriented, draws_in_frame, load_normalized  # noqa: E402
mesh, N, RES = sys.argv[1], 40000, 96
path = f"assets/{mesh}.obj"
_, v_src = load_normalized("assets/isosphere.obj", N, 97, return_volume=True, sample="stratified")
frame = {}
tgt = load_normalized(path, N, 98, match_volume=v_src, sample="stratified", frame=frame)
box = float(np.abs(tgt).max()) * 1.25
px = 2.0 * box / RES
m = _oriented(path)
Vw = (np.asarray(m.vertices, np.float64) - frame["offset"]) * frame["scale"]
scene = G.o3d.t.geometry.RaycastingScene()
scene.add_triangles(G.o3d.core.Tensor(Vw.astype(np.float32)), G.o3d.core.Tensor(np.asarray(m.faces, np.uint32)))
sds = [PH.inside_sd(PH.silhouette(scene, box, RES, th, ph)) for th, ph in PH.views]
g = -box - px + px * np.arange(RES + 3)
Gp = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
f = np.full(len(Gp), np.inf)
for (th, ph), sd in zip(PH.views, sds):
    r, u, _ = PH.basis(th, ph)
    f = np.minimum(f, G.map_coordinates(sd, [(Gp @ r + box) / px, (Gp @ u + box) / px], order=1, mode="constant", cval=-1e9))
f = f.reshape(len(g), len(g), len(g))
f[[0, -1], :, :] = f[:, [0, -1], :] = f[:, :, [0, -1]] = -1.0
occ0 = f > 0
idx_np = PH.surface(occ0)
idx_t = G.surface_t(torch.as_tensor(occ0))
print("surface voxels", len(idx_np), len(idx_t), "same set", set(map(tuple, idx_np)) == set(map(tuple, idx_t.numpy())))
order = np.lexsort(idx_np.T[::-1]); idx_np = idx_np[order]
idx_t = idx_t[np.lexsort(idx_t.numpy().T[::-1])]
g0n = np.array([g[0]] * 3)
vis_np = PH.visibility(occ0, idx_np, g0n, px)
vis_t = G.visibility_t(torch.as_tensor(occ0), idx_t, torch.full((3,), float(g[0]), dtype=torch.float64), px).numpy()
print("visibility differs", int((vis_np != vis_t).sum()), "of", vis_np.size)
samples = [tgt] + draws_in_frame(path, frame, N, [98 + 100 * k for k in range(1, 8)])
pitch = 0.708 * G.D7.median_kth_spacing(tgt, 8, 20000)
T0 = torch.as_tensor(tgt); center = T0.mean(0)
lat = Lattice(center, 2.8 * float((T0 - center).norm(dim=1).max())); h = min(box / RES, 0.92 * pitch)
per = []
for s in samples:
    p, gg, _, _ = lat.discs(ZhuBridson(torch.as_tensor(s), pitch), h, refine=False)
    per.append(exterior_targets(p, torch.nn.functional.normalize(gg, dim=1), PH.views, RES, box, PH.cfg.sil_k, PH.AMB))
Al = np.stack([np.stack([x.numpy() for x in e[0]]) for e in per]).astype(np.float64)
Sl = np.stack([np.stack([x.numpy() for x in e[1]]) for e in per]).astype(np.float64)
X = g0n + idx_np * px
a_np = np.stack([PH.lookup(Al[d], X, box, px) for d in range(8)])
a_t = G.lookup_t(torch.as_tensor(Al), torch.as_tensor(X), box, px).numpy()
print("lookup max diff", float(np.abs(a_np - a_t).max()))
cd = np.stack([PH.lookup(Sl[d], X, box, px) for d in range(8)]) / np.where(a_np > 0, a_np, 1.0)
use = vis_np & (a_np > 0).all(0)
s2 = np.maximum(cd.var(0, ddof=1) / 8, 1e-12)
T_np, n_np, _ = PH.stat(cd.mean(0), use, s2)
T_t, n_t, _ = G.stat_t(torch.as_tensor(cd.mean(0)), torch.as_tensor(use).double(), torch.as_tensor(s2))
ok = use.sum(1) >= 3
print("T rel diff (testable)", float(np.abs(T_np[ok] - T_t.numpy()[ok]).max() / max(np.abs(T_np[ok]).max(), 1e-30)),
      "normal max diff", float(np.abs(n_np[ok] - n_t.numpy()[ok]).max()))
