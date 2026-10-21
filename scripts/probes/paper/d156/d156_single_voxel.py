"""d156_single_voxel.py MESH [K] -- D156c's diagnostic (uses the true mesh; not part of any carve): at the visual hull,
does removing ONE phantom surface voxel (at least one voxel outside the mesh) lower the run's render loss? K phantom
and K on-mesh surface voxels drawn at random (seed 0), each removed alone from the carried sample (d156_rcarve.py's:
the visual hull's sample at the mesh's density, shell particles in their nearest occupied node); the loss change of
each. Also the coordinated removal of ALL phantom surface voxels at once, and of all phantom voxels (the whole excess
at the hull's grid), for comparison with D156's carve path."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d156")
import d156_photohull as PH  # noqa: E402
from d156_photohull import D7, o3d, torch, nnf, map_coordinates  # noqa: E402
from physmorph.pipeline.render_loss import d_exterior, exterior_targets  # noqa: E402
from physmorph.render.exterior import Lattice, ZhuBridson  # noqa: E402
from physmorph.sampling.mesh import _oriented, draws_in_frame, fill_volume_in_frame, load_normalized, sample_in_frame  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402

cfg, AMB, views, SEED = PH.cfg, PH.AMB, PH.views, PH.SEED
mesh = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 40
N, RES = 40000, 96
A = "/data/relcfd/chayo/physmorph_v2/output/gpu/d156/assets_hull"
path, hpath = f"assets/{mesh}.obj", f"{A}/{mesh}_hull18_r96.obj"
_, v_src = load_normalized("assets/isosphere.obj", N, SEED, return_volume=True, sample="stratified")
frame = {}
tgt = load_normalized(path, N, SEED + 1, match_volume=v_src, sample="stratified", frame=frame)
box = float(np.abs(tgt).max()) * 1.25
px = 2.0 * box / RES
m = _oriented(path)
Vw = (np.asarray(m.vertices, np.float64) - frame["offset"]) * frame["scale"]
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.core.Tensor(Vw.astype(np.float32)), o3d.core.Tensor(np.asarray(m.faces, np.uint32)))
tc, _ = D7.close_holes(Vw, np.asarray(m.faces, np.int64))
true = D7.Surf(np.asarray(tc.vertices), np.asarray(tc.faces))
samples = [tgt] + draws_in_frame(path, frame, N, [SEED + 1 + 100 * k for k in range(1, 8)])
pitch = 0.708 * D7.median_kth_spacing(tgt, 8, 20000)
T0 = torch.as_tensor(tgt)
center = T0.mean(0)
lattice = Lattice(center, 2.8 * float((T0 - center).norm(dim=1).max()))
h = min(box / RES, 0.92 * pitch)


def discs(x):
    p, g, _, _ = lattice.discs(ZhuBridson(torch.as_tensor(np.asarray(x, np.float32)), pitch), h, refine=False)
    return p, nnf.normalize(g, dim=1)


per = [exterior_targets(*discs(s), views, RES, box, cfg.sil_k, AMB) for s in samples]
sils = [torch.stack(v).mean(0) for v in zip(*[e[0] for e in per])]
shade = [torch.stack(v).mean(0) for v in zip(*[e[1] for e in per])]


def loss(x):
    p, n = discs(x)
    s, b = d_exterior(p, n, sils, shade, views, RES, box, cfg.sil_k, cfg.w_hole, cfg.w_spray, AMB)
    return float(s + cfg.w_pbr * b)


sds = [PH.inside_sd(PH.silhouette(scene, box, RES, th, ph)) for th, ph in views]
g = -box - px + px * np.arange(RES + 3)
G = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
f = np.full(len(G), np.inf)
for (th, ph), sd in zip(views, sds):
    r, u, _ = PH.basis(th, ph)
    f = np.minimum(f, map_coordinates(sd, [(G @ r + box) / px, (G @ u + box) / px], order=1, mode="constant", cval=-1e9))
f = f.reshape(len(g), len(g), len(g))
f[[0, -1], :, :] = f[:, [0, -1], :] = f[:, :, [0, -1]] = -1.0
occ0 = f > 0
shape = np.array(occ0.shape)
n_vh = int(round(N * fill_volume_in_frame(hpath, frame, N) / fill_volume_in_frame(path, frame, N)))
xs, _ = sample_in_frame(hpath, frame, n_vh, SEED + 1)
node = np.clip(np.rint((xs - g[0]) / px).astype(np.int64), 0, shape - 1)
_, near = distance_transform_edt(~occ0, return_indices=True)
node = np.stack([near[a][node[:, 0], node[:, 1], node[:, 2]] for a in range(3)], 1)
nk = (node[:, 0] * shape[1] + node[:, 1]) * shape[2] + node[:, 2]
L0 = loss(xs)
idx = PH.surface(occ0)
X = g[0] + idx * px
d = true.signed(X)
allocc = np.argwhere(occ0)
dall = true.signed(g[0] + allocc * px)
rng = np.random.default_rng(0)
out = dict(mesh=mesh, loss_hull=L0, px=px)
for name, mask in (("phantom", d >= px), ("on_mesh", np.abs(d) <= 0.5 * px)):
    pick = rng.choice(np.flatnonzero(mask), min(K, int(mask.sum())), replace=False)
    dl = []
    for i in pick:
        key = (idx[i, 0] * shape[1] + idx[i, 1]) * shape[2] + idx[i, 2]
        keep = nk != key
        dl.append(loss(xs[keep]) - L0 if (~keep).any() else 0.0)
    dl = np.array(dl)
    out[name] = dict(n=len(dl), share_lower=float((dl < 0).mean()), median_rel=float(np.median(dl) / L0),
                     q10_rel=float(np.quantile(dl, .1) / L0), q90_rel=float(np.quantile(dl, .9) / L0))
for name, sel in (("all_phantom_surface", idx[d >= px]), ("all_excess_voxels", allocc[dall >= px])):
    keys = set(((sel[:, 0] * shape[1] + sel[:, 1]) * shape[2] + sel[:, 2]).tolist())
    keep = ~np.isin(nk, np.fromiter(keys, np.int64))
    out[name] = dict(voxels=len(sel), particles=int((~keep).sum()), loss_rel=(loss(xs[keep]) - L0) / L0)
print(json.dumps(out), flush=True)
