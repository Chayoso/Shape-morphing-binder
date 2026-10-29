import sys, json, numpy as np, torch
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/repo_d156"); sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d156")
import d156_photohull as PH
from d156_photohull import D7, o3d, trimesh, marching_cubes, nnf
from physmorph.pipeline.render_loss import d_exterior, exterior_targets
from physmorph.render.exterior import Lattice, ZhuBridson
from physmorph.sampling.mesh import _oriented, load_normalized, sample_in_frame, stratified_draws, fill_volume_in_frame, draws_in_frame
cfg, AMB, views = PH.cfg, PH.AMB, PH.views
A = "/data/relcfd/chayo/physmorph_v2/output/gpu/d156/assets_hull"
mesh = sys.argv[1]; N = 40000; RES = 96
path, hp = f"assets/{mesh}.obj", f"{A}/{mesh}_hull18_r96.obj"
_, v_src = load_normalized("assets/isosphere.obj", N, 97, return_volume=True, sample="stratified")
fr = {}; tgt = load_normalized(path, N, 98, match_volume=v_src, sample="stratified", frame=fr)
box = float(np.abs(tgt).max()) * 1.25
pitch = 0.708 * D7.median_kth_spacing(tgt, 8, 20000)
T0 = torch.as_tensor(tgt); center = T0.mean(0)
lattice = Lattice(center, 2.8 * float((T0 - center).norm(dim=1).max())); h = min(box / RES, 0.92 * pitch)
def discs(x):
    p, g, _, _ = lattice.discs(ZhuBridson(torch.as_tensor(np.asarray(x, np.float32)), pitch), h, refine=False)
    return p, nnf.normalize(g, dim=1)
samples = [tgt] + draws_in_frame(path, fr, N, [98 + 100 * k for k in range(1, 8)])
per = [exterior_targets(*discs(s), views, RES, box, cfg.sil_k, AMB) for s in samples]
sils = [torch.stack(v).mean(0) for v in zip(*[e[0] for e in per])]; shade = [torch.stack(v).mean(0) for v in zip(*[e[1] for e in per])]
def loss(x):
    p, n = discs(x); s, b = d_exterior(p, n, sils, shade, views, RES, box, cfg.sil_k, cfg.w_hole, cfg.w_spray, AMB)
    return round(float(s + cfg.w_pbr * b) * 1e3, 3), round(float(s) * 1e3, 3), round(float(b) * 1e3, 3), len(x)
n_vh = int(round(N * fill_volume_in_frame(hp, fr, N) / fill_volume_in_frame(path, fr, N)))
out = {"mesh": mesh, "n_vh": n_vh}
for sd in (98, 198):
    xs, _ = sample_in_frame(hp, fr, n_vh, sd); out[f"carried_seed{sd}"] = loss(xs)
hm = _oriented(hp); Hw = (np.asarray(hm.vertices, np.float64) - fr["offset"]) * fr["scale"]
H = trimesh.Trimesh(Hw, np.asarray(hm.faces), process=True)
for sd in (98, 198):
    x = stratified_draws(H, n_vh, [sd])[0]; out[f"world_hull_seed{sd}"] = loss(x)
    out[f"world_hull_seed{sd}_mean_offset"] = [round(float(v), 4) for v in (x.mean(0) - xs.mean(0))]
print(json.dumps(out), flush=True)
