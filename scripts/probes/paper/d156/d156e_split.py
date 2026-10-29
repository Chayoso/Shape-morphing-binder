"""d156e_split.py MESH -- D156e's diagnostic: where the disc surrogate's gain comes from. With the solved d (the npz of
d156_sfs_e.py), the run's loss of the hull's discs (p0, n0), of the solved surrogate (p0 - d n0, normals from grad d),
of the positions alone (p0 - d n0, n0), of the normals alone (p0, normals from grad d), and of the discs the true
operator finds on a sample of the solved solid (the mean over 8 seeds, the pipeline's sampler); and the solid's discs
against the surrogate's: the distance from each solid disc to the nearest surrogate disc and the normal angle there."""
import os
import sys
from pathlib import Path
os.environ.setdefault("D156_DEV", "cpu")
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import d156_sfs_e as E  # noqa: E402
from d156_sfs_e import D7, nnf, cKDTree  # noqa: E402
from physmorph.pipeline.render_loss import d_exterior, exterior_targets  # noqa: E402
from physmorph.render.exterior import Lattice, key, make_field  # noqa: E402
from physmorph.sampling.mesh import draws_in_frame, fill_volume_in_frame, load_normalized  # noqa: E402

torch.set_grad_enabled(False)
mesh, N, RES = sys.argv[1], 40000, 96
A = "/data/relcfd/chayo/physmorph_v2/output/gpu/d156/assets_hull"
path, hpath, spath = f"assets/{mesh}.obj", f"{A}/{mesh}_hull18_r96.obj", f"{A}/{mesh}_sfse18_r96.obj"
z = np.load(f"{A}/{mesh}_sfse18_r96_d.npz")
p0, n0, d = (torch.as_tensor(z[k], dtype=torch.float32) for k in ("p0", "n0", "d"))
_, v_src = load_normalized("assets/isosphere.obj", N, 97, return_volume=True, sample="stratified")
frame = {}
tgt = load_normalized(path, N, 98, match_volume=v_src, sample="stratified", frame=frame)
box = float(np.abs(tgt).max()) * 1.25
px = 2 * box / RES
pitch = 0.708 * D7.median_kth_spacing(tgt, 8, 20000)
T0 = torch.as_tensor(tgt); center = T0.mean(0)
lattice = Lattice(center, 2.8 * float((T0 - center).norm(dim=1).max())); h = min(box / RES, 0.92 * pitch)
cfg, AMB, views = E.cfg, E.AMB, E.views


def discs_of(x):
    p, g, _, _ = lattice.discs(make_field("zb", torch.as_tensor(np.asarray(x, np.float32)), pitch), h, refine=False)
    return p, nnf.normalize(g, dim=1)


per = [exterior_targets(*discs_of(s), views, RES, box, cfg.sil_k, AMB)
       for s in [tgt] + draws_in_frame(path, frame, N, [98 + 100 * k for k in range(1, 8)])]
sils = [torch.stack(v).mean(0) for v in zip(*[e[0] for e in per])]
shade = [torch.stack(v).mean(0) for v in zip(*[e[1] for e in per])]


def L(p, n):
    s, b = d_exterior(p, n, sils, shade, views, RES, box, cfg.sil_k, cfg.w_hole, cfg.w_spray, AMB)
    return round(float(s) * 1e3, 3), round(float(b) * 1e3, 3), round(float(s + cfg.w_pbr * b) * 1e3, 3)


# the surrogate's normals (as d156_sfs_e: tangential least-squares gradient over the lattice neighbours)
M = len(p0)
cell = torch.floor((p0 - lattice.origin) / h).long()
ck = key(cell); order = torch.argsort(ck); sk = ck[order]
I, J = [], []
for off in torch.stack(torch.meshgrid(*[torch.arange(-1, 2)] * 3, indexing="ij"), -1).reshape(-1, 3):
    if bool((off == 0).all()):
        continue
    nk = key(cell + off); at = torch.searchsorted(sk, nk).clamp(max=M - 1); hit = sk[at] == nk
    I.append(torch.nonzero(hit).squeeze(1)); J.append(order[at[hit]])
I, J = torch.cat(I), torch.cat(J)
dp = p0[J] - p0[I]; dpt = dp - (dp * n0[I]).sum(1, keepdim=True) * n0[I]
Am = torch.zeros(M, 3, 3).index_add_(0, I, dpt[:, :, None] * dpt[:, None, :])
Am = Am + n0[:, :, None] * n0[:, None, :] * (Am.diagonal(dim1=1, dim2=2).sum(1)[:, None, None] + 1e-12)
Minv = torch.linalg.inv(Am.double()).float()
S = torch.zeros(M, 3).index_add(0, I, dpt * (d[J] - d[I])[:, None])
gs = torch.einsum("mij,mj->mi", Minv, S); gs = gs - (gs * n0).sum(1, keepdim=True) * n0
pd, nd = p0 - d[:, None] * n0, nnf.normalize(n0 + gs, dim=1)
out = dict(mesh=mesh, hull_discs=L(p0, n0), surrogate=L(pd, nd), positions_only=L(pd, n0), normals_only=L(p0, nd))
n_s = int(round(len(tgt) * fill_volume_in_frame(spath, frame, N) / fill_volume_in_frame(path, frame, N)))
sol = [discs_of(x) for x in draws_in_frame(spath, frame, n_s, [98 + 100 * k for k in range(8)])]
vals = np.array([L(p, n) for p, n in sol])
out["solid_samples_mean"] = [round(float(v), 3) for v in vals.mean(0)]
ps, ns = sol[0]
dist, j = cKDTree(pd.numpy()).query(ps.numpy())
ang = np.degrees(np.arccos(np.clip((ns.numpy() * nd.numpy()[j]).sum(1), -1, 1)))
ang0 = np.degrees(np.arccos(np.clip((ns.numpy() * n0.numpy()[cKDTree(p0.numpy()).query(ps.numpy())[1]]).sum(1), -1, 1)))
out["solid_vs_surrogate"] = dict(dist_px_q=[round(float(v), 3) for v in np.quantile(dist / px, [.5, .9])],
                                 normal_deg_vs_surrogate_q=[round(float(v), 2) for v in np.quantile(ang, [.5, .9])],
                                 normal_deg_vs_hull_q=[round(float(v), 2) for v in np.quantile(ang0, [.5, .9])],
                                 surrogate_tilt_deg_q=[round(float(v), 2) for v in np.quantile(
                                     np.degrees(np.arccos(np.clip((nd * n0).sum(1).numpy(), -1, 1))), [.5, .9, .99])])
print(json.dumps(out), flush=True)
