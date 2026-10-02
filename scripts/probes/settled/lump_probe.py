"""lump_probe.py ARCHIVE_NPZ RUN_JSON TERMS_DIR x0,y0,x1,y1 [MIN_OUT_SP=1.0] — the particles behind a lump seen in the
4K picture: who they are, when they got there, what moved them, and what the objective says about them.

The lump is the set of particles of the last delivered frame that project into the pixel box (the 4K renderer's
camera) and lie more than MIN_OUT_SP particle spacings from the target sample. For that set, window by window: the
distance to the target, the share in the outer layer, and the motion of each particle that its neighbours do not
share (its displacement over the window minus an affine fit of its 24 nearest neighbours' displacements), along the
outward normal: the grid moves a neighbourhood together, so this part is the position channels' (the u control, the
layer relaxation, the bonds). The same for the whole outer layer as the reference. Then, from the term dump, each
objective term's pull on the set at chosen windows (inward cosine and size against the all-particle rms)."""
import glob, json, math, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from physmorph.pipeline.window.layer import layer_by_asymmetry, layer_spacing  # noqa: E402
from physmorph.render.covariance_torch import world_to_view_torch  # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402

W, H, AZ, EL = 3840, 2160, 35.0, 18.0
dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
run = json.load(open(sys.argv[2]))
arm = run["arms"]["render_full_dt_iso_nn"]
box = [float(v) for v in sys.argv[4].split(",")]
min_out = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
a = float(run["provenance"]["mpm"]["dx"]) / float(run["provenance"]["ppc"]) ** (1 / 3)
frames = z["frames"]
n_del = int(z["deliver_n"])
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
tknn = gpu.KNN(tgt)
center = tgt.mean(0)
radius = float((tgt - center).norm(dim=1).max())
az, el = math.radians(AZ), math.radians(EL)
cam = center + 3.6 * radius * center.new_tensor((math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)))
view = world_to_view_torch(cam, center)
tan_y = math.tan(math.radians(30) / 2); tan_x = tan_y * W / H


def X(raw):
    return torch.as_tensor(np.asarray(frames[raw], np.float32), device=dev)


def pixels(x):
    p = x @ view[:3, :3].T + view[:3, 3]
    return (p[:, 0] / (p[:, 2] * tan_x) + 1) / 2 * W, (1 - p[:, 1] / (p[:, 2] * tan_y)) / 2 * H


xe = X(n_del - 1)
px, py = pixels(xe)
dte = tknn.query(xe, 1)[0][:, 0].float() / a
inbox = (px >= box[0]) & (px <= box[2]) & (py >= box[1]) & (py <= box[3])
print(f"N {len(xe)}, spacing a {a:.4f} wu, delivered frames {n_del}; particles projecting into the box {int(inbox.sum())}; of them farther than 0.75 / 1 / 1.5 / 2 spacings from the target sample: "
      + " / ".join(str(int((inbox & (dte > t)).sum())) for t in (.75, 1, 1.5, 2)) + f"; over the whole body: " + " / ".join(str(int((dte > t).sum())) for t in (.75, 1, 1.5, 2)))
lump = torch.nonzero(inbox & (dte > min_out)).squeeze(1)
print(f"the lump: {len(lump)} particles (> {min_out} spacings out, in the box); distance median {float(dte[lump].median()):.2f}, max {float(dte[lump].max()):.2f} spacings; "
      f"world centre {[round(float(v), 2) for v in xe[lump].mean(0)]}; indices (the 12 farthest): {lump[torch.argsort(dte[lump], descending=True)][:12].tolist()}")
x0 = X(0)
dk, ik = knn_self_torch(x0, 34)
outer0 = (x0[ik[:, 1:]].mean(1) - x0).norm(dim=1) > 0.35 * dk[:, -1]
depth0 = gpu.KNN(x0[outer0]).query(x0, 1)[0][:, 0].float() / a
print(f"origin: depth below the source surface median {float(depth0[lump].median()):.1f} spacings (p10 {float(torch.quantile(depth0[lump], .1)):.1f}, p90 {float(torch.quantile(depth0[lump], .9)):.1f}); "
      f"source height y {float(x0[lump, 1].min()):+.2f}..{float(x0[lump, 1].max()):+.2f}")

com = [r for r in arm["history"] if r.get("frame_end") and not r.get("null_commit") and r["frame_end"] <= n_del]
print(f"committed windows {len(com)}")


def unshared(xa, xb, ids):
    """Displacement of the particles ids from xa to xb minus the affine fit of their 24 nearest neighbours' displacements."""
    d, idx = gpu.KNN(xa).query(xa[ids], 25)
    idx, d = idx[:, 1:], d[:, 1:].float()
    P, D = xa[idx], (xb - xa)[idx]
    w = torch.exp(-(d / d[:, -1:]) ** 2)[..., None]
    w = w / w.sum(1, keepdim=True)
    pc, dc = (w * P).sum(1), (w * D).sum(1)
    Pc, Dc = P - pc[:, None], D - dc[:, None]
    M = (w * Pc).transpose(1, 2) @ Pc
    M = M + 1e-3 * (M.diagonal(dim1=1, dim2=2).sum(1) / 3)[:, None, None] * torch.eye(3, device=dev)
    A = ((w * Dc).transpose(1, 2) @ Pc) @ torch.linalg.inv(M)
    return (xb - xa)[ids] - (dc + (A @ (xa[ids] - pc)[..., None])[..., 0])


print("window | raw start..end | lump: distance to target median (max), in the outer layer %, unshared motion along the normal: median (p90) spacings, total displacement median | "
      "outer layer (reference): unshared along the normal median, |unshared| p90 | record: kin, alpha, u_gate")
rows = list(range(0, min(len(com), 24))) + list(range(24, len(com), max(1, (len(com) - 24) // 12)))
for i in rows:
    r = com[i]
    fe = r["frame_end"]
    xa, xb = X(fe - 41), X(fe - 1)
    mask, nrm = layer_by_asymmetry(xa, layer_spacing(xa))
    layer = torch.nonzero(mask).squeeze(1)
    ul = unshared(xa, xb, lump)
    ur = unshared(xa, xb, layer[:: max(1, len(layer) // 3000)])
    nl = (ul * nrm[lump]).sum(1) / a
    nr = (ur * nrm[layer[:: max(1, len(layer) // 3000)]]).sum(1) / a
    dta = tknn.query(xb[lump], 1)[0][:, 0].float() / a
    print(f"{i:4d} | {fe - 41:5d}..{fe - 1:5d} | {float(dta.median()):5.2f} ({float(dta.max()):5.2f}) {100 * float(mask[lump].float().mean()):3.0f} % {float(nl.median()):+.3f} ({float(torch.quantile(nl, .9)):+.3f}) "
          f"{float(((xb - xa)[lump].norm(dim=1) / a).median()):.3f} | {float(nr.median()):+.3f} {float(torch.quantile(ur.norm(dim=1) / a, .9)):.3f} | "
          f"{r.get('kin'):.2g} {r.get('alpha_last'):.2g} {r.get('u_gate')}")

files = sorted(glob.glob(os.path.join(sys.argv[3], "terms_*.npz")))
if files:
    print(f"\nterm gradients on the lump ({len(files)} dumped windows): window | per term: inward component of the descent, mean over the lump, in units of the all-particle rms of the summed gradient "
          "(ot surf near spray rend) | share of the lump with the near band on, the spray on")
    T = ("ot", "surf", "near", "spray", "rend")
    for k in sorted({min(k, len(files) - 1) for k in (6, 8, 10, 12, 16, 24, 40, 80, 150, len(files) - 1)}):
        d = np.load(files[k])
        x = torch.as_tensor(d["x"], device=dev)
        g = {t: torch.as_tensor(d["g_" + t], device=dev) for t in T}
        tot = sum(g.values())
        u = float(tot.pow(2).sum(1).mean().sqrt())
        it = tknn.query(x[lump], 1)[1][:, 0]
        n = torch.nn.functional.normalize(tgt[it] - x[lump], dim=1)
        dt = (tgt[it] - x[lump]).norm(dim=1) / a
        print(f"   {os.path.basename(files[k])} (distance {float(dt.median()):.2f} sp) | " + " ".join(f"{float(((-g[t][lump] * n).sum(1)).mean()) / u:+7.2f}" for t in T)
              + f" | {100 * float((g['near'][lump].norm(dim=1) > 0).float().mean()):.0f} % {100 * float((g['spray'][lump].norm(dim=1) > 0).float().mean()):.0f} %")
