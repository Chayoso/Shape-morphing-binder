"""clump_probe.py ARCHIVE_NPZ RUN_JSON TERMS_DIR — the floating Gaussians as clumps: where the rendered particles far
from the target sit at the end, how they are grouped, and how the largest groups came to be.

Floaters = particles of the last delivered frame more than 3 target spacings from the target sample and rendered
(support > 0). They are grouped by single linkage at one coverage radius. Per group: size, distance to the target
(target spacings and loss cells), density (8th-neighbour distance over the coverage radius: below 1.2 the spray
cleanup's isolation gate is shut), whether the near band reaches it (within one loss cell), 4K pixel, world centre.
For the largest groups, window by window from the term dump: distance, density, radius of the group, the share on
which the near band and the spray cleanup have a non-zero gradient, each term's inward pull, the group's source
bonds that stay inside the group, and the motion of its centre toward the target."""
import glob, json, math, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from physmorph.render.covariance_torch import world_to_view_torch  # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.support import live_support              # noqa: E402

W, H, AZ, EL = 3840, 2160, 35.0, 18.0
dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
run = json.load(open(sys.argv[2]))
frames = z["frames"]
n_del = int(z["deliver_n"])
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
td = knn_self_torch(tgt, 9)[0]
sp, cov = float(td[:, 1].median()), float(td[:, 8].median())
ldx = 2 * (1.25 * float(max(np.abs(np.asarray(z["src"])).max(), np.abs(np.asarray(z["tgt"])).max())) + 2 * float(run["provenance"]["mpm"]["dx"])) / int(run["arms"]["render_full_dt_iso_nn"]["config"]["loss_res"])
tknn = gpu.KNN(tgt)
center = tgt.mean(0)
radius = float((tgt - center).norm(dim=1).max())
az, el = math.radians(AZ), math.radians(EL)
cam = center + 3.6 * radius * center.new_tensor((math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)))
view = world_to_view_torch(cam, center)
tan_y = math.tan(math.radians(30) / 2); tan_x = tan_y * W / H


def X(raw):
    return torch.as_tensor(np.asarray(frames[raw], np.float32), device=dev)


def pix(c):
    p = view[:3, :3] @ c + view[:3, 3]
    return float((p[0] / (p[2] * tan_x) + 1) / 2 * W), float((1 - p[1] / (p[2] * tan_y)) / 2 * H)


xe = X(n_del - 1)
d = knn_self_torch(xe, 33)[0]
sup = live_support(d, cov, sp)
dt = tknn.query(xe, 1)[0][:, 0].float() / sp
fl = torch.nonzero((dt > 3) & (sup > 0)).squeeze(1)
print(f"N {len(xe)}, last frame {n_del - 1}; target spacing {sp:.4f} wu, coverage radius {cov / sp:.2f} sp, loss cell {ldx:.4f} wu = {ldx / sp:.1f} sp; "
      f"rendered floaters (> 3 sp): {len(fl)}; unrendered particles beyond 3 sp (support 0): {int(((dt > 3) & (sup == 0)).sum())}")
# single linkage at one coverage radius
P = xe[fl]
adj = torch.cdist(P, P) < cov
label = torch.arange(len(P), device=dev)
for _ in range(200):
    new = torch.where(adj, label[None, :].expand_as(adj), label[:, None].expand_as(adj)).min(1).values
    if bool((new == label).all()):
        break
    label = new
groups = sorted(((int((label == l).sum()), int(l)) for l in label.unique().tolist()), reverse=True)
print(f"groups: {len(groups)}; sizes of the ten largest: {[g[0] for g in groups[:10]]}; floaters in groups of 9 or more: {sum(g[0] for g in groups if g[0] >= 9)}, single or in groups under 9: {sum(g[0] for g in groups if g[0] < 9)}")
print("group | size | distance to the target sp (median, max) = loss cells | 8NN/coverage median | beyond one loss cell % | 4K pixel | world centre")
big = []
for size, l in groups[:10]:
    ids = fl[label == l]
    c = xe[ids].mean(0)
    u, v = pix(c)
    print(f"  {size:4d} | {float(dt[ids].median()):5.1f} {float(dt[ids].max()):5.1f} = {float(dt[ids].median()) * sp / ldx:.2f} cells | {float((d[ids, 8] / cov).median()):.2f} | "
          f"{100 * float((dt[ids] * sp > ldx).float().mean()):3.0f} % | ({u:5.0f}, {v:5.0f}) | {[round(float(t), 2) for t in c]}")
    big.append(ids)
# all floaters: which definition leaves them
far = dt[fl] * sp > ldx
dense = d[fl, 8] / cov < 1.2
print(f"all floaters: beyond one loss cell {100 * float(far.float().mean()):.0f} %; denser than the isolation gate's lower edge (8NN/coverage < 1.2) {100 * float(dense.float().mean()):.0f} %; "
      f"both (no local term reaches them by definition) {100 * float((far & dense).float().mean()):.0f} %")

files = sorted(glob.glob(os.path.join(sys.argv[3], "terms_*.npz")))
x0 = X(0)
bond = gpu.knn(x0, 9)[1][:, 1:]
T = ("ot", "surf", "near", "spray", "rend")
for gi, ids in enumerate(big[:3]):
    inside = torch.zeros(len(xe), dtype=torch.bool, device=dev); inside[ids] = True
    print(f"\n== group {gi} ({len(ids)} particles): source bonds (8 a particle) that stay inside the group {100 * float(inside[bond[ids]].float().mean()):.0f} %; "
          f"source positions spread (rms radius) {float((x0[ids] - x0[ids].mean(0)).norm(dim=1).pow(2).mean().sqrt()) / sp:.1f} sp")
    print("   window | distance sp median | 8NN/coverage median | group radius sp | rendered % | near band on % | spray on % | inward pull per term (ot surf near spray rend), units of the all-particle rms of the sum | centre moved toward the target since the last row (sp)")
    prev = None
    for k in sorted({min(k, len(files) - 1) for k in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, len(files) - 1)}):
        f = np.load(files[k])
        x = torch.as_tensor(f["x"], device=dev)
        g = {t: torch.as_tensor(f["g_" + t], device=dev) for t in T}
        rms_all = float(sum(g.values()).pow(2).sum(1).mean().sqrt())
        dd = knn_self_torch(x, 33)[0]
        s = live_support(dd, cov, sp)
        q, it = tknn.query(x[ids], 1)
        n = torch.nn.functional.normalize(tgt[it[:, 0]] - x[ids], dim=1)
        dist = q[:, 0].float() / sp
        c = x[ids].mean(0)
        rad = float((x[ids] - c).norm(dim=1).pow(2).mean().sqrt()) / sp
        moved = "" if prev is None else f"{float(prev - dist.median()):+.2f}"
        prev = dist.median()
        print(f"   {k:4d} | {float(dist.median()):5.1f} | {float((dd[ids, 8] / cov).median()):.2f} | {rad:5.1f} | {100 * float((s[ids] > 0).float().mean()):3.0f} | "
              f"{100 * float((g['near'][ids].norm(dim=1) > 0).float().mean()):3.0f} | {100 * float((g['spray'][ids].norm(dim=1) > 0).float().mean()):3.0f} | "
              + " ".join(f"{float((-g[t][ids] * n).sum(1).mean()) / rms_all:+6.2f}" for t in T) + f" | {moved}")
