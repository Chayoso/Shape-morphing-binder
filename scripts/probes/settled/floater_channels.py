"""floater_channels.py ARCHIVE_NPZ RUN_JSON TERMS_DIR [LINK_A] — what moves the floating material, channel by channel.

The floaters: the end state's rendered particles that are not connected to the body (single linkage at LINK_A particle
spacings, default 2) and lie beyond the sampling berth of the target. The term dump of a run made with the channel
record (telemetry.channel_record) holds, per committed window and particle, the displacement by the MPM advection
(sum of dt v), by the u control and by the rest (the relaxation on the layer; the bond projection on decoupled
particles), with the layer mask, the decoupling flag, the control size and det F.

Per window, for all floaters together and for the largest components:
  toward the target: each channel's displacement along the particle's direction to its nearest target point (mean, sp);
  contraction: minus the mean radial part of each channel's displacement about the component's centre (sp; positive =
    the component shrinks), with the component's rms radius;
  the centre's motion along the centre's direction to the target, per channel (sp);
  det F at the window's ends (median), the mean control increment against the all-particle median, the share on the
  layer and the share decoupled at the last step."""
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
link_a = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
cfg = run["arms"]["render_full_dt_iso_nn"]["config"]
frames = z["frames"]
n_del = int(z["deliver_n"])
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
td = knn_self_torch(tgt, 9)[0]
sp, cov = float(td[:, 1].median()), float(td[:, 8].median())
ldx = 2 * (1.25 * float(max(np.abs(np.asarray(z["src"])).max(), np.abs(np.asarray(z["tgt"])).max())) + 2 * float(run["provenance"]["mpm"]["dx"])) / int(cfg["loss_res"])
berth = float(cfg.get("nn_berth_k", 1.97)) * sp
tknn = gpu.KNN(tgt)
center = tgt.mean(0)
radius = float((tgt - center).norm(dim=1).max())
az, el = math.radians(AZ), math.radians(EL)
cam = center + 3.6 * radius * center.new_tensor((math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)))
view = world_to_view_torch(cam, center)
tan_y = math.tan(math.radians(30) / 2); tan_x = tan_y * W / H
X = lambda raw: torch.as_tensor(np.asarray(frames[raw], np.float32), device=dev)  # noqa: E731
a = float(knn_self_torch(X(0), 2)[0][:, 1].median())
link = link_a * a


def pix(c):
    p = view[:3, :3] @ c + view[:3, 3]
    return float((p[0] / (p[2] * tan_x) + 1) / 2 * W), float((1 - p[1] / (p[2] * tan_y)) / 2 * H)


def components(x, k=33):
    d, nb = knn_self_torch(x, k)
    N = len(x)
    ok = d[:, 1:] < link
    i = torch.arange(N, device=dev)[:, None].expand(-1, k - 1)[ok]
    j = nb[:, 1:][ok]
    label = torch.arange(N, device=dev)
    while True:
        new = label.clone()
        new.scatter_reduce_(0, i, label[j], "amin")
        new.scatter_reduce_(0, j, label[i], "amin")
        new = new[new]
        if bool((new == label).all()):
            return label, d
        label = new


xe = X(n_del - 1)
label, d = components(xe)
u_, cnt = label.unique(return_counts=True)
body = u_[cnt.argmax()]
dt = tknn.query(xe, 1)[0][:, 0].float()
vis = (label != body) & (live_support(d, cov, sp) > 0)
fl = torch.nonzero(vis & (dt > berth)).squeeze(1)
lab = label[fl]
groups = sorted(((int((lab == l).sum()), int(l)) for l in lab.unique().tolist()), reverse=True)
print(f"N {len(xe)}, raw {n_del - 1}: detached rendered particles {int(vis.sum())}, of them beyond the berth ({berth / sp:.2f} sp) {len(fl)} in {len(groups)} components "
      f"(beyond one loss cell = {ldx / sp:.1f} sp: {int((dt[fl] > ldx).sum())}); sp {sp:.4f} wu, a {a:.4f} wu, linkage {link / sp:.1f} sp")
print("   size | distance sp | 8NN/coverage | 4K pixel | world centre")
big = []
for size, l in groups[:8]:
    g = fl[lab == l]
    c = xe[g].mean(0)
    px = pix(c)
    print(f"   {size:4d} | {float(dt[g].median()) / sp:5.1f} | {float((d[g, 8] / cov).median()):.2f} | ({px[0]:5.0f}, {px[1]:5.0f}) | {[round(float(t), 2) for t in c]}")
    big.append(g)

files = sorted(glob.glob(os.path.join(sys.argv[3], "terms_*.npz")))
CH = ("d_adv", "d_u", "d_rest")
sets = {"all floaters": fl, **{f"component {i} ({len(g)})": g for i, g in enumerate(big[:4])}}
out = {k: [] for k in sets}
tot = {k: torch.zeros(3) for k in sets}
for k, fpath in enumerate(files):
    f = np.load(fpath)
    if "c_x0" not in f.files:
        raise SystemExit("this term dump has no channel record (run it on a repo with telemetry.channel_record)")
    x0 = torch.as_tensor(f["c_x0"], device=dev)
    ch = {c: torch.as_tensor(f["c_" + c], device=dev) for c in CH}
    lmask, frag = torch.as_tensor(f["c_lmask"], device=dev), torch.as_tensor(f["c_frag"], device=dev)
    dfc, J0, J1 = (torch.as_tensor(f["c_" + c], device=dev) for c in ("dfc", "J0", "J1"))
    dfc_med = float(dfc.median())
    d8 = knn_self_torch(x0, 9)[0][:, 8] / cov
    for name, m in sets.items():
        q, it = tknn.query(x0[m], 1)
        n = torch.nn.functional.normalize(tgt[it[:, 0]] - x0[m], dim=1)
        c0 = x0[m].mean(0)
        r = x0[m] - c0
        rad = float(r.pow(2).sum(1).mean().sqrt()) / sp
        rh = torch.nn.functional.normalize(r, dim=1)
        nc = torch.nn.functional.normalize(tgt[tknn.query(c0[None], 1)[1][0, 0]] - c0, dim=0)
        toward = [float((ch[c][m] * n).sum(1).mean()) / sp for c in CH]
        contr = [-float(((ch[c][m] - ch[c][m].mean(0)) * rh).sum(1).mean()) / sp for c in CH]
        cen = [float(ch[c][m].mean(0) @ nc) / sp for c in CH]
        tot[name] += torch.tensor(toward)
        if k % 2 == 0 or k == len(files) - 1:
            out[name].append(f"   {k:4d} | {float(q[:, 0].float().median()) / sp:5.1f} | {float(d8[m].median()):.2f} | {rad:5.1f} | "
                             + " ".join(f"{v:+6.2f}" for v in toward) + " | " + " ".join(f"{v:+6.2f}" for v in contr) + " | " + " ".join(f"{v:+6.2f}" for v in cen)
                             + f" | {float(J0[m].median()):.2f} -> {float(J1[m].median()):.2f} | {float(dfc[m].mean()) / max(dfc_med, 1e-30):5.2f} | "
                             f"{100 * float((lmask[m] > 0.5).float().mean()):3.0f} | {100 * float((frag[m] > 0.5).float().mean()):3.0f}")
for name, lines in out.items():
    print(f"\n== {name}: over the run, displacement toward the target summed over the windows (sp): advection {float(tot[name][0]):+.1f}, u {float(tot[name][1]):+.1f}, rest {float(tot[name][2]):+.1f}")
    print("   window | distance sp (start) | 8NN/coverage | rms radius sp | toward the target (advection, u, rest) | contraction (advection, u, rest) | centre toward the target (advection, u, rest) | "
          "det F start -> end | control / all-particle median | on the layer % | decoupled %")
    print("\n".join(lines))
