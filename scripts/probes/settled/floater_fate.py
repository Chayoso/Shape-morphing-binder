"""floater_fate.py ARCHIVE_NPZ RUN_JSON TERMS_DIR [LINK_A] — which detached material is brought in, by which channel.

At the start of a few windows (term dump with the channel record), the particles that are not connected to the body
(single linkage at LINK_A particle spacings, default 2) and lie beyond the sampling berth of the target are split by
the size of their component at that moment. Per class: how many, the share within the berth at the end of the run, the
distance to the target then and at the end, and the displacement toward the target (along each particle's direction
to its nearest target point at each window's start) summed from that window to the end, per channel: the MPM
advection, the u control, the rest (relaxation on the layer; bond projection on decoupled particles). And the share of
each particle's 32 nearest neighbours (the layer normal's neighbourhood) that are fellow members of its component."""
import glob, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402

dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
run = json.load(open(sys.argv[2]))
link_a = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
cfg = run["arms"]["render_full_dt_iso_nn"]["config"]
frames = z["frames"]
n_del = int(z["deliver_n"])
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
sp = float(knn_self_torch(tgt, 2)[0][:, 1].median())
berth = float(cfg.get("nn_berth_k", 1.97)) * sp
tknn = gpu.KNN(tgt)
X = lambda raw: torch.as_tensor(np.asarray(frames[raw], np.float32), device=dev)  # noqa: E731
a = float(knn_self_torch(X(0), 2)[0][:, 1].median())
link = link_a * a
xe = X(n_del - 1)
d_end = tknn.query(xe, 1)[0][:, 0].float()
files = sorted(glob.glob(os.path.join(sys.argv[3], "terms_*.npz")))
CH = ("d_adv", "d_u", "d_rest")
CLASSES = (("1", 1, 1), ("2-4", 2, 4), ("5-23", 5, 23), ("24 or more", 24, 10 ** 9))


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
            return label, nb
        label = new


def toward(k):
    """(N, 3): each channel's displacement of window k along the direction to the nearest target point at its start."""
    f = np.load(files[k])
    x0 = torch.as_tensor(f["c_x0"], device=dev)
    n = torch.nn.functional.normalize(tgt[tknn.query(x0, 1)[1][:, 0]] - x0, dim=1)
    return torch.stack([(torch.as_tensor(f["c_" + c], device=dev) * n).sum(1) for c in CH], 1)


T = [toward(k) for k in range(len(files))]
print(f"N {len(xe)}, {len(files)} dumped windows, end frame {n_del - 1}; sp {sp:.4f} wu, berth {berth / sp:.2f} sp, linkage {link / sp:.1f} sp")
for k0 in (4, 8, 12, 16, 24):
    if k0 >= len(files):
        continue
    f = np.load(files[k0])
    x0 = torch.as_tensor(f["c_x0"], device=dev)
    lmask = torch.as_tensor(f["c_lmask"], device=dev) > 0.5
    label, nb = components(x0)
    u_, inv, cnt = label.unique(return_inverse=True, return_counts=True)
    size = cnt[inv]
    body = size == cnt.max()
    d0 = tknn.query(x0, 1)[0][:, 0].float()
    out = ~body & (d0 > berth)
    fellow = ((label[nb[:, 1:]] == label[:, None]).float().mean(1))
    summed = torch.stack(T[k0:]).sum(0) / sp
    print(f"\n== window {k0} start: {int((~body).sum())} detached particles, {int(out.sum())} of them beyond the berth")
    print("   component size | particles | on the layer % | fellow members among the 32 nearest % | distance sp then -> at the end (median) | within the berth at the end % | "
          "toward the target from here to the end, sp (advection, u, rest; mean)")
    for name, lo, hi in CLASSES:
        m = out & (size >= lo) & (size <= hi)
        c = int(m.sum())
        if c == 0:
            print(f"   {name:10s} | 0")
            continue
        print(f"   {name:10s} | {c:6d} | {100 * float(lmask[m].float().mean()):3.0f} | {100 * float(fellow[m].mean()):3.0f} | {float(d0[m].median()) / sp:5.1f} -> {float(d_end[m].median()) / sp:5.1f} | "
              f"{100 * float((d_end[m] <= berth).float().mean()):3.0f} | " + " ".join(f"{float(summed[m, i].mean()):+6.2f}" for i in range(3)))
