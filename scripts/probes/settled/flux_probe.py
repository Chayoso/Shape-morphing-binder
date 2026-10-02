"""flux_probe.py ARCHIVE_NPZ TERMS_DIR — D11 addendum: is the fringe a fixed set of particles or an exchange?

Particles are binned by their distance to the target cloud (< 1, 1-1.5, 1.5-3, > 3 target spacings) at several
committed windows; the tables count where the particles of each bin are at a later window. Then, for the particles
1.5-3 and > 3 spacings out at a window, the inward motion realised to the next dumped window against the inward pull
of the summed gradient (is the pull followed?), and whether the particle is in the outer layer of the body at that
state (the layer that carries the u control)."""
import glob, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402

dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
sp = float(knn_self_torch(tgt, 9)[0][:, 1].median())
tknn = gpu.KNN(tgt)
files = sorted(glob.glob(os.path.join(sys.argv[2], "terms_*.npz")))
EDGES = torch.tensor([1.0, 1.5, 3.0], device=dev)
NAMES = ("< 1", "1-1.5", "1.5-3", "> 3")


def load(k, grads=False):
    d = np.load(files[k])
    x = torch.as_tensor(d["x"], device=dev)
    dt, it = tknn.query(x, 1)
    out = [x, dt[:, 0] / sp, torch.nn.functional.normalize(tgt[it[:, 0]] - x, dim=1)]
    if grads:
        out.append(sum(torch.as_tensor(d[g], device=dev) for g in ("g_ot", "g_surf", "g_near", "g_spray", "g_rend")))
        out.append(torch.as_tensor(d["g_near"], device=dev))
        out.append(torch.as_tensor(d["g_rend"], device=dev))
    return out


last = len(files) - 1
for a, b in ((8, 20), (20, last), (min(40, last - 1), last)):
    xa, da, _ = load(a)
    xb, db, _ = load(b)
    ba, bb = torch.bucketize(da, EDGES), torch.bucketize(db, EDGES)
    print(f"\n== where the particles of window {a} are at window {b} (rows: bin at {a}; columns: bin at {b}; target spacings from the target cloud)")
    print("   " + " " * 9 + " | " + " ".join(f"{n:>8s}" for n in NAMES) + " | total")
    for i, n in enumerate(NAMES):
        row = [int(((ba == i) & (bb == j)).sum()) for j in range(4)]
        print(f"   {n:>9s} | " + " ".join(f"{v:8d}" for v in row) + f" | {sum(row)}")
    print("   " + f"{'total':>9s} | " + " ".join(f"{int((bb == j).sum()):8d}" for j in range(4)))

print("\n== is the pull followed? per window: particles of a bin, the median inward motion to the next dumped window (sp), the share moving inward,")
print("   the share whose summed gradient pulls inward, and the outer-layer share (centroid of 33 neighbours off the particle)")
for k in (6, 12, 20, 30, 45, last - 1):
    if k + 1 > last:
        continue
    x, d, n, g, gn, gr = load(k, grads=True)
    x2, d2, _ = load(k + 1)
    dk, ik = knn_self_torch(x, 34)
    outer = (x[ik[:, 1:]].mean(1) - x).norm(dim=1) > 0.35 * dk[:, -1]
    inward = ((x2 - x) * n).sum(1) / sp
    for name, m in (("1.5-3", (d > 1.5) & (d <= 3)), ("> 3", d > 3), ("0.3-1", (d > 0.3) & (d <= 1))):
        c = int(m.sum())
        if c == 0:
            continue
        pull = (-g[m] * n[m]).sum(1)
        near_on = gn[m].norm(dim=1) > 0
        s = f"   window {k:3d} -> {k + 1:3d} | {name:>6s}: {c:6d} | inward motion median {float(inward[m].median()):+.4f} sp (p10 {float(torch.quantile(inward[m], .1)):+.4f}, p90 {float(torch.quantile(inward[m], .9)):+.4f}), moving inward {100 * float((inward[m] > 0).float().mean()):3.0f} % | pulled inward {100 * float((pull > 0).float().mean()):3.0f} % | outer layer {100 * float(outer[m].float().mean()):3.0f} %"
        if int(near_on.sum()) > 0:
            s += f" | where the near band is on ({int(near_on.sum())}): inward motion median {float(inward[m][near_on].median()):+.4f} sp, distance change {float((d2[m][near_on] - d[m][near_on]).median()):+.4f} sp"
        print(s)
