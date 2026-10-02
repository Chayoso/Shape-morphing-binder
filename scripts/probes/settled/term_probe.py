"""term_probe.py ARCHIVE_NPZ TERMS_DIR — D11: on the floating particles of a run, which objective term pulls where.

Per dumped window (terms_<animation>.npz: the committed state x and the position gradient of the transport, the
surface term, the near band, the spray cleanup and the weighted render), for sets of particles chosen at that state
(above the ear base and more than 3 target spacings out; 1.5-3 spacings out; anywhere more than 3 out; on the surface
at 0.3-1 spacing as the reference): the share on which each local term is exactly zero, each term's rms gradient per
particle, the cosine of each term's descent direction with the direction to the nearest target point, and the cosine
of the near band's pull with the sum of the other terms."""
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
td, _ = knn_self_torch(tgt, 9)
sp, cov_r = float(td[:, 1].median()), float(td[:, 8].median())
tknn = gpu.KNN(tgt)
y0, y1 = float(tgt[:, 1].min()), float(tgt[:, 1].max())
y_base = y0 + 0.72 * (y1 - y0)
files = sorted(glob.glob(os.path.join(sys.argv[2], "terms_*.npz")))
TERMS = ("ot", "surf", "near", "spray", "rend")
print(f"{len(files)} dumped windows; target spacing {sp:.4f}, coverage radius {cov_r / sp:.2f} sp, ear base y {y_base:+.2f}")


def load(f):
    d = np.load(f)
    x = torch.as_tensor(d["x"], device=dev)
    g = {k: torch.as_tensor(d["g_" + k], device=dev) for k in TERMS}
    dt, it = tknn.query(x, 1)
    n = torch.nn.functional.normalize(tgt[it[:, 0]] - x, dim=1)
    d8 = knn_self_torch(x, 9)[0][:, 8] / cov_r
    return x, g, dt[:, 0] / sp, n, d8


def rms(v):
    return float(v.pow(2).sum(1).mean().sqrt()) if len(v) else float("nan")


def cosm(a, b):
    """Cosine over a set, taken as one long vector."""
    na, nb = float(a.norm()), float(b.norm())
    return float((a * b).sum()) / (na * nb) if na > 0 and nb > 0 else float("nan")


def table(k, f, detail=False):
    x, g, dt, n, d8 = load(f)
    tot = sum(g.values())
    sets = {"ears > 3 sp": (dt > 3) & (x[:, 1] > y_base), "ears 1.5-3 sp": (dt > 1.5) & (dt <= 3) & (x[:, 1] > y_base),
            "anywhere > 3 sp": dt > 3, "anywhere 1.5-3 sp": (dt > 1.5) & (dt <= 3), "surface 0.3-1 sp": (dt > 0.3) & (dt <= 1)}
    print(f"\n== window {k} ({os.path.basename(f)}): rms gradient per particle over all particles: " + ", ".join(f"{t} {rms(g[t]):.2e}" for t in TERMS) + f", sum {rms(tot):.2e}")
    for name, m in sets.items():
        c = int(m.sum())
        if c == 0:
            print(f"   {name:18s}: none"); continue
        zn, zs = (g["near"][m].norm(dim=1) == 0), (g["spray"][m].norm(dim=1) == 0)
        line = (f"   {name:18s}: {c:6d} particles | 8NN/coverage median {float(d8[m].median()):.2f} | near band zero on {100 * float(zn.float().mean()):5.1f} %, "
                f"spray zero on {100 * float(zs.float().mean()):5.1f} %, both zero on {100 * float((zn & zs).float().mean()):5.1f} % | rms per particle: "
                + " ".join(f"{t} {rms(g[t][m]):.1e}" for t in TERMS) + f" sum {rms(tot[m]):.1e}")
        print(line)
        print(f"   {'':18s}  pull toward the nearest target point (cosine of the descent direction): "
              + " ".join(f"{t} {cosm(-g[t][m], n[m]):+.2f}" for t in TERMS) + f" sum {cosm(-tot[m], n[m]):+.2f}"
              + f" | near band against the other four {cosm(g['near'][m], (tot - g['near'])[m]):+.2f}, spray against the other four {cosm(g['spray'][m], (tot - g['spray'])[m]):+.2f}"
              + f" | sum opposes the inward direction on {100 * float(((-tot[m] * n[m]).sum(1) < 0).float().mean()):.0f} % of them")
        if detail and c <= 400:
            ids = torch.nonzero(m).squeeze(1)
            ids = ids[torch.argsort(dt[ids], descending=True)][:12]
            print("      index | distance sp | 8NN/cov | inward component of the descent per term (ot surf near spray rend), in units of the all-particle rms of the sum")
            u = rms(tot)
            for i in ids:
                i = int(i)
                print(f"      {i:7d} | {float(dt[i]):5.2f} | {float(d8[i]):.2f} | " + " ".join(f"{float((-g[t][i] * n[i]).sum()) / u:+8.2f}" for t in TERMS)
                      + f" | world {' '.join(f'{float(v):+.2f}' for v in x[i])}")


last = len(files) - 1
for k in sorted({min(k, last) for k in (1, 2, 4, 6, 8, 12, 16, 20, 30, 45, last)}):
    table(k, files[k], detail=k in (6, 12, last))
