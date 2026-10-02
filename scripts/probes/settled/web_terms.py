"""web_terms.py ARCHIVE_NPZ TERMS_DIR [Y_FRAC=0.72] — D11 addendum: which term's direction the far particles of the top
region follow. For the particles above the base height and more than 3 target spacings out at a window: the realised
displacement to the next window against each term's descent direction (cosine over the set), each term's upward
component, and the realised motion's own direction (upward, toward the nearest target point)."""
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
y0, y1 = float(tgt[:, 1].min()), float(tgt[:, 1].max())
y_base = y0 + (float(sys.argv[3]) if len(sys.argv) > 3 else 0.72) * (y1 - y0)
files = sorted(glob.glob(os.path.join(sys.argv[2], "terms_*.npz")))
TERMS = ("ot", "surf", "near", "spray", "rend")
up = torch.tensor([0., 1., 0.], device=dev)


def cosm(a, b):
    na, nb = float(a.norm()), float(b.norm())
    return float((a * b).sum()) / (na * nb) if na > 0 and nb > 0 else float("nan")


print("window | set size | realised motion: median sp, upward cos, toward-target cos | cos(realised, descent of term): " + " ".join(TERMS) + " sum | upward cos of the descent: " + " ".join(TERMS)
      + " | rms per particle: " + " ".join(TERMS))
for k in range(0, min(12, len(files) - 1)):
    d = np.load(files[k])
    x = torch.as_tensor(d["x"], device=dev)
    x2 = torch.as_tensor(np.load(files[k + 1])["x"], device=dev)
    g = {t: torch.as_tensor(d["g_" + t], device=dev) for t in TERMS}
    dt, it = tknn.query(x, 1)
    n = torch.nn.functional.normalize(tgt[it[:, 0]] - x, dim=1)
    m = (dt[:, 0] / sp > 3) & (x[:, 1] > y_base)
    if int(m.sum()) < 10:
        continue
    dx = (x2 - x)[m]
    tot = sum(g.values())
    print(f"{k:4d} -> {k + 1:2d} | {int(m.sum()):6d} | {float(dx.norm(dim=1).median() / sp):5.2f} sp, up {cosm(dx, up.expand_as(dx)):+.2f}, to target {cosm(dx, n[m]):+.2f} | "
          + " ".join(f"{cosm(dx, -g[t][m]):+.2f}" for t in TERMS) + f" {cosm(dx, -tot[m]):+.2f} | "
          + " ".join(f"{cosm(-g[t][m], up.expand_as(dx)):+.2f}" for t in TERMS) + " | "
          + " ".join(f"{float(g[t][m].pow(2).sum(1).mean().sqrt()):.1e}" for t in TERMS))
