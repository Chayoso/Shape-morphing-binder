"""d173_edge.py OUT_JSONL JOBS -- D173 (2): how wide is the band's colour edge on the surface, and where does the width come
from? (one card for the exterior's operators)

JOBS lines: `NAME FRAMES12` (the band runs: their src_albedo, the band plane y = tgt_y from the archive's appearance), and
the target paint itself (the archive's tgt sample painted by tgt_albedo) once per archive, `NAME@tgt`.
Edge width = the 10-90 % rise of the red share across the band plane, in pitches of the run's sample (a = 0.708 x the
8th-neighbour median distance of its target sample), read on a profile of the signed height s = (y - tgt_y) / a in bins of
0.25 pitch over [-8, 8] (the profile made monotone by its running maximum; the 10 % and 90 % crossings interpolated):
  particles   the state's outer layer (the pipeline's rule), each particle's own paint (red or white): the identity layout
  discs_k3    the exterior discs (the loss's: Zhu-Bridson radius 3 pitches, offset 0.8, lattice 0.46 a as the deliverable's
              d160_studio) coloured by render/exterior.found_disc_albedo (the kernel (1 - d^2/R^2)^3 over the particles
              within R = 3 pitches + one pitch of skin): what the loss and the display show
  discs_nn    the same discs coloured by their nearest particle alone (the identity, no kernel)
  discs_k1    the same discs, found_disc_albedo's kernel at R = 1 pitch (the identity smoothed over one pitch)
Also: the share of outer-layer particles on the wrong side of the plane by more than 2 pitches (the identity scatter)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path.cwd()))
from physmorph import gpu  # noqa: E402
from physmorph.render.exterior import Lattice, found_disc_albedo, make_field  # noqa: E402

out_path, jobs = sys.argv[1], [x.split() for x in open(sys.argv[2]) if x.strip()]
RED = np.array([0.9, 0.25, 0.2])
BINS = np.arange(-8.0, 8.0001, 0.25)


def layer(X):
    sub = X[np.random.default_rng(0).choice(len(X), min(20000, len(X)), replace=False)]
    sp = float(np.median(cKDTree(sub).query(sub, 9)[0][:, 8])) * (len(sub) / len(X)) ** (1 / 3)
    _, nb = cKDTree(X).query(X, 33, workers=16)
    return np.linalg.norm(X - X[nb[:, 1:]].mean(1), axis=1) >= 0.5 * sp


def width(s, red):
    """(10-90 % width, the 50 % crossing) of the red share against s; None where the profile does not span 0.1-0.9."""
    idx = np.digitize(s, BINS) - 1
    ok = (idx >= 0) & (idx < len(BINS) - 1)
    num = np.bincount(idx[ok], weights=red[ok], minlength=len(BINS) - 1)
    cnt = np.bincount(idx[ok], minlength=len(BINS) - 1)
    c = 0.5 * (BINS[1:] + BINS[:-1])
    keep = cnt >= 5
    if keep.sum() < 4:
        return None, None, []
    prof = np.maximum.accumulate(num[keep] / cnt[keep])
    cc = c[keep]
    if prof[0] > 0.1 or prof[-1] < 0.9:
        return None, None, list(zip(cc.tolist(), prof.tolist()))
    x = lambda q: float(np.interp(q, prof + np.arange(len(prof)) * 1e-9, cc))  # noqa: E731
    return x(0.9) - x(0.1), x(0.5), list(zip(cc.tolist(), prof.tolist()))


def is_red(alb):
    return np.linalg.norm(np.asarray(alb, np.float64)[:, :3] - RED, axis=1) < 0.05


def read(name, X, red, tgt, tgt_y):
    a = .708 * float(np.median(cKDTree(tgt[:20000]).query(tgt[:20000], 9)[0][:, 8])) * (20000 / len(tgt)) ** (1 / 3)
    row = dict(name=name, n=len(X), pitch=a)
    lay = layer(X)
    s = (X[lay, 1] - tgt_y) / a
    row["particles"] = dict(zip(("width", "mid"), width(s, red[lay].astype(np.float64))[:2]))
    row["scatter_2a"] = float((((s > 2) & ~red[lay]) | ((s < -2) & red[lay])).sum() / max(1, ((s > 2) | (s < -2)).sum()))
    Xt = gpu.tensor(X.astype(np.float32))
    alb = gpu.tensor(np.where(red[:, None], RED, 1.0).astype(np.float32))
    c = Xt.mean(0)
    lat = Lattice(c, 2.8 * float((Xt - c).norm(dim=1).max()))
    with torch.no_grad():
        f3 = make_field("zb", Xt, a, radius=3.0, offset=0.8)
        p, g, _, _ = lat.discs(f3, 0.46 * a, refine=False)
        k3 = found_disc_albedo(f3, p, alb, alb.mean(0), a).cpu().numpy()
        f1 = make_field("zb", Xt, a, radius=1.0, offset=0.8 / 3.0)
        k1 = found_disc_albedo(f1, p, alb, alb.mean(0), a).cpu().numpy()
    P = p.cpu().numpy().astype(np.float64)
    nn = cKDTree(X).query(P, 1, workers=16)[1]
    sd = (P[:, 1] - tgt_y) / a
    share = lambda col: np.clip((1.0 - col[:, 1]) / (1.0 - RED[1]), 0, 1)       # noqa: E731  (the green channel: 1 white, 0.25 red)
    for k, v in (("discs_k3", share(k3)), ("discs_k1", share(k1)), ("discs_nn", red[nn].astype(np.float64))):
        w_, m_, prof = width(sd, v)
        row[k] = dict(width=w_, mid=m_)
        if k == "discs_k3":
            row["profile_k3"] = prof
    row["discs"] = int(len(P))
    return row


def main():
    out = open(out_path, "a")
    for job in jobs:
        name, path = job[0], job[1]
        z = np.load(path, allow_pickle=True)
        app = json.loads(str(z["appearance"]))
        tgt, ty = np.asarray(z["tgt"], np.float64), float(app["tgt_y"])
        if name.endswith("@tgt"):
            X, red = tgt, is_red(z["tgt_albedo"])
        else:
            X, red = np.asarray(z["frames"][-1], np.float64), is_red(z["src_albedo"])
        row = read(name, X, red, tgt, ty)
        print(json.dumps({k: v for k, v in row.items() if k != "profile_k3"}), flush=True)
        out.write(json.dumps(row) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
