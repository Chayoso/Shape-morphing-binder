"""d160_surfdisp.py OUT_JSONL LABEL=RUN_FRAMES12,TWIN_FRAMES12 [...] -- D160d's reading: how much a run's surface geometry
differs from its twin's at the delivered end, apart from where the material went (measurement only; a card if visible).

Per pair (the delivered frames; the exterior discs of each state, Zhu-Bridson radius 3, offset 0.8 a, the loss's
lattice, d160_read.py's geometry):
  surface   each run disc's signed offset from the twin's surface along the twin's normal at its nearest twin disc,
            in pitches: rms, p90 of |offset|, mean (+ outward)
  particles the outer layer (the particles the run's discs read): each particle's end displacement from its place in
            the twin, split into the part along the run surface's normal there (the nearest run disc's) and the part
            along the surface: rms and p90 of each, in pitches
A twin-against-twin pair (two lambda = 0 runs of one seed, other paints: one colour-blind physics) is the floor."""
import json
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d160")
from d160_read import Ctx, geom  # noqa: E402

out = open(sys.argv[1], "a")
ctxs = {}
for spec in sys.argv[2:]:
    label, paths = spec.split("=", 1)
    pr, pt = paths.split(",")
    zr, zt = np.load(pr, allow_pickle=True), np.load(pt, allow_pickle=True)
    tgt = np.asarray(zr["tgt"], np.float32)
    key = (len(tgt), float(tgt[:, 1].sum()))
    ctx = ctxs.setdefault(key, Ctx(tgt))
    a = ctx.a
    xr, xt = np.asarray(zr["frames"][-1], np.float32), np.asarray(zt["frames"][-1], np.float32)
    p_r, n_r, idx_r, _ = geom(xr, ctx)
    p_t, n_t, _, _ = geom(xt, ctx)
    p_r, n_r = p_r.cpu().numpy(), n_r.cpu().numpy()
    p_t, n_t = p_t.cpu().numpy(), n_t.cpu().numpy()
    _, j = cKDTree(p_t).query(p_r)
    off = ((p_r - p_t[j]) * n_t[j]).sum(1) / a
    outer = np.unique(idx_r[idx_r >= 0].cpu().numpy())
    _, k = cKDTree(p_r).query(xr[outer])
    d = (xr[outer] - xt[outer]) / a
    dn = (d * n_r[k]).sum(1)
    dt = np.linalg.norm(d - dn[:, None] * n_r[k], axis=1)
    row = dict(label=label, run=pr.split("/")[-1], twin=pt.split("/")[-1], pitch=a,
               surf_rms=float(np.sqrt((off ** 2).mean())), surf_p90=float(np.quantile(np.abs(off), .9)),
               surf_mean=float(off.mean()), outer=int(len(outer)),
               normal_rms=float(np.sqrt((dn ** 2).mean())), normal_p90=float(np.quantile(np.abs(dn), .9)),
               tangential_rms=float(np.sqrt((dt ** 2).mean())), tangential_p90=float(np.quantile(dt, .9)))
    out.write(json.dumps(row) + "\n")
    out.flush()
    print(json.dumps({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in row.items()}), flush=True)
    del p_r, p_t
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
