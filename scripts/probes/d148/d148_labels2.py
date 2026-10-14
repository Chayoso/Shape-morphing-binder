"""D148 design measurement, part 2 (read-only, CPU): the definition as proposed (gap nodes outside the target only, the
side by the debiased transport image), on FB's dumped windows: how many nodes split, how many particles they reach, the
agreement with the run's end, and two isolation counts per window:
  link    a particle whose side at its heaviest split node differs from every one of its links (1.5 pitches)
  kernel  a particle that is the only one of its side among the particles of its heaviest split node's stencil
(the second is the debris mode: its field at that node is itself). Also the bunny's gap nodes (SB's target).
Usage: d148_labels2.py FB_FRAMES FB_DUMP_DIR SB_FRAMES OUT_JSON
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap_split import GapPlanes, bspline1  # noqa: E402

fb_frames, dump_dir, sb_frames, out_json = sys.argv[1:5]
out = {}
for name, path, dx, ngrid, lres in (("bunny", sb_frames, 0.30542453333862823, 36, 71),
                                    ("dragon", fb_frames, 0.2948594285057975, 44, 87)):
    z = np.load(path)
    tgt = z["tgt"].astype(np.float64)
    half = dx * ngrid / 2.0
    gp = GapPlanes(tgt, dx, np.full(3, -half), (ngrid,) * 3, 2.0 * half / lres)
    out[name] = {"gap_nodes": int(gp.n_gap), "three_pieces": int(gp.n_three), "inside_candidates": int(gp.n_inside)}
    print(name, out[name], flush=True)
    if name != "dragon":
        continue
    end = z["frames"][-1].astype(np.float64)
    gmin = np.full(3, -half)
    ii = np.stack(np.nonzero(np.ones((ngrid,) * 3, bool)), -1)
    xg_all = gmin + ii * dx
    pitch = 0.708 * float(np.median(cKDTree(tgt).query(tgt[::50], k=9)[0][:, 8]))
    rows = []
    for w in (1, 3, 5, 7, 9, 12, 15, 18, 20, 22, 24, 26, 28, 30):
        d = np.load(os.path.join(dump_dir, f"d137_{w:04d}.npz"))
        x = d["x"].astype(np.float64)
        q = x + (d["T_tgt"].astype(np.float64) - d["T_self"].astype(np.float64))
        b = np.floor((x - gmin) / dx).astype(np.int64) - 1
        P, G, W = [], [], []
        for oi in range(4):
            for oj in range(4):
                for ok in range(4):
                    nid = b + np.array([oi, oj, ok])
                    inb = ((nid >= 0) & (nid < ngrid)).all(1)
                    g = np.where(inb, (nid[:, 0] * ngrid + nid[:, 1]) * ngrid + nid[:, 2], 0)
                    sel = inb & gp.mask[g]
                    p = np.nonzero(sel)[0]
                    r = (xg_all[g[sel]] - x[p]) / dx
                    P.append(p)
                    G.append(g[sel])
                    W.append(bspline1(r[:, 0]) * bspline1(r[:, 1]) * bspline1(r[:, 2]))
        P, G, W = np.concatenate(P), np.concatenate(G), np.concatenate(W)
        s = gp.side(q[P], G)
        so = gp.side(end[P], G)
        both = np.zeros(len(gp.mask), np.int8)
        np.bitwise_or.at(both, G, np.where(s == 1, 2, 1).astype(np.int8))
        act = (both == 3)[G]
        Pa, Ga, Wa, sa = P[act], G[act], W[act], s[act]
        o = np.lexsort((-Wa, Pa))
        first = np.ones(len(o), bool)
        first[1:] = Pa[o][1:] != Pa[o][:-1]
        pp, gstar, sp = Pa[o][first], Ga[o][first], sa[o][first]
        # kernel isolation: the particles of g*'s stencil on p's side, other than p
        cnt = {}
        key = Ga * 2 + sa
        uk, uc = np.unique(key, return_counts=True)
        lut = dict(zip(uk.tolist(), uc.tolist()))
        alone = np.array([lut.get(int(g) * 2 + int(v), 0) <= 1 for g, v in zip(gstar, sp)])
        # link isolation: every link on the other side (at g*)
        nb = cKDTree(x).query_ball_point(x[pp], r=1.5 * pitch)
        lens = np.array([len(a) - 1 for a in nb])
        flat = np.concatenate([np.asarray(a, np.int64) for a in nb])
        owner = np.repeat(np.arange(len(pp)), lens + 1)
        keep = flat != pp[owner]
        flat, owner = flat[keep], owner[keep]
        sn = gp.side(q[flat], gstar[owner])
        diff = np.bincount(owner, weights=(sn != sp[owner]).astype(float), minlength=len(pp))
        link_all = (lens > 0) & (diff >= lens)
        rec = {"window": w, "t": round(w / 6.0, 3), "active_nodes": int((both == 3).sum()),
               "particles_at_active": int(len(pp)), "agree": round(float((Wa * (sa == so[act])).sum() / Wa.sum()), 4),
               "side1_frac": round(float((Wa * (sa == 1)).sum() / Wa.sum()), 4),
               "iso_link_all": int(link_all.sum()), "iso_kernel": int(alone.sum()),
               "iso_both": int((alone & link_all).sum())}
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    out["dragon"]["windows"] = rows
json.dump(out, open(out_json, "w"), indent=1)
print("done", flush=True)
