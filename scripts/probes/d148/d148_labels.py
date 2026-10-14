"""D148 design measurement (read-only, CPU): the side labels' error on D135's dragon (FB) before anything is built.

For every gap node of the target (physmorph/mpm/gap_split.GapPlanes on the MPM grid, the loss cell's occupancy) and
every particle in its kernel's reach at a dumped window, the particle's side at that node by three candidate rules:
  deb  the debiased transport image x + (T_tgt - T_self) (what the pipeline's u gate computes at every window start)
  raw  the transport image T_tgt
  pos  the particle's own position x (CPIC's colour: the side of the plane the particle is on)
against its outcome: the side of the same plane its position at the run's last frame lies on (and, where defined, the
piece of the node's box its end position's cell belongs to). Read: agreement (kernel-weighted), isolation (a particle
whose side at its heaviest gap node differs from the majority of its links within 1.5 pitches: a debris candidate),
the window-to-window flips, how many nodes a rule would split (both sides present in the stencil), per gap region.
Usage: d148_labels.py FRAMES_NPZ DUMP_DIR DX NGRID LOSS_RES OUT_JSON
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.environ.get("REPO", "."))
try:
    from physmorph.mpm.gap_split import GapPlanes, bspline1  # noqa: E402
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gap_split import GapPlanes, bspline1  # noqa: E402

frames_npz, dump_dir, dx, ngrid, loss_res, out_json = sys.argv[1:7]
dx, ngrid, loss_res = float(dx), int(ngrid), int(loss_res)
half = dx * ngrid / 2.0
gmin = np.full(3, -half)
ldx = 2.0 * half / loss_res
t0 = time.time()
z = np.load(frames_npz)
tgt = z["tgt"].astype(np.float64)
frames = z["frames"]
end = frames[-1].astype(np.float64)
print(f"loaded {frames.shape} in {time.time() - t0:.1f}s", flush=True)
t0 = time.time()
gp = GapPlanes(tgt, dx, gmin, (ngrid,) * 3, ldx, keep_boxes=True)
print(f"gap nodes {gp.n_gap} (three or more pieces {gp.n_three}) in {time.time() - t0:.1f}s", flush=True)
mask3 = gp.mask.reshape((ngrid,) * 3)
reg, nreg = ndimage.label(mask3, structure=np.ones((3, 3, 3), bool))
reg = reg.reshape(-1)
ii = np.stack(np.nonzero(np.ones((ngrid,) * 3, bool)), -1)
xg_all = gmin + ii * dx
d8 = cKDTree(tgt).query(tgt[::50], k=9)[0][:, 8]
pitch = 0.708 * float(np.median(d8))
link = 1.5 * pitch
rsizes = np.bincount(reg, minlength=nreg + 1)
big = np.argsort(-rsizes[1:])[:8] + 1
info = {"dx": dx, "ldx": ldx, "pitch": pitch, "gap_nodes": int(gp.n_gap), "three_pieces": int(gp.n_three),
        "regions": [{"id": int(r), "nodes": int(rsizes[r]), "centre": xg_all[reg == r].mean(0).round(3).tolist()}
                    for r in big], "windows": []}
print(json.dumps({k: info[k] for k in ("dx", "ldx", "pitch", "gap_nodes", "three_pieces", "regions")}), flush=True)


def pairs(x):
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
                gg = g[sel]
                r = (xg_all[gg] - x[p]) / dx
                P.append(p)
                G.append(gg)
                W.append(bspline1(r[:, 0]) * bspline1(r[:, 1]) * bspline1(r[:, 2]))
    return np.concatenate(P), np.concatenate(G), np.concatenate(W)


rng = np.random.default_rng(0)
tree_t = cKDTree(tgt)
prev = None
files = sorted(glob.glob(os.path.join(dump_dir, "d137_*.npz")))
for f in files:
    d = np.load(f)
    w = int(d["window"])
    x = d["x"].astype(np.float64)
    if "T_tgt" not in d.files:
        continue
    T, S = d["T_tgt"].astype(np.float64), d["T_self"].astype(np.float64)
    q = {"deb": x + (T - S), "raw": T, "pos": x}
    P, G, W = pairs(x)
    rec = {"window": w, "frame_end": int(d["frame_end"]), "t": w / 6.0, "pairs": int(len(P)),
           "particles": int(len(np.unique(P))), "bary_grid": str(d["bary_grid"]) if "bary_grid" in d.files else ""}
    s_out = gp.side(end[P], G)
    # the outcome by the pieces, on a random sample of the pairs (a Python loop)
    smp = rng.choice(len(P), size=min(len(P), 150000), replace=False) if len(P) else np.zeros(0, int)

    _, je = tree_t.query(end[P[smp]])
    oc = gp.side_comp(tgt[je], G[smp])
    for name, qq in q.items():
        s = gp.side(qq[P], G)
        agree = float((W * (s == s_out)).sum() / max(W.sum(), 1e-30))
        # nodes this rule splits: both sides present among the node's particles
        both = np.zeros(len(gp.mask), np.int8)
        np.bitwise_or.at(both, G, np.where(s == 1, 2, 1).astype(np.int8))
        active = both == 3
        act_pairs = active[G]
        agree_act = float((W[act_pairs] * (s[act_pairs] == s_out[act_pairs])).sum() / max(W[act_pairs].sum(), 1e-30))
        # the pieces' outcome where defined
        _, jq = tree_t.query(qq[P[smp]])
        lc = gp.side_comp(tgt[jq], G[smp])
        okc = (oc >= 0) & (lc >= 0)
        agree_comp = float((lc[okc] == oc[okc]).mean()) if okc.any() else float("nan")
        plane_vs_comp = float((s[smp][lc >= 0] == lc[lc >= 0]).mean()) if (lc >= 0).any() else float("nan")
        # isolation: each particle's side at its heaviest active gap node against its links' sides at that node
        Pa, Ga, Wa = P[act_pairs], G[act_pairs], W[act_pairs]
        iso = float("nan")
        n_iso = 0
        if len(Pa):
            o = np.lexsort((-Wa, Pa))
            first = np.ones(len(o), bool)
            first[1:] = Pa[o][1:] != Pa[o][:-1]
            pp, gstar = Pa[o][first], Ga[o][first]
            sp = gp.side(qq[pp], gstar)
            tree = cKDTree(x)
            nb = tree.query_ball_point(x[pp], r=link)
            lens = np.array([len(a) - 1 for a in nb])
            flat = np.concatenate([np.asarray(a, np.int64) for a in nb])
            owner = np.repeat(np.arange(len(pp)), lens + 1)
            notself = flat != pp[owner]
            flat, owner = flat[notself], owner[notself]
            sn = gp.side(qq[flat], gstar[owner])
            diff = np.bincount(owner, weights=(sn != sp[owner]).astype(float), minlength=len(pp))
            iso_mask = (lens > 0) & (diff > 0.5 * lens)
            n_iso = int(iso_mask.sum())
            iso = float(iso_mask.mean())
        rec[name] = {"agree": round(agree, 4), "active_nodes": int(active.sum()), "agree_active": round(agree_act, 4),
                     "agree_pieces": round(agree_comp, 4), "plane_vs_pieces": round(plane_vs_comp, 4),
                     "isolated": n_iso, "isolated_frac": round(iso, 5),
                     "side1_frac": round(float((W * (s == 1)).sum() / max(W.sum(), 1e-30)), 4)}
        if name == "deb":
            key = P * len(gp.mask) + G
            if prev is not None:
                common, ia, ib = np.intersect1d(prev[0], key, return_indices=True)
                rec["deb"]["flip_vs_prev"] = round(float((prev[1][ia] != s[ib]).mean()), 4) if len(common) else None
            prev = (key, s)
            # per region, the active nodes and the agreement
            rr = reg[G]
            rec["deb"]["regions"] = {int(r): {"active": int((active & (reg == r)).sum()),
                                             "agree": round(float((W[rr == r] * (s[rr == r] == s_out[rr == r])).sum()
                                                                  / max(W[rr == r].sum(), 1e-30)), 4)}
                                     for r in big}
    info["windows"].append(rec)
    print(json.dumps(rec), flush=True)
json.dump(info, open(out_json, "w"), indent=1)
print("done", flush=True)
