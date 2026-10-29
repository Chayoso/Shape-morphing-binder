"""d155b_relief.py MESH [MESH ...] -- D155b: the end state's sub-pitch surface itself (CPU): on the outer layer, the
relief (each layer particle's normal offset from the mean of its 24 nearest, the quantity the layer relaxation sets;
rms, pitches), the surface sampling (nearest-neighbour distance among layer particles: median and coefficient of
variation), against the independent sample's own; and the relief at the ramp's end / the tag's arrival. Seed 97, tag and
A3G, render arm and twin."""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
O = f"{G}/d155/gate"


def surf(x, pitch):
    tree = cKDTree(x)
    sp = float(np.median(tree.query(x[::15], 9)[0][:, 8]))
    _, nb = tree.query(x, 33, workers=2)
    off = x - x[nb[:, 1:]].mean(1)
    lay = np.linalg.norm(off, axis=1) >= 0.5 * sp
    c24 = x[nb[:, 1:25]].mean(1)
    n = x - c24
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    # relief: offset along n from the plane through the mean of the 24 nearest LAYER particles
    li = np.flatnonzero(lay)
    lt = cKDTree(x[li])
    _, j = lt.query(x[li], 25, workers=2)
    cl = x[li][j[:, 1:]].mean(1)
    rel = ((x[li] - cl) * n[li]).sum(1) / pitch
    d1 = lt.query(x[li], 2, workers=2)[0][:, 1] / pitch
    return dict(n=int(len(li)), relief_rms=float(np.sqrt((rel ** 2).mean())), relief_mean=float(rel.mean()),
                nn_med=float(np.median(d1)), nn_cv=float(d1.std() / d1.mean()))


def arrival(name):
    a = next(iter(json.load(open(f"{O}/{name}.json"))["arms"].values()))
    acc = [h for h in a["history"] if h.get("frame_end") and not h.get("null_commit")]
    t = next((h["frame_end"] / 240 for h in acc if (h.get("u_gate") or 0) >= 0.5), 0.0)
    if any("morph_s" in h for h in acc):
        t = max(t, next(h["frame_end"] / 240 for h in acc if (h.get("morph_s") or 0) >= 1.0))
    return t


for mesh in sys.argv[1:]:
    ind = np.asarray(np.load(f"{G}/d92/ind_{mesh}.npz")["tgt"], np.float64)
    pitch = 0.708 * float(np.median(cKDTree(ind).query(ind[::20], 9)[0][:, 8]))
    s = surf(ind, pitch)
    print(f"{mesh} IND: relief rms {s['relief_rms']:.3f} mean {s['relief_mean']:+.3f}, layer nn median "
          f"{s['nn_med']:.3f} cv {s['nn_cv']:.3f}", flush=True)
    for arm in ("V97", "V99", "V101", "A3GL97", "W97", "W99", "W101", "A3GP97"):
        z = np.load(f"{O}/{mesh}40k_{arm}_frames12.npz")
        raws = np.asarray(z["raws"])
        e = surf(np.asarray(z["frames"][-1], np.float64), pitch)
        ta = arrival(f"{mesh}40k_{arm}")
        i = int(np.argmin(np.abs(raws / 240.0 - ta)))
        a = surf(np.asarray(z["frames"][i], np.float64), pitch)
        print(f"  {arm:7s} end: relief rms {e['relief_rms']:.3f} mean {e['relief_mean']:+.3f}, nn median "
              f"{e['nn_med']:.3f} cv {e['nn_cv']:.3f} | at arrival {ta:.2f} s: relief rms {a['relief_rms']:.3f}, "
              f"nn cv {a['nn_cv']:.3f}", flush=True)
