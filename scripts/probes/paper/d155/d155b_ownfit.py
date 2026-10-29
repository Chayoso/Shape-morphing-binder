"""d155b_ownfit.py MESH [MESH ...] -- D155b: does the switched run end closer to its OWN target sample and farther from an
independent one than the tag (a copy of the own sample's noise)? For the end frame of each run: the outer layer's
signed normal offset rms (pitches) against the own sample's outer layer (the run's 'tgt') and against the independent
sample's, and the same at the run's arrival (the tag: u's gate over half the layer; A3G: the ramp's end) and at +1 /
+2 / +3 / +5 s after it. The tag at seeds 97 / 99 / 101 (the own sample is the same for every seed? no: a run's target
sample is drawn with seed + 1, so the seeds differ)."""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
O = f"{G}/d155/gate"


def layer_normals(x, k_lay=33, k_n=25):
    tree = cKDTree(x)
    sp = float(np.median(tree.query(x[::15], 9)[0][:, 8]))
    _, nb = tree.query(x, k_lay, workers=2)
    off = x - x[nb[:, 1:]].mean(1)
    lay = np.linalg.norm(off, axis=1) >= 0.5 * sp
    off_n = x - x[nb[:, 1:k_n]].mean(1)
    return lay, off_n / np.maximum(np.linalg.norm(off_n, axis=1, keepdims=True), 1e-12)


def ref(y):
    lay, n = layer_normals(y)
    return y[lay], n[lay]


def rms_off(x, R, pitch):
    Y, NY = R
    lay, _ = layer_normals(x)
    xl = x[lay]
    _, j = cKDTree(xl).query(Y, 8, workers=2)
    o = ((xl[j] - Y[:, None, :]) * NY[:, None, :]).sum(2).mean(1) / pitch
    return float(np.sqrt((o ** 2).mean()))


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
    RI = ref(ind)
    print(f"== {mesh} (pitch {pitch:.4f})", flush=True)
    for arm in ("V97", "V99", "V101", "A3L97", "A3GL97", "W97", "W99", "W101", "A3P97", "A3GP97"):
        z = np.load(f"{O}/{mesh}40k_{arm}_frames12.npz")
        own = np.asarray(z["tgt"], np.float64)
        RO = ref(own)
        raws = np.asarray(z["raws"])
        ta = arrival(f"{mesh}40k_{arm}")
        row = []
        for dt in (0, 1, 2, 3, 5, None):
            i = len(raws) - 1 if dt is None else int(np.argmin(np.abs(raws / 240.0 - (ta + dt))))
            x = np.asarray(z["frames"][i], np.float64)
            row.append((("end" if dt is None else f"+{dt}s"), rms_off(x, RO, pitch), rms_off(x, RI, pitch)))
        print(f"  {arm:7s} own / independent: " + "  ".join(f"{k} {a:.3f} / {b:.3f}" for k, a, b in row)
              + f"   (own sample against the independent {rms_off(own, RI, pitch):.3f})", flush=True)
