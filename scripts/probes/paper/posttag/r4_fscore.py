"""r4_fscore.py OUT_JSONL MESH:METHOD:LABEL=ARCHIVE [...] -- D145 reading 4 (3): the F-score of every method's end frame
against the independent 300k sample, sample against sample (CPU; scipy).

For tau = 1 and 2 pitches (pitch = 0.708 x the independent sample's median 8th-neighbour distance, the harness's):
precision = share of the state's points within tau of the independent sample, recall = share of the independent
sample's points within tau of the state, F = 2 P R / (P + R); the same for the archive's own target sample (a second
sample of the mesh: the floor). The last frame that is not NaN. Against the exact mesh this is D157's, not repeated."""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

B = "/data/relcfd/chayo/physmorph_v2"
TAUS = (1.0, 2.0)
ref = {}


def fs(x, tree_i, tree_x, I, a):
    dx = tree_i.query(x, 1, workers=6)[0]
    di = tree_x.query(I, 1, workers=6)[0]
    out = {}
    for t in TAUS:
        p, r = float((dx <= t * a).mean()), float((di <= t * a).mean())
        out[f"F{t:g}"] = 2 * p * r / (p + r) if p + r else 0.0
        out[f"P{t:g}"], out[f"R{t:g}"] = p, r
    return out


with open(sys.argv[1], "a") as fh:
    for spec in sys.argv[2:]:
        key, path = spec.split("=", 1)
        mesh, method, label = key.split(":")
        if mesh not in ref:
            I = np.asarray(np.load(f"{B}/output/gpu/d90/{mesh}_ind300k.npz")["tgt"], np.float64)
            ti = cKDTree(I)
            a = 0.708 * float(np.median(ti.query(I[::20], 9, workers=6)[0][:, 8]))
            ref[mesh] = (I, ti, a)
        I, ti, a = ref[mesh]
        z = np.load(path, allow_pickle=True)
        F = z["frames"]
        k = len(F) - 1
        while k > 0 and not np.isfinite(np.asarray(F[k][:16])).all():
            k -= 1
        x = np.asarray(F[k], np.float64)
        own = np.asarray(z["tgt"], np.float64)
        row = dict(mesh=mesh, method=method, label=label, pitch=a, **fs(x, ti, cKDTree(x), I, a),
                   floor=fs(own, ti, cKDTree(own), I, a))
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print(mesh, method, label, {k2: round(v, 4) for k2, v in row.items() if k2.startswith("F") and k2 != "floor"},
              flush=True)
