"""d157_gate_box.py MESH SEED -- D157's 300k gate on D154's own box: the pooled 1 - IoU (1024 px, 18 training views,
raw referee) of the mesh-seed's own, L and P states with the orthographic box D154 used at 300k (1.25 x the largest
|x| of D90's independent sample), against D154's exact_1024. Isolates the box from the reader."""
import glob
import json
import sys

import numpy as np

import d157_read as R

mesh, seed = sys.argv[1], int(sys.argv[2])
G = R.G
ind = np.asarray(np.load(f"{G}/d90/{mesh}_ind300k.npz")["tgt"], np.float32)
ext154 = float(np.abs(ind).max()) * 1.25
theirs = {}
for f in glob.glob(f"{G}/d154/read/end_*.jsonl"):
    for ln in open(f):
        r = json.loads(ln)
        if r["mesh"] == mesh and r["seed"] == seed and r["raw"] in ("end", "tgt") and "exact_1024" in r:
            theirs.setdefault(r["label"], r["exact_1024"]["one_minus_iou"])
m = np.load(f"{R.MESHDIR['300k']}/{mesh}_{seed}.npz")
t, _ = R.close_holes(np.asarray(m["V"], np.float64), np.asarray(m["F"], np.int64))
ref = R.Surf(np.asarray(t.vertices), np.asarray(t.faces))
tr = [R.ortho_rays(v, ext154, 1024) for v in R.TRAIN]
T = [ref.occluded(r).reshape(1024, 1024) for r in tr]
for lab in ("own", "L", "P"):
    npz = f"{G}/d145t_b1/{mesh}300k_{'L' if lab == 'own' else lab}{seed}_end1.npz"
    x = R.state_x(npz, "tgt" if lab == "own" else "end")
    a = .708 * R.median_kth_spacing(np.asarray(np.load(npz)["tgt"], np.float32), 8, 20000)
    a154 = .708 * R.median_kth_spacing(ind, 8, 20000)              # D154's pitch at 300k: the independent sample's
    for name, pitch in (("run pitch", a), ("D154 pitch", a154)):
        st, _ = R.field_mesh("zb", x, pitch)
        S = [st.occluded(r).reshape(1024, 1024) for r in tr]
        inter = sum(int((s & q).sum()) for s, q in zip(S, T))
        uni = sum(int((s | q).sum()) for s, q in zip(S, T))
        v = 1 - inter / uni
        print(f"{mesh} {seed} {lab} {name}: box {ext154:.5f} pitch {pitch:.6f} -> 1 - IoU {v:.7f}; D154 "
              f"{theirs.get(lab, float('nan')):.7f}; diff {v - theirs.get(lab, float('nan')):+.2e}", flush=True)
