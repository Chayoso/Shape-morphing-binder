"""d155_d157cmp.py -- D155 (the user's question), CPU: the render's end effect against the exact mesh, frozen pipeline
against D155d's ramp, with D157's reader (d157_read.py; the tag's V / W end states read by D157 itself, D155d's A3DL /
A3DP read the same way). Per metric the error e (1 - score for IoU-like, F and normal consistency; the value for
errors and distances) and r = (e_twin - e_arm) / e_twin (> 0: the render arm closer to the mesh), per mesh: the tag at
seed 97 and the median over its seeds 97 / 99 / 101, the ramp at seed 97."""
import glob
import json

import numpy as np

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
rows = {}
for f in glob.glob(f"{G}/d157/read40_*.jsonl") + glob.glob(f"{G}/d155/gate/d157_A3D_*.jsonl"):
    for line in open(f):
        r = json.loads(line)
        rows[(r["mesh"], r["seed"], r["label"])] = r


def g(r, path):
    x = r
    for k in path.split("."):
        if x is None:
            return None
        x = x.get(k) if isinstance(x, dict) else None
    return x


M = (("outline 1 - IoU, 1024 px, held-out views", "sil.zb.cor.1024.held.iou_err", False),
     ("Boundary IoU error (2 %), held-out", "sil.zb.cor.1024.held.biou2pct_err", False),
     ("boundary F error within 0.5 pitch, held-out", "sil.zb.cor.1024.held.bf05a_err", False),
     ("contour Chamfer (pitches), held-out", "sil.zb.cor.1024.held.chamfer_a", False),
     ("3D F-score at 0.5 pitch", "d3.zb.cor.f0.5", True),
     ("3D Chamfer-L1 (pitches)", "d3.zb.cor.chamfer_l1_a", False),
     ("normal consistency", "d3.zb.cor.nc", True))


def err(r, path, score):
    v = g(r, path)
    return None if v is None else (1 - v if score else v)


def rr(arm, twin, path, score):
    a, t = err(arm, path, score), err(twin, path, score)
    return None if a is None or t is None or t == 0 else (t - a) / t


print("r = (e_twin - e_arm) / e_twin, > 0: the render arm closer to the exact mesh")
for name, path, score in M:
    print(name)
    for m in ("bunny", "dragon", "teapot", "armadilo", "homer"):
        tag = [rr(rows.get((m, s, "V")), rows.get((m, s, "W")), path, score) for s in (97, 99, 101)]
        tag = [x for x in tag if x is not None]
        ramp = rr(rows.get((m, 97, "A3DL")), rows.get((m, 97, "A3DP")), path, score)
        t97 = rr(rows.get((m, 97, "V")), rows.get((m, 97, "W")), path, score)
        f = lambda x: "-" if x is None else f"{x:+.3f}"  # noqa: E731
        ea = err(rows.get((m, 97, "A3DL")), path, score)
        ev = err(rows.get((m, 97, "V")), path, score)
        print(f"  {m:9s} tag 97 {f(t97)}, tag median of {len(tag)} seeds {f(float(np.median(tag)) if tag else None)} | "
              f"ramp 97 {f(ramp)} || arm error: tag 97 {'-' if ev is None else f'{ev:.4f}'}, ramp 97 "
              f"{'-' if ea is None else f'{ea:.4f}'}")
