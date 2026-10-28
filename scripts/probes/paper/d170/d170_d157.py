"""d170_d157.py OUT_MD READ_JSONL... -- D170: D157's decision set (14 measures: held-out silhouettes at 1024 px against the
corrected referee, 3D on the exterior, held-out studio images) for Rx on the 40k gallery against the tag's V (and d167's
R at seed 97) and the twin W, per mesh-seed. The V / W rows come from D157's own reads (output/gpu/d157/read40_*.jsonl);
the Rx / R rows from READ_JSONL (tmp/d157/d157_read.py on the D170 runs). Error form e (1 - score for F / NC / SSIM);
r = (e_V - e_Rx) / e_V (> 0: Rx better). Writes OUT_MD (a table of <= 10 rows and a verdict line)."""
import glob
import json
import sys

import numpy as np

DECISION = ["sil/zb/cor/1024/held/biou2pct_err", "sil/zb/cor/1024/held/bfdavis_err", "sil/zb/cor/1024/held/chamfer_a",
            "sil/zb/cor/1024/held/hd95_a", "d3/zb/cor/f0.5", "d3/zb/cor/f1", "d3/zb/cor/nc", "d3/zb/cor/chamfer_l1_a",
            "img/cor/shaded/held/mse", "img/cor/shaded/held/ssim", "img/cor/shaded/held/lpips",
            "img/cor/normal/held/mse", "img/cor/normal/held/ssim", "img/cor/normal/held/lpips"]
SCORE = ("f0.5", "f1", "nc", "ssim")
MESHES = ("bunny", "dragon", "armadilo", "bob", "homer", "teapot")


def get(row, path):
    x = row
    for k in path.split("/"):
        if not isinstance(x, dict) or k not in x:
            return None
        x = x[k]
    if not isinstance(x, (int, float)):
        return None
    return 1.0 - x if path.split("/")[-1] in SCORE else float(x)


def load(paths):
    out = {}
    for p in paths:
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                if "label" in r and "sil" in r:
                    out[(r["mesh"], int(r["seed"]), r["label"])] = r
    return out


out_md = sys.argv[1]
base = load(sorted(glob.glob("/data/relcfd/chayo/physmorph_v2/output/gpu/d157/read40_*.jsonl")))
mine = load(sys.argv[2:])
rows = ["| mesh | seed | Rx worse than V by > 5 % (of 14) | Rx better by > 5 % | median r vs V | W (twin) worse than V | R vs V (seed 97: run spread) |",
        "|---|---|---|---|---|---|---|"]
allr, worse_tot, n_pairs = [], 0, 0
for m in MESHES:
    for s in (97, 99):
        rx, v, w, rr = mine.get((m, s, "Rx")), base.get((m, s, "V")), base.get((m, s, "W")), mine.get((m, s, "R"))
        if rx is None or v is None:
            continue
        rs, wo, be, ww, rv = [], 0, 0, 0, 0
        for k in DECISION:
            a, b = get(rx, k), get(v, k)
            if a is None or b is None or b <= 0:
                continue
            r = (b - a) / b
            rs.append(r)
            wo += r < -0.05
            be += r > 0.05
            if w is not None and get(w, k) is not None:
                ww += (b - get(w, k)) / b < -0.05
            if rr is not None and get(rr, k) is not None:
                rv += (b - get(rr, k)) / b < -0.05
        allr += rs
        worse_tot += wo
        n_pairs += 1
        rows.append(f"| {m} | {s} | {wo} | {be} | {np.median(rs):+.3f} | {ww if w is not None else '-'} | {rv if rr is not None else '-'} |")
verdict = (f"Rx against the tag (V) on D157's decision set: median r {np.median(allr):+.3f} over {len(allr)} mesh-seed-measures "
           f"({n_pairs} mesh-seeds); worse by > 5 % on {worse_tot} of {14 * n_pairs}." if allr else "no pairs read")
L = ["# D170 gallery: D157's decision set, Rx vs the tag (V) and the twin (W), 40k", "", *rows[:12], "",
     "Measures: held-out silhouettes at 1024 px against the corrected referee (BIoU 2 %, boundary F DAVIS, contour Chamfer, HD95), "
     "the exterior in 3D (F@0.5, F@1, normal consistency, Chamfer-L1), the held-out studio images (shaded, normal: MSE, SSIM, "
     "LPIPS). r = (e_V - e_Rx) / e_V; the W and R columns count the measures on which the twin / d167's rule run is worse "
     "than V by > 5 % (the size of a single-run swing).", "", "VERDICT: " + verdict]
open(out_md, "w").write("\n".join(L) + "\n")
print(verdict)
