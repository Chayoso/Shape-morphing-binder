"""d162c_motion.py OUT_JSONL [RUN ...] -- D162c (A): the realized motion of the outer particles over 48 frames (about a
window) from the kept positions, split as d162c_lib.decompose, per state, before and after arrival; the runs of RUNS below
(or the named ones). One JSON line per state. CPU."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d162c_lib as C  # noqa: E402

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
RUNS = {}
for s in (97, 99):
    for sc, base in (("coin", "coin40k"), ("bunny", "bunny40k"), ("band", "bunny40k_band")):
        RUNS[f"F_{sc}_{s}"] = ("fail F", G / "d162/runs40" / f"{base}_F_{s}")
        RUNS[f"S_{sc}_{s}"] = ("fail S", G / "d162b/runs40" / f"{base}_S_{s}")
    for sc in ("coin", "bunny"):
        RUNS[f"D161c_{sc}_{s}"] = ("success D161c", G / "d161/runs40" / f"{sc}40k_full_{s}_lh")
        RUNS[f"twin_{sc}_{s}"] = ("twin D161", G / "d161/runs40" / f"{sc}40k_twin_{s}")
    RUNS[f"D160d_band_{s}"] = ("success D160d", G / "d160/runs40" / f"bunny40k_band_D{s}")
    RUNS[f"P_band_{s}"] = ("twin D160 P", G / "d160/runs40" / f"bunny40k_band_P{s}")
RUNS["D163H_plaque_97"] = ("success D163 H", G / "d163/runs300" / "plq300k_H_97")
RUNS["T_plaque_97"] = ("twin D163 T", G / "d163/runs300" / "plq300k_T_97")
STEP = 48


def arrival_frame(prefix):
    js = json.load(open(f"{prefix}.json"))
    com = [h for h in list(js["arms"].values())[0]["history"] if h.get("frame_end") and not h.get("null_commit")]
    te = np.array([h.get("transport_energy") or np.nan for h in com])
    return com[int(np.argmax(te < 2 * te[-1]))]["frame_end"]


def main(out, names):
    for name in names:
        group, prefix = RUNS[name]
        t0 = time.time()
        z = np.load(f"{prefix}_frames12.npz", allow_pickle=True)
        raws = np.asarray(z["raws"])
        a = C.pitch_of(z["tgt"])
        arr = arrival_frame(prefix)
        pos = {int(r): k for k, r in enumerate(raws)}
        pairs = [(k, pos[int(r) + STEP]) for k, r in enumerate(raws) if int(r) + STEP in pos]
        cap = 20 if len(z["tgt"]) > 100000 else 30
        pick = [pairs[i] for i in sorted(set(np.linspace(0, len(pairs) - 1, min(cap, len(pairs))).astype(int)))]
        for k1, k2 in pick:
            x1 = np.asarray(z["frames"][k1], np.float64)
            x2 = np.asarray(z["frames"][k2], np.float64)
            idx, n = C.outer_layer(x1, a)
            B, deg = C.band_operator(x1[idx], a)
            d = C.decompose((x2 - x1)[idx] / a, n, B)
            row = dict(run=name, group=group, raw=int(raws[k1]), t=float(raws[k1]) / 240.0,
                       phase="after" if raws[k1] >= arr else "before", n_outer=int(len(idx)), nbrs=deg, **d)
            with open(out, "a") as fo:
                fo.write(json.dumps(row) + "\n")
        print(f"[d162c A] {name}: {len(pick)} states, arrival frame {arr}, pitch {a:.4f}, {time.time() - t0:.0f} s",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or list(RUNS))
