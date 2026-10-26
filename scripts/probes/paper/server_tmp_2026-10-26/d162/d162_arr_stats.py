"""d162_arr_stats.py ARRANGE_JSONL -- D162: the arrangement reader's rows in a few numbers per run (after 2 s: how much a
and b swing from frame to frame, how often the weight's excess e_a is zero, the shape excess e_b) and the end's
full-vs-twin gap split into shape (b) and arrangement (a - b)."""
import json
import sys

import numpy as np

rows = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    rows.setdefault(r["run"], []).append(r)


def get(run):
    rs = rows[run]
    fl = next(r for r in rs if r["state"] == "floor_sample")
    fr = sorted((r for r in rs if r["state"] == "frame"), key=lambda r: r["raw"])
    return fl, fr


for run in rows:
    fl, fr = get(run)
    lf, bf = fl["a"], fl["b"]
    late = [r for r in fr if r["t"] >= 2.0]
    a = np.array([r["a"] for r in late])
    b = np.array([r["b"] for r in late])
    ea, eb = np.maximum(0.0, 1.0 - lf / a), 1.0 - bf / b
    e = fr[-1]
    print(f"{run}: {len(fr)} frames ({len(late)} after 2 s); a mean {a.mean():.3e} (cv {a.std() / a.mean():.2f}), "
          f"a / L_floor {np.mean(a / lf):.2f}; b mean {b.mean():.3e} (cv {b.std() / b.mean():.2f}); e_a = 0 at "
          f"{np.mean(ea == 0):.0%} of them; e_b mean {eb.mean():.3f}, min {eb.min():.3f}; end t {e['t']:.1f} a {e['a']:.3e} "
          f"b {e['b']:.3e}")
for full, twin in (("coin40k_F_97", "coin40k_TC_97"), ("coin40k_FC_97", "coin40k_TC_97"),
                   ("bunny40k_band_F_97", "bunny40k_band_TC_97")):
    if full not in rows or twin not in rows:
        continue
    A, T = get(full)[1][-1], get(twin)[1][-1]
    ga, gb = T["a"] - A["a"], T["b"] - A["b"]
    print(f"END {full} vs {twin}: gap a {ga:.3e}, gap b {gb:.3e}: shape share {gb / ga:.2f}, arrangement {1 - gb / ga:.2f}")
