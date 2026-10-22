"""d162_arrange_sum.py OUT_MD ARRANGE_JSONL -- D162: the arrangement reader's rows (d162_arrange.py) as one table per run
and the pre-registered reading (docs, D162 amendment 3).

Per kept frame: the time, the window and its weight (lambda; on F / FC the floor-referenced one), the render term on the
state (a) and the weight's excess on it e_a = 1 - L_floor / a (what the weight reads), the render term of the state's
uniform resamples (b, its standard error) and the shape's excess on it e_b = 1 - b_floor / b (b_floor: the floor sample
through the same extraction and resampling, so the extraction's own loss cancels), and the shares of the render term's
descent since the first kept frame: shape (b_0 - b) / (a_0 - a) and arrangement 1 - that.

Reading: at the first kept frame whose e_a <= 0.05 (the weight at most 5 % of its first value), the weight has stopped
pushing; if there e_b exceeds 0.05 by more than two standard errors (b's and b_floor's, through e_b), the weight
vanished with shape work left: arrangement ate the floor. If e_b is within that, the weight vanished with the shape at
its floor. If no kept frame has e_a <= 0.05, the weight never vanished."""
import json
import sys
from collections import defaultdict

import numpy as np

out, src = sys.argv[1], sys.argv[2]
rows = defaultdict(list)
for line in open(src):
    r = json.loads(line)
    rows[r["run"]].append(r)
L = ["# D162: arrangement against shape in the render term's descent", "",
     "a = the render term on the state (the weight reads it); b = the mean over eight uniform resamples of the state's "
     "solid (shape kept, arrangement dropped). e_a = 1 - L_floor / a; e_b = 1 - b_floor / b. Shares: of the descent of a "
     "since the first kept frame.", ""]
for run, rs in rows.items():
    fs = {r["state"]: r for r in rs if r["state"] != "frame"}
    fr = sorted((r for r in rs if r["state"] == "frame"), key=lambda r: r["raw"])
    fl, un = fs.get("floor_sample"), fs.get("uniform_true_mesh")
    if fl is None or not fr:
        continue
    Lf = fl.get("L_floor_run") or fl["a"]
    bf, bf_se = fl["b"], fl["b_se"]
    L += [f"## {run}", "",
          f"L_floor {Lf:.4e} (its own read here {fl['a']:.4e}); the floor sample through the extraction b_floor "
          f"{bf:.4e} ± {bf_se:.1e}; uniform samples of the true mesh {un['a']:.4e} ± {un['a_se']:.1e} "
          f"(the in-picture floor sits {100 * (1 - fl['a'] / un['a']):.0f} % below the uniform one)", "",
          "| frame | t (s) | window | lambda | a | e_a | b ± se | e_b | shape share | arrangement share | solid closed |",
          "|" + "---|" * 11]
    a0, b0 = fr[0]["a"], fr[0]["b"]
    verdict = "the weight never fell to 5 % of its first value at a kept frame"
    for r in fr:
        w = r.get("window") or {}
        a, b, se = r["a"], r["b"], r["b_se"]
        ea = max(0.0, 1.0 - Lf / a)
        eb = 1.0 - bf / b
        se_eb = (bf / b) * float(np.hypot(se / b, bf_se / bf))
        da, db = a0 - a, b0 - b
        sh = db / da if da > 0 else None
        lam = w.get("lambda")
        L.append(f"| {r['raw']} | {r['t']:.2f} | {w.get('animation')} | {'-' if lam is None else f'{lam:.3g}'} | "
                 f"{a:.4e} | {ea:.3f} | {b:.4e} ± {se:.1e} | {eb:.3f} | {'-' if sh is None else f'{sh:.2f}'} | "
                 f"{'-' if sh is None else f'{1 - sh:.2f}'} | {r.get('watertight')} |")
        if verdict.startswith("the weight never") and ea <= 0.05:
            if eb - 0.05 > 2 * se_eb:
                verdict = (f"ARRANGEMENT ATE THE FLOOR: at frame {r['raw']} (t {r['t']:.2f} s) e_a = {ea:.3f} while "
                           f"e_b = {eb:.3f} ± {se_eb:.3f}")
            else:
                verdict = (f"the weight vanished with the shape at its floor: at frame {r['raw']} (t {r['t']:.2f} s) "
                           f"e_a = {ea:.3f}, e_b = {eb:.3f} ± {se_eb:.3f}")
    L += ["", f"Reading: {verdict}", ""]
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
