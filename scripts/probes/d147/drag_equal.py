"""drag_equal.py RUNS_DIR OLD_A,OLD_B,NEW -- D147, the damping's speed-up: is the block-sum damping with its hand adjoint
(NEW) the atomic one (OLD) within float summation order? The scene is not repeatable for the same input (the transfers'
atomics), so the old code is run twice and NEW is read against OLD_A on the scale of OLD_B against OLD_A: per window
and body the linear and angular momentum, the committed merit, each body's terms, and the positions at each window's
end (spacings); and each run's seconds per window (start, gradient, line search, commit). Prints JSON. CPU."""
import json
import sys
from pathlib import Path

import numpy as np

runs = Path(sys.argv[1])
tags = sys.argv[2].split(",")
J = {t: json.loads((runs / f"{t}.json").read_text()) for t in tags}
Z = {t: np.load(runs / f"{t}.npz") for t in tags}
H = {t: [r for r in J[t]["history"] if "bodies" in r] for t in tags}
n_b = int(Z[tags[0]]["rows"][0][1] - Z[tags[0]]["rows"][0][0])
sp = (4.18879 * J[tags[0]]["scene"]["R"] ** 3 / n_b) ** (1 / 3)
T = int(J[tags[0]]["config"]["T"])


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def compare(t, ref):
    out = {"P": 0.0, "L": 0.0, "merit": 0.0, "transport": 0.0, "d_sil": 0.0, "x_sp": 0.0}
    for ra, rb in zip(H[t], H[ref]):
        for b in range(len(ra["bodies"])):
            ba, bb = ra["bodies"][b], rb["bodies"][b]
            out["P"] = max(out["P"], rel(ba["P"], bb["P"]))
            out["L"] = max(out["L"], rel(ba["L"], bb["L"]))
            out["transport"] = max(out["transport"], rel(ba["transport"], bb["transport"]))
            out["d_sil"] = max(out["d_sil"], rel(ba["d_sil"], bb["d_sil"]))
        if ra.get("loss") is not None and rb.get("loss") is not None:
            out["merit"] = max(out["merit"], rel(ra["loss"], rb["loss"]))
    fa, fb = Z[t]["frames"], Z[ref]["frames"]
    ra_, rb_ = list(Z[t]["raws"]), list(Z[ref]["raws"])
    for k in range(1, len(H[t]) + 1):
        s = k * 2 * T
        if s in ra_ and s in rb_:
            d = np.linalg.norm(fa[ra_.index(s)].astype(np.float64) - fb[rb_.index(s)].astype(np.float64), axis=1)
            out["x_sp"] = max(out["x_sp"], float(d.max()) / sp)
    return {k: round(v, 8) for k, v in out.items()}


old_a, old_b, new = tags
res = {"noise_old_b_vs_old_a": compare(old_b, old_a), "new_vs_old_a": compare(new, old_a),
       "new_vs_old_b": compare(new, old_b), "windows": len(H[new])}
for t in tags:
    h = H[t][1:] if len(H[t]) > 1 else H[t]                   # the first window carries the module loads
    res[f"seconds_{t}"] = {k: round(float(np.mean([r.get(k) or 0.0 for r in h])), 3)
                           for k in ("t_start", "t_grad", "t_ls", "t_commit")}
    res[f"seconds_{t}"]["window"] = round(sum(res[f"seconds_{t}"].values()), 3)
print(json.dumps(res, indent=1))
