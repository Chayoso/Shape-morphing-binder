"""angL145.py DIR/NAME ... -- D145: the angular momentum per unit mass the run carries and who changes it, from the runs own
per-window records (telemetry.below_grid_record: L_start at each committed window, and the window changes L_grid,
L_space, L_u, L_relax); CPU only. Per run: |L_start| median / max over windows and at the last window; per update
|sum_w dL| (net) and sum_w |dL| (path); the sum of all parts against L_end - L_start (closure)."""
import json, sys
import numpy as np
G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
for spec in sys.argv[1:]:
    d, name = spec.split("/")
    hist = next(iter(json.load(open(f"{G}/{d}/{name}.json"))["arms"].values()))["history"]
    rows = [h for h in hist if h.get("L_start") is not None]
    Ls = np.array([h["L_start"] for h in rows])
    nL = np.linalg.norm(Ls, axis=1)
    parts = {k: np.array([h[k] for h in rows]) for k in ("L_grid", "L_space", "L_u", "L_relax")}
    s = " ".join(f"{k[2:]} {np.linalg.norm(v.sum(0)):.2e}/{np.linalg.norm(v, axis=1).sum():.2e}" for k, v in parts.items())
    print(f"{name:18s} w={len(rows):3d} |L_start| med {np.median(nL):.2e} max {nL.max():.2e} last {nL[-1]:.2e} | dL net/path: {s}")
