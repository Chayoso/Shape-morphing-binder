"""cut.py RUNS_DIR TAG[,TAG...] -- D147: the delivered video's cut, by the rule written before the 300k runs were read
(the user's choice, 2026-10-14: keep the physics, cut after the first floor bounces and before the second contact):
the last kept frame before the bodies come within one sample spacing of each other again after they part (they part
at the first kept frame after the first contact with no particle of one body within a spacing of the other). No
second contact: the whole run. Amended (the coordinator, 2026-10-14 17:35; the user's intent "before the second
collision", the spacing a proxy for contact): the earlier of that frame and the last kept frame at or before the start
of the first window with a grid contact impulse after the parting (at 300k the MPM cell is wider than the spacing).
Prints one JSON line per run (both cuts, kept-frame indices, simulated steps and seconds). Read-only, CPU (scipy)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def touching(a, b, sp):
    if (np.maximum(a.min(0) - b.max(0), b.min(0) - a.max(0)) > sp).any():
        return False                                   # the boxes more than a spacing apart
    return bool(np.isfinite(cKDTree(b).query(a, distance_upper_bound=sp)[0]).any())


runs = Path(sys.argv[1])
for tag in sys.argv[2].split(","):
    j = json.loads((runs / f"{tag}.json").read_text())
    z = np.load(runs / f"{tag}.npz")
    fr, rows, raws = z["frames"], z["rows"], z["raws"]
    dt, R = j["mpm"]["dt"], j["scene"]["R"]
    sp = (4.18879 * R ** 3 / (rows[0][1] - rows[0][0])) ** (1 / 3)
    (s0, e0), (s1, e1) = rows
    first = part = again = None
    for k in range(len(fr)):
        c = touching(np.asarray(fr[k, s0:e0]), np.asarray(fr[k, s1:e1]), sp)
        if first is None:
            first = k if c else None
        elif part is None:
            part = None if c else k
        elif c:
            again = k
            break
    cut_sp = (again - 1) if again is not None else len(fr) - 1
    # the amendment (the coordinator, 17:35): the earlier of that and the first grid contact impulse after the parting
    # (the last kept frame at or before the start of the first window with an impulse that starts after the parting):
    # at 300k the MPM cell is wider than the spacing, so the grid contact begins before the spacing test fires
    win = 2 * int(j["config"]["T"])
    hist = [r for r in j["history"] if "bodies" in r]
    imp = next((r for r in hist if part is not None and np.linalg.norm(r["contact_impulse"]) > 0
                and r["animation"] * win >= raws[part]), None)
    cut_imp = int(np.searchsorted(raws, imp["animation"] * win, side="right")) - 1 if imp is not None else len(fr) - 1
    cut = min(cut_sp, cut_imp)
    sec = lambda k: None if k is None else round(float(raws[k] * dt), 3)   # noqa: E731
    print(json.dumps({"tag": tag, "first_contact": first, "parted": part, "second_contact": again, "cut": cut,
                      "cut_spacing": cut_sp, "t_cut_spacing": sec(cut_sp), "cut_impulse": cut_imp,
                      "t_cut_impulse": sec(cut_imp),
                      "cut_step": int(raws[cut]), "t_first": sec(first), "t_part": sec(part), "t_second": sec(again),
                      "t_cut": sec(cut), "frames": len(fr)}), flush=True)
