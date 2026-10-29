"""d154_bands.py OUT_DIR -- D154 (2): the shares of switch.json in window bands (before: windows -5..0; after: 1-3, 4-6,
7-12, 13-24), per measure and run, with B1's lead at 6 s; a share is reported only where that lead is at least 0.03
(a ratio to a smaller lead is not read). Writes switch_bands.json and prints."""
import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
S = json.load(open(out / "switch.json"))["series"]
BANDS = {"before": (-5, 0), "1-3": (1, 3), "4-6": (4, 6), "7-12": (7, 12), "13-24": (13, 24)}
res = {}
for key in sorted(S):
    for m, d in S[key].items():
        lead = d.get("lead_LP_at_switch")
        if lead is None:
            leads = [r.get("lead_LP") for r in d["rows"] if -2 <= r["n"] <= 0 and r.get("lead_LP") is not None]
            lead = float(np.mean(leads)) if leads else None
        row = dict(lead_LP_6=lead)
        for b, (lo, hi) in BANDS.items():
            v = [r["share"] for r in d["rows"] if lo <= r["n"] <= hi and r.get("share") is not None]
            row[b] = float(np.mean(v)) if v and lead is not None and lead >= 0.03 else None
        res.setdefault(m, {})[key] = row
json.dump(res, open(out / "switch_bands.json", "w"), indent=1)
f = lambda v: "  -  " if v is None else f"{v:5.2f}"  # noqa: E731
for m, d in res.items():
    for key, r in d.items():
        print(f"{m:15s} {key:17s} lead@6 {f(r['lead_LP_6'])} | " + " ".join(f"{b} {f(r[b])}" for b in BANDS))
