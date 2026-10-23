"""d162c_sum.py OUT_MD A_JSONL [A_JSONL ...] -- D162c's tables: per run and phase (before / after arrival) the median shares
of the four energies (band-normal, band-tangential, sub-normal, sub-tangential), the normal and band totals and the rms
(pitches a 48-frame step for A; arbitrary units for B), for the realized motion (A) and the render's push (B, lines that
carry L_r); then the group medians after arrival (succeeding / failing / twins), and for A each run's after-arrival shares
against its own twin's (D161's twins for coin / bunny, D160's P for band, D163's T for the plaque)."""
import json
import sys
from collections import defaultdict

import numpy as np

KEYS = ("band_normal", "band_tan", "sub_normal", "sub_tan", "normal", "band", "rms")
rows = []
for f in sys.argv[2:]:
    rows += [json.loads(line) for line in open(f)]
out = ["# D162c: where the render's push and the motion go (outer layer)", ""]


def twin_of(run):
    sc, s = run.split("_")[-2], run.split("_")[-1]
    return {"coin": f"twin_coin_{s}", "bunny": f"twin_bunny_{s}", "band": f"P_band_{s}", "plaque": f"T_plaque_{s}"}.get(sc)


for kind, sel in (("A, the realized motion over 48 frames", lambda r: "L_r" not in r),
                  ("B, the render's push -dL_r/dx", lambda r: "L_r" in r)):
    R = [r for r in rows if sel(r)]
    if not R:
        continue
    med = defaultdict(dict)
    by = defaultdict(list)
    for r in R:
        by[(r["run"], r["group"], r["phase"])].append(r)
    out += [f"## {kind}", "", "| run | group | phase | states | " + " | ".join(KEYS) + " |", "|" + "---|" * (4 + len(KEYS))]
    for (run, group, ph), rs in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        m = {k: float(np.median([r[k] for r in rs])) for k in KEYS}
        med[run][ph] = (group, m)
        out.append(f"| {run} | {group} | {ph} | {len(rs)} | " + " | ".join(f"{m[k]:.3f}" for k in KEYS) + " |")
    out += ["", f"Group medians after arrival ({kind[:1]}):", "", "| group | runs | " + " | ".join(KEYS[:-1]) + " |",
            "|" + "---|" * (2 + len(KEYS) - 1)]
    groups = defaultdict(list)
    for run, d in med.items():
        if "after" in d:
            groups[d["after"][0]].append(d["after"][1])
    for g, ms in sorted(groups.items()):
        out.append(f"| {g} | {len(ms)} | " + " | ".join(f"{np.median([m[k] for m in ms]):.3f}" for k in KEYS[:-1]) + " |")
    if kind.startswith("A"):
        out += ["", "Against the twin after arrival (A; arm minus its twin, share points; rms ratio):", "",
                "| run | group | " + " | ".join(KEYS[:-1]) + " | rms / twin |", "|" + "---|" * (3 + len(KEYS) - 1)]
        for run, d in sorted(med.items()):
            t = twin_of(run)
            if run.startswith(("twin", "P_", "T_")) or t not in med or "after" not in d or "after" not in med[t]:
                continue
            g, m = d["after"]
            mt = med[t]["after"][1]
            out.append(f"| {run} | {g} | " + " | ".join(f"{m[k] - mt[k]:+.3f}" for k in KEYS[:-1]) +
                       f" | {m['rms'] / max(mt['rms'], 1e-12):.2f} |")
    out.append("")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
print("\n".join(out))
