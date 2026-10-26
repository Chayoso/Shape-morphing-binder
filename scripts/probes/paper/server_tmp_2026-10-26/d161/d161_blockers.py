"""d161_blockers.py OUT_MD CURVE_JSONL [...] -- D161: which term of the physics objective blocks a relief, from the curve's
stored values (no new evaluation). For each (mesh, N, w, dt): the minimiser ds* (pitches; a parabola through the three
lowest of the nine ds points) of the physics terms plus lambda_adapt x the render term, with every term (all), with one
term left out (-T transport, -P surface proximity, -nn near band, -sp spray cleanup), and with the transport alone
(T only: P, nn and sp left out). The share ds* / dt. A term blocks where leaving it out moves ds* most."""
import json
import sys

import numpy as np


def pmin(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    j = int(np.argmin(ys))
    if j == 0 or j == len(xs) - 1:
        return float(xs[j])
    c = np.polyfit(xs[j - 1:j + 2], ys[j - 1:j + 2], 2)
    return float(-c[1] / (2 * c[0])) if c[0] > 0 else float(xs[j])


out, files = sys.argv[1], sys.argv[2:]
rows = []
for f in files:
    for ln in open(f):
        r = json.loads(ln)
        if "w" in r:
            rows.append(r)
sets = {"all": ("T", "P", "nn", "sp"), "-T": ("P", "nn", "sp"), "-P": ("T", "nn", "sp"), "-nn": ("T", "P", "sp"),
        "-sp": ("T", "P", "nn"), "T only": ("T",), "P only": ("P",)}
lines = ["| mesh | N | w | dt | " + " | ".join(sets) + " | render alone | blocker |", "|" + "---|" * (len(sets) + 6)]
summary = {}
for r in sorted(rows, key=lambda r: (r["mesh"], r["N"], r["w"], r["dt"])):
    DS = r["ds"]
    lam = r["totals"]["adapt"]["lam"]
    Lr = [r["render"][str(d)]["L"] for d in DS]
    res = {}
    for k, terms in sets.items():
        E = [sum(r["phys"][str(d)]["val"][t] for t in terms) + lam * Lr[i] for i, d in enumerate(DS)]
        res[k] = pmin(DS, E)
    ro = r["render_only"]["ds_star"]
    gain = {t: res["-" + t] - res["all"] for t in ("T", "P", "nn", "sp")}
    blk = max(gain, key=gain.get)
    summary.setdefault((r["mesh"], r["N"]), []).append((r["w"], r["dt"], blk, gain[blk], res))
    lines.append(f"| {r['mesh']} | {r['N'] // 1000}k | {r['w']:.1f} | {r['dt']} | "
                 + " | ".join(f"{res[k]:.2f} ({res[k] / r['dt']:.2f})" for k in sets) + f" | {ro:.2f} | {blk} |")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
