"""b10sum.py B10_JSONL ARM_A ARM_B MESH [...] [--md OUT_MD] -- D145 B10 post-tag: how far each loss term falls, render arm
against twin, from b10_curves.py's per-window records (CPU only).

Per run, over its committed windows (simulated time t = frame_end / 240 s): each term at the end (the last committed
window), at t = 1 / 2 / 4 / 8 s (linear interpolation in t), and its fall (the first committed window's value over the
end value). Terms: total merit (outer_merit; the twin's has no render term, so the two totals are not the same sum),
the physics part (outer_merit_phys, which the record keeps relative to the epoch's first committed window: selection.py
divides by that window's value, so it cannot be subtracted from the absolute terms), transport energy, released motion
(stab), end drift (stab_end), the silhouette and shading terms (d_sil, d_pbr; recorded at weight zero in the twin) and,
where recorded (--telemetry runs), the proximity (sup_B) and the near band (merit_far). Per mesh and term: A's and
B's min / median / max over seeds and whether the ranges overlap (D145's reading)."""
import json
import re
import sys

import numpy as np

args = sys.argv[1:]
md = None
if "--md" in args:
    i = args.index("--md")
    md = args[i + 1]
    del args[i:i + 2]
rows = [json.loads(s) for s in open(args[0]) if s.startswith("{")]
A, B = args[1], args[2]
meshes = args[3:]
TERMS = [("outer_merit", "total merit"), ("outer_merit_phys", "physics part (rel.)"), ("transport_energy", "transport"),
         ("stab", "released motion"), ("stab_end", "end drift"), ("d_sil", "silhouette"),
         ("d_pbr", "shading"), ("sup_B", "proximity"), ("merit_far", "near band")]
TIMES = (1.0, 2.0, 4.0, 8.0)


def series(r, key):
    w = [x for x in r["windows"] if x.get("outer_accepted") and x.get("t") is not None]
    if key == "local":
        v = [x["outer_merit_phys"] - (x.get("transport_energy") or 0) - (x.get("stab") or 0) - (x.get("stab_end") or 0)
             if x.get("outer_merit_phys") is not None else None for x in w]
    else:
        v = [x.get(key) for x in w]
    pts = [(x["t"], y) for x, y in zip(w, v) if y is not None]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def reads(r, key):
    t, y = series(r, key)
    if len(t) < 2:
        return None
    out = {"end": float(y[-1]), "fall": float(y[0] / y[-1]) if y[-1] > 0 else None, "t_end": float(t[-1])}
    for T in TIMES:
        out[f"t{T:g}"] = float(np.interp(T, t, y)) if t[0] <= T <= t[-1] else None
    return out


def pick(mesh, arm):
    pat = re.compile(rf"^{mesh}300k_{arm}(\d+)$")
    return [r for r in rows if pat.match(r["name"])]


lines = []
f = lambda v: "-" if not v else f"{min(v):.3g} / {np.median(v):.3g} / {max(v):.3g}"
for mesh in meshes:
    ra, rb = pick(mesh, A), pick(mesh, B)
    lines += [f"### B10 {mesh}: {A} (n {len(ra)}) against {B} (n {len(rb)})", "",
              f"| term | at | {A} min / med / max | {B} min / med / max | {A} / {B} (medians) | ranges |",
              "|---|---|---|---|---|---|"]
    for key, label in TERMS:
        ga = [reads(r, key) for r in ra]
        gb = [reads(r, key) for r in rb]
        ga, gb = [g for g in ga if g], [g for g in gb if g]
        if not ga and not gb:
            continue
        for at in ("t1", "t2", "t4", "t8", "end", "fall"):
            va = [g[at] for g in ga if g.get(at) is not None]
            vb = [g[at] for g in gb if g.get(at) is not None]
            if not va and not vb:
                continue
            ratio = "-" if not (va and vb) or np.median(vb) == 0 else f"{np.median(va) / np.median(vb):.2f}"
            ov = "-" if not (va and vb) else ("overlap" if min(va) <= max(vb) and min(vb) <= max(va) else "**apart**")
            lab = {"t1": "t = 1 s", "t2": "t = 2 s", "t4": "t = 4 s", "t8": "t = 8 s", "end": "end",
                   "fall": "first / end"}[at]
            lines.append(f"| {label} | {lab} | {f(va)} | {f(vb)} | {ratio} | {ov} |")
    lines.append("")
text = "\n".join(lines)
print(text)
if md:
    open(md, "w").write(text + "\n")
