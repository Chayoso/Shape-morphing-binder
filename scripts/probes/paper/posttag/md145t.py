"""md145t.py BAND_JSON TITLE MEASURES [--detail] -- D145 post-tag: a markdown table from band145t.py's comparison JSON.

MEASURES: comma-separated measure keys (band145t.py's names), each optionally KEY:SCALE:LABEL (the value multiplied by
SCALE, e.g. yard_ext_sil:1e4:yard sil e-4). Compact (default): one row per mesh, a cell per measure "A med / B med
(change, * = the ranges do not overlap)". --detail: one row per mesh and measure with both arms' min / median / max."""
import json
import sys

rows = json.load(open(sys.argv[1]))
title = sys.argv[2]
meas = []
for m in sys.argv[3].split(","):
    p = m.split(":")
    meas.append((p[0], float(p[1]) if len(p) > 1 and p[1] else 1.0, p[2] if len(p) > 2 else p[0]))
detail = "--detail" in sys.argv
by = {}
for r in rows:
    by[(r["mesh"], r["measure"])] = r
meshes = []
for r in rows:
    if r["mesh"] not in meshes:
        meshes.append(r["mesh"])
A = rows[0]["A"] if rows else "A"
B = rows[0]["B"] if rows else "B"


def f(v):
    if v == 0:
        return "0"
    if float(v).is_integer() and abs(v) < 1e6:
        return str(int(v))
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 0.01:
        return f"{v:.4g}" if a < 1 else f"{v:.3g}"
    return f"{v:.2e}"


def med(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


print(f"### {title}\n")
if not detail:
    print("| mesh (n " + A + " / " + B + ") | " + " | ".join(lab for _, _, lab in meas) + " |")
    print("|---|" + "---|" * len(meas))
    for mesh in meshes:
        cells, na, nb = [], 0, 0
        for key, sc, _ in meas:
            r = by.get((mesh, key))
            if not r or not r["a"] or not r["b"]:
                cells.append("-")
                continue
            na, nb = max(na, len(r["a"])), max(nb, len(r["b"]))
            ch = "" if r["rel"] is None else f" ({r['rel'] * 100:+.0f} %{'*' if not r['overlap'] else ''})"
            if r["rel"] is None and not r["overlap"]:
                ch = " (*)"
            cells.append(f"{f(med(r['a']) * sc)} / {f(med(r['b']) * sc)}{ch}")
        print(f"| {mesh} ({na} / {nb}) | " + " | ".join(cells) + " |")
else:
    print(f"| mesh | measure | {A} min / med / max | {B} min / med / max | {A} vs {B} | ranges |")
    print("|---|---|---|---|---|---|")
    for mesh in meshes:
        for key, sc, lab in meas:
            r = by.get((mesh, key))
            if not r or (not r["a"] and not r["b"]):
                continue
            g = lambda v: "-" if not v else f"{f(min(v) * sc)} / {f(med(v) * sc)} / {f(max(v) * sc)}"
            ch = "-" if r["rel"] is None else f"{r['rel'] * 100:+.1f} %"
            ov = "-" if not (r["a"] and r["b"]) else ("overlap" if r["overlap"] else "**apart**")
            print(f"| {mesh} | {lab} | {g(r['a'])} | {g(r['b'])} | {ch} | {ov} |")
print()
