"""md145m.py TITLE MEASURES BAND_JSON [...] -- D145 post-tag: one markdown table from several band145t.py comparisons, a
row per (comparison, mesh): "A med / B med (change, * = the ranges do not overlap)", as md145t.py's compact form.
MEASURES: comma-separated KEY[:SCALE[:LABEL]]."""
import json
import sys

title, spec, files = sys.argv[1], sys.argv[2], sys.argv[3:]
meas = []
for m in spec.split(","):
    p = m.split(":")
    meas.append((p[0], float(p[1]) if len(p) > 1 and p[1] else 1.0, p[2] if len(p) > 2 else p[0]))


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
print("| A vs B | mesh (n A / B) | " + " | ".join(lab for _, _, lab in meas) + " |")
print("|---|---|" + "---|" * len(meas))
for path in files:
    rows = json.load(open(path))
    by, meshes = {}, []
    for r in rows:
        by[(r["mesh"], r["measure"])] = r
        if r["mesh"] not in meshes:
            meshes.append(r["mesh"])
    for mesh in meshes:
        cells, na, nb, A, B = [], 0, 0, "", ""
        for key, sc, _ in meas:
            r = by.get((mesh, key))
            if not r or not r["a"] or not r["b"]:
                cells.append("-")
                continue
            A, B = r["A"], r["B"]
            na, nb = max(na, len(r["a"])), max(nb, len(r["b"]))
            ch = "" if r["rel"] is None else f" ({r['rel'] * 100:+.0f} %{'*' if not r['overlap'] else ''})"
            if r["rel"] is None and not r["overlap"]:
                ch = " (*)"
            cells.append(f"{f(med(r['a']) * sc)} / {f(med(r['b']) * sc)}{ch}")
        if A:
            print(f"| {A} vs {B} | {mesh} ({na} / {nb}) | " + " | ".join(cells) + " |")
print()
