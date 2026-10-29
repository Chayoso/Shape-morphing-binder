"""d155_sum.py RECORDS_JSON OUT_MD -- D155: the tables of the records reading (median over seeds, [min, max])."""
import json
import sys

import numpy as np

R = json.load(open(sys.argv[1]))
out = open(sys.argv[2], "w")
MESH = ("bunny", "dragon", "teapot", "bob", "armadilo", "homer")


def g(r, path):
    x = r
    for p in path.split("."):
        if x is None:
            return None
        x = x.get(p) if isinstance(x, dict) else None
    return x


def fmt(vals, nd=3, pct=False):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "–"
    k = 100.0 if pct else 1.0
    if len(v) == 1:
        return f"{v[0] * k:.{nd}f}"
    return f"{np.median(v) * k:.{nd}f} [{min(v) * k:.{nd}f}, {max(v) * k:.{nd}f}]"


def runs(mesh, arm):
    return [r for r in R if r["mesh"] == mesh and r["arm"] == arm]


def p(s=""):
    print(s)
    out.write(s + "\n")


p("## T1 progress (X0 = first committed record), run fraction; median [min, max] over seeds")
p("| mesh arm (n) | windows | T s | lin P90 | lin R90 | A_lin | log P90 | log R90 | A_log | A_log signed | path 90 % (s) | path 90 % (f) | 90 % of particle movement (s) | (f) |")
p("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for m in MESH:
    for a in ("L", "P"):
        rr = runs(m, a)
        if not rr:
            continue
        p(f"| {m} {a} ({len(rr)}) | {fmt([r['windows'] for r in rr], 0)} | {fmt([r['T'] for r in rr], 1)} | "
          f"{fmt([g(r, 'lin.P90') for r in rr])} | {fmt([g(r, 'lin.R90') for r in rr])} | {fmt([g(r, 'lin.A_abs') for r in rr])} | "
          f"{fmt([g(r, 'log.P90') for r in rr])} | {fmt([g(r, 'log.R90') for r in rr])} | {fmt([g(r, 'log.A_abs') for r in rr])} | "
          f"{fmt([g(r, 'log.A_signed') for r in rr])} | {fmt([r['path90_s'] for r in rr], 2)} | {fmt([r['path90_f'] for r in rr])} | "
          f"{fmt([r.get('move90_s') for r in rr], 2)} | {fmt([r.get('move90_f') for r in rr])} |")
p()
p("## T1s the same crossings in simulated seconds")
p("| mesh arm | lin P90 s | lin R90 s | log P90 s | log R90 s | lin sil 90 s | lin shading 90 s | log sil 90 s | log shading 90 s |")
p("|---|---|---|---|---|---|---|---|---|")
for m in MESH:
    for a in ("L", "P"):
        rr = runs(m, a)
        p(f"| {m} {a} | {fmt([g(r, 'lin.P90_s') for r in rr], 2)} | {fmt([g(r, 'lin.R90_s') for r in rr], 2)} | "
          f"{fmt([g(r, 'log.P90_s') for r in rr], 2)} | {fmt([g(r, 'log.R90_s') for r in rr], 2)} | "
          f"{fmt([g(r, 'lin.S90_s') for r in rr], 2)} | {fmt([g(r, 'lin.B90_s') for r in rr], 2)} | "
          f"{fmt([g(r, 'log.S90_s') for r in rr], 2)} | {fmt([g(r, 'log.B90_s') for r in rr], 2)} |")
p()
p("## T2 share of each term's drop still to come (linear / log) at f = 1/3 and at t = 3 s, 6 s")
p("| mesh arm | Pc lin f33 / 3 s / 6 s | R lin f33 / 3 s / 6 s | Pc log f33 / 3 s / 6 s | R log f33 / 3 s / 6 s |")
p("|---|---|---|---|---|")
for m in MESH:
    for a in ("L", "P"):
        rr = runs(m, a)
        cells = []
        for sc in ("lin", "log"):
            for nm in ("Pc", "R"):
                cells.append(" / ".join(fmt([g(r, f"{sc}.{nm}_left_{q}") for r in rr], 1, True).split(" [")[0]
                                        for q in ("f33", "3s", "6s")))
        p(f"| {m} {a} | " + " | ".join(cells) + " |")
p()
p("## T3 the render in the run's thirds (render arm): render share of the merit lam R / (Pc + lam R); lambda; share of the summed per-window move")
p("| mesh | g_share 1st / 2nd / 3rd | merit share 0-10 % / 1st / 2nd / 3rd | lambda 1st / 3rd | record move share 1st / 2nd / 3rd | kept-frame move share 1st / 2nd / 3rd | first 10 % |")
p("|---|---|---|---|---|---|---|")
for m in MESH:
    rr = runs(m, "L")
    th = lambda key, nm: fmt([g(r, f"thirds.{nm}.{key}") for r in rr], 2).split(" [")[0]   # noqa: E731
    p(f"| {m} | {th('g_share', '1st')} / {th('g_share', '2nd')} / {th('g_share', '3rd')} | "
      f"{th('render_share_merit', '0-10')} / {th('render_share_merit', '1st')} / {th('render_share_merit', '2nd')} / {th('render_share_merit', '3rd')} | "
      f"{fmt([g(r, 'thirds.1st.lam') for r in rr], 4).split(' [')[0]} / {fmt([g(r, 'thirds.3rd.lam') for r in rr], 5).split(' [')[0]} | "
      f"{th('move_share', '1st')} / {th('move_share', '2nd')} / {th('move_share', '3rd')} | "
      f"{fmt([g(r, 'move_thirds.1st') for r in rr], 2).split(' [')[0]} / {fmt([g(r, 'move_thirds.2nd') for r in rr], 2).split(' [')[0]} / "
      f"{fmt([g(r, 'move_thirds.3rd') for r in rr], 2).split(' [')[0]} | {fmt([r.get('move_first10') for r in rr], 2)} |")
p()
p("## T4 where the particle movement happens (kept-frame pairs, all particles, share of the run's total) by simulated time")
p("| mesh arm | 0-1 s | 1-2 s | 2-3 s | 3-6 s | 6 s-end | 50 % at (s) | late rate (pitch / 0.05 s) | total (pitch) |")
p("|---|---|---|---|---|---|---|---|---|")
for m in MESH:
    for a in ("L", "P"):
        rr = runs(m, a)
        p(f"| {m} {a} | " + " | ".join(fmt([g(r, f"move_bins_s.{b}") for r in rr], 0, True)
                                     for b in ("0-1", "1-2", "2-3", "3-6", "6-end"))
          + f" | {fmt([r.get('move50_s') for r in rr], 2)} | {fmt([r.get('move_late_rate') for r in rr], 4)} | "
          f"{fmt([r.get('move_total_pitch') for r in rr], 1)} |")
p()
p("## T5 the late phase: value at f = 1/3, 3 s, 6 s over the end value (records; yardstick and display on kept frames)")
p("| mesh arm | Pc | TE | R (own render) | sil | shading | kin | yardstick (ext. sil+shade vs ind.) | display 1-IoU |")
p("|---|---|---|---|---|---|---|---|---|")
for m in MESH:
    for a in ("L", "P"):
        rr = runs(m, a)
        cells = []
        for nm in ("Pc", "TE", "R", "SIL", "PBR", "kin"):
            cells.append(" / ".join(fmt([g(r, f"late.{nm}.{q}") for r in rr], 2).split(" [")[0] for q in ("f33", "3s", "6s")))
        for key in ("yard_late", "disp_late"):
            cells.append(" / ".join(fmt([g(r, f"{key}.{q}") for r in rr], 2).split(" [")[0] for q in ("f33", "3s", "6s")))
        p(f"| {m} {a} | " + " | ".join(cells) + " |")
p()
p("## T6 yardstick and display crossings (kept frames; X0 = the source frame)")
p("| mesh arm | yard lin R90 (f) | yard lin R90 (s) | yard log R90 (f) | yard log R90 (s) | yard lin left at f33 | yard log left at f33 | display lin R90 (s) | display left at f33 |")
p("|---|---|---|---|---|---|---|---|---|")
for m in ("bunny", "dragon"):
    for a in ("L", "P"):
        rr = runs(m, a)
        p(f"| {m} {a} | {fmt([g(r, 'yard_lin.R90') for r in rr])} | {fmt([g(r, 'yard_lin.R90_s') for r in rr], 2)} | "
          f"{fmt([g(r, 'yard_log.R90') for r in rr])} | {fmt([g(r, 'yard_log.R90_s') for r in rr], 2)} | "
          f"{fmt([g(r, 'yard_lin.left_f33') for r in rr], 0, True)} | {fmt([g(r, 'yard_log.left_f33') for r in rr], 0, True)} | "
          f"{fmt([g(r, 'disp_lin.R90_s') for r in rr], 2)} | {fmt([g(r, 'disp_lin.left_f33') for r in rr], 0, True)} |")
p()
p("## T7 ot_scale sensitivity (meshes without a measured ot_scale use 0.6): P90 lin / A_lin / A_log at s = 0.3 / 0.6 / 1.2")
p("| run | s known | P90 lin | A_lin | A_log | P90 log |")
p("|---|---|---|---|---|---|")
for r in R:
    if r["seed"] != 97 or r["arm"] not in ("L", "P"):
        continue
    sv = r["ot_sens"]
    p(f"| {r['name']} | {r['ot_scale']} | " + " | ".join(
        " / ".join(f"{sv[s][k]:.3f}" for s in ("0.3", "0.6", "1.2")) for k in ("P90", "A_lin", "A_log", "P90_log")) + " |")
out.close()
