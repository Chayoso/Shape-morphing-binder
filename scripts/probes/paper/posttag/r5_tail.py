"""r5_tail.py OUT_MD -- D145 reading 5 (3), r4_tail.py on all six meshes: the part of each render-arm run that comes after
its own silhouette term has reached the sampling floor, and what the end measures do there (CPU; the runs' records and logs only).

Per B1 render-arm run (d145t_b1, seeds 97 / 99 / 101, six meshes): the first committed window whose silhouette term
d_sil is at or below the seed's floor (D152's own_floor.jsonl, state `ind`: the independent sample read against the
run's own pictures); the simulated time of that window's end (frame_end / 240 s); the share of the committed windows
and of the wall time (the attempts' t_attempt) after it; the ratios end / crossing of d_sil, d_pbr, the transport
energy and the kinetic record; where B1 has every-frame readings (t3_ / e3_: bunny, dragon, and from reading 5 L97 of
teapot, bob, armadilo, homer) the yardstick's exterior silhouette and shading and the display front 1 - IoU at the kept
frame nearest the crossing against the last, and the price of the last 50 / 25 / 10 % of the wall."""
import json
import sys
from pathlib import Path

import numpy as np

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
floor = {}
for s in open(G / "d152/own_floor.jsonl"):
    r = json.loads(s)
    if r["state"] == "ind":
        floor[(r["mesh"], r["seed"])] = r


def frames(path, get):
    out = []
    try:
        for s in open(path):
            if s.startswith("{"):
                r = json.loads(s)
                try:
                    out.append((int(r["state"]) / 240.0, float(get(r))))
                except (KeyError, ValueError, TypeError):
                    pass
    except OSError:
        pass
    return sorted(out)


def at(series, t):
    if not series:
        return None
    return min(series, key=lambda p: abs(p[0] - t))[1]


L = ["### After the render's own floor: the tail of each render-arm run (B1 on the tag)", "",
     "| mesh | seed | floor d_sil e-4 | crossing at (s) | run ends (s) | windows after | wall after | d_sil end / crossing | "
     "d_pbr end / crossing | transport end / crossing | kin end / crossing | yardstick sil end / crossing | "
     "yardstick shading end / crossing | display front end / crossing |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
rows = []
for mesh in ("bunny", "dragon", "teapot", "bob", "armadilo", "homer"):
    for seed in (97, 99, 101):
        name = f"{mesh}300k_L{seed}"
        try:
            arm = json.loads((G / "d145t_b1" / f"{name}.json").read_text())["arms"]["render_full_dt_iso_nn"]
        except OSError:
            continue
        h = arm["history"]
        fl = floor.get((mesh, seed))
        if not fl:
            continue
        c = [(i, x) for i, x in enumerate(h) if x.get("outer_accepted") and x.get("d_sil") is not None]
        ic = next((j for j, (_, x) in enumerate(c) if x["d_sil"] <= fl["sil"]), None)
        row = dict(mesh=mesh, seed=seed, floor_sil=fl["sil"], windows=len(c), t_end=c[-1][1]["frame_end"] / 240.0)
        if ic is None:
            row["crossing"] = None
        else:
            ia, xc = c[ic]
            xe = c[-1][1]
            tc = xc["frame_end"] / 240.0
            wall = [x.get("t_attempt") or 0.0 for x in h]
            row.update(crossing=tc, windows_after=(len(c) - 1 - ic) / len(c),
                       wall_after=sum(wall[ia + 1:]) / max(sum(wall), 1e-9),
                       r_sil=xe["d_sil"] / xc["d_sil"], r_pbr=xe["d_pbr"] / xc["d_pbr"],
                       r_transport=xe["transport_energy"] / xc["transport_energy"],
                       r_kin=(xe.get("kin") or np.nan) / (xc.get("kin") or np.nan))
            ys = yp = df = []
            if (G / "d145t_b1" / f"t3_{name}.log").exists():                 # every-frame readings: all six meshes now
                ys = frames(G / "d145t_b1" / f"t3_{name}.log", lambda r: r["exterior"]["sil"])
                yp = frames(G / "d145t_b1" / f"t3_{name}.log", lambda r: r["exterior"]["pbr"])
                df = frames(G / "d145t_b1" / f"e3_{name}.log", lambda r: 1 - r["to_target"]["front"]["iou"])
                for k, s in (("r_yard_sil", ys), ("r_yard_pbr", yp), ("r_disp", df)):
                    if s:
                        row[k] = s[-1][1] / at(s, tc)
            # where the tail's gain comes: per measure, the first time it has made 90 % of its crossing -> end change
            # (kinetic on a log scale), and the share of the run's wall after that time
            cw = np.cumsum([x.get("t_attempt") or 0.0 for x in h])
            wall_at = lambda t: cw[max([i for i, x in c if x["frame_end"] / 240.0 <= t + 1e-9] or [ia])]
            recs = {k: [(x["frame_end"] / 240.0, x.get(k)) for _, x in c[ic:] if x.get(k)]
                    for k in ("d_sil", "d_pbr", "transport_energy", "kin")}
            recs["kin"] = [(t, np.log(v)) for t, v in recs["kin"] if v > 0]
            for k, s in (("yard_sil", ys), ("yard_pbr", yp), ("disp", df)):
                if s:
                    recs[k] = [(t, v) for t, v in s if t >= tc - 1e-9]
            # the price of the last q of the wall: the measure's value where the run's wall reaches (1 - q) of its
            # total against the end, as a share of the end value, and the minutes that q is
            tw = [(x["frame_end"] / 240.0, cw[i]) for i, x in c]
            for q in (0.5, 0.25, 0.1):
                tq = next((t for t, w in tw if w >= (1 - q) * cw[-1]), tw[-1][0])
                row[f"min_last{int(q * 100)}"] = q * cw[-1] / 60.0
                for k, s in (("yard_sil", ys), ("yard_pbr", yp), ("disp", df)):
                    if s:
                        row[f"gain_last{int(q * 100)}_{k}"] = (at(s, tq) - s[-1][1]) / s[-1][1]
            for k, s in recs.items():
                if len(s) < 2 or s[0][1] == s[-1][1]:
                    continue
                g = [(t, (s[0][1] - v) / (s[0][1] - s[-1][1])) for t, v in s]
                t90 = next(t for t, v in g if v >= 0.9)
                row[f"t90_{k}"] = t90
                row[f"wall_after90_{k}"] = (cw[-1] - wall_at(t90)) / max(cw[-1], 1e-9)
        rows.append(row)
        f = lambda k, fmt: "-" if row.get(k) is None else format(row[k], fmt)
        L.append(f"| {mesh} | {seed} | {fl['sil'] * 1e4:.2f} | {f('crossing', '.1f')} | {row['t_end']:.1f} | "
                 f"{f('windows_after', '.0%')} | {f('wall_after', '.0%')} | {f('r_sil', '.2f')} | {f('r_pbr', '.2f')} | "
                 f"{f('r_transport', '.2f')} | {f('r_kin', '.2g')} | {f('r_yard_sil', '.2f')} | {f('r_yard_pbr', '.2f')} | "
                 f"{f('r_disp', '.2f')} |")
ok = [r for r in rows if r.get("crossing") is not None]
if ok:
    g = lambda k: [r[k] for r in ok if r.get(k) is not None and np.isfinite(r[k])]
    L += ["", f"Crossed in {len(ok)} of {len(rows)} runs; windows after the crossing {min(g('windows_after')):.0%}-"
          f"{max(g('windows_after')):.0%} (median {np.median(g('windows_after')):.0%}), wall after "
          f"{min(g('wall_after')):.0%}-{max(g('wall_after')):.0%} (median {np.median(g('wall_after')):.0%}); over that tail "
          f"d_sil x {min(g('r_sil')):.2f}-{max(g('r_sil')):.2f}, d_pbr x {min(g('r_pbr')):.2f}-{max(g('r_pbr')):.2f}, "
          f"transport x {min(g('r_transport')):.2f}-{max(g('r_transport')):.2f}"
          + ("" if not g("r_yard_sil") else f"; with every-frame readings: yardstick silhouette x {min(g('r_yard_sil')):.2f}-"
             f"{max(g('r_yard_sil')):.2f}, shading x {min(g('r_yard_pbr')):.2f}-{max(g('r_yard_pbr')):.2f}, display front x "
             f"{min(g('r_disp')):.2f}-{max(g('r_disp')):.2f}") + ".", ""]
    K = [("d_sil", "d_sil"), ("d_pbr", "d_pbr"), ("transport_energy", "transport"), ("kin", "kinetic (log)"),
         ("yard_sil", "yardstick sil"), ("yard_pbr", "yardstick shading"), ("disp", "display front")]
    L += ["### Where the tail's gain comes: the time each measure has made 90 % of its crossing -> end change, and the "
          "share of the run's wall after it", "",
          "| mesh | seed | crossing (s) | " + " | ".join(n for _, n in K) + " |", "|---|---|---|" + "---|" * len(K)]
    for r in ok:
        cell = lambda k: ("-" if r.get(f"t90_{k}") is None
                          else f"{r[f't90_{k}']:.1f} s, {r[f'wall_after90_{k}']:.0%} after")
        L.append(f"| {r['mesh']} | {r['seed']} | {r['crossing']:.1f} | " + " | ".join(cell(k) for k, _ in K) + " |")
    for k, n in K:
        v = g(f"wall_after90_{k}")
        if v:
            L.append(f"- {n}: wall after its 90 % point {min(v):.0%}-{max(v):.0%} (median {np.median(v):.0%}), {len(v)} runs")
    L += ["", "### The price of the last part of the wall: how much worse the end measure was where the run's wall reached "
          "50 / 75 / 90 % of its total (a share of the end value), and the minutes of that last part", "",
          "| mesh | seed | last 50 %: min, yardstick sil / shading / display | last 25 % | last 10 % |", "|---|---|---|---|---|"]
    for r in ok:
        def cell(q):
            if r.get(f"gain_last{q}_yard_sil") is None:
                return f"{r[f'min_last{q}']:.1f} min, -"
            return (f"{r[f'min_last{q}']:.1f} min, {r[f'gain_last{q}_yard_sil']:+.0%} / {r[f'gain_last{q}_yard_pbr']:+.0%} / "
                    f"{r[f'gain_last{q}_disp']:+.0%}")
        L.append(f"| {r['mesh']} | {r['seed']} | {cell(50)} | {cell(25)} | {cell(10)} |")
    for q in (50, 25, 10):
        for k, n in (("yard_sil", "yardstick sil"), ("yard_pbr", "shading"), ("disp", "display front")):
            v = g(f"gain_last{q}_{k}")
            if v:
                L.append(f"- last {q} % of the wall, {n}: the end is {min(v):.0%}-{max(v):.0%} better (median "
                         f"{np.median(v):.0%}), {len(v)} runs")
    L.append("")
text = "\n".join(L) + "\n"
Path(sys.argv[1]).write_text(text)
Path(sys.argv[1]).with_suffix(".json").write_text(json.dumps(rows, default=float))
print(text)
