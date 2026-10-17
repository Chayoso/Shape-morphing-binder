"""b13_read300.py OUT_JSONL [LDIR] -- B13's 300k reading (CPU only), as pre-registered: per mesh, the default arm L
(B1's render arm on the tag, output/gpu/LDIR, default d145t_b1) against F (--baseline force, scale 1) and FM (force_mf,
scale 1/4) in output/gpu/b13, seeds 97 / 99 / 101.

Per run:
  - end against the independent sample with the floor (endgen145.py: endgen_b13.jsonl / endgen_L.jsonl in b13):
    silhouette IoU and chamfer, the floor;
  - the yardstick at the end (t3end_: exterior silhouette / shading), the display (e3end_: 1 - IoU front / crop / back);
  - the run's own silhouette IoU, chamfer, thin uncovered;
  - at rest: kin_last (tab145.py: the history's last kinetic record);
  - momentum: the probe on every kept frame (m_: centre of mass largest / end in display pitches a, full precision from
    its rows; net rotation; angular net / gross mean); the records' body net (sum of body_vcom over the committed windows,
    in the pitch P of the apart log) and its rotation; the force records (|sum force_vdv|, sum force_dv, |sum force_vdL|,
    sum force_gross, the largest window ratio force_dv / force_gross, windows with force_dv = 0);
  - floating material: apart particles at windows 3-10 (apart_);
  - runtime: windows (committed), attempts, minutes, seconds an attempt (tab145.py's: seconds / history records);
  - the geometry energy at the delivered last window (B13a's measure; observed).
Reading (D145): an arm differs from L on a measure when the two three-run ranges do not overlap. Then the hypothesis
items, each as written. One JSON line per run to OUT_JSONL; tables to stdout."""
import json
import re
import sys
from pathlib import Path

import numpy as np

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
B13 = G / "b13"
LDIR = G / (sys.argv[2] if len(sys.argv) > 2 else "d145t_b1")
MESHES = ["bunny", "dragon", "teapot", "bob", "armadilo", "homer"]
SEEDS = [97, 99, 101]
ARMS = {"L": (LDIR, "L"), "F": (B13, "F"), "FM": (B13, "FM")}


def jlines(p):
    try:
        return [json.loads(s) for s in open(p) if s.startswith("{")]
    except OSError:
        return []


def num(pattern, text, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


ENDGEN = {}
for f in (B13 / "endgen_b13.jsonl", B13 / "endgen_L.jsonl"):
    for r in jlines(f):
        ENDGEN[r["run"]] = r     # the last row per run


def read_run(d, name):
    row = {"name": name}
    try:
        arm = json.loads((d / f"{name}.json").read_text())["arms"]["render_full_dt_iso_nn"]
    except (OSError, KeyError, ValueError):
        return None
    met, hist = arm["metrics"], arm["history"]
    kin = [h["kin"] for h in hist if h.get("kin") is not None]
    h = [r for r in hist if r.get("frame_end") and not r.get("null_commit") and r.get("body_vcom") is not None]
    dn = arm.get("deliver_n")
    hd = [r for r in h if dn is None or r["frame_end"] <= dn] or h
    row.update(sil_own=met.get("sil_iou"), chamfer_own=met.get("chamfer"), thin=met.get("thin_uncovered"),
               windows=len([r for r in hist if r.get("outer_accepted")]), attempts=len(hist),
               minutes=arm.get("seconds", 0) / 60, s_attempt=arm.get("seconds", 0) / max(len(hist), 1),
               kin_last=kin[-1] if kin else None, energy_end=hd[-1].get("transport_energy") if hd else None,
               body_net_wu=float(np.linalg.norm(np.sum([r["body_vcom"] for r in h], 0))) if h else None,
               body_rot_deg=float(np.degrees(np.linalg.norm(np.sum([r["body_vrot"] for r in h], 0))))
               if h and h[0].get("body_vrot") is not None else None)
    hf = [r for r in h if r.get("force_dv") is not None]
    if hf:
        dv = np.array([r["force_dv"] for r in hf]); gr = np.array([r["force_gross"] for r in hf])
        row.update(force_dv_net=float(np.linalg.norm(np.sum([r["force_vdv"] for r in hf], 0))),
                   force_dv_path=float(dv.sum()), force_dL_net=float(np.linalg.norm(np.sum([r["force_vdL"] for r in hf], 0))),
                   force_gross=float(gr.sum()), force_ratio_win_max=float((dv / np.maximum(gr, 1e-30)).max()),
                   force_zero_windows=int((dv == 0).sum()), force_windows=len(hf),
                   force_absmax=float(max(r["force_absmax"] for r in hf)), force_unit_ratio=hf[-1].get("force_ratio"))
    e = ENDGEN.get(name)
    if e:
        row.update(ind_sil=e["ind"]["sil"], ind_chamfer=e["ind"]["chamfer"], floor=e["floor"],
                   eg_own_sil=e["own"]["sil"], eg_own_chamfer=e["own"]["chamfer"])
    t3 = jlines(d / f"t3end_{name}.log")
    if t3:
        row.update(yard_ext_sil=t3[-1]["exterior"]["sil"], yard_ext_pbr=t3[-1]["exterior"]["pbr"])
    e3 = [r for r in jlines(d / f"e3end_{name}.log") if r.get("state") not in ("target", "own")]
    if e3:
        tt = e3[-1].get("to_target", {})
        row.update({f"disp_{v}_1miou": 1 - tt[v]["iou"] for v in ("front", "crop", "back") if v in tt})
    try:
        ap = (d / f"apart_{name}.log").read_text()
        row.update(apart_w3_10_max=num(r"windows 3-10: apart max (\d+)", ap, int), pitch=num(r"spacing ([\d.]+);", ap))
    except OSError:
        pass
    mom = [r for r in jlines(d / f"m_{name}.log") if "com" in r]
    if mom:
        row.update(com_largest_a=max(r["com"] for r in mom), com_end_a=mom[-1]["com"])
        try:
            s = [x for x in (d / f"m_{name}.log").read_text().splitlines() if "kept frames |" in x][-1]
            row.update(rotation_deg=num(r"net rotation over the run ([\d.e+-]+) deg", s),
                       ang_ng_mean=num(r"net / gross angular move: largest [\d.e+-]+, mean ([\d.e+-]+)", s))
        except IndexError:
            pass
    if row.get("pitch") and row.get("body_net_wu") is not None:
        row["body_net_P"] = row["body_net_wu"] / row["pitch"]
    return row


runs = {}
with open(sys.argv[1], "w") as out:
    for m in MESHES:
        for a, (d, tag) in ARMS.items():
            for s in SEEDS:
                r = read_run(d, f"{m}300k_{tag}{s}")
                if r:
                    r.update(mesh=m, arm=a, seed=s)
                    runs[(m, a, s)] = r
                    out.write(json.dumps(r) + "\n")


def vals(m, a, k):
    return [runs[(m, a, s)][k] for s in SEEDS if (m, a, s) in runs and runs[(m, a, s)].get(k) is not None]


def rng(v, fmt):
    if not v:
        return "-"
    return format(min(v), fmt) if len(v) == 1 or min(v) == max(v) else f"{format(min(v), fmt)}–{format(max(v), fmt)}"


def cmp(m, a, k):
    """'>' / '<' when a's range lies wholly above / below L's (both with three runs); '~' overlap; '?' incomplete."""
    x, y = vals(m, a, k), vals(m, "L", k)
    if len(x) < 3 or len(y) < 3:
        return "?"
    return ">" if min(x) > max(y) else ("<" if max(x) < min(y) else "~")


COLS = [("ind_sil", "ind sil IoU", ".4f"), ("ind_chamfer", "ind chamfer", ".4f"), ("floor", "floor", ".4f"),
        ("yard_ext_sil", "yard sil", ".2e"), ("yard_ext_pbr", "yard shade", ".2e"), ("disp_front_1miou", "disp front", ".4f"),
        ("sil_own", "own sil", ".4f"), ("chamfer_own", "own chamfer", ".4f"), ("thin", "thin", ".4f"),
        ("kin_last", "kin", ".2e"), ("com_largest_a", "com largest (a)", ".2e"), ("rotation_deg", "rotation (deg)", ".4f"),
        ("ang_ng_mean", "ang n/g", ".2e"), ("apart_w3_10_max", "apart 3-10", "d"), ("windows", "windows", "d"),
        ("s_attempt", "s/att", ".2f"), ("minutes", "min", ".1f"), ("energy_end", "geometry energy", ".3e")]
print("| mesh | arm | n | " + " | ".join(c[1] for c in COLS) + " |")
print("|" + "---|" * (len(COLS) + 3))
for m in MESHES:
    for a in ARMS:
        n = len([s for s in SEEDS if (m, a, s) in runs])
        if not n:
            continue
        cells = []
        for k, _, fmt in COLS:
            c = rng(vals(m, a, k), fmt)
            if a != "L":
                c += f" {cmp(m, a, k)}"
            cells.append(c)
        print(f"| {m} | {a} | {n} | " + " | ".join(cells) + " |")

print("\nforce records (sum over committed windows; per run seed 97 / 99 / 101):")
for m in MESHES:
    for a in ("F", "FM"):
        for s in SEEDS:
            r = runs.get((m, a, s))
            if r and r.get("force_windows"):
                print(f"{m} {a}{s}: dv net {r['force_dv_net']:.2e} path {r['force_dv_path']:.2e} wu/s | dL net "
                      f"{r['force_dL_net']:.2e} | gross {r['force_gross']:.3e} | dv/gross window max {r['force_ratio_win_max']:.2e} "
                      f"| zero windows {r['force_zero_windows']}/{r['force_windows']} | absmax {r['force_absmax']:.3f} | unit ratio "
                      f"{r['force_unit_ratio']:.3g} | body net {r.get('body_net_P', float('nan')):.2e} P rot {r.get('body_rot_deg') or float('nan'):.3e} deg")

print("\nhypothesis (each item as pre-registered; '?' = a range incomplete):")
def behind(m, a):
    s, c = cmp(m, a, "ind_sil"), cmp(m, a, "ind_chamfer")
    return "?" if "?" in (s, c) else (s == "<" and c == ">")


for a in ("F", "FM"):
    beh = {m: behind(m, a) for m in MESHES}
    nb = sum(1 for v in beh.values() if v is True)
    print(f"[{a}] behind L on independent-sample silhouette IoU AND chamfer: {nb} of 6 meshes "
          f"({', '.join(f'{m} {v}' for m, v in beh.items())}); predicted >= 4")
    print(f"[{a}] own-sample silhouette IoU below L: {[m for m in MESHES if cmp(m, a, 'sil_own') == '<']}; "
          f"chamfer above: {[m for m in MESHES if cmp(m, a, 'chamfer_own') == '>']}")
    print(f"[{a}] kin_last above L's range: {[m for m in MESHES if cmp(m, a, 'kin_last') == '>']} "
          f"(below: {[m for m in MESHES if cmp(m, a, 'kin_last') == '<']}); predicted: all")
    print(f"[{a}] apart 3-10 above L's range: {[m for m in MESHES if cmp(m, a, 'apart_w3_10_max') == '>']} "
          f"(below: {[m for m in MESHES if cmp(m, a, 'apart_w3_10_max') == '<']}); predicted: all")
    print(f"[{a}] windows above L's range: {[m for m in MESHES if cmp(m, a, 'windows') == '>']}")
    for m in MESHES:
        x, y = vals(m, a, "com_largest_a"), vals(m, "L", "com_largest_a")
        rx, ry = vals(m, a, "rotation_deg"), vals(m, "L", "rotation_deg")
        sx, sy = vals(m, a, "s_attempt"), vals(m, "L", "s_attempt")
        line = f"[{a}] {m}:"
        if x and y:
            line += f" com largest min/L max {min(x) / max(y):.3g}x (max/L max {max(x) / max(y):.3g}x)"
        if rx and ry:
            line += f"; rotation min/L max {min(rx) / max(ry):.3g}x"
        if sx and sy:
            line += f"; s/attempt {100 * (np.mean(sx) / np.mean(sy) - 1):+.1f} %"
        z = [runs[(m, a, s)].get("force_zero_windows") for s in SEEDS if (m, a, s) in runs]
        rr = [runs[(m, a, s)].get("force_ratio_win_max") for s in SEEDS if (m, a, s) in runs]
        if z and None not in z:
            line += f"; force_dv zero windows {z}; dv/gross window max {max(rr):.2e}"
        print(line)
