"""d148_gate_read.py dragon GA GT [GA99] bunny BA BT -- D148's 300k gate read item by item as pre-registered
(docs/experiments.md, D148): (2) the shell against the bulk at windows 1-9 in D144's three-run bands widened by their
width; (3) the end in D138's widened band, at rest, the render lead (yardstick silhouette and display front against the
twin); (4) the particles apart from the body in windows 3-6 and 3-10 against the larger of D135's two runs of the same
mesh and arm; (5) seconds an attempt against 1.25 x batch 5's W3; and the split record. Item (1) (the side views) is read
on the renders and the detector's candidates by eye; the relief and roughness from bands_*.log and e3end_."""
import json
import re
import sys

import numpy as np

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
D = G + "/d148"
BANDS = {   # D144's three-run bands (D135's code), windows 1-9
    "bunny": {"rel_med": [(1.32, 1.59), (2.95, 3.31), (3.90, 4.65), (4.72, 5.14), (5.05, 5.59), (5.21, 5.48), (4.81, 5.62),
                          (4.94, 5.48), (5.07, 5.40)],
              "rel_p90": [(1.99, 2.20), (5.07, 5.64), (6.57, 9.96), (8.85, 10.95), (10.48, 12.31), (11.82, 12.90),
                          (12.65, 13.94), (13.97, 14.06), (13.89, 15.21)],
              "rel_face_med": [(1.72, 1.75), (4.59, 4.68), (6.41, 7.64), (6.54, 9.30), (7.47, 8.67), (7.41, 8.28),
                               (6.94, 7.96), (6.64, 8.14), (6.29, 8.24)]},
    "dragon": {"rel_med": [(1.43, 1.46), (3.19, 3.94), (4.62, 6.48), (5.84, 8.45), (6.93, 9.33), (7.96, 9.40), (8.49, 9.57),
                           (9.01, 9.67), (9.32, 9.65)],
               "rel_p90": [(1.92, 2.04), (5.12, 5.48), (7.43, 10.37), (7.95, 15.75), (8.28, 19.38), (9.31, 20.44),
                           (11.04, 19.89), (13.48, 18.70), (15.35, 18.71)]}}
END = {"bunny": (0.9871, 0.0552, 0.052), "dragon": (0.9853, 0.0546, 0.055)}
REST = {"bunny": 1.53e-5, "dragon": 3.3e-5}
W3 = {"bunny": 6.72, "dragon": 9.65}
# (4) the limits: the larger of D135's two runs of the same mesh and arm (D137's LT / PT as recorded there; FB, SB, FT,
# BP0 read with d120_apart.py: tmp/d140/apart_{FB,SB}.log, tmp/d148/apart_base_*.log), windows 3-6 / 3-10
APART = {}


def jrows(p):
    try:
        return [json.loads(s) for s in open(p) if s.startswith("{")]
    except OSError:
        return []


def apart(p):
    rows = []
    try:
        for s in open(p):
            m = re.match(r"frame\s+\d+ raw\s+\d+ window\s+([\d.]+) apart\s+(\d+)", s)
            if m:
                rows.append((float(m.group(1)), int(m.group(2))))
    except OSError:
        return (-1, -1)
    w = np.array([r[0] for r in rows])
    a = np.array([r[1] for r in rows])
    pick = lambda lo, hi: int(a[(w >= lo) & (w <= hi)].max()) if ((w >= lo) & (w <= hi)).any() else -1  # noqa: E731
    return pick(3, 6), pick(3, 10)


def run_json(path):
    try:
        a = next(iter(json.load(open(path))["arms"].values()))
    except (OSError, StopIteration, KeyError):
        return None
    h = [r for r in a["history"] if r.get("frame_end") and not r.get("null_commit") and r["frame_end"] <= a["deliver_n"]]
    att = [r.get("t_attempt") for r in a["history"] if r.get("t_attempt")]
    m = a["metrics"]
    return dict(sil=m["sil_iou"], chamfer=m["chamfer"], thin=m.get("thin_uncovered"), kin=h[-1]["kin"], windows=len(h),
                attempts=len(a["history"]), s_att=float(np.mean(att)) if att else a["seconds"] / max(len(a["history"]), 1),
                minutes=a["seconds"] / 60, deliver=a["deliver_n"], merit=min(r["selection_merit"] for r in h))


def endeval(e3, t3):
    rows = jrows(e3)
    st = [r for r in rows if str(r.get("state")) not in ("own", "target")]
    last = st[-1] if st else None
    t = jrows(t3)
    return (last["field_normals"]["angle_rms"] if last else np.nan,
            1 - last["to_target"]["front"]["iou"] if last else np.nan,
            t[-1]["exterior"]["sil"] if t else np.nan, t[-1]["exterior"]["pbr"] if t else np.nan)


T = "/data/relcfd/chayo/physmorph_v2/tmp"
base = {("dragon", 0): [(12, 2136), apart(f"{T}/d140/apart_FB.log")],
        ("dragon", 1): [(6, 2318), apart(f"{T}/d148/apart_base_dragon300k_FT.log")],
        ("bunny", 0): [(939, -1), apart(f"{T}/d140/apart_SB.log")],
        ("bunny", 1): [(1192, -1), apart(f"{T}/d148/apart_base_bunny300k_BP0.log")]}
ARMS, cur = {}, None
for a in sys.argv[1:]:
    if a in ("dragon", "bunny"):
        cur = a
        ARMS[cur] = []
    else:
        ARMS[cur].append(a)
fails = []
print("== (2) 0-1.5 s: the shell against its bulk at windows 1-9, D144's three-run bands widened by their width")
for mesh, arms in ARMS.items():
    for tag in arms:
        rows = [r for r in jrows(f"{D}/shear_{mesh}300k_{tag}.log") if 1 <= r.get("window", 0) <= 9]
        out = []
        for k, bands in BANDS[mesh].items():
            for r in rows:
                lo, hi = bands[r["window"] - 1]
                w = hi - lo
                if not lo - w <= r[k] <= hi + w:
                    out.append(f"w{r['window']} {k} {r[k]} outside [{lo - w:.2f}, {hi + w:.2f}]")
        print(f"  {mesh}300k_{tag}: {len(rows)} windows; outside: {out if out else 'none'}")
        print("     med " + " ".join(f"{r['rel_med']:.2f}" for r in rows) + " | p90 " +
              " ".join(f"{r['rel_p90']:.2f}" for r in rows) + " | bulk " + " ".join(f"{r['bulk_moved_med']:.2f}" for r in rows))
        if out:
            fails.append(f"(2) {mesh} {tag}")
print("== (3) the end, at rest, the render lead; (5) cost; (4) apart")
for mesh, arms in ARMS.items():
    ev = {}
    for i, tag in enumerate(arms):
        name = f"{mesh}300k_{tag}"
        j = run_json(f"{D}/{name}.json")
        rough, front, ysil, ypbr = endeval(f"{D}/e3end_{name}.log", f"{D}/t3end_{name}.log")
        ev[tag] = (ysil, front)
        a36, a310 = apart(f"{D}/apart_{name}.log")
        twin = tag.endswith("T")
        lim = base[(mesh, int(twin))]
        l36 = max(v[0] for v in lim)
        l310 = max(v[1] for v in lim)
        if j is None:
            print(f"  {name}: no JSON yet")
            continue
        e = END[mesh]
        ok3 = j["sil"] >= e[0] and j["chamfer"] <= e[1] and (j["thin"] or 1) <= e[2]
        ok_rest = j["kin"] <= REST[mesh]
        ok5 = j["s_att"] <= 1.25 * W3[mesh]
        ok4 = a36 <= l36 and a310 <= l310
        print(f"  {name}: silIoU {j['sil']:.4f} (>= {e[0]}) chamfer {j['chamfer']:.4f} (<= {e[1]}) thin "
              f"{100 * (j['thin'] or np.nan):.2f} % (<= {100 * e[2]:.1f}) {'ok' if ok3 else 'OUT'} | kin {j['kin']:.2e} "
              f"(<= {REST[mesh]:.1e}) {'ok' if ok_rest else 'OUT'} | roughness {rough:.2f} deg | display front 1-IoU "
              f"{front:.4f} | yardstick sil {ysil:.2e} pbr {ypbr:.2e} | apart 3-6 {a36} (<= {l36}) 3-10 {a310} (<= {l310}) "
              f"{'ok' if ok4 else 'OUT'} | {j['windows']} windows, {j['attempts']} attempts, {j['s_att']:.2f} s an attempt "
              f"(<= {1.25 * W3[mesh]:.2f}) {'ok' if ok5 else 'OUT'}, {j['minutes']:.1f} min, best merit {j['merit']:.3e}")
        for k, ok in (("(3) end", ok3), ("(3) rest", ok_rest), ("(4) apart", ok4), ("(5) cost", ok5)):
            if not ok:
                fails.append(f"{k} {mesh} {tag}")
    arms_ = [t for t in arms if not t.endswith("T")]
    twins = [t for t in arms if t.endswith("T")]
    for t in arms_:
        for tw in twins:
            lead = ev[t][0] < ev[tw][0] and ev[t][1] < ev[tw][1]
            print(f"  render lead {mesh} {t} vs {tw}: yardstick sil {ev[t][0]:.2e} vs {ev[tw][0]:.2e}, display front "
                  f"{ev[t][1]:.4f} vs {ev[tw][1]:.4f}: {'ahead' if lead else 'NOT ahead'}")
            if not lead:
                fails.append(f"(3) lead {mesh} {t}")
print("== the split record per committed window (first 12; split node-steps / separating, separating mass share)")
for mesh, arms in ARMS.items():
    for tag in arms:
        try:
            g = list(open(f"{D}/gap_{mesh}300k_{tag}.log"))
        except OSError:
            g = []
        vals = [re.search(r"node-steps (\d+), separating (\d+); .* mass ([\d.]+)", s) for s in g]
        vals = [(int(m.group(1)), int(m.group(2)), float(m.group(3))) for m in vals if m]
        print(f"  {mesh} {tag}: {len(vals)} commits; " + "; ".join(f"{a}/{b} {c:.3f}" for a, b, c in vals[:12]))
print("== automatic items failing: " + (", ".join(fails) if fails else "none"))
