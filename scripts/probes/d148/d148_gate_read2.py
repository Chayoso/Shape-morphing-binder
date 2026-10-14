"""d148_gate_read2.py dragon ARM TWIN [ARM ...] bunny ARM TWIN -- D148's 300k gate on the D146b base (refreeze-prep,
--relax_driven --u_adam_carry), item by item as D148's gate, the references from D146b's own runs (output/gpu/d146g):
the render arms CDL, CDL2 (dragon) and CBL, CBL2 (bunny), the twins CDP and CBP. (2) the shell against the bulk at
windows 1-9 in the two arm runs' band widened by its width (the twins against the same band, as D148's gate read them);
(3) the end in the arms' two-run band widened by its width (a twin: its D146b twin's value moved by the arms' width),
at rest (D148's limits), the render lead (yardstick silhouette and display front against the twin); (4) the particles
apart from the body at windows 3-6 and 3-10 not above the larger of the D146b runs of the same mesh and arm
(d120_apart.py: output/gpu/d148/apart_base146_*.log); (5) seconds an attempt not above 1.25 x the mean of the same
D146b runs. Item (1) (the side views) is read on the renders and the detector's tracks by eye."""
import contextlib
import io
import json
import re
import sys

import numpy as np

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d148")
_argv, sys.argv = sys.argv, sys.argv[:1]               # d148_gate_read reads its own arguments at import
with contextlib.redirect_stdout(io.StringIO()):
    from d148_gate_read import BANDS, REST, apart, endeval, jrows, run_json  # noqa: E402
sys.argv = _argv

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
D, B = G + "/d148", G + "/d146g"
BASE = {"dragon": (["CDL", "CDL2"], ["CDP"]), "bunny": (["CBL", "CBL2"], ["CBP"])}
KEYS = {"dragon": ("rel_med", "rel_p90"), "bunny": ("rel_med", "rel_p90", "rel_face_med")}


def band(vals):
    lo, hi = min(vals), max(vals)
    return lo - (hi - lo), hi + (hi - lo)


ARMS, cur = {}, None
for a in sys.argv[1:]:
    if a in ("dragon", "bunny"):
        cur = a
        ARMS[cur] = []
    else:
        ARMS[cur].append(a)
fails = []
for mesh, tags in ARMS.items():
    arms_b, twins_b = BASE[mesh]
    sh = {t: {r["window"]: r for r in jrows(f"{B}/shear_{mesh}300k_{t}.log")} for t in arms_b}
    print(f"== {mesh}: (2) the shell against its bulk at windows 1-9 in the band of {'/'.join(arms_b)} widened by its "
          f"width")
    for tag in tags:
        rows = [r for r in jrows(f"{D}/shear_{mesh}300k_{tag}.log") if 1 <= r.get("window", 0) <= 9]
        out = []
        for r in rows:
            for k in KEYS[mesh]:
                vals = [sh[t][r["window"]][k] for t in arms_b if r["window"] in sh[t]]
                if len(vals) < 2:
                    continue
                lo, hi = band(vals)
                if not lo <= r[k] <= hi:
                    out.append(f"w{r['window']} {k} {r[k]} outside [{lo:.2f}, {hi:.2f}]")
        print(f"  {tag}: {len(rows)} windows; outside: {out if out else 'none'}")
        if out:
            fails.append(f"(2) {mesh} {tag}")
    base = {t: run_json(f"{B}/{mesh}300k_{t}.json") for t in arms_b + twins_b}
    ev_b = {t: endeval(f"{B}/e3end_{mesh}300k_{t}.log", f"{B}/t3end_{mesh}300k_{t}.log") for t in arms_b + twins_b}
    ap_b = {t: apart(f"{D}/apart_base146_{mesh}300k_{t}.log") for t in arms_b + twins_b}
    for t in arms_b + twins_b:
        j = base[t]
        print(f"  base {t}: silIoU {j['sil']:.4f} chamfer {j['chamfer']:.4f} thin {100 * j['thin']:.2f} % kin "
              f"{j['kin']:.2e} | yardstick sil {ev_b[t][2]:.2e} front {ev_b[t][1]:.4f} | apart {ap_b[t]} | "
              f"{j['s_att']:.2f} s an attempt, {j['minutes']:.1f} min")
    w_sil = max(base[t]["sil"] for t in arms_b) - min(base[t]["sil"] for t in arms_b)
    w_ch = max(base[t]["chamfer"] for t in arms_b) - min(base[t]["chamfer"] for t in arms_b)
    w_th = max(base[t]["thin"] for t in arms_b) - min(base[t]["thin"] for t in arms_b)
    print(f"== {mesh}: (3) the end, at rest, the render lead; (4) apart; (5) cost")
    ev = {}
    for tag in tags:
        twin = tag.endswith("T")
        ref = twins_b if twin else arms_b
        j = run_json(f"{D}/{mesh}300k_{tag}.json")
        if j is None:
            print(f"  {tag}: no JSON yet")
            continue
        rough, front, ysil, ypbr = endeval(f"{D}/e3end_{mesh}300k_{tag}.log", f"{D}/t3end_{mesh}300k_{tag}.log")
        ev[tag] = (ysil, front)
        lim_sil = min(base[t]["sil"] for t in ref) - w_sil
        lim_ch = max(base[t]["chamfer"] for t in ref) + w_ch
        lim_th = max(base[t]["thin"] for t in ref) + w_th
        ok3 = j["sil"] >= lim_sil and j["chamfer"] <= lim_ch and j["thin"] <= lim_th
        ok_rest = j["kin"] <= REST[mesh]
        a36, a310 = apart(f"{D}/apart_{mesh}300k_{tag}.log")
        l36, l310 = max(ap_b[t][0] for t in ref), max(ap_b[t][1] for t in ref)
        ok4 = a36 <= l36 and a310 <= l310
        lim5 = 1.25 * float(np.mean([base[t]["s_att"] for t in ref]))
        ok5 = j["s_att"] <= lim5
        print(f"  {tag}: silIoU {j['sil']:.4f} (>= {lim_sil:.4f}) chamfer {j['chamfer']:.4f} (<= {lim_ch:.4f}) thin "
              f"{100 * j['thin']:.2f} % (<= {100 * lim_th:.2f}) {'ok' if ok3 else 'OUT'} | kin {j['kin']:.2e} (<= "
              f"{REST[mesh]:.1e}) {'ok' if ok_rest else 'OUT'} | roughness {rough:.2f} deg | display front 1-IoU "
              f"{front:.4f} | yardstick sil {ysil:.2e} pbr {ypbr:.2e} | apart 3-6 {a36} (<= {l36}) 3-10 {a310} (<= "
              f"{l310}) {'ok' if ok4 else 'OUT'} | {j['windows']} windows, {j['attempts']} attempts, {j['s_att']:.2f} s "
              f"an attempt (<= {lim5:.2f}) {'ok' if ok5 else 'OUT'}, {j['minutes']:.1f} min")
        for k, ok in (("(3) end", ok3), ("(3) rest", ok_rest), ("(4) apart", ok4), ("(5) cost", ok5)):
            if not ok:
                fails.append(f"{k} {mesh} {tag}")
    for t in [t for t in tags if not t.endswith("T") and t in ev]:
        for tw in [t for t in tags if t.endswith("T") and t in ev]:
            lead = ev[t][0] < ev[tw][0] and ev[t][1] < ev[tw][1]
            print(f"  render lead {mesh} {t} vs {tw}: yardstick sil {ev[t][0]:.2e} vs {ev[tw][0]:.2e}, display front "
                  f"{ev[t][1]:.4f} vs {ev[tw][1]:.4f}: {'ahead' if lead else 'NOT ahead'}")
            if not lead:
                fails.append(f"(3) lead {mesh} {t}")
    for tag in tags:
        try:
            g = list(open(f"{D}/gap_{mesh}300k_{tag}.log"))
        except OSError:
            g = []
        vals = [re.search(r"node-steps (\d+), separating (\d+); .* mass ([\d.]+)", s) for s in g]
        vals = [(int(m.group(1)), int(m.group(2)), float(m.group(3))) for m in vals if m]
        print(f"  split record {tag}: {len(vals)} commits; " + "; ".join(f"{a}/{b} {c:.3f}" for a, b, c in vals[:8]))
print("== automatic items failing: " + (", ".join(fails) if fails else "none"))
# Post-hoc secondary reading (the coordinator's ruling 18:05 CDT; not the pre-registered rule): the pre-registered two-run
# bands are degenerate (CDL = CDL2 in silhouette and chamfer, zero width at some windows), so beside the reading by the
# letter: (2) the D146b runs' range widened by D135's three-run width at that window (D144's bands, d148_gate_read.BANDS);
# (3) the D146b runs' range widened by D135's five-run widths (batches 4-7: silhouette 0.0012 dragon / 0.0005 bunny,
# chamfer 0.0001, thin 0.9 / 0.3 points). Twins: no twin band exists for (2) (CDP and CBP have no shear log), no pass/fail.
W135 = {"dragon": (0.0012, 0.0001, 0.009), "bunny": (0.0005, 0.0001, 0.003)}
print("== POST-HOC secondary reading (the pre-registered two-run band is degenerate: CDL = CDL2), not the gate's rule")
for mesh, tags in ARMS.items():
    arms_b, twins_b = BASE[mesh]
    sh = {t: {r["window"]: r for r in jrows(f"{B}/shear_{mesh}300k_{t}.log")} for t in arms_b}
    for tag in tags:
        rows = [r for r in jrows(f"{D}/shear_{mesh}300k_{tag}.log") if 1 <= r.get("window", 0) <= 9]
        if tag.endswith("T"):
            print(f"  {mesh} {tag} (2): no twin band exists (CDP / CBP have no shear log); med "
                  + " ".join(f"{r['rel_med']:.2f}" for r in rows))
            continue
        out = []
        for r in rows:
            for k in KEYS[mesh]:
                vals = [sh[t][r["window"]][k] for t in arms_b if r["window"] in sh[t]]
                if len(vals) < 2 or k not in BANDS[mesh]:
                    continue
                lo135, hi135 = BANDS[mesh][k][r["window"] - 1]
                w = hi135 - lo135
                if not min(vals) - w <= r[k] <= max(vals) + w:
                    out.append(f"w{r['window']} {k} {r[k]} outside [{min(vals) - w:.2f}, {max(vals) + w:.2f}]")
        print(f"  {mesh} {tag} (2) post-hoc: outside {out if out else 'none'}")
    base = {t: run_json(f"{B}/{mesh}300k_{t}.json") for t in arms_b + twins_b}
    ws, wc, wt = W135[mesh]
    for tag in tags:
        j = run_json(f"{D}/{mesh}300k_{tag}.json")
        if j is None:
            continue
        ref = twins_b if tag.endswith("T") else arms_b
        ls_ = min(base[t]["sil"] for t in ref) - ws
        lc = max(base[t]["chamfer"] for t in ref) + wc
        lt = max(base[t]["thin"] for t in ref) + wt
        ok = j["sil"] >= ls_ and j["chamfer"] <= lc + 1e-9 and j["thin"] <= lt
        print(f"  {mesh} {tag} (3) post-hoc: silIoU {j['sil']:.4f} (>= {ls_:.4f}) chamfer {j['chamfer']:.4f} (<= {lc:.4f}) "
              f"thin {100 * j['thin']:.2f} % (<= {100 * lt:.2f}) {'ok' if ok else 'OUT'}")
