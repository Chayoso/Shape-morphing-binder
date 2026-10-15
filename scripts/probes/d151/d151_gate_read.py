"""d151_gate_read.py -- D151's 300k gate read item by item as pre-registered (docs/experiments.md, D151, 2026-10-15 14:02 CDT).

New runs in output/gpu/d151 (MESH300k_NL arm, MESH300k_NP twin); the base in output/gpu/d149: D149's SL / SP (dragon),
SL, SL2 / SP, SP2 (bunny), the refreeze-cand configuration; the secondary spread CDL, CDL2, CDP / CBL, CBL2, CBP (D146b
without D149) where the item names it. Each run's JSON, its yardstick (t3end_), display (e3end_), momentum (m_) and apart
(apart_) logs; the visible pieces from visible_apart.py's whole-run records (vfull_*.jsonl: output/gpu/d148 for the
dragon's SL / SP, output/gpu/d151 for the bunny's base and the new runs).
(1) the unique side-view blob particles over the run and the side-view blobs a window at 1.5-6 s at most half the base's
    (dragon SL / SP; bunny the mean of SL, SL2 / SP, SP2); the other bands and all blobs reported;
(2) windows 1-9 (the committed windows in order): the record's d_vol, d_sil and kin within the base's range widened by
    its width (arms SL, CDL, CDL2 / SL, SL2, CBL, CBL2; twins SP, CDP / SP, SP2, CBP);
(3) silhouette IoU, chamfer, thin uncovered within the base's range widened by its width (arms SL, CDL, CDL2 / SL, SL2;
    twins SP, CDP / SP, SP2), the last kinetic record at D148's rest limits, the arm ahead of its twin on the yardstick's
    silhouette and the display's front 1 - IoU;
(4) the momentum probe's largest and final centre-of-mass displacement, final net rotation and mean angular net / gross,
    and the record's net body translation and rotation, not beyond the base's range (as (3)) widened by its width; the
    render weight lambda and g_share (medians over windows) reported;
(5) seconds an attempt at most 1.05 x the base's mean (SL / SP; SL, SL2 / SP, SP2)."""
import json
import re
from pathlib import Path

import numpy as np

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
NEW, BASE = G / "d151", G / "d149"
REST = {"bunny": 1.53e-5, "dragon": 3.3e-5}
REF = {   # (base for (1) and (5), the band's runs for (2), for (3)/(4))
    ("dragon", "NL"): (["SL"], ["SL", "CDL", "CDL2"], ["SL", "CDL", "CDL2"]),
    ("dragon", "NP"): (["SP"], ["SP", "CDP"], ["SP", "CDP"]),
    ("bunny", "NL"): (["SL", "SL2"], ["SL", "SL2", "CBL", "CBL2"], ["SL", "SL2"]),
    ("bunny", "NP"): (["SP", "SP2"], ["SP", "SP2", "CBP"], ["SP", "SP2"]),
}
BANDS = [(0, 8, "0-1.5 s"), (9, 35, "1.5-6 s"), (36, 89, "6-15 s"), (90, 10 ** 6, "15 s-end")]


def jl(p):
    try:
        return [json.loads(s) for s in open(p) if s.startswith("{")]
    except OSError:
        return []


def run(d, mesh, tag):
    name = f"{mesh}300k_{tag}"
    try:
        a = next(iter(json.loads((d / f"{name}.json").read_text())["arms"].values()))
    except (OSError, StopIteration, KeyError, ValueError):
        return None
    h = [r for r in a["history"] if r.get("frame_end") and not r.get("null_commit") and r["frame_end"] <= a["deliver_n"]]
    att = [r.get("t_attempt") for r in a["history"] if r.get("t_attempt")]
    m = a["metrics"]
    out = dict(name=name, sil=m["sil_iou"], chamfer=m["chamfer"], thin=m.get("thin_uncovered"), kin=h[-1]["kin"],
               windows=len(h), attempts=len(a["history"]), s_att=float(np.mean(att)) if att else None,
               minutes=a["seconds"] / 60, early=[(r.get("d_vol"), r.get("d_sil"), r.get("kin")) for r in h[:9]],
               lam=float(np.median([r["lambda"] for r in h if r.get("lambda") is not None] or [np.nan])),
               share=float(np.median([r["g_share"] for r in h if r.get("g_share") is not None] or [np.nan])))
    out["body_net"] = float(np.linalg.norm(np.sum([r["body_vcom"] for r in h if r.get("body_vcom") is not None], 0)))
    out["body_rot_deg"] = float(np.degrees(np.linalg.norm(np.sum([r["body_vrot"] for r in h
                                                                   if r.get("body_vrot") is not None], 0))))
    mom = [r for r in jl(d / f"m_{name}.log") if "com" in r]
    if mom:
        out.update(com_largest=max(r["com"] for r in mom), com_end=mom[-1]["com"], rot_end=mom[-1].get("rot"))
        try:
            s = [x for x in (d / f"m_{name}.log").read_text().splitlines() if "kept frames |" in x][-1]
            g = re.search(r"net / gross angular move: largest ([\d.e+-]+), mean ([\d.e+-]+)", s)
            out["ang_ng_mean"] = float(g.group(2))
        except (OSError, IndexError, AttributeError):
            pass
    t3 = jl(d / f"t3end_{name}.log")
    if t3:
        out["yard_sil"] = t3[-1]["exterior"]["sil"]
    e3 = [r for r in jl(d / f"e3end_{name}.log") if r.get("state") not in ("target", "own")]
    if e3 and "front" in e3[-1].get("to_target", {}):
        out["front"] = 1 - e3[-1]["to_target"]["front"]["iou"]
    return out


def vis(path):
    rows = jl(path)
    rng = [r for r in rows if r.get("window") == "range"]
    w = [r for r in rows if r.get("window") != "range"]
    if not rng:
        return None
    out = dict(unique_side=rng[0]["side_blob"], unique_blob=rng[0]["blob"])
    for lo, hi, name in BANDS:
        rr = [r for r in w if lo <= r["window"] <= hi]
        out[name] = (float(np.mean([r["side_blob"] for r in rr])) if rr else np.nan,
                     float(np.mean([r["blob"] for r in rr])) if rr else np.nan)
    return out


def vpath(mesh, tag):
    if mesh == "dragon" and tag in ("SL", "SP"):
        return G / "d148" / f"vfull_{tag}.jsonl"
    if mesh == "bunny" and tag in ("SL", "SL2", "SP", "SP2"):
        return NEW / f"vfull_bunny_{tag}.jsonl"
    return NEW / f"vfull_{mesh}_{tag}.jsonl"


def band(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    lo, hi = min(vals), max(vals)
    return lo - (hi - lo), hi + (hi - lo)


fails, rows = [], {}
for (mesh, tag), (b1, b2, b3) in REF.items():
    print(f"\n== {mesh} {tag} (base {'/'.join(b1)}; band (2) {'/'.join(b2)}; band (3)/(4) {'/'.join(b3)})")
    r = run(NEW, mesh, tag)
    if r is None:
        print("  no JSON yet")
        continue
    rows[(mesh, tag)] = r
    base = {t: run(BASE, mesh, t) for t in set(b1 + b2 + b3)}
    # (1)
    v = vis(vpath(mesh, tag))
    vb = [vis(vpath(mesh, t)) for t in b1]
    if v is None or any(x is None for x in vb):
        print("  (1) visible pieces: records missing")
    else:
        us = float(np.mean([x["unique_side"] for x in vb]))
        w16 = float(np.mean([x["1.5-6 s"][0] for x in vb]))
        ok1 = v["unique_side"] <= 0.5 * us and v["1.5-6 s"][0] <= 0.5 * w16
        print(f"  (1) unique side-view blob particles {v['unique_side']} (base {us:.1f}, half {0.5 * us:.1f}); side blobs a "
              f"window at 1.5-6 s {v['1.5-6 s'][0]:.2f} (base {w16:.2f}) {'ok' if ok1 else 'FAIL'}; unique blobs "
              f"{v['unique_blob']} (base {np.mean([x['unique_blob'] for x in vb]):.1f}); by band side / all: " +
              ", ".join(f"{n} {v[n][0]:.1f} / {v[n][1]:.1f} (base {np.mean([x[n][0] for x in vb]):.1f} / "
                        f"{np.mean([x[n][1] for x in vb]):.1f})" for _, _, n in BANDS))
        if not ok1:
            fails.append(f"(1) {mesh} {tag}")
    # (2)
    out2 = []
    for k in range(min(9, len(r["early"]))):
        for j, key in enumerate(("d_vol", "d_sil", "kin")):
            vals = [base[t]["early"][k][j] for t in b2 if base[t] and len(base[t]["early"]) > k]
            if len(vals) < 2 or r["early"][k][j] is None:
                continue
            lo, hi = band(vals)
            if not lo <= r["early"][k][j] <= hi:
                out2.append(f"w{k + 1} {key} {r['early'][k][j]:.3e} outside [{lo:.3e}, {hi:.3e}]")
    print(f"  (2) windows 1-9: outside the band: {out2 if out2 else 'none'}")
    if out2:
        fails.append(f"(2) {mesh} {tag}")
    # (3)
    lim = {}
    for key in ("sil", "chamfer", "thin"):
        lim[key] = band([base[t][key] for t in b3 if base[t]])
    ok3 = (r["sil"] >= lim["sil"][0] and r["chamfer"] <= lim["chamfer"][1] and r["thin"] <= lim["thin"][1])
    ok_rest = r["kin"] <= REST[mesh]
    print(f"  (3) silIoU {r['sil']:.4f} (>= {lim['sil'][0]:.4f}) chamfer {r['chamfer']:.4f} (<= {lim['chamfer'][1]:.4f}) "
          f"thin {100 * r['thin']:.2f} % (<= {100 * lim['thin'][1]:.2f}) {'ok' if ok3 else 'FAIL'} | kin {r['kin']:.2e} "
          f"(<= {REST[mesh]:.1e}) {'ok' if ok_rest else 'FAIL'} | yardstick sil {r.get('yard_sil', np.nan):.3e} | display "
          f"front 1-IoU {r.get('front', np.nan):.4f}")
    for k, ok in (("(3) end", ok3), ("(3) rest", ok_rest)):
        if not ok:
            fails.append(f"{k} {mesh} {tag}")
    # (4)
    out4 = []
    for key in ("com_largest", "com_end", "rot_end", "ang_ng_mean", "body_net", "body_rot_deg"):
        vals = [base[t].get(key) for t in b3 if base[t]]
        if r.get(key) is None or len([x for x in vals if x is not None]) < 2:
            continue
        lo, hi = band(vals)
        mark = "" if r[key] <= hi else " BEYOND"
        out4.append(f"{key} {r[key]:.3e} (<= {hi:.3e}){mark}")
        if mark:
            fails.append(f"(4) {mesh} {tag} {key}")
    print(f"  (4) " + "; ".join(out4) + f" | lambda median {r['lam']:.3e} (base " +
          ", ".join(f"{t} {base[t]['lam']:.3e}" for t in b3 if base[t]) + f"), g_share {r['share']:.3f} (base " +
          ", ".join(f"{t} {base[t]['share']:.3f}" for t in b3 if base[t]) + ")")
    # (5)
    sb = [base[t]["s_att"] for t in b1 if base[t] and base[t]["s_att"]]
    ok5 = r["s_att"] is not None and r["s_att"] <= 1.05 * float(np.mean(sb))
    print(f"  (5) {r['s_att']:.2f} s an attempt (<= {1.05 * float(np.mean(sb)):.2f}) {'ok' if ok5 else 'FAIL'}; "
          f"{r['windows']} windows, {r['attempts']} attempts, {r['minutes']:.1f} min")
    if not ok5:
        fails.append(f"(5) {mesh} {tag}")
for mesh in ("dragon", "bunny"):
    a, t = rows.get((mesh, "NL")), rows.get((mesh, "NP"))
    if a and t and a.get("yard_sil") is not None and t.get("yard_sil") is not None:
        lead = a["yard_sil"] < t["yard_sil"] and a.get("front", 1) < t.get("front", 1)
        print(f"\n(3) render lead {mesh}: yardstick sil {a['yard_sil']:.3e} vs {t['yard_sil']:.3e}, display front "
              f"{a.get('front', np.nan):.4f} vs {t.get('front', np.nan):.4f}: {'ahead' if lead else 'NOT ahead'}")
        if not lead:
            fails.append(f"(3) lead {mesh}")
print("\n== items failing: " + (", ".join(fails) if fails else "none"))
