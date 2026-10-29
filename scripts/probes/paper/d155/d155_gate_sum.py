"""d155_gate_sum.py GATE_DIR OUT_DIR -- D155 option A's 40k gate: the pre-registered criteria (2026-10-17 14:10 CDT) and
the tables from gate.jsonl (d155_gate_read.py over the gate's runs) and gate_b3_tag.jsonl (B3's tag runs, the run-to-run
check); one figure (fig_gate.png): cumulative move and window-end kinetic energy over time, bunny / dragon / homer,
render arm, the tag's three seeds against A1 and A3."""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GD, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
_src = f"{GD}/gate_all.jsonl" if os.path.exists(f"{GD}/gate_all.jsonl") else f"{GD}/gate.jsonl"
R = {r["name"]: r for r in (json.loads(l) for l in open(_src))}
B3 = {r["name"]: r for r in (json.loads(l) for l in open(f"{os.path.dirname(GD)}/gate_b3_tag.jsonl"))}
MESH = ("bunny", "dragon", "teapot", "armadilo", "homer")
SEEDS = (97, 99, 101)
md = open(f"{OUT}/gate_tables.md", "w")


def p(s=""):
    print(s)
    md.write(s + "\n")


def g(r, path):
    x = r
    for k in path.split("."):
        x = None if x is None else x.get(k)
    return x


def tag(m, arm, path, src=None):
    """the tag's range (pre-registered): its three re-run seeds and B3's three where B3 has the measure"""
    srcs = [R, B3] if src is None else [src]
    v = [g(s_.get(f"{m}40k_{arm}{s}"), path) for s_ in srcs for s in SEEDS]
    return [x for x in v if x is not None]


def rng(v, nd=3):
    return f"{min(v):.{nd}f}–{max(v):.{nd}f}" if v else "–"


def val(m, arm, path, nd=3):
    x = g(R.get(f"{m}40k_{arm}97"), path)
    return "–" if x is None else f"{x:.{nd}f}"


# ---------------- criteria (render arm of an arm: A3 for D155, A3G for D155b) ----------------
yes = lambda b: "–" if b is None else ("yes" if b else "NO")  # noqa: E731


def judge(arm, item5, item6=False):
  crit = {}
  for m in MESH:
    a3 = R.get(f"{m}40k_{arm}L97")
    if a3 is None:
        continue
    c = {}
    t1 = tag(m, "V", "move_after_1s")
    c["1 timing"] = bool(t1) and a3["move_after_1s"] >= max(t1) + 0.15
    q = {}
    for path, better in (("end.ind_sil", "high"), ("end.yard_sil", "low"), ("end.yard_pbr", "low"),
                         ("end.display", "low"), ("end.thin", "low")):
        tv, x = tag(m, "V", path), g(a3, path)
        q[path] = None if (not tv or x is None) else (x >= min(tv) if better == "high" else x <= max(tv))
    c["2 end"] = q
    p3 = R.get(f"{m}40k_{arm}P97")
    c["2 lead"] = (None if p3 is None or g(a3, "end.yard_sil") is None or g(p3, "end.yard_sil") is None
                   else g(a3, "end.yard_sil") < g(p3, "end.yard_sil"))
    tk = tag(m, "V", "kin_end")
    c["3 rest"] = bool(tk) and a3["kin_end"] <= 1.25 * max(tk)
    tp = tag(m, "V", "kin_peak")
    c["4 peak kinetic"] = bool(tp) and a3["kin_peak"] <= 0.5 * min(tp)
    if item5:
        ta, tb = tag(m, "V", "apart_w3_12.max"), tag(m, "V", "visible_w3_10.bump_max")
        c["5 apart"] = None if not ta or g(a3, "apart_w3_12.max") is None else g(a3, "apart_w3_12.max") <= max(ta)
        c["5 bumps"] = (None if not tb or g(a3, "visible_w3_10.bump_max") is None
                        else g(a3, "visible_w3_10.bump_max") <= max(tb))
    if item6:
        jr = g(a3, "J_ramp_max")
        c["6a ramp J"] = None if jr is None else jr <= 1.03
        tm, tq = tag(m, "V", "J_end.mean"), tag(m, "V", "J_end.p99")
        jm, jq = g(a3, "J_end.mean"), g(a3, "J_end.p99")
        c["6b end J mean"] = None if not tm or jm is None else (min(tm) <= jm <= max(tm))
        c["6b end J p99"] = None if not tq or jq is None else jq <= max(tq)
    crit[m] = c
  p(f"## Criteria ({arm} render arm against the tag's three seeds, V)")
  p("| mesh | 1 timing | 2 IoU (ind.) | 2 yardstick sil. | 2 yardstick shading | 2 display | 2 thin | 2 lead over its twin | 3 at rest | 4 peak kinetic |"
    + (" 5 apart w3-12 | 5 visible bumps w3-10 |" if item5 else "")
    + (" 6a ramp J | 6b end J mean | 6b end J p99 |" if item6 else ""))
  p("|---|---|---|---|---|---|---|---|---|---|" + ("---|---|" if item5 else "") + ("---|---|---|" if item6 else ""))
  for m, c in crit.items():
    q = c["2 end"]
    p(f"| {m} | {yes(c['1 timing'])} | {yes(q['end.ind_sil'])} | {yes(q['end.yard_sil'])} | {yes(q['end.yard_pbr'])} | "
      f"{yes(q['end.display'])} | {yes(q['end.thin'])} | {yes(c['2 lead'])} | {yes(c['3 rest'])} | {yes(c['4 peak kinetic'])} |"
      + (f" {yes(c['5 apart'])} | {yes(c['5 bumps'])} |" if item5 else "")
      + (f" {yes(c['6a ramp J'])} | {yes(c['6b end J mean'])} | {yes(c['6b end J p99'])} |" if item6 else ""))
  n_ok = lambda f: sum(1 for c in crit.values() if f(c))  # noqa: E731
  verdict = {"1": n_ok(lambda c: c["1 timing"]),
             **{k: n_ok(lambda c, k=k: c["2 end"][k]) for k in ("end.ind_sil", "end.yard_sil", "end.yard_pbr",
                                                                 "end.display", "end.thin")},
             "lead": n_ok(lambda c: c["2 lead"]), "3": n_ok(lambda c: c["3 rest"]), "4": n_ok(lambda c: c["4 peak kinetic"])}
  if item5:
      verdict.update({"5 apart": n_ok(lambda c: c["5 apart"]), "5 bumps": n_ok(lambda c: c["5 bumps"])})
  if item6:
      verdict.update({k: n_ok(lambda c, k=k: c[k]) for k in ("6a ramp J", "6b end J mean", "6b end J p99")})
  p()
  p("counts of 5: " + json.dumps(verdict))
  passed = all(v >= 4 for v in verdict.values()) and len(crit) == 5
  p(f"verdict: {'PASS' if passed else 'FAIL'} (each item on >= 4 of 5)")
  p()


judge("A3", False)
if any(k.endswith("A3GL97") for k in R):
    judge("A3G", True)
if any(k.endswith("A3CL97") for k in R):
    judge("A3C", True, True)
# ---------------- tables ----------------
p("## Timing (render arm L / its twin P; the tag: three seeds V / W)")
p("| mesh | move after 1 s: tag L / A1 L / A3 L | after 3 s: tag / A1 / A3 | physics P90 (s, final target) tag / A1 / A3 | render P90 (s) | A_lin tag / A1 / A3 | A_log tag / A1 / A3 | T_e A1 / A3 (s) |")
p("|---|---|---|---|---|---|---|---|")
for m in MESH:
    p(f"| {m} | {rng(tag(m, 'V', 'move_after_1s'))} / {val(m, 'A1L', 'move_after_1s')} / {val(m, 'A3L', 'move_after_1s')} | "
      f"{rng(tag(m, 'V', 'move_after_3s'))} / {val(m, 'A1L', 'move_after_3s')} / {val(m, 'A3L', 'move_after_3s')} | "
      f"{rng(tag(m, 'V', 'lin.P90_s'), 2)} / {val(m, 'A1L', 'lin.P90_s', 2)} / {val(m, 'A3L', 'lin.P90_s', 2)} | "
      f"{rng(tag(m, 'V', 'lin.R90_s'), 2)} / {val(m, 'A1L', 'lin.R90_s', 2)} / {val(m, 'A3L', 'lin.R90_s', 2)} | "
      f"{rng(tag(m, 'V', 'lin.A'))} / {val(m, 'A1L', 'lin.A')} / {val(m, 'A3L', 'lin.A')} | "
      f"{rng(tag(m, 'V', 'log.A'))} / {val(m, 'A1L', 'log.A')} / {val(m, 'A3L', 'log.A')} | "
      f"{val(m, 'A1L', 'T_e', 2)} / {val(m, 'A3L', 'T_e', 2)} |")
p()
p("## The shared shape on the ramp: over [0, T_e] the raw cosine of the physics and render gradients, the render's share of the step and of the merit, capped windows (the tag over the same span)")
p("| mesh | raw cos: tag over [0, 1.19 s] and over [0, 3.26 s] / A1 / A3 | g_share A1 / A3 | merit share tag / A1 / A3 | capped A1 / A3 | raw cos after the ramp tag / A3 |")
p("|---|---|---|---|---|---|")
def span_cos(r, t1):
    """the mean raw cosine of the windows ending at or before t1 (the tag over A3's ramp span)"""
    c = r["curves"]
    v = [x for t, x in zip(c["t"], c["raw_cos"]) if t <= t1 + 1e-9 and x is not None]
    return float(np.mean(v)) if v else None


for m in MESH:
    a3 = R.get(f"{m}40k_A3L97")
    t3 = a3["T_e"] if a3 else 3.26
    tc = [x for x in (span_cos(R[f"{m}40k_V{s}"], t3) for s in SEEDS if f"{m}40k_V{s}" in R) if x is not None]
    p(f"| {m} | {rng(tag(m, 'V', 'ramp.raw_cos'))} and {rng(tc)} / {val(m, 'A1L', 'ramp.raw_cos')} / {val(m, 'A3L', 'ramp.raw_cos')} | "
      f"{val(m, 'A1L', 'ramp.g_share')} / {val(m, 'A3L', 'ramp.g_share')} | {rng(tag(m, 'V', 'ramp.merit_share'))} / "
      f"{val(m, 'A1L', 'ramp.merit_share')} / {val(m, 'A3L', 'ramp.merit_share')} | {val(m, 'A1L', 'ramp.capped', 0)} / "
      f"{val(m, 'A3L', 'ramp.capped', 0)} | {rng(tag(m, 'V', 'after_raw_cos'))} / {val(m, 'A3L', 'after_raw_cos')} |")
p()
p("## End quality against the independent 40k sample (render arm; twin in the second row)")
p("| mesh, arm | IoU ind. tag / A1 / A3 | yardstick sil. (1e-3) | yardstick shading (1e-3) | display 1 - IoU | thin | own IoU |")
p("|---|---|---|---|---|---|---|")
for m in MESH:
    for t_, a1, a3 in (("V", "A1L", "A3L"), ("W", "A1P", "A3P")):
        row = []
        for path, k, nd in (("end.ind_sil", 1, 4), ("end.yard_sil", 1e3, 3), ("end.yard_pbr", 1e3, 3),
                            ("end.display", 1, 4), ("end.thin", 1, 3), ("end.sil_iou", 1, 4)):
            tv = [x * k for x in tag(m, t_, path)]
            x1, x3 = g(R.get(f"{m}40k_{a1}97"), path), g(R.get(f"{m}40k_{a3}97"), path)
            row.append(f"{rng(tv, nd)} / {'–' if x1 is None else f'{x1 * k:.{nd}f}'} / "
                       f"{'–' if x3 is None else f'{x3 * k:.{nd}f}'}")
        p(f"| {m} {'L' if t_ == 'V' else 'P'} | " + " | ".join(row) + " |")
p()
p("## Motion, rest, momentum, floating pieces, cost (render arm)")
p("| mesh | peak kinetic tag / A1 / A3 (time s) | delivered kinetic (1e-5) | com drift (1e-3 wu) | abs(L) max | apart w3-6 / w3-12 max: tag / A1 / A3 | visible apart / blobs w3-10 | windows | minutes | build s a ramp window (A3) |")
p("|---|---|---|---|---|---|---|---|---|---|")
for m in MESH:
    a1, a3 = R.get(f"{m}40k_A1L97") or {}, R.get(f"{m}40k_A3L97") or {}
    ta36, ta312 = tag(m, "V", "apart_w3_6.max"), tag(m, "V", "apart_w3_12.max")
    tv, tb = tag(m, "V", "visible_w3_10.apart_max"), tag(m, "V", "visible_w3_10.blob_max")
    p(f"| {m} | {rng(tag(m, 'V', 'kin_peak'), 2)} / {g(a1, 'kin_peak') or float('nan'):.2f} / "
      f"{g(a3, 'kin_peak') or float('nan'):.2f} ({g(a3, 'kin_peak_t') or float('nan'):.2f}) | "
      f"{rng([x * 1e5 for x in tag(m, 'V', 'kin_end')], 2)} / {(g(a1, 'kin_end') or float('nan')) * 1e5:.2f} / "
      f"{(g(a3, 'kin_end') or float('nan')) * 1e5:.2f} | {rng([x * 1e3 for x in tag(m, 'V', 'com_drift')], 3)} / "
      f"{(g(a1, 'com_drift') or float('nan')) * 1e3:.3f} / {(g(a3, 'com_drift') or float('nan')) * 1e3:.3f} | "
      f"{rng(tag(m, 'V', 'L_max'), 1)} / {g(a1, 'L_max') or float('nan'):.1f} / {g(a3, 'L_max') or float('nan'):.1f} | "
      f"{ta36} {ta312} / {g(a1, 'apart_w3_6.max')} {g(a1, 'apart_w3_12.max')} / {g(a3, 'apart_w3_6.max')} "
      f"{g(a3, 'apart_w3_12.max')} | {tv} {tb} / {g(a1, 'visible_w3_10.apart_max')} {g(a1, 'visible_w3_10.blob_max')} / "
      f"{g(a3, 'visible_w3_10.apart_max')} {g(a3, 'visible_w3_10.blob_max')} | {rng(tag(m, 'V', 'windows'), 0)} / "
      f"{g(a1, 'windows')} / {g(a3, 'windows')} | {rng(tag(m, 'V', 'minutes'), 2)} / {g(a1, 'minutes') or float('nan'):.2f} / "
      f"{g(a3, 'minutes') or float('nan'):.2f} | {g(a3, 'build_s_mean') or float('nan'):.2f} |")
p()
p("## Run-to-run: the tag re-run against B3 (seed 97, render arm)")
p("| mesh | windows B3 / re-run | IoU ind. B3 / re-run | move after 1 s B3 / re-run | peak kinetic B3 / re-run |")
p("|---|---|---|---|---|")
for m in MESH:
    b, r = B3.get(f"{m}40k_V97") or {}, R.get(f"{m}40k_V97") or {}
    p(f"| {m} | {b.get('windows')} / {r.get('windows')} | {g(b, 'end.ind_sil')} / {g(r, 'end.ind_sil')} | "
      f"{b.get('move_after_1s')} / {r.get('move_after_1s')} | {b.get('kin_peak')} / {r.get('kin_peak')} |")
p()
p("## Diagnostic (added after part 1 was read, 19:05 CDT; reported, not judged): u off (--u_off) with and without the ramp, seed 97")
p("| run | move after 1 s | peak kinetic | apart w3-6 / w3-12 max | visible apart / blobs w3-10 | IoU ind. | yardstick sil. (1e-3) | display | thin | windows | minutes |")
p("|---|---|---|---|---|---|---|---|---|---|---|")
for m in ("bunny", "dragon"):
    for x in ("V97", "U97", "A3L97", "A3U97"):
        r = R.get(f"{m}40k_{x}")
        if r is None:
            continue
        p(f"| {m} {x} | {r['move_after_1s']:.3f} | {r['kin_peak']:.3f} | {g(r, 'apart_w3_6.max')} / {g(r, 'apart_w3_12.max')} | "
          f"{g(r, 'visible_w3_10.apart_max')} / {g(r, 'visible_w3_10.blob_max')} | {g(r, 'end.ind_sil') or float('nan'):.4f} | "
          f"{(g(r, 'end.yard_sil') or float('nan')) * 1e3:.3f} | {g(r, 'end.display') or float('nan'):.4f} | "
          f"{g(r, 'end.thin') or float('nan'):.3f} | {r['windows']} | {r['minutes']:.2f} |")
if any(k.endswith("A3GL97") for k in R):
    p()
    p("## D155b: A3G (u reads the final target) against A3 and the tag, render arm (twin rows: P)")
    p("| mesh | move after 1 s tag / A3 / A3G | physics / render P90 s A3 / A3G | ramp cos A3 / A3G | u gate open, mean over 0-1 s: tag / A3 / A3G; first window over half (s) | apart w3-12 tag / A3 / A3G | visible bumps / blobs w3-10 tag / A3 / A3G | peak kinetic A3 / A3G | delivered kinetic 1e-5 tag / A3G | abs(L) max tag / A3G | minutes tag / A3 / A3G | build s A3 / A3G |")
    p("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in MESH:
        a, b = R.get(f"{m}40k_A3L97") or {}, R.get(f"{m}40k_A3GL97") or {}
        f_ = lambda r, k, nd=3: "–" if g(r, k) is None else f"{g(r, k):.{nd}f}"  # noqa: E731
        p(f"| {m} | {rng(tag(m, 'V', 'move_after_1s'))} / {f_(a, 'move_after_1s')} / {f_(b, 'move_after_1s')} | "
          f"{f_(a, 'lin.P90_s', 2)} {f_(a, 'lin.R90_s', 2)} / {f_(b, 'lin.P90_s', 2)} {f_(b, 'lin.R90_s', 2)} | "
          f"{f_(a, 'ramp.raw_cos')} / {f_(b, 'ramp.raw_cos')} | {rng(tag(m, 'V', 'u_gate_1s', R), 2)} / {f_(a, 'u_gate_1s', 2)} / "
          f"{f_(b, 'u_gate_1s', 2)}; {f_(b, 'u_gate_half_t', 2)} | {tag(m, 'V', 'apart_w3_12.max')} / {g(a, 'apart_w3_12.max')} / "
          f"{g(b, 'apart_w3_12.max')} | {tag(m, 'V', 'visible_w3_10.bump_max')} {tag(m, 'V', 'visible_w3_10.blob_max', R)} / "
          f"{g(a, 'visible_w3_10.bump_max')} {g(a, 'visible_w3_10.blob_max')} / {g(b, 'visible_w3_10.bump_max')} "
          f"{g(b, 'visible_w3_10.blob_max')} | {f_(a, 'kin_peak')} / {f_(b, 'kin_peak')} | "
          f"{rng([x * 1e5 for x in tag(m, 'V', 'kin_end')], 2)} / {(g(b, 'kin_end') or float('nan')) * 1e5:.2f} | "
          f"{rng(tag(m, 'V', 'L_max'), 1)} / {f_(b, 'L_max', 1)} | {rng(tag(m, 'V', 'minutes'), 2)} / {f_(a, 'minutes', 2)} / "
          f"{f_(b, 'minutes', 2)} | {f_(a, 'build_s_mean', 2)} / {f_(b, 'build_s_mean', 2)} |")
    p()
    p("| mesh, arm | IoU ind. tag / A3 / A3G | yardstick sil. 1e-3 | yardstick shading 1e-3 | display | thin |")
    p("|---|---|---|---|---|---|")
    for m in MESH:
        for t_, x in (("V", "L"), ("W", "P")):
            row = []
            for path, k, nd in (("end.ind_sil", 1, 4), ("end.yard_sil", 1e3, 3), ("end.yard_pbr", 1e3, 3),
                                ("end.display", 1, 4), ("end.thin", 1, 3)):
                tv = [v * k for v in tag(m, t_, path)]
                va, vb = g(R.get(f"{m}40k_A3{x}97"), path), g(R.get(f"{m}40k_A3G{x}97"), path)
                row.append(f"{rng(tv, nd)} / {'–' if va is None else f'{va * k:.{nd}f}'} / "
                           f"{'–' if vb is None else f'{vb * k:.{nd}f}'}")
            p(f"| {m} {x} | " + " | ".join(row) + " |")
if any(k.endswith("A3CL97") for k in R):
    def jmax(r, t1):
        c = (r or {}).get("curves") or {}
        v = [j for t, j in zip(c.get("t", []), c.get("J", [])) if j is not None and t <= t1 + 1e-9]
        return max(v) if v else None

    f_ = lambda r, k, nd=3: "–" if g(r, k) is None else f"{g(r, k):.{nd}f}"  # noqa: E731
    p()
    p("## D155c: A3C (the targets on the way at the body's density) against A3G and the tag, render arm")
    p("| mesh | ramp's largest mean J: tag over [0, T_e] / A3G / A3C | end mean J tag / A3G / A3C | end J p99 tag / A3G / A3C | k_s (A3C) | target volume before the match | local volume p05..p95: final target / A3C on the way |")
    p("|---|---|---|---|---|---|---|")
    for m in MESH:
        a, b = R.get(f"{m}40k_A3GL97") or {}, R.get(f"{m}40k_A3CL97") or {}
        te = b.get("T_e", 3.26)
        tj = [x for x in (jmax(R.get(f"{m}40k_V{s}"), te) for s in SEEDS) if x is not None]
        mo = b.get("morph") or {}
        fq = mo.get("final_vol_q") or [None, None, None]
        p(f"| {m} | {rng(tj, 4)} / {f_(a, 'J_ramp_max', 4)} / {f_(b, 'J_ramp_max', 4)} | "
          f"{rng(tag(m, 'V', 'J_end.mean'), 4)} / {f_(a, 'J_end.mean', 4)} / {f_(b, 'J_end.mean', 4)} | "
          f"{rng(tag(m, 'V', 'J_end.p99'), 3)} / {f_(a, 'J_end.p99')} / {f_(b, 'J_end.p99')} | "
          f"{mo.get('k')} | {mo.get('vol_ratio')} | {fq[0]}..{fq[2]} / {mo.get('vol_q_p05')}..{mo.get('vol_q_p95')} |")
    p()
    p("| mesh | move after 1 s tag / A3G / A3C | physics / render P90 s A3G / A3C | ramp cos A3G / A3C | apart w3-12 A3G / A3C | visible bumps w3-10 A3G / A3C | peak kinetic A3G / A3C | delivered kinetic 1e-5 tag / A3C | minutes tag / A3G / A3C |")
    p("|---|---|---|---|---|---|---|---|---|")
    for m in MESH:
        a, b = R.get(f"{m}40k_A3GL97") or {}, R.get(f"{m}40k_A3CL97") or {}
        p(f"| {m} | {rng(tag(m, 'V', 'move_after_1s'))} / {f_(a, 'move_after_1s')} / {f_(b, 'move_after_1s')} | "
          f"{f_(a, 'lin.P90_s', 2)} {f_(a, 'lin.R90_s', 2)} / {f_(b, 'lin.P90_s', 2)} {f_(b, 'lin.R90_s', 2)} | "
          f"{f_(a, 'ramp.raw_cos')} / {f_(b, 'ramp.raw_cos')} | {g(a, 'apart_w3_12.max')} / {g(b, 'apart_w3_12.max')} | "
          f"{g(a, 'visible_w3_10.bump_max')} / {g(b, 'visible_w3_10.bump_max')} | {f_(a, 'kin_peak')} / {f_(b, 'kin_peak')} | "
          f"{rng([x * 1e5 for x in tag(m, 'V', 'kin_end')], 2)} / {(g(b, 'kin_end') or float('nan')) * 1e5:.2f} | "
          f"{rng(tag(m, 'V', 'minutes'), 2)} / {f_(a, 'minutes', 2)} / {f_(b, 'minutes', 2)} |")
    p()
    p("| mesh, arm | IoU ind. tag / A3G / A3C | yardstick sil. 1e-3 | yardstick shading 1e-3 | display | thin |")
    p("|---|---|---|---|---|---|")
    for m in MESH:
        for t_, x in (("V", "L"), ("W", "P")):
            row = []
            for path, k, nd in (("end.ind_sil", 1, 4), ("end.yard_sil", 1e3, 3), ("end.yard_pbr", 1e3, 3),
                                ("end.display", 1, 4), ("end.thin", 1, 3)):
                tv = [v * k for v in tag(m, t_, path)]
                va, vb = g(R.get(f"{m}40k_A3G{x}97"), path), g(R.get(f"{m}40k_A3C{x}97"), path)
                row.append(f"{rng(tv, nd)} / {'–' if va is None else f'{va * k:.{nd}f}'} / "
                           f"{'–' if vb is None else f'{vb * k:.{nd}f}'}")
            p(f"| {m} {x} | " + " | ".join(row) + " |")
md.close()

# ---------------- figure ----------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "savefig.dpi": 160, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6, "legend.frameon": False,
                     "legend.fontsize": 7.5})
show = [m for m in ("bunny", "dragon", "homer") if f"{m}40k_A3L97" in R]
if show:
    fig, ax = plt.subplots(2, len(show), figsize=(4.6 * len(show), 7.2), squeeze=False)
    for j, m in enumerate(show):
        for name, col, lab, lw in ([(f"{m}40k_V{s}", MUTED, "the tag (three seeds)" if s == 97 else None, 1.0)
                                    for s in SEEDS] + [(f"{m}40k_A1L97", AQUA, "A1: ramp over the run's own T_e", 1.8),
                                                       (f"{m}40k_A3L97", ORANGE, "A3: ramp over the 300k body's T_e", 2.0),
                                                       (f"{m}40k_A3GL97", BLUE, "A3G: A3, u reads the final target", 2.0)]):
            r = R.get(name)
            if r is None or "curves" not in r:
                continue
            c = r["curves"]
            t = np.array(c["t"])
            mv = np.cumsum(c["move"])
            ax[0, j].plot(t, mv / mv[-1], color=col, lw=lw, label=lab)
            ax[1, j].semilogy(t, c["kin"], color=col, lw=lw, label=lab)
        for i in (0, 1):
            for nm, ls in (("A1L", ":"), ("A3L", "--")):
                r = R.get(f"{m}40k_{nm}97")
                if r:
                    ax[i, j].axvline(r["T_e"], color=AQUA if nm == "A1L" else ORANGE, ls=ls, lw=0.9)
            ax[i, j].set_xlim(0, 8)
            ax[i, j].set_xlabel("simulated time (s)")
        ax[0, j].set_ylabel("share of the run's summed movement done")
        ax[1, j].set_ylabel("kinetic energy at the window end")
        ax[0, j].set_title(f"{m} 40k: when the body moves (lines: where the ramp ends)", fontsize=9)
        ax[1, j].set_title(f"{m} 40k: how fast it moves", fontsize=9)
        ax[0, j].legend(loc="lower right")
    fig.suptitle("D155 option A at 40k: when the body moves and how fast, the tag against A1 (the run's own T_e) and A3 "
                 "(the 300k body's T_e)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_gate.png")
print("written", OUT)
