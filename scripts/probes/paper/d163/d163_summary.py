"""d163_summary.py OUT_MD READ_JSON [SPREAD_JSON] -- D163: the pre-registered verdict (P1-P5 per arm against the twin of
its seed) and the tables from d163_read.py's output (CPU). The seed spread is |T_97 - T_99| from the read itself when
both twins are in it; else (300k without T_99) SPREAD_JSON (the 40k read): its spread relative to its T_97, applied to
this T_97 (the pre-registered rule). Arm names as the user reads them."""
import json
import sys

import numpy as np

NAMES = {"T": "dF_c + layer u", "TC": "dF_c + layer u (placement at the cell)",
         "R": "dF_c + layer u + render loss (full), the rule's weight",
         "H": "dF_c + layer u + render loss (full), weight held (--lambda_ema 0)",
         "H2": "the same as H, pictures at 2x resolution (--render_res_hi)",
         "F": "dF_c + layer u + render loss (full), weight on its floor (D162 F)",
         "FC": "D162 F + placement at the cell (FC)"}
out, rpath = sys.argv[1], sys.argv[2]
r = json.load(open(rpath))
sc, refs, runs, pairs = r["scene"], r["refs"], r["runs"], r["pairs"]
seeds = sorted({int(k[-2:]) for k in runs})


def g(lab, *path):
    v = runs[lab]["end"]
    for p in path:
        v = v[p]
    return v


def ceil(s, key="relief"):
    return float(np.mean([refs[f"relief_s1{s}"][key]["A"], refs[f"relief_s2{s}"][key]["A"]])), \
        float(np.mean([refs[f"relief_s1{s}"][key]["corr"], refs[f"relief_s2{s}"][key]["corr"]]))


sides = [k[len("relief_"):] for k in refs[f"relief_s1{seeds[0]}"] if k.startswith("relief_")]
L = [f"## {sc['scene']} (N = {sc['N']})", "",
     "| state | A (pitch) | share of ceiling | corr | " + " | ".join(f"{s} A / corr" for s in sides) + " | display A / corr |",
     "|---|---|---|---|" + "---|" * len(sides) + "---|"]
for k in ("smooth_tgt", "smooth_ind", "relief_s1", "relief_s2"):
    for s in seeds:
        v = refs[f"{k}{s}"]
        d = v.get("display", {})
        L.append(f"| ref {k} {s} | {v['relief']['A']:+.3f} | {v['relief']['A'] / ceil(s)[0]:+.3f} | {v['relief']['corr']:+.3f} | "
                 + " | ".join(f"{v['relief_' + sd]['A']:+.3f} / {v['relief_' + sd]['corr']:+.3f}" for sd in sides)
                 + f" | {d.get('A', float('nan')):+.3f} / {d.get('corr', float('nan')):+.3f} |")
for lab in runs:
    e = runs[lab]["end"]
    s = int(lab[-2:])
    L.append(f"| **{lab}** end | {e['relief']['A']:+.3f} | {e['relief']['A'] / ceil(s)[0]:+.3f} | {e['relief']['corr']:+.3f} | "
             + " | ".join(f"{e['relief_' + sd]['A']:+.3f} / {e['relief_' + sd]['corr']:+.3f}" for sd in sides)
             + f" | {e['display']['A']:+.3f} / {e['display']['corr']:+.3f} |")
L += ["", "| run | arm | letters' shading err vs relief mesh | all px | held-out 1-IoU vs smooth | Chamfer-L1 vs smooth (a) | "
      "F@0.5 | run sil_iou | thin_uncovered | v_mean end | kin end | pieces >= 10 | windows | wall (min) | lambda med10 | g_share last |",
      "|" + "---|" * 16]
for lab in runs:
    e = runs[lab]["end"]
    im, ge, rc, pc = e["images"], e["geometry"], e["records"], e["pieces"]
    L.append(f"| {lab} | {NAMES.get(lab.rsplit('_', 1)[0], '')} | {im['relief']['mse_pattern']:.3e} | {im['relief']['mse_all']:.3e} | "
             f"{ge['sil_iou_err_held']:.4f} | {ge['smooth']['chamfer_l1_a']:.3f} | {ge['smooth']['f0.5']:.3f} | {rc['sil_iou']:.4f} | "
             f"{rc['thin_uncovered']:.4f} | {rc['v_mean']:.2e} | {rc['kin']:.2e} | {pc['pieces_ge10']} | {rc['windows']} | "
             f"{(rc['wall_s'] or 0) / 60:.1f} | {rc['lam_med10']:.2e} | {rc['g_share_last']} |")
# the spread
METRICS = (("held 1-IoU", ("geometry", "sil_iou_err_held"), +1), ("Chamfer-L1", ("geometry", "smooth", "chamfer_l1_a"), +1),
           ("sil_iou", ("records", "sil_iou"), -1), ("thin_uncovered", ("records", "thin_uncovered"), +1),
           ("letters' shading", ("images", "relief", "mse_pattern"), +1))
spread = {}
if "T_97" in runs and "T_99" in runs:
    for key, path, _ in METRICS:
        spread[key] = abs(g("T_97", *path) - g("T_99", *path))
    spread_src = "T_97 vs T_99 in this read"
elif len(sys.argv) > 3:
    r40 = json.load(open(sys.argv[3]))

    def g40(lab, *path):
        v = r40["runs"][lab]["end"]
        for p in path:
            v = v[p]
        return v
    for key, path, _ in METRICS:
        rel = abs(g40("T_97", *path) - g40("T_99", *path)) / max(abs(g40("T_97", *path)), 1e-12)
        spread[key] = rel * abs(g("T_97", *path))
    spread_src = f"the 40k read's |T_97 - T_99| / T_97 applied to this T_97 ({sys.argv[3]})"
else:
    spread_src = "none (no second twin seed)"
L += ["", f"Seed spread ({spread_src}): " + ", ".join(f"{k} {v:.3g}" for k, v in spread.items()), ""]
verdict = {}
for fl, tw in sc.get("pairs", []):
    if fl not in runs or tw not in runs:
        continue
    s = int(fl[-2:])
    cA, cc = ceil(s)
    Af, cf, At, ct = g(fl, "relief", "A"), g(fl, "relief", "corr"), g(tw, "relief", "A"), g(tw, "relief", "corr")
    P = dict(P1=dict(share=Af / cA, corr=cf, corr_ceiling=cc, twin_share=At / cA, twin_corr=ct,
                     pass_=bool(Af >= 0.5 * cA and cf >= 0.5 * cc and At <= 0.1 * cA and abs(ct) <= 0.2)))
    ef, et = g(fl, "images", "relief", "mse_pattern"), g(tw, "images", "relief", "mse_pattern")
    P["P2"] = dict(full=ef, twin=et, rel=(ef - et) / et, spread=spread.get("letters' shading"),
                   pass_=bool(spread and et - ef > spread["letters' shading"]))
    p3 = []
    for key, path, worse in METRICS[:4]:
        d = worse * (g(fl, *path) - g(tw, *path))
        p3.append(dict(metric=key, worse_by=d, spread=spread.get(key), pass_=bool(spread and d <= spread[key])))
    P["P3"] = p3
    P["P4"] = dict(full=g(fl, "records", "v_mean"), twin=g(tw, "records", "v_mean"),
                   ratio=g(fl, "records", "v_mean") / max(g(tw, "records", "v_mean"), 1e-12),
                   pass_=bool(g(fl, "records", "v_mean") <= 2 * g(tw, "records", "v_mean")))
    P["P5"] = dict(full=g(fl, "pieces", "pieces_ge10"), twin=g(tw, "pieces", "pieces_ge10"),
                   pass_=bool(g(fl, "pieces", "pieces_ge10") <= g(tw, "pieces", "pieces_ge10")))
    P["carves"] = P["P1"]["pass_"]
    P["passes"] = bool(P["P1"]["pass_"] and P["P2"]["pass_"] and all(x["pass_"] for x in p3) and P["P4"]["pass_"]
                       and P["P5"]["pass_"])
    verdict[f"{fl} vs {tw}"] = P
L += ["| arm vs twin | P1 share / corr (ceiling corr) | twin share / corr | P1 | P2 letters' err (full / twin, rel) | P2 | P3 misses | "
      "P4 v_mean ratio | P4 | P5 | PASS |", "|" + "---|" * 11]
for k, P in verdict.items():
    miss = [x["metric"] + f" ({x['worse_by']:+.4g} > {x['spread'] if x['spread'] is None else format(x['spread'], '.3g')})" for x in P["P3"] if not x["pass_"]]
    L.append(f"| {k} | {P['P1']['share']:.3f} / {P['P1']['corr']:.3f} ({P['P1']['corr_ceiling']:.3f}) | "
             f"{P['P1']['twin_share']:+.3f} / {P['P1']['twin_corr']:+.3f} | {'yes' if P['P1']['pass_'] else 'no'} | "
             f"{P['P2']['full']:.3e} / {P['P2']['twin']:.3e} ({P['P2']['rel']:+.1%}) | {'yes' if P['P2']['pass_'] else 'no'} | "
             f"{'; '.join(miss) or 'none'} | {P['P4']['ratio']:.2f} | {'yes' if P['P4']['pass_'] else 'no'} | "
             f"{'yes' if P['P5']['pass_'] else 'no'} | {'PASS' if P['passes'] else ('carves' if P['carves'] else 'no')} |")
# course: the carved relief over time
L += ["", "Carved relief A (pitches) in the particles over time:", "",
      "| t (s) | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
n = max(len(runs[lab]["series"]) for lab in runs)
ref = max(runs, key=lambda lab: len(runs[lab]["series"]))
for i in sorted(set(list(range(0, n, max(1, n // 10))) + [n - 1])):
    cells = []
    for lab in runs:
        ser = runs[lab]["series"]
        cells.append(f"{(ser[i]['relief']['A'] if i < len(ser) and ser[i]['relief']['A'] is not None else float('nan')):+.3f}")
    L.append(f"| {runs[ref]['series'][i]['t']:.2f} | " + " | ".join(cells) + " |")
L += ["", "Full vs twin (same source particles): median particle distance in the letters' region / elsewhere and the layers' "
      "surface distance (pitches); the twins' seed floor last:", ""]
for k, p in pairs.items():
    if k == "spread":
        L.append(f"- twins' seed floor ({' vs '.join(p['labels'])}): end surface distance {p['end_surf']:.3f} a")
        continue
    ser = p["series"]
    if not ser:
        continue
    pick = [ser[j] for j in sorted(set([0, len(ser) // 4, len(ser) // 2, 3 * len(ser) // 4, len(ser) - 1]))]
    L.append(f"- {k}: " + "; ".join(f"t {x['t']:.2f}: {x['pp_med_region'] or float('nan'):.2f} / {x['pp_med_else'] or float('nan'):.2f}, "
                                     f"surf {x['surf']:.2f}" for x in pick) + f"; end surface distance {p['end_surf']:.3f} a")
L.append("")
open(out, "w").write("\n".join(L) + "\n")
json.dump(verdict, open(out.replace(".md", "_verdict.json"), "w"), indent=1, default=float)
print("\n".join(L))
