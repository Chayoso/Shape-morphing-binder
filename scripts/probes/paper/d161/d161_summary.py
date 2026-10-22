"""d161_summary.py OUT_MD READ_JSON [READ_JSON ...] -- D161: the pre-registered verdict (P1-P5) and the tables from
d161_read.py's outputs (CPU). Per scene: the references (ceiling = the relief mesh's own samples, floor = the smooth
target's), each run's carved relief (particles and display), held-out shading in the pattern's pixels, geometry against
the smooth mesh, rest, pieces, records; the render -> physics course (full vs twin per kept frame against the
twin97-twin99 floor)."""
import json
import sys

import numpy as np

out, files = sys.argv[1], sys.argv[2:]
L = []
verdicts = {}
for f in files:
    r = json.load(open(f))
    sc = r["scene"]["scene"]
    refs, runs, pairs = r["refs"], r["runs"], r["pairs"]
    ceil = {s: np.mean([refs[f"relief_s1{s}"]["relief"]["A"], refs[f"relief_s2{s}"]["relief"]["A"]]) for s in (97, 99)}
    L += [f"## {sc}", "", "| state | A (pitch) | share of ceiling | corr | display A | display corr |", "|---|---|---|---|---|---|"]
    for k in ("smooth_tgt", "smooth_ind", "relief_s1", "relief_s2"):
        for s in (97, 99):
            v = refs[f"{k}{s}"]
            d = v.get("display", {})
            L.append(f"| ref {k} {s} | {v['relief']['A']:+.3f} | {v['relief']['A'] / ceil[s]:+.3f} | {v['relief']['corr']:+.3f} | "
                     f"{d.get('A', float('nan')):+.3f} | {d.get('corr', float('nan')):+.3f} |")
    for lab in ("full97", "twin97", "full99", "twin99"):
        e = runs[lab]["end"]
        s = int(lab[-2:])
        L.append(f"| {lab} end | {e['relief']['A']:+.3f} | {e['relief']['A'] / ceil[s]:+.3f} | {e['relief']['corr']:+.3f} | "
                 f"{e['display']['A']:+.3f} | {e['display']['corr']:+.3f} |")
    L += ["", "| run | shading err in pattern px (vs relief mesh) | all px | vs smooth mesh (pattern px) | held-out 1-IoU | "
          "Chamfer-L1 vs smooth (a) | F@0.5 vs smooth | run sil_iou | thin_uncovered | v_mean end | largest piece | windows | "
          "lambda (median last 10) |", "|" + "---|" * 13]
    for lab in ("full97", "twin97", "full99", "twin99"):
        e = runs[lab]["end"]
        im, g, rc, pc = e["images"], e["geometry"], e["records"], e["pieces"]
        L.append(f"| {lab} | {im['relief']['mse_pattern']:.3e} | {im['relief']['mse_all']:.3e} | {im['smooth']['mse_pattern']:.3e} | "
                 f"{g['sil_iou_err_held']:.4f} | {g['smooth']['chamfer_l1_a']:.3f} | {g['smooth']['f0.5']:.3f} | {rc['sil_iou']:.4f} | "
                 f"{rc['thin_uncovered']} | {rc['v_mean']:.2e} | {pc['largest_piece']} | {rc['windows']} | {rc['lam_med10']:.2e} |")
    # verdict
    def g(lab, *path):
        v = runs[lab]["end"]
        for p in path:
            v = v[p]
        return v
    P = {}
    P1 = []
    for s in (97, 99):
        Af, cf = g(f"full{s}", "relief", "A"), g(f"full{s}", "relief", "corr")
        At, ct = g(f"twin{s}", "relief", "A"), g(f"twin{s}", "relief", "corr")
        P1.append(dict(seed=s, share_full=Af / ceil[s], corr_full=cf, share_twin=At / ceil[s], corr_twin=ct,
                       pass_=Af >= 0.5 * ceil[s] and cf >= 0.5 and At <= 0.1 * ceil[s] and abs(ct) <= 0.2,
                       partial=Af >= 0.2 * ceil[s] and cf >= 0.3))
    P["P1"] = P1
    sp = abs(g("twin97", "images", "relief", "mse_pattern") - g("twin99", "images", "relief", "mse_pattern"))
    P["P2"] = [dict(seed=s, full=g(f"full{s}", "images", "relief", "mse_pattern"), twin=g(f"twin{s}", "images", "relief", "mse_pattern"),
                    floor=sp, pass_=g(f"twin{s}", "images", "relief", "mse_pattern") - g(f"full{s}", "images", "relief", "mse_pattern") > sp)
               for s in (97, 99)]
    p3 = []
    for key, path, worse in (("held 1-IoU", ("geometry", "sil_iou_err_held"), +1), ("Chamfer-L1", ("geometry", "smooth", "chamfer_l1_a"), +1),
                             ("sil_iou", ("records", "sil_iou"), -1), ("thin_uncovered", ("records", "thin_uncovered"), +1)):
        spread = max(abs(g("full97", *path) - g("full99", *path)), abs(g("twin97", *path) - g("twin99", *path)))
        for s in (97, 99):
            d = worse * (g(f"full{s}", *path) - g(f"twin{s}", *path))
            p3.append(dict(metric=key, seed=s, worse_by=d, spread=spread, pass_=d <= spread))
    P["P3"] = p3
    P["P4"] = [dict(seed=s, full=g(f"full{s}", "records", "v_mean"), twin=g(f"twin{s}", "records", "v_mean"),
                    pass_=g(f"full{s}", "records", "v_mean") <= 2 * g(f"twin{s}", "records", "v_mean")) for s in (97, 99)]
    P["P5"] = [dict(seed=s, full=g(f"full{s}", "pieces", "pieces_ge10"), twin=g(f"twin{s}", "pieces", "pieces_ge10"),
                    pass_=g(f"full{s}", "pieces", "pieces_ge10") <= g(f"twin{s}", "pieces", "pieces_ge10")) for s in (97, 99)]
    verdicts[sc] = P
    L += ["", "**Verdict items**", "```", json.dumps(P, indent=1, default=float), "```"]
    # course
    L += ["", "| t (s) | seed | full-twin particle median, pattern region (a) | elsewhere (a) | full-twin layer surface distance (a) | "
          "twin97-twin99 surface distance (a, floor) |", "|---|---|---|---|---|---|"]
    floor = {round(x["t"], 3): x["surf"] for x in pairs.get("twin97_twin99", {}).get("series", [])}
    for s in (97, 99):
        ser = pairs.get(f"full_twin{s}", {}).get("series", [])
        idx = sorted(set(list(range(0, len(ser), max(1, len(ser) // 8))) + [len(ser) - 1])) if ser else []
        for i in idx:
            x = ser[i]
            L.append(f"| {x['t']:.2f} | {s} | {x['pp_med_region']:.3f} | {x['pp_med_else']:.3f} | {x['surf']:.3f} | "
                     f"{floor.get(round(x['t'], 3), float('nan')):.3f} |")
    L += [f"end surface distance full-twin 97 / 99: {pairs['full_twin97']['end_surf']:.3f} / {pairs['full_twin99']['end_surf']:.3f} a; "
          f"twin97-twin99 {pairs['twin97_twin99']['end_surf']:.3f} a; full97-full99 {pairs.get('full97_full99', {}).get('end_surf', float('nan')):.3f} a", ""]
    # relief over time (particles), full vs twin
    L += ["| t (s) | " + " | ".join(f"A {lab}" for lab in ("full97", "twin97", "full99", "twin99")) + " |", "|---|---|---|---|---|"]
    n = min(len(runs[lab]["series"]) for lab in runs)
    for i in sorted(set(list(range(0, n, max(1, n // 8))) + [n - 1])):
        L.append(f"| {runs['full97']['series'][i]['t']:.2f} | " + " | ".join(
            f"{(runs[lab]['series'][i]['relief']['A'] or float('nan')):+.3f}" for lab in ("full97", "twin97", "full99", "twin99")) + " |")
    L.append("")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
