"""d162_c23.py OUT_MD -- D162: the colour scenes' C2 (shape) and C3 (rest) and C4 (pieces as recorded) from the runs' own
records (no recomputation): per scene and seed, F against D160's twin P and FC against its own twin TC (the
pre-registered pairing), D160c H and D160d D beside them. C2 cells: the run's sil_iou, chamfer and thin_uncovered; a cell
misses when the arm is worse than its twin by more than the larger of the two arms' seed spreads (|97 - 99|). C3: the
delivered end's mean speed (the history row whose frame_end is deliver_n) at most 2 x the twin's. Also the windows run
and the delivered seconds."""
import json
import sys
from pathlib import Path

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
R2, R0 = G / "d162" / "runs40", G / "d160" / "runs40"


def rec(path):
    js = json.load(open(str(path) + ".json"))
    arm = list(js["arms"].values())[0]
    m, dn = arm["metrics"], arm["deliver_n"]
    com = [h for h in arm["history"] if h.get("frame_end") and not h.get("null_commit")]
    end = next((h for h in com if h["frame_end"] == dn), com[-1])
    return dict(sil=m["sil_iou"], cham=m["chamfer"], thin=m["thin_uncovered"], v=end["v_mean"], win=len(com),
                t=dn / 240.0)


L = ["| scene | seed | arm | twin | sil_iou arm / twin | chamfer arm / twin | thin arm / twin | C2 misses | v_end arm / twin "
     "(ratio) | C3 | windows | delivered s |", "|" + "---|" * 12]
summary = {}
for sc in ("band", "ears"):
    runs = {}
    for s in (97, 99):
        for arm, path in (("S", G / "d162b" / "runs40" / f"bunny40k_{sc}_S_{s}"), ("F", R2 / f"bunny40k_{sc}_F_{s}"), ("FC", R2 / f"bunny40k_{sc}_FC_{s}"),
                          ("TC", R2 / f"bunny40k_{sc}_TC_{s}"), ("P", R0 / f"bunny40k_{sc}_P{s}"),
                          ("H", R0 / f"bunny40k_{sc}_H{s}"), ("D", R0 / f"bunny40k_{sc}_D{s}")):
            if Path(str(path) + ".json").exists():
                runs[(arm, s)] = rec(path)
    for arm, twin in (("S", "P"), ("F", "P"), ("FC", "TC"), ("H", "P"), ("D", "P")):
        if (arm, 97) not in runs or (twin, 97) not in runs:
            continue
        spread = {k: max(abs(runs[(arm, 97)][k] - runs[(arm, 99)][k]), abs(runs[(twin, 97)][k] - runs[(twin, 99)][k]))
                  for k in ("sil", "cham", "thin")}
        for s in (97, 99):
            a, t = runs[(arm, s)], runs[(twin, s)]
            worse = {"sil": t["sil"] - a["sil"], "cham": a["cham"] - t["cham"], "thin": a["thin"] - t["thin"]}
            miss = [k for k in worse if worse[k] > spread[k]]
            c3 = a["v"] <= 2.0 * t["v"]
            summary.setdefault((sc, arm), []).append((len(miss), c3))
            L.append(f"| {sc} | {s} | {arm} | {twin} | {a['sil']:.4f} / {t['sil']:.4f} | {a['cham']:.4f} / {t['cham']:.4f} | "
                     f"{a['thin']:.3f} / {t['thin']:.3f} | {', '.join(miss) or '-'} | {a['v']:.2e} / {t['v']:.2e} "
                     f"({a['v'] / t['v']:.1f}x) | {'pass' if c3 else 'FAIL'} | {a['win']} | {a['t']:.1f} |")
L += [""] + [f"{sc} {arm}: C2 cells missed {sum(m for m, _ in v)} of {3 * len(v)}; C3 {'pass' if all(c for _, c in v) else 'FAIL'}"
             for (sc, arm), v in summary.items()]
open(sys.argv[1], "w").write("\n".join(L) + "\n")
print("\n".join(L))
