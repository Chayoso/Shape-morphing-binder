"""read_load.py LOAD_DIR WORLD_DIR [OUT_PREFIX] -- B9b's assim_load 40k gate (pre-registered in docs/experiments.md, B9b,
"The loads' own effect", before its runs). CPU only. Arms in LOAD_DIR: the definition (1)+(2), assim_load with the
recipe's drag 0.9 in the standing frame (L97 / L99 / L101, twin P97); (1) alone, drag 0 (A97 / A99 / A101); the
no-gravity frozen check (frame fixed at launch, no floor, drag 0.9: bunny ZA97 / ZB97 / ZC97 the current rule, ZL97
assim_load; dragon ZA97 the current rule) with the window-level probe load_equal.json. Option 1's 40k runs in WORLD_DIR.
GATE (the definition): P1 for each mesh the render arms' median end motion at most the largest no-gravity current-rule
run of the mesh, and V1 rest on >= 6 of the 8 runs; P2 every render arm: shape tilt <= 3 deg, end height within one
pitch, silhouette IoU at least the reach bar, and IoU, chamfer, thin, yardstick silhouette and shading no worse than
option 1's worst render arm of the mesh; P3 no serious piece at the end, every window committed; F1 the window-level
probe passes; F2 at the run level the new rule's frame differences from the current-rule runs (RMS, pitches, at the end)
overlap the current rule's own run-to-run differences (D145's convention: apart = no overlap). (1) alone is read by the
same items and reported, not gated."""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import read_b9b as RB  # noqa: E402
import read_share as RS  # noqa: E402

HDR = ["run", "sil IoU ind", "chamfer", "thin %", "yard sil e-4", "yard shading e-4", "height (sp)", "shape tilt deg",
       "material tilt deg", "drop min (sp)", "end motion (sp)", "|v_c|, |w|R wu/s", "V1 rest", "serious at end",
       "windows", "min", "assim read p99 (frozen read p99)"]


def checks(r, worst, bar):
    pf = r["dh_wu"] / r["pitch"]
    ch = {"shape tilt": r.get("shape_tilt_deg") is not None and r["shape_tilt_deg"] <= 3.0, "height": abs(pf) <= 1.0,
          "reach": r.get("sil_ind") is not None and r["sil_ind"] >= bar, "pieces": r["end_serious"] == 0,
          "whole": (not r["invalid"]) and r["windows"] == r["windows_planned"]}
    for k, sense in RS.CHECKS:
        if worst[k] is not None:
            ch[k] = r.get(k) is not None and (r[k] >= worst[k] if sense == "high" else r[k] <= worst[k])
    return ch


def main():
    D, W = Path(sys.argv[1]), Path(sys.argv[2])
    prefix = sys.argv[3] if len(sys.argv) > 3 else str(D / "read_load")
    refs = RB.b9_refs("40k")
    lines, fails, notes = [], [], []
    rest_def, rest_alone = [], []
    lines += ["# B9b 40k: the loads' own effect removed from the assimilation (assim_load)", "",
              "Arms: option 1 (load-free assimilation, drag 0); (1) alone (assim_load, drag 0); (1)+(2) the definition "
              "(assim_load, the recipe's drag 0.9 on m v). End motion = the mean particle move per kept-frame pair over the "
              "last second (pitches).", ""]
    for mesh in ("bunny", "dragon"):
        o1 = RS.rows(W, mesh, RS.ARMS)
        de = RS.rows(D, mesh, RS.ARMS)
        al = RS.rows(D, mesh, [("A", 97), ("A", 99), ("A", 101)])
        nz = RS.rows(D, mesh, [("ZA", 97), ("ZB", 97), ("ZC", 97), ("ZL", 97)])
        bar = min(x["sil_ind"] for x in refs[mesh]["nog"].values()) - 0.01
        cur = [r["move_lastsec_sp"] for r in nz if r["arm"] in ("ZA", "ZB", "ZC")]
        level = max(cur) if cur else float("nan")
        worst = {}
        for k, sense in RS.CHECKS:
            v = [r[k] for r in o1 if r["arm"] == "L" and r.get(k) is not None]
            worst[k] = (min(v) if sense == "high" else max(v)) if v else None
        lines += [f"## {mesh} 40k", "", "No-gravity runs (frame fixed at launch, no floor, drag 0.9): " + ", ".join(
            f"{r['name']} end motion {r['move_lastsec_sp']:.4f}" for r in nz) + f"; the level (current rule) {level:.4f}. "
            f"Reach bar {bar:.4f}.", "", "| " + " | ".join(HDR) + " |", "|" + "---|" * len(HDR)]
        for label, group in (("option 1", o1), ("(1) alone", al), ("(1)+(2)", de), ("no gravity", nz)):
            for r in group:
                lines.append("| " + " | ".join(RS.cells(label, r)) + " |")
        for label, group, store in (("(1)+(2)", de, rest_def), ("(1) alone", al, rest_alone)):
            store += [r["rest"] for r in group]
            arms = [r for r in group if r["arm"] in ("L", "A")]
            med = float(np.median([r["move_lastsec_sp"] for r in arms])) if arms else float("nan")
            bad = [f"{r['name']} {k}" for r in arms for k, ok in checks(r, worst, bar).items() if not ok]
            p1 = med <= level
            lines += ["", f"{label} ({mesh}): render arms' median end motion {med:.4f} against the no-gravity level "
                      f"{level:.4f} ({'yes' if p1 else 'NO'}); P2/P3 items missed: {', '.join(bad) or 'none'}"]
            if label == "(1)+(2)":
                fails += bad
                if not p1:
                    fails.append(f"{mesh} end motion {med:.4f} > the no-gravity level {level:.4f}")
            else:
                notes += bad + ([] if p1 else [f"{mesh} end motion {med:.4f} > {level:.4f}"])
        med1 = float(np.median([r["move_lastsec_sp"] for r in o1 if r["arm"] == "L"])) if o1 else float("nan")
        lines += [f"Option 1's render arms' median end motion {med1:.4f}. P2 bounds (option 1's worst render arm): " + ", ".join(
            f"{k} {RB.fmt(v, '.5g')}" for k, v in worst.items()), ""]
        L97 = next((r for r in de if r["name"].endswith("_L97")), None)
        P97 = next((r for r in de if r["name"].endswith("_P97")), None)
        if L97 and P97:
            lines += [f"Render's influence ((1)+(2), seed 97, arm / twin): silhouette term at the end {L97['d_sil_end']:.3g} / "
                      f"{P97['d_sil_end']:.3g}; silhouette IoU {L97['sil_ind']:.4f} / {P97['sil_ind']:.4f}; yardstick "
                      f"silhouette {RB.fmt(L97['yard_sil'], '.3e')} / {RB.fmt(P97['yard_sil'], '.3e')}, shading "
                      f"{RB.fmt(L97['yard_pbr'], '.3e')} / {RB.fmt(P97['yard_pbr'], '.3e')}; g_share median "
                      f"{L97['g_share_median']:.3f}, lambda median {L97['lam_median']:.3g}; end motion "
                      f"{L97['move_lastsec_sp']:.4f} / {P97['move_lastsec_sp']:.4f}; V1 rest {RB.fmt(L97['rest'])} / "
                      f"{RB.fmt(P97['rest'])}", ""]
        if mesh == "bunny":
            cur_tags = ["bunny40k_ZA97", "bunny40k_ZB97", "bunny40k_ZC97"]
            try:
                within = {f"{a[-4:-2]}-{b[-4:-2]}": RS.frame_diff(D, a, b) for a, b in combinations(cur_tags, 2)}
                cross = {f"ZL-{a[-4:-2]}": RS.frame_diff(D, "bunny40k_ZL97", a) for a in cur_tags}
                wv = [v["end_sp"] for v in within.values()]
                cv = [v["end_sp"] for v in cross.values()]
                f2 = min(cv) <= max(wv) and min(wv) <= max(cv)
                lines += ["Frozen check F2 (run level, no gravity): RMS position difference at the end (pitches): the "
                          "current rule against itself " + ", ".join(f"{k} {v['end_sp']:.3f}" for k, v in within.items())
                          + "; the new rule against it " + ", ".join(f"{k} {v['end_sp']:.3f}" for k, v in cross.items())
                          + f": {'overlap (pass)' if f2 else 'APART (fail)'}; largest over the run: within "
                          f"{max(v['max_sp'] for v in within.values()):.3f}, cross {max(v['max_sp'] for v in cross.values()):.3f}.", ""]
                if not f2:
                    fails.append("F2 run-level frozen check apart")
            except (OSError, KeyError, ValueError) as e:
                lines += [f"(F2 missing: {e})", ""]
                fails.append("F2 missing")
    try:
        le = json.loads((D / "load_equal.json").read_text())
        lines += ["Frozen check F1 (window level, no gravity, no floor, 40k bunny): " + "; ".join(
            f"window {w['window']} ({'moving start' if w['moving_start'] else 'from rest'}): continuations with / without "
            f"the loads {w['zl_vs_zf']:.2e} apart (the frozen end's replay {w['replay_F']:.2e}); Fp {w['Fp_load_vs_frozen']:.2e} "
            f"from the frozen rule's (its replay {w['Fp_replay']:.2e}, the step itself {w['Fp_step']:.2e})"
            for w in le["windows"]) + f": {'PASS' if le['pass'] else 'FAIL'}", ""]
        if not le["pass"]:
            fails.append("F1 window-level frozen check")
    except (OSError, ValueError, KeyError) as e:
        lines += [f"(F1 missing: {e})", ""]
        fails.append("F1 missing")
    nd, na = sum(rest_def), sum(rest_alone)
    if nd < 6:
        fails.append(f"V1 rest on {nd} of {len(rest_def)} runs (needs 6 of 8)")
    lines += [f"V1 rest: (1)+(2) on {nd} of {len(rest_def)} runs; (1) alone on {na} of {len(rest_alone)}.",
              f"(1) alone, reported (not gated): {', '.join(notes) or 'every item held'}.", "",
              "GATE load 40k: " + ("PASS" if not fails else "FAIL (" + "; ".join(fails) + ")")]
    text = "\n".join(lines) + "\n"
    print(text)
    Path(prefix + ".md").write_text(text)


if __name__ == "__main__":
    main()
