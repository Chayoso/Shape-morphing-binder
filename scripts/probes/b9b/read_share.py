"""read_share.py SHARE_DIR WORLD_DIR [OUT_PREFIX] -- B9b's assim_share 40k gate (pre-registered in docs/experiments.md,
B9b, "The controls' share of the assimilation", before its runs). CPU only. Reads the gravity runs of both definitions
(option 1, WORLD_DIR: the load-free assimilation; SHARE_DIR: the controls' share) with read_b9b.scene_row, their 40k
yardstick (t3end_TAG.log, render_terms_probe against the lifted independent sample), and the no-gravity runs in
SHARE_DIR (ZA97 / ZB97 the current rule twice, ZS97 the share; dragon ZA97): their end motion (the level) and, on the
bunny, the share's frame-by-frame difference from the current rule against the current rule's own run-to-run one.
GATE: P1 V1 rest on >= 6 of the 8 gravity runs, and each mesh's render arms' median end motion (the mean particle move
per kept-frame pair over the last second, pitches) at most the largest of that mesh's no-gravity current-rule runs;
P2 every render arm: shape tilt <= 3 deg, end height within one pitch, silhouette IoU at least the reach bar, and IoU,
chamfer, thin, yardstick silhouette and shading no worse than option 1's worst render arm of the mesh; P3 no serious
piece at the end and every window committed."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import read_b9b as RB  # noqa: E402

ARMS = [("L", 97), ("L", 99), ("L", 101), ("P", 97)]
CHECKS = (("sil_ind", "high"), ("chamfer_ind", "low"), ("thin", "low"), ("yard_sil", "low"), ("yard_pbr", "low"))


def yard(R, tag):
    r = RB.last_json(R / f"t3end_{tag}.log", lambda x: "exterior" in x)
    return (r["exterior"]["sil"], r["exterior"]["pbr"]) if r else (None, None)


def rows(R, mesh, arms):
    out = []
    for arm, seed in arms:
        r = RB.scene_row("40k", R, mesh, arm, seed)
        if r is None:
            continue
        r["yard_sil"], r["yard_pbr"] = yard(R, r["name"])
        r["assim"] = None
        try:
            h = json.loads((R / f"{r['name']}.json").read_text())["history"]
            a = [x["assim"] for x in h if x.get("assim")]
            if a:
                r["assim"] = {k: float(np.median([x[k]["p99"] for x in a])) for k in ("read", "frozen_read")}
        except (OSError, KeyError, ValueError):
            pass
        r["rest"] = r["v_com_end"] < 0.01 and r["omegaR_end"] < 0.01 and abs(r["lowest_sp"]) <= 1.0
        out.append(r)
    return out


def frame_diff(R, a, b):
    za, zb = np.load(R / f"{a}.npz"), np.load(R / f"{b}.npz")
    n = min(len(za["frames"]), len(zb["frames"]))
    pitch = json.loads((R / f"{a}_track.json").read_text())["pitch"]
    d = [float(np.sqrt(((np.asarray(za["frames"][k], np.float64) - np.asarray(zb["frames"][k], np.float64)) ** 2)
                       .sum(1).mean())) / pitch for k in range(n)]
    return {"at_1s_sp": d[min(n - 1, 20)], "max_sp": max(d), "end_sp": d[-1]}


def cells(label, r):
    f = RB.fmt
    return [f"{label} {r['name']}", f(r.get("sil_ind"), ".4f"), f(r.get("chamfer_ind"), ".5f"), f(100 * r["thin"], ".2f"),
            f(None if r["yard_sil"] is None else 1e4 * r["yard_sil"], ".2f"),
            f(None if r["yard_pbr"] is None else 1e4 * r["yard_pbr"], ".2f"), f"{r['dh_wu'] / r['pitch']:+.2f}",
            f(r.get("shape_tilt_deg"), ".2f"), f(r["tilt_deg"], ".2f"), f"{r['drop_min_wu'] / r['pitch']:+.2f}",
            f(r["move_lastsec_sp"], ".4f"), f"{r['v_com_end']:.3f}, {r['omegaR_end']:.3f}", f(r["rest"]),
            str(r["end_serious"]), f"{r['windows']} / {r['windows_planned']}", f(r["minutes"], ".1f"),
            "-" if not r["assim"] else f"{r['assim']['read']:.4f} ({r['assim']['frozen_read']:.4f})"]


def main():
    S, W = Path(sys.argv[1]), Path(sys.argv[2])
    prefix = sys.argv[3] if len(sys.argv) > 3 else str(S / "read_share")
    refs = RB.b9_refs("40k")
    lines, fails, gravity_rest = [], [], []
    lines += ["# B9b 40k: the controls' share of the assimilation (assim_share) against option 1", "",
              "End motion = the mean particle move per kept-frame pair over the last second (pitches). Height, drop in "
              "pitches of the sample.", ""]
    hdr = ["run", "sil IoU ind", "chamfer", "thin %", "yard sil e-4", "yard shading e-4", "height (sp)", "shape tilt deg",
           "material tilt deg", "drop min (sp)", "end motion (sp)", "|v_c|, |w|R wu/s", "V1 rest", "serious at end",
           "windows", "min", "assim read p99 (frozen read p99)"]
    for mesh in ("bunny", "dragon"):
        o1, sh = rows(W, mesh, ARMS), rows(S, mesh, ARMS)
        nz = rows(S, mesh, [("ZA", 97), ("ZB", 97), ("ZS", 97)])
        nog = [x["sil_ind"] for x in refs[mesh]["nog"].values()]
        bar = min(nog) - 0.01
        cur = [r["move_lastsec_sp"] for r in nz if r["arm"] in ("ZA", "ZB")]
        level = max(cur) if cur else float("nan")
        lines += [f"## {mesh} 40k", "", "No-gravity runs (standing frame, g 0): " + ", ".join(
            f"{r['name']} end motion {r['move_lastsec_sp']:.4f}" for r in nz) + f"; the level (current rule) {level:.4f}",
            f"Reach bar {bar:.4f}.", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        worst = {}
        for k, sense in CHECKS:
            v = [r[k] for r in o1 if r["arm"] == "L" and r.get(k) is not None]
            worst[k] = (min(v) if sense == "high" else max(v)) if v else None
        for label, group in (("option 1", o1), ("share", sh), ("no gravity", nz)):
            for r in group:
                lines.append("| " + " | ".join(cells(label, r)) + " |")
                if label == "share":
                    gravity_rest.append(r["rest"])
                if label == "share" and r["arm"] == "L":
                    pf = r["dh_wu"] / r["pitch"]
                    ch = {"shape tilt": r.get("shape_tilt_deg") is not None and r["shape_tilt_deg"] <= 3.0,
                          "height": abs(pf) <= 1.0, "reach": r.get("sil_ind") is not None and r["sil_ind"] >= bar,
                          "pieces": r["end_serious"] == 0,
                          "whole": (not r["invalid"]) and r["windows"] == r["windows_planned"]}
                    for k, sense in CHECKS:
                        if worst[k] is not None and r.get(k) is not None:
                            ch[k] = r[k] >= worst[k] if sense == "high" else r[k] <= worst[k]
                        elif worst[k] is not None:
                            ch[k] = False
                    fails += [f"{r['name']} {k}" for k, ok in ch.items() if not ok]
        med = float(np.median([r["move_lastsec_sp"] for r in sh if r["arm"] == "L"])) if sh else float("nan")
        med1 = float(np.median([r["move_lastsec_sp"] for r in o1 if r["arm"] == "L"])) if o1 else float("nan")
        p1 = med <= level
        if not p1:
            fails.append(f"{mesh} end motion {med:.4f} > the no-gravity level {level:.4f}")
        lines += ["", f"P1 ({mesh}): the share's render arms' median end motion {med:.4f} against the no-gravity level "
                  f"{level:.4f} ({'yes' if p1 else 'NO'}); option 1's {med1:.4f}.",
                  f"Option 1's worst render arm (the P2 bounds): " + ", ".join(
                      f"{k} {RB.fmt(v, '.5g')}" for k, v in worst.items()), ""]
        L97 = next((r for r in sh if r["name"].endswith("_L97")), None)
        P97 = next((r for r in sh if r["name"].endswith("_P97")), None)
        if L97 and P97:
            lines += [f"Render's influence (share, seed 97, arm / twin): silhouette term at the end {L97['d_sil_end']:.3g} / "
                      f"{P97['d_sil_end']:.3g}; silhouette IoU {L97['sil_ind']:.4f} / {P97['sil_ind']:.4f}; yardstick "
                      f"silhouette {RB.fmt(L97['yard_sil'], '.3e')} / {RB.fmt(P97['yard_sil'], '.3e')}, shading "
                      f"{RB.fmt(L97['yard_pbr'], '.3e')} / {RB.fmt(P97['yard_pbr'], '.3e')}; g_share median "
                      f"{L97['g_share_median']:.3f}, lambda median {L97['lam_median']:.3g}; end motion "
                      f"{L97['move_lastsec_sp']:.4f} / {P97['move_lastsec_sp']:.4f}", ""]
        if mesh == "bunny":
            try:
                ab = frame_diff(S, "bunny40k_ZA97", "bunny40k_ZB97")
                a_s = frame_diff(S, "bunny40k_ZA97", "bunny40k_ZS97")
                b_s = frame_diff(S, "bunny40k_ZB97", "bunny40k_ZS97")
                lines += ["No gravity, does the share change the frames? RMS position difference (pitches) at 1 s / largest "
                          f"/ end: the current rule against itself {ab['at_1s_sp']:.3f} / {ab['max_sp']:.3f} / "
                          f"{ab['end_sp']:.3f}; the share against the current rule {a_s['at_1s_sp']:.3f} / "
                          f"{a_s['max_sp']:.3f} / {a_s['end_sp']:.3f} and {b_s['at_1s_sp']:.3f} / {b_s['max_sp']:.3f} / "
                          f"{b_s['end_sp']:.3f}.", ""]
            except (OSError, KeyError, ValueError) as e:
                lines += [f"(the no-gravity frame difference is missing: {e})", ""]
    n_rest = sum(gravity_rest)
    if n_rest < 6:
        fails.append(f"V1 rest on {n_rest} of {len(gravity_rest)} gravity runs (needs 6 of 8)")
    lines += [f"V1 rest on {n_rest} of {len(gravity_rest)} gravity runs (share).", "",
              "GATE share 40k: " + ("PASS" if not fails else "FAIL (" + "; ".join(fails) + ")")]
    text = "\n".join(lines) + "\n"
    print(text)
    Path(prefix + ".md").write_text(text)


if __name__ == "__main__":
    main()
