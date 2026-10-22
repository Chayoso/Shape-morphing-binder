"""d162b_read.py OUT_MD -- D162b's window reading from the runs' own records (no recomputation), D162b's S beside D162's F:
per scene and seed, the committed windows, the delivered seconds and how the run ended (its log: early stop after
outer rejections, the null / stale patience, the merit plateau, or the window cap); the weight's on/off switches
(lambda > 0 <-> lambda = 0 between committed windows) and the on share after the first zero excess; the excess at
arrival (the first window whose transport energy is below 2 x its final value) and at the end; the rest question:
the first window whose excess is zero ("the excess is gone"), the mean speed there and at the delivered end, the end
against 2 x the twin's (P4 / C3), and the seconds the run went on after the excess was gone; the shape reading's cost
(t_shape against the window's wall time t_attempt). Twins: D161's (relief), D160's P (colour)."""
import json
import re
import sys
from pathlib import Path

import numpy as np

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
SCENES = (("coin", "coin40k_{arm}_{s}", G / "d161" / "runs40" / "coin40k_twin_{s}"),
          ("bunny", "bunny40k_{arm}_{s}", G / "d161" / "runs40" / "bunny40k_twin_{s}"),
          ("band", "bunny40k_band_{arm}_{s}", G / "d160" / "runs40" / "bunny40k_band_P{s}"),
          ("ears", "bunny40k_ears_{arm}_{s}", G / "d160" / "runs40" / "bunny40k_ears_P{s}"))
ARMS = (("S", G / "d162b" / "runs40"), ("F", G / "d162" / "runs40"))


def end_reason(log):
    if not Path(log).exists():
        return "?"
    t = open(log).read()
    for pat, why in (("consecutive rejected candidates -> early stop", "outer rejections"),
                     (r"frozen after \d+ stale/null", "stale/null patience"), ("delivery merit plateau", "merit plateau"),
                     ("gradient converged at window start", "gradient converged")):
        if re.search(pat, t):
            return why
    return "window cap"


def read(prefix):
    js = json.load(open(f"{prefix}.json"))
    arm = list(js["arms"].values())[0]
    dn, H = arm["deliver_n"], arm["history"]
    com = [h for h in H if h.get("frame_end") and not h.get("null_commit")]
    end = next((h for h in com if h["frame_end"] == dn), com[-1])
    return arm, dn, H, com, end


L = ["| scene | seed | arm | windows | delivered s | end | switches | on share after e=0 | e at arrival / end | "
     "excess gone at (s) | v there | v end / twin (ratio) | rest | s run on after e=0 | t_shape / window (s) |",
     "|" + "---|" * 15]
for sc, pat, twin in SCENES:
    for s in (97, 99):
        tw = read(str(twin).format(s=s))[4]["v_mean"] if Path(str(twin).format(s=s) + ".json").exists() else None
        for armname, d in ARMS:
            p = d / pat.format(arm=armname, s=s)
            if not Path(f"{p}.json").exists():
                continue
            arm, dn, H, com, end = read(p)
            lam = np.array([h.get("lambda") or 0.0 for h in com])
            sw = int(sum((a > 0) != (b > 0) for a, b in zip(lam, lam[1:])))
            exc = [h.get("render_excess") for h in com]
            te = np.array([h.get("transport_energy") or np.nan for h in com])
            arr = int(np.argmax(te < 2 * te[-1]))
            f0 = next((i for i, e in enumerate(exc) if e is not None and e == 0.0), None)
            on = float((lam[f0:] > 0).mean()) if f0 is not None else None
            v0 = com[f0]["v_mean"] if f0 is not None else None
            t0 = com[f0]["frame_end"] / 240.0 if f0 is not None else None
            ratio = end["v_mean"] / tw if tw else None
            ts = [h.get("t_shape") for h in com if h.get("t_shape")]
            ta = [h.get("t_attempt") for h in H if h.get("t_attempt")]
            cost = f"{np.median(ts):.1f} / {np.median(ta):.1f}" if ts and ta else "-"
            L.append(f"| {sc} | {s} | {armname} | {len(com)} | {dn / 240:.1f} | {end_reason(f'{p}.log')} | {sw} | "
                     f"{'-' if on is None else f'{on:.2f}'} | {exc[arr] if exc[arr] is None else round(exc[arr], 3)} / "
                     f"{exc[-1] if exc[-1] is None else round(exc[-1], 3)} | {'-' if t0 is None else f'{t0:.1f}'} | "
                     f"{'-' if v0 is None else f'{v0:.2e}'} | {end['v_mean']:.2e} / {'-' if tw is None else f'{tw:.2e}'} "
                     f"({'-' if ratio is None else f'{ratio:.1f}x'}) | {'-' if ratio is None else ('pass' if ratio <= 2 else 'FAIL')} | "
                     f"{'-' if t0 is None else f'{dn / 240 - t0:.1f}'} | {cost} |")
open(sys.argv[1], "w").write("\n".join(L) + "\n")
print("\n".join(L))
