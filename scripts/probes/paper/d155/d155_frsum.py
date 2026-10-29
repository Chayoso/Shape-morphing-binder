"""d155_frsum.py OUT_DIR -- D155: the kept-frame readings (fr_*.json) at fixed times, in pitches and in 4K pixels.

Pixels: render_deliverable.py's camera (distance 3.6 R from the target's centre, R its largest radius, 30 deg
vertical field of view, 3840 x 2160): at the centre's depth one world unit is 2160 / (2 x 3.6 R x tan 15 deg) pixels.
remaining normal change at t: lag_n (rms over the end's outer layer of (x_t - x_end) . n_end); the summed normal
surface motion (mv_lay_n per kept-frame pair) per phase as a share of the run's total."""
import glob
import json
import math
import os
import sys

import numpy as np

O = sys.argv[1]
G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
T = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
BINS = (("0-1 s", 0, 1), ("1-2 s", 1, 2), ("2-3 s", 2, 3), ("3-6 s", 3, 6), ("6 s-end", 6, 1e9))
out = open(f"{O}/tables_frames.md", "w")
res = {}


def p(s=""):
    print(s)
    out.write(s + "\n")


p("## F1 what is still to come at time t (kept frames, seed 97): the outer layer's remaining NORMAL displacement, rms "
  "pitches [4K pixels]; mean path done")
p("| run | px / pitch | " + " | ".join(f"{t:g} s" for t in T) + " | end (s) |")
p("|---|---|" + "---|" * len(T) + "---|")
for f in sorted(glob.glob(f"{O}/fr_*.json")):
    name = os.path.basename(f)[3:-5]
    r = json.load(open(f))
    z = np.load(r["file"])
    tgt = np.asarray(z["tgt"], np.float64)
    R = float(np.linalg.norm(tgt - tgt.mean(0), axis=1).max())
    px_wu = 2160.0 / (2 * 3.6 * R * math.tan(math.radians(15.0)))
    pxp = px_wu * r["pitch"]
    rows = r["rows"]
    t = np.array([w["raw"] / 240.0 for w in rows])
    ln = np.array([w["lag_n"] for w in rows])
    lp = np.array([w["lag_n_p95"] for w in rows])
    pa = np.array([w["path"] for w in rows])
    ll = np.array([w["lag_lay"] for w in rows])
    mvn = np.array([w.get("mv_lay_n", 0.0) for w in rows[:-1]])
    mvl = np.array([w.get("mv_lay", 0.0) for w in rows[:-1]])
    tm = t[:-1]
    e = dict(name=name, px_per_pitch=pxp, T=float(t[-1]), t=t.tolist(), lag_n=ln.tolist(), lag_n_p95=lp.tolist(),
             path=pa.tolist(), lag_lay=ll.tolist(), mv_lay_n=mvn.tolist(), mv_lay=mvl.tolist())
    e["at"] = {f"{q:g}": dict(lag_n=float(np.interp(q, t, ln)), lag_n_p95=float(np.interp(q, t, lp)),
                              path=float(np.interp(q, t, pa)), lag_lay=float(np.interp(q, t, ll))) for q in T}
    e["below"] = {}
    for thr in (1.0, 0.5, 0.25, 0.1, 1.0 / pxp):
        i = next((k for k in range(len(ln)) if all(ln[k:] < thr)), len(ln) - 1)
        e["below"][f"{thr:.3f}"] = float(t[i])
    tot_n, tot_l = mvn.sum(), mvl.sum()
    e["motion_share_n"] = {lab: float(mvn[(tm >= lo) & (tm < hi)].sum() / tot_n) for lab, lo, hi in BINS}
    e["motion_share"] = {lab: float(mvl[(tm >= lo) & (tm < hi)].sum() / tot_l) for lab, lo, hi in BINS}
    e["motion_total_n"] = float(tot_n)
    res[name] = e
    p(f"| {name} | {pxp:.1f} | " + " | ".join(f"{e['at'][f'{q:g}']['lag_n']:.3f} [{e['at'][f'{q:g}']['lag_n'] * pxp:.1f}]"
                                              for q in T) + f" | {t[-1]:.1f} |")
p()
p("## F2 the time from which the remaining normal displacement stays below a size (s): 1 / 0.5 / 0.25 / 0.1 pitch / one 4K pixel")
p("| run | 1 pitch | 0.5 | 0.25 | 0.1 | 1 px (4K) | run end |")
p("|---|---|---|---|---|---|---|")
for name, e in res.items():
    b = list(e["below"].values())
    p(f"| {name} | " + " | ".join(f"{v:.2f}" for v in b) + f" | {e['T']:.1f} |")
p()
p("## F3 the 95th percentile of the remaining normal displacement (pitches) at 3 / 6 s, the mean path done at 1 / 2 / 3 s")
p("| run | p95 at 3 s | p95 at 6 s | path 1 s | path 2 s | path 3 s |")
p("|---|---|---|---|---|---|")
for name, e in res.items():
    a = e["at"]
    p(f"| {name} | {a['3']['lag_n_p95']:.3f} | {a['6']['lag_n_p95']:.3f} | {a['1']['path']:.3f} | {a['2']['path']:.3f} | "
      f"{a['3']['path']:.3f} |")
p()
p("## F4 the summed surface motion (outer layer, per kept-frame pair; all / normal part) per phase, share of the run's total")
p("| run | " + " | ".join(lab for lab, _, _ in BINS) + " | total normal (pitch) |")
p("|---|" + "---|" * len(BINS) + "---|")
for name, e in res.items():
    p(f"| {name} | " + " | ".join(f"{e['motion_share'][lab]:.2f} / {e['motion_share_n'][lab]:.2f}" for lab, _, _ in BINS)
      + f" | {e['motion_total_n']:.1f} |")
json.dump(res, open(f"{O}/frames_summary.json", "w"))
out.close()

# F5: the Eulerian cross-check: the frame's own outer layer against the end's (signed normal offset, rms), less the
# end's own self-offset (the same reading of the end state against itself, 0.17 pitch: the layer's own roughness about
# its neighbours' tangent planes), in quadrature: sqrt(off(t)^2 - off(end)^2). It does not count particles that
# trade places within the layer (Lagrangian lag_n does), so it is the lower estimate of the visible change.
out = open(f"{O}/tables_frames.md", "a")
p("## F5 the visible surface's distance from its end shape, Eulerian (excess over the end's self-offset), rms pitches "
  "[4K px]: lower estimate beside F1's Lagrangian upper estimate")
p("| run | self-offset at the end | " + " | ".join(f"{t:g} s" for t in T) + " |")
p("|---|---|" + "---|" * len(T))
for f in sorted(glob.glob(f"{O}/fr_*.json")):
    name = os.path.basename(f)[3:-5]
    r = json.load(open(f))
    rows = r["rows"]
    t = np.array([w["raw"] / 240.0 for w in rows])
    off = np.array([w["off_rms"] for w in rows])
    ex = np.sqrt(np.maximum(off ** 2 - off[-1] ** 2, 0.0))
    pxp = res[name]["px_per_pitch"]
    res[name]["off_excess"] = ex.tolist()
    p(f"| {name} | {off[-1]:.3f} | " + " | ".join(f"{np.interp(q, t, ex):.3f} [{np.interp(q, t, ex) * pxp:.1f}]"
                                                  for q in T) + " |")
json.dump(res, open(f"{O}/frames_summary.json", "w"))
out.close()
