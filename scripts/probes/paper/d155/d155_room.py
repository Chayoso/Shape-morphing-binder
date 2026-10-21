"""d155_room.py OUT_JSON -- D155 (the user's question: "physics already moves everything; what does the render do?"), CPU,
existing 40k runs only. Does the render shape the in-between shapes, under the tag's fast arrival and under D155d's
ramp? Per mesh, the render arm against its lambda = 0 twin (same seed, same source sample, so particle i is the same
material in both): the tag V97 / W97 and D155d A3DL97 / A3DP97.
  (1) over time (every 0.1 s and at 0.25 / 0.5 / 1 / 1.5 / 2 / 3 s and the end): the particle-wise rms of x_arm - x_twin
      over the whole body and over the arm's outer layer, and the shape difference (the mean distance of each run's outer
      layer to the other's, both ways), in pitches of the run's target sample; the seed-to-seed shape difference of two
      render arms (V97 / V99; on dragon and homer also A3DL97 / A3DL99) as the floor two runs differ by anyway.
  (2) per phase (0-1 s, 1 s-T_e, after T_e; T_e = 3.26 s, the ramp's end): the render's share of the actual motion, the
      arm's displacement D over 0.2 s against the arm-minus-twin displacement d over the same interval, projected: sum d.D
      / sum |D|^2 (outer layer and all particles), and the summed motion |D|^2 per phase (the share of the run's motion in
      it). As D155's B7 measure, but from two whole runs rather than per-window counterfactuals from one start, so the
      runs' own divergence is in it (read against the seed floor)."""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

O = "/data/relcfd/chayo/physmorph_v2/output/gpu/d155/gate"
TE = 3.26
# the render arm against its twin, and the floor: the same arm run twice (same seed, same source; the runs differ by the
# unordered atomics only: V97 against its re-run VE97, D155b's A3GL97 against A3GE97)
PAIRS = (("tag", "V97", "W97"), ("ramp", "A3DL97", "A3DP97"), ("tag re-run", "V97", "VE97"),
         ("D155b re-run", "A3GL97", "A3GE97"), ("D155b", "A3GL97", "A3GP97"))
out = {}


def load(name):
    z = np.load(f"{O}/{name}_frames12.npz")
    return z["frames"], np.asarray(z["raws"]), z["tgt"].astype(np.float64)


def layer(x):
    tree = cKDTree(x)
    sp = float(np.median(tree.query(x[::15], 9)[0][:, 8]))
    _, nb = tree.query(x, 33, workers=4)
    return np.linalg.norm(x - x[nb[:, 1:]].mean(1), axis=1) >= 0.5 * sp


def shape_diff(a, b, la, lb):
    da = cKDTree(b[lb]).query(a[la], workers=4)[0].mean()
    db = cKDTree(a[la]).query(b[lb], workers=4)[0].mean()
    return 0.5 * (da + db)


def at(raws, t):
    return int(np.argmin(np.abs(raws / 240.0 - t)))


for mesh in ("bunny", "dragon", "teapot", "armadilo", "homer"):
    res = {}
    for lab, arm, twin in PAIRS:
        FL, rl, tgt = load(f"{mesh}40k_{arm}")
        FP, rp, _ = load(f"{mesh}40k_{twin}")
        pitch = 0.708 * float(np.median(cKDTree(tgt).query(tgt[::10], 9)[0][:, 8]))
        tmax = min(rl[-1], rp[-1]) / 240.0
        ts = list(np.round(np.arange(0.1, tmax + 1e-9, 0.1), 2))
        curve = []
        for t in ts:
            a, b = np.asarray(FL[at(rl, t)], np.float64), np.asarray(FP[at(rp, t)], np.float64)
            la, lb = layer(a), layer(b)
            dd = np.linalg.norm(a - b, axis=1) / pitch
            curve.append(dict(t=float(t), all=float(np.sqrt((dd ** 2).mean())), layer=float(np.sqrt((dd[la] ** 2).mean())),
                              shape=float(shape_diff(a, b, la, lb) / pitch)))
        a, b = np.asarray(FL[-1], np.float64), np.asarray(FP[-1], np.float64)
        la, lb = layer(a), layer(b)
        dd = np.linalg.norm(a - b, axis=1) / pitch
        end = dict(t=float(rl[-1] / 240), all=float(np.sqrt((dd ** 2).mean())), layer=float(np.sqrt((dd[la] ** 2).mean())),
                   shape=float(shape_diff(a, b, la, lb) / pitch))
        # (2) the render's share of the actual motion per phase, 0.2 s steps
        acc = {p: dict(dD=0.0, DD=0.0, dDl=0.0, DDl=0.0) for p in ("0-1 s", "1 s-T_e", "after T_e")}
        for t0 in np.arange(0.0, tmax - 0.2 + 1e-9, 0.2):
            i0, i1, j0, j1 = at(rl, t0), at(rl, t0 + 0.2), at(rp, t0), at(rp, t0 + 0.2)
            x0 = np.asarray(FL[i0], np.float64)
            D = np.asarray(FL[i1], np.float64) - x0
            d = D - (np.asarray(FP[j1], np.float64) - np.asarray(FP[j0], np.float64))
            lm = layer(x0)
            p = "0-1 s" if t0 < 1.0 - 1e-9 else ("1 s-T_e" if t0 < TE - 1e-9 else "after T_e")
            acc[p]["dD"] += float((d * D).sum())
            acc[p]["DD"] += float((D * D).sum())
            acc[p]["dDl"] += float((d[lm] * D[lm]).sum())
            acc[p]["DDl"] += float((D[lm] * D[lm]).sum())
        tot = sum(v["DD"] for v in acc.values())
        phases = {p: dict(share_all=v["dD"] / max(v["DD"], 1e-300), share_layer=v["dDl"] / max(v["DDl"], 1e-300),
                          motion_share=v["DD"] / max(tot, 1e-300)) for p, v in acc.items()}
        res[lab] = dict(pitch=pitch, curve=curve, end=end, phases=phases)
    # the seed floor: two render arms of different seeds (different source samples: the shape difference only)
    floors = {}
    for lab, a_, b_ in ():   # the seed floor (different source samples) is dominated by the sampling; the re-run pairs above are the floor
        try:
            FA, ra, tgt = load(f"{mesh}40k_{a_}")
            FB, rb, _ = load(f"{mesh}40k_{b_}")
        except OSError:
            continue
        pitch = 0.708 * float(np.median(cKDTree(tgt).query(tgt[::10], 9)[0][:, 8]))
        tmax = min(ra[-1], rb[-1]) / 240.0
        fl = []
        for t in list(np.round(np.arange(0.1, tmax + 1e-9, 0.1), 2)):
            a, b = np.asarray(FA[at(ra, t)], np.float64), np.asarray(FB[at(rb, t)], np.float64)
            fl.append(dict(t=float(t), shape=float(shape_diff(a, b, layer(a), layer(b)) / pitch)))
        floors[lab] = fl
    res["floors"] = floors
    out[mesh] = res
    for lab, _, _ in PAIRS:
        r = res[lab]
        pick = [c for c in r["curve"] if any(abs(c["t"] - q) < 0.051 for q in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0))]
        print(f"{mesh} {lab}: " + "; ".join(f"{c['t']:.1f}s all {c['all']:.2f} layer {c['layer']:.2f} shape {c['shape']:.3f}"
                                            for c in pick)
              + f"; END {r['end']['t']:.2f}s all {r['end']['all']:.2f} layer {r['end']['layer']:.2f} shape {r['end']['shape']:.3f}",
              flush=True)
        print("    phases: " + "; ".join(f"{p}: render share {v['share_layer']:+.3f} (layer) {v['share_all']:+.3f} (all), "
                                         f"{100 * v['motion_share']:.0f} % of the motion" for p, v in r["phases"].items()),
              flush=True)
    for lab, fl in floors.items():
        pick = [c for c in fl if any(abs(c["t"] - q) < 0.051 for q in (0.5, 1.0, 2.0, 3.0))] + fl[-1:]
        print(f"    floor {lab}: " + "; ".join(f"{c['t']:.1f}s {c['shape']:.3f}" for c in pick), flush=True)
json.dump(out, open(sys.argv[1], "w"))
