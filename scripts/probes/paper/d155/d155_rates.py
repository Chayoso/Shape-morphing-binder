"""d155_rates.py -- D155: FAMO's quantity on the tag: the per-window relative decrease (log decrement) of the physics
core and of the render term, per phase (render arms and twins, median over seeds), and their ratio; also the decades
each term falls from the first record to the end and the path-speed headroom against a ramp of one elastic period."""
import json

import numpy as np

R = json.load(open("/data/relcfd/chayo/physmorph_v2/output/gpu/d155/records.json"))
BINS = (("0-1 s", 0, 1), ("1-2 s", 1, 2), ("2-3 s", 2, 3), ("3-6 s", 3, 6), ("6 s-end", 6, 1e9))
TE = 3.23   # elastic period from the [disc] line (3.23-3.26 s on the six meshes)
for mesh in ("bunny", "dragon", "teapot", "bob", "armadilo", "homer"):
    for arm in ("L", "P"):
        rr = [r for r in R if r["mesh"] == mesh and r["arm"] == arm]
        rowP, rowR, dec, speed = {}, {}, [], []
        for r in rr:
            c = r["curves"]
            t = np.array(c["t"])
            lp, lr = np.log10(c["Pc"]), np.log10(c["R"])
            dec.append((lp[0] - lp[-1], lr[0] - lr[-1]))
            for lab, lo, hi in BINS:
                m = (t > lo) & (t <= hi)
                idx = np.flatnonzero(m)
                if len(idx) < 2:
                    continue
                i0, i1 = max(idx[0] - 1, 0), idx[-1]
                n = i1 - i0
                rowP.setdefault(lab, []).append((lp[i0] - lp[i1]) / n)
                rowR.setdefault(lab, []).append((lr[i0] - lr[i1]) / n)
            pth = np.array([p if p is not None else np.nan for p in c["path"]])
            ok = np.isfinite(pth)
            tt, pp = np.r_[0.0, t[ok]], np.r_[0.0, pth[ok]]
            v = np.diff(pp) / np.diff(tt)
            speed.append(float(np.max(v)))
        dP = np.median([d[0] for d in dec])
        dR = np.median([d[1] for d in dec])
        s = "  ".join(f"{lab}: {np.median(rowP[lab]):.3f}/{np.median(rowR[lab]):.3f} ({np.median(rowR[lab]) / np.median(rowP[lab]):.2f})"
                      for lab, _, _ in BINS if lab in rowP)
        print(f"{mesh:8s} {arm} decades P {dP:.2f} R {dR:.2f} | log10 decrement a window phys/render (ratio): {s} | "
              f"peak path speed {np.median(speed):.2f}/s, ramp peak {1.5 / TE:.2f}/s, headroom {np.median(speed) / (1.5 / TE):.2f}")
