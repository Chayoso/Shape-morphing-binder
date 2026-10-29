"""next_read.py OUT_MD RUNDIR -- D175 measurements (b) and (c) of tmp/d175/MEAS_a.md. CPU.
(b) For the reruns R_S99_1, R_S99_2 and R_S97_1 (the 208 flags plus the keyframe-only replay record): every committed or
nulled window's E_final - E_accept, split into the end drift (lv), the released motion (stab), lambda x render and the
rest, against the tolerance; whether the commit read the render on a disc set found anew. (c) P_dFc_y13: the body's
second-moment aspect at every kept frame (dF_c alone toward the stretched sphere as a 3D target)."""
import json
import sys
from pathlib import Path

import numpy as np


def arm(p):
    return next(iter(json.load(open(p + ".json"))["arms"].values()))


def aspect(x):
    x = np.asarray(x, np.float64)
    v = ((x - x.mean(0)) ** 2).mean(0)
    return float(np.sqrt(v[1] / (0.5 * (v[0] + v[2]))))


def main():
    out, rd = sys.argv[1], sys.argv[2]
    L = ["# D175 (b) replay split and (c) the physics' pace", "", "## (b) E_final - E_accept by term (keyframe runs)", "",
         "| run | window | entry | outcome | E_final - E_accept | tol | d end drift (lv) | d stab | lambda d render | rest | "
         "render's share of the |terms| | discs found anew at the commit |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    agg = {"seg": [], "fin": []}
    for name in ("R_S99_1", "R_S99_2", "R_S97_1"):
        try:
            h = arm(f"{rd}/{name}")["history"]
        except OSError:
            L.append(f"| {name} | (missing) |" + " |" * 10)
            continue
        for r in h:
            s = r.get("replay_split")
            if not s:
                continue
            outcome = r.get("null_reason") or ("rejected" if r.get("outer_rejected") else "accepted")
            dE = s["E_final"] - s["E_accept"]
            parts = [abs(s["d_lv"]), abs(s["d_stab"]), abs(s["d_render_w"])]
            shr = parts[2] / max(sum(parts), 1e-30)
            seg = s.get("keyframe") is not None
            agg["seg" if seg else "fin"].append((shr, outcome == "commit_replay"))
            if seg or outcome == "commit_replay":
                L.append(f"| {name} | {r['animation'] + 1} | {'keyframe' if seg else 'target'} | {outcome} | {dE:.3e} | "
                         f"{s['tol']:.3e} | {s['d_lv']:+.3e} | {s['d_stab']:+.3e} | {s['d_render_w']:+.3e} | "
                         f"{s['d_rest']:+.3e} | {shr:.3f} | {s['ext_rebuilt']} |")
    for k, v in agg.items():
        if v:
            L.append(f"\n{'Segment' if k == 'seg' else 'Final-target'} windows: {len(v)} commits, commit_replay nulls "
                     f"{sum(b for _, b in v)}, the render's median share of the replay difference {np.median([a for a, _ in v]):.3f}.")
    L += ["", "## (c) dF_c alone toward the stretched sphere as a 3D target (40k, seed 97; keyframe aspect 1.480)", ""]
    try:
        z = np.load(f"{rd}/P_dFc_y13_frames12.npz")
        raws, fr = np.asarray(z["raws"]), z["frames"]
        asp = [aspect(x) for x in fr]
        first = lambda thr: next((int(r) for r, a in zip(raws, asp) if a >= thr), None)  # noqa: E731
        k240 = int(np.flatnonzero(raws == 240)[0]) if 240 in raws else None
        L.append("aspect by frame: " + ", ".join(f"{int(r)}:{a:.3f}" for r, a in zip(raws, asp) if r % 48 == 0 or r == raws[-1]))
        L.append(f"\nAt frame 240 (t = 1 s): {asp[k240]:.3f}; 1.40 first at frame {first(1.40)}, 1.48 at frame {first(1.48)}; "
                 f"the run's last kept frame {int(raws[-1])} at {asp[-1]:.3f}." if k240 is not None else
                 f"\nThe run ended at frame {int(raws[-1])} (aspect {asp[-1]:.3f}); 1.40 first at {first(1.40)}.")
        L.append(f"VERDICT (c): {'enough' if k240 is not None and asp[k240] >= 1.40 else 'NOT enough'} (aspect >= 1.40 by frame 240)")
    except (OSError, KeyError) as e:
        L.append(f"(c) not read: {e!r}")
    Path(out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
