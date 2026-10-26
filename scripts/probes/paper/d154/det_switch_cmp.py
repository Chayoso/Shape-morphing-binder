"""det_switch_cmp.py REF_prefix SW_prefix T_S EXPECT_BEFORE -- D154: a switched run (--render_until_s / --render_from_s
T_S) against the unswitched run it must equal before T_S (both under Warp's run-to-run deterministic mode,
tmp/rt/det_run.py). Before T_S: every kept frame with raw <= the switch step and every history field of the windows that
start before it (clocks and the switch's own record left out) bit for bit. EXPECT_BEFORE (1 / 0) is the switch's state
there; from T_S on the windows must carry the other state, with lambda 0 when off and > 0 when on. Prints SAME-BEFORE /
DIFFERENT and the windows after the switch."""
import json
import sys

import numpy as np

CLOCK = ("t_", "seconds", "prof", "wall", "time")
ref, sw, T = sys.argv[1], sys.argv[2], float(sys.argv[3])
before = int(sys.argv[4])
za, zb = (np.load(p + "_frames12.npz", allow_pickle=True) for p in (ref, sw))
ja, jb = (list(json.load(open(p + ".json"))["arms"].values())[0] for p in (ref, sw))
dt = float(json.load(open(ref + ".json"))["provenance"]["mpm"]["dt"])
step = round(T / dt)
bad = 0
ra, rb = [int(r) for r in za["raws"]], [int(r) for r in zb["raws"]]
shared = [r for r in ra if r <= step and r in rb]
for r in shared:
    if not np.array_equal(za["frames"][ra.index(r)], zb["frames"][rb.index(r)]):
        print("frame differs at raw", r)
        bad += 1
if not np.array_equal(za["tgt"], zb["tgt"]):
    print("tgt differs")
    bad += 1


def starts(hist):
    s, out = 0, []
    for h in hist:
        out.append(s)
        if h.get("frame_end") and not h.get("null_commit"):
            s = int(h["frame_end"]) - 1
    return out


sa, sb = starts(ja["history"]), starts(jb["history"])
n_cmp = n_rows = 0
for h, s, (g, t) in zip(ja["history"], sa, zip(jb["history"], sb)):
    if s >= step or t >= step:
        break
    n_rows += 1
    if s != t:
        print("window starts differ", s, t)
        bad += 1
    if g.get("render_switch_on") != before:
        print("switch state before T is", g.get("render_switch_on"), "at row", n_rows - 1)
        bad += 1
    for k in sorted(set(h) | set(g)):
        if k == "render_switch_on" or k.startswith(CLOCK) or k in CLOCK:
            continue
        n_cmp += 1
        if json.dumps(h.get(k), sort_keys=True) != json.dumps(g.get(k), sort_keys=True):
            print("history differs:", k, h.get(k), g.get(k))
            bad += 1
after = [(t, g) for g, t in zip(jb["history"], sb) if t >= step]
for t, g in after:
    on = g.get("render_switch_on")
    lam = g.get("lambda")
    ok = on == 1 - before and ((lam or 0.0) == 0.0 if on == 0 else (lam or 0.0) > 0.0)
    if g.get("frame_end") or g.get("outer_rejected"):
        print(f"after: window start {t * dt:.3f} s switch {on} lambda {lam} g_share {g.get('g_share')}"
              + ("" if ok else "  <- UNEXPECTED"))
        bad += 0 if ok else 1
print(f"switch at {T} s = step {step}: kept frames compared {len(shared)}, windows before {n_rows}, fields {n_cmp}, "
      f"windows after {len(after)}:", "SAME-BEFORE" if bad == 0 and after else f"DIFFERENT ({bad})")
