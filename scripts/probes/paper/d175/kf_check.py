"""kf_check.py K1 K2 K3 -- D175 gate: the keyframe smoke runs (t_k = 0.5 s = frame 120; K1 full with --telemetry, K2 the
twin, K3 the naive arm) read for what the design defines: (a) every arm's committed windows inside the segment end at frames 40,
80, 120 (frame_end, the frames held, is the end frame + 1; a commit or a hold adds 2T frames, the boundary falls where frame_of puts it); (b) inside K1's segment the merit
is the soundness terms plus the held render term alone, loss = ot_scale stab_end + stab + lambda render, and on the final
target it is not (the transport and the cleanup join); (c) the held lambda is one value through K1's segment and the
pre-pass's value; (d) the final target's windows open a new selection epoch. Prints OK / FAIL lines and a last line
VERDICT."""
import json
import sys


def arm(prefix):
    return next(iter(json.load(open(prefix + ".json"))["arms"].values()))


ok = True


def check(cond, msg):
    global ok
    ok = ok and bool(cond)
    print(("OK   " if cond else "FAIL ") + msg)


for name, p in zip(("K1 full", "K2 twin", "K3 naive"), sys.argv[1:4]):
    h = arm(p)["history"]
    seg = [r for r in h if r.get("keyframe") == 0 and r.get("frame_end") and not r.get("null_commit")]
    fe = [r["frame_end"] for r in seg]
    check(fe == [41, 81, 121], f"{name}: segment commits end at frames {[f - 1 for f in fe]} (holds {sum(int(bool(r.get('hold'))) for r in seg)})")
    fin = [r for r in h if "keyframe" in r and r["keyframe"] is None and r.get("frame_end")]
    check(len(fin) > 0 and fin[0]["frame_end"] == 161, f"{name}: the first final-target window ends at frame "
          f"{fin[0]['frame_end'] - 1 if fin else None}")
    if fin and not fin[0].get("null_commit") and "selection_epoch" in fin[0]:
        check(fin[0]["selection_epoch"] >= 1, f"{name}: the final target's epoch {fin[0]['selection_epoch']}")
    gs = [r.get("g_raw_cos") for r in h if r.get("keyframe") == 0 and r.get("g_raw_cos") is not None]
    print(f"     {name}: segment g_raw_cos {[round(g, 3) for g in gs]}, g_share "
          f"{[round(r['g_share'], 3) for r in h if r.get('keyframe') == 0 and r.get('g_share') is not None]}")

h = arm(sys.argv[1])["history"]
opt = [r for r in h if r.get("keyframe") == 0 and not r.get("hold") and r.get("loss") is not None
       and r.get("ot_scale") is not None]
res = []
for r in opt:
    rest = r["ot_scale"] * r["stab_end"] + r["stab"] + r["lambda"] * (r["d_render"] + r["d_pbr"])
    res.append(abs(r["loss"] - rest) / max(abs(r["loss"]), 1e-30))
check(len(opt) > 0 and max(res) < 1e-5, f"K1: segment merit = soundness + held render, residuals {[f'{x:.1e}' for x in res]}")
fin = [r for r in h if "keyframe" in r and r["keyframe"] is None and r.get("loss") is not None
       and r.get("ot_scale") is not None and not r.get("null_commit")]
if fin:
    r = fin[0]
    rest = r["ot_scale"] * r["stab_end"] + r["stab"] + r["lambda"] * (r["d_render"] + r["d_pbr"])
    check(abs(r["loss"] - rest) / abs(r["loss"]) > 1e-3, f"K1: on the final target the transport and cleanup join "
          f"(residual {abs(r['loss'] - rest) / abs(r['loss']):.3g})")
lams = sorted({r["lambda"] for r in h if r.get("keyframe") == 0 and not r.get("hold") and r.get("lambda")})
check(len(lams) == 1, f"K1: one held lambda through the segment {lams}")
print("VERDICT:", "PASS" if ok else "FAIL")
