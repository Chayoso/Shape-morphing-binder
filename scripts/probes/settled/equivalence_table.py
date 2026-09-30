"""equivalence_table.py NEW_DIR NEW_PREFIX OLD_DIR OLD_PREFIX N TARGET [TARGET ...] — per target, the new code's run
(NEW_DIR/NEW_PREFIX<N>_<t>.json) against the reference run of the old code (OLD_DIR/OLD_PREFIX<N>_<t>.json): silhouette
IoU, chamfer, det F min, windows, wall minutes, and the IoU difference against the pre-registered band (S3: +-0.002)."""
import json
import os
import sys

new_dir, new_p, old_dir, old_p, N, targets = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:]


def rec(path):
    if not os.path.exists(path):
        return None
    arm = json.load(open(path))["arms"]["render_full_dt_iso_nn"]
    m, h = arm["metrics"], arm["history"]
    wins = sum(1 for r in h if r.get("frame_end") and not r.get("null_commit"))
    return dict(sil=m["sil_iou"], ch=m["chamfer"], detF=m.get("detF_min", float("nan")), win=wins, sec=arm["seconds"])


print(f"{'target':12s} | {'new: sil    chamf  detF  win  min':34s} | {'old: sil    chamf  detF  win  min':34s} | dSil     band")
out = []
for t in targets:
    a, b = rec(os.path.join(new_dir, f"{new_p}{N}_{t}.json")), rec(os.path.join(old_dir, f"{old_p}{N}_{t}.json"))
    f = lambda r: ("(pending)".ljust(34) if r is None else
                   f"{r['sil']:.4f} {r['ch']:.4f} {r['detF']:5.3f} {r['win']:3d} {r['sec'] / 60:4.1f}")
    d = None if (a is None or b is None) else a["sil"] - b["sil"]
    band = "" if d is None else ("inside" if abs(d) <= 0.002 else "OUTSIDE")
    if d is not None:
        out.append((t, d, a["sec"] / b["sec"]))
    print(f"{t:12s} | {f(a)} | {f(b)} | {'' if d is None else f'{d:+.4f}'} {band}")
if out:
    ds = sorted(abs(d) for _, d, _ in out)
    print(f"n={len(out)}  median |dSil| {ds[len(ds) // 2]:.4f}  max {ds[-1]:.4f}  outside: {[t for t, d, _ in out if abs(d) > 0.002]}"
          f"  wall ratio new/old median {sorted(r for _, _, r in out)[len(out) // 2]:.2f}")
