"""d174_cut.py RUN_PREFIX [STARTUP_S SETTLE_S] -- D174: what a shorter run would deliver. For a finished run (its JSON and
kept frames, every 12th): at the end of window w (the last kept frame at or before it) the silhouette IoU, chamfer and
thin coverage against the target, the window's merit and end v_mean, and the wall time a run stopped there would take
(the start-up, the attempts' seconds up to w, the settle). Run from the repo (GPU: the metrics' KNN)."""
import json
import sys

import numpy as np

import physmorph  # noqa: F401
from physmorph import metrics
from physmorph.thin import thin_metrics, thin_set_cached

pre = sys.argv[1]
startup = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
settle_s = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
d = json.load(open(pre + ".json"))
arm = list(d["arms"].values())[0]
z = np.load(pre + "_frames12.npz")
frames, raws, tgt = z["frames"], list(z["raws"]), z["tgt"]
prov = d["provenance"]
cell = prov.get("cell_shape") or prov["mpm"]["dx"]
ts = thin_set_cached(tgt, cell, 40000)
e = metrics.target_extent(tgt)
h = arm["history"]
att = [r for r in h if not r.get("settle")]
cum, t = [], 0.0
for r in att:
    t += float(r.get("t_attempt") or 0.0)
    cum.append(t)
rows = []
kept = [(i, r) for i, r in enumerate(att) if r.get("frame_end") and not r.get("null_commit")]
pick = sorted(set([k for k in range(0, len(kept), 10)] + [len(kept) - 1]))
print(f"{pre.split('/')[-1]}: {len(att)} attempts, {len(kept)} kept; start-up {startup:.0f} s, settle {settle_s:.0f} s")
print("| kept window | attempt | frame | sil_iou | chamfer | thin uncovered | merit | v_mean | wall if stopped here (min) |")
print("|---|---|---|---|---|---|---|---|---|")
for k in pick:
    i, r = kept[k]
    fe = int(r["frame_end"]) - 1
    j = max(q for q in range(len(raws)) if raws[q] <= fe)
    x = frames[j]
    s = metrics.sil_iou(x, tgt, extent=e)
    c = metrics.chamfer(x, tgt)
    th = thin_metrics(x, ts).get("thin_uncovered")
    wall = (startup + cum[i] + settle_s) / 60.0
    print(f"| {k + 1} | {r['animation'] + 1} | {raws[j]} (end {fe}) | {s:.4f} | {c:.4f} | {th:.4f} | {r['selection_merit']:.3e} | "
          f"{r['v_mean']:.2e} | {wall:.1f} |", flush=True)
m = arm["metrics"]
print(f"delivered (after the settle): sil_iou {m['sil_iou']:.4f}, chamfer {m['chamfer']:.4f}, thin {m.get('thin_uncovered'):.4f}")
