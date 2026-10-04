"""render_twin_plot.py OUT_PNG TITLE LABEL=PROBE_LOG,RUN_JSON [...] — runs of one mesh and N on one plot, every frame
and every window: what the render term changes against the physics-only twin (the first run given is the reference).

Per frame (the rows of surface_layer_probe.py, lattice): the state against the target sample as the exterior draws both
(intersection over union of the solid regions and the pictures' mean absolute difference, front camera, its crop, far
side), the crop's soft pixels, the field's roughness. Per window (the run's JSON): the silhouette and shading terms as
the run's own objective reads them, the transport energy, the render gradient's share and the render weight.
Printed: each measure's mean over the frames (windows) all runs have, and its last value, per run."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out, title = sys.argv[1], sys.argv[2]
runs = []
for spec in sys.argv[3:]:
    label, paths = spec.split("=")
    log, js = paths.split(",")
    rows = [json.loads(l) for l in open(log) if l.startswith("{")]
    rows = [r for r in rows if r["state"] != "target" and "to_target" in r]
    hist = [h for h in next(iter(json.load(open(js))["arms"].values()))["history"] if "d_sil" in h]
    runs.append((label, rows, hist))

frame = [("front: IoU with the target's solid region", lambda r: r["to_target"]["front"]["iou"], False),
         ("front: mean |picture - target's|", lambda r: r["to_target"]["front"]["difference"], True),
         ("thin part's crop: IoU", lambda r: r["to_target"]["crop"]["iou"], False),
         ("thin part's crop: mean |picture - target's|", lambda r: r["to_target"]["crop"]["difference"], True),
         ("far side: IoU", lambda r: r["to_target"]["back"]["iou"], False),
         ("far side: mean |picture - target's|", lambda r: r["to_target"]["back"]["difference"], True),
         ("crop's soft pixels (exterior)", lambda r: r["exterior"]["crop_soft"], True),
         ("field's normal against its neighbours' mean (deg)", lambda r: r["field_normals"]["angle_rms"], False)]
window = [("silhouette term of the objective (d_sil)", "d_sil", True), ("shading term (d_pbr)", "d_pbr", True),
          ("transport energy", "transport_energy", True), ("render gradient's share (g_share)", "g_share", False)]
fig, ax = plt.subplots(6, 2, figsize=(15, 21))
ax = ax.ravel()
colors = ["k", "tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
n_frames = min(len(r[1]) for r in runs)
n_windows = min(len(r[2]) for r in runs)
print(f"== {title}: " + "; ".join(f"{label}: {len(rows)} frames, {len(hist)} windows" for label, rows, hist in runs)
      + f"; means over the first {n_frames} frames and {n_windows} windows (those every run has), then the last value")
for k, (name, get, log) in enumerate(frame):
    line = []
    for c, (label, rows, hist) in zip(colors, runs):
        v = [get(r) for r in rows]
        ax[k].plot([int(r["state"]) / 40 for r in rows], v, c, label=label, lw=1.2)
        line.append(f"{label} {np.mean(v[:n_frames]):.5g} / {v[-1]:.5g}")
    ax[k].set_title(name, fontsize=10); ax[k].set_yscale("log" if log else "linear")
    print(f"   {name}: " + "; ".join(line))
for k, (name, key, log) in enumerate(window, start=len(frame)):
    line = []
    for c, (label, rows, hist) in zip(colors, runs):
        v = [h[key] for h in hist]
        ax[k].plot([h["frame_end"] / 40 for h in hist], v, c, label=label, lw=1.2)
        line.append(f"{label} {np.mean(v[:n_windows]):.5g} / {v[-1]:.5g}")
    ax[k].set_title(name, fontsize=10); ax[k].set_yscale("log" if log else "linear")
    print(f"   {name}: " + "; ".join(line))
for a_ in ax:
    a_.grid(alpha=.3); a_.set_xlabel("window", fontsize=8); a_.legend(fontsize=8)
fig.suptitle(title, fontsize=13)
fig.tight_layout(rect=(0, 0, 1, .98)); fig.savefig(out, dpi=100); plt.close(fig)
print(f"   wrote {out}")
