"""render_twin_plot.py OUT_PNG TITLE LABEL=PROBE_LOG,RUN_JSON,TERMS_LOG [...] — runs of one mesh and N on one plot,
every kept frame and every window: what the render term changes against the physics-only twin (given first).

Per frame, one yardstick for every run (render_terms_probe.py): the silhouette and shading terms read on the exterior
and on the particle cloud. Per frame, the display (surface_layer_probe.py, lattice): the state against the target
sample as the exterior draws both (intersection over union of the solid regions and the pictures' mean absolute
difference: front camera, its thin crop, the far side), the crop's soft pixels, the field's roughness. Per window (the
run's JSON): the silhouette and shading terms as the run's own objective reads them, the transport energy, the render
gradient's share.
Printed per measure and run: the mean over the frames (windows) every run has after window 10, and the last value."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out, title = sys.argv[1], sys.argv[2]
rows_of = lambda path: [json.loads(l) for l in open(path) if l.startswith("{")]  # noqa: E731
runs = []
for spec in sys.argv[3:]:
    label, paths = spec.split("=")
    log, js, terms = paths.split(",")
    shown = [r for r in rows_of(log) if r["state"] != "target" and "to_target" in r] if log != "-" else []    # "-": not drawn yet
    hist = [h for h in next(iter(json.load(open(js))["arms"].values()))["history"] if h.get("frame_end") is not None and h.get("d_sil") is not None]    # the committed windows
    runs.append((label, dict(terms=rows_of(terms), shown=shown, window=hist)))

T = lambda part, key: lambda r: r[part][key]                   # noqa: E731
S = lambda view, key: lambda r: r["to_target"][view][key]      # noqa: E731
panels = [("yardstick, exterior: silhouette term", "terms", T("exterior", "sil"), True),
          ("yardstick, exterior: shading term", "terms", T("exterior", "pbr"), True),
          ("yardstick, particle cloud: silhouette term", "terms", T("particles", "sil"), True),
          ("yardstick, particle cloud: shading term", "terms", T("particles", "pbr"), True),
          ("display, front: IoU with the target's solid region", "shown", S("front", "iou"), False),
          ("display, front: mean |picture - target's|", "shown", S("front", "difference"), True),
          ("display, thin part's crop: IoU", "shown", S("crop", "iou"), False),
          ("display, thin part's crop: mean |picture - target's|", "shown", S("crop", "difference"), True),
          ("display, far side: IoU", "shown", S("back", "iou"), False),
          ("display, far side: mean |picture - target's|", "shown", S("back", "difference"), True),
          ("display, crop's soft pixels", "shown", lambda r: r["exterior"]["crop_soft"], True),
          ("field's normal against its neighbours' mean (deg)", "shown", lambda r: r["field_normals"]["angle_rms"], False),
          ("the run's own silhouette term (d_sil)", "window", lambda r: r["d_sil"], True),
          ("the run's own shading term (d_pbr)", "window", lambda r: r["d_pbr"], True),
          ("transport energy", "window", lambda r: r["transport_energy"], True),
          ("render gradient's share (g_share)", "window", lambda r: r["g_share"], False)]
at = dict(terms=lambda r: int(r["state"]) / 40, shown=lambda r: int(r["state"]) / 40, window=lambda r: r["frame_end"] / 40)
fig, ax = plt.subplots(8, 2, figsize=(15, 28))
colors = ["k", "tab:red", "tab:blue", "tab:cyan", "tab:green", "tab:orange"]
print(f"== {title}: " + "; ".join(f"{label}: {len(d['shown'])} frames, {len(d['window'])} windows" for label, d in runs))
print("   per measure and run: mean over the common range after window 10 / last value")
for a_, (name, kind, get, log) in zip(ax.ravel(), panels):
    have = [(c, label, d[kind]) for c, (label, d) in zip(colors, runs) if d[kind]]
    if not have:
        a_.set_title(name + ": not drawn", fontsize=10)
        continue
    n = min(len(rows) for _, _, rows in have)
    line = []
    for c, label, rows in have:
        w = np.array([at[kind](r) for r in rows])
        v = np.array([get(r) for r in rows], float)
        a_.plot(w, v, c, label=label, lw=1.1)
        common = (np.arange(len(v)) < n) & (w > 10)
        line.append(f"{label} {v[common].mean():.5g} / {v[-1]:.5g}")
    a_.set_title(name, fontsize=10); a_.set_yscale("log" if log else "linear")
    a_.grid(alpha=.3); a_.set_xlabel("window", fontsize=8); a_.legend(fontsize=8)
    print(f"   {name}: " + "; ".join(line))
fig.suptitle(title, fontsize=13)
fig.tight_layout(rect=(0, 0, 1, .985)); fig.savefig(out, dpi=90); plt.close(fig)
print(f"   wrote {out}")
