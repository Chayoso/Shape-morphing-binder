"""loss_plot.py OUT_PNG TITLE LABEL=RUN_JSON [...] — every term of the objective and of the step, per committed window,
from the runs' records: the window objective and the selection merit, the transport energy, the volume term, the
kinetic terms (released end, over the window, variance), the stability terms, the cleanup (W1), the render terms
(silhouette, shading) and their weight, the two gradients' norms and the render's share of the step."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out, title = sys.argv[1], sys.argv[2]
panels = [("window objective (loss)", "loss", True), ("selection merit (the acceptance's number)", "selection_merit", True),
          ("transport energy", "transport_energy", True), ("volume term D_vol", "d_vol", True),
          ("kinetic energy at the released end", "kin", True), ("kinetic energy over the window", "kin_run", True),
          ("released motion (stability)", "stab", True), ("end drift", "stab_end", True),
          ("cleanup (W1 to the target)", "d_dt", True), ("render: silhouette term (the run's own)", "d_sil", True),
          ("render: shading term (the run's own)", "d_pbr", True), ("render weight lambda", "lambda", True),
          ("|g_physics|", "g_phys_norm", True), ("|g_render| (projected)", "g_rend_norm", True),
          ("render gradient's share of the step", "g_share", False), ("mean particle move per window", "move", True)]
fig, ax = plt.subplots(8, 2, figsize=(15, 30))
colors = ["k", "tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]   # as render_twin_plot and momentum_plot
for i, spec in enumerate(sys.argv[3:]):
    label, path = spec.split("=", 1)
    h = [r for r in json.load(open(path))["arms"]["render_full_dt_iso_nn"]["history"] if r.get("frame_end")]
    w = [r["frame_end"] / 40 for r in h]
    for (name, key, log), a in zip(panels, ax.flat):
        v = [r.get(key) for r in h]
        pts = [(x, y) for x, y in zip(w, v) if y is not None and (not log or y > 0)]
        if pts:
            a.plot(*zip(*pts), color=colors[i % len(colors)], lw=1.1, label=label)
for (name, key, log), a in zip(panels, ax.flat):
    a.set_title(name, fontsize=10)
    if log and a.lines:
        a.set_yscale("log")
    a.set_xlabel("window", fontsize=8)
    a.grid(alpha=.3)
    if a.lines:
        a.legend(fontsize=7)
fig.suptitle(title, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, .98))
fig.savefig(out, dpi=85)
print("wrote", out)
