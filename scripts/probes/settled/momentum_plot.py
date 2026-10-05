"""momentum_plot.py OUT_PNG TITLE LABEL=MOM_LOG,RUN_JSON [...] — the momentum of runs over every kept frame and every
committed window: the centre of mass's displacement, the net-over-gross linear and angular move, the net rotation
summed so far, the mean particle move per kept-frame pair (the probe's rows); the centre of mass's velocity, the
kinetic energy and the centre of mass's position (the run's record)."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out, title = sys.argv[1], sys.argv[2]
fig, ax = plt.subplots(4, 2, figsize=(15, 15))
colors = ["k", "tab:red", "tab:blue", "tab:cyan", "tab:green", "tab:orange", "tab:purple"]
panels = [("centre of mass: displacement from the first frame (pitches)", "com", False), ("net / gross linear move per kept-frame pair", "lin", True),
          ("net / gross angular move per kept-frame pair", "ang", True), ("net rotation summed so far (deg)", "rot", False),
          ("mean particle move per kept-frame pair (pitches)", "move", True)]
record = [("centre of mass: velocity per committed window (world units)", lambda r: sum(c * c for c in r["v_com"]) ** .5, True),
          ("kinetic energy per committed window", lambda r: r["kin"], True),
          ("centre of mass: distance from the origin per committed window (world units)", lambda r: sum(c * c for c in r["com"]) ** .5, False)]
for i, spec in enumerate(sys.argv[3:]):
    label, rest = spec.split("=")
    mom, run = rest.split(",")
    rows = [json.loads(l) for l in open(mom) if l.startswith("{")]
    h = [r for r in json.load(open(run))["arms"]["render_full_dt_iso_nn"]["history"] if r.get("frame_end")]
    w = [r["state"] / 40 for r in rows]
    for (name, key, log), a in zip(panels, ax.flat):
        a.plot(w, [r[key] for r in rows], color=colors[i % 7], lw=1, label=label)
    hw = [r["frame_end"] / 40 for r in h]
    for (name, f, log), a in zip(record, list(ax.flat)[5:]):
        a.plot(hw, [f(r) for r in h], color=colors[i % 7], lw=1, label=label)
for (name, key, log), a in zip(panels + record, ax.flat):
    a.set_title(name, fontsize=10)
    if log:
        a.set_yscale("log")
    a.set_xlabel("window", fontsize=8)
    a.grid(alpha=.3)
    a.legend(fontsize=7)
fig.suptitle(title, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, .97))
fig.savefig(out, dpi=90)
print("wrote", out)
