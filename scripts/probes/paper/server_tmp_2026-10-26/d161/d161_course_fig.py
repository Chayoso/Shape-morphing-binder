"""d161_course_fig.py OUT_PNG READ_JSON [READ_JSON ...] -- D161: the render -> physics course per scene (CPU).
Top: the relief carved into the particles over time, as a share of what the relief mesh's own sample carries (full and
twin, both seeds; frames where fewer than 1000 layer particles are within reach of the surface are left out).
Bottom: how far the full arm's surface is from its twin's, against how far two twins of different seeds are."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

out, files = sys.argv[1], sys.argv[2:]
fig, axes = plt.subplots(2, len(files), figsize=(6.2 * len(files), 7.5), squeeze=False)
for c, f in enumerate(files):
    r = json.load(open(f))
    sc = r["scene"]["scene"]
    ax = axes[0, c]
    for lab, col, ls in (("full97", "#c0392b", "-"), ("full99", "#c0392b", "--"), ("twin97", "#2471a3", "-"), ("twin99", "#2471a3", "--")):
        s = int(lab[-2:])
        ceil = 0.5 * (r["refs"][f"relief_s1{s}"]["relief"]["A"] + r["refs"][f"relief_s2{s}"]["relief"]["A"])
        ser = [x for x in r["runs"][lab]["series"] if x["relief"].get("A") is not None and x["relief"]["n"] >= 1000]
        name = ("dF_c + layer u + render loss (full)" if lab.startswith("full") else "dF_c + layer u") + f", seed {s}"
        ax.plot([x["t"] for x in ser], [x["relief"]["A"] / ceil for x in ser], ls, color=col, label=name)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(1, color="#555", lw=0.8, ls=":")
    ax.text(0.1, 1.02, "the relief mesh's own sample", fontsize=8, color="#555")
    ax.set_ylim(-0.2, 1.15)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("relief carved into the particles (share)")
    ax.set_title(f"{sc}: how much of the letters the particles carry")
    ax.legend(fontsize=7)
    ax = axes[1, c]
    fl = r["pairs"]["twin97_twin99"]["series"]
    ax.plot([x["t"] for x in fl], [x["surf"] for x in fl], color="#7f8c8d", lw=2, label="two twins, seeds 97 and 99 (the floor)")
    for s, ls in ((97, "-"), (99, "--")):
        ser = r["pairs"][f"full_twin{s}"]["series"]
        ax.plot([x["t"] for x in ser], [x["surf"] for x in ser], ls, color="#c0392b", label=f"full vs twin, seed {s}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance between the two surfaces (pitches)")
    ax.set_title(f"{sc}: does the render move the surface beyond seed noise?")
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(out, dpi=130)
