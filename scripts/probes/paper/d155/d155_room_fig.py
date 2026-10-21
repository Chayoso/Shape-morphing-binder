"""d155_room_fig.py ROOM_JSON OUT_PNG -- the figure of d155_room.py: how far the render moves the body's shape away from the
physics-only twin's, over time, under the tag's fast arrival and under D155d's ramp; the same arm run twice (the re-run
floor: what two identical runs differ by anyway) dashed. Top: the shape difference of the outer layers (pitches);
bottom: the outer layer's particle-wise difference (pitches). Five meshes."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = json.load(open(sys.argv[1]))
INK2, MUTED, GRID, SURF = "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "savefig.dpi": 160, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "lines.linewidth": 2.0, "legend.frameon": False, "legend.fontsize": 7.5})
meshes = [m for m in ("bunny", "dragon", "teapot", "armadilo", "homer") if m in R]
fig, ax = plt.subplots(2, len(meshes), figsize=(3.3 * len(meshes), 6.6), squeeze=False)
for j, m in enumerate(meshes):
    for key, col, ls, lab in (("tag", MUTED, "-", "frozen: render arm vs its physics-only twin"),
                              ("ramp", ORANGE, "-", "ramp (D155d): render arm vs its twin"),
                              ("tag re-run", MUTED, ":", "frozen: the same run twice (the floor)"),
                              ("D155b re-run", ORANGE, ":", "ramp (D155b): the same run twice (the floor)")):
        r = R[m].get(key)
        if not r:
            continue
        t = [c["t"] for c in r["curve"]]
        ax[0, j].plot(t, [c["shape"] for c in r["curve"]], color=col, ls=ls, label=lab)
        ax[1, j].plot(t, [c["layer"] for c in r["curve"]], color=col, ls=ls, label=lab)
    for i in (0, 1):
        ax[i, j].axvline(3.26, color=INK2, lw=0.9, ls="--")
        ax[i, j].set_xlim(0, 8)
        ax[i, j].set_xlabel("simulated time (s); dashed: the ramp ends")
    ax[0, j].set_title(f"{m} 40k", fontsize=9)
    ax[0, j].set_ylabel("how far the surfaces differ (pitches)")
    ax[1, j].set_ylabel("how far the surface particles differ (pitches)")
ax[0, 0].legend(loc="lower right")
fig.suptitle("What the render changes in the shape over time: the render arm against its physics-only twin, frozen pipeline "
             "against the elastic-period ramp (dotted: two identical runs)", fontsize=10)
fig.tight_layout()
fig.savefig(sys.argv[2])
print("written", sys.argv[2])
