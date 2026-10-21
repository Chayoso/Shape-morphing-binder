"""d155_demo_chart.py OUT_PNG -- D155 demo (the user's question: can we stop almost everything happening in the first
0.75-1 s?): how much of the body's whole movement is done by each time, the frozen pipeline (the tag, render arm, three
seeds) against the elastic-period ramp (D155d, render arm, seed 97), 40k, five meshes; the share done by 1 s marked."""
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = {r["name"]: r for r in (json.loads(line) for line in open("/data/relcfd/chayo/physmorph_v2/output/gpu/d155/gate/gate_all.jsonl"))}
INK2, MUTED, GRID, SURF = "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
ORANGE = "#eb6834"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "savefig.dpi": 160, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "lines.linewidth": 2.2, "legend.frameon": False, "legend.fontsize": 9})
meshes = ("bunny", "dragon", "teapot", "armadilo", "homer")
fig, ax = plt.subplots(1, 5, figsize=(17, 3.9), sharey=True)


def cum(r):
    t = np.asarray(r["curves"]["t"])
    m = np.asarray(r["curves"]["move"], float)
    c = np.cumsum(m) / m.sum()
    return np.concatenate([[0.0], t]), np.concatenate([[0.0], c])


for j, m in enumerate(meshes):
    for s in (97, 99, 101):
        r = R.get(f"{m}40k_V{s}")
        if r:
            t, c = cum(r)
            ax[j].plot(t, 100 * c, color=MUTED, lw=1.6, label="frozen pipeline (3 runs)" if s == 97 else None)
    r = R.get(f"{m}40k_A3DL97")
    if r:
        t, c = cum(r)
        ax[j].plot(t, 100 * c, color=ORANGE, label="elastic-period ramp")
        f1 = 100 * np.interp(1.0, t, c)
        ax[j].annotate(f"{f1:.0f} % by 1 s", (1.0, f1), (1.6, max(f1 - 12, 3)), color=INK2, fontsize=9,
                       arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    rv = R.get(f"{m}40k_V97")
    if rv:
        t, c = cum(rv)
        f1 = 100 * np.interp(1.0, t, c)
        ax[j].annotate(f"{f1:.0f} % by 1 s", (1.0, f1), (1.6, min(f1 + 4, 96)), color=INK2, fontsize=9,
                       arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax[j].axvline(1.0, color=INK2, lw=0.8, ls=":")
    ax[j].set_xlim(0, 5)
    ax[j].set_ylim(0, 102)
    ax[j].set_title(f"{m}", fontsize=10)
    ax[j].set_xlabel("simulated time (s)")
ax[0].set_ylabel("share of the whole movement done (%)")
ax[0].legend(loc="center right")
fig.suptitle("How much of the morph is done by each moment: frozen pipeline (grey) against the elastic-period ramp "
             "(orange), 40k, render arm", fontsize=11)
fig.tight_layout()
fig.savefig(sys.argv[1])
print("written", sys.argv[1])
