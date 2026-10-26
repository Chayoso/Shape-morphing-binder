"""d154_figs.py OUT_DIR -- D154's figures from table1.json, midrun.json (d154_tables.py) and switch.json (d154_switch.py).
One question per panel, plain-word legends, linear ratios, one scale per figure (the user's visualisation rule).
Colours: the reference palette's first three categorical slots (render arm blue, twin orange, switched run aqua),
grey for the perfect sample; never colour alone (markers / line styles too)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

out = Path(sys.argv[1])
BLUE, ORANGE, AQUA, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c8c7c2", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
                     "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.6, "legend.frameon": False})
MESHES = ["bunny", "dragon", "teapot", "bob", "armadilo", "homer"]

# ---------- (1) the end against the exact mesh: render arm / twin per measure, seed range; the perfect sample ----------
t1 = out / "table1.json"
if t1.exists():
    T = json.load(open(t1))
    rows = [("yard_ind_sil", "yardstick silhouette\n(vs a sample, as before)"), ("yard_ind_pbr", "yardstick shading\n(vs a sample, as before)"),
            ("yard_mesh_sil", "yardstick silhouette\n(vs the exact mesh)"), ("yard_mesh_pbr", "yardstick shading\n(vs the exact mesh)"),
            ("iou1024", "1 - IoU, exact\nsilhouettes, 1024 px"), ("outline_sd", "outline shape error\n(offset spread)"),
            ("normal_cl", "normal angle at the\nnearest mesh point"), ("shading", "shading error, exact\n1024 px"),
            ("dist_sd", "surface distance\nspread")]
    fig, axes = plt.subplots(1, 6, figsize=(13.5, 5.2), sharey=True, sharex=True)
    for ax, mesh in zip(axes, MESHES):
        for i, (k, _) in enumerate(rows):
            d = T.get(mesh, {}).get(k)
            if not d or not d["n"]:
                continue
            y = len(rows) - 1 - i
            r = d["ratio"]
            ax.plot([r[1], r[2]], [y, y], color=BLUE, lw=2, solid_capstyle="round")
            ax.plot(r[0], y, "o", color=BLUE, ms=6, mec="white", mew=1.2, zorder=3)
            if d.get("own") and d.get("P") and d["P"][0]:
                ownr = [p["own"] / p["P"] for p in d["per_seed"] if p["P"]]
                ax.plot(np.median(ownr), y, "|", color=GREY, ms=12, mew=2, zorder=2)
        ax.axvline(1.0, color=ORANGE, lw=1.5, ls="--")
        ax.set_title(mesh, color=INK)
        ax.set_xlim(0.2, 1.4)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([lab for _, lab in rows][::-1])
    fig.supxlabel("error of the render arm relative to its twin (twin = 1, dashed; dot = median over seeds, bar = seed range;"
                  " grey tick = a perfect particle sample of the mesh)", color=INK2)
    fig.suptitle("End of the run: how much smaller is the render arm's error than its twin's?", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig1_end_vs_mesh.png", dpi=160)
    plt.close(fig)

# ---------- (1) mid-run: when does the lead appear (bunny, dragon seed 97) ----------
mr = out / "midrun.json"
if mr.exists():
    M = json.load(open(mr))
    keys = [("yard_ind_sil", "yardstick silhouette (vs a sample)", "-", "o"), ("outline_sd", "outline shape error (exact)", "--", "s"),
            ("normal_cl", "normal angle (exact)", ":", "^")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, mesh in zip(axes, ("bunny", "dragon")):
        d = M.get(mesh, {})
        ts = sorted(int(t) for t in d if t != "end")
        for (k, lab, ls, mk), col in zip(keys, (BLUE, AQUA, ORANGE)):
            v = [d[str(t)][k]["rel"] * 100 for t in ts]
            ax.plot(ts, v, ls=ls, marker=mk, color=col, lw=2, ms=6, mec="white", label=lab)
            if d.get("end"):
                ax.plot(12, d["end"][k]["rel"] * 100, marker=mk, color=col, ms=7, mec="white")
        ax.axhline(0, color=GREY, lw=1)
        ax.set_title(f"{mesh}, seed 97", color=INK)
        ax.set_xticks([1, 2, 3, 4, 5, 6, 8, 10, 12])
        ax.set_xticklabels(["1", "2", "3", "4", "5", "6", "8", "10", "end"])
        ax.set_xlabel("simulated time (s)")
    axes[0].set_ylabel("render arm's lead over its twin (%)\n(twin's error - arm's) / twin's")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("When does the render arm's lead appear?", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig2_lead_over_time.png", dpi=160)
    plt.close(fig)

# ---------- (2) the switch at 6 s: error relative to the twin, around 6 s ----------
sw = out / "switch.json"
if sw.exists():
    S = json.load(open(sw))["series"]
    for m, title in (("yard_sil", "yardstick silhouette (vs the independent sample)"),
                     ("yard_pbr", "yardstick shading (vs the independent sample)"),
                     ("rec_sil", "the run's own silhouette term (every window's record)"),
                     ("outline_sd", "outline shape error against the exact mesh")):
        cases = [(mesh, seed) for mesh in ("bunny", "dragon") for seed in (97, 99)]
        fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
        any_ = False
        for j, (mesh, seed) in enumerate(cases):
            for i, arm in enumerate(("Loff6", "Pon6")):
                ax = axes[i, j]
                d = S.get(f"{mesh}_{seed}_{arm}", {}).get(m)
                if not d:
                    ax.set_axis_off()
                    continue
                rows = [r for r in d["rows"] if r.get("P")]
                t = [r["t"] for r in rows]
                lp = [r["L"] / r["P"] if r.get("L") is not None else np.nan for r in rows]
                xp = [r["X"] / r["P"] if r.get("X") is not None else np.nan for r in rows]
                ax.axhline(1.0, color=ORANGE, lw=1.5, ls="--", label="twin (= 1)")
                ax.plot(t, lp, color=BLUE, lw=2, label="render arm, render on all along")
                ax.plot(t, xp, color=AQUA, lw=2, ls="-", marker="o", ms=3, label="render switched at 6 s (off or on: panel title)")  # noqa: E501
                ax.axvline(6.0, color=GREY, lw=1)
                ax.set_title(f"{mesh} {seed}: render {'off' if arm == 'Loff6' else 'on'} from 6 s", color=INK, fontsize=9)
                any_ = True
        if not any_:
            plt.close(fig)
            continue
        for ax in axes[1]:
            ax.set_xlabel("simulated time (s)")
        for ax in axes[:, 0]:
            ax.set_ylabel("error / the twin's error")
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper right", ncol=3, fontsize=8)
        fig.suptitle(f"Switching the render at 6 s: {title}", color=INK, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / f"fig3_switch_{m}.png", dpi=160)
        plt.close(fig)
print("figures written to", out)
