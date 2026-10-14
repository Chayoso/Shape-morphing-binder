"""fig145.py KIND OUT_PNG [inputs] -- D145's paper figures (matplotlib, light print surface; run on the server).

KIND loss     OUT B10_JSONL MESH RENDER_PREFIX TWIN_PREFIX: B10, the loss terms per window over simulated time, render arm
              (blue) against the λ = 0 twin (orange), the median of the seeds as a line and their range as a band; one
              panel a term (each on its own log axis): total merit, physics part, transport, released motion, end drift,
              silhouette, shading, local terms (physics part less transport and settling). Names are matched as
              MESH300k_<PREFIX><seed>, e.g. L / P.
KIND lead     OUT DIR MESH RENDER_PREFIX TWIN_PREFIX: the render lead over the trajectory, from read300.sh's every-frame
              readings (t3_: yardstick exterior silhouette and shading; e3_: display 1 - IoU front), the same encoding.
KIND gallery  OUT GAL_JSON: B3, per mesh the render arm minus the twin on the independent silhouette IoU at each seed
              (dots) and their mean (bar tick), meshes ordered by the mean, zero line.
KIND progress OUT HARNESS_JSONL [LABEL=Name ...]: the transport progress p over normalised time per method; ours blue,
              Xu et al. orange, Buonomo et al. aqua, OT displacement grey dashed; markers at p = 0.25 / 0.5 / 0.75.
Palette (validated with the dataviz skill's validator, light mode: blue #2a78d6, orange #eb6834, aqua #1baf7a pass the
adjacent CVD and normal-vision checks; aqua is below 3:1 contrast, so every series is labelled directly)."""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "font.size": 9, "lines.linewidth": 2.0, "savefig.dpi": 200})


def band(ax, curves, color, label):
    """curves: list of (t, y); the median on a common grid as a line, the min-max as a band, a direct label."""
    if not curves:
        return
    t0 = max(c[0][0] for c in curves)
    t1 = min(c[0][-1] for c in curves)
    grid = np.linspace(t0, t1, 200)
    ys = np.array([np.interp(grid, c[0], c[1]) for c in curves])
    ax.fill_between(grid, ys.min(0), ys.max(0), color=color, alpha=0.18, linewidth=0)
    ax.plot(grid, np.median(ys, 0), color=color, label=label)
    ax.annotate(label, (grid[-1], np.median(ys, 0)[-1]), xytext=(4, 0), textcoords="offset points", color=INK2,
                fontsize=8, va="center")


def by_prefix(rows, mesh, prefix):
    pat = re.compile(rf"^{mesh}300k_{prefix}(\d+)$")
    return [r for r in rows if pat.match(r["name"])]


def kind_loss(out, path, mesh, pa, pb):
    rows = [json.loads(s) for s in open(path) if s.startswith("{")]
    terms = [("outer_merit", "total merit"), ("outer_merit_phys", "physics part"), ("transport_energy", "transport"),
             ("stab", "released motion"), ("stab_end", "end drift"), ("d_sil", "silhouette"), ("d_pbr", "shading"),
             ("local", "local terms")]
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.6), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, terms):
        for prefix, color, label in ((pa, BLUE, "render arm"), (pb, ORANGE, "twin (λ = 0)")):
            curves = []
            for r in by_prefix(rows, mesh, prefix):
                w = [x for x in r["windows"] if x.get("outer_accepted") and x.get("t") is not None]
                if key == "local":
                    y = [x["outer_merit_phys"] - (x["transport_energy"] or 0) - (x["stab"] or 0) - (x["stab_end"] or 0)
                         if x.get("outer_merit_phys") is not None else None for x in w]
                else:
                    y = [x.get(key) for x in w]
                pts = [(x["t"], v) for x, v in zip(w, y) if v is not None and v > 0]
                if len(pts) > 1:
                    curves.append((np.array([p[0] for p in pts]), np.array([p[1] for p in pts])))
            band(ax, curves, color, label)     # the twin's render terms are recorded at weight zero, when present
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_yscale("log")
        ax.set_xlabel("simulated time (s)")
    fig.suptitle(f"{mesh} 300k: the loss terms per window (median of seeds, band = range)", fontsize=10)
    fig.savefig(out)


def frame_rows(path):
    try:
        return [json.loads(s) for s in open(path) if s.startswith("{")]
    except OSError:
        return []


def kind_lead(out, d, mesh, pa, pb):
    D = Path(d)
    panels = [("yardstick silhouette (exterior)", "t3", lambda r: r["exterior"]["sil"]),
              ("yardstick shading (exterior)", "t3", lambda r: r["exterior"]["pbr"]),
              ("display 1 - IoU (front)", "e3", lambda r: 1 - r["to_target"]["front"]["iou"])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    for ax, (title, kind, get) in zip(axes, panels):
        for prefix, color, label in ((pa, BLUE, "render arm"), (pb, ORANGE, "twin (λ = 0)")):
            curves = []
            for f in sorted(D.glob(f"{kind}_{mesh}300k_{prefix}[0-9]*.log")):
                if not re.search(rf"_{prefix}\d+\.log$", f.name):
                    continue
                pts = []
                for r in frame_rows(f):
                    try:
                        pts.append((int(r["state"]) / 240.0, get(r)))
                    except (KeyError, ValueError, TypeError):
                        continue
                pts.sort()
                if len(pts) > 1:
                    curves.append((np.array([p[0] for p in pts]), np.array([p[1] for p in pts])))
            band(ax, curves, color, label)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_yscale("log")
        ax.set_xlabel("simulated time (s)")
    fig.suptitle(f"{mesh} 300k: against an independent sample of the mesh, every kept frame", fontsize=10)
    fig.savefig(out)


def kind_gallery(out, path):
    rows = [r for r in json.load(open(path)) if r.get("diff")]
    rows.sort(key=lambda r: r["diff_mean"])
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    for i, r in enumerate(rows):
        ax.scatter([i] * len(r["diff"]), r["diff"], s=14, color=BLUE, zorder=3, edgecolors=SURF, linewidths=1)
        ax.plot([i - 0.3, i + 0.3], [r["diff_mean"]] * 2, color=INK, linewidth=2)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(range(len(rows)), [r["mesh"] for r in rows], rotation=60, ha="right")
    ax.set_ylabel("silhouette IoU, render arm - twin")
    ax.set_title("40k gallery against independent samples: three seeds (dots) and their mean (bar)", fontsize=10)
    fig.savefig(out)


def kind_progress(out, path, names):
    rows = [json.loads(s) for s in open(path) if s.startswith("{")]
    style = {"ours": (BLUE, "-"), "xu": (ORANGE, "-"), "buo": (AQUA, "-"), "ot": (GREY, "--")}
    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    for r in rows:
        name = names.get(r["label"], r["label"])
        key = next((k for k in style if name.lower().startswith(k)), "ours")
        color, ls = style[key]
        raw = np.array([c[0] for c in r["curve"]], float)
        p = np.array([c[1] for c in r["curve"]], float)
        tau = raw / max(raw[-1], 1)
        ax.plot(tau, p, color=color, linestyle=ls, label=name)
        for q in ("p0.25", "p0.5", "p0.75"):
            v = r["read"].get(q)
            if v:
                ax.scatter([v["tau"]], [v["p"]], s=36, color=color, edgecolors=SURF, linewidths=2, zorder=3)
        ax.annotate(name, (tau[-1], p[-1]), xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=8,
                    va="center")
    ax.set_xlabel("normalised time")
    ax.set_ylabel("transport progress p")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out)


if __name__ == "__main__":
    kind, out = sys.argv[1], sys.argv[2]
    if kind == "loss":
        kind_loss(out, *sys.argv[3:7])
    elif kind == "lead":
        kind_lead(out, *sys.argv[3:7])
    elif kind == "gallery":
        kind_gallery(out, sys.argv[3])
    elif kind == "progress":
        kind_progress(out, sys.argv[3], dict(s.split("=", 1) for s in sys.argv[4:]))
