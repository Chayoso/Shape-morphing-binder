"""d162d_trace.py OUT_MD OUT_PNG -- D162d's per-window weight trace (docs, D162d) on arm Lp: per committed window the weight
as a share of its first value (lambda / lambda_0) and the excess the low-passed pictures still read (e); at the kept
frames what is still to do (relief: 1 - the carved share of the ceiling, D161's reader; colour: the mis-coloured surface
above its floor as a share of the twin's, D160's reader). One panel per scene and seed, one 0-1 axis. The table: arrival,
the weight there, the first window with e <= 0.05 (R1's time), what is still to do then and at the end, and R1 (e <= 0.05
by the first kept frame within 0.1 of the run's last progress)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
R, D = G / "d162d" / "runs40", G / "d162d" / "after"
SCENES = (("coin", "coin40k_Lp_{s}"), ("bunny", "bunny40k_Lp_{s}"), ("band", "bunny40k_band_Lp_{s}"),
          ("ears", "bunny40k_ears_Lp_{s}"))
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"


def windows(name):
    arm = list(json.load(open(R / f"{name}.json"))["arms"].values())[0]
    com = [h for h in arm["history"] if h.get("frame_end") and not h.get("null_commit")]
    te = np.array([h.get("transport_energy") or np.nan for h in com])
    arr = int(np.argmax(te < 2 * te[-1]))
    lam0 = com[0].get("lam0") or com[0]["lambda"]
    t = np.array([h["frame_end"] / 240.0 for h in com])
    return t, np.array([h["lambda"] / lam0 for h in com]), np.array([h.get("render_excess") or 0.0 for h in com]), arr


def todo(sc, s):
    """(t, still to do) at the kept frames."""
    if sc in ("coin", "bunny"):
        f = D / f"read_{sc}_Lp.json"
        if not f.exists():
            return None
        d = json.load(open(f))
        ceil = np.mean([d["refs"][f"relief_s1{k}"]["relief"]["A"] for k in (97, 99) if f"relief_s1{k}" in d["refs"]])
        ser = d["runs"][f"full{s}"]["series"]
        return np.array([p["t"] for p in ser]), np.clip(1.0 - np.array([p["relief"]["A"] for p in ser]) / ceil, 0, 1.2)
    f = D / "read_colour.jsonl"
    if not f.exists():
        return None
    rows = {json.loads(line)["name"]: json.loads(line) for line in open(f)}
    a, tw = rows.get(f"bunny40k_{sc}_Lp_{s}"), rows.get(f"bunny40k_{sc}_P{s}")
    if a is None or tw is None or not a.get("timeline"):
        return None
    tl = np.asarray(a["timeline"], np.float64)
    fl, mt = a["floor"], tw["end"]["mis"]
    return tl[:, 0], np.clip((tl[:, 1] - fl) / max(mt - fl, 1e-9), 0, 1.2)


fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True, constrained_layout=True)
L = ["| scene | seed | arrival (s) | weight there | first e <= 0.05 (s) | still to do then | still to do at the end | "
     "R1 |", "|" + "---|" * 8]
for j, (sc, pat) in enumerate(SCENES):
    for i, s in enumerate((97, 99)):
        ax = axes[i, j]
        name = pat.format(s=s)
        if not (R / f"{name}.json").exists():
            ax.set_visible(False)
            continue
        t, w, e, arr = windows(name)
        ax.step(t, w, where="post", color=BLUE, lw=2, label="weight, share of its first value")
        ax.step(t, e, where="post", color=ORANGE, lw=2, ls="--", label="excess the smoothed pictures still read")
        td = todo(sc, s)
        if td is not None:
            ax.plot(td[0], td[1], color=AQUA, lw=2, ls=":", marker="o", ms=3,
                    label="still to do (letters to carve / paint to route)")
            ax.text(td[0][-1], td[1][-1], " to do", color=MUTED, va="center", fontsize=8)
        ax.text(t[-1], w[-1], " weight", color=MUTED, va="center", fontsize=8)
        ax.axvline(t[arr], color=GRID, lw=1)
        ax.set_title(f"{sc}, seed {s}", color=INK, fontsize=11)
        ax.set_ylim(-0.02, 1.15)
        ax.grid(True, color=GRID, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED)
        if i == 1:
            ax.set_xlabel("time (s)", color=MUTED)
        k0 = int(np.argmax(e <= 0.05)) if (e <= 0.05).any() else None
        t0 = t[k0] if k0 is not None else None
        r1 = None
        if td is not None:
            done = 1.0 - td[1]
            kstar = int(np.argmax(done >= done[-1] - 0.1))
            r1 = t0 is not None and t0 <= td[0][kstar] + 1e-9
            then = float(np.interp(t0, td[0], td[1])) if t0 is not None else None
        L.append(f"| {sc} | {s} | {t[arr]:.1f} | {w[arr]:.2f} | {'-' if t0 is None else f'{t0:.1f}'} | "
                 f"{'-' if td is None or t0 is None else f'{then:.2f}'} | {'-' if td is None else f'{td[1][-1]:.2f}'} | "
                 f"{'-' if r1 is None else ('yes' if r1 else 'no')} |")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("D162d (arm Lp): the render's weight per window, the excess it reads, and what is left to do "
             "(thin grey line: arrival)", color=INK)
fig.savefig(sys.argv[2], dpi=130, bbox_inches="tight", facecolor="#fcfcfb")
open(sys.argv[1], "w").write("\n".join(L) + "\n")
print("\n".join(L))
