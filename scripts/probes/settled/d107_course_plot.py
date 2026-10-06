"""d107_course_plot.py OUT_PNG MESH LOG_DIR — D107 (1): the share of the mesh's relief the display carries along the morph,
per band (family B of ag2_bands.py), for D98's P and L and D105's PV and LV, from LOG_DIR/course_MESH_ARM.log (the target
first, then the run's kept frames in order and its end frame) and, for the render arms, early_MESH_ARM.log (earlier frames). The horizontal axis is the kept frame's raw step over the
run's last; the target's share is drawn as a dashed line (the cap at this N). Also the share of discs left out (farther
than two pitches from the mesh: the body not yet arrived) from LOG_DIR/course_p_MESH_ARM.log. Prints the rows."""
import json, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out, mesh, d = sys.argv[1], sys.argv[2], sys.argv[3]
BANDS = ("2.7", "5.4", "10.8", "21.6")
ARMS = {"P": ("D98 physics only", "#eb6834", "--"), "L": ("D98 with render", "#2a78d6", "--"),
        "PV": ("D105 physics only", "#eb6834", "-"), "LV": ("D105 with render", "#2a78d6", "-")}


def family_b(path):
    """{state name: {band: kept share}} from an ag2_bands.py log."""
    res, on = {}, False
    for line in open(path):
        if line.startswith("== family B"):
            on = True
            continue
        if on and line.strip() == "":
            break
        m = re.match(r"\s*[0-9.]+\s*-\s*[0-9.]+\s+([0-9.]+)\s", line)
        if not on or not m:
            continue
        band = m.group(1)
        for cell in line.split("|")[1:]:
            name, vals = cell.split(":", 1)
            kept = float(vals.split("/")[2].split("+-")[0])
            res.setdefault(name.strip(), {})[band] = kept
    return res


apart = {}
fig, ax = plt.subplots(1, len(BANDS), figsize=(4.4 * len(BANDS), 3.8), sharey=True)
for arm, (label, colour, ls) in ARMS.items():
    try:
        rows = family_b(f"{d}/course_{mesh}_{arm}.log")
    except FileNotFoundError:
        continue
    try:                                                        # the early frames (d107_early.sh), the render arms only
        rows.update(family_b(f"{d}/early_{mesh}_{arm}.log"))
    except FileNotFoundError:
        pass
    for log in (f"{d}/course_p_{mesh}_{arm}.log", f"{d}/early_p_{mesh}_{arm}.log"):
        try:
            for line in open(log):
                if line.startswith("{"):
                    r = json.loads(line)
                    apart[(arm, str(r["state"]))] = r["apart"]
        except FileNotFoundError:
            pass
    target = next(v for k, v in rows.items() if k.endswith("_target"))
    states = sorted(((int(k.rsplit("_", 1)[1]), v) for k, v in rows.items() if not k.endswith("_target")), key=lambda e: e[0])
    last = states[-1][0]
    for c, band in enumerate(BANDS):
        ax[c].plot([s / last for s, _ in states], [v[band] for _, v in states], color=colour, ls=ls, lw=2, marker="o", ms=4, label=label)
        ax[c].axhline(target[band], color="#52514e", ls=":", lw=1)
    print(arm, " ".join(f"raw {s}: " + "/".join(f"{v[b]:+.2f}" for b in BANDS) + f" (apart {apart.get((arm, str(s)), float('nan')):.3f})"
                        for s, v in states), "| target", "/".join(f"{target[b]:+.2f}" for b in BANDS), flush=True)
for c, band in enumerate(BANDS):
    ax[c].set_title(f"wavelength {band} pitches")
    ax[c].set_xlabel("morph progress (raw step / last)")
    ax[c].axhline(0, color="#e4e3df", lw=1)
ax[0].set_ylabel("share of the mesh's relief carried")
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle(f"{mesh} 300k: the relief along the morph (dotted: the 300k target sample, the cap)")
fig.tight_layout()
fig.savefig(out, dpi=130)
print("wrote", out)
