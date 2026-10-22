"""d161_fig.py OUT_PNG COIN40 COIN300 BUNNY300 FEATURES -- D161: the response curve in two plain panels per N (CPU).

Left, per N: how much of a target relief (1 pitch) the run would keep, against the relief's wavelength: what the
pictures ask for, what the whole physics lets through, without the surface proximity, with the transport alone; the
real features' wavelength bands (bunny wrinkles, dragon scales) marked with their amplitude. Right: what holds one
pitch of relief, per term (transport, surface proximity, near band, spray cleanup), against the render's push."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def pmin(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    j = int(np.argmin(ys))
    if j == 0 or j == len(xs) - 1:
        return float(xs[j])
    c = np.polyfit(xs[j - 1:j + 2], ys[j - 1:j + 2], 2)
    return float(-c[1] / (2 * c[0])) if c[0] > 0 else float(xs[j])


def load(f, dt=1.0):
    rows = [json.loads(ln) for ln in open(f)]
    rows = sorted([r for r in rows if r.get("dt") == dt], key=lambda r: r["w"])
    out = []
    for r in rows:
        DS = r["ds"]
        lam = r["totals"]["adapt"]["lam"]
        Lr = [r["render"][str(d)]["L"] for d in DS]
        keep = {}
        for k, terms in (("all", "T P nn sp"), ("noP", "T nn sp"), ("Tonly", "T")):
            E = [sum(r["phys"][str(d)]["val"][t] for t in terms.split()) + lam * Lr[i] for i, d in enumerate(DS)]
            keep[k] = pmin(DS, E) / dt
        hold = r["phys"]["1.0"]["der"]
        out.append(dict(w=r["w"], ask=r["render_only"]["ds_star"] / dt, push=-lam * r["render"]["0.0"]["der"], hold=hold, **keep))
    return out


out, c40, c300, b300, featf = sys.argv[1:6]
feat = {f["mesh"]: f for f in json.load(open(featf))}
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for row, (name, f, key) in enumerate((("coin 40k", c40, "40k"), ("coin 300k", c300, "300k"))):
    d = load(f)
    w = [x["w"] for x in d]
    ax = axes[row, 0]
    ax.plot(w, [x["ask"] for x in d], "--", color="#555", label="what the pictures ask for")
    ax.plot(w, [x["all"] for x in d], "-o", color="#c0392b", lw=2.2, label="kept by the run (all physics terms)")
    ax.plot(w, [x["noP"] for x in d], "-s", color="#d68910", label="kept if the surface proximity did not act")
    ax.plot(w, [x["Tonly"] for x in d], "-^", color="#2471a3", label="kept if only the transport acted")
    for m, col in (("bunny", "#7f8c8d"), ("dragon", "#16a085")):
        b = feat[m]["bands"]
        for band in b[1:4]:
            lo, hi = band["wavelength_" + key]
            amp = band["top_rms_" + key]
            ax.axvspan(lo, hi, ymin=0.0 if m == "bunny" else 0.05, ymax=0.05 if m == "bunny" else 0.10, color=col, alpha=0.5)
        ax.text(b[2]["wavelength_" + key][0], 1.02 + (0.06 if m == "dragon" else 0.0),
                f"{'bunny wrinkles' if m == 'bunny' else 'dragon scales'}: {b[2]['top_rms_' + key]:.2f}-{b[3]['top_rms_' + key]:.2f} pitch",
                color=col, fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(-0.1, 1.2)
    ax.set_xlabel("relief wavelength (particle pitches)")
    ax.set_ylabel("share of a 1-pitch relief kept")
    ax.set_title(f"{name}: how much carved relief survives")
    ax.legend(fontsize=8, loc="center right")
    ax = axes[row, 1]
    T = np.array([x["hold"]["T"] for x in d])
    P = np.array([x["hold"]["P"] for x in d])
    N = np.array([x["hold"]["nn"] for x in d])
    S = np.array([x["hold"]["sp"] for x in d])
    tot = T + P + N + S
    x = np.arange(len(w))
    ax.bar(x, T / tot, color="#2471a3", label="transport")
    ax.bar(x, P / tot, bottom=T / tot, color="#c0392b", label="surface proximity")
    ax.bar(x, N / tot, bottom=(T + P) / tot, color="#d68910", label="near band")
    ax.bar(x, S / tot, bottom=(T + P + N) / tot, color="#7f8c8d", label="spray cleanup")
    for i, xx in enumerate(d):
        ax.text(i, 1.02, f"x{tot[i] / max(xx['push'], 1e-30):.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, [f"{v:.0f}" for v in w])
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("relief wavelength (particle pitches)")
    ax.set_ylabel("share of the physics' hold on 1 pitch of relief")
    ax.set_title(f"{name}: who holds the relief (above: hold / render push)")
    ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(out, dpi=130)
