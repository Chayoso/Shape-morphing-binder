"""WHERE the render channel changes dFc, on one run (docs/surface_gradient.md §6-7, §14; the gradient dump
of pipeline/optimizer.py --grad_dump). Per window and per particle, from the dump:

  share_p  = lam |g_rend,p| / (|g_phys,p| + lam |g_rend,p|)     the render channel's share of the control
                                                             gradient AT this particle (g = the leaf gradient
                                                             norm over the window's T steps, lam = the balancer's
                                                             lambda of the window) — where it exceeds 1/2 the
                                                             render channel decides the update of dFc there
  |g_rend,p| lam                                             the render pull on dFc, absolute (log colour)
  resp_rend,p = |xT_rend - xT_base| / spacing                the one-window response to the render channel's
                                                             gradient ALONE (same control norm as the accepted
                                                             step) — where the physics actually moves particles
                                                             when the render channel steers it
  resp_phys,p = |xT_phys - xT_base| / spacing                the same for the physics channel

Draws, for the chosen windows, one row of four orthographic particle maps (azimuth 35, elevation 18, the
photoreal view; depth-sorted), and prints the aggregate: the share of particles where the render channel
dominates, its mean on the outer layer vs the interior, the fraction of the render pull on the outer layer,
the correlation of the two responses.

usage: grad_where.py <dump_dir> <out_dir> [--wins 2,mid,last] [--az 35 --el 18]
"""
from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from physmorph.render.surface_recon import layer_by_asymmetry  # noqa: E402


def view(x, az=35.0, el=18.0):
    a, e = np.radians(az), np.radians(el)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    p = x @ (Rx @ Ry).T
    return p[:, 0], p[:, 1], np.argsort(p[:, 2])


def spacing_of(x):
    sub = x[np.random.default_rng(0).choice(len(x), min(len(x), 20000), replace=False)]
    return float(np.median(cKDTree(sub).query(sub, k=9, workers=-1)[0][:, -1])) * (min(len(x), 20000) / len(x)) ** (1 / 3)


def panel(ax, x, vals, title, cmap, vmin=None, vmax=None, log=False, s=1.4):
    px, py, order = view(x)
    v = np.log10(np.maximum(vals, 1e-12)) if log else vals
    sc = ax.scatter(px[order], py[order], c=v[order], s=s, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)
    ax.set_aspect("equal"); ax.set_axis_off(); ax.set_title(title, fontsize=10)
    return sc


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dump, out = args[0], args[1]
    wins_arg = sys.argv[sys.argv.index("--wins") + 1] if "--wins" in sys.argv else "2,mid,last"
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(dump, "win_*.npz")))
    n = len(files)
    sel = []
    for w in wins_arg.split(","):
        sel.append({"mid": n // 2, "last": n - 2, "first": 0}.get(w, int(w) if w.isdigit() else n // 2))
    sel = [min(max(k, 0), n - 1) for k in sel]
    rows = []
    lines = []
    for k in sel:
        z = np.load(files[k])
        x0 = z["x0"].astype(np.float32)
        sp = spacing_of(x0)
        lam = float(z["lam_r"])
        gp = z["gl_phys_pnorm"].astype(np.float64)
        gr = (z["gl_rend_pnorm"] if "gl_rend_pnorm" in z.files else z["gl_sil_pnorm"]).astype(np.float64)
        share = lam * gr / np.maximum(gp + lam * gr, 1e-30)
        pull = lam * gr
        xb = z["xT_base"].astype(np.float64)
        r_rend = np.linalg.norm(z["xT_rend"].astype(np.float64) - xb, axis=1) / sp if "xT_rend" in z.files else np.linalg.norm(z["xT_sil"].astype(np.float64) - xb, axis=1) / sp
        r_phys = np.linalg.norm(z["xT_phys"].astype(np.float64) - xb, axis=1) / sp
        mask, _ = layer_by_asymmetry(x0, sp)
        lay = mask > 0.5
        dom = share > 0.5
        stats = dict(win=k, lam=lam, g_share=float(z["g_share"]),
                     dom_all=float(dom.mean()), dom_layer=float(dom[lay].mean()), dom_int=float(dom[~lay].mean()),
                     share_layer=float(share[lay].mean()), share_int=float(share[~lay].mean()),
                     pull_on_layer=float(pull[lay].sum() / max(pull.sum(), 1e-30)), layer_frac=float(lay.mean()),
                     resp_rend_layer=float(r_rend[lay].mean()), resp_rend_int=float(r_rend[~lay].mean()),
                     resp_phys_layer=float(r_phys[lay].mean()), resp_phys_int=float(r_phys[~lay].mean()),
                     corr_resp=float(np.corrcoef(r_rend, r_phys)[0, 1]))
        line = (f"win {k:3d}: lam {lam:.3f} g_share {stats['g_share']:.3f} | render dominates dFc at {100 * stats['dom_all']:.1f} % of particles "
                f"(outer layer {100 * stats['dom_layer']:.1f} %, interior {100 * stats['dom_int']:.1f} %); mean share layer {stats['share_layer']:.2f} / interior {stats['share_int']:.2f}; "
                f"{100 * stats['pull_on_layer']:.0f} % of the render pull on the layer ({100 * stats['layer_frac']:.0f} % of particles) | "
                f"one-window response (spacings): render layer {stats['resp_rend_layer']:.3f} / interior {stats['resp_rend_int']:.3f}, "
                f"physics layer {stats['resp_phys_layer']:.3f} / interior {stats['resp_phys_int']:.3f}; corr(resp) {stats['corr_resp']:.2f}")
        print(line, flush=True); lines.append(line)
        rows.append((k, x0, share, pull, r_rend, r_phys, stats))
    fig, axes = plt.subplots(len(rows), 4, figsize=(17, 4.3 * len(rows)))
    axes = np.atleast_2d(axes)
    pmax = max(np.percentile(r[3], 99) for r in rows)
    rmax = max(np.percentile(np.concatenate([r[4], r[5]]), 99) for r in rows)
    for i, (k, x0, share, pull, r_rend, r_phys, st) in enumerate(rows):
        sc0 = panel(axes[i, 0], x0, share, f"window {k}: render share of the dFc gradient (>0.5 = render decides)", "RdBu_r", 0, 1)
        sc1 = panel(axes[i, 1], x0, pull / max(np.median(pull), 1e-30), f"window {k}: render pull on dFc, λ|g_rend| / median (log10)", "magma", -1.5, 1.5, log=True)
        sc2 = panel(axes[i, 2], x0, r_rend, f"window {k}: response to the render channel alone (spacings)", "viridis", 0, rmax)
        sc3 = panel(axes[i, 3], x0, r_phys, f"window {k}: response to the physics channel alone (spacings)", "viridis", 0, rmax)
        for sc, ax in ((sc0, axes[i, 0]), (sc1, axes[i, 1]), (sc2, axes[i, 2]), (sc3, axes[i, 3])):
            fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
    fig.suptitle("Where the render channel changes dFc — " + os.path.basename(os.path.normpath(dump)) + f" ({n} windows)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "grad_where.png"), dpi=110)
    with open(os.path.join(out, "grad_where.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    # the per-window trace: share of particles where render dominates, and on the layer
    tr = []
    for f in files:
        zz = np.load(f)
        lam = float(zz["lam_r"]); gp = zz["gl_phys_pnorm"].astype(np.float64)
        gr = (zz["gl_rend_pnorm"] if "gl_rend_pnorm" in zz.files else zz["gl_sil_pnorm"]).astype(np.float64)
        sh = lam * gr / np.maximum(gp + lam * gr, 1e-30)
        m = zz["layer_mask"] > 0.5 if "layer_mask" in zz.files else layer_by_asymmetry(zz["x0"].astype(np.float32), spacing_of(zz["x0"]))[0] > 0.5
        tr.append((float(zz["g_share"]), float((sh > 0.5).mean()), float((sh[m] > 0.5).mean()), float(sh[m].mean()), float(sh[~m].mean())))
    tr = np.array(tr)
    fig2, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(tr[:, 0], label="g_share (global: λ‖g_r‖ / (‖g_p‖ + λ‖g_r‖))", color="k")
    ax.plot(tr[:, 1], label="share of particles where render dominates", color="#B5442E")
    ax.plot(tr[:, 2], label="… on the outer layer", color="#B5442E", ls="--")
    ax.plot(tr[:, 3], label="mean per-particle share, outer layer", color="#1F6F78")
    ax.plot(tr[:, 4], label="mean per-particle share, interior", color="#1F6F78", ls="--")
    ax.set_xlabel("window"); ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=2)
    ax.set_title("the render channel's share of the dFc gradient, per window")
    fig2.tight_layout(); fig2.savefig(os.path.join(out, "grad_where_trace.png"), dpi=110)
    print("saved", os.path.join(out, "grad_where.png"), os.path.join(out, "grad_where_trace.png"))


if __name__ == "__main__":
    main()
