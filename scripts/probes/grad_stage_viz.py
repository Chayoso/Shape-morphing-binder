"""Figures of the gradient-stage dumps (docs/surface_gradient.md §6-7; probes/grad_stage.py).

usage: grad_stage_viz.py --out <dir> --win <k> <name>=<dump_dir> [<name>=<dump_dir> ...]

The FIRST dump is the one whose window <k> is drawn as particle maps; every dump enters the
per-window and correlation plots. Particle maps are orthographic views (azimuth 35, elevation
18, the photoreal view), depth-sorted, interior particles grey where a field lives on the
outer layer only. Writes fig1_covector.png ... fig7_u.png.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from grad_stage import analyse, corr_at, spacing_of  # noqa: E402
from physmorph.render.surface_recon import layer_by_asymmetry, plane_residual  # noqa: E402

GREY = "#d9d4cb"


def view(x, az=35.0, el=18.0):
    a, e = np.radians(az), np.radians(el)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    p = x @ (Rx @ Ry).T
    order = np.argsort(p[:, 2])          # far first, near last
    return p[:, 0], p[:, 1], order


def nbr_mean(vals, P, sp, r_sp=2.0):
    kd = cKDTree(P)
    pairs = kd.query_pairs(r_sp * sp, output_type="ndarray")
    acc = vals.copy(); cnt = np.ones(len(vals))
    np.add.at(acc, pairs[:, 0], vals[pairs[:, 1]]); np.add.at(acc, pairs[:, 1], vals[pairs[:, 0]])
    np.add.at(cnt, pairs[:, 0], 1); np.add.at(cnt, pairs[:, 1], 1)
    return acc / cnt


def scatter_all(ax, x, vals, title, cmap="viridis", log=True, s=1.2):
    sx, sy, order = view(x)
    v = vals.copy()
    if log:
        v = np.log10(np.maximum(v, 1e-12) / max(float(v.max()), 1e-12))
        vmin, vmax = -3.0, 0.0
    else:
        vmin, vmax = None, None
    sc = ax.scatter(sx[order], sy[order], c=v[order], s=s, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=9)
    return sc


def scatter_layer(ax, x, mask, vals, title, cmap="RdBu_r", s=2.0, sym=True, vmax=None):
    sx, sy, order = view(x)
    m = mask[order]
    ax.scatter(sx[order][~m], sy[order][~m], c=GREY, s=0.6, linewidths=0, rasterized=True)
    v = np.full(len(x), np.nan); v[mask] = vals
    vv = v[order][m]
    if vmax is None:
        vmax = float(np.nanquantile(np.abs(vv), 0.98)) if sym else float(np.nanquantile(vv, 0.98))
    sc = ax.scatter(sx[order][m], sy[order][m], c=vv, s=s, cmap=cmap, vmin=(-vmax if sym else 0), vmax=vmax, linewidths=0, rasterized=True)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=9)
    return sc


def main():
    args = sys.argv[1:]
    out = args[args.index("--out") + 1]; win = int(args[args.index("--win") + 1])
    dumps = [a.split("=", 1) for a in args if "=" in a and not a.startswith("--")]
    os.makedirs(out, exist_ok=True)
    name0, dir0 = dumps[0]
    files0 = sorted(glob.glob(os.path.join(dir0, "win_*.npz")))
    z = np.load(files0[win])
    x0 = z["x0"].astype(np.float32); xT0 = z["xT0"].astype(np.float32)
    sp = spacing_of(x0)
    mask, nrm = layer_by_asymmetry(xT0, sp)
    _, npl = plane_residual(xT0, mask, nrm, sp)
    n_layer = np.zeros_like(nrm); n_layer[mask] = npl
    P = xT0[mask]
    chans = [("gx_phys", "physics"), ("gx_sil", "silhouette"), ("gx_pbr", "shading")]

    # ---- fig 1: the covector on the particles ---------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    for j, (c, lab) in enumerate(chans):
        g = z[c]
        e = np.linalg.norm(g, axis=1)
        share = float((e[mask] ** 2).sum() / ((e ** 2).sum() + 1e-30))
        scatter_all(axes[0, j], xT0, e, f"{lab}: |g| per particle (log, 3 decades)\nouter layer carries {share*100:.0f} % of |g|²")
        gn = (g[mask] * n_layer[mask]).sum(1)
        nsh = float((gn ** 2).sum() / ((g[mask] ** 2).sum() + 1e-30))
        scatter_layer(axes[1, j], xT0, mask, gn, f"{lab}: normal component on the layer (+ = outward pull)\n{nsh*100:.0f} % of the layer's |g|² along the normal")
    fig.suptitle(f"Stage 1 — terminal covectors on the particles ({name0}, window {win}, N {len(x0)}, layer {mask.mean()*100:.1f} % of particles)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_covector.png"), dpi=140); plt.close(fig)

    # ---- fig 2: smooth vs rough decomposition of the silhouette covector on the layer ---------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    for i, (c, lab) in enumerate([("gx_sil", "silhouette"), ("gx_pbr", "shading")]):
        g = z[c]; gn = (g[mask] * n_layer[mask]).sum(1)
        mean = nbr_mean(gn, P, sp); rough = gn - mean
        fr = float((rough ** 2).sum() / ((gn ** 2).sum() + 1e-30))
        vmax = float(np.quantile(np.abs(gn), 0.98))
        scatter_layer(axes[i, 0], xT0, mask, gn, f"{lab}: normal component g·n", vmax=vmax)
        scatter_layer(axes[i, 1], xT0, mask, mean, f"{lab}: its 2-spacing neighbourhood mean (the smooth part)", vmax=vmax)
        scatter_layer(axes[i, 2], xT0, mask, rough, f"{lab}: the residual (the ROUGH part) — {fr*100:.0f} % of the energy", vmax=vmax)
    fig.suptitle("Stage 1 — what a render covector asks of the outer layer: smooth (features, outline) vs rough (the sampling noise of the target images)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_rough.png"), dpi=140); plt.close(fig)

    # ---- fig 3: the two paths after the adjoint ------------------------------------------------------
    has_u = "u_final" in z.files
    lm = (z["layer_mask"] > 0.5) if has_u else None
    fig, axes = plt.subplots(2 if has_u else 1, 3, figsize=(13, 8.6 if has_u else 4.4), squeeze=False)
    for j, (c, lab) in enumerate([("gl_phys", "physics"), ("gl_sil", "silhouette"), ("gl_pbr", "shading")]):
        pn = z[c + "_pnorm"]
        scatter_all(axes[0, j], x0, pn, f"{lab} → STRESS control (dFc): per-particle |∂L/∂dFc| (log)\nthe grid's correlation length: corr 0.87 at 2 sp, 0.6 at 4, 0.2 at 8")
        if has_u:
            gu = z["gu_" + c[3:]]
            scatter_layer(axes[1, j], x0, lm, gu[lm], f"{lab} → u CHANNEL: ∂L/∂u on the layer (+ = push outward)\ncorr at 2 sp: {corr_at(gu[lm][:, None], x0[lm], sp, 2):.2f}")
    fig.suptitle(f"Stage 2 — the same covectors after the MPM adjoint: through the stress control (top) and through the position channel u (bottom) — {name0}, window {win}", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_paths.png"), dpi=140); plt.close(fig)

    # ---- fig 4: correlation vs distance, averaged over windows, every dump ------------------------------
    rs = (1, 2, 4, 8)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for j, (c, lab) in enumerate(chans):
        ax = axes[j]
        for name, d in dumps:
            fl = sorted(glob.glob(os.path.join(d, "win_*.npz")))
            cov, st, uu = [], [], []
            for f in fl:
                zz = np.load(f)
                xx0 = zz["x0"].astype(np.float32); xxT = zz["xT0"].astype(np.float32)
                spp = spacing_of(xx0)
                mk, nr = layer_by_asymmetry(xxT, spp)
                _, np_ = plane_residual(xxT, mk, nr, spp)
                gn = (zz[c][mk] * np_).sum(1)
                cov.append([corr_at(gn[:, None], xxT[mk], spp, r) for r in rs])
                st.append([corr_at(zz["gl_" + c[3:] + "_tmean"], xx0, spp, r) for r in rs])
                if "u_final" in zz.files:
                    lmm = zz["layer_mask"] > 0.5
                    gu = zz["gu_" + c[3:]][lmm]
                    uu.append([corr_at(gu[:, None], xx0[lmm], spp, r) for r in rs])
            ax.plot(rs, np.nanmean(cov, 0), "o--", label=f"{name}: covector on the layer")
            ax.plot(rs, np.nanmean(st, 0), "s-", label=f"{name}: through the stress control")
            if uu:
                ax.plot(rs, np.nanmean(uu, 0), "^-", label=f"{name}: through the u channel")
        ax.set_xscale("log", base=2); ax.set_xticks(rs); ax.set_xticklabels([str(r) for r in rs])
        ax.set_xlabel("neighbour distance (spacings)"); ax.set_ylabel("correlation"); ax.set_ylim(-0.05, 1.0)
        ax.set_title(f"{lab}", fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=6.5)
    fig.suptitle("Stage 2 — how much of each gradient's structure survives at each scale (mean over windows)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_corr.png"), dpi=140); plt.close(fig)

    # ---- fig 5: the window's response to each channel alone ----------------------------------------------
    xb = z["xT_base"].astype(np.float32)
    mb, nb_ = layer_by_asymmetry(xb, sp)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    for j, c in enumerate(["phys", "sil", "pbr"]):
        dx = z["xT_" + c].astype(np.float32) - xb
        dn = (dx[mb] * nb_[mb]).sum(1) / sp
        scatter_layer(axes[0, j], xb, mb, dn, f"{c} via the STRESS control: layer normal displacement (sp)\nmean |dx|: layer {np.linalg.norm(dx[mb], axis=1).mean()/sp:.2f}, interior {np.linalg.norm(dx[~mb], axis=1).mean()/sp:.2f}")
        if has_u and ("xT_" + c + "_u") in z.files:
            xb0 = z["xT_base_u0"].astype(np.float32); mb0, nb0 = layer_by_asymmetry(xb0, sp)
            dxu = z["xT_" + c + "_u"].astype(np.float32) - xb0
            dnu = (dxu[mb0] * nb0[mb0]).sum(1) / sp
            scatter_layer(axes[1, j], xb0, mb0, dnu, f"{c} via the u CHANNEL: layer normal displacement (sp)\nmean |dx|: layer {np.linalg.norm(dxu[mb0], axis=1).mean()/sp:.2f}, interior {np.linalg.norm(dxu[~mb0], axis=1).mean()/sp:.3f}")
        else:
            axes[1, j].axis("off")
    fig.suptitle(f"Stage 3 — what one window does when it follows ONE channel (same control norm as the accepted step) — {name0}, window {win}", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig5_response.png"), dpi=140); plt.close(fig)

    # ---- fig 6: per-window trajectories and the response RMS bars, every dump --------------------------------
    rows_all = {name: [analyse(f, None) for f in sorted(glob.glob(os.path.join(d, "win_*.npz")))] for name, d in dumps}
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.15, 1.4]})
    for name, rows in rows_all.items():
        w = np.arange(len(rows))
        axes[0].plot(w, [r["lam_r"] for r in rows], "o-", label=name)
        axes[1].plot(w, [r["g_share"] for r in rows], "o-", label=name)
        axes[2].plot(w, [r["rms_xT0"] for r in rows], "o--", label=f"{name} start")
        axes[2].plot(w, [r.get("rms_final", np.nan) for r in rows], "s-", label=f"{name} end")
    axes[0].set_title("λ (render weight)"); axes[1].set_title("g_share (render share of the accepted step)")
    axes[2].set_title("outer-layer plane-residual RMS (spacings)\nwindow start (dashed) → end (solid)")
    for ax in axes[:3]:
        ax.set_xlabel("window"); ax.grid(alpha=0.3); ax.legend(fontsize=6.5)
    # bars: layer RMS at the window's end per channel, stress path (all dumps) and u path (dumps with u)
    labels, vals, cols = [], [], []
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, (name, rows) in enumerate(rows_all.items()):
        m = lambda key: np.nanmean([r[key] for r in rows if key in r])
        for c in ("base", "actual"):
            labels.append(f"{name}\n{c}"); vals.append(m("rms_base") if c == "base" else m("rms_final")); cols.append(palette[k % 10])
        for c in ("phys", "sil", "pbr"):
            labels.append(f"{name}\n{c}/stress"); vals.append(m(f"rms_{c}")); cols.append(palette[k % 10])
        if any(f"rmsu_{c}" in r for r in rows for c in ("sil",)):
            for c in ("phys", "sil", "pbr"):
                labels.append(f"{name}\n{c}/u"); vals.append(m(f"rmsu_{c}")); cols.append(palette[k % 10])
    axes[3].bar(range(len(vals)), vals, color=cols)
    axes[3].set_xticks(range(len(vals))); axes[3].set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=6, rotation=70, ha="right")
    axes[3].set_title("layer RMS at the window's end\nfollowing ONE channel (mean over windows)", fontsize=10); axes[3].grid(alpha=0.3, axis="y")
    axes[3].set_ylabel("spacings")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig6_windows.png"), dpi=140); plt.close(fig)

    # ---- fig 7: the accepted u field ----------------------------------------------------------------------
    if has_u:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
        uf = z["u_final"] / sp
        scatter_layer(axes[0], x0, lm, uf[lm], f"accepted u of window {win} on the layer (spacings; + = outward)\nRMS {np.sqrt((uf[lm]**2).mean()):.2f} sp, at the clip {(np.abs(uf[lm]) >= 0.999).mean()*100:.0f} %", vmax=1.0)
        allu = np.concatenate([np.load(f)["u_final"][np.load(f)["layer_mask"] > 0.5] / spacing_of(np.load(f)["x0"]) for f in files0])
        axes[1].hist(allu, bins=80, color="#2f6f73"); axes[1].axvline(-1, color="k", ls="--"); axes[1].axvline(1, color="k", ls="--")
        axes[1].set_title("u / spacing over all windows (dashed: the one-spacing clip)"); axes[1].set_xlabel("u (spacings)"); axes[1].grid(alpha=0.3)
        rows = rows_all[name0]
        axes[2].plot(range(len(rows)), [r["u_rms"] for r in rows], "o-", label="RMS u (spacings)")
        axes[2].plot(range(len(rows)), [r["u_clip"] for r in rows], "s-", label="fraction at the clip")
        axes[2].set_xlabel("window"); axes[2].grid(alpha=0.3); axes[2].legend(fontsize=7); axes[2].set_title("the channel's use per window")
        fig.suptitle(f"The position-mode control channel u ({name0})", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig7_u.png"), dpi=140); plt.close(fig)
    print("figures written to", out)


if __name__ == "__main__":
    main()
