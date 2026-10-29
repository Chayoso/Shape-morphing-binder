"""ri_figs.py OUT_DIR -- render influence (2026-10-22; CPU): the four panels and the combined figure.

1 fig1_distance.png  how far apart the full arm's and the twin's surfaces are over time, against the run-to-run noise
                     (D159's R2 reader: course_tag = L97 vs P97, course_floor = P99 vs P97; this probe: course_rep =
                     RG97 vs L97, course_full2 = RG97 vs P97)
2 fig2_push.png      the render's share and size of the push per window (L97's history via D159's window.jsonl; the
                     surface layer's share from RGV's tag check of RG97), and the surface's move per window
3 fig3_endmap.png    where on the end surface the render changed the shape (ri_endmap / ri_endrender)
4 fig4_end.png       the end against the exact mesh (D157 table: per-mesh median r) and the render term's lead split
                     into arrangement and shape (D156h2)
Arms: "dF_c + layer u + render loss (full)" (L) and "dF_c + layer u" (the lambda = 0 twin, P)."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
T = Path("/data/relcfd/chayo/physmorph_v2/tmp/render_influence")
MESHES = ("bunny", "dragon")
FULL, FULL2 = "#f0934f", "#f4c7a1"          # the render / the full arm (RGV's render orange; its re-run lighter)
INK, INK2, MUTED, GRID = "#eceef0", "#b8bfc7", "#a0a8b2", "#5c636c"   # RGV's text inks on the studio backdrop
LGREY, DARKTXT = "#bec4cc", "#16191d"        # RGV's physics grey; dark text on light bars
NOISE = "#c9cfd6"                           # the run-to-run band (drawn at alpha 0.30 on the backdrop)
SURF = "none"                               # transparent; the studio backdrop is composited under it
FULLNAME = "dF_c + layer u + render loss (full)"
TWINNAME = "dF_c + layer u (twin, λ = 0)"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#78808a", "text.color": INK, "legend.labelcolor": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False, "axes.axisbelow": True})


def backdrop(w, h):
    """StudioRaster's backdrop (the linear gradient (.115,.137,.165) -> (.215,.237,.265) top to bottom, tone map, sRGB)."""
    s = (np.arange(h) / max(1, h - 1))[:, None]
    lin = (1 - s) * np.array((.115, .137, .165)) + s * np.array((.215, .237, .265))
    m = lin / (1 + lin)
    srgb = np.where(m <= .0031308, 12.92 * m, 1.055 * np.power(m, 1 / 2.4) - .055)
    return Image.fromarray(np.repeat((srgb * 255 + .5).astype(np.uint8)[:, None, :], w, 1), "RGB")


def save(fig, path, dpi):
    """The figure drawn on a transparent canvas, then put on the studio backdrop (the RGV videos' look)."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.25, transparent=True)
    plt.close(fig)
    im = Image.open(path).convert("RGBA")
    bg = backdrop(*im.size).convert("RGBA")
    Image.alpha_composite(bg, im).convert("RGB").save(path)


def course(path, label):
    t, v = [], []
    for ln in open(path):
        r = json.loads(ln)
        if r["t_s"] is None or label not in r["arms"]:
            continue
        t.append(r["t_s"])
        v.append(r["arms"][label]["rms_a"])
    o = np.argsort(t)
    return np.asarray(t)[o], np.asarray(v)[o]


def windows():
    W = {}
    for p in (G / "d159/read/window.jsonl",):
        for ln in open(p):
            r = json.loads(ln)
            W[r["label"]] = r
    return W


def arrival_s(W, mesh):
    return (W[f"{mesh}_tag_L97"]["arrival_window"] + 1) / 6.0


def fig1(out, W, S):
    fig, axs = plt.subplots(2, 1, figsize=(11, 7.6), constrained_layout=True)
    for ax, mesh in zip(axs, MESHES):
        R = G / "d159/read"
        tA, A = course(R / f"course_tag_{mesh}.jsonl", "L97")
        tB, B = course(out / f"course_full2_{mesh}.jsonl", "RG97")
        tN, N = course(out / f"course_rep_{mesh}.jsonl", "RG97")
        tS, Sd = course(R / f"course_floor_{mesh}.jsonl", "P99")
        ta = arrival_s(W, mesh)
        ax.axvspan(0, ta, color="white", alpha=0.07, lw=0, zorder=0)
        ax.fill_between(tN, 0, N, color=NOISE, alpha=0.30, lw=0, zorder=1,
                        label="run-to-run noise: the full arm run twice, same seed (L97 vs RG97)")
        ax.plot(tS, Sd, color=LGREY, lw=1.5, ls=(0, (4, 3)), zorder=2,
                label="the twin at two seeds (P97 vs P99): another valid run of the twin")
        ax.plot(tB, B, color=FULL2, lw=2, zorder=3, label="full vs twin, the full arm's re-run (RG97 vs P97)")
        ax.plot(tA, A, color=FULL, lw=2, zorder=4, label="full vs twin (L97 vs P97): what the render changed + noise")
        tmax = max(tA.max(), tB.max(), tN.max(), tS.max())
        ax.set_xlim(0, tmax)
        ymax = 1.15 * max(A.max(), B.max(), N.max(), Sd.max())
        ax.set_ylim(0, ymax)
        ax.text(ta / 2, 0.97 * ymax, "arriving", ha="center", va="top", color=INK2, fontsize=9)
        ax.annotate(f"arrival: 90 % of the motion done ({ta:.1f} s)", xy=(ta, 0.80 * ymax), xytext=(ta + 0.02 * tmax, 0.80 * ymax),
                    color=INK2, fontsize=9, va="center")
        ax.axvline(ta, color=INK2, lw=1, zorder=5)
        # numbers: the peak during arrival and the end
        k = np.argmax(A[tA <= ta + 0.5])
        ax.annotate(f"{A[k]:.2f}", xy=(tA[k], A[k]), xytext=(4, 4), textcoords="offset points", color=FULL, fontsize=9)
        ax.annotate(f"end {A[-1]:.2f}", xy=(tA[-1], A[-1]), xytext=(-4, 8), textcoords="offset points", ha="right",
                    color=FULL, fontsize=9)
        ax.annotate(f"noise {N[-1]:.2f}", xy=(tN[-1], N[-1]), xytext=(-4, -12), textcoords="offset points", ha="right",
                    color=INK2, fontsize=9)
        ax.set_title(f"{mesh} 300k", loc="left", fontsize=12, color=INK)
        ax.set_ylabel("surface apart (rms, pitches)")
        S[mesh]["fig1"] = dict(arrival_s=ta, full_twin_peak=float(A[k]), full_twin_peak_t=float(tA[k]),
                               full_twin_end=float(A[-1]), full2_twin_end=float(B[-1]), rep_end=float(N[-1]),
                               seed_twin_end=float(Sd[-1]),
                               rep_peak=float(N[tN <= ta + 0.5].max()), seed_peak=float(Sd[tS <= ta + 0.5].max()),
                               full2_peak=float(B[tB <= ta + 0.5].max()),
                               at={str(x): dict(full_twin=float(np.interp(x, tA, A)), full2_twin=float(np.interp(x, tB, B)),
                                                rep=float(np.interp(x, tN, N)), seed=float(np.interp(x, tS, Sd)))
                                   for x in (0.5, 1.0, 2.0, 4.0)})
    axs[1].set_xlabel("simulated time (s)")
    axs[0].legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.0, 0.72))
    fig.suptitle("1  How far apart are the full arm's and the twin's surfaces over time?", x=0.01, ha="left",
                 fontsize=16, color=INK)
    fig.text(0.01, -0.035, "Surface apart = rms over the twin's outer-layer particles of the difference of the two surfaces' "
             "exterior fields (Zhu-Bridson, radius 3: the surface the loss reads), i.e. how far one surface lies outside or "
             "inside the other along the normal. 1 pitch = 0.708 x the target sample's median 8th-neighbour distance "
             f"(the bodies are {S['bunny']['size_pitch']:.0f} (bunny) and {S['dragon']['size_pitch']:.0f} (dragon) pitches across). Tag freeze-2026-10-15, seed 97. Each line ends where the shorter run of its pair ends (the stop rule "
             "ends runs at different times); the twin at two seeds differs already at t = 0 by its different particle sample.",
             fontsize=7.5, color=MUTED, wrap=True)
    save(fig, out / "fig1_distance.png", 150)
    plt.close(fig)


def fig2(out, W, S):
    fig, axs = plt.subplots(3, 2, figsize=(13, 9.4), constrained_layout=True, sharex="col")
    tc = {m: json.load(open(T / f"{m}_tagcheck.json"))["new"]["rows"] for m in MESHES}
    for j, mesh in enumerate(MESHES):
        L, P = W[f"{mesh}_tag_L97"]["per_window"], W[f"{mesh}_twin_P97"]["per_window"]
        t = (np.array([r["w"] for r in L]) + 0.5) / 6.0
        tp = (np.array([r["w"] for r in P]) + 0.5) / 6.0
        f = lambda rows, k: np.array([np.nan if r[k] is None else r[k] for r in rows], float)   # noqa: E731
        lam_gr, gp, share, mv, mvp = f(L, "lam_gr"), f(L, "gp"), f(L, "share"), f(L, "move_sp"), f(P, "move_sp")
        ta = arrival_s(W, mesh)
        k = W[f"{mesh}_tag_L97"]["arrival_window"]
        rows = tc[mesh]
        tl = np.array([(r["t0"] + r["t1"]) / 2 for r in rows])
        agg = np.array([r["agg"] for r in rows])
        for i in range(3):
            ax = axs[i, j]
            ax.axvspan(0, ta, color="white", alpha=0.07, lw=0, zorder=0)
            ax.axvline(ta, color=INK2, lw=1)
            ax.set_xlim(0, max(t.max(), tp.max()))
        ax = axs[0, j]
        ax.plot(t, share, color=LGREY, lw=1.5, ls=(0, (4, 3)), label="all controls: fixed at 1/3 by the balancing rule")
        ax.plot(tl, agg, color=FULL, lw=2, label="on the surface layer (measured on the re-run RG97)")
        ax.set_ylim(0, 0.6)
        ax.set_ylabel("render's share of the push")
        ax.set_title(f"{mesh} 300k", loc="left", fontsize=12, color=INK)
        ax = axs[1, j]
        ok = np.isfinite(gp) & np.isfinite(lam_gr)
        ax.plot(t[ok], gp[ok] * 1e4, color=LGREY, lw=2, label="physics push |g_physics|")
        ax.plot(t[ok], lam_gr[ok] * 1e4, color=FULL, lw=2, label="render push λ|g_render|")
        ax.plot(t[~ok], np.zeros((~ok).sum()), ls="none", marker="|", ms=9, mew=1.5, color=MUTED,
                label="window nulled by the commit check (no step, no record)")
        bef, aft = slice(0, k + 1), slice(k + 1, None)
        mb, ma = np.nanmedian(gp[bef]), np.nanmedian(gp[aft])
        rb, ra = np.nanmedian(lam_gr[bef]), np.nanmedian(lam_gr[aft])
        lam = f(L, "lam")
        lb, la = np.nanmedian(lam[bef]), np.nanmedian(lam[aft])
        ymax = np.nanmax(gp) * 1e4
        ax.set_ylim(-0.03 * ymax, 1.08 * ymax)
        ax.text(0.98, 0.92, f"median before arrival: physics {mb * 1e4:.2f}, render {rb * 1e4:.2f}\n"
                f"median after arrival: physics {ma * 1e4:.3f}, render {ra * 1e4:.3f}  ({mb / ma:.0f}× smaller)\n"
                f"render weight λ: {lb:.4f} → {la:.4f} ({lb / la:.0f}× smaller)",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
        ax.set_ylabel("push size (×10⁻⁴, gradient norm)")
        ax = axs[2, j]
        ax.plot(tp, mvp, color=LGREY, lw=1.5, ls=(0, (4, 3)), label=TWINNAME)
        ax.plot(t, mv, color=FULL, lw=2, label=FULLNAME)
        mvb, mva = np.nanmedian(mv[bef]), np.nanmedian(mv[aft])
        ax.text(0.98, 0.92, f"median move per window: {mvb:.2f} before, {mva:.3f} after arrival",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
        ax.set_ylim(0, 1.08 * np.nanmax(np.r_[mv, mvp]))
        ax.set_ylabel("surface moved per window\n(target spacings)")
        ax.set_xlabel("simulated time (s); one window = 1/6 s")
        S[mesh]["fig2"] = dict(arrival_window=k, phys_before=mb, phys_after=ma, rend_before=rb, rend_after=ra,
                               lam_before=float(lb), lam_after=float(la), nulled_windows=int((~ok).sum()),
                               move_before=mvb, move_after=mva, layer_share_before=float(np.median(agg[tl <= ta])),
                               layer_share_after=float(np.median(agg[tl > ta])), windows=len(L))
    axs[0, 0].legend(loc="upper right", fontsize=8.5)
    axs[1, 0].legend(loc="center right", fontsize=8.5)
    axs[2, 0].legend(loc="center right", fontsize=8.5)
    fig.suptitle("2  How strong is the render's push over time?", x=0.01, ha="left", fontsize=13, color=INK,
                 )
    fig.text(0.01, -0.05, "Per optimisation window of the delivered run L97 (its history; D159 R6's reader). The rule sets "
             "λ so that λ|g_render| = |g_physics| / 2 in every window: the render's share of the whole push is 1/3 "
             "by construction, so its size falls with the physics' after arrival. Surface-layer share: Σ|λg_render| / "
             "Σ(|λg_render| + |g_physics|) over the outer layer's particles, every non-render term counted as physics "
             "(RGV's tag check on RG97, the same arm and seed run again). Shaded: arriving (until 90 % of the run's motion is "
             "done).", fontsize=7.5, color=MUTED, wrap=True)
    save(fig, out / "fig2_push.png", 150)
    plt.close(fig)


def fig3(out, S, dmax):
    fig = plt.figure(figsize=(12, 15.0))
    W_, H_ = 0.47, 0.47 * 12 / 15.0                              # square image axes, manual layout
    for i, mesh in enumerate(MESHES):
        E = json.load(open(out / f"endmap_{mesh}.json"))
        S[mesh]["fig3"] = E
        for j, vn in enumerate(("front", "back")):
            ax = fig.add_axes([0.02 + j * 0.49, (0.586 if i == 0 else 0.159), W_, H_])
            img = np.asarray(Image.open(out / f"endmap_{mesh}_{vn}_map.png"))
            ax.imshow(img)
            ax.set_axis_off()
            ax.set_title(f"{mesh} 300k, end, {vn}", loc="left", fontsize=12, color=INK)
            if j == 1:
                tw = E["toward_mesh"]["L97"]
                pr = E["pair_rms"]
                s_f2 = np.mean([pr[k] ** 2 for k in ("L97-RG97", "L97-T97", "RG97-T97")]) / 2   # one full run's noise
                s_t2 = pr["P99-P99r"] ** 2 / 2                                                 # one twin run's noise
                d2 = np.mean([v ** 2 for v in E["delta_rms"].values()])
                eff = float(np.sqrt(max(d2 - s_f2 - s_t2, 0.0)))
                E["systematic_rms"], E["one_run_noise_rms"] = eff, float(np.sqrt((s_f2 + s_t2) / 2))
                noise = E.get("beyond_share_pairs_loo") or E.get("beyond_share_pairs") or {}
                nz = (f"; the same rule on a re-run pair: {100 * min(noise.values()):.0f}-{100 * max(noise.values()):.0f} %"
                      if noise else "")
                fig.text(0.98, (0.578 if i == 0 else 0.151),
                        f"coloured (beyond noise): {100 * E['beyond_share']:.0f} % of the surface "
                        f"({100 * E['beyond_out_share']:.0f} % out, {100 * E['beyond_in_share']:.0f} % in){nz}\n"
                        f"of the coloured area {100 * tw['beyond']:.0f} % moved toward the true mesh "
                        f"(error there {tw['mean_abs_err_twin_beyond']:.2f} → {tw['mean_abs_err_full_beyond']:.2f} pitch)\n"
                        f"change full vs twin {E['delta_rms']['L97']:.2f} pitch rms (re-run pairs "
                        f"{min(pr.values()):.2f}-{max(pr.values()):.2f}); its reproducible part "
                        f"≈ {eff:.2f} pitch rms, one run's own noise ≈ {E['one_run_noise_rms']:.2f}",
                         ha="right", va="top", fontsize=8.5, color=INK2)
    cax = fig.add_axes([0.06, 0.085, 0.88, 0.014])
    from matplotlib.colors import LinearSegmentedColormap
    grey = tuple(np.array([0.64, 0.69, 0.72]) ** (1 / 2.2))
    cm = LinearSegmentedColormap.from_list("ri", ["#2a78d6", grey, "#e34948"])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-dmax, dmax), cmap=cm), cax=cax, orientation="horizontal")
    cb.set_label("blue: the render arm's end surface lies INSIDE the twin's;  red: OUTSIDE  (pitches);  "
                 "grey: within run-to-run noise at that spot", color=INK2)
    fig.suptitle("3  Where on the surface did the render change the end shape?", x=0.01, y=0.995, ha="left", fontsize=16,
                 color=INK)
    fig.text(0.01, 0.045, "Body: the twin's end display surface (zbs) in the studio look. Value: the full arm L97's exterior "
             "field minus the twin P97's at each point (how far the full arm's surface lies outside or inside). Noise at a spot: "
             "the rms of the same difference between re-runs (L97 / RG97 / T97 of the full arm at seed 97, and the twin at seed 99 "
             "twice), averaged over a box about 4.5 pitches wide; grey where the change is below twice that. 'Toward the true "
             "mesh': each surface's distance from the exact target mesh grown by D157's offset c. Reproducible part: the mean square "
             "change of the three full runs minus one full run's and one twin run's noise (half the re-run pairs' mean "
             "squares). Images flipped back left-right "
             f"(the studio camera mirrors). One colour scale (±{dmax:g} pitch) for both meshes.", fontsize=7.5,
             color=MUTED, wrap=True, va="top")
    pass                                        # the end-map numbers only; panel 3 is the video and its hero sheet
    plt.close(fig)


def fig4(out, S):
    M = json.load(open(T / "d157_table.json"))["metrics"]
    meas = [("sil/zb/cor/1024/held/biou2pct_err", "outline near the true one\n(Boundary IoU, unseen views)"),
            ("sil/zb/cor/1024/held/chamfer_a", "outline distance\n(contour Chamfer, unseen views)"),
            ("d3/zb/cor/f0.5", "surface within half a pitch\n(3D F-score)"),
            ("d3/zb/cor/chamfer_l1_a", "surface distance\n(3D Chamfer)"),
            ("d3/zb/cor/nc", "normals agree\n(normal consistency)"),
            ("img/cor/shaded/held/mse", "studio picture error\n(unseen views)")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 7.0), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.5, 1]))
    y = np.arange(len(meas))[::-1]
    h = 0.26
    rows = {}

    def scale(m):
        x = M[f"300k:{m}"]
        tw, ar = x["twin_med"], x["arm_med"]
        A = "→"
        if m.endswith("biou2pct_err"):
            return f"Boundary IoU {1 - tw:.3f} (twin) {A} {1 - ar:.3f} (full)"
        if m.endswith("chamfer_a") or m.endswith("chamfer_l1_a"):
            return f"{tw:.2f} (twin) {A} {ar:.2f} (full) pitch"
        if m.endswith("f0.5"):
            return f"F-score {1 - tw:.2f} (twin) {A} {1 - ar:.2f} (full)"
        if m.endswith("nc"):
            return f"NC {1 - tw:.3f} (twin) {A} {1 - ar:.3f} (full)"
        if m.endswith("mse"):
            return f"PSNR {10 * np.log10(1 / tw):.1f} (twin) {A} {10 * np.log10(1 / ar):.1f} (full) dB"
        return ""
    meas = [(m, f"{lab}\n{scale(m)}") for m, lab in meas]
    for off, key, col, name in ((h, "bunny", FULL, "bunny (median of 5 seeds)"), (0, "dragon", "#86b6ef", "dragon (5 seeds)"),
                                (-h, "all", LGREY, "all six 300k meshes (median of 22 pairs)")):
        v = [100 * (M[f"300k:{m}"]["r_med"] if key == "all" else M[f"300k:{m}"]["per_mesh_r"][key]) for m, _ in meas]
        rows[key] = v
        a1.barh(y + off, v, height=h * 0.9, color=col, label=name)
        for yy, vv in zip(y + off, v):
            a1.text(vv + 0.6, yy, f"{vv:.0f} %", va="center", fontsize=8, color=INK2)
    a1.set_yticks(y)
    a1.set_yticklabels([lab for _, lab in meas], fontsize=8.5)
    a1.set_xlabel("share of the twin's error the render loss removes (%)")
    a1.axvline(0, color="#c3c2b7", lw=1)
    a1.set_xlim(0, 1.18 * max(max(v) for v in rows.values()))
    a1.legend(loc="lower right", fontsize=8.5)
    a1.set_title("Is the end closer to the exact target mesh?  (D157, held-out)", loc="left", fontsize=12,
                 color=INK)
    S["fig4_d157"] = {lab.split("\n")[0]: dict(zip(("bunny", "dragon", "all"), (rows["bunny"][i], rows["dragon"][i], rows["all"][i])))
                      for i, (_, lab) in enumerate(meas)}
    H = [json.loads(ln) for ln in open(G / "d156h/d156h2.jsonl") if ln.strip()]
    H = [r for r in H if "L" in r and "floor" in r]
    bars = {}
    for mesh in MESHES:
        rr = [r for r in H if r["mesh"] == mesh]
        seen = {}
        for r in rr:
            seen[r["seed"]] = r                                 # the last line of a seed (the relaunch)
        rr = list(seen.values())
        L = np.array([r["L"]["total"] for r in rr])
        P = np.array([r["P"]["total"] for r in rr])
        Fl = np.array([r["floor"]["total"] for r in rr])
        lead = P - L
        arr = (Fl - L) / lead
        bars[mesh] = dict(arrangement=float(np.mean(arr)), rest=float(1 - np.mean(arr)), arr_min=float(arr.min()),
                          arr_max=float(arr.max()), seeds=sorted(seen), L_over_floor=float(np.mean(L / Fl)),
                          P_over_floor=float(np.mean(P / Fl)))
    S["fig4_d156h2"] = bars
    yb = np.arange(len(MESHES))[::-1]
    for yy, mesh in zip(yb, MESHES):
        b = bars[mesh]
        a2.barh(yy, 100 * b["arrangement"], height=0.36, color=FULL2, label="arrangement: an ordered, smoother surface layer\n"
                "(the part below what a uniform sample of the true mesh scores)" if mesh == "bunny" else None)
        a2.barh(yy, 100 * b["rest"], left=100 * b["arrangement"] + 0.6, height=0.36, color=FULL,
                label="the rest: at most shape" if mesh == "bunny" else None)
        a2.text(100 * b["arrangement"] / 2, yy, f"{100 * b['arrangement']:.0f} %\n({100 * b['arr_min']:.0f}-{100 * b['arr_max']:.0f})",
                ha="center", va="center", fontsize=8.5, color=DARKTXT)
        a2.text(100 * b["arrangement"] + 100 * b["rest"] / 2, yy, f"{100 * b['rest']:.0f} %", ha="center", va="center",
                fontsize=8.5, color=DARKTXT)
    a2.set_yticks(yb)
    a2.set_yticklabels([f"{m} 300k" for m in MESHES])
    a2.set_xlim(0, 101)
    a2.set_xlabel("the full arm's lead over the twin on its own render score (% of the lead)")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.5, ncol=1)
    a2.set_title("How much of the render's own score is shape vs arrangement?  (D156h2)", loc="left", fontsize=12,
                 color=INK)
    a2.grid(axis="y", visible=False)
    fig.suptitle("4  Is the end better, and what is the render's own score made of?", x=0.01, ha="left", fontsize=16,
                 color=INK)
    fig.text(0.01, -0.07, "Left: under each label the measure for the twin and the full arm (medians over all 22 pairs). r = (twin's error - full arm's error) / twin's error against the exact target mesh grown by "
             "the surface offset c (D157's corrected referee), on 46 views the loss never used; per-mesh medians over seeds "
             "97-105 (bunny, dragon) and the median over all 22 pairs of six meshes; every bar is positive in 22 of 22 pairs "
             "except contour Chamfer and normals (21 of 22). Right: the render term the run optimises, at the end, for the "
             "full arm L, the twin P and eight uniform samples of the true mesh (the floor); the lead P - L split at the floor "
             "(mean of five seeds, range in brackets). A body of the true shape sampled uniformly does not score below the "
             "floor, so the part below it is the surface layer's order, not shape.", fontsize=7.5, color=MUTED, wrap=True)
    save(fig, out / "fig4_end.png", 150)
    plt.close(fig)


def combine(out):
    names = ["fig1_distance.png", "fig2_push.png"] + (["fig3_hero_sheet.png"] if (out / "fig3_hero_sheet.png").exists() else []) + ["fig4_end.png"]
    ims = [Image.open(out / f).convert("RGB") for f in names]
    w = max(i.width for i in ims)
    ims = [i.resize((w, int(i.height * w / i.width)), Image.LANCZOS) if i.width != w else i for i in ims]
    canvas = Image.new("RGB", (w, sum(i.height for i in ims) + 40 * (len(ims) - 1)), (40, 45, 52))   # RGV grid line colour
    y = 0
    for i in ims:
        canvas.paste(i, (0, y))
        y += i.height + 40
    canvas.save(out / "fig_render_influence_all.png")
    small = canvas.resize((canvas.width // 2, canvas.height // 2), Image.LANCZOS)
    small.save(out / "fig_render_influence_all_half.png")


def main():
    out = Path(sys.argv[1])
    dmax = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    S = {m: {} for m in MESHES}
    for m in MESHES:
        tgt = np.asarray(np.load(G / f"d145t_b1/{m}300k_P97_end1.npz")["tgt"], np.float64)
        a = json.load(open(out / f"endmap_{m}.json"))["pitch"]
        S[m]["size_pitch"] = float((tgt.max(0) - tgt.min(0)).max() / a)
    W = windows()
    fig1(out, W, S)
    fig2(out, W, S)
    fig3(out, S, dmax)
    fig4(out, S)
    combine(out)
    json.dump(S, open(out / "summary.json", "w"), indent=1, default=float)
    print(json.dumps({m: {k: v for k, v in S[m].items() if k != "fig3"} for m in MESHES}, indent=1, default=float))
    print(json.dumps({k: S[k] for k in ("fig4_d157", "fig4_d156h2")}, indent=1, default=float))


if __name__ == "__main__":
    main()
