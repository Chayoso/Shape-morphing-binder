"""b7_proof.py OUT DUMP_DIR CF_DIR RUN_JSON RUN_LOG L_FRAMES P_FRAMES L_DIRNAME IND_NPZ -- D145 B7, the render gradient's
proof on the deliverable code (bunny 300k, seed 97; CPU only): WHERE it acts, HOW it reaches the controls, WHAT it
changes, and what it does not.

Regions, on each window's own start state x0 (no target used):
  layer  = the pipeline's outer layer (the dump's layer_mask, the u channel's particles);
  depth  = distance to the nearest layer particle in pitches (pitch = 0.708 x the target sample's median 8th-neighbour
           distance, D95); bulk = depth > 2, sub = layer excluded and depth <= 2;
  rim    = layer particles on the silhouette outline of any of six axis views (an occupancy grid of one pitch, the
           occupied cells with an empty 4-neighbour);
  thin   = particles whose local thickness is below two MPM cells: the max-ball thickness (Hildebrand & Rueegsegger
           1997) of the body's own occupancy on a voxel grid of one pitch, as physmorph/thin.py measures the target.
Gradients (the dump: the window's first gradient, before PCGrad and lambda unless named _pc): on the stress control dFc
per particle the sum over the driven steps of the Frobenius norm (gl_*_psum) and the time-mean 9-vector (gl_*_tmean); on
u per layer particle (gu_*). Shares = the squared-norm share of a channel inside a region; cos per region = the cosine of
the two channels' time-mean dFc vectors (or u values) summed over the region's particles.
Writes fig1_where_wNN.png, fig2_how.png, fig3_what.png, fig4_run.png, numbers.json, numbers.txt."""
import ast
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

out, dump_dir, cf_dir, run_json, run_log, l_frames, p_frames, l_dir, ind_npz = sys.argv[1:10]
os.makedirs(out, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "savefig.dpi": 150})

tgt = np.load(l_frames, mmap_mode="r")["tgt"].astype(np.float64)
pitch = 0.708 * float(np.median(cKDTree(tgt).query(tgt[::50], 9)[0][:, 8]))
m = re.search(r"dx = diag [\d.]+ / [\d.]+ = ([\d.]+) wu", open(run_log).read())
cell = float(m.group(1)) if m else 6.0 * pitch
print(f"pitch {pitch:.5f} wu, MPM cell {cell:.4f} wu ({cell / pitch:.2f} pitches)", flush=True)
PROJ = [(0, 1), (2, 1), (0, 2)]                            # the three axis projections (the outline is shared by opposite views)


def regions(x0, layer):
    tree = cKDTree(x0[layer])
    depth = tree.query(x0, 1)[0] / pitch
    rim = np.zeros(len(x0), bool)
    for h, v in PROJ:
        ij = np.floor((x0[:, [h, v]] - x0[:, [h, v]].min(0)) / pitch).astype(int)
        occ = np.zeros(ij.max(0) + 3, bool)
        occ[ij[:, 0] + 1, ij[:, 1] + 1] = True
        edge = occ & ~ndimage.binary_erosion(occ)
        rim |= edge[ij[:, 0] + 1, ij[:, 1] + 1]
    rim &= layer
    lo = x0.min(0) - 3 * pitch
    dims = np.ceil((x0.max(0) + 3 * pitch - lo) / pitch).astype(int) + 1
    ijk = np.floor((x0 - lo) / pitch).astype(int)
    occ = np.zeros(dims, bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    occ = ndimage.binary_fill_holes(ndimage.binary_closing(occ, iterations=1))
    D = ndimage.distance_transform_edt(occ)
    thick = np.zeros(dims)
    for r in range(1, int(D.max()) + 1):
        core = D >= r
        if not core.any():
            break
        thick[(ndimage.distance_transform_edt(~core) <= r) & occ] = 2.0 * r * pitch
    h = thick[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    thin = (h > 0) & (h < 2.0 * cell)
    return dict(layer=layer, sub=(~layer) & (depth <= 2), bulk=depth > 2, rim=rim, thin=thin), depth


def sq_share(g2, mask):
    return float(g2[mask].sum() / max(g2.sum(), 1e-300))


def cos_region(a, b, mask):
    a, b = a[mask].reshape(-1), b[mask].reshape(-1)
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def draw(ax, x, val, view, norm, title, s=1.2):
    h, v, d = view
    order = np.argsort(x[:, d])                          # far first (the camera on the +d side), the nearest on top
    sc = ax.scatter(x[order, h], x[order, v], c=val[order], s=s, cmap="magma", norm=norm, linewidths=0, rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=8, color=INK)
    plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.01, shrink=0.7)


nums = {"pitch": pitch, "cell": cell, "windows": {}, "cf": {}, "run": {}}
hist = next(iter(json.load(open(run_json))["arms"].values()))["history"]
dumps = {int(os.path.basename(f)[4:8]) + 1: f for f in sorted(glob.glob(os.path.join(dump_dir, "win_*.npz")))}
FRONT, SIDE = (0, 1, 2), (2, 1, 0)
DEPTH_BINS = [0, 1, 2, 4, 6, 8, 12, 100]
for w, f in dumps.items():
    z = np.load(f)
    x0 = z["x0"].astype(np.float64)
    layer = z["layer_mask"] > 0.5
    R, depth = regions(x0, layer)
    lam = float(z["lam_r"])
    gr, gp = z["gl_rend_psum"].astype(np.float64), z["gl_phys_psum"].astype(np.float64)
    grpc = z["gl_rend_pc_psum"].astype(np.float64)
    tr, tp = z["gl_rend_tmean"].astype(np.float64), z["gl_phys_tmean"].astype(np.float64)
    ur, up = z["gu_rend"].astype(np.float64), z["gu_phys"].astype(np.float64)
    row = {"lam": lam, "g_share": float(z["g_share"]), "n": int(len(x0))}
    for name, msk in R.items():
        row[f"frac_{name}"] = float(msk.mean())
        row[f"rend_dFc_share_{name}"] = sq_share(gr ** 2, msk)
        row[f"phys_dFc_share_{name}"] = sq_share(gp ** 2, msk)
        row[f"rend_over_phys_median_{name}"] = float(np.median(lam * gr[msk]) / max(np.median(gp[msk]), 1e-300))
        row[f"cos_dFc_{name}"] = cos_region(tr, tp, msk)
        if name in ("layer", "rim", "thin"):
            mk = msk & layer
            row[f"rend_u_share_{name}"] = sq_share(ur ** 2, mk)
            row[f"phys_u_share_{name}"] = sq_share(up ** 2, mk)
            row[f"cos_u_{name}"] = cos_region(ur, up, mk)
    row["cos_dFc_all"] = cos_region(tr, tp, np.ones(len(x0), bool))
    row["cos_u_all"] = cos_region(ur, up, layer)
    gx_r = (z["gx_sil"] + z["gx_pbr"]).astype(np.float64)
    n2 = (gx_r ** 2).sum(1)
    row["rend_position_share_layer"] = sq_share(n2, layer)
    row["rend_dFc_share_layer_vs_position"] = row["rend_dFc_share_layer"]
    med_s = float(np.median(gr[layer]))
    row["reach_all"] = float((gr > 0.01 * med_s).mean())
    row["reach_bulk"] = float((gr[R["bulk"]] > 0.01 * med_s).mean())
    row["depth_profile_rend"] = [float(np.median(gr[(depth >= DEPTH_BINS[i]) & (depth < DEPTH_BINS[i + 1])]) / med_s)
                                 if ((depth >= DEPTH_BINS[i]) & (depth < DEPTH_BINS[i + 1])).any() else None
                                 for i in range(len(DEPTH_BINS) - 1)]
    med_p = float(np.median(gp[layer]))
    row["depth_profile_phys"] = [float(np.median(gp[(depth >= DEPTH_BINS[i]) & (depth < DEPTH_BINS[i + 1])]) / med_p)
                                 if ((depth >= DEPTH_BINS[i]) & (depth < DEPTH_BINS[i + 1])).any() else None
                                 for i in range(len(DEPTH_BINS) - 1)]
    # what each channel alone moves in the window (the linear response of a unit step along it), by depth
    mv_r = np.linalg.norm((z["xT_rend"] - z["xT_base"]).astype(np.float64), axis=1)
    mv_p = np.linalg.norm((z["xT_phys"] - z["xT_base"]).astype(np.float64), axis=1)
    row["move_profile_rend"] = [float(np.median(mv_r[(depth >= DEPTH_BINS[i]) & (depth < DEPTH_BINS[i + 1])]) / max(np.median(mv_r[layer]), 1e-300))
                                if ((depth >= DEPTH_BINS[i]) & (depth < DEPTH_BINS[i + 1])).any() else None
                                for i in range(len(DEPTH_BINS) - 1)]
    row["move_profile_phys"] = [float(np.median(mv_p[(depth >= DEPTH_BINS[i]) & (depth < DEPTH_BINS[i + 1])]) / max(np.median(mv_p[layer]), 1e-300))
                                if ((depth >= DEPTH_BINS[i]) & (depth < DEPTH_BINS[i + 1])).any() else None
                                for i in range(len(DEPTH_BINS) - 1)]
    row["pcgrad_changed_rel"] = float(np.linalg.norm(gr - grpc) / max(np.linalg.norm(gr), 1e-300))
    nums["windows"][w] = row
    print(f"window {w}: lam {lam:.3g}; render dFc share layer {row['rend_dFc_share_layer']:.2f} rim {row['rend_dFc_share_rim']:.2f} "
          f"thin {row['rend_dFc_share_thin']:.2f} bulk {row['rend_dFc_share_bulk']:.2f} (fractions {row['frac_layer']:.2f} / "
          f"{row['frac_rim']:.3f} / {row['frac_thin']:.3f} / {row['frac_bulk']:.2f}); cos dFc all {row['cos_dFc_all']:.2f} "
          f"layer {row['cos_dFc_layer']:.2f} rim {row['cos_dFc_rim']:.2f} thin {row['cos_dFc_thin']:.2f} bulk {row['cos_dFc_bulk']:.2f}; "
          f"reach all {row['reach_all']:.2f} bulk {row['reach_bulk']:.2f}; PCGrad change {row['pcgrad_changed_rel']:.3f}", flush=True)
    if w in (1, 2, 3, 5, 10, 20, 40):
        fig, axs = plt.subplots(2, 3, figsize=(12, 8.4), constrained_layout=True)
        for i, (view, vname) in enumerate(((FRONT, "front (along z)"), (SIDE, "side (along x)"))):
            for j, (val, lab) in enumerate(((lam * gr, "lambda |dL_render/d dFc| (sum over steps)"),
                                             (gp, "|dL_physics/d dFc| (sum over steps)"),
                                             (np.where(layer, lam * np.abs(ur), np.nan), "lambda |dL_render/du| (layer)"))):
                v = np.nan_to_num(val, nan=0.0)
                hi = float(np.percentile(v[v > 0], 99)) if (v > 0).any() else 1.0
                norm = LogNorm(hi * 1e-3, hi)
                draw(axs[i, j], x0 if j < 2 else x0[layer], np.clip(v if j < 2 else v[layer], hi * 1e-3, hi), view,
                     norm, f"{lab}\n{vname}")
        fig.suptitle(f"bunny 300k, window {w}: where the render gradient acts (log colour, 3 decades below p99); "
                     f"render share of the step {row['g_share']:.2f}, lambda {lam:.3g}", fontsize=10)
        fig.savefig(os.path.join(out, f"fig1_where_w{w:02d}.png"))
        plt.close(fig)

# ---- the counterfactuals: the same window from the same start, render on against lambda = 0 ----
ind = np.load(ind_npz)["tgt"].astype(np.float64)
ind_tree = cKDTree(ind)
for f in sorted(glob.glob(os.path.join(cf_dir, "cf_*.npz"))):
    w = int(os.path.basename(f)[3:7]) + 1
    z = np.load(f, allow_pickle=True)
    if not len(z["x_phys"]) or not len(z["x_render"]):
        continue
    x0, xr, xp = (z[k].astype(np.float64) for k in ("x0", "x_render", "x_phys"))
    dmp = dumps.get(w)
    layer = (np.load(dmp)["layer_mask"] > 0.5) if dmp else None
    if layer is None:
        continue
    R, depth = regions(x0, layer)
    dx = np.linalg.norm(xr - xp, axis=1) / pitch
    dr, dp = ind_tree.query(xr, 1)[0] / pitch, ind_tree.query(xp, 1)[0] / pitch    # distance to the target's sample
    rec_r, rec_p = ast.literal_eval(str(z["render"])), ast.literal_eval(str(z["cf"]))
    row = {"d_sil_render": rec_r.get("d_sil"), "d_sil_phys": rec_p.get("d_sil"), "d_pbr_render": rec_r.get("d_pbr"),
           "d_pbr_phys": rec_p.get("d_pbr"), "d_vol_render": rec_r.get("d_vol"), "d_vol_phys": rec_p.get("d_vol"),
           "kin_render": rec_r.get("kin"), "kin_phys": rec_p.get("kin"), "stab_render": rec_r.get("stab"),
           "stab_phys": rec_p.get("stab"), "L_start": rec_r.get("L_start")}
    for name, msk in R.items():
        row[f"dx_median_{name}"] = float(np.median(dx[msk]))
        row[f"moved01_{name}"] = float((dx[msk] > 0.1).mean())
        row[f"closer_{name}"] = float(np.median((dp - dr)[msk]))          # > 0: the render arm nearer the target sample
    com_r, com_p = xr.mean(0), xp.mean(0)
    row["com_diff_pitches"] = float(np.linalg.norm(com_r - com_p) / pitch)
    nums["cf"][w] = row
    print(f"cf window {w}: |dx| median layer {row['dx_median_layer']:.3f} rim {row['dx_median_rim']:.3f} thin {row['dx_median_thin']:.3f} "
          f"bulk {row['dx_median_bulk']:.3f} pitches; nearer the target (median, pitches) layer {row['closer_layer']:+.4f} rim "
          f"{row['closer_rim']:+.4f} thin {row['closer_thin']:+.4f}; d_sil {row['d_sil_render']:.4g} / {row['d_sil_phys']:.4g}, "
          f"d_vol {row['d_vol_render']:.4g} / {row['d_vol_phys']:.4g}; centre of mass apart {row['com_diff_pitches']:.2e} pitches", flush=True)
    if w in (1, 5, 10, 20):
        fig, axs = plt.subplots(1, 2, figsize=(9, 4.6), constrained_layout=True)
        hi = max(float(np.percentile(dx, 99)), 0.05)
        for ax, (view, vname) in zip(axs, ((FRONT, "front"), (SIDE, "side"))):
            draw(ax, x0, np.clip(dx, hi * 1e-3, hi), view, LogNorm(hi * 1e-3, hi), f"|x_end(render) - x_end(lambda = 0)|, pitches, {vname}")
        fig.suptitle(f"window {w}: the same window from the same start, render on against off", fontsize=10)
        fig.savefig(os.path.join(out, f"fig3_what_cf_w{w:02d}.png"))
        plt.close(fig)

# ---- the run: render arm L against its twin P (same seed, same particles), matched simulated frames ----
zl, zp = np.load(l_frames), np.load(p_frames)
FL, FP = zl["frames"], zp["frames"]                  # loaded once (an npz member is re-read at every access)
rl, rp = list(np.asarray(zl["raws"]).astype(int)), list(np.asarray(zp["raws"]).astype(int))
common = sorted(set(rl) & set(rp))
traj = []
for raw in common[:: max(1, len(common) // 24)] + [common[-1]]:
    xl = np.asarray(FL[rl.index(raw)], np.float64)
    xpp = np.asarray(FP[rp.index(raw)], np.float64)
    tree = cKDTree(xl)
    sp = float(np.median(tree.query(xl[::15], 9)[0][:, 8]))
    nb = tree.query(xl, 33)[1][:, 1:]
    lay = np.linalg.norm(xl - xl[nb].mean(1), axis=1) >= 0.5 * sp
    dep = cKDTree(xl[lay]).query(xl, 1)[0] / pitch
    d = np.linalg.norm(xl - xpp, axis=1) / pitch
    traj.append({"raw": raw, "t": raw / 240.0, "diff_layer": float(np.median(d[lay])), "diff_bulk": float(np.median(d[dep > 2])),
                 "diff_bulk_p90": float(np.percentile(d[dep > 2], 90)),
                 "com_diff": float(np.linalg.norm(xl.mean(0) - xpp.mean(0)) / pitch)})
nums["run"]["matched"] = traj
xle = np.asarray(FL[-1], np.float64)
xpe = np.asarray(FP[-1], np.float64)
for nm, xe in (("L", xle), ("P", xpe)):
    tree = cKDTree(xe)
    sp = float(np.median(tree.query(xe[::15], 9)[0][:, 8]))
    nb = tree.query(xe, 33)[1][:, 1:]
    lay = np.linalg.norm(xe - xe[nb].mean(1), axis=1) >= 0.5 * sp
    R, _ = regions(xe, lay)
    de = ind_tree.query(xe, 1)[0] / pitch
    nums["run"][nm] = {f"dist_{k}_median": float(np.median(de[v])) for k, v in R.items()}
    nums["run"][nm].update({f"dist_{k}_p90": float(np.percentile(de[v], 90)) for k, v in R.items()})
print("run, matched frames (median |x_L - x_P| in pitches, layer / bulk):",
      " ".join(f"{r['t']:.2f}s {r['diff_layer']:.2f}/{r['diff_bulk']:.2f}" for r in traj), flush=True)
print("end, distance to the independent sample (median / p90, pitches):",
      {k: (round(nums['run']['L'][f'dist_{k}_median'], 3), round(nums['run']['P'][f'dist_{k}_median'], 3),
           round(nums['run']['L'][f'dist_{k}_p90'], 3), round(nums['run']['P'][f'dist_{k}_p90'], 3))
       for k in ("layer", "rim", "thin", "bulk")}, flush=True)

# ---- fig2: how it reaches the controls ----
W = sorted(nums["windows"])
fig, axs = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
mid = [0.5, 1.5, 3, 5, 7, 10, 14]
for w, c in zip((1, 5, 20), (BLUE, AQUA, ORANGE)):
    if w in nums["windows"]:
        r = nums["windows"][w]
        axs[0].plot(mid, r["depth_profile_rend"], "-o", color=c, ms=3, label=f"render, window {w}")
        axs[0].plot(mid, r["depth_profile_phys"], ":", color=c, label=f"physics, window {w}")
        axs[1].plot(mid, r["move_profile_rend"], "-o", color=c, ms=3, label=f"render, window {w}")
        axs[1].plot(mid, r["move_profile_phys"], ":", color=c, label=f"physics, window {w}")
axs[0].set_yscale("log"); axs[0].set_xlabel("depth below the outer layer (pitches)")
axs[0].set_ylabel("median |gradient on dFc| / the layer's"); axs[0].set_title("gradient on dFc by depth\n(median, relative to the outer layer)", fontsize=9)
axs[0].legend(frameon=False, fontsize=7)
axs[1].set_yscale("log"); axs[1].set_xlabel("depth below the outer layer (pitches)")
axs[1].set_ylabel("median move / the layer's"); axs[1].set_title("what one step along each channel moves,\nby depth (relative to the outer layer)", fontsize=9)
axs[1].legend(frameon=False, fontsize=7)
for key, c, lab in (("layer", BLUE, "outer layer"), ("rim", AQUA, "silhouette rim"), ("thin", ORANGE, "thin parts"),
                    ("bulk", INK2, "bulk (> 2 pitches deep)")):
    axs[2].plot(W, [nums["windows"][w][f"rend_dFc_share_{key}"] for w in W], "-o", ms=3, color=c, label=lab)
    axs[2].plot(W, [nums["windows"][w][f"frac_{key}"] for w in W], ":", color=c)
axs[2].set_xscale("log"); axs[2].set_ylim(0, 1); axs[2].set_xlabel("window")
axs[2].set_title("render gradient: squared-norm share by region\n(dotted: the region's share of particles)", fontsize=9); axs[2].legend(frameon=False, fontsize=7)
hw = [h for h in hist if h.get("g_share") is not None and h.get("frame_end")]
ws = np.arange(1, len(hw) + 1)
axs[3].plot(ws, [h["g_share"] for h in hw], color=BLUE, label="render share of the step")
axs[3].plot(ws, [h["g_raw_cos"] for h in hw], color=INK2, lw=1, label="cos(g_phys, g_render), raw")
axs[3].plot(ws, [h["g_cos"] for h in hw], color=ORANGE, lw=1, ls="--", label="cos after PCGrad")
axs[3].axhline(0, color=GRID, lw=1); axs[3].set_xlabel("window"); axs[3].legend(frameon=False, fontsize=7)
axs[3].set_title("render share of the step and\ncos(g_phys, g_render) per window", fontsize=9)
fig.savefig(os.path.join(out, "fig2_how.png"))
plt.close(fig)
nums["pcgrad_windows_active"] = [i + 1 for i, h in enumerate(hw) if (h.get("g_raw_cos") or 0) < 0]
nums["lambda_per_window"] = [h.get("lambda") for h in hw]

# ---- fig4: the run ----
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
axs[0].plot([r["t"] for r in traj], [r["diff_layer"] for r in traj], color=BLUE, label="outer layer")
axs[0].plot([r["t"] for r in traj], [r["diff_bulk"] for r in traj], color=INK2, label="bulk (> 2 pitches deep)")
axs[0].plot([r["t"] for r in traj], [r["com_diff"] for r in traj], color=ORANGE, ls="--", label="centre of mass")
axs[0].set_yscale("log"); axs[0].set_xlabel("simulated time (s)"); axs[0].set_ylabel("median |x_render arm - x_twin| (pitches)")
axs[0].set_title("render arm against its twin, same particles, same time"); axs[0].legend(frameon=False, fontsize=7)
keys = ["layer", "rim", "thin", "bulk"]
xx = np.arange(len(keys))
axs[1].bar(xx - 0.18, [nums["run"]["L"][f"dist_{k}_p90"] for k in keys], 0.36, color=BLUE, label="render arm (p90)")
axs[1].bar(xx + 0.18, [nums["run"]["P"][f"dist_{k}_p90"] for k in keys], 0.36, color=ORANGE, label="twin (p90)")
axs[1].set_xticks(xx, keys); axs[1].set_ylabel("distance to the independent sample (pitches)")
axs[1].set_title("the end: where each is off the target"); axs[1].legend(frameon=False, fontsize=7)
fig.savefig(os.path.join(out, "fig4_run.png"))
plt.close(fig)
json.dump(nums, open(os.path.join(out, "numbers.json"), "w"), indent=1, default=float)
print("written", out, flush=True)
