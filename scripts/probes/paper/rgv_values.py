"""rgv_values.py RUN_JSON FRAMES12_NPZ DUMP_DIR CF_DIR OUT_NPZ [NUMBERS_JSON] -- the render-gradient videos' values (RGV;
CPU only): one npz with a run's kept frames and, per COMMITTED window (the history's records with frame_end, in order),
the per-particle records the renderer (scripts/render_gradient_viz.py) colours the body with:

  g_rend  lambda x the render gradient on dF_c after PCGrad (the window's first gradient), the sum over the driven steps
          of the per-step Frobenius norm (the dump's gl_rend_pc_psum x lam_r)                           (W, N)
  g_phys  the physics gradient on dF_c (with the cleanup's), the same reduction (gl_phys_psum)          (W, N)
  gu_rend lambda |dL_render/du| after PCGrad (u: the outer layer's normal offset)                       (W, N)
  layer   the window's outer layer (the u channel's particles)                                          (W, N)
  ddfc    Delta dF_c^render = dF_c(render on) - dF_c(lambda = 0), mean over the driven steps, the same window
          optimised twice from the same start (the counterfactual)                                      (W, N, 9)
  du      g (u(render on) - u(lambda = 0)) in pitches: the change of the layer's normal displacement      (W, N)
  ugate   u's gate g (0 or 1: u acts only where the body has arrived; 5-12 % of the layer early)       (W, N)
  dfc0n   |dF_c(lambda = 0)|, the Frobenius norm of the lambda = 0 control's mean over the driven steps     (W, N)
  cosrp   the cosine per particle between the render and the physics gradients on dF_c (before PCGrad; their
          time means over the steps as 9-vectors, the dump's gl_*_tmean)                               (W, N)
  dx      |x_end(render on) - x_end(lambda = 0)| in pitches                                              (W, N)
win_of_frame maps each kept frame to the window whose rollout holds it (frame 0, the source, to the first). The
colour scales are fixed for the whole run (scales, JSON): v1 three decades below the 99.5th percentile of both
gradients pooled on the outer layer over all windows; v2 three decades below the 99.5th percentile of |Delta dF_c| on
the layer; v3 the 99th percentile of |du| where u acts (layer, gate open); the streaks' full length at the 99th
percentile of the largest |eigenvalue| of sym(Delta dF_c) on the layer. NUMBERS_JSON: per window the scalars the frames
print and the readings of how much the render changes the control (relative to the lambda = 0 control, by region)."""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

run_json, f12, dump_dir, cf_dir, out = sys.argv[1:6]
num_out = sys.argv[6] if len(sys.argv) > 6 else os.path.splitext(out)[0] + "_numbers.json"
z = np.load(f12)
frames, raws, tgt = z["frames"], np.asarray(z["raws"]).astype(int), np.asarray(z["tgt"], np.float64)
pitch = 0.708 * float(np.median(cKDTree(tgt).query(tgt[::50], 9)[0][:, 8]))      # b7_proof.py's pitch
hist = next(iter(json.load(open(run_json))["arms"].values()))["history"]
committed = [(int(r["animation"]), int(r["frame_end"])) for r in hist if r.get("frame_end")]
raw_cos = [r.get("g_raw_cos") for r in hist if r.get("frame_end")]       # < 0: PCGrad acted in that window
N = frames.shape[1]
W = len(committed)
print(f"pitch {pitch:.5f}; {len(raws)} kept frames (last raw {raws[-1]}); {W} committed windows; N {N}", flush=True)

g_rend = np.zeros((W, N), np.float32)
g_phys = np.zeros((W, N), np.float32)
gu_rend = np.zeros((W, N), np.float32)
layer = np.zeros((W, N), bool)
ddfc = np.zeros((W, N, 9), np.float32)
du = np.zeros((W, N), np.float32)
ugate = np.zeros((W, N), np.float32)
dfc0n = np.zeros((W, N), np.float32)
cosrp = np.zeros((W, N), np.float32)
dx = np.zeros((W, N), np.float32)
scal = {k: np.zeros(W) for k in ("w_anim", "w_lam", "w_gshare", "w_t0", "w_t1", "w_ratio_layer", "w_ddfc_rel_layer",
                                 "w_du_rms_layer", "w_u0_rms_layer", "w_gphys_p99_layer", "w_dfc0_p99_layer",
                                 "w_layer_frac", "w_uactive_frac", "w_du_rms_active", "w_u0_rms_active",
                                 "w_graw_cos", "w_tot_rend", "w_tot_phys")}
rows = []
fe_prev = 0
for w, (a, fe) in enumerate(committed):
    d = np.load(os.path.join(dump_dir, f"win_{a:04d}.npz"))
    c = np.load(os.path.join(cf_dir, f"cf_{a:04d}.npz"), allow_pickle=True)
    lam = float(d["lam_r"])
    gr = d["gl_rend_pc_psum"] if "gl_rend_pc_psum" in d.files else d["gl_rend_psum"]
    g_rend[w], g_phys[w] = lam * gr, d["gl_phys_psum"]
    gu_rend[w] = lam * np.abs(d["gu_rend_pc"] if "gu_rend_pc" in d.files else d["gu_rend"])
    lay = d["layer_mask"] > 0.5
    tr, tp = d["gl_rend_tmean"].astype(np.float64), d["gl_phys_tmean"].astype(np.float64)   # (N, 9), before PCGrad
    cosrp[w] = (tr * tp).sum(1) / np.maximum(np.linalg.norm(tr, axis=1) * np.linalg.norm(tp, axis=1), 1e-300)
    scal["w_graw_cos"][w] = raw_cos[w] if raw_cos[w] is not None else np.nan
    scal["w_tot_rend"][w], scal["w_tot_phys"][w] = float(g_rend[w].sum()), float(g_phys[w].sum())
    layer[w] = lay
    row = {"window": w + 1, "animation": a, "frame_start": fe_prev, "frame_end": fe, "lam": lam,
           "g_share": float(d["g_share"]), "cf_complete": "ddfc_mean" in c.files}
    bulk = ~lay
    row["rend_over_phys_median_layer"] = float(np.median(g_rend[w][lay]) / max(np.median(g_phys[w][lay]), 1e-300))
    row["rend_over_phys_median_bulk"] = float(np.median(g_rend[w][bulk]) / max(np.median(g_phys[w][bulk]), 1e-300))
    row["rend_dFc_share_layer"] = float((g_rend[w][lay] ** 2).sum() / max((g_rend[w] ** 2).sum(), 1e-300))
    row["grend_p99_layer"] = float(np.percentile(g_rend[w][lay], 99))
    row["gphys_p99_layer"] = float(np.percentile(g_phys[w][lay], 99))
    if row["cf_complete"]:
        dd, d0 = c["ddfc_mean"], c["dfc_phys_mean"]
        ddfc[w] = dd
        dfc0n[w] = np.linalg.norm(d0, axis=1)
        g, gp = c["u_gate"], c["u_gate_phys"]
        du[w] = g * (c["u_render"] - c["u_phys"]) / pitch
        ugate[w] = g
        act = lay & (g > 0.5)                      # u acts only where its gate is open (binary: the arrived scope)
        row["layer_frac"], row["uactive_frac_of_layer"] = float(lay.mean()), float(act.sum() / max(lay.sum(), 1))
        row["du_rms_active"] = float(np.sqrt(np.mean(du[w][act] ** 2))) if act.any() else 0.0
        row["u0_rms_active"] = float(np.sqrt(np.mean((gp * c["u_phys"] / pitch)[act] ** 2))) if act.any() else 0.0
        dx[w] = np.linalg.norm(c["x_render"] - c["x_phys"], axis=1) / pitch
        u0 = gp * c["u_phys"] / pitch
        for nm, m in (("layer", lay), ("bulk", bulk), ("all", np.ones(N, bool))):
            row[f"ddfc_rel_{nm}"] = float(np.linalg.norm(dd[m]) / max(np.linalg.norm(d0[m]), 1e-300))
            row[f"ddfc_median_{nm}"] = float(np.median(np.linalg.norm(dd[m], axis=1)))
            row[f"dfc0_median_{nm}"] = float(np.median(np.linalg.norm(d0[m], axis=1)))
            row[f"dfc0_p99_{nm}"] = float(np.percentile(np.linalg.norm(d0[m], axis=1), 99))
            row[f"dx_median_{nm}"] = float(np.median(dx[w][m]))
        row["ddfc_psum_median_layer"] = float(np.median(c["ddfc_psum"][lay]))
        row["du_rms_layer"] = float(np.sqrt(np.mean(du[w][lay] ** 2)))
        row["u0_rms_layer"] = float(np.sqrt(np.mean(u0[lay] ** 2)))
        row["du_rel_layer"] = float(np.linalg.norm(du[w][lay]) / max(np.linalg.norm(u0[lay]), 1e-300))
        row["gate_same"] = float(np.abs(g - gp).max())
        rec_r, rec_0 = eval(str(c["render"])), eval(str(c["cf"]))          # noqa: S307  (our own repr of a dict)
        for k in ("d_sil", "d_pbr", "loss_end", "iters", "accepted"):
            row[f"{k}_render"], row[f"{k}_cf"] = rec_r.get(k), rec_0.get(k)
    scal["w_anim"][w], scal["w_lam"][w], scal["w_gshare"][w] = a, lam, row["g_share"]
    scal["w_t0"][w], scal["w_t1"][w] = fe_prev / 240.0, fe / 240.0
    scal["w_ratio_layer"][w] = row["rend_over_phys_median_layer"]
    scal["w_gphys_p99_layer"][w] = float(np.percentile(g_phys[w][lay], 99))
    scal["w_dfc0_p99_layer"][w] = row.get("dfc0_p99_layer", np.nan)
    scal["w_ddfc_rel_layer"][w] = row.get("ddfc_rel_layer", np.nan)
    scal["w_du_rms_layer"][w] = row.get("du_rms_layer", np.nan)
    scal["w_u0_rms_layer"][w] = row.get("u0_rms_layer", np.nan)
    scal["w_layer_frac"][w] = float(lay.mean())
    scal["w_uactive_frac"][w] = row.get("uactive_frac_of_layer", np.nan)
    scal["w_du_rms_active"][w] = row.get("du_rms_active", np.nan)
    scal["w_u0_rms_active"][w] = row.get("u0_rms_active", np.nan)
    rows.append(row)
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    fe_prev = fe

ends = np.array([fe for _, fe in committed])
win_of = np.array([0 if k == 0 else int(min(np.searchsorted(ends, k), W - 1)) for k in raws])
L = layer
pool = np.concatenate((g_rend[L], g_phys[L]))
v1_hi = float(np.percentile(pool[pool > 0], 99.5))
dn = np.linalg.norm(ddfc, axis=2)
v2_hi = float(np.percentile(dn[L][dn[L] > 0], 99.5))
v3_max = float(np.percentile(np.abs(du[L & (ugate > 0.5)]), 99))   # where u acts
rng = np.random.default_rng(0)
samp = ddfc[L][rng.choice(int(L.sum()), size=min(200000, int(L.sum())), replace=False)].reshape(-1, 3, 3).astype(np.float64)
ev = np.linalg.eigvalsh(0.5 * (samp + samp.transpose(0, 2, 1)))
glyph_ref = float(np.percentile(np.abs(ev).max(1), 99))
# the per-window relative scales (each window divided by its own physics gradient's, or its own lambda = 0 control's,
# 99th percentile on the outer layer): the same colour bar for every window, read relative to that window
rel1 = np.concatenate([np.concatenate((g_rend[w][layer[w]], g_phys[w][layer[w]])) / scal["w_gphys_p99_layer"][w]
                       for w in range(W)])
rel2 = np.concatenate([dn[w][layer[w]] / scal["w_dfc0_p99_layer"][w] for w in range(W)
                       if np.isfinite(scal["w_dfc0_p99_layer"][w]) and scal["w_dfc0_p99_layer"][w] > 0])
v1w_hi, v2w_hi = float(np.percentile(rel1[rel1 > 0], 99.5)), float(np.percentile(rel2[rel2 > 0], 99.5))
b_hi = float(np.percentile(g_rend[L][g_rend[L] > 0], 99.5))   # panel B: one log scale, 4 decades below
scales = dict(v1_lo=v1_hi * 1e-3, v1_hi=v1_hi, v2_lo=v2_hi * 1e-3, v2_hi=v2_hi, v3_max=v3_max, glyph_ref=glyph_ref,
              v1w_lo=v1w_hi * 1e-3, v1w_hi=v1w_hi, v2w_lo=v2w_hi * 1e-3, v2w_hi=v2w_hi, b_lo=b_hi * 1e-4, b_hi=b_hi,
              pitch=pitch)
print("scales", scales, flush=True)
np.savez(out, frames=frames, raws=raws, tgt=z["tgt"], deliver_n=z["deliver_n"], pitch=np.float64(pitch),
         win_of_frame=win_of, g_rend=g_rend, g_phys=g_phys, gu_rend=gu_rend, layer=layer, ddfc=ddfc, du=du, dx=dx,
         ugate=ugate, dfc0n=dfc0n, cosrp=cosrp, scales=np.str_(json.dumps(scales)), **scal)
json.dump({"pitch": pitch, "scales": scales, "windows": rows,
           "kept_frames_per_window": np.bincount(win_of, minlength=W).tolist()}, open(num_out, "w"), indent=1)
print("wrote", out, num_out, flush=True)
