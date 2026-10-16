"""rgv_tag_check.py OUT_JSON OLD_VIZ NEW_VIZ NEW_DUMP_DIR -- the RGV values on the tag freeze-2026-10-15 against the
pre-tag version (CPU only): per committed window, on the outer layer, the quantities the panels show, and then the
medians over the window bands the README reads (1, 2-5, 6-10, 11-20, 21-40, 41-end).

  g_share   the balancer's share of the window's first gradient (|lambda g_r| / (|lambda g_r| + |g_p|) over ALL
            controls; 1/3 by construction where it is not capped)
  agg       sum |lambda g_r| / sum (|lambda g_r| + |g_p|) over the layer's particles (per-particle norms)
  A         the per-particle share s on the layer: median, p90, the fraction over 50 %
  C         the per-window cosine (not the panel's 5-window mean) where the render acts (layer, s >= 5 %): the fraction
            above +tau (agree) and below -tau (conflict), tau 0, 0.2, 0.3; PCGrad active (the history's g_raw_cos < 0)
  D         |Delta dF_c| / |dF_c(lambda = 0)| on the layer where |dF_c(lambda = 0)| >= 1 % of its p99: median, the
            fraction over 20 %; the norm ratio over the layer (ddfc_rel_layer)
  U         u's open share of the layer, rms g Delta u where open (pitches)
The new values count every non-render term as the physics (rgv_patch2's gl_nonrend_*); the same quantities with the
physics core alone (the dump's gl_phys_psum / gl_phys_tmean, the pre-tag definition) are recorded beside them (core_*),
so the change from the code and the change from the definition read apart."""
import json
import os
import sys

import numpy as np

out, old_p, new_p, dump_dir = sys.argv[1:5]
BANDS = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 40), (41, 10 ** 6)]
TAUS = (0.0, 0.2, 0.3)


def per_window(z, w, gp=None, cos=None):
    lay = np.asarray(z["layer"][w], bool)
    r = np.asarray(z["g_rend"][w], np.float64)[lay]
    p = (np.asarray(z["g_phys"][w], np.float64) if gp is None else gp)[lay]
    s = r / np.maximum(r + p, 1e-300)
    c = (np.asarray(z["cosrp"][w], np.float64) if cos is None else cos)[lay]
    act = s >= 0.05
    row = {"agg": float(r.sum() / max((r + p).sum(), 1e-300)), "A_median": float(np.median(s)),
           "A_p90": float(np.percentile(s, 90)), "A_over50": float((s > 0.5).mean()), "acts": float(act.mean())}
    for t in TAUS:
        row[f"C_agree_{t}"] = float((c[act] > t).mean()) if act.any() else np.nan
        row[f"C_conflict_{t}"] = float((c[act] < -t).mean()) if act.any() else np.nan
    return row, lay


def rows_of(z, dump_dir=None):
    W = len(z["w_lam"])
    gsh, lam, gcos = np.asarray(z["w_gshare"]), np.asarray(z["w_lam"]), np.asarray(z["w_graw_cos"])
    anim = np.asarray(z["w_anim"]).astype(int)
    ddfc, dfc0, du, ug = z["ddfc"], z["dfc0n"], z["du"], z["ugate"]
    rows = []
    for w in range(W):
        row, lay = per_window(z, w)
        row.update(window=w + 1, t0=float(z["w_t0"][w]), t1=float(z["w_t1"][w]), g_share=float(gsh[w]),
                   lam=float(lam[w]), pcgrad=bool(np.isfinite(gcos[w]) and gcos[w] < 0),
                   cleanup_core_layer=float(z["w_cleanup_core_layer"][w]) if "w_cleanup_core_layer" in z else np.nan)
        d0 = np.asarray(dfc0[w], np.float64)
        if d0.any():                                         # the counterfactual is complete for this window
            dn = np.linalg.norm(np.asarray(ddfc[w], np.float64), axis=1)
            ok = lay & (d0 >= 0.01 * np.percentile(d0[lay], 99))
            rel = dn[ok] / d0[ok]
            row.update(D_median=float(np.median(rel)), D_over20=float((rel > 0.2).mean()),
                       D_rel_layer=float(np.linalg.norm(dn[lay]) / max(np.linalg.norm(d0[lay]), 1e-300)))
            g = np.asarray(ug[w]) > 0.5
            on = lay & g
            row.update(U_open=float(on.sum() / max(lay.sum(), 1)),
                       U_rms=float(np.sqrt(np.mean(np.asarray(du[w], np.float64)[on] ** 2))) if on.any() else 0.0)
        if dump_dir is not None:                             # the pre-tag definition on the tag run: the core alone
            d = np.load(os.path.join(dump_dir, f"win_{anim[w]:04d}.npz"))
            tr, tp = d["gl_rend_tmean"].astype(np.float64), d["gl_phys_tmean"].astype(np.float64)
            cc = (tr * tp).sum(1) / np.maximum(np.linalg.norm(tr, axis=1) * np.linalg.norm(tp, axis=1), 1e-300)
            core, _ = per_window(z, w, gp=d["gl_phys_psum"].astype(np.float64), cos=cc)
            row.update({f"core_{k}": v for k, v in core.items()})
            lay_d = d["layer_mask"] > 0.5
            row["nonrend_over_core_layer"] = float(np.median(d["gl_nonrend_psum"][lay_d]
                                                             / np.maximum(d["gl_phys_psum"][lay_d], 1e-300)))
        rows.append(row)
    return rows


def bands(rows):
    res = {}
    keys = [k for k in rows[0] if k not in ("window", "t0", "t1", "pcgrad")]
    for a, b in BANDS:
        sel = [r for r in rows if a <= r["window"] <= b]
        if not sel:
            continue
        nm = f"w{a}" if a == b else f"w{a}-{min(b, rows[-1]['window'])}"
        res[nm] = {k: float(np.nanmedian([r.get(k, np.nan) for r in sel])) for k in keys}
        res[nm]["pcgrad_windows"] = int(sum(r["pcgrad"] for r in sel))
        res[nm]["t"] = [sel[0]["t0"], sel[-1]["t1"]]
    return res


res = {}
for name, p, dd in (("old", old_p, None), ("new", new_p, dump_dir)):
    zf = np.load(p)
    z = {k: zf[k] for k in zf.files if k not in ("frames", "raws", "tgt", "deliver_n", "dx", "gu_rend")}  # each read once
    rows = rows_of(z, dd)
    pc = [r["window"] for r in rows if r["pcgrad"]]
    res[name] = {"physics_def": str(z["physics_def"]) if "physics_def" in z else "core", "windows": len(rows),
                 "pcgrad_windows": len(pc), "pcgrad_first": pc[0] if pc else None, "bands": bands(rows), "rows": rows}
    print(f"== {name}: {len(rows)} windows, physics {res[name]['physics_def']}, PCGrad in {len(pc)} "
          f"(first {pc[0] if pc else None})", flush=True)
    for bn, v in res[name]["bands"].items():
        print(f"  {bn:8s} t {v['t'][0]:.2f}-{v['t'][1]:.2f}  gshare {v['g_share']:.3f} lam {v['lam']:.3g} agg {v['agg']:.3f}"
              f"  A med {v['A_median']:.3f} p90 {v['A_p90']:.3f} >50 {v['A_over50']:.3f} acts {v['acts']:.2f}"
              f"  C+/- (0.2) {v['C_agree_0.2']:.2f}/{v['C_conflict_0.2']:.2f} (0) {v['C_agree_0.0']:.2f}/{v['C_conflict_0.0']:.2f}"
              f" (0.3) {v['C_agree_0.3']:.2f}/{v['C_conflict_0.3']:.2f}"
              f"  D med {v.get('D_median', np.nan):.3f} >20 {v.get('D_over20', np.nan):.2f} rel {v.get('D_rel_layer', np.nan):.3f}"
              f"  U open {v.get('U_open', np.nan):.2f} rms {v.get('U_rms', np.nan):.3f}  pcg {v['pcgrad_windows']}"
              f"  cleanup/core {v['cleanup_core_layer']:.2f}", flush=True)
        if "core_A_median" in v:
            print(f"  {'':8s} core only: agg {v['core_agg']:.3f} A med {v['core_A_median']:.3f} p90 {v['core_A_p90']:.3f}"
                  f" acts {v['core_acts']:.2f} C+/- (0.2) {v['core_C_agree_0.2']:.2f}/{v['core_C_conflict_0.2']:.2f}"
                  f" (0) {v['core_C_agree_0.0']:.2f}/{v['core_C_conflict_0.0']:.2f}; all/core pull {v['nonrend_over_core_layer']:.3f}",
                  flush=True)
json.dump(res, open(out, "w"), indent=1, default=float)
print("wrote", out, flush=True)
