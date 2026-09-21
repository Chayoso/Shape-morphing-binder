"""Gradient-stage analysis (docs/surface_gradient.md): where does each channel's gradient
act, and what does it do to the SURFACE?  Reads the per-window dumps written by
`--grad_dump <dir>` (pipeline/optimizer.py) and prints, per window and averaged:

  stage 1 — terminal covectors on the particles (gx_phys, gx_sil, gx_pbr):
      layer share   = fraction of |g|^2 carried by the outer particle layer (asymmetry rule)
      normal share  = fraction of the layer's |g|^2 along the local surface normal
      rough share   = fraction of the layer's normal component that is NOT explained by its
                      2-spacing neighbourhood mean (the bump-band content; 0 = a smooth
                      field, 1 = uncorrelated between neighbours)
  stage 2 — control (leaf) gradients after the MPM adjoint (gl_*_tmean, (N,9)):
      corr(r)       = correlation of the per-particle control gradient with its neighbours at
                      distance r (1, 2, 4, 8 spacings): the grid's low-pass shows as a long
                      correlation length; a channel whose particle covector was rough but whose
                      control gradient is smooth has lost its rough content in the pull-back
      layer share   = fraction of |gl|^2 on the outer layer
  stage 3 — linear response of the window to each channel alone (xT_phys/sil/pbr/rend vs
      xT_base, same control norm as the accepted change):
      |dx| layer/interior (spacings), normal share of the layer displacement, rough share of
      it, and the plane residual RMS of the outer layer (the bump amplitude, spacings) for
      x0, xT_base, xT_final and each channel's end state.

usage: grad_stage.py <dump_dir> [--spacing <wu>]
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from physmorph.render.surface_recon import layer_by_asymmetry, plane_residual  # noqa: E402


def spacing_of(x):
    kd = cKDTree(x)
    return float(np.median(kd.query(x, k=9, workers=-1)[0][:, -1]) * (9 / 8) ** 0)   # 8-NN median


def rough_share(vals, x, mask, spacing, r_sp=2.0):
    """1 - (energy of the neighbourhood mean) / (energy): the part of a scalar field on the
    layer that its 2-spacing mean does not explain."""
    P = x[mask]; v = vals
    if len(P) < 10 or float((v ** 2).sum()) <= 0:
        return float("nan")
    kd = cKDTree(P)
    pairs = kd.query_pairs(r_sp * spacing, output_type="ndarray")
    acc = v.copy(); cnt = np.ones(len(v))
    np.add.at(acc, pairs[:, 0], v[pairs[:, 1]]); np.add.at(acc, pairs[:, 1], v[pairs[:, 0]])
    np.add.at(cnt, pairs[:, 0], 1); np.add.at(cnt, pairs[:, 1], 1)
    mean = acc / cnt
    return float(1.0 - (mean ** 2).sum() / (v ** 2).sum())


def corr_at(field, x, spacing, r_sp):
    """Correlation of a per-particle vector field between particles ~r apart (a thin shell)."""
    kd = cKDTree(x)
    lo, hi = (r_sp - 0.25) * spacing, (r_sp + 0.25) * spacing
    pairs_hi = kd.query_pairs(hi, output_type="ndarray")
    if len(pairs_hi) == 0:
        return float("nan")
    d = np.linalg.norm(x[pairs_hi[:, 0]] - x[pairs_hi[:, 1]], axis=1)
    sel = pairs_hi[d >= lo]
    if len(sel) > 200000:
        sel = sel[np.random.default_rng(0).choice(len(sel), 200000, replace=False)]
    f = field - field.mean(0)
    num = (f[sel[:, 0]] * f[sel[:, 1]]).sum()
    den = np.sqrt((f[sel[:, 0]] ** 2).sum() * (f[sel[:, 1]] ** 2).sum()) + 1e-30
    return float(num / den)


def analyse(path, spacing_arg):
    z = np.load(path)
    x0 = z["x0"].astype(np.float32); xT0 = z["xT0"].astype(np.float32)
    sp = spacing_arg or spacing_of(x0)
    mask, nrm = layer_by_asymmetry(xT0, sp)
    res0, npl = plane_residual(xT0, mask, nrm, sp)
    n_layer = np.zeros_like(nrm); n_layer[mask] = npl
    out = {"n": len(x0), "layer_frac": float(mask.mean()), "spacing": sp,
           "lam_r": float(z["lam_r"]), "g_share": float(z["g_share"]), "step_norm": float(z["step_norm"]),
           "rms_xT0": float(np.sqrt((res0 ** 2).mean()))}
    # stage 1
    for c in ("gx_phys", "gx_sil", "gx_pbr"):
        g = z[c]
        if g.size == 0:
            continue
        e = (g ** 2).sum(1)
        tot = float(e.sum()) + 1e-30
        gn = (g[mask] * n_layer[mask]).sum(1)
        out[c + "_norm"] = float(np.sqrt(tot))
        out[c + "_layer"] = float(e[mask].sum() / tot)
        out[c + "_normal"] = float((gn ** 2).sum() / (e[mask].sum() + 1e-30))
        out[c + "_rough"] = rough_share(gn, xT0, mask, sp)
    # stage 2
    for c in ("gl_phys", "gl_sil", "gl_pbr", "gl_rend"):
        k = c + "_tmean"
        if k not in z.files:
            continue
        gl = z[k]; pn = z[c + "_pnorm"]
        out[c + "_layer"] = float((pn[mask] ** 2).sum() / ((pn ** 2).sum() + 1e-30))
        for r in (1, 2, 4, 8):
            out[f"{c}_corr{r}"] = corr_at(gl, x0, sp, r)
    # stage 2b / 3b — the position-mode channel u (docs/surface_gradient.md §7)
    if "u_final" in z.files:
        lm = z["layer_mask"] > 0.5
        xl = x0[lm]
        out["u_layer_n"] = float(lm.sum())
        uf = z["u_final"][lm]
        out["u_rms"] = float(np.sqrt((uf ** 2).mean()) / sp)
        out["u_clip"] = float((np.abs(uf) >= 0.999 * sp).mean())
        out["u_corr2"] = corr_at(uf[:, None], xl, sp, 2)
        for c in ("gu_phys", "gu_sil", "gu_pbr", "gu_rend"):
            if c not in z.files:
                continue
            gu = z[c][lm]
            e = float((gu ** 2).sum())
            out[c + "_norm"] = float(np.sqrt(e))
            out[c + "_rough"] = rough_share(gu, x0, lm, sp)
            for r in (1, 2, 4, 8):
                out[f"{c}_corr{r}"] = corr_at(gu[:, None], xl, sp, r)
        if "xT_base_u0" in z.files:
            xb0 = z["xT_base_u0"].astype(np.float32)
            mb0, nb0 = layer_by_asymmetry(xb0, sp)
            resb0, _ = plane_residual(xb0, mb0, nb0, sp)
            out["rms_base_u0"] = float(np.sqrt((resb0 ** 2).mean()))
            for c in ("phys", "sil", "pbr", "rend"):
                k = f"xT_{c}_u"
                if k not in z.files:
                    continue
                xc = z[k].astype(np.float32)
                dx = xc - xb0
                dn = (dx[mb0] * nb0[mb0]).sum(1)
                out[f"dxu_{c}_layer"] = float(np.linalg.norm(dx[mb0], axis=1).mean() / sp)
                out[f"dxu_{c}_interior"] = float(np.linalg.norm(dx[~mb0], axis=1).mean() / sp)
                out[f"dxu_{c}_rough"] = rough_share(dn, xb0, mb0, sp)
                mc, nc = layer_by_asymmetry(xc, sp)
                resc, _ = plane_residual(xc, mc, nc, sp)
                out[f"rmsu_{c}"] = float(np.sqrt((resc ** 2).mean()))
    # stage 3
    if "xT_base" in z.files:
        xb = z["xT_base"].astype(np.float32)
        mb, nb_ = layer_by_asymmetry(xb, sp)
        resb, _ = plane_residual(xb, mb, nb_, sp)
        out["rms_base"] = float(np.sqrt((resb ** 2).mean()))
        xf = z["xT_final"].astype(np.float32)
        mf, nf = layer_by_asymmetry(xf, sp)
        resf, _ = plane_residual(xf, mf, nf, sp)
        out["rms_final"] = float(np.sqrt((resf ** 2).mean()))
        for c in ("phys", "sil", "pbr", "rend"):
            k = "xT_" + c
            if k not in z.files:
                continue
            xc = z[k].astype(np.float32)
            dx = xc - xb
            dn = (dx[mb] * nb_[mb]).sum(1)
            out[f"dx_{c}_layer"] = float(np.linalg.norm(dx[mb], axis=1).mean() / sp)
            out[f"dx_{c}_interior"] = float(np.linalg.norm(dx[~mb], axis=1).mean() / sp)
            out[f"dx_{c}_normal"] = float((dn ** 2).sum() / ((dx[mb] ** 2).sum() + 1e-30))
            out[f"dx_{c}_rough"] = rough_share(dn, xb, mb, sp)
            mc, nc = layer_by_asymmetry(xc, sp)
            resc, _ = plane_residual(xc, mc, nc, sp)
            out[f"rms_{c}"] = float(np.sqrt((resc ** 2).mean()))
    return out


def main():
    d = sys.argv[1]
    sp_arg = float(sys.argv[sys.argv.index("--spacing") + 1]) if "--spacing" in sys.argv else None
    files = sorted(glob.glob(os.path.join(d, "win_*.npz")))
    rows = [analyse(f, sp_arg) for f in files]
    if not rows:
        print("no dumps"); return
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    print(f"{len(rows)} windows, N {rows[0]['n']}, spacing {rows[0]['spacing']:.4f} wu, layer {rows[0]['layer_frac']*100:.1f} % of particles")
    print("\nSTAGE 1 — terminal covector on the particles (share of |g|^2 on the outer layer; of that, along the normal; of that, rough at 2 sp)")
    print(f"{'channel':<10} {'norm':>10} {'layer':>7} {'normal':>7} {'rough':>7}")
    for c in ("gx_phys", "gx_sil", "gx_pbr"):
        if c + "_norm" in rows[0]:
            m = lambda k: np.nanmean([r[k] for r in rows])
            print(f"{c:<10} {m(c+'_norm'):10.3e} {m(c+'_layer'):7.3f} {m(c+'_normal'):7.3f} {m(c+'_rough'):7.3f}")
    print("\nSTAGE 2 — control gradient after the MPM adjoint (layer share; neighbour correlation at 1/2/4/8 spacings)")
    print(f"{'channel':<10} {'layer':>7} {'corr1':>7} {'corr2':>7} {'corr4':>7} {'corr8':>7}")
    for c in ("gl_phys", "gl_sil", "gl_pbr", "gl_rend"):
        if c + "_layer" in rows[0]:
            m = lambda k: np.nanmean([r[k] for r in rows])
            print(f"{c:<10} {m(c+'_layer'):7.3f} {m(c+'_corr1'):7.3f} {m(c+'_corr2'):7.3f} {m(c+'_corr4'):7.3f} {m(c+'_corr8'):7.3f}")
    if "rms_base" in rows[0]:
        print("\nSTAGE 3 — the window's response to each channel alone (same control norm as the accepted step)")
        print(f"{'channel':<8} {'|dx| layer':>10} {'|dx| int':>9} {'normal':>7} {'rough':>7} {'rms end':>8}   (spacings; rms = outer-layer plane residual)")
        m = lambda k: np.nanmean([r[k] for r in rows])
        print(f"{'x0':<8} {'':>10} {'':>9} {'':>7} {'':>7} {m('rms_xT0'):8.3f}   (state at the window's first iteration)")
        print(f"{'base':<8} {'':>10} {'':>9} {'':>7} {'':>7} {m('rms_base'):8.3f}   (no step)")
        print(f"{'actual':<8} {'':>10} {'':>9} {'':>7} {'':>7} {m('rms_final'):8.3f}   (the accepted step)")
        for c in ("phys", "sil", "pbr", "rend"):
            if f"dx_{c}_layer" in rows[0]:
                print(f"{c:<8} {m(f'dx_{c}_layer'):10.4f} {m(f'dx_{c}_interior'):9.4f} {m(f'dx_{c}_normal'):7.3f} {m(f'dx_{c}_rough'):7.3f} {m(f'rms_{c}'):8.3f}")
    if "u_rms" in rows[0]:
        m = lambda k: np.nanmean([r[k] for r in rows])
        print("\nSTAGE 2b — the u channel (position-mode control on the outer layer): each term's u-gradient — rough share at 2 sp, neighbour correlation")
        print(f"{'channel':<10} {'norm':>10} {'rough':>7} {'corr1':>7} {'corr2':>7} {'corr4':>7} {'corr8':>7}")
        for c in ("gu_phys", "gu_sil", "gu_pbr", "gu_rend"):
            if c + "_norm" in rows[0]:
                print(f"{c:<10} {m(c+'_norm'):10.3e} {m(c+'_rough'):7.3f} {m(c+'_corr1'):7.3f} {m(c+'_corr2'):7.3f} {m(c+'_corr4'):7.3f} {m(c+'_corr8'):7.3f}")
        print(f"accepted u: RMS {m('u_rms'):.3f} spacings on {rows[0]['u_layer_n']:.0f} layer particles, at the clip {m('u_clip')*100:.1f} %, corr at 2 sp {m('u_corr2'):.3f}")
        if "rms_base_u0" in rows[0]:
            print("\nSTAGE 3b — the window's response to each channel through u ALONE (start control, |u| = the accepted u's norm)")
            print(f"{'channel':<8} {'|dx| layer':>10} {'|dx| int':>9} {'rough':>7} {'rms end':>8}   (base u=0: {m('rms_base_u0'):.3f})")
            for c in ("phys", "sil", "pbr", "rend"):
                if f"dxu_{c}_layer" in rows[0]:
                    print(f"{c:<8} {m(f'dxu_{c}_layer'):10.4f} {m(f'dxu_{c}_interior'):9.4f} {m(f'dxu_{c}_rough'):7.3f} {m(f'rmsu_{c}'):8.3f}")
    print("\nper window: lam_r, g_share, step_norm, rms xT0 -> final")
    for i, r in enumerate(rows):
        print(f"  win {i:3d}: lam {r['lam_r']:.3g}  g_share {r['g_share']:.3f}  step {r['step_norm']:.3g}  rms {r['rms_xT0']:.3f} -> {r.get('rms_final', float('nan')):.3f}")


if __name__ == "__main__":
    main()
