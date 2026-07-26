"""PhysMorph-ST top-level loop — physical 3D->3D style transfer for 3DGS.

The deformation is driven by the RENDER style gradient (appearance + geometric-texture)
and realized by a SMOOTH elastic MPM flow (neighborhood-preserving), so the texture
RIDES the deformation F (method.md A1-A4) — unlike the measure-preserving-but-scrambling
OT shape-morph (which is the content_weight=0, texture-free special case in
morph_physical.morph_elastoplastic).

  TRANSPORT  (weak, heavily smoothed sliced-OT)  -> coarse geometry basin + identity anchor
  RENDER     (multi-view silhouette/color)       -> appearance + geometric-texture style
  PHYSICS    (MPM, moderate young, plastic Fp)   -> smooth, volume-respecting, coherent motion
"""
from __future__ import annotations

import numpy as np

from .losses.render_guidance import build_target_color_grams, build_target_views, color_gram_grad, render_guidance
from .mpm import MPMParams, compute_volumes, make_state, mpm_step
from .mpm.constitutive import lame
from .morph_physical import _repel, _smooth_field, _taubin
from .plasticity import balanced_assignment, displacement_jacobian, sliced_ot_displacement, update_fp


def _id(n):
    return np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))


def _rms(a):
    return float(np.sqrt((a ** 2).sum(1).mean())) + 1e-9


def style_transfer_3dgs(content_x, style_x, prm: MPMParams, content_color=None, style_color=None,
                        K=120, T=16, render_gain=1.0, transport_gain=0.5, content_weight=0.3,
                        young=3.0e4, poisson=0.2, vel_gain=10.0, vel_keep=0.2, max_disp=0.6,
                        smooth_iters=10, repel=0.12, taubin_iters=0, color_gain=0.3,
                        kinematic=False, render_views=16, render_res=96, device="cuda", log=print):
    """Stylize `content_x` toward `style_x` with a smooth, texture-coherent physical
    deformation. Returns (frames, Fs, colors, hist): per-frame positions, the source->frame
    deformation gradient (for Sigma=sigma0^2 F F^T), the advected per-particle color, history.

    Smoothness (=texture coherence) is enforced by: heavy field smoothing (`smooth_iters`),
    moderate elasticity, a content-identity leash, and NO auction snap / heavy repel.
    """
    import warp as wp
    N = content_x.shape[0]
    lam, mu = lame(young, poisson)
    x0 = np.ascontiguousarray(content_x, np.float32)
    x = x0.copy()
    F = _id(N); Fp = _id(N); v = np.zeros((N, 3), np.float32)
    color = (np.tile(np.array([0.45, 0.5, 0.55], np.float32), (N, 1))
             if content_color is None else np.ascontiguousarray(content_color, np.float32))
    s0 = make_state(x, 1.0, lam, mu, prm, F=F, Fp=Fp, device=device)
    compute_volumes(s0, prm); vol0 = s0.vol.numpy().copy()

    # identity anchor (transport): each content particle keeps a stable slot (head->head)
    a = x0.copy() if content_weight <= 0 else balanced_assignment(x0, x0, k=16, seed=0)
    # render style targets: silhouette (geometry-style) + color-Gram (appearance-style)
    extent = float(np.abs(style_x).max()) * 1.25
    tgt_views, thetas = build_target_views(style_x, render_views, render_res, extent, device=device)
    cgrams = cmeans = cthetas = None
    if style_color is not None and color_gain > 0:             # render OWNS appearance
        cgrams, cmeans, cthetas = build_target_color_grams(
            style_x, style_color, render_views, render_res, extent, device=device)

    frames, Fs, colors, hist = [x.copy()], [F.copy()], [color.copy()], []
    for k in range(K):
        # (1) RENDER style guidance (smooth, diffuse) -> geometry style toward the exemplar
        d_r = render_guidance(x, tgt_views, thetas, render_res, extent, device=device)
        # (2) weak heavily-smoothed transport prior (coarse basin, NOT a bijection -> stays smooth)
        d_t = _smooth_field(x, sliced_ot_displacement(x, style_x, n_dirs=32, seed=k), iters=6)
        # (3) content identity leash
        d_c = (a - x).astype(np.float32)
        d = (render_gain * d_r / _rms(d_r) + transport_gain * d_t / _rms(d_t)
             + content_weight * d_c / _rms(d_c))
        d = _smooth_field(x, d, iters=smooth_iters)            # KEY: heavy smoothing -> texture coherence
        dn = np.linalg.norm(d, axis=1, keepdims=True)
        d = d * np.minimum(1.0, max_disp / np.maximum(dn, 1e-8))
        # (4) plastic rest-state migration (smooth) + (5) elastic MPM rollout
        if kinematic:                                          # ABLATION (physics off): advect, no MPM
            x = (x + d).astype(np.float32)
            F = displacement_jacobian(x0, x - x0, k=12, diffusion_iters=1).astype(np.float32)
        else:
            Fp = update_fp(Fp, displacement_jacobian(x, d, k=12, diffusion_iters=2),
                           eta=0.15, smin=0.5, smax=2.0, isochoric=True)
            v = (v * vel_keep + vel_gain * d).astype(np.float32)
            st = make_state(x, 1.0, lam, mu, prm, v=v, F=F, Fp=Fp, device=device)
            st.vol = wp.array(vol0, dtype=wp.float32, device=device)
            for _ in range(T):
                mpm_step(st, prm)
            wp.synchronize()
            x = st.x.numpy().astype(np.float32); F = st.F.numpy().astype(np.float32); v = st.v.numpy().astype(np.float32)
        if repel > 0:                                          # light only -> density without scrambling texture
            x = _repel(x, _nn(x), strength=repel, iters=1)
        if taubin_iters > 0:
            x = _taubin(x, iters=taubin_iters)
        # (6) RENDER restyles APPEARANCE: color steps toward the style's color-Gram (then rides F)
        if cgrams is not None:
            d_col, _ = color_gram_grad(x, color, cgrams, cmeans, cthetas, render_res, extent, device=device)
            color = np.clip(color + color_gain * d_col / (_rms(d_col)), 0.0, 1.0).astype(np.float32)
        frames.append(x.copy()); Fs.append(F.copy()); colors.append(color.copy())
        hist.append({"frame": k + 1, "Jmin": float(np.linalg.det(F).min()),
                     "move": float(np.linalg.norm(x - x0, axis=1).mean())})
        log(f"[st] {k+1}/{K} Jmin={hist[-1]['Jmin']:.3f} move={hist[-1]['move']:.3f}")
    return frames, Fs, colors, hist


def _nn(x):
    from scipy.spatial import cKDTree
    return float(np.median(cKDTree(x).query(x, k=2)[0][:, 1]))
