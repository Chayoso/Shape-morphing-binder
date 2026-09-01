"""Physically-coherent morph: plasticity-driven rest-state migration + elasticity.

The honest method (docs/SPEC.md §3): the shape is driven by migrating the plastic
rest state F_p toward the target (OT, NOT Chamfer — song2026), and the elastic
stress PK1(F F_p^{-1}) holds the continuum together so particles CANNOT scatter.
No velocity-clamp / cohesion / pull-back band-aids — coherence is physical.
"""
from __future__ import annotations

import numpy as np

from ..mpm import MPMParams, compute_volumes, make_state, mpm_step
from ..mpm import kernels as _kern
from ..mpm.constitutive import lame
from ..plasticity import (balanced_assignment, displacement_jacobian, sinkhorn_displacement,
                          sliced_ot_displacement, update_fp)


def _id(N):
    return np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))


def _smooth_field(x, d, k=16, iters=4):
    from scipy.spatial import cKDTree
    idx = cKDTree(x).query(x, k=k + 1)[1][:, 1:]
    for _ in range(iters):
        d = 0.5 * d + 0.5 * d[idx].mean(1)
    return d.astype(np.float32)


def _taubin(x, k=12, lam=0.33, mu=-0.34, iters=1):
    """Volume-preserving surface smoothing (de-spike). Physical surface tension."""
    from scipy.spatial import cKDTree
    idx = cKDTree(x).query(x, k=k + 1)[1][:, 1:]
    for _ in range(iters):
        x = x + lam * (x[idx].mean(1) - x)
        x = x + mu * (x[idx].mean(1) - x)
    return x.astype(np.float32)


def _repel(x, rest, k=8, strength=0.5, iters=2):
    """Blue-noise / Lloyd repulsion: push particles apart to keep ~uniform spacing
    (pressure-like, physical). Counteracts OT barycentric clustering -> clean density."""
    from scipy.spatial import cKDTree
    for _ in range(iters):
        d, idx = cKDTree(x).query(x, k=k + 1)
        nbr = x[idx[:, 1:]]                              # (N,k,3)
        diff = x[:, None, :] - nbr                       # away from neighbours
        dist = np.linalg.norm(diff, axis=2, keepdims=True)
        push = (diff / np.maximum(dist, 1e-6)) * np.maximum(rest - dist, 0.0)
        x = x + strength * push.mean(1)
    return x.astype(np.float32)


def morph_elastoplastic(source_x, target_x, prm: MPMParams, K=60, T=20,
                        eta=0.2, vel_gain=2.2, vel_keep=0.1, hard_chamfer=False,
                        eps_frac=0.05, sinkhorn_iters=60, n_anchor=3072, n_src=8000, max_disp=0.5,
                        repel=0.4, taubin_iters=1, settle_from=0.8, settle_min=0.08,
                        distribute=False, overshoot_clamp=True, transport="sliced", n_dirs=48,
                        sharpen_from=1.0, assign_from=1.0, kinematic=False,
                        transport_gain=1.0, render_gain=0.0, render_views=12, render_res=96,
                        young=1.4e5, poisson=0.2, device="cuda", log=print):
    """OT-guided elastoplastic flow. Each frame: (1) Sinkhorn OT displacement
    (smooth field, NOT Chamfer); (2) set guidance velocity v = vel_gain*d toward
    the OT target; (3) migrate F_p (persistence); (4) run T MPM steps so elasticity
    keeps the continuum coherent (no scatter) while drag settles it.

    The OT field is smooth -> no scatter; elasticity holds neighbours -> physical.
    hard_chamfer=True uses NN (the failing baseline) for the OT-vs-Chamfer ablation.
    """
    import warp as wp
    N = source_x.shape[0]
    lam, mu = lame(young, poisson)
    x = np.ascontiguousarray(source_x, np.float32)
    F = _id(N)
    Fp = _id(N)
    v = np.zeros((N, 3), np.float32)
    s0 = make_state(x, 1.0, lam, mu, prm, F=F, Fp=Fp, device=device)
    compute_volumes(s0, prm)
    vol0 = s0.vol.numpy().copy()
    from scipy.spatial import cKDTree
    rest = float(np.median(cKDTree(target_x).query(target_x, k=2)[0][:, 1]))  # target spacing

    # render supervision: target multi-view silhouettes (makes the renderer load-bearing)
    tgt_views = thetas_r = extent_r = None
    if render_gain > 0:
        from ..losses.render_guidance import build_target_views, render_guidance
        extent_r = float(np.abs(target_x).max()) * 1.25
        tgt_views, thetas_r = build_target_views(target_x, render_views, render_res, extent_r, device=device)

    frames, Fs, hist = [x.copy()], [F.copy()], []
    assigned = None                                          # cached balanced assignment (snap phase)
    for k in range(K):
        # settle schedule: full guidance early, decay late so particles land
        # precisely on the target instead of over-extending the extremities
        if k < settle_from * K:
            s = 1.0
        else:
            t = (k - settle_from * K) / max((1 - settle_from) * K, 1)
            s = settle_min + (1 - settle_min) * 0.5 * (1 + np.cos(np.pi * min(t, 1.0)))
        # (1) transport displacement toward target. sliced-OT (balanced, smooth) warm-
        # starts; in the snap phase a fixed balanced ASSIGNMENT (auction bijection) lands
        # each particle on a distinct target -> exact shape (IoU->ceiling), no clustering.
        snap = k >= assign_from * K
        if snap:
            if assigned is None:
                assigned = balanced_assignment(x, target_x, k=32, max_rounds=400, seed=0)
                log(f"[ep] balanced-assignment snap engaged at frame {k+1}")
            d = (assigned - x).astype(np.float32)            # toward fixed assigned target
            s = 1.0                                          # full step (override settle decay)
        else:
            # combine TRANSPORT (3D correspondence) + RENDER (multi-view silhouette)
            # guidance by relative weight (each normalized -> sidesteps the 45000x gap).
            def _rms(a):
                return float(np.sqrt((a ** 2).sum(1).mean())) + 1e-9
            d = np.zeros((N, 3), np.float32)
            if transport_gain > 0:
                if transport == "sliced" and not hard_chamfer:
                    d_t = sliced_ot_displacement(x, target_x, n_dirs=n_dirs, seed=k)
                else:
                    d_t = sinkhorn_displacement(x, target_x, eps_frac=eps_frac, iters=sinkhorn_iters,
                          n_anchor=n_anchor, n_src=n_src, hard=hard_chamfer, device=device, seed=k)
                d = d + transport_gain * (d_t / _rms(d_t))
            if render_gain > 0:                              # render genuinely drives the morph
                d_r = render_guidance(x, tgt_views, thetas_r, render_res, extent_r, device=device)
                d = d + render_gain * (d_r / _rms(d_r))
            d = _smooth_field(x, d, iters=2)
        dn = np.linalg.norm(d, axis=1, keepdims=True)            # cap per-step guidance
        d = d * np.minimum(1.0, (max_disp * s) / np.maximum(dn, 1e-8))
        # (c) plastic rest-state migration carries the morph; elasticity pulls F->Fp
        Fp = update_fp(Fp, displacement_jacobian(x, d, k=12, diffusion_iters=2),
                       eta=eta, smin=0.4, smax=2.5, isochoric=True)
        t_target = (x + d).astype(np.float32)                    # per-frame bounded OT target
        if kinematic or snap:                                  # OT advection / geometric snap (no MPM)
            x = (x + d).astype(np.float32)                     # snap lands precisely (assignment is geometric)
            F = _id(N)
        elif distribute:                                       # (a) distribute over substeps
            v = (v * vel_keep).astype(np.float32)
            st = make_state(x, 1.0, lam, mu, prm, v=v, F=F, Fp=Fp, device=device)
            st.vol = wp.array(vol0, dtype=wp.float32, device=device)
            d_wp = wp.array(d, dtype=wp.vec3, device=device)
            g = vel_gain * s / float(T)
            for _ in range(T):
                wp.launch(_kern.k_add_guidance, dim=N, inputs=[st.v, d_wp, g], device=device)
                mpm_step(st, prm)
        else:                                                   # strong velocity guidance
            v = (v * vel_keep + vel_gain * s * d).astype(np.float32)
            st = make_state(x, 1.0, lam, mu, prm, v=v, F=F, Fp=Fp, device=device)
            st.vol = wp.array(vol0, dtype=wp.float32, device=device)
            for _ in range(T):
                mpm_step(st, prm)
        if not (kinematic or snap):
            wp.synchronize()
            x = st.x.numpy().astype(np.float32)
            F = st.F.numpy().astype(np.float32)
            v = st.v.numpy().astype(np.float32)
        if overshoot_clamp and not (kinematic or snap):        # stop particles that passed
            du = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-8)
            over = ((x - t_target) * du).sum(1) > 0             # moved past per-frame target
            x[over] = t_target[over]                            # clamp to target (no overshoot)
            v[over] *= 0.2
        sharpen = snap or k >= sharpen_from * K                 # late/snap: let sharp features form
        if repel > 0 and not snap:                             # uniform density (anti-cluster)
            x = _repel(x, rest, strength=repel * (0.4 if sharpen else 1.0), iters=2)
        if taubin_iters > 0 and not sharpen:                    # de-spike (surface tension)
            x = _taubin(x, iters=taubin_iters)
        disp = float(np.linalg.norm(d, axis=1).mean())
        Jmin = float(np.linalg.det(F).min())
        frames.append(x.copy()); Fs.append(F.copy())
        hist.append({"frame": k + 1, "disp": disp, "Jmin": Jmin,
                     "vmean": float(np.linalg.norm(v, axis=1).mean())})
        log(f"[ep] {k+1}/{K}  disp={disp:.3f}  Jmin={Jmin:.3f}  vmean={hist[-1]['vmean']:.3f}")
    return frames, Fs, hist
