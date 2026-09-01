"""Single-graph morphing loop: per-frame Adam on dFc + state promotion.

Phase 3: D_vol mass matching only. Plasticity (Phase 3.5) and render loss
(Phase 4) plug into the same loop. See docs/SPEC.md §6.

Stability: gradual deformation via per-particle dFc clip + grad clip, a coarse
(smooth) loss grid, and a domain/finite guard between frames.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from scipy.spatial import cKDTree

from ..losses.silhouette import d_img, ring_thetas, target_silhouettes
from ..losses.volumetric import d_vol, target_mass_grid
from ..mpm.constitutive import lame
from ..mpm.function import RolloutSpec, warp_mpm
from ..mpm.state import MPMParams
from ..plasticity import displacement_jacobian, sinkhorn_displacement, update_fp


def _id(N):
    return np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))


def _clip_dfc(dFc, max_norm):
    with torch.no_grad():
        nrm = dFc.flatten(1).norm(dim=1, keepdim=True)
        scale = (max_norm / nrm.clamp_min(1e-8)).clamp(max=1.0)
        dFc.mul_(scale.view(-1, 1, 1))


def _step_scale(k, K, min_scale, plateau=0.7):
    """Step-size schedule: ~constant (=1) for the first `plateau` fraction so the
    morph progresses evenly across all frames, then cosine-decays to min_scale
    for a smooth settle. Avoids 'done in 20 frames' big-early behaviour."""
    ks = plateau * K
    if k < ks:
        return 1.0
    t = (k - ks) / max(K - ks, 1)
    return min_scale + (1.0 - min_scale) * 0.5 * (1.0 + np.cos(np.pi * min(t, 1.0)))


def _pull_outliers(x, k=8, factor=2.5, alpha=0.8, passes=3):
    """Reincorporate isolated particles toward their kNN centroid (anti-float).
    Multi-pass so chains of stragglers collapse; touches only outliers (no shrink)."""
    if len(x) < k + 2:
        return x
    for _ in range(passes):
        d, idx = cKDTree(x).query(x, k=k + 1)
        out = d[:, -1] > factor * np.median(d[:, -1])
        if not out.any():
            break
        x[out] = (1 - alpha) * x[out] + alpha * x[idx[:, 1:]].mean(1)[out]
    return x.astype(np.float32)


def _condition_F(F, smin=0.5, smax=2.0):
    """Clamp singular values of F to prevent inversion/blow-up accumulation
    across frames (a stabilizer; proper rest-state migration is Phase 3.5)."""
    U, S, Vt = np.linalg.svd(F.astype(np.float32))
    S = np.clip(S, smin, smax)
    return np.einsum("nij,nj,njk->nik", U, S, Vt).astype(np.float32)


def morph_mass(source_x, target_x, prm: MPMParams, T=15, K=30, inner=4, lr=0.02,
               loss_res=32, dfc_clip=0.04, grad_clip=2.0, promote_F=False,
               render_lambda=0.0, render_views=6, render_res=64,
               min_scale=0.1, patience=6, tol=0.003, smooth_outliers=True,
               cohesion_w=8.0, cohesion_k=10, cohesion_slack=1.3, plateau=0.75,
               max_move=0.22, v_max=8.0, plasticity=False, eta=0.1,
               young=1.4e5, poisson=0.2, device="cuda", log=print, save_F=True):
    """Morph source -> target. L = D_vol + render_lambda * D_img (single graph).

    Smoothness/stability: cosine step-decay (large steps early to deform, tiny
    late to settle), convergence freeze (stop sloshing once plateaued), and
    per-frame outlier pull-back (anti-float). Returns (frames, Fs, history)."""
    N = source_x.shape[0]
    if v_max > 0:
        prm = replace(prm, v_max=v_max)               # enable G2P velocity clamp
    lam, mu = lame(young, poisson)
    m = torch.ones(N, device=device)

    thetas = ring_thetas(render_views)
    extent = float(np.abs(target_x).max()) * 1.25
    tgt_sil = None
    if render_lambda > 0:
        tgt_sil = target_silhouettes(torch.tensor(target_x, device=device, dtype=torch.float32),
                                     thetas, render_res, extent)

    dmin = np.asarray(prm.grid_min, np.float32)
    dmax = dmin + prm.dx * np.array([prm.nx, prm.ny, prm.nz], np.float32)
    ldx = float((dmax - dmin).max() / loss_res)
    ldims = (loss_res, loss_res, loss_res)
    lgmin = torch.tensor(dmin, device=device)
    tgt = target_mass_grid(torch.tensor(target_x, device=device, dtype=torch.float32),
                           m, lgmin, ldx, ldims)
    lo = torch.tensor(dmin + 2 * prm.dx, device=device)
    hi = torch.tensor(dmax - 2 * prm.dx, device=device)

    x0 = np.ascontiguousarray(source_x, np.float32)
    F0 = _id(N)
    Fp = _id(N)                                       # plastic rest state
    if plasticity:
        promote_F = True                              # plasticity needs kept F (elasticity)
    frames, Fs, hist = [x0.copy()], [F0.copy()], []
    L0 = float(d_vol(torch.tensor(x0, device=device), m, tgt, lgmin, ldx, ldims).item())
    log(f"[morph] N={N} T={T} K={K} inner={inner} lr={lr} loss_res={loss_res}  init D_vol={L0:.2f}")

    best, stale, frozen = L0, 0, False
    for k in range(K):
        if frozen:                                   # converged: hold a still shape
            frames.append(x0.copy()); Fs.append(F0.copy())
            hist.append({"frame": k + 1, "d_vol": best, "move": 0.0, "Jmin": 1.0, "held": 1})
            continue
        scale = _step_scale(k, K, min_scale, plateau)  # uniform then settle
        lr_k, clip_k = lr * scale, dfc_clip * scale
        spec = RolloutSpec(x0=x0, m=1.0, lam=lam, mu=mu, prm=prm, T=T, F0=F0, Fp=Fp)
        # frame-fixed kNN cohesion: penalize particles straying beyond their rest
        # neighbourhood -> NO scattering during optimization (anti-float at source)
        knn_t = r0_t = None
        if cohesion_w > 0:
            knn = cKDTree(x0).query(x0, k=cohesion_k + 1)[1][:, 1:]
            knn_t = torch.tensor(knn, device=device)
            x0_t = torch.tensor(x0, device=device)
            stray0 = (x0_t - x0_t[knn_t].mean(1)).norm(dim=1)
            # GLOBAL rest radius (median) -> stragglers in sparse regions are
            # penalized relative to typical density, so drifters get pulled back
            r0_t = stray0.median() * cohesion_slack + 1e-3
        dFc = torch.zeros(N, 3, 3, device=device, requires_grad=True)
        opt = torch.optim.Adam([dFc], lr=lr_k)
        for _ in range(inner):
            xT, FT = warp_mpm(dFc, spec)
            L = d_vol(xT, m, tgt, lgmin, ldx, ldims)
            if render_lambda > 0:
                L = L + render_lambda * d_img(xT, tgt_sil, thetas, render_res, extent)
            if cohesion_w > 0:
                stray = (xT - xT[knn_t].mean(1)).norm(dim=1)
                L = L + cohesion_w * torch.clamp(stray - r0_t, min=0).pow(2).mean()
            opt.zero_grad(); L.backward()
            torch.nn.utils.clip_grad_norm_([dFc], grad_clip)
            opt.step()
            _clip_dfc(dFc, clip_k)
        with torch.no_grad():
            xT, FT = warp_mpm(dFc, spec)
            Lf = float(d_vol(xT, m, tgt, lgmin, ldx, ldims).item())
            xT = torch.nan_to_num(xT, 0.0, 0.0, 0.0).clamp(min=lo, max=hi)
            FT = torch.nan_to_num(FT, 0.0, 0.0, 0.0)
            # hard per-particle displacement cap -> no fly-off, gradual morph
            if max_move > 0:
                x0_t = torch.tensor(x0, device=device)
                dpos = xT - x0_t
                dn = dpos.norm(dim=1, keepdim=True)
                xT = x0_t + dpos * (max_move * scale / dn.clamp_min(1e-8)).clamp(max=1.0)
        x_new = xT.cpu().numpy().astype(np.float32)
        if smooth_outliers:
            x_new = _pull_outliers(x_new)             # reincorporate floaters
        move = float(np.linalg.norm(x_new - x0, axis=1).mean())
        x0 = np.ascontiguousarray(x_new)
        F0 = _condition_F(FT.reshape(N, 3, 3).cpu().numpy()) if promote_F else _id(N)
        if plasticity:                                # migrate rest state (OT) -> elasticity ally
            d = sinkhorn_displacement(x0, target_x, eps_frac=0.05, iters=60, seed=k)
            dFp = displacement_jacobian(x0, d, k=12, diffusion_iters=2)
            Fp = update_fp(Fp, dFp, eta=eta, smin=0.4, smax=2.5, isochoric=True)
        frames.append(x0.copy())
        if save_F:
            Fs.append(F0.copy())
        # convergence tracking -> freeze (stops post-convergence oscillation)
        if Lf < best - tol * best:
            best, stale = Lf, 0
        else:
            stale += 1
        if stale >= patience:
            frozen = True
            log(f"[morph] converged at frame {k+1} (D_vol={Lf:.2f}); holding still")
        hist.append({"frame": k + 1, "d_vol": Lf, "move": move, "scale": float(scale),
                     "Jmin": float(np.linalg.det(F0).min())})
        log(f"[morph] frame {k+1}/{K}  D_vol={Lf:.2f}  move={move:.4f}  scale={scale:.2f}")
    return frames, Fs, hist
