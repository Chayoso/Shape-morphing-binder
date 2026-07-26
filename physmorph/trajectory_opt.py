"""Trajectory optimisation of the dFc control SEQUENCE — the Warp counterpart of the C++
CompGraph::OptimizeDefGradControlSequence.

Why this exists. `morph.py:morph_mass` solves a different problem from the C++ original:

                        C++ CompGraph                     morph_mass
  control               dFc[t], one per layer             ONE dFc shared by all T steps
  reset between frames  never                             zeroed every frame
  loss                  terminal only (EndLayerMassLoss)  every frame
  scheme                trajectory optimisation           greedy / myopic

The greedy scheme cannot represent "move differently now so the shape is reachable later"; the
trajectory scheme can. This module matches the C++ formulation, and adds the three step-control
devices the C++ has and the Warp loops lacked:

  * backtracking LINE SEARCH  — a step that does not decrease the loss (or produces a non-finite
    rollout) is REJECTED and the control + optimiser state are restored, alpha halved. This is the
    structural answer to "|dFc| is inside the clamp but the rollout blows up".
  * ADAPTIVE ALPHA            — alpha scaled by target_norm / ||grad|| so the first step is sane.
  * NORM-BASED LAMBDA         — lambda = alpha_lam * ||grad phys|| / ||grad render||, so the render
    term is balanced against the physics term automatically instead of by a hand-tuned constant.
    (Measured: at the default lambda the render term sat ~90x below D_vol and did nothing.)

Adam state persists across iterations, as in the C++ (`adam_timestep_` is global there).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .losses.silhouette import d_img, ring_thetas, target_silhouettes
from .losses.volumetric import d_vol, target_mass_grid
from .mpm.constitutive import lame
from .mpm.function import RolloutSpec, warp_mpm
from .mpm.state import MPMParams


@dataclass
class TrajOptConfig:
    T: int = 20                     # rollout length = number of control layers (C++ num_timesteps)
    iters: int = 60                 # outer optimisation iterations over the whole sequence
    alpha: float = 0.02             # initial step size (C++ initial_alpha)
    max_ls_iters: int = 10          # backtracking line-search attempts (C++ max_ls_iters)
    adaptive_alpha: bool = True     # C++ adaptive_alpha_enabled
    target_norm: float = 2500.0     # C++ adaptive_alpha_target_norm
    min_alpha_scale: float = 0.1    # C++ adaptive_alpha_min_scale
    gd_tol: float = 1e-3            # stop when ||grad|| < gd_tol * ||grad_0||
    dfc_clip: float = 0.0           # optional per-particle |dFc| cap (0 = off, C++ has none)
    w_ctrl: float = 0.0             # control-energy weight: sum_t ||dFc[t]||^2 / (T N)

    loss_res: int = 32              # D_vol grid resolution
    render_lambda: float = 0.0      # 0 = physics only; >0 = fixed weight
    lambda_auto: float = 0.0        # >0 = norm-based balancing, overrides render_lambda
    render_views: int = 6
    render_res: int = 64
    young: float = 1.4e5
    poisson: float = 0.2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    device: str = "cuda"
    history: list = field(default_factory=list)


def _grad_norm(g: torch.Tensor) -> float:
    return float(g.norm().item())


def optimize_sequence(source_x, target_x, prm: MPMParams, cfg: TrajOptConfig, log=print,
                      F0=None, Fp=None, v0=None, C0=None):
    """Optimise dFc[0..T-1] so the state after T steps matches the target. Returns
    (frames, dFc, history) where frames are the states along the FINAL rollout."""
    dev = cfg.device
    N = source_x.shape[0]
    assert target_x.shape[0] == N, ("D_vol compares unit-mass clouds: source and target need the "
                                    f"same particle count (got {N} vs {target_x.shape[0]})")
    lam_e, mu_e = lame(cfg.young, cfg.poisson)
    m = torch.ones(N, device=dev)

    dmin = np.asarray(prm.grid_min, np.float32)
    dmax = dmin + prm.dx * np.array([prm.nx, prm.ny, prm.nz], np.float32)
    ldx = float((dmax - dmin).max() / cfg.loss_res)
    ldims = (cfg.loss_res,) * 3
    lgmin = torch.tensor(dmin, device=dev)
    tgt_t = torch.tensor(target_x, device=dev, dtype=torch.float32)
    tgt_grid = target_mass_grid(tgt_t, m, lgmin, ldx, ldims)

    use_render = cfg.render_lambda > 0 or cfg.lambda_auto > 0
    thetas = ring_thetas(cfg.render_views)
    extent = float(np.abs(target_x).max()) * 1.25
    tgt_sil = target_silhouettes(tgt_t, thetas, cfg.render_res, extent) if use_render else None

    spec = RolloutSpec(x0=np.ascontiguousarray(source_x, np.float32), m=1.0, lam=lam_e, mu=mu_e,
                       prm=prm, T=cfg.T, F0=F0, Fp=Fp, v0=v0, C0=C0, device=dev)

    # the control SEQUENCE — one field per layer, never reset (the C++ semantics)
    dFc = torch.zeros(cfg.T, N, 3, 3, device=dev, requires_grad=True)
    # persistent Adam moments, as in C++ PointCloud::Descend_Adam
    mom = torch.zeros_like(dFc)
    vel = torch.zeros_like(dFc)
    adam_t = 0

    def terms(dfc):
        """Forward + the two loss terms, kept separate so they can be balanced by norm."""
        xT, _ = warp_mpm(dfc, spec)
        lv = d_vol(xT, m, tgt_grid, lgmin, ldx, ldims)
        li = d_img(xT, tgt_sil, thetas, cfg.render_res, extent) if use_render else None
        return xT, lv, li

    def ctrl_cost(dfc):
        """Terminal-only mass matching rewards ARRIVING, not arriving gently: a solution that
        launches the material ballistically scores the same as one that walks it there. Nothing in
        D_vol or D_img sees how hard the body was driven, so the optimiser is free to slam it and
        the run diverges. This is the missing running cost."""
        return dfc.pow(2).sum() / (dfc.shape[0] * dfc.shape[1])

    def total(lv, li, lam_r, dfc=None):
        L = lv if li is None else lv + lam_r * li
        if cfg.w_ctrl > 0 and dfc is not None:
            L = L + cfg.w_ctrl * ctrl_cost(dfc)
        return L

    lam_r = cfg.render_lambda
    with torch.no_grad():
        x_init, lv0, li0 = terms(dFc)
    log(f"[traj] N={N} T={cfg.T} iters={cfg.iters} alpha={cfg.alpha} "
        f"ls={cfg.max_ls_iters} adaptive={cfg.adaptive_alpha} "
        f"render={'auto' if cfg.lambda_auto > 0 else cfg.render_lambda}")
    log(f"[traj] initial D_vol={float(lv0):.3f}" +
        (f"  D_img={float(li0):.5f}" if li0 is not None else ""))

    alpha = cfg.alpha
    g0_norm = None
    accepted = rejected = 0
    for it in range(cfg.iters):
        # ---- gradient of the current point (and, if balancing, of each term separately) ----
        if cfg.lambda_auto > 0:
            xT, lv, li = terms(dFc)
            gp = torch.autograd.grad(lv, dFc, retain_graph=True)[0]
            gr = torch.autograd.grad(li, dFc)[0]
            np_, nr_ = _grad_norm(gp), _grad_norm(gr)
            lam_r = cfg.lambda_auto * np_ / max(nr_, 1e-12)      # C++: alpha*phys/render
            if cfg.w_ctrl > 0:
                gc = torch.autograd.grad(cfg.w_ctrl * ctrl_cost(dFc), dFc)[0]
                gp = gp + gc
            g = gp + lam_r * gr
            cur = float(total(lv, li, lam_r, dFc))
        else:
            xT, lv, li = terms(dFc)
            L = total(lv, li, lam_r, dFc)
            g = torch.autograd.grad(L, dFc)[0]
            cur = float(L)
        if not np.isfinite(cur):
            log(f"[traj] iter {it}: non-finite loss, aborting")
            break
        gn = _grad_norm(g)
        if g0_norm is None:
            g0_norm = max(gn, 1e-12)
        if gn < cfg.gd_tol * g0_norm:
            log(f"[traj] converged at iter {it} (||g||={gn:.4g} < {cfg.gd_tol}*{g0_norm:.4g})")
            break

        # ---- adaptive alpha: keep the first move sane when gradients are large (C++) ----
        a_try = alpha
        if cfg.adaptive_alpha:
            a_try *= max(cfg.min_alpha_scale, min(1.0, cfg.target_norm / max(gn, 1e-6)))

        # ---- backtracking line search over the Adam step (C++ dFc_bak / restore / halve) ----
        bak_d, bak_m, bak_v = dFc.detach().clone(), mom.clone(), vel.clone()
        step_ok = False
        for _ls in range(cfg.max_ls_iters):
            with torch.no_grad():
                t_ = adam_t + 1
                mom.mul_(cfg.beta1).add_(g, alpha=1 - cfg.beta1)
                vel.mul_(cfg.beta2).addcmul_(g, g, value=1 - cfg.beta2)
                mh = mom / (1 - cfg.beta1 ** t_)
                vh = vel / (1 - cfg.beta2 ** t_)
                dFc -= a_try * mh / (vh.sqrt() + cfg.eps)
                if cfg.dfc_clip > 0:
                    n = dFc.flatten(2).norm(dim=2, keepdim=True).unsqueeze(-1)
                    dFc *= (cfg.dfc_clip / n.clamp_min(1e-8)).clamp(max=1.0)
                _, lv_n, li_n = terms(dFc)
                new = float(total(lv_n, li_n, lam_r, dFc))
            if np.isfinite(new) and new < cur:
                adam_t = t_
                alpha = min(a_try * 1.1, cfg.alpha)               # C++ grows alpha on acceptance
                step_ok = True
                accepted += 1
                break
            with torch.no_grad():                                 # reject: restore and halve
                dFc.copy_(bak_d); mom.copy_(bak_m); vel.copy_(bak_v)
            a_try *= 0.5
        if not step_ok:
            rejected += 1
            log(f"[traj] iter {it}: line search failed (loss {cur:.4f}, ||g||={gn:.3g}) — "
                f"alpha exhausted from {alpha:.2e}")
            alpha *= 0.5
            if alpha < 1e-8:
                log("[traj] alpha underflow, stopping")
                break
            continue

        with torch.no_grad():
            _, lv_c, li_c = terms(dFc)
        cfg.history.append({"iter": it, "loss": new, "d_vol": float(lv_c),
                            "d_img": float(li_c) if li_c is not None else None,
                            "lambda": lam_r, "grad_norm": gn, "alpha": a_try,
                            "dfc_absmax": float(dFc.abs().max())})
        if it % max(1, cfg.iters // 12) == 0 or it == cfg.iters - 1:
            log(f"[traj] iter {it:3d}  L={new:.4f}  D_vol={float(lv_c):.3f}" +
                (f"  D_img={float(li_c):.5f}  lam={lam_r:.3g}" if li_c is not None else "") +
                f"  |g|={gn:.3g}  alpha={a_try:.2e}  |dFc|max={float(dFc.abs().max()):.4f}")

    # ---- final rollout, keeping every intermediate state for rendering/metrics ----
    with torch.no_grad():
        from .mpm.traj import Trajectory
        import warp as wp
        dc = dFc.detach().contiguous()
        seq = [wp.from_torch(dc[t].view(N, 3, 3), dtype=wp.mat33) for t in range(cfg.T)]
        tr = Trajectory(spec.x0, 1.0, lam_e, mu_e, prm, cfg.T, F0=F0, Fp=Fp, v0=v0, C0=C0,
                        dFc=seq, device=dev, requires_grad=False)
        tr.rollout()
        frames = [tr.x[t].numpy().copy() for t in range(cfg.T + 1)]
        # the FULL end state — promoting only (x, F) drops momentum and the APIC affine field,
        # which leaves an elastically loaded body frozen and re-injects that energy every commit
        end = {"F": tr.F[cfg.T].numpy().copy(), "v": tr.v[cfg.T].numpy().copy(),
               "C": tr.C[cfg.T].numpy().copy()}
    log(f"[traj] done: {accepted} steps accepted, {rejected} line-search failures")
    return frames, dFc.detach(), cfg.history, end


def _condition_F(F, smin=0.5, smax=2.0):
    """Clamp singular values, and REPAIR inversion instead of preserving it.

    The version in morph.py rebuilds U diag(clip(S)) V^T directly. numpy's SVD may return a U, V
    pair with det(U)det(V^T) = -1, and clamping the (positive) singular values keeps that
    reflection: feeding it det F = -27 returns det = -8, still inverted. So the "stabiliser"
    cannot undo an inversion, it only rescales one. Flipping the last column of U when the pair is
    improper — exactly what corotated_R does in constitutive.py — makes the output a genuine
    orientation-preserving deformation. Non-finite rows are reset to identity first (numpy raises
    on NaN input)."""
    F = np.ascontiguousarray(F, np.float32).reshape(-1, 3, 3)
    bad = ~np.isfinite(F).all(axis=(1, 2))
    if bad.any():
        F = F.copy()
        F[bad] = np.eye(3, dtype=np.float32)
    U, S, Vt = np.linalg.svd(F)
    flip = np.linalg.det(U) * np.linalg.det(Vt) < 0        # improper pair -> reflection
    n_flip = int(flip.sum())
    if n_flip:
        U = U.copy()
        U[flip, :, -1] *= -1.0
    out = np.einsum("nij,nj,njk->nik", U, np.clip(S, smin, smax), Vt).astype(np.float32)
    return out, int(bad.sum()), n_flip


def optimize_morph(source_x, target_x, prm: MPMParams, cfg: TrajOptConfig, animations=30,
                   promote_F=True, guard=True, log=print):
    """The FULL C++ structure: an outer animation loop around the inner trajectory optimisation.

    The C++ driver reads `cg.get_point_cloud(num_timesteps - 1)` after each `run_optimization`:
    it optimises a T-layer control sequence, COMMITS THE WHOLE HORIZON, and restarts from the end
    state. Crucially a C++ layer is a whole MaterialPoint — x, v, F AND C — so the restart carries
    the COMPLETE state. Promoting only (x, F) freezes an elastically loaded body (F != I means
    stored energy) at v = 0, C = 0, which re-releases that energy every commit and diverges after a
    few animations. That is a formulation bug, not something a clamp should paper over.

    `guard` only COUNTS domain clamps / non-finite F resets; with the full state promoted the
    counters should stay at zero, which is the test that the fix is real rather than masked.
    """
    x = np.ascontiguousarray(source_x, np.float32)
    st = {"F": None, "v": None, "C": None}
    dmin = np.asarray(prm.grid_min, np.float32)
    dmax = dmin + prm.dx * np.array([prm.nx, prm.ny, prm.nz], np.float32)
    lo, hi = dmin + 2 * prm.dx, dmax - 2 * prm.dx
    frames, hist_all = [x.copy()], []
    for a in range(animations):
        cfg.history = []
        fr, dFc, hist, end = optimize_sequence(x, target_x, prm, cfg, log=lambda *_: None,
                                               F0=st["F"], v0=st["v"], C0=st["C"])
        if not hist:
            log(f"[morph-traj] anim {a + 1}: no accepted step, stopping")
            break
        x_new = np.ascontiguousarray(fr[-1])
        n_out = int(((x_new < lo) | (x_new > hi)).any(1).sum()) if guard else 0
        n_nan = int((~np.isfinite(x_new).all(1)).sum())
        x = np.clip(np.nan_to_num(x_new), lo, hi).astype(np.float32) if guard else x_new
        Fc, nbad, nflip = _condition_F(end["F"]) if promote_F else (None, 0, 0)
        st = {"F": Fc, "v": end["v"], "C": end["C"]}          # FULL state promoted
        frames.extend(fr[1:])
        hist_all.append({"animation": a, "iters": len(hist),
                         "d_vol": hist[-1]["d_vol"], "d_img": hist[-1]["d_img"],
                         "lambda": hist[-1]["lambda"], "dfc_absmax": float(dFc.abs().max()),
                         "clamped": n_out, "nan_x": n_nan, "F_reset": nbad, "F_flip": nflip,
                         "v_absmax": float(np.abs(end["v"]).max()),
                         "Jmin": float(np.linalg.det(Fc).min()) if Fc is not None else 1.0})
        h = hist_all[-1]
        if a % max(1, animations // 10) == 0 or a == animations - 1 or n_out or nbad or n_nan or nflip:
            log(f"[morph-traj] anim {a + 1}/{animations}  D_vol={h['d_vol']:.3f}" +
                (f"  D_img={h['d_img']:.5f}  lam={h['lambda']:.3g}" if h['d_img'] is not None else "") +
                f"  |dFc|max={h['dfc_absmax']:.4f}  |v|max={h['v_absmax']:.3f}  Jmin={h['Jmin']:.4f}" +
                (f"  CLAMPED={n_out}" if n_out else "") + (f"  NAN_x={n_nan}" if n_nan else "") +
                (f"  F_reset={nbad}" if nbad else "") + (f"  F_flip={nflip}" if nflip else ""))
    return frames, hist_all
