"""run_pipeline — outer commit loop of the v2 blessed path (docs/pipeline_v2.md §3.5, §3.7).

Per commit: optimise one window, promote the FULL state (x, F, v, C — partial promotion was
the v1 energy-re-injection bug), assimilate an eta-fraction of the elastic stretch into the
plastic rest state Fp (exact polar relaxation — render→Fp channel), track convergence on the
RAW loss components (λ-free; a drifting λ must not decide the freeze) and freeze on plateau.

Guards are containment + telemetry: a fired guard means the run is INVALID (gate G2 requires
every counter to read zero); the sanitisation only prevents one poisoned state from cascading
into meaningless downstream telemetry. F is repaired for numerical pathologies only
(non-finite rows, reflections — both counted); there is NO silent singular-value projection
in this path. The archived frames are the PROMOTED states, so metrics, plasticity and the
next window all describe the same trajectory.
"""
from __future__ import annotations

import numpy as np
import torch

from ..losses.volumetric import d_vol, target_mass_grid
from ..mpm.conditioning import condition_F
from ..mpm.state import MPMParams
from ..plasticity import assimilate_elastic
from .config import PipelineConfig
from .optimizer import TargetPack, optimize_window
from .render_loss import (LambdaBalancer, d_render, make_views, shade_targets,
                          target_dts, target_silhouettes)
from .surface_local import surface_local_pass


def _id(N):
    return np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))


def build_target(target_x, prm: MPMParams, cfg: PipelineConfig) -> TargetPack:
    dev = cfg.device
    N = target_x.shape[0]
    m = torch.ones(N, device=dev)
    dmin = np.asarray(prm.grid_min, np.float32)
    dmax = dmin + prm.dx * np.array([prm.nx, prm.ny, prm.nz], np.float32)
    ldx = float((dmax - dmin).max() / cfg.loss_res)
    ldims = (cfg.loss_res,) * 3
    lgmin = torch.tensor(dmin, device=dev)
    tgt_t = torch.tensor(np.ascontiguousarray(target_x, np.float32), device=dev)
    grid = target_mass_grid(tgt_t, m, lgmin, ldx, ldims)
    views = make_views(cfg.render_views, cfg.render_elevs)
    extent = float(np.abs(target_x).max()) * 1.25
    sils = shade = dts = None
    if cfg.lambda_auto > 0:
        sils = target_silhouettes(tgt_t, views, cfg.render_res, extent, cfg.sil_k)
        if cfg.w_pbr > 0:
            shade = shade_targets(tgt_t, views, cfg.render_res, extent,
                                  lgmin, ldx, ldims, cfg.sil_k, cfg.pbr_ambient)
        if cfg.w_dt > 0:
            dts = target_dts(sils, extent, cfg.dt_clamp_frac, cfg.dt_mask_thresh)
    return TargetPack(grid=grid, lgmin=lgmin, ldx=ldx, ldims=ldims, m=m,
                      views=views, sils=sils, extent=extent, shade=shade, dts=dts)


def run_pipeline(source_x, target_x, prm: MPMParams, cfg: PipelineConfig, log=print,
                 on_commit=None, on_iter=None):
    """Morph source -> target. Returns a result dict (frames, F_frames, history, guards, s,
    n_held, converged). frames/F_frames archive the PROMOTED per-step states.
    on_commit(a, x, F, v, rec) fires after each promoted commit; on_iter(it, xT, FT, tele)
    streams each accepted optimisation iteration (live viewer hooks)."""
    src = np.ascontiguousarray(source_x, np.float32)
    N = src.shape[0]
    assert target_x.shape[0] == N, ("D_vol compares unit-mass clouds: source and target need "
                                    f"the same particle count (got {N} vs {target_x.shape[0]})")
    tgt = build_target(target_x, prm, cfg)
    balancer = LambdaBalancer(cfg.lambda_auto, cfg.lambda_ema, cfg.lambda_cap)
    # the local-global pass calibrates λ in ITS OWN variable space (u, joules) — sharing
    # the global balancer both mis-scales the pass and poisons the global EMA
    lg_balancer = LambdaBalancer(cfg.lambda_auto, cfg.lambda_ema, cfg.lambda_cap)

    dmin = np.asarray(prm.grid_min, np.float32)
    dmax = dmin + prm.dx * np.array([prm.nx, prm.ny, prm.nz], np.float32)
    lo, hi = dmin + 2 * prm.dx, dmax - 2 * prm.dx

    x = src.copy()
    st = {"F": None, "v": None, "C": None}
    Fp = _id(N)
    s, dfc_prev = None, None
    frames, F_frames, hist = [x.copy()], [_id(N)], []
    guards = {"clamped": 0, "nan_x": 0, "nan_state": 0, "F_reset": 0, "F_flip": 0,
              "F_invert_steps": 0}
    # freeze tracks the RAW components (λ-free): the physics track and the render track
    best_phys, best_rend, stale, frozen, n_held = None, None, 0, False, 0

    log(f"[v2] N={N} T={cfg.T} iters={cfg.iters} animations={cfg.animations} "
        f"render={'on(a=%g)' % cfg.lambda_auto if cfg.lambda_auto > 0 else 'OFF'} "
        f"material={'on' if cfg.opt_material else 'off'} assim={cfg.assim} "
        f"w_kin={cfg.w_kin} w_box={cfg.w_box}")

    for a in range(cfg.animations):
        if frozen:
            if cfg.hold_after_converge:
                frames.append(x.copy()); F_frames.append(F_frames[-1].copy())
                hist.append({"animation": a, "held": 1})
                n_held += 1
                continue
            break
        # coarse-to-fine: sharpen the render targets late in the run (thin features)
        if (cfg.c2f_at > 0 and cfg.lambda_auto > 0
                and a == int(cfg.c2f_at * cfg.animations)):
            cfg.render_res = cfg.render_res_hi
            tgt = build_target(target_x, prm, cfg)
            best_rend, stale = None, 0              # rescaled track must not inherit a
            hist.append({"animation": a, "c2f_render_res": cfg.render_res})   # near-full
            log(f"[v2] c2f at anim {a + 1}: render targets rebuilt at {cfg.render_res}px")
            # plateau counter (adversarial finding: freeze could fire one commit later)
        x_start = x.copy()
        fr, F_seq, end, s, whist, stats = optimize_window(
            x_start, prm, cfg, tgt, balancer, F0=st["F"], Fp=Fp, v0=st["v"], C0=st["C"],
            s_init=s, dfc_init=dfc_prev, on_iter=on_iter, log=lambda *_: None)
        if cfg.warm_start:
            dfc_prev = stats.get("dfc")
        if not whist:
            if stats.get("grad_converged"):
                frozen = True                       # zero gradient at the start: at the optimum
                hist.append({"animation": a, "grad_converged": 1})
                log(f"[v2] anim {a + 1}: gradient converged at window start; holding still")
                continue
            log(f"[v2] anim {a + 1}: no accepted step, stopping")
            break

        # ---- FULL state promotion + guard counters (must stay zero, gate G2) ----
        x_new = np.ascontiguousarray(fr[-1], np.float32)
        n_out = int(((x_new < lo) | (x_new > hi)).any(1).sum())
        n_nan = int((~np.isfinite(x_new).all(1)).sum())
        x = np.clip(np.nan_to_num(x_new), lo, hi).astype(np.float32)
        # repair numerical pathologies ONLY (counted); no singular-value projection here
        Fc, n_bad, n_flip, _ = condition_F(end["F"], clamp=False)
        n_ns = int((~np.isfinite(end["v"]).all(1)).sum()
                   + (~np.isfinite(end["C"]).all(axis=(1, 2))).sum())
        v_p = np.nan_to_num(end["v"]).astype(np.float32)
        C_p = np.nan_to_num(end["C"]).astype(np.float32)
        st = {"F": Fc, "v": v_p, "C": C_p}
        # whole-window F health, not just the endpoint (an inversion mid-window that
        # recovers by T would otherwise be invisible)
        dets = np.stack([np.linalg.det(Fs) for Fs in F_seq[1:]])
        n_inv = int((dets <= 0.0).any(0).sum())
        guards["clamped"] += n_out; guards["nan_x"] += n_nan; guards["nan_state"] += n_ns
        guards["F_reset"] += n_bad; guards["F_flip"] += n_flip
        guards["F_invert_steps"] += n_inv

        # ---- LOCAL phase (local-global): band-limited surface GS pass on the render
        # residual, interior pinned to the global solution just promoted. Runs BEFORE
        # assimilation so one ratchet covers global+local. λ is calibrated by its OWN
        # balancer in the local energy's variable space (adversarial blocker). ----
        lg_tele = None
        if cfg.lg_sweeps > 0 and balancer.active:
            out = surface_local_pass(x, Fc, Fp, tgt, cfg, lg_balancer, prm)
            if out is not None:
                x_lg, F_lg, lg_tele = out
                n_lg_nan = int((~np.isfinite(x_lg).all(1)).sum())
                n_lg_out = int(((x_lg < lo) | (x_lg > hi)).any(1).sum())   # COUNT the clip
                n_inv2 = int((np.linalg.det(np.nan_to_num(F_lg)) <= 0).sum())  # pre-repair
                Fc2, n_b2, n_f2, _ = condition_F(F_lg, clamp=False)
                guards["nan_x"] += n_lg_nan; guards["clamped"] += n_lg_out
                guards["F_reset"] += n_b2; guards["F_flip"] += n_f2
                guards["F_invert_steps"] += n_inv2
                x = np.clip(np.nan_to_num(x_lg), lo, hi).astype(np.float32)
                Fc = Fc2
                st["F"] = Fc
                n_out += n_lg_out; n_nan += n_lg_nan
                n_bad += n_b2; n_flip += n_f2; n_inv += n_inv2

        # ---- plastic assimilation of the ELASTIC stretch (§3.5): F_e -> R_e S_e^{1-eta},
        # exact and per-particle — displacement-field estimation mismatched the
        # dFc-inflated F and spiked stress at every commit boundary (measured) ----
        if cfg.assim > 0:
            Fp = assimilate_elastic(Fc, Fp, eta=cfg.assim,
                                    smin=cfg.assim_smin, smax=cfg.assim_smax)

        # archive the PROMOTED states (identical to raw when no guard fired)
        frames.extend(f.copy() for f in fr[1:-1]); frames.append(x.copy())
        F_frames.extend(F_seq[1:-1]); F_frames.append(Fc.copy())

        w = whist[-1]
        # after a local pass the archived state differs from the window's last iterate —
        # the freeze and the logged trace must describe the ARCHIVED state (adversarial
        # finding), so recompute the data terms on the final x
        if lg_tele is not None:
            with torch.no_grad():
                xt = torch.as_tensor(x, device=cfg.device)
                w = dict(w)
                w["d_vol"] = float(d_vol(xt, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx,
                                         tgt.ldims))
                if tgt.sils is not None:
                    w["d_render"] = float(d_render(xt, tgt.sils, tgt.views,
                                                   cfg.render_res, tgt.extent, cfg.sil_k,
                                                   cfg.w_hole, cfg.w_spray))
        rec = {"animation": a, "iters": len(whist), "loss": w["loss"], "d_vol": w["d_vol"],
               "grad_norm": w.get("grad_norm"), "d_pbr": w.get("d_pbr"),
               "kin": w["kin"], "d_render": w["d_render"], "lambda": w["lambda"],
               "dfc_absmax": w["dfc_absmax"], "s_absmax": w["s_absmax"],
               "accepted": stats["accepted"], "rejected": stats["rejected"],
               "v_absmax": float(np.abs(v_p).max()),
               "v_mean": float(np.linalg.norm(v_p, axis=1).mean()),
               "move": float(np.linalg.norm(x - x_start, axis=1).mean()),
               "Jmin": float(np.linalg.det(Fc).min()),
               "Jmin_traj": float(dets.min()),
               "clamped": n_out, "nan_x": n_nan, "nan_state": n_ns,
               "F_reset": n_bad, "F_flip": n_flip, "F_invert_steps": n_inv}
        if lg_tele is not None:
            rec.update(lg_tele)
        hist.append(rec)
        if on_commit is not None:
            on_commit(a, x, Fc, v_p, rec)

        # ---- plateau freeze on RAW components (λ-free; stops post-convergence sloshing) ----
        phys_track = rec["d_vol"] + cfg.w_kin * rec["kin"]
        rend_track = rec["d_render"]
        improved = best_phys is None or phys_track < best_phys - cfg.tol * abs(best_phys)
        if rend_track is not None and best_rend is not None:
            improved = improved or rend_track < best_rend - cfg.tol * abs(best_rend)
        best_phys = phys_track if best_phys is None else min(best_phys, phys_track)
        if rend_track is not None:
            best_rend = rend_track if best_rend is None else min(best_rend, rend_track)
        stale = 0 if improved else stale + 1
        if stale >= cfg.patience:
            frozen = True
            log(f"[v2] converged at anim {a + 1} (phys={phys_track:.4f}); holding still")

        any_guard = n_out or n_nan or n_ns or n_bad or n_flip or n_inv
        if a % max(1, cfg.animations // 10) == 0 or a == cfg.animations - 1 or any_guard:
            log(f"[v2] anim {a + 1}/{cfg.animations}  L={rec['loss']:.4f}  D_vol={rec['d_vol']:.3f}" +
                (f"  D_r={rec['d_render']:.5f}  lam={rec['lambda']:.3g}" if rec["d_render"] is not None else "") +
                f"  kin={rec['kin']:.4f}  |v|max={rec['v_absmax']:.3f}  move={rec['move']:.4f}" +
                f"  Jmin={rec['Jmin_traj']:.3f}  acc/rej={rec['accepted']}/{rec['rejected']}" +
                (f"  GUARD clamp={n_out} nanx={n_nan} nanst={n_ns} Freset={n_bad} "
                 f"Fflip={n_flip} Finv={n_inv}" if any_guard else ""))

    return {"frames": frames, "F_frames": F_frames, "history": hist, "guards": guards,
            "s": s, "Fp": Fp, "n_held": n_held, "converged": frozen}
