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

from ..losses.volumetric import (coverage_shortfall, d_vol, d_w1, target_dt_grid,
                                 target_mass_grid)
from ..mpm.conditioning import condition_F
from ..mpm.state import MPMParams
from ..mpm.traj import compute_rest_volumes
from ..plasticity import assimilate_elastic
from .config import PipelineConfig
from .optimizer import TargetPack, optimize_window
from .render_loss import (LambdaBalancer, d_render, make_views, shade_targets,
                          target_silhouettes)
from .surface_local import surface_local_pass


def _id(N):
    return np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))


def _surface_weights(x: np.ndarray, k: int, fraction: float, floor: float) -> np.ndarray:
    """Persistent soft surface score from one-sided local neighbour directions.

    A volume-interior particle sees approximately cancelling unit directions; a boundary
    particle does not.  The score is computed once in source/material coordinates so the
    active set cannot flicker between optimisation windows.
    """
    from scipy.spatial import cKDTree
    n = len(x)
    kk = min(n, max(2, 3 * int(k) + 1))  # tolerate duplicate voxel samples
    _, idx = cKDTree(x).query(x, k=kk, workers=-1)
    d = x[idx[:, 1:]] - x[:, None, :]
    dn = np.linalg.norm(d, axis=2, keepdims=True)
    valid = dn[..., 0] > 1e-8
    unit = d / np.maximum(dn, 1e-8)
    score = np.linalg.norm((unit * valid[..., None]).sum(1) /
                           np.maximum(valid.sum(1, keepdims=True), 1), axis=1)
    threshold = float(np.quantile(score, 1.0 - float(fraction)))
    width = max(float(np.std(score)) * 0.15, 1e-4)
    soft = 1.0 / (1.0 + np.exp(np.clip(-(score - threshold) / width, -40.0, 40.0)))
    return np.ascontiguousarray(floor + (1.0 - floor) * soft, np.float32)


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
    sils = shade = dt3 = None
    if cfg.lambda_auto > 0:
        sils = target_silhouettes(tgt_t, views, cfg.render_res, extent, cfg.sil_k)
        if cfg.w_pbr > 0:
            shade = shade_targets(tgt_t, views, cfg.render_res, extent,
                                  lgmin, ldx, ldims, cfg.sil_k, cfg.pbr_ambient)
    dtgmin, dtdx, dtdims, tmass3 = None, 0.0, (), None
    if cfg.w_dt > 0 or cfg.w_fill > 0 or cfg.w_grow > 0:
        # fine target-fitted grid shared by both one-signed W1 terms — independent of
        # the render channel (Opus finding 2: the loss grid's ~1-unit cells made a dead
        # radius covering the whole fringe band). Cube spans 1.5x extent: everything the
        # box leash allows stays on a live DT slope; the EDT is build-time.
        dtdims = (cfg.dt_res,) * 3
        dtdx = 3.0 * extent / cfg.dt_res
        dtgmin = torch.tensor([-1.5 * extent] * 3, device=dev)
        dt_mass = target_mass_grid(tgt_t, m, dtgmin, dtdx, dtdims)
        if cfg.w_dt > 0:
            dt3 = target_dt_grid(dt_mass, dtdx, dtdims,
                                 clamp=cfg.dt_clamp_frac * extent)
        if cfg.w_fill > 0 or cfg.w_grow > 0:
            tmass3 = dt_mass
    gauss = None
    if cfg.use_gauss_loss and cfg.lambda_auto > 0:
        from ..render.covariance import sigma0_from_nn
        from .gauss_loss import GaussViews
        target_mask = None
        gaussian_points = target_x
        if cfg.render_surface_only:
            tw = _surface_weights(target_x, cfg.surface_grad_k,
                                  cfg.surface_grad_frac, cfg.surface_grad_floor)
            target_mask = torch.as_tensor(tw > 0.5, device=dev)
            gaussian_points = target_x[tw > 0.5]
        gauss = GaussViews(views, extent,
                           sigma0_from_nn(gaussian_points, cfg.gauss_sigma_scale),
                           cfg.gauss_res, dev, child_count=cfg.gauss_children,
                           child_sigma_scale=cfg.gauss_child_sigma_scale,
                           child_offset_scale=cfg.gauss_child_offset_scale,
                           child_k=cfg.gauss_child_k)
        gauss.bake_targets(tgt_t, mask=target_mask)
    pts, nn_sp = None, 0.0
    if cfg.w_nn > 0:
        from scipy.spatial import cKDTree
        nn_sp = float(np.median(cKDTree(target_x).query(target_x, k=2,
                                                        workers=-1)[0][:, 1]))
        pts = tgt_t
    return TargetPack(grid=grid, lgmin=lgmin, ldx=ldx, ldims=ldims, m=m,
                      views=views, sils=sils, extent=extent, shade=shade,
                      dt3=dt3, dtgmin=dtgmin, dtdx=dtdx, dtdims=dtdims, tmass3=tmass3,
                      pts=pts, nn_spacing=nn_sp, gauss=gauss)


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
    if cfg.lg_sweeps > 0 and (cfg.w_dt > 0 or cfg.w_fill > 0):
        # Codex finding 7 (+stack-review f11): the local pass's exact-quadratic energy
        # excludes BOTH one-signed W1 terms, so it can undo an accepted W1/fill step and
        # assimilation then ratchets the regression
        raise ValueError("lg_sweeps>0 with w_dt>0 or w_fill>0 is unsupported (local "
                         "energy has no W1/fill term; a non-quadratic term breaks its "
                         "exact line search)")
    tgt = build_target(target_x, prm, cfg)
    balancer = LambdaBalancer(cfg.lambda_auto, cfg.lambda_ema, cfg.lambda_cap)
    # the local-global pass calibrates λ in ITS OWN variable space (u, joules) — sharing
    # the global balancer both mis-scales the pass and poisons the global EMA
    lg_balancer = LambdaBalancer(cfg.lambda_auto, cfg.lambda_ema, cfg.lambda_cap)
    # fill v3: its own norm balancer (alpha = w_fill) — dominance structurally bounded
    fill_balancer = (LambdaBalancer(cfg.w_fill, cfg.lambda_ema, cap=100.0)
                     if cfg.w_fill > 0 else None)

    dmin = np.asarray(prm.grid_min, np.float32)
    dmax = dmin + prm.dx * np.array([prm.nx, prm.ny, prm.nz], np.float32)
    lo, hi = dmin + 2 * prm.dx, dmax - 2 * prm.dx

    x = src.copy()
    vol0 = (compute_rest_volumes(src, 1.0, prm, cfg.device)
            if cfg.persistent_rest_volume else None)
    surface_w = (_surface_weights(src, cfg.surface_grad_k, cfg.surface_grad_frac,
                                  cfg.surface_grad_floor)
                 if cfg.surface_grad_frac > 0 else None)
    if cfg.render_surface_only:
        if surface_w is None:
            raise ValueError("render_surface_only requires surface_grad_frac > 0")
        surface_w = np.ascontiguousarray(surface_w > 0.5, np.float32)
    if tgt.gauss is not None and cfg.gauss_children > 1:
        if not cfg.render_surface_only:
            raise ValueError("gauss_children>1 requires render_surface_only material parents")
        tgt.gauss.configure_source(src, surface_w > 0.5)
    st = {"F": None, "v": None, "C": None}
    Fp = _id(N)
    s, dfc_prev = None, None
    frames, F_frames, hist = [x.copy()], [_id(N)], []
    guards = {"clamped": 0, "nan_x": 0, "nan_state": 0, "F_reset": 0, "F_flip": 0,
              "F_invert_steps": 0}
    # freeze tracks the RAW components (λ-free): physics, render, and W1 tracks. The W1
    # track is SEPARATE (Opus finding 4: folded into phys_track it sat at the tolerance
    # noise floor, so fringe-only progress could still stale out)
    best_phys, best_rend, best_dt, best_fill = None, None, None, None
    stale, frozen, n_held = 0, False, 0
    anneal = 1.0                     # plateau-scheduled step scale (zigzag forensic)
    mom_prev = None                  # cross-window Adam moments (mom_carry)
    outer_scales = outer_prev = prev_disp = None
    # Once the trajectory first reaches the small-motion regime, keep the outer
    # trust gate active.  Without this latch, one accepted large-motion candidate
    # disables the gate again and the optimizer can re-enter a long limit cycle.
    outer_gate_latched = False

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
            outer_scales = outer_prev = prev_disp = None
            outer_gate_latched = False              # render track rescaled: re-earn the latch
            hist.append({"animation": a, "c2f_render_res": cfg.render_res})   # near-full
            log(f"[v2] c2f at anim {a + 1}: render targets rebuilt at {cfg.render_res}px")
            # plateau counter (adversarial finding: freeze could fire one commit later)
        x_start = x.copy()
        rollback = {
            "st": {k: (None if q is None else q.copy()) for k, q in st.items()},
            "Fp": Fp.copy(), "s": None if s is None else s.copy(),
            "dfc": dfc_prev, "mom": mom_prev, "lam": balancer.lam,
            "frames": len(frames), "F_frames": len(F_frames), "guards": dict(guards),
        }
        fr, F_seq, end, s, whist, stats = optimize_window(
            x_start, prm, cfg, tgt, balancer, F0=st["F"], Fp=Fp, v0=st["v"], C0=st["C"],
            s_init=s, dfc_init=dfc_prev, on_iter=on_iter, log=lambda *_: None,
            fill_bal=fill_balancer, alpha_scale=anneal, mom_init=mom_prev, vol0=vol0,
            surface_w=surface_w)
        if cfg.warm_start:
            dfc_prev = stats.get("dfc")
        if not whist:
            if stats.get("grad_converged"):
                frozen = True                       # zero gradient at the start: at the optimum
                hist.append({"animation": a, "grad_converged": 1})
                log(f"[v2] anim {a + 1}: gradient converged at window start; holding still")
                continue
            # line-search exhaustion is NOT convergence (Codex stack-review f6: a hard
            # stop here bypassed patience — hero7_base truncated at anim 106). Null
            # commit: hold the state, let the patience counter decide the freeze.
            log(f"[v2] anim {a + 1}: no accepted step — null commit (stale {stale + 1})")
            frames.append(x.copy()); F_frames.append(F_frames[-1].copy())
            hist.append({"animation": a, "null_commit": 1})
            stale += 1
            mom_prev = None
            if cfg.anneal_stale > 0:
                # Retrying the identical state/moments at the identical alpha reproduces
                # the same line-search exhaustion forever.  Null commits must participate
                # in the same plateau schedule as rejected outer candidates.
                anneal = max(0.05, anneal * cfg.anneal_stale)
            if stale >= cfg.patience:
                frozen = True
                log(f"[v2] frozen after {cfg.patience} stale/null commits")
            continue

        if cfg.mom_carry > 0:            # only a committed window donates its moments
            mom_prev = stats.get("mom_out")
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
            if cfg.w_grow > 0 and tgt.tmass3 is not None:
                from ..losses.volumetric import growth_demand
                from ..plasticity.assimilation import assimilate_growth
                dem = growth_demand(torch.as_tensor(x, device=cfg.device), tgt.m,
                                    tgt.tmass3, tgt.dtgmin, tgt.dtdx, tgt.dtdims,
                                    cfg.fill_sigma)
                Fp = assimilate_growth(Fc, Fp, eta=cfg.assim,
                                       smin=cfg.assim_smin, smax=cfg.assim_smax,
                                       isochoric=cfg.assim_iso,
                                       grow=1.0 + cfg.w_grow * dem,
                                       grow_band=cfg.grow_band)
            else:
                Fp = assimilate_elastic(Fc, Fp, eta=cfg.assim,
                                        smin=cfg.assim_smin, smax=cfg.assim_smax,
                                        isochoric=cfg.assim_iso)

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
        d_dt = None
        if tgt.dt3 is not None:      # W1 term on the ARCHIVED state — the freeze track
            with torch.no_grad():    # must see it (Codex finding 6). UNGATED on purpose
                # (stack-review f13: a per-window gate makes the track fall when the
                # GATE dies rather than when particles move — the §7.4 pathology in
                # the convergence signal): a gate-independent geometric statistic.
                xt = torch.as_tensor(x, device=cfg.device)
                d_dt = float(d_w1(xt, tgt.m, tgt.dt3,
                                  tgt.dtgmin, tgt.dtdx, tgt.dtdims))
        d_fill = None
        if cfg.w_fill > 0 and tgt.tmass3 is not None:   # fill telemetry (f9/F9: the
            with torch.no_grad():                        # term was fully unobservable)
                xt = torch.as_tensor(x, device=cfg.device)
                d_fill = coverage_shortfall(xt, tgt.m, tgt.tmass3, tgt.dtgmin,
                                            tgt.dtdx, tgt.dtdims, cfg.fill_sigma)
        rec = {"animation": a, "iters": len(whist), "loss": w["loss"], "d_vol": w["d_vol"],
               "grad_norm": w.get("grad_norm"), "d_pbr": w.get("d_pbr"), "d_dt": d_dt,
               "d_fill": d_fill, "g_cos": stats.get("g_cos"),
               "g_raw_cos": stats.get("g_raw_cos"), "g_share": stats.get("g_share"),
               "g_phys_norm": stats.get("g_phys_norm"), "g_rend_norm": stats.get("g_rend_norm"),
               "render_work": stats.get("render_work"),
               "render_work_x": stats.get("render_work_x"),
               "render_work_F": stats.get("render_work_F"),
               "phys_work": stats.get("phys_work"),
               "phys_work_x": stats.get("phys_work_x"),
               "phys_work_F": stats.get("phys_work_F"),
               "phys_work_v": stats.get("phys_work_v"),
               "step_norm": stats.get("step_norm"),
               "predicted_decrease": stats.get("predicted_decrease"),
               "fill_lam": stats.get("fill_lam"),
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
        if tgt.gauss is not None:
            from .gauss_loss import gaussian_shape_diagnostics
            rec.update(gaussian_shape_diagnostics(
                torch.as_tensor(Fc, device=cfg.device), tgt.gauss.primitive_sigma,
                reference_spacing=tgt.nn_spacing if tgt.nn_spacing > 0 else None))
        if lg_tele is not None:
            rec.update(lg_tele)

        # Fixed-scale outer trust gate.  The inner objective contains an adaptive
        # render lambda, so it cannot safely decide whether a whole physical state
        # should be committed across windows.  Normalize each raw channel once per
        # target resolution and require monotone progress in that fixed merit.
        phys_track = rec["d_vol"] + cfg.w_kin * rec["kin"]
        components = {"phys": phys_track}
        if rec["d_render"] is not None:
            components["render"] = rec["d_render"]
        if d_dt is not None:
            components["dt"] = d_dt
        if d_fill is not None:
            components["fill"] = d_fill
        disp = (x - x_start).reshape(-1)
        reversal_cos = None
        if prev_disp is not None:
            reversal_cos = float(np.dot(disp, prev_disp) /
                                 max(np.linalg.norm(disp) * np.linalg.norm(prev_disp), 1e-12))
        rec["reversal_cos"] = reversal_cos

        # λ-free plateau tracks, evaluated BEFORE the outer gate: "no track improved"
        # is this pipeline's validated definition of near-stationarity (driver #4 —
        # at small move the residual motion is honest descent UNTIL these tracks
        # stall). The previous latch trigger (normalized merit <= outer_gate_merit_max
        # at small move) fired at anim ~20 of 300 — pace + a large w_kin make moves
        # small long before the descent is done — and then rejected every window to
        # a fake "converged" at anim 27 (final_hires20k_child4_latched forensic).
        rend_track = rec["d_render"]
        improved = best_phys is None or phys_track < best_phys - cfg.tol * abs(best_phys)
        if rend_track is not None and best_rend is not None:
            improved = improved or rend_track < best_rend - cfg.tol * abs(best_rend)
        if d_dt is not None and best_dt is not None:     # own track + tolerance (Opus f4)
            improved = improved or d_dt < best_dt - cfg.tol * abs(best_dt)
        if d_fill is not None and best_fill is not None:  # fill track (stack-review F9)
            improved = improved or d_fill < best_fill - cfg.tol * abs(best_fill)

        outer_reject = False
        outer_gain = None
        if cfg.outer_merit:
            if outer_scales is None:
                outer_scales = {k: max(abs(v), 1e-8) for k, v in components.items()}
            score = float(sum(v / outer_scales[k] for k, v in components.items()))
            if outer_prev is not None:
                outer_gain = (outer_prev - score) / max(abs(outer_prev), 1e-8)
                near_stationary = (not improved
                                   and rec["move"] <= cfg.outer_gate_move_frac * tgt.extent)
                outer_gate_latched = outer_gate_latched or near_stationary
                outer_reject = outer_gate_latched and outer_gain < cfg.outer_merit_tol
                if (outer_gate_latched and reversal_cos is not None
                        and reversal_cos < cfg.outer_reversal_cos
                        and outer_gain < cfg.outer_reversal_gain):
                    outer_reject = True
            rec.update({"outer_merit": score, "outer_gain": outer_gain,
                        "reversal_cos": reversal_cos,
                        "outer_gate_latched": int(outer_gate_latched),
                        "outer_accepted": 0 if outer_reject else 1})
            if outer_reject:
                # Undo every mutation made after the window start, including plastic
                # assimilation and optimizer/balancer state.  Rejected trial frames
                # never enter the deliverable trajectory.
                x = x_start
                st = rollback["st"]
                Fp, s = rollback["Fp"], rollback["s"]
                dfc_prev, mom_prev, balancer.lam = rollback["dfc"], rollback["mom"], rollback["lam"]
                del frames[rollback["frames"]:]
                del F_frames[rollback["F_frames"]:]
                guards = rollback["guards"]
                rec.update({"null_commit": 1, "outer_rejected": 1})
                hist.append(rec)
                stale += 1
                if cfg.anneal_stale > 0:
                    anneal = max(0.05, anneal * cfg.anneal_stale)
                if on_commit is not None:
                    F_hold = _id(N) if st["F"] is None else st["F"]
                    v_hold = np.zeros_like(x) if st["v"] is None else st["v"]
                    on_commit(a, x, F_hold, v_hold, rec)
                log(f"[v2] anim {a + 1}: outer merit rejected candidate "
                    f"(gain={outer_gain:.3g}, reversal={reversal_cos})")
                if stale >= cfg.patience:
                    frozen = True
                continue
            outer_prev = score
            prev_disp = disp.copy()
        else:
            prev_disp = disp.copy()
        hist.append(rec)
        if on_commit is not None:
            on_commit(a, x, Fc, v_p, rec)

        # ---- plateau freeze on RAW components (λ-free; stops post-convergence sloshing).
        # `improved` was computed above, against the pre-commit bests; the bests only
        # absorb ACCEPTED commits (a gate-rejected candidate must not raise the bar). ----
        best_phys = phys_track if best_phys is None else min(best_phys, phys_track)
        if rend_track is not None:
            best_rend = rend_track if best_rend is None else min(best_rend, rend_track)
        if d_dt is not None:
            best_dt = d_dt if best_dt is None else min(best_dt, d_dt)
        if d_fill is not None:
            best_fill = d_fill if best_fill is None else min(best_fill, d_fill)
        stale = 0 if improved else stale + 1
        if cfg.anneal_stale > 0:     # optimizer-side zigzag damping (docs/oscillation.md)
            anneal = (min(1.0, anneal * 1.15) if improved
                      else max(0.05, anneal * cfg.anneal_stale))
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
            "s": s, "Fp": Fp, "n_held": n_held, "converged": frozen,
            "render_mask": ((surface_w > 0.5) if cfg.render_surface_only else None)}
