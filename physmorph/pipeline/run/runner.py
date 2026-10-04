"""run_pipeline — the window loop of settled transport, on the device.

Per window: optimise the controls (window.optimize_window), promote the FULL committed
state (x, F, v, C) with its guard counts, make an eta-fraction of the elastic stretch
plastic, score the window with the selection merit and accept or reject it
(run.selection). The material bonds' rest lengths are refreshed at every window start for
particles still coupled to the body. The archive keeps every simulated frame; the
deliverable ends at the best window.
"""
from __future__ import annotations

import dataclasses

import torch

from ... import gpu
from ...losses.volumetric import d_vol_density, d_w1, rasterize_mass
from ...mpm.state import MPMParams
from ...mpm.traj import compute_rest_volumes
from ...plasticity import assimilate_elastic
from ...thin import thin_metrics
from ..config import PipelineConfig
from ..render_loss import LambdaBalancer
from ..target import build_target, calibrate_units, rebuild_for_resolution
from ..window import StartState, optimize_window
from ..window.setup import domain_bounds
from ..window.telemetry import write_term_dump
from .selection import Selection, best_window
from .state import FrameStore, fragment_mask, promote

GUARDS = ("clamped", "nan_x", "nan_state", "F_reset", "F_flip", "F_invert_steps")
_STAT_FIELDS = ("g_cos", "g_raw_cos", "g_share", "g_phys_norm", "g_rend_norm", "render_work",
                "render_work_x", "render_work_F", "phys_work", "phys_work_x", "phys_work_F",
                "phys_work_v", "step_norm", "render_cos", "phys_cos", "predicted_decrease",
                "ls_trials", "ls_fail_merit", "ls_fail_state", "ls_fail_state_reason", "ls_probe", "iter_probe",
                "zero_ok", "zero_reason", "warm_ok", "warm_reason", "start_ok", "start_reason", "commit_reason",
                "replay_dx_max", "replay_dx_rms", "replay_dlv", "replay_dlk", "replay_dlr",
                "t_start", "t_grad", "t_ls", "t_commit", "merit_far", "merit_w1_gap",
                "g_transport", "g_surf", "g_spray", "g_near", "n_spray", "n_near", "n_near_active", "ot_scale",
                "active_set", "ctrl_mag", "ctrl_rough", "prof",
                "sup_E", "sup_B", "sup_w_eff", "sup_pen_max", "sup_pen_p99", "sup_pen_med", "sup_grad_ratio",
                "ext_discs", "ext_builds")
_NULL_FIELDS = ("null_reason", "prof", "ls_trials", "ls_fail_merit", "ls_fail_state", "ls_fail_state_reason", "ls_probe",
                "iter_probe", "E_accept", "commit_E_final", "commit_jt", "commit_reason", "replay_rel",
                "zero_ok", "zero_reason", "warm_ok", "warm_reason", "start_ok", "start_reason")


def _host(t):
    return t.detach().cpu().numpy()


def run_pipeline(source_x, target_x, prm: MPMParams, cfg: PipelineConfig, log=print,
                 on_commit=None, on_iter=None, F_stride: int | None = None, thin=None):
    """Morph source -> target. Returns a dict with the archived frames (FrameStore), the
    per-window history, the guard counts and the delivered slice. on_commit(a, x, F, v, rec)
    fires after every judged window and on_iter(it, x, F, tele) after every accepted
    iteration (live viewer hooks, host arrays). thin: a physmorph.thin.ThinSet whose coverage
    every committed window records (measurement only)."""
    gpu.require_cuda()
    cfg = dataclasses.replace(cfg)                      # c2f edits render_res on this copy
    log(f"[v2] settled transport: {cfg.T} controlled + {cfg.T} released steps per commit; "
        "fixed initial render weight per resolution")
    src = gpu.tensor(source_x)
    N = src.shape[0]
    if len(target_x) != N:
        raise ValueError(f"source and target need the same particle count (got {N} vs {len(target_x)})")
    tgt = build_target(target_x, prm, cfg)
    calibrate_units(tgt, src, cfg)
    log(f"[v2] density units: D_vol legacy({cfg.unit_ref_res}^3)/density = {tgt.unit_ratio:.4g} "
        f"(weights), gradient ratio = {tgt.unit_grad_ratio:.4g} (eps/target_norm), "
        f"n_support={tgt.n_support} m_ref={tgt.m_ref:.3g}")
    # the relative cap: 20x the first raw ratio (the same meaning in any loss unit)
    balancer = LambdaBalancer(cfg.lambda_auto, cfg.lambda_ema, None, cap_rel=20.0)
    lo, hi = domain_bounds(prm)
    x = src.clone()
    vol0 = compute_rest_volumes(src, 1.0, prm, cfg.device)
    coh_nbr = gpu.knn(src, cfg.coh_k + 1)[1][:, 1:]    # frozen source-material neighbours
    bond_rest = None
    F = v = C = None
    Fp = torch.eye(3, device=gpu.DEVICE).repeat(N, 1, 1)
    dfc_prev = None
    Fp_pre = None                                   # the last commit's plastic state before its assimilation
    frames = FrameStore(src, F_stride or cfg.T)
    hist, guards = [], {k: 0 for k in GUARDS}
    sel = Selection(cfg)
    shadow = Selection(cfg)            # a record: the same rule read with the dense distance added (the merit until R13)
    frozen = False
    log(f"[v2] N={N} T={cfg.T} iters={cfg.iters} animations={cfg.animations} "
        f"render=on(a={cfg.lambda_auto:g}) x{cfg.render_weight_scale:g} assim={cfg.assim}")
    c2f_pending = cfg.c2f_event and cfg.render_res_hi > cfg.render_res
    for a in range(cfg.animations):
        if not frozen and sel.plateau(a):
            log(f"[v2] delivery merit plateau at anim {a + 1}")
            frozen = True
        if frozen and c2f_pending:
            # coarse-to-fine: the run at the coarse resolution has stopped (plateau, patience or the
            # rejection streak); the render targets are rebuilt at the fine resolution and the run goes
            # on to its own stop there, a new cost epoch with its own render weight and convergence test
            c2f_pending, frozen = False, False
            cfg.render_res = cfg.render_res_hi
            tgt = rebuild_for_resolution(tgt, target_x, prm, cfg)
            for s_ in (sel, shadow):
                s_.new_epoch()
                s_.stale, s_.lam = 0, None
                s_.reject_streak, s_.last_reject_score = 0, None
            hist.append({"animation": a, "c2f_render_res": cfg.render_res})
            log(f"[v2] c2f at anim {a + 1}: render targets rebuilt at {cfg.render_res}px")
        if frozen:
            if cfg.hold_after_converge:
                frames.hold()
            break
        x_start = x.clone()
        rollback = {"F": F, "v": v, "C": C, "Fp": Fp, "dfc": dfc_prev, "lam": balancer.lam,
                    "frames": len(frames), "guards": dict(guards)}
        # bonds: rest lengths refresh only for particles coupled at this window's start; a
        # broken-off particle keeps its last coupled lengths, so the bonds pull it back
        d_now = (x_start[coh_nbr] - x_start[:, None, :]).norm(dim=2)
        frag = fragment_mask(x_start, prm)
        bond_rest = d_now if bond_rest is None else torch.where(frag[:, None], bond_rest, d_now)
        n_frag = int(frag.sum())
        if a % 10 == 0 or n_frag:
            log(f"[v2] anim {a + 1}: fragments {n_frag} particles")
        res = optimize_window(StartState(x=x_start, Fp=Fp, F=F, v=v, C=C), prm, cfg, tgt, balancer,
                              vol0, (coh_nbr, bond_rest, frag.float()), dfc_init=dfc_prev,
                              alpha_scale=sel.anneal, on_iter=on_iter, log=lambda *_: None)
        stats = res.stats
        dfc_prev = stats["dfc"]
        if res.commit is None:
            if stats.get("invalid_selection"):
                dfc_prev = rollback["dfc"]
            if stats.get("grad_converged"):
                frozen = True               # zero gradient at the window start: the optimum
                hist.append({"animation": a, "grad_converged": 1})
                log(f"[v2] anim {a + 1}: gradient converged at window start; holding still")
                continue
            log(f"[v2] anim {a + 1}: no accepted step - null commit (stale {sel.stale + 1})")
            rec = {"animation": a, "null_commit": 1, "no_simulated_time": 1, **{k: stats.get(k) for k in _NULL_FIELDS}}
            if cfg.ls_probe and stats.get("start_ok") == 0 and Fp_pre is not None:
                # diagnostic: is the dead start state made by the last commit's plastic assimilation? The free
                # rollout is re-run from the same state with the assimilation undone (Fp as the committing
                # window had it) and, for reference, as it is
                bonds = (coh_nbr, bond_rest, frag.float())
                rec["dead_free"] = _free_probe(StartState(x=x_start, Fp=Fp, F=F, v=v, C=C), prm, cfg, tgt, vol0, bonds)
                rec["dead_free_noassim"] = _free_probe(StartState(x=x_start, Fp=Fp_pre, F=F, v=v, C=C), prm, cfg,
                                                       tgt, vol0, bonds)
                log(f"[v2] anim {a + 1}: dead start state; free rollout {rec['dead_free']}, "
                    f"without the last assimilation {rec['dead_free_noassim']}")
            hist.append(rec)
            shadow.null()
            if sel.null():
                frozen = True
                log(f"[v2] frozen after {cfg.patience} stale/null commits")
            continue
        commit = res.commit
        x, F, v, C, counts = promote(commit, lo, hi)
        for k in GUARDS:
            guards[k] += counts[k]
        Fp_pre = Fp                                     # the plastic state the committing window ran with
        if cfg.assim > 0:
            Fp = assimilate_elastic(F, Fp, eta=cfg.assim, smin=cfg.assim_smin, smax=cfg.assim_smax,
                                    isochoric=True)
        frames.add_window(commit.x[1:-1], commit.F[1:-1], x, F)
        rec = _record(a, res, x, x_start, v, F, counts, commit, tgt, cfg, prm, thin)
        if cfg.term_dump and stats.get("term_grads") is not None:
            write_term_dump(cfg.term_dump, a, x, stats.pop("term_grads"))
        res.commit = commit = None          # release the window's buffers before the next one
        lam, render = float(res.hist[-1]["lambda"] or 0.0), rec["d_render"] + cfg.w_pbr * rec["d_pbr"]
        sel.check_lambda(rec, lam, render)
        components = {"phys": rec["transport_energy"], "render": rec["d_sil"], "dt": rec["d_dt"]}
        disp = (x - x_start).reshape(-1)
        outer_reject, brake_reject, improved = sel.judge(rec, components, disp)
        alt = _shadow_judge(shadow, rec, components, disp, lam, render, improved)
        if outer_reject:
            # undo every mutation made after the window start (plasticity, lambda); a
            # rejected lineage is not retried: cold restart, no warm start or step memory
            x, F, v, C, Fp = x_start, rollback["F"], rollback["v"], rollback["C"], rollback["Fp"]
            balancer.lam = rollback["lam"]
            dfc_prev = Fp_pre = None
            tgt.settled_step = None
            frames.truncate(rollback["frames"])
            guards = rollback["guards"]
            rec.update({"null_commit": 1, "outer_rejected": 1, "brake_reject": int(brake_reject)})
            hist.append(rec)
            stop = sel.rejected(rec, brake_reject, stats.get("replay_rel", 0.0))
            rec["shadow_stop"] = int(shadow.rejected(alt, brake_reject, stats.get("replay_rel", 0.0)))
            rec["actual_stop"] = int(stop)
            _notify(on_commit, a, x, F, v, rec)
            log(f"[v2] anim {a + 1}: outer merit rejected candidate (gain={_fmt(rec['outer_gain'])}, "
                f"physics gain={_fmt(rec['phys_gain'])}, reversal={rec['reversal_cos']})")
            if stop:
                if sel.reject_streak >= cfg.reject_stop > 0:
                    log(f"[v2] anim {a + 1}: {sel.reject_streak} consecutive rejected candidates -> "
                        "early stop at the best commit")
                frozen = True
            continue
        rec["frame_end"] = len(frames)
        converged = sel.accepted(rec, a, disp, improved)
        rec["shadow_stop"] = int(shadow.accepted(alt, a, disp, bool(rec["shadow_improved"])))
        rec["actual_stop"] = int(converged)
        hist.append(rec)
        _notify(on_commit, a, x, F, v, rec)
        if converged:
            frozen = True
            phys_track = tgt.ot_scale * (rec["transport_energy"] + rec["stab_end"]) + rec["stab"]
            log(f"[v2] converged at anim {a + 1} (phys={phys_track:.4f}); holding still")
        any_guard = any(counts[k] for k in GUARDS)
        if a % max(1, cfg.animations // 10) == 0 or a == cfg.animations - 1 or any_guard:
            log(f"[v2] anim {a + 1}/{cfg.animations}  L={rec['loss']:.4f}  D_vol={rec['d_vol']:.3f}"
                f"  D_r={rec['d_render']:.5f}  lam={rec['lambda']:.3g}  kin={rec['kin']:.4f}"
                f"  |v|max={rec['v_absmax']:.3f}  move={rec['move']:.4f}  Jmin={rec['Jmin_traj']:.3f}"
                f"  acc/rej={rec['accepted']}/{rec['rejected']}"
                + (f"  GUARD {counts}" if any_guard else ""))
    deliver_n, trunc = (best_window(hist, len(frames), cfg.tol, cfg.w_pbr) if cfg.best_truncate
                        else (len(frames), None))
    if trunc is not None:
        log(f"[v2] deliverable ends at best commit anim {trunc['best_animation']} "
            f"(deliver {deliver_n} of {len(frames)} frames; all frames archived)")
    return {"truncation": trunc, "deliver_n": deliver_n, "frames": frames, "history": hist,
            "guards": guards, "Fp": _host(Fp), "n_held": 0, "converged": frozen,
            "balancer": {"cap": balancer.cap, "cap_rel": balancer.cap_rel, "alpha_lam": balancer.alpha_lam}}


def _free_probe(start: StartState, prm, cfg, tgt, vol0, bonds) -> dict:
    """Diagnostic: the zero-control rollout from a start state (rollout.free_rollout_probe on a throwaway window)."""
    from ..window.objective import Objective
    from ..window.rollout import free_rollout_probe
    from ..window.setup import Window
    win = Window(start, prm, cfg, tgt, vol0, bonds)
    try:
        return free_rollout_probe(win, Objective(win))
    finally:
        del win


def _record(a, res, x, x_start, v, F, counts, commit, tgt, cfg, prm, thin=None) -> dict:
    """The window's history record, measured on the PROMOTED state."""
    w, stats = res.hist[-1], res.stats
    with torch.no_grad():
        d_vol = float(d_vol_density(x, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims, tgt.m_ref,
                                    tgt.n_support))
        d_dt = float(d_w1(x, tgt.m, tgt.dt3, tgt.dtgmin, tgt.dtdx, tgt.dtdims))
        energy = float(tgt.grid_ot.state_energy(x, tgt.m))
        detF = torch.linalg.det(F)
        jmin = float(detF.min())
        diag = {}
        if cfg.work_telemetry:                      # diagnostic records, not read by the run
            ot_div = float(tgt.grid_ot(rasterize_mass(x, tgt.m, tgt.lgmin, tgt.ldx, tgt.ldims)))
            sv = torch.linalg.svdvals(F)
            aniso = sv[:, 0] / sv[:, -1].clamp_min(1e-9)
            qs = torch.tensor([.01, .5, .9, .99], device=F.device, dtype=detF.dtype)
            jq, aq = torch.quantile(detF, qs), torch.quantile(aniso.to(detF.dtype), qs)
            diag = {"ot_div": ot_div, "J_p01": float(jq[0]), "J_p50": float(jq[1]), "J_p99": float(jq[3]),
                    "aniso_p50": float(aq[1]), "aniso_p90": float(aq[2]), "aniso_p99": float(aq[3])}
    rec = {"animation": a, "iters": len(res.hist), "loss": w["loss"], "d_vol": d_vol,
           "grad_norm": w["grad_norm"], "d_pbr": w["d_pbr"], "d_dt": d_dt, "d_sil": w["d_sil"],
           **{k: stats.get(k) for k in _STAT_FIELDS},
           "kin": w["kin"], "kin_run": w["kin_run"], "kin_var": w["kin_var"], "stab": w["stab"], "stab_end": w["stab_end"],
           "alpha_last": w["alpha"],
           "d_render": w["d_render"], "lambda": w["lambda"], "lambda_capped": stats.get("lambda_capped"),
           "u_gate": stats.get("u_gate"), "dfc_absmax": w["dfc_absmax"],
           "accepted": stats["accepted"], "rejected": stats["rejected"],
           "v_absmax": float(v.abs().max()), "v_mean": float(v.norm(dim=1).mean()),
           "com": x.mean(0).tolist(), "v_com": v.mean(0).tolist(),
           "move": float((x - x_start).norm(dim=1).mean()), "Jmin": jmin,
           "Jmin_traj": commit.jmin_traj, **counts,
           "selection_merit": stats["selection_merit"], "transport_energy": energy, **diag}
    if thin is not None and cfg.work_telemetry:
        m = thin_metrics(x, thin)
        rec.update(thin_uncovered=m.get("thin_uncovered"), thin_uncovered_world=m.get("thin_uncovered_world"))
    return rec


def _notify(on_commit, a, x, F, v, rec):
    if on_commit is not None:
        N = x.shape[0]
        on_commit(a, _host(x), _host(torch.eye(3, device=x.device).repeat(N, 1, 1) if F is None else F),
                  _host(torch.zeros_like(x) if v is None else v), rec)


def _fmt(v):
    return "n/a" if v is None else format(v, ".3g")


def _shadow_judge(shadow, rec, components, disp, lam, render, improved) -> dict:
    """A record, no effect on the run: the selection rule judged with the merit that also carries the dense
    body-to-target distance (the merit until R13), on the same trajectory. Writes the shadow's verdicts beside the
    actual ones into rec and returns the shadow's copy of the record."""
    alt = dict(rec)
    alt["selection_merit"] = rec["selection_merit"] + (rec.get("merit_w1_gap") or 0.0)
    shadow.check_lambda(alt, lam, render)
    s_reject, s_brake, s_improved = shadow.judge(alt, components, disp)
    rec.update({"shadow_merit": alt["selection_merit"], "shadow_reject": int(s_reject), "shadow_brake": int(s_brake),
                "shadow_improved": int(s_improved), "judge_improved": int(improved)})
    return alt
