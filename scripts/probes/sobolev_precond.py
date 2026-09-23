"""Probe — render-channel work share under (1) Sobolev preconditioning of the render
covector BEFORE the MPM adjoint pullback and (2) weaker F-smoothing, measured in REAL
optimisation windows (docs/root_analysis.md §2; results: docs/probes/sobolev_precond.md).

Protocol (pre-registered in the task):
  * flagship config as scripts/dress_bench.py minus the gauss channel
    (lambda_auto=0.5, w_kin=5, w_dt=0.2, w_nn=0.2, w_jvol=50, assim_iso=True,
    work_telemetry=True, surface mask as dress_bench), N=20000, T=20, iters=8;
  * per condition: 6 consecutive commits from the same source/seed, each commit = one
    optimize_window call + runner-style promotion (grid clip, condition_F(clamp=False),
    isochoric assimilate_elastic) so the state evolves as in production;
  * recorded per commit: the returned stats' work telemetry (render_work, phys_work, the
    x/F/v splits), g_cos / g_raw_cos / g_share / norms, accepted/rejected, and d_vol /
    d_render evaluated on the PROMOTED x, plus every guard counter the runner keeps.

Two share definitions are reported. `share_last` = render_work/(render_work+phys_work)
from `stats` (the LAST accepted iteration — exactly what runner.py logs as the headline
P-render metric). `share_sum` sums both works over every accepted iteration of the window
(from `hist`) before taking the ratio — the window-integrated version.

F-smoothing note (physmorph/mpm/kernels.py k_update, eq (9)):
    F_out = (1 - s) * F_new + s * F_old
so s is the retention weight on the OLD F: s=1.0 FREEZES F (maximal smoothing, the
control still reaches the stress through F+dFc), s=0.0 is "no smoothing". The task's
literal sweep {0.955, 0.98, 1.0} therefore INCREASES the low-pass; the direction that
reduces it is s -> 0. Both are run.

Nothing here edits production code; results go to a /tmp-side directory (never /data).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from physmorph.losses.volumetric import d_vol  # noqa: E402
from physmorph.mpm import MPMParams  # noqa: E402
from physmorph.mpm.conditioning import condition_F  # noqa: E402
from physmorph.mpm.traj import compute_rest_volumes  # noqa: E402
from physmorph.pipeline import PipelineConfig  # noqa: E402
from physmorph.pipeline.grid_smooth import smooth_particle_field  # noqa: E402
from physmorph.pipeline.optimizer import optimize_window  # noqa: E402
from physmorph.pipeline.render_loss import LambdaBalancer, d_render  # noqa: E402
from physmorph.pipeline.runner import _surface_weights, build_target  # noqa: E402
from physmorph.plasticity import assimilate_elastic  # noqa: E402
from physmorph.sampling import load_normalized as load  # noqa: E402

# condition -> (render_gs_iters, render_gs_kappa, MPMParams.smoothing)
CONDITIONS = {
    "base":     dict(gs=0,  kappa=4.0,  s=0.955),
    "base_rep": dict(gs=0,  kappa=4.0,  s=0.955),   # repeat: run-to-run noise (CUDA atomics)
    # (1) Sobolev / screened-diffusion preconditioning of the render covector.
    # grid_smooth solves (I + kappa*(I-avg6)) u = g_hat on the 64^3 LOSS grid (cell 0.5 wu,
    # = MPM dx) with `gs` red-black GS sweeps: screening length sqrt(kappa/6) cells
    # (k1 0.41, k4 0.82, k16 1.63) but each sweep only propagates ~1 cell, so the reach is
    # ~min(sqrt(kappa/6), gs) cells. gs20_k4 is pipeline_run.py's --render_gs_iters default.
    "gs2_k1":   dict(gs=2,  kappa=1.0,  s=0.955),
    "gs2_k4":   dict(gs=2,  kappa=4.0,  s=0.955),
    "gs2_k16":  dict(gs=2,  kappa=16.0, s=0.955),
    "gs6_k4":   dict(gs=6,  kappa=4.0,  s=0.955),
    "gs6_k16":  dict(gs=6,  kappa=16.0, s=0.955),
    "gs20_k4":  dict(gs=20, kappa=4.0,  s=0.955),
    # (2) F-smoothing s (retention of OLD F per step; see module docstring).
    "s0.98":    dict(gs=0,  kappa=4.0,  s=0.98),    # requested: MORE smoothing
    "s1.0":     dict(gs=0,  kappa=4.0,  s=1.0),     # requested: F frozen (not 'no smoothing')
    "s0.8":     dict(gs=0,  kappa=4.0,  s=0.8),     # less smoothing (tau 5 steps)
    "s0.5":     dict(gs=0,  kappa=4.0,  s=0.5),     # less smoothing (tau 2 steps)
    "s0.0":     dict(gs=0,  kappa=4.0,  s=0.0),     # NO smoothing: F integrates fully
}

STAT_KEYS = ("render_work", "render_work_x", "render_work_F",
             "phys_work", "phys_work_x", "phys_work_F", "phys_work_v",
             "g_cos", "g_raw_cos", "g_share", "g_rend_norm", "g_phys_norm",
             "accepted", "rejected", "grad_converged", "pace_bound",
             "step_norm", "predicted_decrease", "L_start")


def make_cfg(gs: int, kappa: float) -> PipelineConfig:
    """scripts/dress_bench.py flagship config, gauss channel OFF."""
    cfg = PipelineConfig(T=20, iters=8, animations=8, loss_res=64)
    cfg.lambda_auto, cfg.w_kin = 0.5, 5.0
    cfg.w_dt, cfg.w_nn, cfg.w_jvol = 0.2, 0.2, 50.0
    cfg.render_surface_only, cfg.surface_grad_frac = True, 0.50
    cfg.assim_iso = True
    cfg.work_telemetry = True
    cfg.render_gs_iters, cfg.render_gs_kappa = int(gs), float(kappa)
    return cfg


def eval_losses(x, tgt, cfg):
    """d_vol / d_render of a promoted (numpy) state — the runner's tracked data terms."""
    with torch.no_grad():
        xt = torch.as_tensor(np.ascontiguousarray(x, np.float32), device=cfg.device)
        dv = float(d_vol(xt, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims))
        dr = float(d_render(xt, tgt.sils, tgt.views, cfg.render_res, tgt.extent,
                            cfg.sil_k, cfg.w_hole, cfg.w_spray))
    return dv, dr


def covector_stats(x, sw, tgt, cfg):
    """Raw render covector at the window-start state (masked as the optimizer masks it)
    and, when the condition preconditions, its cosine against the Sobolev-smoothed field:
    cos ~ 1 = the preconditioner barely changes the direction, cos << 1 = it does."""
    xt = torch.as_tensor(np.ascontiguousarray(x, np.float32),
                         device=cfg.device).requires_grad_(True)
    lr = d_render(xt, tgt.sils, tgt.views, cfg.render_res, tgt.extent,
                  cfg.sil_k, cfg.w_hole, cfg.w_spray)
    gx = torch.autograd.grad(lr, xt)[0].detach()
    gx = gx * torch.as_tensor(sw, device=cfg.device).view(-1, 1)
    out = {"gx_norm": float(gx.norm()),
           "gx_nonzero_frac": float((gx.norm(dim=1) > 0).float().mean()),
           "precond_cos": None, "gxs_norm": None}
    if cfg.render_gs_iters > 0:
        with torch.no_grad():
            gxs = smooth_particle_field(xt.detach(), gx, tgt.lgmin, tgt.ldx, tgt.ldims,
                                        cfg.render_gs_iters, cfg.render_gs_kappa)
        out["precond_cos"] = float((gx * gxs).sum() / (gx.norm() * gxs.norm()).clamp_min(1e-30))
        out["gxs_norm"] = float(gxs.norm())
    return out


def gpu_mem_mib() -> float | None:
    """This process's device footprint (torch + warp) from nvidia-smi; None if unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=20).stdout
        for line in out.strip().splitlines():
            pid, mem = [t.strip() for t in line.split(",")[:2]]
            if int(pid) == os.getpid():
                return float(mem)
    except Exception:
        pass
    return None


def _share(rw, pw):
    if rw is None or pw is None:
        return None
    den = rw + pw
    return rw / den if abs(den) > 1e-30 else None


def _ratio(rw, pw):
    if rw is None or pw is None or abs(pw) < 1e-30:
        return None
    return rw / pw


def run_condition(name, spec, src, tgt, sw, vol0, n_commits, log, tag=""):
    cfg = make_cfg(spec["gs"], spec["kappa"])
    prm = MPMParams(smoothing=float(spec["s"]))
    balancer = LambdaBalancer(cfg.lambda_auto, cfg.lambda_ema, cfg.lambda_cap)
    N = len(src)
    dmin = np.asarray(prm.grid_min, np.float32)
    dmax = dmin + prm.dx * np.array([prm.nx, prm.ny, prm.nz], np.float32)
    lo, hi = dmin + 2 * prm.dx, dmax - 2 * prm.dx        # runner's promotion clip box

    x, st = src.copy(), {"F": None, "v": None, "C": None}
    Fp = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
    dv0, dr0 = eval_losses(x, tgt, cfg)
    commits = []
    t_cond = time.perf_counter()
    for c in range(n_commits):
        x_prev = x
        cov = covector_stats(x, sw, tgt, cfg)
        t0 = time.perf_counter()
        fr, F_seq, end, _s, whist, stats = optimize_window(
            x, prm, cfg, tgt, balancer, F0=st["F"], Fp=Fp, v0=st["v"], C0=st["C"],
            log=lambda *_: None, surface_w=sw, vol0=vol0)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rec = {"commit": c, "time_s": dt, "null_commit": not whist,
               "lambda": balancer.lam, "n_hist": len(whist)}
        rec.update(cov)
        for k in STAT_KEYS:
            rec[k] = stats.get(k)
        # window had an accepted step (telemetry exists) but the commit rollout's
        # trajectory/replay check discarded it (optimizer.py "replay_bad"/jt_final)
        rec["replay_discard"] = bool((not whist) and stats.get("render_work") is not None
                                     and not stats.get("grad_converged"))
        rec["share_last"] = _share(rec["render_work"], rec["phys_work"])
        rec["ratio_last"] = _ratio(rec["render_work"], rec["phys_work"])
        lam = rec["lambda"] or 0.0
        rec["wshare_last"] = (_share(lam * rec["render_work"], rec["phys_work"])
                              if rec["render_work"] is not None else None)
        rw = [h["render_work"] for h in whist if h.get("render_work") is not None]
        pw = [h["phys_work"] for h in whist if h.get("phys_work") is not None]
        rec["render_work_sum"] = float(sum(rw)) if rw else None
        rec["phys_work_sum"] = float(sum(pw)) if pw else None
        rec["share_sum"] = _share(rec["render_work_sum"], rec["phys_work_sum"])
        rec["ratio_sum"] = _ratio(rec["render_work_sum"], rec["phys_work_sum"])
        rec["wshare_sum"] = (_share(lam * rec["render_work_sum"], rec["phys_work_sum"])
                             if rw else None)
        rec["render_work_iters"] = rw
        rec["phys_work_iters"] = pw
        rec["loss_iters"] = [h["loss"] for h in whist]
        rec["d_vol_iters"] = [h["d_vol"] for h in whist]
        rec["d_render_iters"] = [h["d_render"] for h in whist]
        rec["kin_iters"] = [h["kin"] for h in whist]
        rec["d_vol_last_iter"] = whist[-1]["d_vol"] if whist else None
        rec["d_render_last_iter"] = whist[-1]["d_render"] if whist else None
        # realized vs linearized render decrease INSIDE the window: first accepted
        # candidate -> last accepted candidate, against the summed first-order work
        # (the telemetry's prediction). << 1 = the covector's linear model is not
        # realized by the steps actually taken.
        if len(whist) >= 2 and rw:
            realized = whist[0]["d_render"] - whist[-1]["d_render"]
            pred = float(sum(rw[1:])) if len(rw) > 1 else None
            rec["d_render_realized"] = realized
            rec["d_render_realization"] = (realized / pred if pred and abs(pred) > 1e-30
                                           else None)
        else:
            rec["d_render_realized"] = rec["d_render_realization"] = None
        if whist:
            # ---- runner-style promotion (runner.py: clip -> condition_F -> assimilate) ----
            x_new = np.ascontiguousarray(fr[-1], np.float32)
            n_out = int(((x_new < lo) | (x_new > hi)).any(1).sum())
            n_nan = int((~np.isfinite(x_new).all(1)).sum())
            x = np.clip(np.nan_to_num(x_new), lo, hi).astype(np.float32)
            Fc, n_bad, n_flip, _ = condition_F(end["F"], clamp=False)
            n_ns = int((~np.isfinite(end["v"]).all(1)).sum()
                       + (~np.isfinite(end["C"]).all(axis=(1, 2))).sum())
            st = {"F": Fc, "v": np.nan_to_num(end["v"]).astype(np.float32),
                  "C": np.nan_to_num(end["C"]).astype(np.float32)}
            dets = np.stack([np.linalg.det(Fs) for Fs in F_seq[1:]])
            rec.update({"Jmin": float(np.linalg.det(Fc).min()),
                        "Jmin_traj": float(dets.min()),
                        "Jmax": float(np.linalg.det(Fc).max()),
                        "clamped": n_out, "nan_x": n_nan, "nan_state": n_ns,
                        "F_reset": n_bad, "F_flip": n_flip,
                        "F_invert_steps": int((dets <= 0.0).any(0).sum()),
                        "v_absmax": float(np.abs(st["v"]).max()),
                        "move": float(np.linalg.norm(x - x_prev, axis=1).mean())})
            Fp = assimilate_elastic(Fc, Fp, eta=cfg.assim, smin=cfg.assim_smin,
                                    smax=cfg.assim_smax, isochoric=cfg.assim_iso)
            sv = np.linalg.svd(Fp, compute_uv=False)
            rec["Fp_sv_min"], rec["Fp_sv_max"] = float(sv.min()), float(sv.max())
        else:
            rec.update({"Jmin": None, "Jmin_traj": None, "Jmax": None, "clamped": 0,
                        "nan_x": 0, "nan_state": 0, "F_reset": 0, "F_flip": 0,
                        "F_invert_steps": 0, "v_absmax": None, "move": 0.0})
        rec["d_vol"], rec["d_render"] = eval_losses(x, tgt, cfg)
        commits.append(rec)
        log(f"[{name}] commit {c}: {dt:.1f}s acc/rej={rec['accepted']}/{rec['rejected']}"
            f"{' NULL' if rec['null_commit'] else ''}"
            f"{' (replay-discard)' if rec['replay_discard'] else ''} lam={rec['lambda']:.3g} "
            f"rw={_fmt(rec['render_work'])} pw={_fmt(rec['phys_work'])} "
            f"share_last={_pct(rec['share_last'])} wshare={_pct(rec['wshare_last'])} "
            f"share_sum={_pct(rec['share_sum'])} realiz={_fmt(rec['d_render_realization'])} "
            f"g_cos={_fmt(rec['g_cos'], 3)} pcos={_fmt(rec['precond_cos'])} "
            f"|gx|={_fmt(rec['gx_norm'])} |gr|={_fmt(rec['g_rend_norm'])} "
            f"d_vol={rec['d_vol']:.3f} d_render={rec['d_render']:.5f} "
            f"Jmin={_fmt(rec['Jmin'], 3)} Jtraj={_fmt(rec['Jmin_traj'], 3)} "
            f"Finv={rec['F_invert_steps']} nan={rec['nan_x'] + rec['nan_state']}")
    total_t = time.perf_counter() - t_cond
    mem = gpu_mem_mib()
    summ = summarize(name, spec, commits, dv0, dr0, total_t, mem, tag)
    return {"name": name, "tag": tag, "spec": spec,
            "cfg": {"gs": spec["gs"], "kappa": spec["kappa"], "smoothing": spec["s"]},
            "d_vol0": dv0, "d_render0": dr0, "commits": commits, "summary": summ,
            "total_time_s": total_t, "gpu_mem_mib": mem}


def _fmt(v, nd=3):
    if v is None:
        return "None"
    return f"{v:.{nd}g}"


def _pct(v):
    return "None" if v is None else f"{100 * v:.3g}%"


def _median(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else None


def summarize(name, spec, commits, dv0, dr0, total_t, mem, tag=""):
    live = [c for c in commits if not c["null_commit"]]
    fin = commits[-1]
    j_traj = [c["Jmin_traj"] for c in live if c["Jmin_traj"] is not None]
    return {
        "name": name, "tag": tag, "gs": spec["gs"], "kappa": spec["kappa"],
        "smoothing": spec["s"],
        "n_commits": len(commits), "null_commits": len(commits) - len(live),
        "replay_discards": int(sum(bool(c.get("replay_discard")) for c in commits)),
        "accepted": int(sum(c["accepted"] for c in commits)),
        "rejected": int(sum(c["rejected"] for c in commits)),
        "share_last_med": _median([c["share_last"] for c in live]),
        "wshare_last_med": _median([c.get("wshare_last") for c in live]),
        "ratio_last_med": _median([c["ratio_last"] for c in live]),
        "share_sum_med": _median([c["share_sum"] for c in live]),
        "wshare_sum_med": _median([c.get("wshare_sum") for c in live]),
        "realization_med": _median([c.get("d_render_realization") for c in live]),
        "precond_cos_med": _median([c.get("precond_cos") for c in commits]),
        "gx_norm_med": _median([c.get("gx_norm") for c in commits]),
        "pullback_ratio_med": _median([(c["g_rend_norm"] / c["gx_norm"])
                                       if c.get("g_rend_norm") and c.get("gx_norm")
                                       else None for c in commits]),
        "d_vol_min": min(c["d_vol"] for c in commits),
        "d_render_min": min(c["d_render"] for c in commits),
        "share_last_min": (min(c["share_last"] for c in live if c["share_last"] is not None)
                           if live else None),
        "share_last_max": (max(c["share_last"] for c in live if c["share_last"] is not None)
                           if live else None),
        "render_work_med": _median([c["render_work"] for c in live]),
        "phys_work_med": _median([c["phys_work"] for c in live]),
        "render_work_x_med": _median([c["render_work_x"] for c in live]),
        "render_work_F_med": _median([c["render_work_F"] for c in live]),
        "g_cos_med": _median([c["g_cos"] for c in commits]),
        "g_raw_cos_med": _median([c["g_raw_cos"] for c in commits]),
        "g_share_med": _median([c["g_share"] for c in commits]),
        "g_rend_norm_med": _median([c["g_rend_norm"] for c in commits]),
        "g_phys_norm_med": _median([c["g_phys_norm"] for c in commits]),
        "lambda_last": fin["lambda"],
        "d_vol0": dv0, "d_vol_final": fin["d_vol"], "d_vol_drop": dv0 - fin["d_vol"],
        "d_vol_drop_rel": (dv0 - fin["d_vol"]) / max(dv0, 1e-12),
        "d_render0": dr0, "d_render_final": fin["d_render"],
        "d_render_drop": dr0 - fin["d_render"],
        "d_render_drop_rel": (dr0 - fin["d_render"]) / max(dr0, 1e-12),
        "Jmin_final": fin["Jmin"], "Jmin_traj_min": min(j_traj) if j_traj else None,
        "F_invert_steps": int(sum(c["F_invert_steps"] for c in commits)),
        "nonfinite_events": int(sum(c["nan_x"] + c["nan_state"] + c["F_reset"]
                                    for c in commits)),
        "F_flip": int(sum(c["F_flip"] for c in commits)),
        "clamped": int(sum(c["clamped"] for c in commits)),
        "move_mean": float(np.mean([c["move"] for c in commits])),
        "time_per_commit_s": float(np.mean([c["time_s"] for c in commits])),
        "total_time_s": total_t, "gpu_mem_mib": mem,
    }


def _load_rows(out_dir: Path):
    rows = []
    for f in sorted(out_dir.glob("*.json")):
        d = json.loads(f.read_text())
        s = d["summary"]
        s.setdefault("tag", d.get("tag", ""))
        # back-fill fields the first run's JSONs predate (computed from commits)
        cs = d["commits"]
        live = [c for c in cs if not c["null_commit"]]
        s.setdefault("d_vol_min", min(c["d_vol"] for c in cs))
        s.setdefault("d_render_min", min(c["d_render"] for c in cs))
        s.setdefault("replay_discards", int(sum(
            c["null_commit"] and c.get("render_work") is not None
            and not c.get("grad_converged") for c in cs)))
        if s.get("wshare_last_med") is None:
            s["wshare_last_med"] = _median([
                _share((c["lambda"] or 0.0) * c["render_work"], c["phys_work"])
                if c.get("render_work") is not None else None for c in live])
        for k in ("realization_med", "precond_cos_med", "gx_norm_med",
                  "pullback_ratio_med", "wshare_sum_med"):
            s.setdefault(k, None)
        rows.append(s)
    order = list(CONDITIONS)
    rows.sort(key=lambda r: (order.index(r["name"]) if r["name"] in order else 999,
                             r.get("tag") or ""))
    return rows


def report(out_dir: Path) -> str:
    """Markdown tables from every <cond>[__tag].json in out_dir (the doc's tables)."""
    rows = _load_rows(out_dir)

    def p(v, nd=3):
        return "-" if v is None else f"{v:.{nd}g}"

    def pc(v):
        return "-" if v is None else f"{100 * v:.3g}%"

    def nm(r):
        return r["name"] + (f" [{r['tag']}]" if r.get("tag") else "")

    lines = ["| condition | gs/kappa | s | share_last med | rw/pw med | share_sum med | "
             "lam-weighted share med | g_cos med | g_share med | d_render drop (rel) | "
             "d_vol drop (rel) | d_vol min | acc/rej | null (replay) | Jmin_traj min | "
             "Finv | nonfinite |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {nm(r)} | {r['gs']}/{r['kappa']:g} | {r['smoothing']:g} | "
            f"{pc(r['share_last_med'])} | {p(r['ratio_last_med'])} | "
            f"{pc(r['share_sum_med'])} | {pc(r['wshare_last_med'])} | {p(r['g_cos_med'])} | "
            f"{p(r['g_share_med'])} | {p(r['d_render_drop'], 4)} ({pc(r['d_render_drop_rel'])}) | "
            f"{p(r['d_vol_drop'], 4)} ({pc(r['d_vol_drop_rel'])}) | {p(r['d_vol_min'], 4)} | "
            f"{r['accepted']}/{r['rejected']} | {r['null_commits']} ({r['replay_discards']}) | "
            f"{p(r['Jmin_traj_min'])} | {r['F_invert_steps']} | {r['nonfinite_events']} |")
    lines += ["", "| condition | rw med | pw med | rw_x med | rw_F med | |gx| raw med | "
              "|g_rend| pullback med | pullback/raw | precond cos | realization med | "
              "|g_phys| med | lambda end | d_vol end | d_render end | Jmin end | "
              "move/commit | s/commit | GPU MiB |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {nm(r)} | {p(r['render_work_med'])} | {p(r['phys_work_med'])} | "
            f"{p(r['render_work_x_med'])} | {p(r['render_work_F_med'])} | "
            f"{p(r['gx_norm_med'])} | {p(r['g_rend_norm_med'])} | {p(r['pullback_ratio_med'])} | "
            f"{p(r['precond_cos_med'])} | {p(r['realization_med'])} | "
            f"{p(r['g_phys_norm_med'])} | {p(r['lambda_last'])} | "
            f"{p(r['d_vol_final'], 4)} | {p(r['d_render_final'], 4)} | {p(r['Jmin_final'])} | "
            f"{p(r['move_mean'])} | {p(r['time_per_commit_s'])} | {p(r['gpu_mem_mib'], 4)} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--commits", type=int, default=6)
    ap.add_argument("--conditions", default=",".join(CONDITIONS),
                    help="comma list of CONDITIONS keys (default: all)")
    ap.add_argument("--out", default="probe_out/sobolev_precond",
                    help="result dir (kept OFF /data on purpose)")
    ap.add_argument("--report", action="store_true",
                    help="only print the markdown tables from existing JSONs")
    ap.add_argument("--force", action="store_true", help="re-run conditions with a JSON")
    ap.add_argument("--tag", default="", help="replicate tag: results go to <cond>__<tag>.json")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.report:
        print(report(out))
        return

    def log(msg):
        print(msg, flush=True)

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conds if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions {unknown}; known: {list(CONDITIONS)}")

    prm = MPMParams()
    src = load("assets/isosphere.obj", args.n, 1)
    tgt_x = load("assets/bunny.obj", args.n, 2)
    cfg0 = make_cfg(0, 4.0)
    t0 = time.perf_counter()
    tgt = build_target(tgt_x, prm, cfg0)      # prm enters only through grid geometry
    sw = _surface_weights(src, cfg0.surface_grad_k, cfg0.surface_grad_frac,
                          cfg0.surface_grad_floor)
    sw = np.ascontiguousarray(sw > 0.5, np.float32)
    vol0 = compute_rest_volumes(src, 1.0, prm, cfg0.device)   # runner: V_p0 once at source
    log(f"[probe] N={args.n} T={cfg0.T} iters={cfg0.iters} views={len(tgt.views)} "
        f"loss_res={cfg0.loss_res} ldx={tgt.ldx:.3f} surface_frac={sw.mean():.3f} "
        f"target build {time.perf_counter() - t0:.1f}s; conditions={conds}")
    for name in conds:
        f = out / (f"{name}__{args.tag}.json" if args.tag else f"{name}.json")
        if f.exists() and not args.force:
            log(f"[probe] {name}: exists, skipping (use --force)")
            continue
        torch.cuda.empty_cache()
        res = run_condition(name, CONDITIONS[name], src, tgt, sw, vol0, args.commits, log,
                            tag=args.tag)
        f.write_text(json.dumps(res, indent=1))
        s = res["summary"]
        log(f"[probe] {name} DONE: share_last_med={_pct(s['share_last_med'])} "
            f"share_sum_med={_pct(s['share_sum_med'])} g_cos_med={_fmt(s['g_cos_med'])} "
            f"d_render {s['d_render0']:.5f}->{s['d_render_final']:.5f} "
            f"d_vol {s['d_vol0']:.3f}->{s['d_vol_final']:.3f} "
            f"Jmin_traj_min={_fmt(s['Jmin_traj_min'])} Finv={s['F_invert_steps']} "
            f"nonfinite={s['nonfinite_events']} {s['total_time_s']:.0f}s "
            f"gpu={_fmt(s['gpu_mem_mib'], 4)}MiB")
    print(report(out))


if __name__ == "__main__":
    main()
