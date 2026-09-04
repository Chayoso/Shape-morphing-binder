"""v2 A/B runner — the SAME blessed path with the render channel off/on (+ material arm).

Arms (docs/pipeline_v2.md §5):
  phys        lambda_auto=0, opt_material=False       (Xu et al. objective, v2 stability)
  render      lambda_auto>0                           (render drives dFc; asym D_render)
  render_mat  render + opt_material                   (render also drives per-particle Lame)

Gates evaluated here:
  G1a plumbing (constant sequence == shared control),
  G1b channels (finite-difference check that dL/ds reaches the material leaves, and that
      the v_T adjoint is live — a dead channel 2/4 would otherwise pass every other gate),
  G2 guards==0, G3 rest (tail jitter over SIMULATED frames AND terminal-velocity drift),
  G4 holes (absolute 2% + vs-physics comparison), G5 supremacy (render vs phys).
G6 (visual QA) is server-side via quicklook/make_gif over the FULL frame range.

Run (hyde06):
  CUDA_VISIBLE_DEVICES=0 python scripts/pipeline_run.py --arms phys,render --out output/v2_ab
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from physmorph import metrics  # noqa: E402
from physmorph.mpm import MPMParams  # noqa: E402
from physmorph.mpm.constitutive import lame  # noqa: E402
from physmorph.mpm.function import RolloutSpec, warp_mpm, warp_mpm_full  # noqa: E402
from physmorph.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from physmorph.sampling import load_normalized as load  # noqa: E402


def gate1_plumbing(src, prm, T=6, device="cuda"):
    """G1a: a constant dFc sequence must equal the shared control (atomic-add ULP tolerance)."""
    N = len(src)
    spec = RolloutSpec(x0=src, m=1.0, lam=1.0e5, mu=5.0e4, prm=prm, T=T, device=device)
    torch.manual_seed(0)
    c = torch.randn(N, 3, 3, device=device) * 1e-3
    with torch.no_grad():
        x_shared, _ = warp_mpm(c, spec)
        x_seq, _ = warp_mpm(c.unsqueeze(0).repeat(T, 1, 1, 1).contiguous(), spec)
    d = float((x_shared - x_seq).abs().max())
    tol = 1e-6 * max(1.0, float(x_shared.abs().max()))
    ok = d <= tol
    print(f"[G1a] constant-seq vs shared: max|dx|={d:.3e} (tol {tol:.1e}) -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return {"max_abs_diff": d, "tol": tol, "pass": bool(ok)}


def gate1_channels(src, prm, young=1.4e5, poisson=0.2, device="cuda"):
    """G1b: the NEW plumbing must be alive — dL/ds (material leaves) checked against a
    central finite difference on a small subproblem, with the v_T adjoint in the loss.
    This is a connectivity check (is the channel wired, roughly correct), not a precision
    gradcheck: float32 MPM FD is noisy, hence the loose tolerance."""
    lam0, mu0 = lame(young, poisson)
    n = min(len(src), 512)
    xs = np.ascontiguousarray(src[:n], np.float32)
    T = 6
    spec = RolloutSpec(x0=xs, m=1.0, lam=lam0, mu=mu0, prm=prm, T=T, device=device)
    torch.manual_seed(1)
    dfc = torch.randn(T, n, 3, 3, device=device) * 5e-2      # nonzero stress -> motion
    s = torch.zeros(2, n, device=device, requires_grad=True)

    def L_of(shift):
        lam_t = lam0 * torch.exp(s[0] + shift)
        mu_t = mu0 * torch.exp(s[1] + shift)
        xT, _, vT = warp_mpm_full(dfc, spec, lam_t, mu_t)
        return xT.pow(2).sum() * 1e-3 + vT.pow(2).sum() * 1e-3   # exercises x AND v adjoints

    L = L_of(0.0)
    (g,) = torch.autograd.grad(L, s)
    g_ok = bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 1e-10
    eps = 1e-2
    with torch.no_grad():
        Lp, Lm = float(L_of(+eps)), float(L_of(-eps))
    fd = (Lp - Lm) / (2 * eps)
    an = float(g.sum())          # uniform shift on both rows == sum of all dL/ds entries
    rel = abs(fd - an) / max(abs(fd), abs(an), 1e-9)
    ok = g_ok and rel < 0.25
    print(f"[G1b] material/v_T channels: dL/ds sum analytic={an:.4e} fd={fd:.4e} "
          f"rel_err={rel:.3f} finite={g_ok} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return {"analytic": an, "fd": fd, "rel_err": rel, "grad_finite_nonzero": g_ok,
            "pass": bool(ok)}


def arm_config(arm: str, args) -> PipelineConfig:
    cfg = PipelineConfig(T=args.T, iters=args.iters, animations=args.animations,
                         alpha=args.alpha, w_kin=args.w_kin, w_ctrl=args.w_ctrl,
                         w_box=args.w_box, assim=args.assim, render_views=args.render_views,
                         render_res=args.render_res, loss_res=args.loss_res,
                         eps=args.eps, w_tctrl=args.w_tctrl, w_cov=args.w_cov,
                         surface_grad_frac=args.surface_grad_frac,
                         render_surface_only=args.render_surface_only,
                         control_h1_iters=args.control_h1_iters,
                         nn_tail_frac=args.nn_tail_frac,
                         outer_merit=args.outer_merit,
                         persistent_rest_volume=not args.legacy_recompute_volumes,
                          gauss_covariance=not args.legacy_gauss_centers_only,
                          gauss_sigma_scale=args.gauss_sigma_scale,
                          gauss_children=(1 if args.gauss_children is None
                                          else args.gauss_children),
                          gauss_child_sigma_scale=args.gauss_child_sigma_scale,
                          gauss_child_offset_scale=args.gauss_child_offset_scale,
                          patience=args.patience, tol=args.tol)
    if arm == "phys":
        cfg.lambda_auto = 0.0
    elif arm == "render":
        cfg.lambda_auto = args.lambda_auto
    elif arm == "render_mat":
        cfg.lambda_auto = args.lambda_auto
        cfg.opt_material = True
    elif arm == "render_ws":                       # warm-started dFc (safeguarded)
        cfg.lambda_auto = args.lambda_auto
        cfg.warm_start = True
    elif arm == "render_gs":                       # Sobolev/grid-GS render direction
        cfg.lambda_auto = args.lambda_auto
        cfg.render_gs_iters = args.render_gs_iters
    elif arm == "render_pbr":                      # + PBR-lite shading channel
        cfg.lambda_auto = args.lambda_auto
        cfg.w_pbr = args.w_pbr
    elif arm == "render_pc":                       # + PCGrad conflict projection
        cfg.lambda_auto = args.lambda_auto
        cfg.grad_project = True
    elif arm == "render_c2f":                      # + coarse-to-fine render targets
        cfg.lambda_auto = args.lambda_auto
        cfg.c2f_at = 0.5
    elif arm == "render_pace":                     # + paced trajectory ONLY (attribution)
        cfg.lambda_auto = args.lambda_auto
        cfg.pace = args.pace
    elif arm == "render_lg":                       # + LOCAL-GLOBAL surface band pass
        cfg.lambda_auto = args.lambda_auto
        cfg.lg_sweeps = args.lg_sweeps
    elif arm == "render_creg":                     # + control-field smoothness (fringe fix)
        cfg.lambda_auto = args.lambda_auto
        cfg.w_creg = args.w_creg
    elif arm == "render_full_lg":                  # flagship + local surface consolidation
        cfg.lambda_auto = args.lambda_auto
        cfg.w_pbr = args.w_pbr
        cfg.grad_project = True
        cfg.c2f_at = 0.5
        cfg.pace = args.pace
        cfg.dfc_clip = args.dfc_clip
        cfg.lg_sweeps = args.lg_sweeps
    elif arm == "render_full_creg":                # flagship + control smoothness
        cfg.lambda_auto = args.lambda_auto
        cfg.w_pbr = args.w_pbr
        cfg.grad_project = True
        cfg.c2f_at = 0.5
        cfg.pace = args.pace
        cfg.dfc_clip = args.dfc_clip
        cfg.w_creg = args.w_creg
    elif arm == "render_full_iso":                 # flagship + isochoric plasticity
        cfg.lambda_auto = args.lambda_auto
        cfg.w_pbr = args.w_pbr
        cfg.grad_project = True
        cfg.c2f_at = 0.5
        cfg.pace = args.pace
        cfg.dfc_clip = args.dfc_clip
        cfg.assim_iso = True
    elif arm == "render_full_dt_iso":              # FLAGSHIP: W1 + isochoric + jvol.
        cfg.lambda_auto = args.lambda_auto         # grad_project OFF since h13 ablation
        cfg.w_pbr = args.w_pbr                     # (in-bundle PCGrad contribution <= 0:
        cfg.grad_project = False                   # bunny tie, armadillo -1.1%/+0.8pt)
        cfg.c2f_at = 0.5
        cfg.pace = args.pace
        cfg.dfc_clip = args.dfc_clip
        cfg.w_creg = args.w_creg
        cfg.w_dt = args.w_dt
        cfg.w_jvol = args.w_jvol
        cfg.assim_iso = True
    elif arm == "render_full_dt_iso_nopc":         # flagship MINUS PCGrad (attribution:
        cfg.lambda_auto = args.lambda_auto         #   standalone pc was falsified in v4;
        cfg.w_pbr = args.w_pbr                     #   in-bundle contribution never isolated,
        cfg.grad_project = False                   #   and the stack has changed since)
        cfg.c2f_at = 0.5
        cfg.pace = args.pace
        cfg.dfc_clip = args.dfc_clip
        cfg.w_creg = args.w_creg
        cfg.w_dt = args.w_dt
        cfg.w_jvol = args.w_jvol
        cfg.assim_iso = True
    elif arm == "render_full_dt_iso_nn":           # FLAGSHIP since h15 (fork-halo -70%,
                                                   # chamfer -5.7%, late g_cos conflict gone)
        cfg.lambda_auto = args.lambda_auto
        cfg.w_pbr = args.w_pbr
        cfg.grad_project = True
        cfg.c2f_at = 0.5
        cfg.pace = args.pace
        cfg.dfc_clip = args.dfc_clip
        cfg.w_creg = args.w_creg
        cfg.w_dt = args.w_dt
        cfg.w_jvol = args.w_jvol
        cfg.w_nn = args.w_nn
        cfg.nn_far_k = args.nn_far_k
        cfg.w_kde = args.w_kde
        cfg.nn_berth_k = args.nn_berth_k
        cfg.mom_carry = args.mom_carry
        cfg.anneal_stale = args.anneal
        cfg.pace_budget = args.pace_budget
        cfg.assim_iso = True
    elif arm == "render_full_fill_iso":            # full stack + norm-balanced fill v3
        cfg.lambda_auto = args.lambda_auto
        cfg.w_pbr = args.w_pbr
        cfg.grad_project = False
        cfg.c2f_at = 0.5
        cfg.pace = args.pace
        cfg.dfc_clip = args.dfc_clip
        cfg.w_creg = args.w_creg
        cfg.w_dt = args.w_dt
        cfg.w_nn = args.w_nn
        cfg.nn_far_k = args.nn_far_k
        cfg.w_kde = args.w_kde
        cfg.nn_berth_k = args.nn_berth_k
        cfg.w_fill = args.w_fill
        cfg.w_jvol = args.w_jvol
        cfg.assim_iso = True
    elif arm == "render_dt":                       # + pointwise-W1 spray (fringe residue)
        cfg.lambda_auto = args.lambda_auto
        cfg.w_dt = args.w_dt
    elif arm == "render_full_dt":                  # flagship + creg + W1 spray (hero3)
        cfg.lambda_auto = args.lambda_auto
        cfg.w_pbr = args.w_pbr
        cfg.grad_project = True
        cfg.c2f_at = 0.5
        cfg.pace = args.pace
        cfg.dfc_clip = args.dfc_clip
        cfg.w_creg = args.w_creg
        cfg.w_dt = args.w_dt
    elif arm == "render_full":                     # PBR + PCGrad + c2f + paced trajectory
        cfg.lambda_auto = args.lambda_auto
        cfg.w_pbr = args.w_pbr
        cfg.grad_project = True
        cfg.c2f_at = 0.5
        cfg.pace = args.pace
        cfg.dfc_clip = args.dfc_clip
    elif arm == "render_full_grow":                # full stack + GROWTH channel
        cfg.lambda_auto = args.lambda_auto
        cfg.w_pbr = args.w_pbr
        cfg.grad_project = False
        cfg.c2f_at = 0.5
        cfg.pace = args.pace
        cfg.dfc_clip = args.dfc_clip
        cfg.w_creg = args.w_creg
        cfg.w_dt = args.w_dt
        cfg.w_nn = args.w_nn
        cfg.nn_far_k = args.nn_far_k
        cfg.w_kde = args.w_kde
        cfg.nn_berth_k = args.nn_berth_k
        cfg.w_fill = args.w_fill
        cfg.w_jvol = args.w_jvol
        cfg.w_grow = args.w_grow
        cfg.assim_iso = True
    elif arm == "render_full_gauss":               # flagship with the REAL 3DGS loss
        cfg.lambda_auto = args.lambda_auto         # replacing the CIC soft-silhouette
        cfg.w_pbr = 0.0
        cfg.grad_project = False
        if args.gauss_mix <= 0:                    # pure replacement: res-fixed targets
            cfg.c2f_at = 0.0                       # hybrid keeps c2f on the silhouette
                                                   # (g3 confound: mix was tested c2f-off)
        cfg.pace = args.pace
        cfg.dfc_clip = args.dfc_clip
        cfg.w_creg = args.w_creg
        cfg.w_dt = args.w_dt
        cfg.w_nn = args.w_nn
        cfg.nn_far_k = args.nn_far_k
        cfg.w_kde = args.w_kde
        cfg.nn_berth_k = args.nn_berth_k
        cfg.w_jvol = args.w_jvol
        cfg.gauss_mix = args.gauss_mix
        cfg.mom_carry = args.mom_carry
        cfg.anneal_stale = args.anneal
        cfg.pace_budget = args.pace_budget
        cfg.assim_iso = True
        cfg.use_gauss_loss = True
        cfg.gauss_res = args.gauss_res
    elif arm == "render_flag_dress":               # Tier D ladder: FLAGSHIP global solver
        cfg.lambda_auto = args.lambda_auto         # (byte-identical window) + gauss built
        cfg.w_pbr = args.w_pbr                     # for dressing/telemetry only
        cfg.grad_project = False
        cfg.c2f_at = 0.5
        cfg.pace = args.pace
        cfg.dfc_clip = args.dfc_clip
        cfg.w_creg = args.w_creg
        cfg.w_dt = args.w_dt
        cfg.w_nn = args.w_nn
        cfg.nn_far_k = args.nn_far_k
        cfg.w_kde = args.w_kde
        cfg.nn_berth_k = args.nn_berth_k
        cfg.w_jvol = args.w_jvol
        cfg.assim_iso = True
        cfg.mom_carry = args.mom_carry
        cfg.anneal_stale = args.anneal
        cfg.pace_budget = args.pace_budget
        cfg.use_gauss_loss = True
        cfg.gauss_in_objective = False
        cfg.gauss_res = args.gauss_res
        cfg.gauss_children = 4 if args.gauss_children is None else args.gauss_children
        cfg.render_surface_only = True             # gauss parents = frozen surface subset
        cfg.surface_grad_frac = (args.surface_grad_frac
                                 if args.surface_grad_frac > 0 else 0.50)
        cfg.surface_mask_objective = False         # ...but the window covector is unmasked
        cfg.local_dress_iters = args.dress_iters
    elif arm == "render_stable_gauss":             # production: exact render + trust gates
        cfg.lambda_auto = args.lambda_auto
        cfg.w_pbr = 0.0
        cfg.grad_project = False
        cfg.c2f_at = 0.5
        cfg.pace = args.pace
        cfg.dfc_clip = args.dfc_clip
        cfg.w_creg = args.w_creg
        cfg.w_dt = args.w_dt
        cfg.w_nn = args.w_nn
        cfg.nn_far_k = args.nn_far_k
        cfg.w_kde = args.w_kde
        cfg.nn_berth_k = args.nn_berth_k
        cfg.w_jvol = args.w_jvol
        cfg.gauss_mix = args.gauss_mix if args.gauss_mix > 0 else 0.25
        cfg.mom_carry = args.mom_carry if args.mom_carry > 0 else 0.8
        cfg.anneal_stale = args.anneal if args.anneal > 0 else 0.5
        cfg.patience = max(cfg.patience, 8)
        cfg.w_kin = max(args.w_kin, 20.0)
        cfg.w_tctrl = args.w_tctrl if args.w_tctrl > 0 else 10.0
        # w_cov RETIRED as an arm default (s1 forensic 2026-09-02): the band
        # penalty on total F fights the transient stretch the morph transport
        # REQUIRES (sval>2 to move mass into the ears); at a17-19 its gradient
        # overwhelmed the data terms (gp 6.4->18), collapsed dFc 0.020->0.003
        # and regressed every track 20-60%. At convergence assimilation absorbs
        # F into Fp (svals->1), so the penalty only ever binds mid-run, where it
        # does damage. Knob + diagnostics stay for explicit A/Bs.
        cfg.w_cov = args.w_cov
        # High-resolution production uses many small surface splats.  Do not inflate
        # them to hide sparse sampling holes: the 20k calibration owns this value.
        cfg.gauss_sigma_scale = args.gauss_sigma_scale
        cfg.gauss_children = (4 if args.gauss_children is None
                              else args.gauss_children)
        cfg.nn_tail_frac = args.nn_tail_frac
        cfg.surface_grad_frac = (args.surface_grad_frac
                                 if args.surface_grad_frac > 0 else 0.50)
        cfg.render_surface_only = True
        cfg.control_h1_iters = args.control_h1_iters
        cfg.pace_budget = (args.pace_budget if args.pace_budget > 0 else 0.01)
        cfg.local_dress_iters = args.dress_iters
        cfg.outer_merit = True
        cfg.assim_iso = True
        cfg.use_gauss_loss = True
        cfg.gauss_res = args.gauss_res
    else:
        raise SystemExit(f"unknown arm {arm!r} (phys|render|render_mat|render_ws|render_gs"
                         "|render_pbr|render_pc|render_c2f|render_pace|render_lg|render_full)")
    return cfg


def eval_gates(tag, res, met, prm, T, rel_tol=0.003, hole_tol=0.02):
    g = res["guards"]
    # G3 rest: tail jitter over SIMULATED frames AND the drift a further window would
    # produce from the promoted terminal velocity (held padding proves nothing).
    v_mean = next((h["v_mean"] for h in reversed(res["history"]) if "v_mean" in h), 0.0)
    drift_rel = v_mean * prm.dt * T / max(met["bbox_diag"], 1e-9)
    gates = {
        "G2_guards": all(v == 0 for v in g.values()),
        "G3_rest": met["jitter_rel"] < rel_tol and drift_rel < rel_tol,
        # target-relative ceiling (pre-registered 2026-09-02): the A->C TARGET itself
        # measures 5.76% under this splat metric, so an absolute 2% is unattainable for
        # that pair — a body at the target's own hole level has no coverage defect
        "G4_holes_abs": met["hole_frac"] <= max(hole_tol,
                                                met.get("hole_frac_tgt", 0.0) + 0.005),
        "G4_ejection": met["outside_max"] == 0.0 and met["stray_max"] < 2e-3,
    }
    if "render_out_nn_far_frac" in met:
        gates["G4_render_surface"] = met["render_out_nn_far_frac"] < 2e-3
    print(f"[{tag}] gates: " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}"
                                         for k, v in gates.items()) +
          f"   (guards={g}, jitter_rel={met['jitter_rel']:.5f}, drift_rel={drift_rel:.5f}, "
          f"hole={met['hole_frac']*100:.2f}% tgt={met['hole_frac_tgt']*100:.2f}%, "
          f"outside_max={met['outside_max']*100:.3f}%, stray_max={met['stray_max']*100:.3f}%)",
          flush=True)
    gates["drift_rel"] = drift_rel
    return gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="assets/isosphere.obj")
    ap.add_argument("--tgt", default="assets/bunny.obj")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--animations", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--lambda_auto", type=float, default=0.5)
    ap.add_argument("--w_kin", type=float, default=0.5)
    ap.add_argument("--w_ctrl", type=float, default=1e-3)
    ap.add_argument("--w_tctrl", type=float, default=0.0)
    ap.add_argument("--w_box", type=float, default=10.0)
    ap.add_argument("--assim", type=float, default=0.5)
    ap.add_argument("--render_views", type=int, default=6)
    ap.add_argument("--render_res", type=int, default=64)
    ap.add_argument("--loss_res", type=int, default=32)
    ap.add_argument("--render_gs_iters", type=int, default=20)
    ap.add_argument("--w_pbr", type=float, default=1.0)
    ap.add_argument("--pace", type=float, default=0.0)   # r3/r5 (2026-09-03): 0.12 capped convergence depth
    ap.add_argument("--lg_sweeps", type=int, default=8)
    ap.add_argument("--w_creg", type=float, default=100.0)
    ap.add_argument("--w_dt", type=float, default=0.2)   # SUM form: per-particle pull
                                                         # = w_dt (Opus parity estimate)
    ap.add_argument("--w_fill", type=float, default=0.1)  # fill v3 ALPHA (norm-balanced)
    ap.add_argument("--w_grow", type=float, default=0.02)
    ap.add_argument("--grow_band", type=float, default=1.5)
    ap.add_argument("--gauss_res", type=int, default=96)
    ap.add_argument("--gauss_sigma_scale", type=float, default=1.0)
    ap.add_argument("--gauss_children", type=int, default=None,
                    help="render children/parent (default: 4 for render_stable_gauss, else 1)")
    ap.add_argument("--gauss_child_sigma_scale", type=float, default=0.55)
    ap.add_argument("--gauss_child_offset_scale", type=float, default=0.35)
    ap.add_argument("--w_cov", type=float, default=0.0)
    ap.add_argument("--surface_grad_frac", type=float, default=0.0)
    ap.add_argument("--render_surface_only", action="store_true")
    ap.add_argument("--control_h1_iters", type=int, default=0)
    ap.add_argument("--outer_merit", dest="outer_merit", action="store_true", default=True)
    ap.add_argument("--no_outer_merit", dest="outer_merit", action="store_false")  # gate v3 brake is a safety net (r5: 0 rejects)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--tol", type=float, default=0.003)  # plateau-track relative improvement threshold
    ap.add_argument("--pace_budget", type=float, default=0.0)
    ap.add_argument("--dress_iters", type=int, default=0)  # Tier D stage ladder
    ap.add_argument("--legacy_recompute_volumes", action="store_true")
    ap.add_argument("--legacy_gauss_centers_only", action="store_true")
    ap.add_argument("--anneal", type=float, default=0.0)  # plateau step decay
    ap.add_argument("--gauss_mix", type=float, default=0.0)  # hybrid sil+gauss render
    ap.add_argument("--mom_carry", type=float, default=0.0)  # cross-window Adam moments
    ap.add_argument("--w_nn", type=float, default=0.2)
    ap.add_argument("--nn_far_k", type=float, default=1000.0)  # E4 adopted: own all far particles
    ap.add_argument("--w_kde", type=float, default=0.0)  # particle-scale density matching (1 = equal-norm to D_vol)
    ap.add_argument("--nn_berth_k", type=float, default=1.0)  # near-band berth (x2 adopted: no dead band)
    ap.add_argument("--nn_tail_frac", type=float, default=0.0)
    ap.add_argument("--live_port", type=int, default=0)  # >0: stream this run
                                        # for live.html / the /quad dashboard
    ap.add_argument("--w_jvol", type=float, default=50.0)  # h12 ladder: detFmin
                                        # 0.0005->0.497, |J-1|>0.3 13.7->0.0%,
                                        # chamfer/silIoU best-ever (docs 2026-09-02)
    ap.add_argument("--dfc_clip", type=float, default=0.02)
    ap.add_argument("--arms", default="phys,render")
    ap.add_argument("--save_F_stride", type=int, default=0,
                    help="save every k-th F frame (0 = T, i.e. commit boundaries)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="output/v2_ab")
    args = ap.parse_args()

    src = load(args.src, args.n, args.seed)
    tgt = load(args.tgt, args.n, args.seed + 1)
    prm = MPMParams()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"[v2run] {args.src} -> {args.tgt}  N={args.n}  T={args.T}  iters={args.iters}  "
          f"anims={args.animations} | dx={prm.dx} dt={prm.dt:.5f} smoothing={prm.smoothing}",
          flush=True)
    print(f"[v2run] baseline chamfer (undeformed) = {metrics.chamfer(src, tgt):.4f}", flush=True)

    live = None
    if args.live_port:
        from physmorph.viewer.server import LiveServer
        live = LiveServer(args.live_port)

    tracked = [Path("physmorph/pipeline/config.py"),
               Path("physmorph/pipeline/optimizer.py"),
               Path("physmorph/pipeline/runner.py"),
               Path("physmorph/pipeline/gauss_loss.py"),
               Path("physmorph/losses/volumetric.py"),
               Path("physmorph/mpm/kernels.py"),
               Path("physmorph/mpm/traj.py")]
    code_hash = hashlib.sha256(b"".join(p.read_bytes() for p in tracked)).hexdigest()[:16]
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                          stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    out = {"provenance": {**vars(args), "mpm": dataclasses.asdict(prm),
                           "git_sha": git_sha, "code_hash": code_hash},   # AGENTS rule 4:
           "G1a": gate1_plumbing(src, prm),                                # discretisation
           "G1b": gate1_channels(src, prm), "arms": {}}                    # travels with numbers

    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        cfg = arm_config(arm, args)
        # snapshot BEFORE the run: c2f mutates cfg.render_res mid-run (the archived
        # config must record what the run STARTED with; the c2f switch is in history)
        cfg_dump = {k: v for k, v in dataclasses.asdict(cfg).items() if k != "history"}
        print(f"\n[v2run] ===== ARM {arm} =====", flush=True)
        t0 = time.time()
        cbs = (None, None)
        if live is not None:
            from physmorph.render.covariance import sigma0_from_nn
            cbs = live.begin_run(arm, src, tgt, prm, cfg, sigma0_from_nn(tgt, 0.9))
        res = run_pipeline(src, tgt, prm, cfg, on_commit=cbs[0], on_iter=cbs[1])
        dt = time.time() - t0
        dn = res.get("deliver_n") or len(res["frames"])   # metrics on the DELIVERED slice
        met = metrics.summarize(res["frames"][:dn], tgt, F_frames=res["F_frames"][:dn],
                                n_held=res["n_held"], render_mask=res.get("render_mask"))
        # trajectory evenness: CV of per-commit displacement (snap-to-target -> high CV)
        mv = [h["move"] for h in res["history"] if "move" in h]
        # <3 commits IS the snap pathology — score it worst, not best (adversarial finding)
        met["move_cv"] = (float(np.std(mv) / max(np.mean(mv), 1e-9))
                          if len(mv) > 2 else float("inf"))
        met["move_first_frac"] = (float(sum(mv[:3]) / max(sum(mv), 1e-9)) if mv else 1.0)
        gates = eval_gates(arm, res, met, prm, args.T)
        stride = args.save_F_stride if args.save_F_stride > 0 else args.T
        nF = len(res["F_frames"])
        idx = sorted(set(range(0, nF, stride)) | {nF - 1})
        archive_extra = {}
        if cfg.use_gauss_loss and cfg.lambda_auto > 0:
            from physmorph.pipeline.runner import _surface_weights
            from physmorph.render.children import tangent_child_offsets
            from physmorph.render.covariance import sigma0_from_nn
            src_mask = (res["render_mask"] if res.get("render_mask") is not None
                        else np.ones(len(src), bool))
            tgt_mask = ((_surface_weights(tgt, cfg.surface_grad_k,
                                          cfg.surface_grad_frac,
                                          cfg.surface_grad_floor) > 0.5)
                        if cfg.render_surface_only else np.ones(len(tgt), bool))
            sigma0 = sigma0_from_nn(tgt[tgt_mask], cfg.gauss_sigma_scale)
            archive_extra = {
                "target_render_mask": tgt_mask, "sigma0": np.float32(sigma0),
                "source_child_offsets": tangent_child_offsets(
                    src, src_mask, sigma0, cfg.gauss_children,
                    cfg.gauss_child_offset_scale, cfg.gauss_child_k),
                "target_child_offsets": tangent_child_offsets(
                    tgt, tgt_mask, sigma0, cfg.gauss_children,
                    cfg.gauss_child_offset_scale, cfg.gauss_child_k),
                "gauss_child_sigma_scale": np.float32(
                    cfg.gauss_child_sigma_scale if cfg.gauss_children > 1 else 1.0),
            }
        np.savez_compressed(
            f"{args.out}_{arm}.npz", src=src, tgt=tgt,
            frames=np.stack(res["frames"]), deliver_n=np.int64(dn),
            truncation=json.dumps(res.get("truncation")),
            F_samples=np.stack([res["F_frames"][i] for i in idx]),
            F_sample_idx=np.array(idx),
            render_mask=(res["render_mask"] if res.get("render_mask") is not None
                          else np.ones(len(src), bool)),
            s=res["s"] if res["s"] is not None else np.zeros(0, np.float32),
            **archive_extra)
        out["arms"][arm] = {"config": cfg_dump, "metrics": met,
                            "gates": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                                      for k, v in gates.items()},
                            "guards": res["guards"], "converged": res["converged"],
                            "n_held": res["n_held"], "seconds": dt, "history": res["history"]}
        print(f"[v2run] ARM {arm}: chamfer={met['chamfer']:.4f}  silIoU={met['sil_iou']:.4f}  "
              f"hole={met['hole_frac']*100:.2f}%  jitter_rel={met['jitter_rel']:.5f}  "
              f"detFmin={met.get('detF_min', 1.0):.4f}  move_cv={met['move_cv']:.2f}  "
              f"first3={met['move_first_frac']*100:.0f}%  ({dt/60:.1f} min)", flush=True)

    # ---- cross-arm gates: every render-driven arm vs its physics-only baseline ----
    base = "phys" if "phys" in out["arms"] else None
    if base:
        mp = out["arms"][base]["metrics"]
        out["G5"] = {}
        for arm, rec in out["arms"].items():
            if arm == base:
                continue
            mr = rec["metrics"]
            g5 = {"sil_iou_up": mr["sil_iou"] > mp["sil_iou"],
                  "chamfer_ok": mr["chamfer"] <= mp["chamfer"] * 1.02,
                  "holes_down": mr["hole_frac"] <= mp["hole_frac"] + 1e-9}
            out["G5"][arm] = {**{k: bool(v) for k, v in g5.items()},
                              "pass": bool(all(g5.values()))}
            print(f"[G5:{arm} vs {base}] silIoU {mp['sil_iou']:.4f}->{mr['sil_iou']:.4f}  "
                  f"chamfer {mp['chamfer']:.4f}->{mr['chamfer']:.4f}  "
                  f"hole {mp['hole_frac']*100:.2f}%->{mr['hole_frac']*100:.2f}%  -> "
                  f"{'PASS' if out['G5'][arm]['pass'] else 'FAIL'}", flush=True)

    Path(f"{args.out}.json").write_text(json.dumps(out))
    print(f"\nsaved {args.out}.json", flush=True)


if __name__ == "__main__":
    main()
