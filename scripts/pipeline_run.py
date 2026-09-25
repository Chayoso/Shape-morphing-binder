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
                         w_box=args.w_box, assim=args.assim, assim_consensus=args.assim_consensus,
                         young=args.young, poisson=args.poisson, render_until=args.render_until,
                         ot_handoff=args.ot_handoff,
                         render_views=args.render_views,
                         render_res=args.render_res, loss_res=args.loss_res,
                         grad_dump=args.grad_dump, layer_relax=args.layer_relax, layer_frac=args.layer_frac,
                         disc_ref=args.disc_ref, stop_on_cycle=args.stop_on_cycle, u_rprop=args.u_rprop, commit_pic=args.commit_pic, rebound_probe=args.rebound_probe, rest_commit=args.rest_commit, rest_commit_gate=args.rest_commit_gate, rest_commit_reversal=args.rest_commit_reversal, pace_project=args.pace_project, outer_latch_reversal=args.outer_latch_reversal, plan_native=args.plan_native, ctrl_rprop=args.ctrl_rprop, ctrl_rprop_smooth=args.ctrl_rprop_smooth, ctrl_rprop_k=args.ctrl_rprop_k, ctrl_rprop_arrived=args.ctrl_rprop_arrived, ctrl_rprop_hold=args.ctrl_rprop_hold, ctrl_rprop_hold_onset=args.ctrl_rprop_hold_onset, freeze_arrived=args.freeze_arrived, settle_eta=args.settle_eta, settle_commit=args.settle_commit, settle_pin=args.settle_pin, settle_pin_clear=args.settle_pin_clear, settle_pin_assim=args.settle_pin_assim, settle_pin_ray=args.settle_pin_ray, settle_pin_yield=args.settle_pin_yield, settle_pin_slip=args.settle_pin_slip, settle_pin_kkt=args.settle_pin_kkt, settle_pin_follow=args.settle_pin_follow, arrive_cap=args.arrive_cap, u_rprop_floor=args.u_rprop_floor, shift_sub=args.shift_sub, shift_h_sp=args.shift_h_sp, layer_ctrl=args.layer_ctrl, pbr_denoised=args.pbr_denoised, layer_F=args.layer_F, layer_F_depth=args.layer_F_depth, layer_gate=args.layer_gate, layer_gate_geom=args.layer_gate_geom, layer_gate_geom_cells=args.layer_gate_geom_cells, layer_gate_ot=args.layer_gate_ot, layer_gate_ot_cells=args.layer_gate_ot_cells, layer_gate_ot_normal=args.layer_gate_ot_normal, layer_u_render_only=args.layer_u_render_only,
                         layer_ctrl_smooth=args.layer_ctrl_smooth, sil_kernel=args.sil_kernel,
                         eps=args.eps, w_tctrl=args.w_tctrl, w_cov=args.w_cov,
                         surface_grad_frac=args.surface_grad_frac,
                         render_surface_only=args.render_surface_only,
                         control_h1_iters=args.control_h1_iters, grad_h1=args.grad_h1,
                         nn_tail_frac=args.nn_tail_frac,
                         outer_merit=args.outer_merit,
                         persistent_rest_volume=not args.legacy_recompute_volumes,
                          gauss_covariance=not args.legacy_gauss_centers_only,
                          gauss_sigma_scale=args.gauss_sigma_scale,
                          gauss_children=(1 if args.gauss_children is None
                                          else args.gauss_children),
                          gauss_child_sigma_scale=args.gauss_child_sigma_scale,
                          gauss_child_offset_scale=args.gauss_child_offset_scale,
                          patience=args.patience, tol=args.tol,
                          outer_reversal_always=args.reversal_always,
                          anneal_on_reversal=args.anneal_rev,
                          # render-controls-physics contract (docs/render_controls_physics.md)
                          control_grid=args.control_grid,
                          control_tknots=args.control_tknots,
                          render_F_geom=args.render_F_geom,
                          w_kin_running=args.w_kin_running,
                          w_kin_var=args.w_kin_var,
                          w_coh=args.w_coh, coh_k=args.coh_k,
                          continuity=args.continuity, bonds=args.bonds, reattach=args.reattach,
                          phys_loss=args.phys_loss, ot_eps_cells=args.ot_eps_cells,
                          ot_samples=args.ot_samples, ot_iters=args.ot_iters, ot_debias=args.ot_debias,
                          ot_tol=args.ot_tol,
                          eject_veto=args.eject_veto, eject_iso_k=args.eject_iso_k,
                          w_esc=args.w_esc, esc_k=args.esc_k, archive_stride=args.archive_stride,
                          w_bond=args.w_bond, bond_s0=args.bond_s0,
                          vol_frontier=args.vol_frontier,
                          warm_start=args.warm_start,
                          grad_project_mode=args.grad_project_mode,
                          cagrad_c=args.cagrad_c,
                          render_gs_cheb=args.render_gs_cheb,
                          gauss_robust_eps=args.gauss_robust_eps,
                          loss_units=args.loss_units, dvol_form=args.dvol_form)
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
        cfg.w_jdens = args.w_jdens
        cfg.w_h1 = args.w_h1
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
        cfg.w_jdens = args.w_jdens
        cfg.w_h1 = args.w_h1
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
        cfg.w_jdens = args.w_jdens
        cfg.w_h1 = args.w_h1
        cfg.nn_berth_k = args.nn_berth_k
        cfg.w_fill = args.w_fill
        cfg.w_jvol = args.w_jvol
        cfg.w_grow = args.w_grow
        cfg.assim_iso = True
    elif arm in ("render_ctrl", "render_ctrl_gauss", "render_ctrl_first"):
        # RENDER-CONTROLS-PHYSICS arms (docs/render_controls_physics.md §8, 2026-09-14):
        # the flagship stack (W1 + near-band + jvol + isochoric assimilation) with the
        # control on a coarse basis (default 12^3 nodes x 4 time knots), the render
        # covariance on the geometric F, a running kinetic term and Chebyshev covector
        # smoothing. `_gauss` adds the hybrid 3DGS image loss on the surface parents;
        # `_first` is the render-first composite (physics component projected off the
        # render direction when they conflict) — the pre-registered A/B ladder.
        cfg.lambda_auto = args.lambda_auto
        cfg.w_pbr = args.w_pbr
        cfg.c2f_at = 0.5
        cfg.pace = args.pace
        cfg.dfc_clip = args.dfc_clip
        cfg.w_creg = 0.0                           # the basis is smooth by construction
        cfg.w_dt = args.w_dt
        cfg.w_jvol = args.w_jvol
        cfg.w_nn = args.w_nn
        cfg.nn_far_k = args.nn_far_k
        cfg.w_h1 = args.w_h1
        cfg.nn_berth_k = args.nn_berth_k
        cfg.anneal_stale = args.anneal
        cfg.assim_iso = True
        cfg.control_grid = args.control_grid if args.control_grid > 0 else 12
        cfg.control_tknots = args.control_tknots if args.control_tknots > 0 else 4
        cfg.render_F_geom = True
        cfg.w_kin_running = args.w_kin_running if args.w_kin_running > 0 else 1.0
        cfg.render_gs_iters = args.render_gs_iters
        cfg.render_gs_cheb = True
        cfg.grad_project = arm == "render_ctrl_first"
        cfg.grad_project_mode = "phys" if arm == "render_ctrl_first" else args.grad_project_mode
        if arm in ("render_ctrl_gauss", "render_ctrl_first"):
            cfg.use_gauss_loss = True
            cfg.gauss_res = args.gauss_res
            cfg.gauss_mix = args.gauss_mix if args.gauss_mix > 0 else 0.25
            cfg.gauss_robust_eps = args.gauss_robust_eps if args.gauss_robust_eps > 0 else 0.02
            cfg.gauss_children = 4 if args.gauss_children is None else args.gauss_children
            cfg.render_surface_only = True
            cfg.surface_grad_frac = (args.surface_grad_frac
                                     if args.surface_grad_frac > 0 else 0.50)
            cfg.surface_mask_objective = False
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
        cfg.w_jdens = args.w_jdens
        cfg.w_h1 = args.w_h1
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
        cfg.w_jdens = args.w_jdens
        cfg.w_h1 = args.w_h1
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
        cfg.w_jdens = args.w_jdens
        cfg.w_h1 = args.w_h1
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
    # REFUTE-2 F16: the drift must describe the DELIVERED slice (a truncated tail's
    # terminal velocity was being read before)
    dn = res.get("deliver_n_used")
    recs = [h for h in res["history"] if "v_mean" in h
            and (dn is None or (h.get("frame_end") or 0) <= dn)]
    v_mean = recs[-1]["v_mean"] if recs else 0.0
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
          f"outside_max={met['outside_max']*100:.3f}%, stray_max={met['stray_max']*100:.3f}%; "
          f"layer breathing: flips {met.get('layer_flip_frac', float('nan')):.2f}, net/summed "
          f"{met.get('layer_net_ratio', float('nan')):.2f}, step {met.get('layer_step_sp', float('nan')):.3f} sp)",
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
    ap.add_argument("--young", type=float, default=1.4e5, help="Young's modulus of the body (material study)")
    ap.add_argument("--poisson", type=float, default=0.2, help="Poisson ratio of the body (material study)")
    ap.add_argument("--render_until", type=int, default=0,
                    help=">0: switch the render channel off from this window on (intervention)")
    ap.add_argument("--ot_handoff", action="store_true",
                    help="ot_pace: hand the window target to the FIXED target once every deficit cell is adjacent to the body")
    ap.add_argument("--assim_consensus", action="store_true")  # neighbourhood-consensus plasticity
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
    ap.add_argument("--grad_h1", action="store_true")          # Sobolev descent direction (material kNN)
    ap.add_argument("--outer_merit", dest="outer_merit", action="store_true", default=True)
    ap.add_argument("--no_outer_merit", dest="outer_merit", action="store_false")  # gate v3 brake is a safety net (r5: 0 rejects)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--tol", type=float, default=0.003)  # plateau-track relative improvement threshold
    ap.add_argument("--reversal_always", action="store_true")  # v4: reversal reject without latch
    ap.add_argument("--anneal_rev", type=float, default=0.0)  # v6: alpha x this on commit reversal
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
    ap.add_argument("--w_jdens", type=float, default=0.0)  # density-measured volume prior (1 = equal-norm to D_vol)
    ap.add_argument("--w_h1", type=float, default=0.0)  # non-local H^-1 mass balance (1 = equal-norm to D_vol)
    ap.add_argument("--nn_berth_k", type=float, default=1.0)  # near-band berth (x2 adopted: no dead band)
    ap.add_argument("--nn_tail_frac", type=float, default=0.0)
    ap.add_argument("--live_port", type=int, default=0)  # >0: stream this run
                                        # for live.html / the /quad dashboard
    ap.add_argument("--live_dir", default="",  # persistent file-backed viewer sink
                    help="publish states to DIR/<out-name>_<arm> for scripts/viewer_serve.py")
    # ---- render-controls-physics contract (docs/render_controls_physics.md) ----
    ap.add_argument("--control_grid", type=int, default=0)     # nodes/axis (0 = per particle)
    ap.add_argument("--control_tknots", type=int, default=0)   # time knots (0 = per step)
    ap.add_argument("--render_F_geom", action="store_true")
    ap.add_argument("--w_kin_running", type=float, default=0.0)
    ap.add_argument("--w_kin_var", type=float, default=0.0)   # window velocity-variance term
    ap.add_argument("--w_coh", type=float, default=0.0)       # material-coherence prior (thin-feature vanguard)
    ap.add_argument("--continuity", action="store_true")      # discrete-continuity line-search feasibility (ejection fix)
    ap.add_argument("--bonds", action="store_true")           # material bonds for decoupled particles (numerical-fracture repair)
    ap.add_argument("--reattach", action="store_true")        # merge grid-disconnected particles back onto the body at each commit
    ap.add_argument("--phys_loss", default="density", choices=["density", "ot", "ot_leash", "ot_pace", "ot_resid", "ot_shape", "auto"],
                    help="density: cell sum; ot: transport map L2 (H3); ot_leash: cell sum + hinge beyond the plan blur; "
                         "ot_pace: cell sum against the displacement-interpolated target (one blur radius per window)")
    ap.add_argument("--ot_eps_cells", type=float, default=0.0,
                    help="Sinkhorn sqrt(eps) in loss cells; 0 = the particle spacing (default)")
    ap.add_argument("--ot_samples", type=int, default=8192)
    ap.add_argument("--ot_iters", type=int, default=400, help="Sinkhorn sweep cap per plan")
    ap.add_argument("--ot_tol", type=float, default=1e-2,
                    help="row-marginal error at which a plan counts as converged")
    ap.add_argument("--ot_debias", action="store_true")
    ap.add_argument("--domain", default="fixed", choices=["fixed", "auto"])  # auto: grid = leash box + stencil margin
    ap.add_argument("--v_max", type=float, default=0.0)        # G2P speed cap [wu/s], 0 = off (MPMParams.v_max)
    ap.add_argument("--eject_veto", action="store_true")       # reject windows that add isolated particles
    ap.add_argument("--eject_iso_k", type=float, default=6.0)  #   isolation radius in target spacings
    ap.add_argument("--w_esc", type=float, default=0.0)        # escape-velocity hinge (window-end v vs neighbours)
    ap.add_argument("--esc_k", type=float, default=3.0)
    ap.add_argument("--archive_stride", type=int, default=1)   # keep every k-th step in the archive
    ap.add_argument("--coh_k", type=int, default=8)
    ap.add_argument("--w_bond", type=float, default=0.0)      # one-sided bond-stretch bound
    ap.add_argument("--bond_s0", type=float, default=0.3)
    ap.add_argument("--vol_frontier", action="store_true")    # frontier-restricted D_vol
    ap.add_argument("--warm_start", action="store_true",       # decayed control warm start
                    help="init each window's control from the previous window's (decayed; "
                         "projected onto the basis) - continuity across windows for the "
                         "measured window-locked limit cycle")
    ap.add_argument("--grad_project_mode", default="render",
                    choices=["render", "phys", "cagrad", "blend"])
    ap.add_argument("--cagrad_c", type=float, default=0.5)
    ap.add_argument("--render_gs_cheb", action="store_true")
    ap.add_argument("--gauss_robust_eps", type=float, default=0.0)
    ap.add_argument("--loss_units", default="legacy", choices=["legacy", "density"])
    ap.add_argument("--dvol_form", default="log", choices=["log", "linear"],
                    help="density-unit residual: log(1+m/m_ref) (default) or linear (m-m_t)/m_ref")
    ap.add_argument("--sample", default="volume", choices=["volume", "shell"],
                    help="particle sampling: uniform volume (default) or the C++ oracle's shell-biased scheme")
    ap.add_argument("--shell_ratio", type=float, default=6.0, help="interior / shell particle spacing (C++: 6)")
    ap.add_argument("--shell_cells", type=float, default=2.0, help="shell thickness in MPM cells (C++: 2)")
    ap.add_argument("--ppc", type=float, default=0.0,
                    help=">0: derive dx/grid/loss_res/sigma from N and the source volume "
                         "for this particles-per-cell (docs/render_controls_physics.md §7)")
    ap.add_argument("--cell_diag", type=float, default=0.0,
                    help=">0: the MPM cell from the SHAPE, dx = source bbox diagonal / cell_diag, "
                         "ppc = N dx^3 / V (docs/method.md 10.9; 26 = the finest fracture-free cell)")
    ap.add_argument("--gate_lo", type=float, default=0.0,   # support-gated APIC (Yao-Zhao 2026):
                    help="omega = smoothstep((n/n0 - lo)/(hi - lo)) on the P2G affine term")
    ap.add_argument("--gate_hi", type=float, default=0.0)   # hi <= lo = off (plain APIC)
    ap.add_argument("--w_jvol", type=float, default=50.0)  # h12 ladder: detFmin
                                        # 0.0005->0.497, |J-1|>0.3 13.7->0.0%,
                                        # chamfer/silIoU best-ever (docs 2026-09-02)
    ap.add_argument("--dfc_clip", type=float, default=0.02)
    ap.add_argument("--arms", default="phys,render")
    ap.add_argument("--save_F_stride", type=int, default=0,
                    help="save every k-th F frame (0 = T, i.e. commit boundaries)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="output/v2_ab")
    ap.add_argument("--grad_dump", default="", help="directory for the per-window gradient-stage dumps (probes/grad_stage.py)")
    ap.add_argument("--layer_relax", action="store_true",
                    help="outer-layer relaxation projection in the forward model (docs/surface_gradient.md §6)")
    ap.add_argument("--layer_frac", type=float, default=0.0,
                    help="projection fraction per step (0 = 1/T over one window; 1 = hard per-step constraint)")
    ap.add_argument("--layer_ctrl", action="store_true",
                    help="position-mode control channel on the outer layer (docs/surface_gradient.md §7)")
    ap.add_argument("--layer_F", action="store_true",
                    help="P3: the u channel through the deformation gradient (F <- (I + grad delta) F on the layer; docs/final_plan.md)")
    ap.add_argument("--layer_F_depth", type=float, default=1.0,
                    help="P3: depth of the normal extension in spacings; 0 = tangential gradient only")
    ap.add_argument("--layer_gate", action="store_true",
                    help="P2: u acts only where the particle-scale density residual exceeds the sampling floor (docs/surface_gradient.md 12)")
    ap.add_argument("--layer_gate_geom", action="store_true",
                    help="the geometric gate: u acts only where the target's outer layer is within --layer_gate_geom_cells MPM cells (docs/surface_gradient.md 15)")
    ap.add_argument("--layer_gate_geom_cells", type=float, default=1.0,
                    help="radius of the geometric gate in MPM cells (the grid's resolution)")
    ap.add_argument("--rest_commit", action="store_true",
                    help="windows from rest: v and C zeroed at accepted commits once the transport has arrived "
                         "(config.rest_commit / rest_commit_gate; docs/method.md 10.21)")
    ap.add_argument("--rest_commit_gate", type=float, default=1.0,
                    help="the u transport gate fraction from which windows start from rest (latched); 0 = every commit")
    ap.add_argument("--rebound_probe", action="store_true",
                    help="diagnostic: a zero-control rollout from every accepted commit, its displacement projected on "
                         "the committed one (the elastic rebound fraction); config.rebound_probe")
    ap.add_argument("--commit_pic", action="store_true",
                    help="project each window's displacement onto the grid-representable subspace at the commit "
                         "(the null-space / XPIC filter once per window; config.commit_pic; docs/method.md 10.20)")
    ap.add_argument("--pace_project", action="store_true",
                    help="the support-preserving paced target: the paced step projected onto the divergence-free "
                         "fields on the body before the window target is rasterised (config.pace_project; "
                         "docs/method.md 10.22)")
    ap.add_argument("--plan_native", action="store_true",
                    help="with --disc_ref: the OT plan's blur from the native target spacing (the sample-derived "
                         "formula is N-independent already; config.plan_native; docs/method.md 10.17a correction)")
    ap.add_argument("--ctrl_rprop", action="store_true",
                    help="per-particle Rprop on the control step: halve a particle's step when its window displacement "
                         "reversed the previous accepted one, raise it x1.2 (to 1) when it kept its direction, no floor "
                         "(config.ctrl_rprop; docs/method.md 10.24)")
    ap.add_argument("--ctrl_rprop_smooth", action="store_true",
                    help="with --ctrl_rprop: the reversal read and the scale applied on the material neighbourhood (config.ctrl_rprop_smooth)")
    ap.add_argument("--ctrl_rprop_k", type=int, default=0,
                    help="with --ctrl_rprop_smooth: the smoothing kNN (0 = the coherence neighbourhood; creg_k = 8 the regulariser's)")
    ap.add_argument("--ctrl_rprop_arrived", action="store_true",
                    help="with --ctrl_rprop: halve only ARRIVED particles' steps (the paced target's per-particle arrival mask); config.ctrl_rprop_arrived")
    ap.add_argument("--ctrl_rprop_hold", action="store_true",
                    help="with --ctrl_rprop: the global step (alpha, the anneal) never grows while the rule is on (config.ctrl_rprop_hold)")
    ap.add_argument("--ctrl_rprop_hold_onset", action="store_true",
                    help="hold the global step only from the alternation's onset (config.ctrl_rprop_hold_onset; with --ctrl_rprop)")
    ap.add_argument("--freeze_arrived", action="store_true",
                    help="freeze particles that have arrived and reversed twice: full plastic assimilation, control and u "
                         "zeroed, velocity zeroed at commits (config.freeze_arrived; needs --ctrl_rprop)")
    ap.add_argument("--settle_eta", action="store_true",
                    help="the settled body's viscosity: arrived, twice-reversed particles get eta = 1/(T dt) in the rollout (config.settle_eta; needs --ctrl_rprop)")
    ap.add_argument("--settle_pin", action="store_true",
                    help="pin arrived, twice-reversed particles inside the rollout (no motion at all; config.settle_pin; needs --ctrl_rprop)")
    ap.add_argument("--settle_pin_clear", action="store_true",
                    help="with --settle_pin: pin only where no unarrived particle lies within the pace radius (config.settle_pin_clear)")
    ap.add_argument("--settle_pin_ray", action="store_true",
                    help="with --settle_pin: pin only where no unarrived particle's plan ray passes within the pace radius (config.settle_pin_ray)")
    ap.add_argument("--arrive_cap", action="store_true",
                    help="the arrival snap respects the target's capacity (config.arrive_cap; method.md 10.28)")
    ap.add_argument("--settle_pin_follow", action="store_true",
                    help="with --settle_pin: release a pinned particle whose plan image moved beyond the pace radius, re-pin when arrived (config.settle_pin_follow)")
    ap.add_argument("--settle_pin_kkt", action="store_true",
                    help="with --settle_pin: release a pinned particle whose cell-sum gradient exceeds the free median (config.settle_pin_kkt)")
    ap.add_argument("--settle_pin_slip", action="store_true",
                    help="with --settle_pin: the pinned body is a grid-level separating (slip) collider — pinned mass out of the momentum average (config.settle_pin_slip)")
    ap.add_argument("--settle_pin_yield", action="store_true",
                    help="with --settle_pin_ray: release settled particles within the kernel support of a transit ray for the window (config.settle_pin_yield)")
    ap.add_argument("--settle_pin_assim", action="store_true",
                    help="with --settle_pin: assimilate the elastic stretch in full at pin time (stress-free pinned body; config.settle_pin_assim)")
    ap.add_argument("--settle_commit", action="store_true",
                    help="with --rest_commit: the accepted state settled by one zero-control window under that viscosity before the next window (config.settle_commit)")
    ap.add_argument("--u_rprop_floor", type=float, default=0.05,
                    help="the u channel's Rprop floor (config.u_rprop_floor); 0 removes it")
    ap.add_argument("--outer_latch_reversal", action="store_true",
                    help="arm the outer merit gate's low-gain reversal reject at the alternation's onset (the second "
                         "accepted commit in a row reversing the previous one) and keep it armed "
                         "(config.outer_latch_reversal; docs/method.md 10.21 addendum 3)")
    ap.add_argument("--rest_commit_reversal", action="store_true",
                    help="with --rest_commit: latch windows from rest at the second accepted commit in a row whose "
                         "displacement reverses the previous one (config.rest_commit_reversal; docs/method.md 10.21)")
    ap.add_argument("--u_rprop", action="store_true",
                    help="damp the u channel's per-window bound per particle by its sign history (Rprop 0.5 / 1.2; "
                         "config.u_rprop; docs/method.md 10.19)")
    ap.add_argument("--stop_on_cycle", action="store_true",
                    help="converge when the net displacement over the last --patience windows is no more than a "
                         "random walk's (median net / summed <= 1/sqrt(patience)); the tail then holds "
                         "(config.stop_on_cycle; docs/oscillation.md Addendum 9)")
    ap.add_argument("--shift_sub", action="store_true",
                    help="Fickian shifting of the sub-cell particle arrangement at every window commit "
                         "(config.shift_sub; docs/method.md 10.18): dx = -1/2 h^2 grad C, positions only, "
                         "the outer layer tangentially")
    ap.add_argument("--shift_h_sp", type=float, default=1.0, help="shifting kernel width in native spacings (Gaussian; 1 = the SPH smoothing length)")
    ap.add_argument("--disc_ref", action="store_true",
                    help="the reference discretisation: every spacing-derived length and every neighbour count of the "
                         "pipeline at the reference spacing s (N / mass_ref_n)^(1/3) and count x N / mass_ref_n "
                         "(config.disc_ref); at N <= mass_ref_n a no-op")
    ap.add_argument("--layer_gate_ot", action="store_true",
                    help="the transport gate: u acts only where the remaining transport to the OT image is within --layer_gate_ot_cells MPM cells (docs/surface_gradient.md 15)")
    ap.add_argument("--layer_gate_ot_cells", type=float, default=1.0,
                    help="radius of the transport gate in MPM cells")
    ap.add_argument("--layer_gate_ot_normal", action="store_true",
                    help="transport gate on the NORMAL component of the remaining transport only (docs/surface_gradient.md 15e)")
    ap.add_argument("--layer_u_render_only", action="store_true",
                    help="P1: the u channel driven by the render channel only (docs/surface_gradient.md 13)")
    ap.add_argument("--sil_kernel", default="cic", choices=["cic", "quad"],
                    help="splat kernel of the silhouette/shading rasterisers (quad = quadratic B-spline, C1)")
    ap.add_argument("--sampler", default="replacement", choices=["replacement", "stratified"],
                    help="particle sampling of source and target: with replacement (the v8 sampler) or one "
                         "jittered particle per fill voxel (G5, docs/surface_gradient.md §4)")
    ap.add_argument("--layer_ctrl_smooth", action="store_true",
                    help="project the u channel's step onto the layer's smooth subspace (the relaxation's W)")
    ap.add_argument("--pbr_denoised", action="store_true",
                    help="G1: shading target from the target's reconstructed surface; morph normals on the pixel grid")
    args = ap.parse_args()

    src, v_src = load(args.src, args.n, args.seed, return_volume=True, sample=args.sampler)
    tgt, v_tgt = load(args.tgt, args.n, args.seed + 1, match_volume=v_src, sample=args.sampler,
                      return_volume=True)
    print(f"[v2run] volumes: source {v_src:.2f} target(matched) {v_tgt:.2f} wu^3 "
          f"(target bbox diag now {float(np.linalg.norm(tgt.max(0) - tgt.min(0))):.2f})",
          flush=True)
    prm = MPMParams()
    if args.cell_diag > 0:
        # discretisation contract v2 (2026-09-17, docs/method.md 10.9): the MPM cell follows
        # the SHAPE — dx = source bbox diagonal / cell_diag — and the particles per cell
        # follow N (ppc = N dx^3 / V). The ladders showed the ejection variable is the cell
        # size relative to the shape (dx 0.20 wu fractures at ppc 8 and 27 alike; 0.31 holds
        # at ppc 27 and 91 alike); 26 = the finest fracture-free cell measured (0.31 wu on
        # the 8 wu normalisation). N then refines the quadrature inside a geometry-set grid.
        diag_src = float(np.linalg.norm(src.max(0) - src.min(0)))
        dx_req = diag_src / float(args.cell_diag)
        args.ppc = float(args.n * dx_req ** 3 / v_src)
        print(f"[disc] cell from the shape: dx = diag {diag_src:.3f} / {args.cell_diag:g} = {dx_req:.4f} wu "
              f"-> ppc = N dx^3 / V = {args.ppc:.1f}", flush=True)
    if args.ppc > 0:                       # discretisation contract: dx follows N
        from physmorph.mpm.discretisation import derive, report
        mat = PipelineConfig(young=args.young, poisson=args.poisson)   # the material the arms actually use
        domain_half = -prm.grid_min[0]
        if args.domain == "auto":
            # the leash box the objective already assumes (runner.build_target: extent =
            # 1.25 x max|target|; w_box pulls particles back inside it) + the 4^3 stencil
            # margin: nothing outside it receives grid forces by design
            dx0 = float((v_src * args.ppc / args.n) ** (1.0 / 3.0))
            leash = 1.25 * float(max(np.abs(src).max(), np.abs(tgt).max()))
            domain_half = leash + 2.0 * dx0
        # the dynamics mass per particle is mass_ref_n / N (config.mass_ref_n, method.md 10.17), so the
        # printed density, sound speed and CFL must use it — with the unit mass the diagnostic was off
        # by N / mass_ref_n at every N != 40k (the independent audit, docs/diagnosis_300k_20260923.md)
        _mref = int(PipelineConfig().mass_ref_n or 0)
        _mass = (_mref / float(args.n)) if (_mref > 0 and args.n != _mref) else 1.0
        disc = derive(args.n, v_src, float(np.linalg.norm(src.max(0) - src.min(0))),
                      prm.dt, mat.young, mat.poisson, ppc=args.ppc,
                      domain_half=domain_half, mass=_mass)
        prm = dataclasses.replace(prm, dx=disc.dx, nx=disc.grid_n, ny=disc.grid_n,
                                  nz=disc.grid_n, grid_min=(disc.grid_min,) * 3)
        if args.domain == "auto":
            # the density-unit calibration measures the legacy ratio on a 0.5 wu reference
            # cell (the grid every legacy weight was tuned on); keep that cell size
            args.unit_ref_res = int(round(2.0 * domain_half / 0.5))
            print(f"[disc] domain auto: half-width {domain_half:.2f} wu (leash {leash:.2f} + 2 dx), "
                  f"grid {disc.grid_n}^3 = {disc.grid_n**3/1e6:.2f} M cells (was {int(np.ceil(32.0/disc.dx))}^3), "
                  f"unit_ref_res {args.unit_ref_res} (0.5 wu reference cell)", flush=True)
        # REFUTE F4 (2026-09-15): the legacy D_vol is a CELL SUM, so letting loss_res
        # follow dx (32 -> 109 at 20k/ppc 8) multiplied it ~500x against every fixed
        # weight. The loss grid follows the MPM cell only in density units, which are
        # resolution-invariant by construction; in legacy units loss_res is untouched.
        if args.loss_units == "density":
            args.loss_res = disc.loss_res
        print(report(disc, src), flush=True)
        print(report(disc, tgt).splitlines()[-1].replace("[disc] measured", "[disc] TARGET measured"),
              flush=True)
        print(f"[disc] loss_res {'follows dx: ' + str(disc.loss_res) if args.loss_units == 'density' else 'kept at ' + str(args.loss_res) + ' (legacy units are a cell sum)'}",
              flush=True)
    w_src = w_tgt = None
    if args.sample == "shell":
        # C++ oracle sampling (LoadShellBiasedMPMPointCloudFromObj): a surface shell of
        # shell_cells MPM cells sampled at spacing h_s, the interior at ratio x h_s; the
        # particle masses follow the rest volumes so the density stays uniform. The cell then
        # holds (dx / h_s)^3 particles in the shell — the decoupling gap in surface spacings.
        thick = float(args.shell_cells) * float(prm.dx)
        src, v_src, w_src = load(args.src, args.n, args.seed, return_volume=True,
                                 shell=(args.shell_ratio, thick))
        tgt, v_tgt, w_tgt = load(args.tgt, args.n, args.seed + 1, match_volume=v_src,
                                 return_volume=True, shell=(args.shell_ratio, thick))
        from scipy.spatial import cKDTree
        h_s = float(np.median(cKDTree(src).query(src, k=2, workers=-1)[0][:, 1]))
        print(f"[v2run] shell-biased sampling: shell {args.shell_cells} cells = {thick:.3f} wu, "
              f"interior/shell spacing ratio {args.shell_ratio}; source median NN spacing {h_s:.4f} wu "
              f"-> cell = {prm.dx / h_s:.2f} shell spacings", flush=True)
    if args.v_max > 0:                     # forward model: G2P speed cap (ejection ladder 2026-09-16)
        prm = dataclasses.replace(prm, v_max=args.v_max)
        print(f"[v2run] G2P speed cap v_max={args.v_max} wu/s (40k archives: body p95 0.25-0.28 wu/s, "
              f"early max 2.6-2.8; ejected particles 5-6 wu/s)", flush=True)
    if args.gate_hi > args.gate_lo:        # forward model: support-gated APIC
        prm = dataclasses.replace(prm, gate_r_lo=args.gate_lo, gate_r_hi=args.gate_hi)
        print(f"[v2run] support-gated APIC on: r_lo={args.gate_lo} r_hi={args.gate_hi} "
              f"(n0 = source median 3^3-cell count, set by the runner)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"[v2run] {args.src} -> {args.tgt}  N={args.n}  T={args.T}  iters={args.iters}  "
          f"anims={args.animations} | dx={prm.dx} dt={prm.dt:.5f} smoothing={prm.smoothing}",
          flush=True)
    print(f"[v2run] baseline chamfer (undeformed) = {metrics.chamfer(src, tgt):.4f}", flush=True)

    live = None
    if args.live_port:
        from physmorph.viewer.server import LiveServer
        live = LiveServer(args.live_port)
    live_dir = Path(args.live_dir) if args.live_dir else None

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
    if git_sha is None:                       # tarball deploys carry the sha in VERSION
        vf = Path(__file__).resolve().parent.parent / "VERSION"
        git_sha = vf.read_text().strip() if vf.exists() else None
    out = {"provenance": {**vars(args), "mpm": dataclasses.asdict(prm),
                           "git_sha": git_sha, "code_hash": code_hash},   # AGENTS rule 4:
           "G1a": gate1_plumbing(src, prm),                                # discretisation
           "G1b": gate1_channels(src, prm), "arms": {}}                    # travels with numbers

    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        cfg = arm_config(arm, args)
        if getattr(args, "unit_ref_res", 0):
            # --domain auto: the density-unit calibration keeps its 0.5 wu reference cell
            # (bug 2026-09-16: the auto block set args.unit_ref_res but cfg kept 64 over the
            # smaller box -> a 0.2 wu reference cell and weights ~3x off vs the fixed domain)
            cfg.unit_ref_res = int(args.unit_ref_res)
        # snapshot BEFORE the run: c2f mutates cfg.render_res mid-run (the archived
        # config must record what the run STARTED with; the c2f switch is in history)
        cfg_dump = {k: v for k, v in dataclasses.asdict(cfg).items() if k != "history"}
        print(f"\n[v2run] ===== ARM {arm} =====", flush=True)
        t0 = time.time()
        cbs = (None, None)
        sink = live
        if live_dir is not None:           # file-backed sink (viewer_serve.py reads it)
            from physmorph.viewer.server import LiveServer
            sink = LiveServer.to_dir(live_dir / f"{Path(args.out).name}_{arm}")
        if sink is not None:
            from physmorph.render.covariance import sigma0_from_nn
            cbs = sink.begin_run(arm, src, tgt, prm, cfg, sigma0_from_nn(tgt, 0.9))
        res = run_pipeline(src, tgt, prm, cfg, on_commit=cbs[0], on_iter=cbs[1],
                           w_src=w_src, w_tgt=w_tgt)
        dt = time.time() - t0
        dn = res.get("deliver_n") or len(res["frames"])   # metrics on the DELIVERED slice
        res["deliver_n_used"] = dn
        met = metrics.summarize(res["frames"][:dn], tgt, F_frames=res["F_frames"][:dn],
                                n_held=res["n_held"], render_mask=res.get("render_mask"),
                                window=max(int(args.T) - 1, 1))
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
        from physmorph.sampling.orientation import orient_name as _orient_name
        # 2026-09-23 (speed): zlib on a 300k x 1000-frame stack is minutes of single-thread CPU at
        # the end of the run (17 s for 100 frames); above 100k particles the archive is written
        # uncompressed (float32 positions compress poorly anyway; disk is not the constraint)
        _saver = np.savez if len(src) >= 100000 else np.savez_compressed
        _saver(
            f"{args.out}_{arm}.npz", src=src, tgt=tgt,
            orient=np.str_(_orient_name(args.tgt)),        # the loader already rotated the asset to y-up
            frames=np.stack(res["frames"]), deliver_n=np.int64(dn),
            pinned=np.asarray(res["pinned"] if res.get("pinned") is not None else np.zeros(len(src), bool), bool),
            pinned_at=np.asarray(res["pinned_at"] if res.get("pinned_at") is not None else np.full(len(src), -1), np.int32),
            gx_last=(np.asarray(res["gx_last"], np.float32) if res.get("gx_last") is not None else np.zeros((0, 3), np.float32)),
            Fp=(np.asarray(res["Fp"], np.float32).reshape(-1, 3, 3) if res.get("Fp") is not None else np.zeros((0, 3, 3), np.float32)),
            truncation=json.dumps(res.get("truncation")),
            F_samples=np.stack([res["F_frames"][i] for i in idx]),
            F_sample_idx=np.array(idx),
            render_mask=(res["render_mask"] if res.get("render_mask") is not None
                          else np.ones(len(src), bool)),
            s=res["s"] if res["s"] is not None else np.zeros(0, np.float32),
            # F_g at accepted commits, DELIVERED slice only (REFUTE F10) — the same
            # per-commit cadence as F_samples at the default stride
            Fg_commit_idx=np.array([i for i, _ in res.get("Fg_commits", []) if i <= dn],
                                   np.int64),
            Fg_commits=(np.stack([f for i, f in res["Fg_commits"] if i <= dn])
                        if any(i <= dn for i, _ in res.get("Fg_commits", []))
                        else np.zeros((0, 0, 3, 3), np.float32)),
            **archive_extra)
        out["arms"][arm] = {"config": cfg_dump, "metrics": met,
                            "gates": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                                      for k, v in gates.items()},
                            "guards": res["guards"], "converged": res["converged"],
                            "balancer": res.get("balancer"), "deliver_n": int(dn),
                            "truncation": res.get("truncation"),
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
