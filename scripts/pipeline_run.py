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
                         render_res=args.render_res, loss_res=args.loss_res)
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
        cfg.anneal_stale = args.anneal
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
        cfg.w_fill = args.w_fill
        cfg.w_jvol = args.w_jvol
        cfg.w_grow = args.w_grow
        cfg.assim_iso = True
    elif arm == "render_full_gauss":               # flagship with the REAL 3DGS loss
        cfg.lambda_auto = args.lambda_auto         # replacing the CIC soft-silhouette
        cfg.w_pbr = 0.0
        cfg.grad_project = False
        cfg.c2f_at = 0.0                           # gauss targets are res-fixed for now
        cfg.pace = args.pace
        cfg.dfc_clip = args.dfc_clip
        cfg.w_creg = args.w_creg
        cfg.w_dt = args.w_dt
        cfg.w_nn = args.w_nn
        cfg.w_jvol = args.w_jvol
        cfg.gauss_mix = args.gauss_mix
        cfg.anneal_stale = args.anneal
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
    ap.add_argument("--lambda_auto", type=float, default=0.5)
    ap.add_argument("--w_kin", type=float, default=0.5)
    ap.add_argument("--w_ctrl", type=float, default=1e-3)
    ap.add_argument("--w_box", type=float, default=10.0)
    ap.add_argument("--assim", type=float, default=0.5)
    ap.add_argument("--render_views", type=int, default=6)
    ap.add_argument("--render_res", type=int, default=64)
    ap.add_argument("--loss_res", type=int, default=32)
    ap.add_argument("--render_gs_iters", type=int, default=20)
    ap.add_argument("--w_pbr", type=float, default=1.0)
    ap.add_argument("--pace", type=float, default=0.12)
    ap.add_argument("--lg_sweeps", type=int, default=8)
    ap.add_argument("--w_creg", type=float, default=100.0)
    ap.add_argument("--w_dt", type=float, default=0.2)   # SUM form: per-particle pull
                                                         # = w_dt (Opus parity estimate)
    ap.add_argument("--w_fill", type=float, default=0.1)  # fill v3 ALPHA (norm-balanced)
    ap.add_argument("--w_grow", type=float, default=0.02)
    ap.add_argument("--grow_band", type=float, default=1.5)
    ap.add_argument("--gauss_res", type=int, default=96)
    ap.add_argument("--anneal", type=float, default=0.0)  # plateau step decay
    ap.add_argument("--gauss_mix", type=float, default=0.0)  # hybrid sil+gauss render
    ap.add_argument("--w_nn", type=float, default=0.2)
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

    out = {"provenance": {**vars(args), "mpm": dataclasses.asdict(prm)},   # AGENTS rule 4:
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
            cbs = live.begin_run(arm, src, tgt, prm, cfg, sigma0_from_nn(src, 0.7))
        res = run_pipeline(src, tgt, prm, cfg, on_commit=cbs[0], on_iter=cbs[1])
        dt = time.time() - t0
        met = metrics.summarize(res["frames"], tgt, F_frames=res["F_frames"],
                                n_held=res["n_held"])
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
        np.savez_compressed(
            f"{args.out}_{arm}.npz", src=src, tgt=tgt,
            frames=np.stack(res["frames"]),
            F_samples=np.stack([res["F_frames"][i] for i in idx]),
            F_sample_idx=np.array(idx),
            s=res["s"] if res["s"] is not None else np.zeros(0, np.float32))
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
