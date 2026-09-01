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
        cfg.opt_material = False
    elif arm == "render":
        cfg.lambda_auto = args.lambda_auto
        cfg.opt_material = False
    elif arm == "render_mat":
        cfg.lambda_auto = args.lambda_auto
        cfg.opt_material = True
    else:
        raise SystemExit(f"unknown arm {arm!r} (phys|render|render_mat)")
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
        "G4_holes_abs": met["hole_frac"] <= hole_tol,       # spec: absolute 2% (AND <= phys arm,
    }                                                       # evaluated cross-arm below)
    print(f"[{tag}] gates: " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}"
                                         for k, v in gates.items()) +
          f"   (guards={g}, jitter_rel={met['jitter_rel']:.5f}, drift_rel={drift_rel:.5f}, "
          f"hole={met['hole_frac']*100:.2f}% tgt={met['hole_frac_tgt']*100:.2f}%, "
          f"outside={met['outside_frac']*100:.3f}%)", flush=True)
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

    out = {"provenance": {**vars(args), "mpm": dataclasses.asdict(prm)},   # AGENTS rule 4:
           "G1a": gate1_plumbing(src, prm),                                # discretisation
           "G1b": gate1_channels(src, prm), "arms": {}}                    # travels with numbers

    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        cfg = arm_config(arm, args)
        print(f"\n[v2run] ===== ARM {arm} =====", flush=True)
        t0 = time.time()
        res = run_pipeline(src, tgt, prm, cfg)
        dt = time.time() - t0
        met = metrics.summarize(res["frames"], tgt, F_frames=res["F_frames"],
                                n_held=res["n_held"])
        gates = eval_gates(arm, res, met, prm, args.T)
        stride = args.save_F_stride if args.save_F_stride > 0 else args.T
        nF = len(res["F_frames"])
        idx = sorted(set(range(0, nF, stride)) | {nF - 1})
        cfg_dump = {k: v for k, v in dataclasses.asdict(cfg).items() if k != "history"}
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
              f"detFmin={met.get('detF_min', 1.0):.4f}  ({dt/60:.1f} min)", flush=True)

    # ---- cross-arm gates: G4 second half + G5 supremacy (render vs phys) ----
    if "phys" in out["arms"] and "render" in out["arms"]:
        mp = out["arms"]["phys"]["metrics"]
        mr = out["arms"]["render"]["metrics"]
        g4_vs = mr["hole_frac"] <= mp["hole_frac"] + 1e-9
        g5 = {"sil_iou_up": mr["sil_iou"] > mp["sil_iou"],
              "chamfer_ok": mr["chamfer"] <= mp["chamfer"] * 1.02,
              "holes_down": g4_vs}
        out["G4_vs_phys"] = bool(g4_vs)
        out["G5"] = {**{k: bool(v) for k, v in g5.items()}, "pass": bool(all(g5.values()))}
        print(f"\n[G4b] render holes <= phys: {mp['hole_frac']*100:.2f}% -> "
              f"{mr['hole_frac']*100:.2f}%  {'PASS' if g4_vs else 'FAIL'}", flush=True)
        print(f"[G5] render vs phys: silIoU {mp['sil_iou']:.4f}->{mr['sil_iou']:.4f} "
              f"chamfer {mp['chamfer']:.4f}->{mr['chamfer']:.4f}  -> "
              f"{'PASS' if out['G5']['pass'] else 'FAIL'}", flush=True)

    Path(f"{args.out}.json").write_text(json.dumps(out))
    print(f"\nsaved {args.out}.json", flush=True)


if __name__ == "__main__":
    main()
