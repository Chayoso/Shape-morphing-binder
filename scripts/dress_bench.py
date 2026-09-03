"""Stage 0b — GPU microbenchmark for the Tier D dressing solve (design v2 §10).

Measures the global commit p50 (real optimize_window calls on a warm state) vs
the dressing-solve p50/p95 at iters {5,10,20} at production res/children/N, and
prints the pre-registered selection: the largest budget whose p50 overhead is
<= 50% of the global commit p50. If even 5 exceeds the bound, the arm is VOID.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from physmorph.mpm import MPMParams  # noqa: E402
from physmorph.pipeline import PipelineConfig  # noqa: E402
from physmorph.pipeline.dressing import DressState, solve_dressing  # noqa: E402
from physmorph.pipeline.render_loss import LambdaBalancer  # noqa: E402
from physmorph.pipeline.runner import (_surface_weights, build_target)  # noqa: E402
from physmorph.pipeline.optimizer import optimize_window  # noqa: E402
from physmorph.sampling import load_normalized as load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--commits", type=int, default=6)
    ap.add_argument("--gauss_res", type=int, default=96)
    args = ap.parse_args()

    prm = MPMParams()
    src = load("assets/isosphere.obj", args.n, 1)
    tgt_x = load("assets/bunny.obj", args.n, 2)
    cfg = PipelineConfig(T=20, iters=8, animations=8, loss_res=64)
    cfg.lambda_auto, cfg.w_kin = 0.5, 5.0
    cfg.w_dt, cfg.w_nn, cfg.w_jvol = 0.2, 0.2, 50.0
    cfg.use_gauss_loss, cfg.gauss_res = True, args.gauss_res
    cfg.gauss_mix, cfg.gauss_children = 0.25, 4
    cfg.render_surface_only, cfg.surface_grad_frac = True, 0.50
    cfg.assim_iso = True
    tgt = build_target(tgt_x, prm, cfg)
    sw = _surface_weights(src, cfg.surface_grad_k, cfg.surface_grad_frac,
                          cfg.surface_grad_floor)
    sw = np.ascontiguousarray(sw > 0.5, np.float32)
    tgt.gauss.configure_source(src, sw > 0.5)
    print(f"[bench] N={args.n} gauss res={tgt.gauss.res} children=4 "
          f"views={len(tgt.gauss.cams)}")

    balancer = LambdaBalancer(cfg.lambda_auto, cfg.lambda_ema, cfg.lambda_cap)
    x, st, times = src.copy(), {"F": None, "v": None, "C": None}, []
    Fp = np.tile(np.eye(3, dtype=np.float32), (args.n, 1, 1))
    Fc = None
    for c in range(args.commits):
        t0 = time.perf_counter()
        fr, F_seq, end, _s, whist, stats = optimize_window(
            x, prm, cfg, tgt, balancer, F0=st["F"], Fp=Fp, v0=st["v"], C0=st["C"],
            log=lambda *_: None, surface_w=sw)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if whist:
            x = np.ascontiguousarray(fr[-1], np.float32)
            st = {"F": end["F"], "v": end["v"], "C": end["C"]}
            Fc = end["F"]
            times.append(dt)
        print(f"[bench] commit {c}: {dt:.2f}s accepted={bool(whist)}")
    g50 = float(np.median(times))
    print(f"[bench] global commit p50 = {g50:.2f}s over {len(times)} accepted")

    picked = 0
    for iters in (5, 10, 20):
        ds = DressState(tgt.gauss, src, sw > 0.5, cfg.dress_cap_frac, cfg.device)
        samples = []
        for _ in range(5):
            ds.coeff += 0.0   # keep state; each solve continues from previous
            t0 = time.perf_counter()
            tele = solve_dressing(ds, x, Fc, iters, cfg.ls_noise_rel)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)
        p50, p95 = np.percentile(samples, 50), np.percentile(samples, 95)
        ok = p50 <= 0.5 * g50
        print(f"[bench] iters={iters}: p50={p50:.2f}s p95={p95:.2f}s "
              f"overhead={100 * p50 / g50:.0f}% used={tele['dress_iters_used']} "
              f"dL={tele['dress_dL']:.3g} -> {'OK' if ok else 'OVER'}")
        if ok:
            picked = iters
    print(f"[bench] SELECTED local_dress_iters={picked}"
          + ("" if picked else "  (ARM VOID: even 5 exceeds the 50% bound)"))


if __name__ == "__main__":
    main()
