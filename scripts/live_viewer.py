"""Live viewer server for the MAIN LINE (dynamic elasto + render-adjoint pipeline).

The sim runs HERE (hyde06); the browser watches through an SSH tunnel. Streams:
  * every ACCEPTED optimiser iteration (phase 'iter'): x_T, F_T + window telemetry —
    the optimisation process itself;
  * every promoted COMMIT (phase 'commit'): full telemetry + the loss-grid physical
    distributions (per-node mass and mass-weighted |v| on the D_vol grid) for the
    viewer's grid layer/histogram.

Zero dependencies beyond the repo (stdlib HTTP, binary /state).

Run (hyde06, free GPU):
  CUDA_VISIBLE_DEVICES=1 setsid nohup python scripts/live_viewer.py \
      --n 20000 --port 8765 > output/live_viewer.log 2>&1 < /dev/null &
Local machine:
  ssh -N -L 8765:localhost:8765 -J chayo@hyde01.dabh.io chayo@hyde06.dabh.io
  -> open http://localhost:8765
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

from physmorph.mpm import MPMParams  # noqa: E402
from physmorph import metrics  # noqa: E402
from physmorph.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from physmorph.render.covariance import sigma0_from_nn  # noqa: E402
from physmorph.render.children import tangent_child_offsets  # noqa: E402
from physmorph.sampling import load_normalized  # noqa: E402
from physmorph.viewer.server import LiveServer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="assets/isosphere.obj")
    ap.add_argument("--tgt", default="assets/bunny.obj")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--animations", type=int, default=300)
    ap.add_argument("--lambda_auto", type=float, default=0.5)
    ap.add_argument("--w_kin", type=float, default=20.0)
    ap.add_argument("--render_views", type=int, default=6)
    ap.add_argument("--render_res", type=int, default=64)
    ap.add_argument("--loss_res", type=int, default=32)
    ap.add_argument("--gauss_res", type=int, default=128)
    ap.add_argument("--gauss_sigma_scale", type=float, default=1.0)
    ap.add_argument("--gauss_children", type=int, default=4)
    ap.add_argument("--gauss_child_sigma_scale", type=float, default=0.55)
    ap.add_argument("--gauss_child_offset_scale", type=float, default=0.35)
    ap.add_argument("--surface_frac", type=float, default=0.50)
    ap.add_argument("--outer_gate_move_frac", type=float, default=0.006,
                    help="latch the fixed-merit outer gate below this move/extent ratio")
    ap.add_argument("--outer_gate_merit_max", type=float, default=0.55,
                    help="also require normalized fixed merit below this value before latching")
    ap.add_argument("--loop", action="store_true")     # default: ONE run, then hold
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="output/live_stable")
    args = ap.parse_args()

    src = load_normalized(args.src, args.n, args.seed)
    tgt = load_normalized(args.tgt, args.n, args.seed + 1)
    prm = MPMParams()
    cfg = PipelineConfig(T=args.T, iters=args.iters, animations=args.animations,
                         lambda_auto=args.lambda_auto, w_kin=args.w_kin,
                         device=args.device, hold_after_converge=False,
                         render_views=args.render_views, render_res=args.render_res,
                         loss_res=args.loss_res, use_gauss_loss=True,
                          gauss_mix=0.25, gauss_res=args.gauss_res,
                          gauss_sigma_scale=args.gauss_sigma_scale,
                          gauss_children=args.gauss_children,
                          gauss_child_sigma_scale=args.gauss_child_sigma_scale,
                          gauss_child_offset_scale=args.gauss_child_offset_scale,
                         w_dt=0.2, w_nn=0.3, nn_tail_frac=0.0,
                         w_jvol=50.0, w_creg=100.0,
                         dfc_clip=0.02, pace=0.12, assim_iso=True,
                         mom_carry=0.8, anneal_stale=0.5,
                          outer_merit=True, w_tctrl=10.0, w_cov=25.0,
                          outer_gate_move_frac=args.outer_gate_move_frac,
                          outer_gate_merit_max=args.outer_gate_merit_max,
                          surface_grad_frac=args.surface_frac, render_surface_only=True,
                         control_h1_iters=0, patience=8)
    from physmorph.pipeline.runner import _surface_weights
    tgt_mask = _surface_weights(tgt, cfg.surface_grad_k, cfg.surface_grad_frac,
                                cfg.surface_grad_floor) > 0.5
    sigma0 = sigma0_from_nn(tgt[tgt_mask], cfg.gauss_sigma_scale)
    src_mask = (_surface_weights(src, cfg.surface_grad_k, cfg.surface_grad_frac,
                                 cfg.surface_grad_floor) > 0.5)
    src_child_offsets = tangent_child_offsets(
        src, src_mask, sigma0, cfg.gauss_children, cfg.gauss_child_offset_scale,
        cfg.gauss_child_k)
    tgt_child_offsets = tangent_child_offsets(
        tgt, tgt_mask, sigma0, cfg.gauss_children, cfg.gauss_child_offset_scale,
        cfg.gauss_child_k)
    live = LiveServer(args.port)
    run_i = 0
    while True:
        cb_commit, cb_iter = live.begin_run("render_stable_gauss", src, tgt, prm, cfg, sigma0)
        archive_x = [src.copy()]
        archive_F = [np.tile(np.eye(3, dtype=np.float32), (len(src), 1, 1))]
        archive_animation = [-1]

        def on_commit(a, x, F, v, rec):
            cb_commit(a, x, F, v, rec)
            # The deliverable animation is commit-time, not every internal MPM
            # substep.  This preserves every state change while avoiding multi-GB
            # archives at a 300-commit production budget; null states are identical.
            archive_x.append(np.asarray(x, np.float32).copy())
            archive_F.append(np.asarray(F, np.float32).copy())
            archive_animation.append(int(a))

        t0 = time.time()
        res = run_pipeline(src, tgt, prm, cfg, log=lambda *a: print(*a, flush=True),
                           on_commit=on_commit, on_iter=cb_iter)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(out_path) + ".npz", src=src, tgt=tgt,
                            frames=np.stack(archive_x), F_samples=np.stack(archive_F),
                            F_sample_idx=np.arange(len(archive_x), dtype=np.int64),
                            animation=np.asarray(archive_animation, dtype=np.int64),
                            render_mask=(res["render_mask"]
                                         if res.get("render_mask") is not None
                                         else np.ones(len(src), bool)),
                            target_render_mask=tgt_mask,
                            sigma0=np.float32(sigma0),
                            source_child_offsets=src_child_offsets,
                            target_child_offsets=tgt_child_offsets,
                            gauss_child_sigma_scale=np.float32(
                                cfg.gauss_child_sigma_scale
                                if cfg.gauss_children > 1 else 1.0))
        met = metrics.summarize(res["frames"], tgt, F_frames=res["F_frames"],
                                n_held=res["n_held"], render_mask=res.get("render_mask"))
        from physmorph.render.support import MaterialSupport
        support_alpha = MaterialSupport.from_rest(src, 8).opacity(res["frames"][-1])
        visible = np.asarray(res["render_mask"], bool)
        render_diag = {
            "parent_sigma0": float(sigma0),
            "primitive_sigma0": float(sigma0 * (cfg.gauss_child_sigma_scale
                                                  if cfg.gauss_children > 1 else 1.0)),
            "three_sigma_radius": float(3.0 * sigma0 * (cfg.gauss_child_sigma_scale
                                                         if cfg.gauss_children > 1 else 1.0)),
            "children_per_parent": int(cfg.gauss_children),
            "surface_count": int(visible.sum()),
            "support_faded_count": int((visible & (support_alpha < 0.5)).sum()),
            "support_faded_frac": float((support_alpha[visible] < 0.5).mean())}
        (out_path.with_suffix(".json")).write_text(json.dumps({
            "discretization": {"N": len(src), "T": cfg.T, "dt": prm.dt,
                               "dx": prm.dx, "smoothing": prm.smoothing},
            "config": dataclasses.asdict(cfg), "metrics": met,
            "render_diagnostics": render_diag, "guards": res["guards"],
            "history": res["history"]}, indent=2))
        print(f"[live] saved {out_path}.npz/.json", flush=True)
        print(f"[live] run {run_i} finished in {time.time()-t0:.1f}s "
              f"(converged={res['converged']}); "
              f"{'looping' if args.loop else 'holding — POST /restart or the viewer button'}",
              flush=True)
        if not args.loop:
            live.hub.restart.wait()
            live.hub.restart.clear()
        run_i += 1
        time.sleep(1.0)


if __name__ == "__main__":
    main()
