"""Morph a source mesh into a target mesh with settled transport (README.md). CUDA only.

Run on hyde06 (scripts/ops/hyde06_env.sh sets REPO, OUT and PY):
  $PY scripts/pipeline_run.py --tgt assets/bunny.obj --n 300000 --seed 97 --out $OUT/bunny

Outputs: <out>_render_full_dt_iso_nn.npz (the archive the renderers read: frames, the
delivered slice, F samples) and <out>.json (provenance, config, metrics, gates and the
per-window history). The prepare stage (mesh sampling, cached) is the only CPU work.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physmorph import gpu, metrics  # noqa: E402
from physmorph.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from physmorph.surface import surface_roughness  # noqa: E402
from physmorph.thin import thin_metrics, thin_set  # noqa: E402
from physmorph.prepare import prepare  # noqa: E402
from physmorph.sampling.orientation import orient_name  # noqa: E402

# the archive name of the adopted recipe, kept so new archives read like the earlier ones
ARM = "render_full_dt_iso_nn"
TRACKED = ("physmorph/gpu.py", "physmorph/prepare.py", "physmorph/pipeline/config.py",
           "physmorph/pipeline/target.py", "physmorph/pipeline/render_loss.py",
           "physmorph/pipeline/window", "physmorph/pipeline/run", "physmorph/losses",
           "physmorph/mpm/kernels.py", "physmorph/mpm/traj.py", "physmorph/mpm/function.py")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="assets/isosphere.obj")
    ap.add_argument("--tgt", default="assets/bunny.obj")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=97)
    ap.add_argument("--out", default="output/run")
    ap.add_argument("--animations", type=int, default=300, help="window budget")
    ap.add_argument("--patience", type=int, default=5,
                    help="windows without a merit improvement that stop the run")
    ap.add_argument("--reject_stop", type=int, default=3,
                    help="consecutive rejected windows that stop the run at the best window")
    ap.add_argument("--render_weight_scale", type=float, default=1.0,
                    help="multiplies the render weight; 0 = the render-off twin")
    ap.add_argument("--ot_iters", type=int, default=1600, help="Sinkhorn sweep budget per solve")
    ap.add_argument("--support_weight", type=float, default=8.0, help="local support bound weight")
    ap.add_argument("--support_target_ref", action="store_true",
                    help="support floor from the target density at the nearest target point")
    ap.add_argument("--support_form", choices=("log", "ratio", "proximity"), default="log",
                    help="per-particle support penalty: log deficit squared, or missing mass fraction squared")
    ap.add_argument("--loss_follows_n", action="store_true",
                    help="transport grid and blur follow the particle spacing above mass_ref_n")
    ap.add_argument("--cell_diag", type=float, default=26.0,
                    help="the MPM cell from the shape: dx = source bbox diagonal / cell_diag")
    ap.add_argument("--save_F_stride", type=int, default=0,
                    help="archive every k-th frame's F (0 = every T frames)")
    ap.add_argument("--grad_dump", default="", help="directory of per-window gradient dumps")
    ap.add_argument("--ls_probe", action="store_true",
                    help="diagnostic: split every failed line-search trial by control channel")
    ap.add_argument("--live_port", type=int, default=0, help=">0: stream to the live viewer")
    ap.add_argument("--live_dir", default="", help="file-backed viewer sink (scripts/viewer_serve.py)")
    return ap.parse_args()


def eval_gates(res, met, prm, T, rel_tol=0.003, hole_tol=0.02):
    """G2 guards all zero; G3 rest (tail jitter and the drift of the delivered terminal
    velocity); G4 holes (absolute, or the target's own level) and ejection."""
    dn = res["deliver_n"]
    recs = [h for h in res["history"] if "v_mean" in h and h.get("frame_end") is not None
            and not h.get("null_commit") and h["frame_end"] <= dn]
    v_mean = recs[-1]["v_mean"] if recs else 0.0
    drift_rel = v_mean * prm.dt * T / max(met["bbox_diag"], 1e-9)
    gates = {"G2_guards": all(v == 0 for v in res["guards"].values()),
             "G3_rest": met["jitter_rel"] < rel_tol and drift_rel < rel_tol,
             "G4_holes_abs": met["hole_frac"] <= max(hole_tol, met.get("hole_frac_tgt", 0.0) + 0.005),
             "G4_ejection": met["outside_max"] == 0.0 and met["stray_max"] < 2e-3}
    print("[gates] " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in gates.items())
          + f"   (guards={res['guards']}, jitter_rel={met['jitter_rel']:.5f}, drift_rel={drift_rel:.5f}, "
          f"hole={met['hole_frac'] * 100:.2f}% tgt={met['hole_frac_tgt'] * 100:.2f}%, "
          f"outside_max={met['outside_max'] * 100:.3f}%, stray_max={met['stray_max'] * 100:.3f}%)", flush=True)
    gates["drift_rel"] = drift_rel
    return gates


def provenance(args, prm) -> dict:
    root = Path(__file__).resolve().parent.parent
    files = sorted(p for t in TRACKED for p in ((root / t).rglob("*.py") if (root / t).is_dir() else [root / t]))
    code_hash = hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()[:16]
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=root,
                                          stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        vf = root / "VERSION"                  # tarball deploys carry the sha in VERSION
        git_sha = vf.read_text().strip() if vf.exists() else None
    return {**vars(args), "mpm": dataclasses.asdict(prm), "git_sha": git_sha, "code_hash": code_hash}


def live_hooks(args, src, tgt, prm, cfg):
    if not (args.live_port or args.live_dir):
        return None, None
    from physmorph.render.covariance import sigma0_from_nn
    from physmorph.viewer.server import LiveServer
    sink = (LiveServer.to_dir(Path(args.live_dir) / f"{Path(args.out).name}_{ARM}") if args.live_dir
            else LiveServer(args.live_port))
    return sink.begin_run(ARM, src, tgt, prm, cfg, sigma0_from_nn(tgt, 0.9))


def main():
    args = parse_args()
    gpu.require_cuda()
    cfg0 = PipelineConfig(support_target_ref=args.support_target_ref, support_form=args.support_form,
                          loss_follows_n=args.loss_follows_n)
    prep = prepare(args.src, args.tgt, args.n, args.seed, args.cell_diag, cfg0.young, cfg0.poisson,
                   cfg0.nn_far_k, log=lambda s: print(s, flush=True),
                   loss_ref_n=cfg0.mass_ref_n if cfg0.loss_follows_n else 0)
    src, tgt, prm = prep.src, prep.tgt, prep.prm
    cfg = dataclasses.replace(cfg0, animations=args.animations, patience=args.patience,
                              reject_stop=args.reject_stop, render_weight_scale=args.render_weight_scale,
                              ot_iters=args.ot_iters, support_weight=args.support_weight,
                              loss_res=prep.loss_res, unit_ref_res=prep.unit_ref_res,
                              nn_berth_k=prep.nn_berth_k, grad_dump=args.grad_dump, ls_probe=args.ls_probe)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"[v2run] {args.src} -> {args.tgt}  N={args.n}  T={cfg.T}  iters={cfg.iters}  "
          f"anims={cfg.animations} | dx={prm.dx} dt={prm.dt:.5f} smoothing={prm.smoothing}", flush=True)
    print(f"[v2run] baseline chamfer (undeformed) = {metrics.chamfer(src, tgt):.4f}", flush=True)
    out = {"provenance": {**provenance(args, prm), "ppc": prep.ppc}, "arms": {}}
    cfg_dump = dataclasses.asdict(cfg)                 # before the run: c2f edits render_res
    print(f"\n[v2run] ===== ARM {ARM} =====", flush=True)
    t_thin = time.time()
    ts = thin_set(tgt, prm.dx, cfg.mass_ref_n)                 # the thin part of the target (measurement)
    print(f"[v2run] thin set: {len(ts.points)} of {ts.n_outer} outer target points below two MPM cells "
          f"({time.time() - t_thin:.1f} s)", flush=True)
    on_commit, on_iter = live_hooks(args, src, tgt, prm, cfg)
    t0 = time.time()
    stride = args.save_F_stride if args.save_F_stride > 0 else cfg.T
    res = run_pipeline(src, tgt, prm, cfg, log=lambda s: print(s, flush=True), on_commit=on_commit,
                       on_iter=on_iter, F_stride=stride, thin=ts)
    seconds = time.time() - t0
    frames, dn = res["frames"], res["deliver_n"]
    delivered = [h for h in res["history"] if h.get("frame_end") and not h.get("null_commit")
                 and h["frame_end"] <= dn]
    detF_min = min([1.0] + [h["Jmin_traj"] for h in delivered])
    met = metrics.summarize(frames.x[:dn], tgt, n_held=res["n_held"], detF_min=detF_min)
    met.update(thin_metrics(frames.x[dn - 1], ts))
    met.update(surface_roughness(frames.x[dn - 1], tgt))
    mv = [h["move"] for h in res["history"] if "move" in h]
    met["move_cv"] = float(np.std(mv) / max(np.mean(mv), 1e-9)) if len(mv) > 2 else float("inf")
    met["move_first_frac"] = float(sum(mv[:3]) / max(sum(mv), 1e-9)) if mv else 1.0
    gates = eval_gates(res, met, prm, cfg.T)
    idx, F_samples = frames.archive_F()
    np.savez(f"{args.out}_{ARM}.npz", src=src, tgt=tgt, orient=np.str_(orient_name(args.tgt)),
             frames=np.stack(frames.x), deliver_n=np.int64(dn), truncation=json.dumps(res["truncation"]),
             F_samples=np.stack(F_samples), F_sample_idx=np.array(idx),
             render_mask=np.ones(len(src), bool), s=np.zeros(0, np.float32),
             Fg_commit_idx=np.zeros(0, np.int64), Fg_commits=np.zeros((0, 0, 3, 3), np.float32))
    out["arms"][ARM] = {"config": cfg_dump, "metrics": met,
                        "gates": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in gates.items()},
                        "guards": res["guards"], "converged": res["converged"], "balancer": res["balancer"],
                        "deliver_n": int(dn), "truncation": res["truncation"], "n_held": res["n_held"],
                        "seconds": seconds, "history": res["history"]}
    print(f"[v2run] ARM {ARM}: chamfer={met['chamfer']:.4f}  silIoU={met['sil_iou']:.4f}  "
          f"hole={met['hole_frac'] * 100:.2f}%  jitter_rel={met['jitter_rel']:.5f}  "
          f"detFmin={met['detF_min']:.4f}  move_cv={met['move_cv']:.2f}  "
          f"first3={met['move_first_frac'] * 100:.0f}%  ({seconds / 60:.1f} min)", flush=True)
    print(f"[v2run] thin: uncovered {met.get('thin_uncovered', float('nan')) * 100:.1f}% (world "
          f"{met.get('thin_uncovered_world', float('nan')) * 100:.1f}%) of {met.get('thin_n', 0)} thin outer points",
          flush=True)
    Path(f"{args.out}.json").write_text(json.dumps(out))
    print(f"\nsaved {args.out}.json", flush=True)


if __name__ == "__main__":
    main()
