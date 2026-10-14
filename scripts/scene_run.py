"""The two-body scene (D147): two spheres thrown toward each other collide at the apex, bounce apart, land and tumble,
each morphing into its own bunny in its own frame. CUDA only; run on hyde06.

  $PY scripts/scene_run.py --n 40000 --seed 97 --out $OUT/scene40k            # the render arm
  $PY scripts/scene_run.py --n 40000 --seed 97 --render_weight_scale 0 --out $OUT/scene40k_phys

Each body is the frozen single-body run (scripts/pipeline_run.py's defaults: its prepare, its target, its objective)
with seeds seed and seed + 1000, drag 0; the scene's numbers are physmorph/pipeline/scene/spec.SceneSpec (or --scene,
a JSON file of its fields). Outputs: <out>.npz (every simulated frame, the bodies' rows, their targets and frames' references)
and <out>.json (the scene's numbers, the per-window history, each body's end metrics in its own frame).
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

import physmorph  # noqa: E402,F401  (before torch: the CuPy / NVRTC order)
import torch  # noqa: E402

from physmorph import gpu, metrics  # noqa: E402
from physmorph.pipeline.scene.frame import body_frame  # noqa: E402
from physmorph.pipeline.scene.runner import run_scene  # noqa: E402
from physmorph.pipeline.scene.spec import SceneSpec, layout  # noqa: E402
from physmorph.pipeline.scene.window import Body  # noqa: E402
from physmorph.pipeline.target import build_target, calibrate_units, target_relief  # noqa: E402
from physmorph.thin import thin_metrics, thin_set  # noqa: E402
from scripts.pipeline_run import parse_args as run_args, prepared, provenance, run_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="", help="a JSON file of SceneSpec fields (default: the pre-registered scene)")
    ap.add_argument("--n", type=int, default=40000, help="particles per body")
    ap.add_argument("--seed", type=int, default=97)
    ap.add_argument("--src", default="assets/isosphere.obj")
    ap.add_argument("--tgt", default="assets/bunny.obj")
    ap.add_argument("--out", default="output/scene")
    ap.add_argument("--render_weight_scale", type=float, default=1.0, help="0: the render-off twin")
    ap.add_argument("--windows", type=int, default=0, help="0: the scene's length")
    ap.add_argument("--save_stride", type=int, default=1, help="archive every k-th simulated frame and the last")
    a = ap.parse_args()
    gpu.require_cuda()
    spec = SceneSpec(**json.loads(Path(a.scene).read_text())) if a.scene else SceneSpec()
    log = lambda s: print(s, flush=True)                                          # noqa: E731
    base = run_args(["--n", str(a.n), "--src", a.src, "--tgt", a.tgt, "--seed", str(a.seed),
                     "--render_weight_scale", str(a.render_weight_scale), "--drag", "0", "--out", a.out])
    bodies, srcs, vps, thins, tgts, start = [], [], [], [], [], 0
    cfg = None
    for b, seed in enumerate((a.seed, a.seed + 1000)[:spec.bodies]):   # body 1: its own samples (none of body 0's)
        cfg0, prep = prepared(base, seed)
        cfg_b = run_config(base, cfg0, prep)
        cfg = cfg or cfg_b
        tgt = build_target(prep.tgt, prep.prm, cfg_b, draws=prep.tgt_draws)
        if prep.tgt_surface is not None:
            tgt.relief = target_relief(tgt.pts, prep.tgt_surface, cfg_b)
        X = np.asarray(prep.src, np.float64)
        X = X - X.mean(0)
        calibrate_units(tgt, gpu.tensor(X.astype(np.float32)), cfg_b)
        # the gate's volume per particle: the sample's own (one jittered particle per voxel of the sampler's fill; after
        # the density match the source's fill is the target's), not the mesh's 110^3 volume / N, which read 0.917 of the
        # cell at interior nodes in the prototype (M-b) and let the gate fire only after an overlap of a spacing
        vp = (prep.density["target_fill"] if prep.density else prep.v_src) / a.n
        bodies.append(Body(start=start, stop=start + a.n, Xq=gpu.tensor(X.astype(np.float32)), tgt=tgt, vp=vp,
                           mass=cfg_b.mass_ref_n / a.n, cfg=cfg_b))
        srcs.append(X)
        vps.append(vp)
        tgts.append(prep.tgt)
        thins.append(thin_set(prep.tgt, prep.cell_shape or prep.prm.dx, cfg_b.mass_ref_n))
        if b == 0:
            dx, dt, prov = prep.prm.dx, prep.prm.dt, provenance(base, prep.prm)
        start += a.n
    x0, v0, R0, prm, info = layout(spec, srcs, vps, cfg.young, cfg.poisson, cfg.mass_ref_n, dx, dt, cfg.T)
    if spec.frame == "fixed":                   # V0: the objective in the frame the body has at launch (the world's)
        for bd, R in zip(bodies, R0):
            bd.fixed = (gpu.tensor(x0[bd.rows].mean(0)), gpu.tensor(R.astype(np.float32)))
    windows = a.windows or info["windows"]
    log(f"[scene] {json.dumps(info)}")
    t0 = time.time()
    res = run_scene(bodies, gpu.tensor(x0), gpu.tensor(v0), prm, cfg, windows, spec.mu, log=log, F_stride=cfg.T)
    seconds = time.time() - t0
    frames = res["frames"]
    n_frames = len(frames.x)
    raws = sorted(set(range(0, n_frames, a.save_stride)) | {n_frames - 1})
    xs = np.stack([frames.x[i] for i in raws])

    def shape(y, tgt, ts):
        e = metrics.target_extent(tgt)
        m = {"chamfer": metrics.chamfer(y, tgt), "sil_iou": metrics.sil_iou(y, tgt, extent=e),
             "hole_frac": metrics.hole_frac(y, e)}
        m.update(thin_metrics(y, ts))
        return {k: m[k] for k in ("chamfer", "sil_iou", "hole_frac", "thin_uncovered", "thin_uncovered_world") if k in m}

    met = []                                    # each body's end in its objective's frame, and in its Buckens frame
    for bd, tgt, ts in zip(bodies, tgts, thins):
        with torch.no_grad():
            xb = gpu.tensor(xs[-1][bd.rows])
            m = shape(bd.frame(xb)[0].cpu().numpy(), tgt, ts)
            if bd.fixed is not None:
                m["buckens"] = shape(body_frame(xb, bd.Xq)[0].cpu().numpy(), tgt, ts)
        met.append(m)
    log(f"[scene] done: {n_frames} frames, {seconds / 60:.1f} min; " + "  ".join(
        f"b{b}: silIoU={m['sil_iou']:.4f} chamfer={m['chamfer']:.4f} thin={m.get('thin_uncovered', float('nan')) * 100:.1f}%"
        for b, m in enumerate(met)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(f"{a.out}.npz", frames=xs, raws=np.array(raws), rows=np.array([[bd.start, bd.stop] for bd in bodies]),
             **{f"tgt{b}": t for b, t in enumerate(tgts)}, **{f"src{b}": s.astype(np.float32) for b, s in enumerate(srcs)},
             R0=np.stack(R0[:len(bodies)]), floor_y=np.float64(prm.floor_y))
    Path(f"{a.out}.json").write_text(json.dumps({
        "provenance": {**prov, "scene_args": vars(a)}, "scene": info, "mpm": dataclasses.asdict(prm),
        "config": dataclasses.asdict(cfg), "metrics": met, "guards": res["guards"], "seconds": seconds,
        "history": res["history"]}))
    log(f"saved {a.out}.json")


if __name__ == "__main__":
    main()
