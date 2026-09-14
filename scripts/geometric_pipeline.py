"""Server-only joint geometric dFc diagnostic with native HQ live images.

This is the approved gradient-flow milestone, not a finished morph-quality run.
No simulator or renderer runs locally. See docs/geometric_control_contract.md.
"""
from dataclasses import asdict
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physmorph.losses.volumetric import d_vol, target_mass_grid
from physmorph.mpm.constitutive import lame
from physmorph.mpm.function import RolloutSpec
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import compute_rest_volumes
from physmorph.pipeline.gauss_loss import GaussViews, _gs
from physmorph.pipeline.geometric import GeometricConfig, next_window_spec, optimize_geometric_window
from physmorph.pipeline.render_loss import make_views
from physmorph.pipeline.runner import _surface_weights
from physmorph.render.covariance import sigma0_from_nn
from physmorph.viewer.geometric import GeometryMonitor, encode_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="npz containing src/tgt; only those initial arrays are read")
    ap.add_argument("--out")
    ap.add_argument("--port", type=int, default=8774)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--step_size", type=float, default=.003)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--view_resolution", type=int, default=1024)
    ap.add_argument("--children", type=int, default=1, choices=(1, 4))
    ap.add_argument("--image_loss", choices=("l1", "l2"), default="l2")
    ap.add_argument("--physics_only", action="store_true")
    ap.add_argument("--exit", action="store_true", help="exit after saving frames, instead of serving replay")
    ap.add_argument("--replay", help="serve saved PNG substeps from an existing diagnostic directory")
    args = ap.parse_args()
    if socket.gethostname() != "hyde06":
        raise SystemExit("All simulation and native rendering runs go to hyde06")
    if not args.replay and not args.out:
        ap.error("--out is required for a new run")
    if args.replay:
        from PIL import Image
        folder = Path(args.replay)
        meta = json.loads((folder/"metadata.json").read_text())
        manifest = json.loads((folder/"qa_manifest.json").read_text())
        rgb_frames = [np.asarray(Image.open(folder/name).convert("RGB"))/255.
                      for name in manifest["files"]]
        target_rgb = np.asarray(Image.open(folder/"target.png").convert("RGB"))/255.
        monitor = GeometryMonitor(args.port)
        try:
            while True:
                for i, rgb in enumerate(rgb_frames):
                    monitor.publish(rgb, target_rgb, {"phase": "saved physical trajectory replay", "frame": i,
                        "frames": len(rgb_frames), "resolution": meta["display_resolution"],
                        "simulation_time": i*meta["discretization"]["dt"], "replay_frame_seconds": .2,
                        "quality_gate": "diagnostic; final morph quality not approved"})
                    time.sleep(.2)
        finally:
            monitor.close()
    if min(args.windows, args.steps, args.resolution, args.view_resolution) < 1:
        raise ValueError("windows, steps and resolutions must be positive")
    torch.set_num_threads(4)
    out = Path(args.out)
    if (out/"metadata.json").exists():
        raise ValueError("choose a fresh output directory to preserve prior results")
    out.mkdir(parents=True, exist_ok=True)
    if args.input:
        with np.load(args.input, allow_pickle=False) as data:
            source, target = data["src"].copy(), data["tgt"].copy()
        prm = MPMParams(dx=.5, dt=1/240, nx=64, ny=64, nz=64, grid_min=(-16.,)*3)
        lam, mu = lame(1.4e5, .2)
        loss_dims, loss_dx = (48,)*3, 32/48
        fraction = .3
    else:
        # Smooth, connected smoke fixture. It does not claim sphere->bunny quality.
        rng = np.random.default_rng(914)
        candidates = rng.uniform(-1, 1, (10000, 3))
        source = candidates[np.linalg.norm(candidates, axis=1) < 1][:2048].astype(np.float32)
        target = source*np.array([1.13, .84, 1.04], np.float32)
        prm = MPMParams(dx=.5, dt=1/120, nx=16, ny=16, nz=16, grid_min=(-4.,)*3)
        lam, mu = 800., 400.
        loss_dims, loss_dx = (24,)*3, 8/24
        fraction = .4
    source = np.ascontiguousarray(source, np.float32)
    target = np.ascontiguousarray(target, np.float32)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("src/tgt must have equal (N,3) shapes")
    mask = _surface_weights(source, 24, fraction, 0.) > .5
    tmask = _surface_weights(target, 24, fraction, 0.) > .5
    mt = torch.tensor(mask, device="cuda")
    target_t = torch.tensor(target, device="cuda")
    sigma = sigma0_from_nn(target[tmask], 1.)
    views = make_views(2, (0., .35))
    extent = float(np.abs(np.concatenate([source, target])).max())*1.25
    identity = np.tile(np.eye(3, dtype=np.float32), (len(source), 1, 1))
    identity_t = torch.tensor(identity, device="cuda")
    def build_renderer(res, selected_views):
        bundle = GaussViews(selected_views, extent, sigma, res, "cuda", child_count=args.children)
        if args.children > 1:
            bundle.configure_source(source, mask)
        bundle.bake_targets(target_t, identity_t, torch.tensor(tmask, device="cuda"))
        return bundle
    render = build_renderer(args.resolution, views)
    display = build_renderer(args.view_resolution, [views[0]])
    gmin = torch.tensor(prm.grid_min, device="cuda")
    mass = torch.ones(len(source), device="cuda")
    target_grid = target_mass_grid(target_t, mass, gmin, loss_dx, loss_dims)
    # Preserve the baseline mass sum. No factor is inferred from gradient norms.
    def physical_loss(state, control):
        return (d_vol(state[0], mass, target_grid, gmin, loss_dx, loss_dims)
                + .5*state[2].square().sum(1).mean()
                + .001*control.square().sum()/(len(source)*args.steps))
    def render_loss(x, fg):
        terms = []
        for cam, target_image in zip(render.cams, render.targets):
            residual = render._render(x.contiguous(), cam, fg, mt, render.source_offsets)-target_image
            terms.append(residual.abs().mean() if args.image_loss == "l1" else .5*residual.square().mean())
        return torch.stack(terms).mean()
    def picture(x, fg):
        with torch.no_grad():
            image = display._render(torch.as_tensor(x, device="cuda"), display.cams[0],
                                      torch.as_tensor(fg, device="cuda"), mt, display.source_offsets)
        return image.permute(1, 2, 0).cpu().numpy()
    target_picture = display.targets[0].permute(1, 2, 0).cpu().numpy()
    monitor = GeometryMonitor(args.port) if args.port else None
    cfg = GeometricConfig(iterations=args.iterations, step_size=args.step_size,
                           render_weight=0. if args.physics_only else 1.)
    spec = RolloutSpec(source, 1., lam, mu, prm, args.steps, device="cuda",
                       vol0=compute_rest_volumes(source, 1., prm, "cuda"))
    frames, model_frames, geom_frames, history, controls = [source.copy()], [identity.copy()], [identity.copy()], [], []
    metadata = {"method": "geometric_joint_dfc_v1", "status": "gradient-flow diagnostic",
                "F_semantics": {"F_model": "constitutive/control", "F_geom": "original-reference advection"},
                "discretization": {**asdict(prm), "N": len(source), "T": args.steps, "lam": lam, "mu": mu,
                                   "loss_dims": loss_dims, "loss_dx": loss_dx},
                "optimizer": asdict(cfg), "image_loss": args.image_loss,
                "loss_resolution": render.res, "display_resolution": display.res,
                "surface_count": int(mask.sum()), "children": args.children, "opacity": .9,
                "sigma0": sigma, "fixed_source_surface": True,
                "mass_loss_reduction": "sum (unchanged Xu baseline)",
                "limitations": ["Surface membership is an estimate frozen at source.",
                    "No no-floater or finished-morph claim; inspect every saved frame.",
                    "Per-window adjoints; no differentiation through past window commits.",
                    "F_model constitutive consistency and native raster derivative are separate gates."]}
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    extension = importlib.import_module(_gs()[0].__name__+"._C")
    metadata["provenance"] = {"script_sha256": sha(__file__), "torch": torch.__version__,
        "warp": wp.__version__, "raster_sha256": sha(extension.__file__),
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (ROOT/"physmorph").rglob("*.py")}}
    (out/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if monitor:
        monitor.publish(picture(source, identity), target_picture,
                        {"phase": "initial", "resolution": display.res, "loss_resolution": render.res})
    try:
        for window in range(args.windows):
            def callback(record, trajectory):
                rec = {**record, "window": window}
                print(json.dumps(rec), flush=True)
                with (out/"iterations.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec)+"\n")
                if monitor:
                    monitor.publish(picture(trajectory.x[-1].numpy(), trajectory.F_geom[-1].numpy()),
                                    target_picture, {"phase": "optimizer endpoint preview", **rec})
            result = optimize_geometric_window(spec, physical_loss, render_loss, mask, cfg, on_iteration=callback)
            tr = result["trajectory"]
            frames.extend(a.numpy().copy() for a in tr.x[1:])
            model_frames.extend(a.numpy().copy() for a in tr.F[1:])
            geom_frames.extend(a.numpy().copy() for a in tr.F_geom[1:])
            controls.append(result["control"].cpu().numpy())
            history.append({"window": window, "loss": result["loss"], "health": result["health"],
                            "accepted_steps": result["accepted_steps"], "iterations": result["history"]})
            spec = next_window_spec(spec, tr)
        np.savez_compressed(out/"trajectory.npz", src=source, tgt=target, frames=np.asarray(frames),
                            F_model_frames=np.asarray(model_frames), F_geom_frames=np.asarray(geom_frames),
                            controls=np.asarray(controls), surface_mask=mask, target_surface_mask=tmask,
                            vol0=spec.vol0, sigma0=np.float32(sigma))
        (out/"history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        # Every physical substep is saved, without subsampling or hidden opacity masks.
        qa_dir = out/"frames"
        qa_dir.mkdir(exist_ok=True)
        (out/"target.png").write_bytes(encode_png(target_picture))
        rendered = []
        for index, (x, fg) in enumerate(zip(frames, geom_frames)):
            rgb = picture(x, fg)
            (qa_dir/f"frame_{index:05d}.png").write_bytes(encode_png(rgb))
            rendered.append(rgb)
        (out/"qa_manifest.json").write_text(json.dumps({"frame_count": len(frames),
            "files": [f"frames/frame_{i:05d}.png" for i in range(len(frames))],
            "visual_review": "pending", "opacity": .9, "resolution": display.res}, indent=2), encoding="utf-8")
        print("SAVED", out, flush=True)
        if monitor and not args.exit:
            while True:
                for i, rgb in enumerate(rendered):
                    monitor.publish(rgb, target_picture, {"phase": "physical trajectory replay", "frame": i,
                                    "frames": len(frames), "resolution": display.res,
                                    "simulation_time": i*prm.dt, "replay_frame_seconds": .2,
                                    "quality_gate": "diagnostic; final morph quality not approved"})
                    time.sleep(.2)
    except Exception as exc:
        if monitor:
            monitor.update_status(phase="failed", error=str(exc))
        raise
    finally:
        if monitor:
            monitor.close()


if __name__ == "__main__":
    main()
