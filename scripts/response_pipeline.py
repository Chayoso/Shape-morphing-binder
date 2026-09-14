"""Server-only measured-response surface-control experiment and HQ frame export."""
from dataclasses import asdict, fields
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import socket
import sys

import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physmorph.losses.volumetric import rasterize_mass, target_mass_grid
from physmorph.mpm.function import RolloutSpec
from physmorph.mpm.state import MPMParams
from physmorph.pipeline.gauss_loss import GaussViews, _gs
from physmorph.pipeline.render_loss import make_views
from physmorph.pipeline.response_control import ResponseConfig, SurfaceStrainBasis, optimize_response_window
from physmorph.viewer.geometric import encode_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="previous diagnostic directory; reuse its fixed source vol0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--objective", choices=("image", "physics"), default="image")
    ap.add_argument("--linear", action="store_true", help="24 modes rather than 6 constant surface strain modes")
    ap.add_argument("--radius", type=float, default=.05)
    ap.add_argument("--max_control", type=float, default=.3)
    ap.add_argument("--max_particle_control", type=float, default=.4)
    ap.add_argument("--fd_strain", type=float, default=.005)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--view_resolution", type=int, default=1024)
    ap.add_argument("--frames", action="store_true")
    args = ap.parse_args()
    if socket.gethostname() != "hyde06":
        raise SystemExit("Run simulation and native rendering on hyde06 only")
    torch.set_num_threads(4)
    out, reference = Path(args.out), Path(args.reference)
    if (out/"metadata.json").exists():
        raise ValueError("choose a fresh result directory")
    if args.steps <= 0 or args.iterations <= 0:
        raise ValueError("positive horizon and iteration budget are required")
    out.mkdir(parents=True, exist_ok=True)
    previous = json.loads((reference/"metadata.json").read_text())
    with np.load(reference/"trajectory.npz", allow_pickle=False) as a:
        source, target = a["src"].copy(), a["tgt"].copy()
        vol0 = a["vol0"].copy()
        mask, target_mask = a["surface_mask"].copy(), a["target_surface_mask"].copy()
    # Pure internal actuation cannot translate the body's center of mass. Both
    # comparison arms use this same achievable, center-aligned target fixture.
    target_translation = source.mean(0)-target.mean(0)
    target += target_translation
    d = previous["discretization"]
    prm = MPMParams(**{f.name: d[f.name] for f in fields(MPMParams)})
    spec = RolloutSpec(source, 1., d["lam"], d["mu"], prm, args.steps, device="cuda", vol0=vol0)
    basis = SurfaceStrainBasis(source, mask, "cuda", args.linear)
    cfg = ResponseConfig(iterations=args.iterations, radius=args.radius, fd_strain=args.fd_strain,
                          max_control_rms=args.max_control, max_particle_control=args.max_particle_control,
                          objective=args.objective)
    mt, tmt = torch.tensor(mask, device="cuda"), torch.tensor(target_mask, device="cuda")
    target_t = torch.tensor(target, device="cuda")
    mass = torch.ones(len(source), device="cuda")
    gmin = torch.tensor(prm.grid_min, device="cuda")
    dims, loss_dx = tuple(d["loss_dims"]), d["loss_dx"]
    tmass = target_mass_grid(target_t, mass, gmin, loss_dx, dims)
    # Frame a bounding sphere including the initial 3-sigma support. A coordinate
    # extent alone can crop a perspective target and conceal silhouette error.
    framing_radius = (float(np.linalg.norm(np.concatenate([source, target]), axis=1).max())
                      + 3*previous["sigma0"])/np.sin(.7/2)*1.1
    extent = framing_radius/2.6  # GaussViews' fixed radius-to-extent convention
    views = make_views(2, (0., .35))
    render = GaussViews(views, extent, previous["sigma0"], args.resolution, "cuda")
    identity = torch.eye(3, device="cuda").repeat(len(source), 1, 1)
    render.bake_targets(target_t, identity, tmt)
    count_image_values = len(views)*3*render.res**2
    def residuals(state, control):
        cur = rasterize_mass(state[0], mass, gmin, loss_dx, dims)
        rm = torch.log1p(cur)-torch.log1p(tmass)
        rp = torch.cat([rm, state[2].flatten()/len(source)**.5,
                         control.flatten()*(.002/(len(source)*spec.T))**.5])
        ri = torch.cat([(render._render(state[0], cam, state[3], mt)-ti).flatten()
                         for cam, ti in zip(render.cams, render.targets)])/count_image_values**.5
        return {"mass": rm, "physics": rp, "image": ri}
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    extension = importlib.import_module(_gs()[0].__name__+"._C")
    metadata = {"method": "surface_response_constrained", "objective": args.objective,
        "discretization": {**asdict(prm), "N": len(source), "T": spec.T, "lam": d["lam"], "mu": d["mu"],
                           "loss_dims": dims, "loss_dx": loss_dx},
        "control": {"modes": basis.count, "temporal_basis": "constant dFc repeated on every substep",
                    "interior_dFc": "fixed zero", "metric": "mean surface Frobenius squared per substep",
                    "spatial_support": "fixed shell, hard boundary; constant or linear fields within shell"},
        "optimizer": asdict(cfg), "target_translation": target_translation.tolist(),
        "loss_resolution": render.res, "display_resolution": args.view_resolution,
        "camera": {"radius": framing_radius, "fov_radians": .7,
                   "framing": "source/target bounding sphere plus 3 sigma, 10 percent margin",
                   "views": [list(v) for v in views]},
        "sigma0": previous["sigma0"], "opacity": .9, "children": 1, "surface_count": int(mask.sum()),
        "provenance": {"reference_archive_sha256": sha(reference/"trajectory.npz"),
            "script_sha256": sha(__file__), "raster_sha256": sha(extension.__file__),
            "torch": torch.__version__, "warp": wp.__version__,
            "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (ROOT/"physmorph").rglob("*.py")}},
        "limitations": ["Constrained image task, not the original weighted-sum objective.",
            "This reduced experiment freezes interior control; interior dynamics remain full MPM.",
            "Changing horizon changes elapsed time and accumulated model-control injection.",
            "Surface attachment, floaters and finished morph quality require separate QA."]}
    (out/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    def callback(record, tr):
        brief = {k: v for k, v in record.items() if k not in ("fd", "image_response_eigenvalues")}
        print(json.dumps(brief), flush=True)
        with (out/"iterations.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record)+"\n")
    result = optimize_response_window(spec, basis, residuals, cfg, on_iteration=callback)
    tr = result["trajectory"]
    report = {k: v for k, v in result.items() if k not in ("coefficients", "control", "trajectory")}
    frames = np.stack([a.numpy() for a in tr.x])
    fg = np.stack([a.numpy() for a in tr.F_geom])
    fm = np.stack([a.numpy() for a in tr.F])
    from scipy.spatial import cKDTree
    # Independent quality diagnostics use raw simulation states only.
    def chamfer(x):
        return float(cKDTree(target).query(x)[0].mean()+cKDTree(x).query(target)[0].mean())*.5
    report["raw_state_metrics"] = {"chamfer_source": chamfer(source), "chamfer_final": chamfer(frames[-1]),
        "max_displacement": float(np.linalg.norm(frames[-1]-source, axis=1).max()),
        "max_center_drift": float(np.linalg.norm(frames.mean(1)-source.mean(0), axis=1).max()),
        "interior_control_max": float(result["control"][:, ~mt].abs().max())}
    (out/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(out/"trajectory.npz", src=source, tgt=target, frames=frames,
        F_model_frames=fm, F_geom_frames=fg, controls=result["control"].cpu().numpy()[None],
        coefficients=result["coefficients"].cpu().numpy(), basis_modes=basis.modes.cpu().numpy(),
        surface_mask=mask, target_surface_mask=target_mask, vol0=vol0, sigma0=np.float32(previous["sigma0"]))
    if args.frames:
        display = GaussViews([views[0]], extent, previous["sigma0"], args.view_resolution, "cuda")
        display.bake_targets(target_t, identity, tmt)
        def rgb(image):
            return image.permute(1, 2, 0).cpu().numpy()
        (out/"target.png").write_bytes(encode_png(rgb(display.targets[0])))
        (out/"frames").mkdir(exist_ok=True)
        with torch.no_grad():
            for index, (x, f) in enumerate(zip(frames, fg)):
                image = display._render(torch.tensor(x, device="cuda"), display.cams[0],
                                          torch.tensor(f, device="cuda"), mt)
                (out/"frames"/f"frame_{index:05d}.png").write_bytes(encode_png(rgb(image)))
        (out/"qa_manifest.json").write_text(json.dumps({"frame_count": len(frames),
            "files": [f"frames/frame_{i:05d}.png" for i in range(len(frames))],
            "visual_review": "pending", "opacity": .9, "resolution": display.res}, indent=2), encoding="utf-8")
    print("RESULT", json.dumps({"initial": result["initial"], "final": result["final"],
        "raw": report["raw_state_metrics"], "seconds": result["seconds"], "rollouts": result["rollout_evaluations"]}), flush=True)


if __name__ == "__main__":
    main()
