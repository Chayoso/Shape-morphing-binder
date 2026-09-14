"""Server-only sphere/bunny development and sealed paired control benchmark."""
from dataclasses import asdict
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch
import warp as wp
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physmorph import metrics
from physmorph.losses.volumetric import rasterize_mass, target_mass_grid
from physmorph.mpm.function import RolloutSpec
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import compute_rest_volumes
from physmorph.pipeline.gauss_loss import GaussViews, _gs
from physmorph.pipeline.image_metric import ScreenedImageResidual
from physmorph.pipeline.response_control import ResponseConfig, SurfaceStrainBasis, TemporalSurfaceBasis, optimize_response_window
from physmorph.pipeline.runner import _surface_weights
from physmorph.render.covariance import sigma0_from_nn
from physmorph.render.material_surface import source_surface, MaterialSurfaceViews
from physmorph.sampling.mesh import (load_mesh, sample_volume, filled_volume, _fill_centers,
                                     seal_internal_voxel_voids, sample_voxel_centers)
from physmorph.sampling.closed_mesh import ClosedMeshSampler, center_target_by_quadrature, validate_surface_arrays
from physmorph.viewer.geometric import encode_png


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def views(split):
    if split == "train":
        return [(j*np.pi/2, e) for e in (-.25, .25) for j in range(4)]
    if split == "validation":
        return [(np.pi/4+j*np.pi/2, e) for e in (-.1, .4) for j in range(4)]
    if split == "test":
        return [(np.pi/8+j*np.pi/4, e) for e in (-.5, 0., .5) for j in range(8)]
    raise ValueError(split)


def raw_quality(x, target, extent, split="validation"):
    result = {}
    for res in (96, 128, 160):
        ious = []
        for az, el in views(split):
            a = metrics._splat_body(x, res, az, el, extent)
            b = metrics._splat_body(target, res, az, el, extent)
            ious.append(float((a & b).sum()/max(int((a | b).sum()), 1)))
        result[f"point_silhouette_iou_{res}"] = float(np.mean(ious))
        result[f"point_silhouette_error_{res}"] = 1-float(np.mean(ious))
    clipped = []
    for az, el in views(split):
        right = np.array([np.cos(az), 0., -np.sin(az)])
        up = np.array([-np.sin(el)*np.sin(az), np.cos(el), -np.sin(el)*np.cos(az)])
        clipped.append(float((np.abs(np.stack([x@right, x@up], 1)) >= extent).any(1).mean()))
    result.update(projected_outside_max=max(clipped),
                  chamfer=.5*metrics.chamfer(x, target), stray_fraction=metrics.stray_frac(x),
                  hole_fraction=metrics.hole_frac(x, extent), outside_fraction=metrics.outside_frac(x, extent))
    return result


def prepare(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError("fixture directory must be new")
    out.mkdir(parents=True)
    smesh = load_mesh(str(Path(args.assets)/"isosphere.obj"))
    tmesh = load_mesh(str(Path(args.assets)/"bunny.obj"))
    source_scale = 2*np.sqrt(3)/np.linalg.norm(smesh.extents)
    if args.material_surface:
        spitch, tpitch = float(smesh.extents.max())/110, float(tmesh.extents.max())/110
        sc, source_seal = seal_internal_voxel_voids(_fill_centers(smesh, spitch), spitch)
        tc, target_seal = seal_internal_voxel_voids(_fill_centers(tmesh, tpitch), tpitch)
        volume = len(sc)*spitch**3*source_scale**3
        target_scale = (volume/(len(tc)*tpitch**3))**(1/3)
        source = sample_voxel_centers(sc, spitch, args.n, args.seed)*source_scale
        target = sample_voxel_centers(tc, tpitch, args.n, args.seed+1)*target_scale
    else:
        volume = filled_volume(smesh)*source_scale**3
        target_scale = (volume/filled_volume(tmesh))**(1/3)
        source = sample_volume(smesh, args.n, args.seed)*source_scale
        target = sample_volume(tmesh, args.n, args.seed+1)*target_scale
    target_center = target.mean(0)
    source = np.ascontiguousarray(source-source.mean(0), np.float32)
    target = np.ascontiguousarray(target-target.mean(0), np.float32)
    target += source.mean(0)-target.mean(0)
    mask = _surface_weights(source, 24, .4, 0) > .5
    tmask = _surface_weights(target, 24, .4, 0) > .5
    # Fixed total mass/density across particle-count studies.
    mass = 1000.*volume/args.n
    if args.dx <= 0 or not np.isclose(round(8/args.dx)*args.dx, 8.):
        raise ValueError("dx must partition the fixed eight-unit domain")
    ng = round(8/args.dx)
    prm = MPMParams(dx=args.dx, dt=args.dt, nx=ng, ny=ng, nz=ng, grid_min=(-4.,)*3)
    vol0 = compute_rest_volumes(source, mass, prm, "cuda")
    sigma = sigma0_from_nn(source[mask], scale=args.sigma_scale)
    extra = {}
    surface_metadata = {}
    if args.material_surface:
        import trimesh
        sx, sf = source_surface(source, subdivisions=2)
        pitch, centers = tpitch, tc
        mesh = trimesh.voxel.ops.points_to_marching_cubes(centers, pitch=pitch)
        if mesh.volume < 0:
            mesh.invert()
        components = len(trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces))))
        if not mesh.is_watertight or not mesh.is_winding_consistent or components != 1:
            raise ValueError("filled target surface must be one closed oriented component")
        # Dense 3-D supervision and the target surface share the same filled volume.
        extra = dict(surface0=sx, surface_faces=sf,
                     target_surface_vertices=np.asarray(mesh.vertices*target_scale-target_center+source.mean(0), np.float32),
                     target_surface_faces=np.asarray(mesh.faces, np.int64),
                     dense_target=np.asarray(centers*target_scale-target_center+source.mean(0), np.float32))
        surface_metadata = {"source_vertices": len(sx), "source_faces": len(sf),
            "target_vertices": len(mesh.vertices), "target_faces": len(mesh.faces),
            "target_components": components, "dense_mass_samples": len(centers),
            "source_voxel_seal": source_seal, "target_voxel_seal": target_seal,
            "target_voxel_pitch_world": pitch*target_scale,
            "surface_transport": "same MPM cubic grid velocity at every substep; no particle mass",
            "target_volume": "same axis-filled voxel set for dense 3D mass and target surface"}
    np.savez_compressed(out/"fixture.npz", src=source, tgt=target, surface_mask=mask,
                        target_surface_mask=tmask, vol0=vol0, **extra)
    meta = {"pair": "sphere_to_bunny", "seed": args.seed, "N": args.n,
            "source_volume": volume, "particle_mass": mass, "density": 1000.,
            "source_mesh_scale": source_scale, "target_mesh_scale": target_scale,
            "discretization": asdict(prm), "lam": 5000., "mu": 3000.,
            "loss_dims": [round(8/args.loss_dx)]*3, "loss_dx": args.loss_dx, "sigma0": sigma,
            "material_surface": surface_metadata,
            "sigma_scale": args.sigma_scale,
            "extent": metrics.target_extent(target), "surface_count": int(mask.sum()),
            "fixture_sha256": sha(out/"fixture.npz"), "script_sha256": sha(__file__),
            "assets": {name: sha(Path(args.assets)/name) for name in ("isosphere.obj", "bunny.obj")}}
    write_json(out/"metadata.json", meta)
    print("PREPARED", json.dumps(meta), flush=True)


def prepare_closed(args):
    """Reference-mesh surface and direct interior sampling, with no axis filling."""
    import trimesh
    import open3d
    out = Path(args.out)
    if out.exists():
        raise ValueError("fixture directory must be new")
    out.mkdir(parents=True)
    repaired = Path(args.repaired_target)
    repair_meta = json.loads((repaired.parent/"report.json").read_text())
    if (not repair_meta.get("valid") or repair_meta["output_sha256"] != sha(repaired)
            or repair_meta["input_sha256"] != sha(Path(args.assets)/"bunny.obj")):
        raise ValueError("verified repair of the same input bunny is required")
    with np.load(repaired, allow_pickle=False) as a:
        target_mesh = trimesh.Trimesh(a["vertices"], a["faces"], process=False)
    ss = ClosedMeshSampler(load_mesh(str(Path(args.assets)/"isosphere.obj")))
    ts = ClosedMeshSampler(target_mesh)
    source_scale = 2*np.sqrt(3)/np.linalg.norm(ss.mesh.extents)
    volume = float(ss.mesh.volume)*source_scale**3
    target_scale = (volume/float(ts.mesh.volume))**(1/3)
    source = ss.sample(args.n, args.seed)*source_scale
    target = ts.sample(args.n, args.seed+1)*target_scale
    sc = source.mean(0, dtype=np.float64)
    source = np.asarray(source-sc, np.float32)
    surface = ss.mesh.copy()
    for _ in range(3):
        surface = surface.subdivide()
    sx = np.asarray(surface.vertices*source_scale-sc, np.float32)
    sf = np.asarray(surface.faces, np.int64)
    tf = np.asarray(ts.mesh.faces, np.int64)
    dense, pitch = ts.quadrature(110)
    (target, tx, dense), tc = center_target_by_quadrature(target, ts.mesh.vertices*target_scale,
        dense.astype(np.float64)*target_scale, source.mean(0, dtype=np.float64))
    source_validation = validate_surface_arrays(sx, sf)
    target_validation = validate_surface_arrays(tx, tf)
    mask = _surface_weights(source, 24, .4, 0) > .5
    tmask = _surface_weights(target, 24, .4, 0) > .5
    if min(args.dx, args.loss_dx, args.dt) <= 0:
        raise ValueError("positive spatial and temporal discretization required")
    ng = round(8/args.dx)
    if not np.isclose(ng*args.dx, 8.) or not np.isclose(round(8/args.loss_dx)*args.loss_dx, 8.):
        raise ValueError("simulation and mass grids must partition the eight-unit domain")
    prm = MPMParams(dx=args.dx, dt=args.dt, nx=ng, ny=ng, nz=ng, grid_min=(-4.,)*3)
    mass = 1000.*volume/args.n
    vol0 = compute_rest_volumes(source, mass, prm, "cuda")
    np.savez_compressed(out/"fixture.npz", src=source, tgt=target, surface_mask=mask,
        target_surface_mask=tmask, vol0=vol0, surface0=sx, surface_faces=sf,
        target_surface_vertices=tx, target_surface_faces=tf, dense_target=dense)
    metadata = {"pair": "sphere_to_bunny", "seed": args.seed, "N": args.n,
        "source_volume": volume, "particle_mass": mass, "density": 1000.,
        "source_mesh_scale": source_scale, "target_mesh_scale": target_scale,
        "source_center_scaled": sc.tolist(), "target_center_scaled": tc.tolist(),
        "discretization": asdict(prm), "lam": 5000., "mu": 3000.,
        "loss_dims": [round(8/args.loss_dx)]*3, "loss_dx": args.loss_dx,
        "sigma0": sigma0_from_nn(source[mask], scale=args.sigma_scale), "sigma_scale": args.sigma_scale,
        "extent": metrics.target_extent(target), "surface_count": int(mask.sum()),
        "fixture_sha256": sha(out/"fixture.npz"), "script_sha256": sha(__file__),
        "assets": {name: sha(Path(args.assets)/name) for name in ("isosphere.obj", "bunny.obj")},
        "material_surface": {"source_vertices": len(sx), "source_faces": len(sf),
            "target_vertices": len(tx), "target_faces": len(tf),
            "source_euler": int(surface.euler_number), "target_euler": int(ts.mesh.euler_number),
            "source_final_validation": source_validation, "target_final_validation": target_validation,
            "centering": "one common target translation from dense mass quadrature center",
            "dense_minus_source_com": (dense.mean(0, dtype=np.float64)-source.mean(0, dtype=np.float64)).tolist(),
            "dense_mass_samples": len(dense), "dense_quadrature_pitch_world": pitch*target_scale,
            "sampling": "uniform rejection against closed meshes; Open3D occupancy with 3 rays",
            "open3d_version": open3d.__version__, "target_repair": repair_meta,
            "surface_transport": "refined original source mesh; every vertex advects with MPM grid"}}
    write_json(out/"metadata.json", metadata)
    print("PREPARED", json.dumps(metadata), flush=True)


def load_fixture(folder):
    folder = Path(folder)
    meta = json.loads((folder/"metadata.json").read_text())
    if sha(folder/"fixture.npz") != meta["fixture_sha256"]:
        raise ValueError("fixture hash changed")
    with np.load(folder/"fixture.npz", allow_pickle=False) as a:
        arrays = {k: a[k].copy() for k in a.files}
    return arrays, meta


def run(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError("result directory must be new")
    out.mkdir(parents=True)
    data, fixture = load_fixture(args.fixture)
    source, target = data["src"], data["tgt"]
    mask, tmask = data["surface_mask"], data["target_surface_mask"]
    prm = MPMParams(**fixture["discretization"])
    spec = RolloutSpec(source, fixture["particle_mass"], fixture["lam"], fixture["mu"],
                       prm, args.steps, device="cuda", vol0=data["vol0"])
    use_surface = args.observation == "material_surface"
    if use_surface:
        spec.surface0, spec.surface_faces = data["surface0"], data["surface_faces"]
    basis = SurfaceStrainBasis(source, mask, "cuda", patches=args.patches)
    if args.phases > 1:
        basis = TemporalSurfaceBasis(basis, args.steps, args.phases)
    config = ResponseConfig(iterations=args.iterations, fd_strain=args.fd_strain,
                            radius=.03, max_radius=.1, max_control_rms=.3,
                            max_particle_control=.4, objective=args.objective,
                            minimum_physics_progress=args.minimum_physics_progress,
                            geometric_response_constraints=args.geometric_constraints,
                            global_surface_checks=use_surface,
                            surface_response_constraints=args.surface_constraints)
    target_t = torch.tensor(target, device="cuda")
    mt, tmt = torch.tensor(mask, device="cuda"), torch.tensor(tmask, device="cuda")
    identity = torch.eye(3, device="cuda").repeat(len(source), 1, 1)
    mass = torch.full((len(source),), fixture["particle_mass"], device="cuda")
    grid_min = torch.tensor(prm.grid_min, device="cuda")
    dims, dx = tuple(fixture["loss_dims"]), fixture["loss_dx"]
    mass_target = torch.tensor(data["dense_target"], device="cuda") if use_surface else target_t
    target_masses = torch.full((len(mass_target),), float(mass.sum())/len(mass_target), device="cuda")
    tmass = target_mass_grid(mass_target, target_masses, grid_min, dx, dims)
    camera_radius = (float(np.linalg.norm(np.concatenate([source, target]), axis=1).max())
                     + 3*fixture["sigma0"])/np.sin(.35)*1.1
    render = GaussViews(views("train"), camera_radius/2.6, fixture["sigma0"], 1024, "cuda")
    render.bake_targets(target_t, identity, tmt)
    if use_surface:
        render = MaterialSurfaceViews(views("train"), camera_radius, spec.surface_faces,
            data["target_surface_vertices"], data["target_surface_faces"], 1024)
    # Fixed gray colors and white background make R=G=B exactly. A single
    # channel preserves the RGB mean-squared objective and its Gram matrix.
    image_count = len(render.cams)*render.res**2
    image_metric = (ScreenedImageResidual(render.res, render.res, args.screen_length, "cuda")
                    if args.image_metric == "screened" else lambda x: x)
    phase_index = torch.arange(args.steps, device="cuda")*args.phases//args.steps
    phase_counts = torch.bincount(phase_index, minlength=args.phases)
    phase_starts = torch.cat([phase_counts.new_zeros(1), phase_counts.cumsum(0)[:-1]])
    control_weights = (.002*phase_counts/(args.steps*len(source))).sqrt().reshape(-1, 1, 1, 1)
    def residuals(state, control):
        rm = torch.log1p(rasterize_mass(state[0], mass, grid_min, dx, dims))-torch.log1p(tmass)
        # Constant time basis: its control penalty can be represented once.
        rp = torch.cat([rm, state[2].flatten()/len(source)**.5,
                         (control[phase_starts]*control_weights).flatten()])
        covariance_F = state[3] if args.observation == "material" else identity
        ri = torch.cat([image_metric((render.render(state[4], c) if use_surface else
                                     render._render(state[0], c, covariance_F, mt))[0]-ti[0]).flatten()
                         for c, ti in zip(render.cams, render.targets)])/image_count**.5
        return {"mass": rm, "physics": rp, "image": ri}
    extension = importlib.import_module(_gs()[0].__name__+"._C")
    metadata = {"fixture": fixture, "optimizer": asdict(config), "steps": args.steps,
                "loss_resolution": render.res, "camera_radius": camera_radius,
                "training_views": views("train"), "patches": args.patches, "modes": basis.count,
                "basis_gram_condition": basis.gram_condition,
                "observation": args.observation,
                "image_metric": args.image_metric, "screen_length": args.screen_length,
                "color": [0.35]*3, "image_channels": "one; exact RGB-mean equivalent for this gray fixture",
                "temporal_phases": args.phases,
                "basis_sha256": hashlib.sha256(basis.modes.cpu().numpy().tobytes()).hexdigest(),
                "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (ROOT/"physmorph").rglob("*.py")},
                "script_sha256": sha(__file__), "raster_sha256": sha(extension.__file__),
                "selection": "last accepted iterate; no selection using validation/test quality",
                "interior_dFc": "fixed zero; shared between arms", "phase": "development"}
    write_json(out/"metadata.json", metadata)
    def callback(record, trajectory):
        brief = {k: v for k, v in record.items() if k not in ("fd", "image_response_eigenvalues")}
        print(json.dumps(brief), flush=True)
        with (out/"iterations.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record)+"\n")
        # Inspection artifact, never an edited physical state or a restart checkpoint.
        pending = out/f"latest_state.{os.getpid()}.pending"
        with pending.open("wb") as stream:
            np.savez_compressed(stream, x=trajectory.x[-1].numpy(),
                F_geom=trajectory.F_geom[-1].numpy(), iteration_attempt=record["iteration"],
                **({"surface": trajectory.surface_x[-1].numpy()} if use_surface else {}))
        os.replace(pending, out/"latest_state.npz")
    result = optimize_response_window(spec, basis, residuals, config, on_iteration=callback)
    tr = result["trajectory"]
    frames = np.stack([a.numpy() for a in tr.x])
    fg = np.stack([a.numpy() for a in tr.F_geom])
    fm = np.stack([a.numpy() for a in tr.F])
    velocities = np.stack([a.numpy() for a in tr.v])
    report = {k: v for k, v in result.items() if k not in ("trajectory", "coefficients", "control")}
    report["validation"] = raw_quality(frames[-1], target, fixture["extent"])
    report["initial_validation"] = raw_quality(source, target, fixture["extent"])
    report["trajectory_metrics"] = {
        "max_displacement": float(np.linalg.norm(frames[-1]-source, axis=1).max()),
        "max_center_drift": float(np.linalg.norm(frames.mean(1)-source.mean(0), axis=1).max()),
        "max_stray_fraction": max(metrics.stray_frac(x) for x in frames),
        "rms_acceleration": float(np.sqrt(np.mean(np.diff(velocities, axis=0)**2))/prm.dt),
        "interior_control_max": float(result["control"][:, ~mt].abs().max())}
    write_json(out/"report.json", report)
    np.savez_compressed(out/"trajectory.npz", **data, frames=frames, F_geom_frames=fg,
                         F_model_frames=fm, velocities=velocities,
                         controls=result["control"].cpu().numpy()[None],
                         coefficients=result["coefficients"].cpu().numpy(),
                         **({"surface_frames": np.stack([a.numpy() for a in tr.surface_x]),
                             "surface_F_frames": np.stack([a.numpy() for a in tr.surface_F]),
                             "surface_density": np.stack([a.numpy() for a in tr.surface_density])} if use_surface else {}))
    print("RESULT", json.dumps({"initial": result["initial"], "final": result["final"],
                                "validation": report["validation"], "health": result["health"],
                                "trajectory": report["trajectory_metrics"]}), flush=True)


def validate_pair_metadata(im, pm, reference_mode="matched"):
    for key in ("fixture", "steps", "camera_radius", "training_views", "basis_sha256", "source_sha256", "script_sha256", "raster_sha256"):
        if im[key] != pm[key]:
            raise ValueError(f"unmatched {key}")
    if im.get("observation", "material") != pm.get("observation", "material"):
        raise ValueError("unmatched Gaussian observation")
    if im.get("temporal_phases", 1) != pm.get("temporal_phases", 1):
        raise ValueError("unmatched temporal control basis")
    for key, default in (("image_metric", "pixel"), ("screen_length", .08)):
        if im.get(key, default) != pm.get(key, default):
            raise ValueError(f"unmatched {key}")
    a, b = dict(im["optimizer"]), dict(pm["optimizer"])
    if a.pop("objective") != "image" or b.pop("objective") != "physics":
        raise ValueError("comparison requires image versus physics objectives")
    ia, ib = a.pop("iterations"), b.pop("iterations")
    expected = ia if reference_mode == "matched" else 2*ia
    if ib != expected or a != b:
        raise ValueError("unmatched optimizer configuration or reference budget")


def compare(args):
    image_dir, physics_dir = Path(args.image), Path(args.physics)
    im = json.loads((image_dir/"metadata.json").read_text())
    pm = json.loads((physics_dir/"metadata.json").read_text())
    validate_pair_metadata(im, pm, args.reference_mode)
    with np.load(image_dir/"trajectory.npz") as a, np.load(physics_dir/"trajectory.npz") as b:
        image_x, physics_x, target = a["frames"][-1], b["frames"][-1], a["tgt"]
        for key in ("src", "tgt", "vol0", "surface_mask", "target_surface_mask"):
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f"unmatched {key}")
    if args.split == "test":
        if not args.lock or not Path(args.lock).exists():
            raise ValueError("sealed evaluation requires a prior configuration-lock artifact")
        raise ValueError("test evaluation is disabled during development; freeze protocol first")
    qi = raw_quality(image_x, target, im["fixture"]["extent"], args.split)
    qp = raw_quality(physics_x, target, im["fixture"]["extent"], args.split)
    ep, ei = qp["point_silhouette_error_128"], qi["point_silhouette_error_128"]
    improvement = (ep-ei)/ep if ep > 1e-6 else None
    ir = json.loads((image_dir/"report.json").read_text())
    pr = json.loads((physics_dir/"report.json").read_text())
    target_holes = metrics.hole_frac(target, im["fixture"]["extent"])
    gates = {
        "relative_error_20_percent": improvement is not None and improvement >= .2,
        "footprint_sensitivity": all(qi[f"point_silhouette_error_{r}"] <= qp[f"point_silhouette_error_{r}"] for r in (96, 160)),
        "chamfer_no_regression": qi["chamfer"] <= qp["chamfer"]*1.02,
        "hole_no_regression": qi["hole_fraction"] <= max(qp["hole_fraction"], target_holes)+.005,
        "no_projected_clipping": qi["projected_outside_max"] == qp["projected_outside_max"] == 0.,
        "stray_no_regression": ir["trajectory_metrics"]["max_stray_fraction"] <= pr["trajectory_metrics"]["max_stray_fraction"]+.001,
        "physical_health": ir["health"]["valid"] and pr["health"]["valid"]}
    result = {"split": args.split, "image": qi, "physics": qp,
              "reference_mode": args.reference_mode, "gates": gates,
              "relative_error_reduction": improvement,
              "absolute_iou_increase": qi["point_silhouette_iou_128"]-qp["point_silhouette_iou_128"],
              "20_percent_numerical_threshold": gates["relative_error_20_percent"],
              "all_numerical_development_gates": all(gates.values()),
              "evaluation_script_sha256": sha(__file__),
              "quality_claim": "development only; topology/trajectory/visual and independent-seed gates remain"}
    write_json(args.out, result)
    print(json.dumps(result), flush=True)


def export(args):
    folder = Path(args.out)
    meta = json.loads((folder/"metadata.json").read_text())
    fixture = meta["fixture"]
    with np.load(folder/"trajectory.npz") as a:
        frames, fg, target = a["frames"], a["F_geom_frames"], a["tgt"]
        mask, tmask = a["surface_mask"], a["target_surface_mask"]
        use_surface = meta.get("observation") == "material_surface"
        if use_surface:
            surface_frames, surface_faces = a["surface_frames"], a["surface_faces"]
            target_surface_vertices, target_surface_faces = a["target_surface_vertices"], a["target_surface_faces"]
    render = GaussViews([views("validation")[0]], meta["camera_radius"]/2.6,
                        fixture["sigma0"], 1024, "cuda")
    identity = torch.eye(3, device="cuda").repeat(len(target), 1, 1)
    render.bake_targets(torch.tensor(target, device="cuda"), identity, torch.tensor(tmask, device="cuda"))
    if use_surface:
        render = MaterialSurfaceViews([views("validation")[0]], meta["camera_radius"], surface_faces,
                                      target_surface_vertices, target_surface_faces, 1024)
    (folder/"frames").mkdir(exist_ok=True)
    def save(path, image):
        path.write_bytes(encode_png(image.permute(1, 2, 0).cpu().numpy()))
    save(folder/"target.png", render.targets[0])
    with torch.no_grad():
        for i, (x, f) in enumerate(zip(frames, fg)):
            covariance_F = torch.tensor(f, device="cuda") if meta.get("observation", "material") == "material" else identity
            img = (render.render(torch.tensor(surface_frames[i], device="cuda"), render.cams[0]) if use_surface else
                   render._render(torch.tensor(x, device="cuda"), render.cams[0], covariance_F, torch.tensor(mask, device="cuda")))
            save(folder/"frames"/f"frame_{i:05d}.png", img)
    write_json(folder/"qa_manifest.json", {"frame_count": len(frames), "resolution": 1024,
        "files": [f"frames/frame_{i:05d}.png" for i in range(len(frames))], "visual_review": "pending"})
    # Compatibility with the existing saved-frame monitor.
    meta.update(display_resolution=1024, discretization=fixture["discretization"])
    write_json(folder/"metadata.json", meta)
    print("EXPORTED", len(frames), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("prepare", "prepare_closed", "run", "compare", "export"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--assets", default="../../assets")
    ap.add_argument("--fixture")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--sigma_scale", type=float, default=1.2)
    ap.add_argument("--dx", type=float, default=.25)
    ap.add_argument("--dt", type=float, default=1/120)
    ap.add_argument("--loss_dx", type=float, default=.25)
    ap.add_argument("--material_surface", action="store_true")
    ap.add_argument("--repaired_target")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--patches", type=int, default=12)
    ap.add_argument("--phases", type=int, default=1)
    ap.add_argument("--minimum_physics_progress", type=float, default=0.)
    ap.add_argument("--geometric_constraints", action="store_true")
    ap.add_argument("--surface_constraints", action="store_true")
    ap.add_argument("--image_metric", choices=("pixel", "screened"), default="pixel")
    ap.add_argument("--screen_length", type=float, default=.08)
    ap.add_argument("--fd_strain", type=float, default=.001)
    ap.add_argument("--objective", choices=("image", "physics"), default="image")
    ap.add_argument("--observation", choices=("material", "fixed_kernel", "material_surface"), default="material")
    ap.add_argument("--image")
    ap.add_argument("--physics")
    ap.add_argument("--split", choices=("validation", "test"), default="validation")
    ap.add_argument("--lock")
    ap.add_argument("--reference_mode", choices=("matched", "strong_physics"), default="matched")
    args = ap.parse_args()
    if socket.gethostname() != "hyde06":
        raise SystemExit("all benchmark operations run on hyde06")
    torch.set_num_threads(4)
    globals()[args.action](args)


if __name__ == "__main__":
    main()
