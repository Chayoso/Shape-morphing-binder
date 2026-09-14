"""Read-only derivative audit of the original pipeline; run only on hyde06.

No optimization, parameter tuning, state promotion, or rendered deliverables.
The source checkout must be the original eca06f5 revision. Extra code in this
script only measures its loss/adjoint branches; it does not replace them.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import socket
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from physmorph.losses.volumetric import d_vol, target_mass_grid
from physmorph.mpm.function import RolloutSpec, warp_mpm_full
from physmorph.mpm.state import MPMParams
from physmorph.mpm.constitutive import lame
from physmorph.mpm.traj import compute_rest_volumes
from physmorph.pipeline.gauss_loss import GaussViews, _gs
from physmorph.pipeline.grid_smooth import smooth_particle_field
from physmorph.pipeline.render_loss import d_render, make_views, target_silhouettes
from physmorph.pipeline.runner import _surface_weights


def norm(x):
    return 0.0 if x is None else float(x.detach().norm())


def rel(a, b):
    return norm(a - b) / max(norm(a), norm(b), 1e-20)


def cosine(a, b):
    return float((a * b).sum()) / max(norm(a) * norm(b), 1e-20)


def control_parts(loss, state, control):
    seeds = torch.autograd.grad(loss, state, retain_graph=True, allow_unused=True)
    parts = []
    for output, seed in zip(state, seeds):
        parts.append(torch.zeros_like(control) if seed is None else
                     torch.autograd.grad(output, control, seed, retain_graph=True)[0])
    full = torch.autograd.grad(loss, control, retain_graph=True)[0]
    return seeds, parts, full


def cpp_update_algebra():
    """Compare source expressions to autodiff; this does NOT execute the .so."""
    gen = torch.Generator().manual_seed(94)
    F = (torch.eye(3, dtype=torch.float64) + .1 * torch.randn(3, 3, generator=gen,
                                                           dtype=torch.float64)).requires_grad_()
    c = (.02 * torch.randn(3, 3, generator=gen, dtype=torch.float64)).requires_grad_()
    C = (.1 * torch.randn(3, 3, generator=gen, dtype=torch.float64)).requires_grad_()
    seed = torch.randn(3, 3, generator=gen, dtype=torch.float64)
    dt, smoothing = 1 / 240, .955
    A = torch.eye(3, dtype=torch.float64) + dt * C
    following = (1 - smoothing) * A @ (F + c) + smoothing * F
    gF, gc, gC = torch.autograd.grad((following * seed).sum(), (F, c, C))
    source_gF = (1 - smoothing) * A.T @ seed + smoothing * seed
    source_gC = dt * seed @ (F + c).T
    return {"scope": "isolated source-formula check; prebuilt C++ not executed",
            "dt": dt, "smoothing": smoothing,
            "old_F_gradient_relative_error": rel(gF, source_gF),
            "C_gradient_norm_source_over_true": norm(source_gC) / norm(gC),
            "control_gradient_norm_old_F_over_true": norm(source_gF) / norm(gc),
            "control_gradient_relative_error_if_old_F_reused": rel(source_gF, gc),
            "expected_missing_smoothing_factor": 1 - smoothing}


def environment_record():
    package = _gs()[0]
    extension = importlib.import_module(package.__name__ + "._C")
    return {"torch": torch.__version__, "warp": wp.__version__,
            "raster_module": package.__file__,
            "raster_module_sha256": hashlib.sha256(Path(package.__file__).read_bytes()).hexdigest(),
            "raster_extension": extension.__file__,
            "raster_extension_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "gpu": torch.cuda.get_device_name()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--children", type=int, default=1, choices=(1, 4))
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--initial_archive", default="",
                    help="Read only src/tgt from a 12k or larger archive; use production discretization")
    ap.add_argument("--fd_entries", type=int, default=3)
    ap.add_argument("--directional", action="store_true",
                    help="Resolve weak scalar finite differences with larger-signal unit directions")
    args = ap.parse_args()
    if not socket.gethostname().startswith("hyde06"):
        raise SystemExit("Run derivative rollouts on hyde06 only.")
    expected = {
        "pipeline/optimizer.py": "8ead488cbfcb2572aaab14337c0bc12643c9e0875c9d376681c677511b236ec1",
        "pipeline/gauss_loss.py": "c5310e9bfe294afa2057bde60fae26a062f1166ff9e8c3b9fbd28adef002d4d8",
        "mpm/kernels.py": "d90b6f73d898f654821ba4f466271e48908fc90e263de71c19d7db84ac9bed59",
    }
    for name, digest in expected.items():
        if hashlib.sha256((ROOT / "physmorph" / name).read_bytes()).hexdigest() != digest:
            raise SystemExit("Source differs from audit baseline eca06f5: " + name)
    if args.fd_entries < 0:
        raise SystemExit("fd_entries must be nonnegative")
    torch.set_num_threads(4)
    dev = "cuda"
    rng = np.random.default_rng(20260914)
    n, horizon = 256, 3
    source = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    target = source * np.array([1.13, .84, 1.04], np.float32)
    fraction, neighbor_k, extent = .4, 16, 1.8
    lam, mu, sigma = 800., 400., .16
    loss_dx, loss_dims = .75, (16,) * 3
    prm = MPMParams(dx=.75, dt=1 / 120, nx=16, ny=16, nz=16,
                    grid_min=(-6., -6., -6.))
    views = make_views(2, (0., .35))
    if args.initial_archive:
        with np.load(args.initial_archive, allow_pickle=False) as archive:
            source, target = archive["src"].copy(), archive["tgt"].copy()
        n, horizon = len(source), 20
        fraction, neighbor_k = .3, 24
        extent = float(np.abs(target).max()) * 1.25
        lam, mu = lame(1.4e5, .2)
        prm = MPMParams(dx=.5, dt=1 / 240, nx=64, ny=64, nz=64,
                        grid_min=(-16., -16., -16.))
        loss_dx, loss_dims = 32 / 48, (48,) * 3
        views = make_views(2, (0., .5, -.5))
    mask_np = _surface_weights(source, neighbor_k, fraction, .05) > .5
    target_mask_np = _surface_weights(target, neighbor_k, fraction, .05) > .5
    if args.initial_archive:
        from physmorph.render.covariance import sigma0_from_nn
        sigma = sigma0_from_nn(target[target_mask_np], 1.)
    mask = torch.as_tensor(mask_np, device=dev)
    target_mask = torch.as_tensor(target_mask_np, device=dev)
    vol0 = compute_rest_volumes(source, 1., prm, dev)
    spec = RolloutSpec(x0=source, m=1., lam=lam, mu=mu, prm=prm, T=horizon,
                       device=dev, vol0=vol0)
    render = GaussViews(views, extent, sigma, args.res, dev, child_count=args.children)
    target_t = torch.as_tensor(target, device=dev)
    render.bake_targets(target_t, mask=target_mask)
    if args.children > 1:
        render.configure_source(source, mask_np)
    control = torch.tensor(rng.normal(0, .001, (horizon, n, 3, 3)).astype(np.float32),
                           device=dev, requires_grad=True)
    state = warp_mpm_full(control, spec)
    x, F, v = state
    gmin = torch.tensor(prm.grid_min, device=dev)
    mass = torch.ones(n, device=dev)
    grid = target_mass_grid(target_t, mass, gmin, loss_dx, loss_dims)
    silhouettes = target_silhouettes(target_t, views, 64, extent)
    losses = {
        "volume_sum": d_vol(x, mass, grid, gmin, loss_dx, loss_dims),
        "silhouette_full_cloud_mean": d_render(x, silhouettes, views, 64, extent),
        "gaussian_surface_mean": render.loss(x, F, mask=mask),
        "kinetic_mean": v.square().sum(1).mean(),
    }
    losses["physics_core_example"] = (losses["volume_sum"] + 20 * losses["kinetic_mean"]
                                       + .001 * control.square().sum() / (horizon * n))
    results, gradients, raw = {}, {}, {}
    for name, loss in losses.items():
        seeds, parts, full = control_parts(loss, state, control)
        # physics_core_example also has a direct control regularizer.
        explicit = (.002 * control / (horizon * n)
                    if name == "physics_core_example" else torch.zeros_like(control))
        results[name] = {"value": float(loss.detach()), "control_norm": norm(full),
                         "endpoint_norms_x_F_v": [norm(z) for z in seeds],
                         "control_path_norms_x_F_v": [norm(z) for z in parts],
                         "control_norm_per_layer": [norm(z) for z in full],
                         "chain_relative_error": rel(full, sum(parts) + explicit),
                         "interior_control_norm": norm(full[:, ~mask]),
                         "surface_control_norm": norm(full[:, mask]),
                         "endpoint_interior_norms_x_F_v": [norm(z[~mask]) if z is not None
                                                            else 0.0 for z in seeds]}
        gradients[name], raw[name] = full, (seeds, parts)

    gs_name, phys_name = "gaussian_surface_mean", "physics_core_example"
    gr, gp = gradients[gs_name], gradients[phys_name]
    gx, gF, _ = raw[gs_name][0]
    smoothed = smooth_particle_field(x.detach(), gx * mask[:, None], gmin, loss_dx,
                                     loss_dims, 8, 4.)
    original_smooth = torch.autograd.grad(x, control, smoothed, retain_graph=True)[0]
    F_path = raw[gs_name][1][1]
    repeated = torch.autograd.grad(losses[gs_name], control, retain_graph=True)[0]
    summary = {
        "physics_over_render_control_norm": norm(gp) / norm(gr),
        "physics_render_control_cosine": cosine(gp, gr),
        "render_position_covariance_path_cosine": cosine(raw[gs_name][1][0], F_path),
        "repeated_backward_relative_error": rel(gr, repeated),
        "surface_observation_interior_control_fraction": norm(gr[:, ~mask]) / norm(gr),
        "smoothing_original_control_norm": norm(original_smooth),
        "smoothing_missing_F_path_norm": norm(F_path),
        "smoothing_omission_relative_to_two_seed_pullback":
            norm(F_path) / max(norm(original_smooth + F_path), 1e-20),
        "smoothed_endpoint_interior_norm": norm(smoothed[~mask]),
        "volume_sum_to_grid_mean_factor": int(np.prod(loss_dims)),
        "gaussian_mean_to_all_view_rgb_sum_factor": len(views) * 3 * render.res ** 2,
        "note": "sum/mean factors are algebraic diagnostics, not proposed gradient weights",
    }
    fd_records = []
    for name in ("volume_sum", gs_name):
        grad = gradients[name]
        for entry in grad.abs().flatten().topk(args.fd_entries).indices.tolist():
            analytic = float(grad.flatten()[entry])
            estimates = []
            for epsilon in (.001, .003, .01):
                values = []
                with torch.no_grad():
                    for sign in (1, -1):
                        dc = control.detach().clone()
                        dc.flatten()[entry] += sign * epsilon
                        xx, ff, _ = warp_mpm_full(dc, spec)
                        ll = (d_vol(xx, mass, grid, gmin, loss_dx, loss_dims)
                              if name == "volume_sum" else render.loss(xx, ff, mask=mask))
                        values.append(float(ll))
                fd = (values[0] - values[1]) / (2 * epsilon)
                estimates.append({"epsilon": epsilon, "fd": fd,
                                  "relative_error": abs(fd - analytic) /
                                  max(abs(fd), abs(analytic), 1e-12)})
            fd_records.append({"term": name, "index": list(np.unravel_index(entry, control.shape)),
                               "analytic": analytic, "estimates": estimates,
                               "two_of_three_below_8_percent": sum(
                                   r["relative_error"] < .08 for r in estimates) >= 2})

    directional = None
    if args.directional:
        def render_control_value(dc):
            with torch.no_grad():
                xx, ff, _ = warp_mpm_full(dc, spec)
                return float(render.loss(xx, ff, mask=mask))

        def seeded_mpm_value(dc):
            with torch.no_grad():
                xx, ff, _ = warp_mpm_full(dc, spec)
                # A fixed linear terminal covector isolates MPM from raster changes.
                return float(((xx.double() - x.detach().double()) * gx.detach().double()).sum()
                             + ((ff.double() - F.detach().double()) * gF.detach().double()).sum())

        replays = [render_control_value(control.detach()) for _ in range(5)]
        records = []
        for label, vector in (("full_render", gr), ("render_x_path", raw[gs_name][1][0]),
                               ("render_F_path", F_path)):
            direction = vector.detach() / max(norm(vector), 1e-20)
            analytic = float((gr * direction).sum())
            checks = []
            for epsilon in (.005, .02, .05):
                plus = [render_control_value(control.detach() + epsilon * direction) for _ in range(2)]
                minus = [render_control_value(control.detach() - epsilon * direction) for _ in range(2)]
                fd = (np.mean(plus) - np.mean(minus)) / (2 * epsilon)
                linear_fd = (seeded_mpm_value(control.detach() + epsilon * direction)
                             - seeded_mpm_value(control.detach() - epsilon * direction)) / (2 * epsilon)
                checks.append({"epsilon_l2_control": epsilon, "fd": float(fd),
                               "plus_replays": plus, "minus_replays": minus,
                               "relative_error": abs(fd - analytic) /
                                   max(abs(fd), abs(analytic), 1e-12),
                               "fixed_seed_mpm_fd": linear_fd,
                               "fixed_seed_mpm_relative_error": abs(linear_fd - analytic) /
                                   max(abs(linear_fd), abs(analytic), 1e-12)})
            records.append({"direction": label, "analytic": analytic, "checks": checks})
        endpoint = []
        for label, vector in (("x", gx), ("F", gF)):
            direction = vector.detach() / max(norm(vector), 1e-20)
            estimates = []
            for epsilon in (.0001, .001, .01):
                with torch.no_grad():
                    if label == "x":
                        lp = render.loss(x.detach() + epsilon * direction, F.detach(), mask=mask)
                        lm = render.loss(x.detach() - epsilon * direction, F.detach(), mask=mask)
                    else:
                        lp = render.loss(x.detach(), F.detach() + epsilon * direction, mask=mask)
                        lm = render.loss(x.detach(), F.detach() - epsilon * direction, mask=mask)
                    fd = float((lp - lm) / (2 * epsilon))
                estimates.append({"epsilon_l2_endpoint": epsilon, "fd": fd,
                                  "relative_error": abs(fd - norm(vector)) /
                                  max(abs(fd), norm(vector), 1e-12)})
            endpoint.append({"endpoint": label, "analytic": norm(vector), "checks": estimates})
        directional = {"identical_control_loss_replays": replays,
                       "loss_replay_range": max(replays) - min(replays), "records": records,
                       "renderer_only_endpoint_checks": endpoint}

    bypass = []
    for label, step_dt, stiffness in (("dt_zero", 0., (lam, mu)),
                                      ("zero_stiffness", prm.dt, (0., 0.))):
        test_spec = dataclasses.replace(spec, T=1, lam=stiffness[0], mu=stiffness[1],
                                         prm=dataclasses.replace(prm, dt=step_dt))
        dc = torch.zeros(1, n, 3, 3, device=dev, requires_grad=True)
        xx, ff, vv = warp_mpm_full(dc, test_spec)
        loss = render.loss(xx, ff, mask=mask)
        derivative = torch.autograd.grad(loss, dc)[0]
        with torch.no_grad():
            delta = torch.zeros_like(dc)
            delta[..., 0, 0] = .1
            xc, fc, _ = warp_mpm_full(delta, test_spec)
            alternate = render.loss(xc, fc, mask=mask)
        bypass.append({"case": label, "T": 1, "dt": step_dt,
                       "lambda": stiffness[0], "mu": stiffness[1],
                       "position_change_max": float((xc - xx).abs().max()),
                       "F_change_max": float((fc - ff).abs().max()),
                       "render_loss_before": float(loss.detach()),
                       "render_loss_after_control_0_1_xx": float(alternate),
                       "render_control_gradient_norm": norm(derivative),
                       "velocity_norm": norm(vv)})

    paths = sorted(p.relative_to(ROOT / "physmorph").as_posix()
                   for p in (ROOT / "physmorph").rglob("*.py"))
    result = {"baseline": "eca06f5", "host": socket.gethostname(),
              "source_hashes": {p: hashlib.sha256((ROOT / "physmorph" / p).read_bytes()).hexdigest()
                                for p in paths},
              "environment": environment_record(),
              "discretization": {"N": n, "T": horizon, "mpm": dataclasses.asdict(prm),
                                  "loss_grid": loss_dims, "loss_dx": loss_dx,
                                  "sigma0": sigma, "extent": extent, "gauss_res_requested": args.res,
                                  "gauss_res_actual": render.res, "sil_res": 64,
                                  "views": len(views), "children": args.children,
                                  "surface_count": int(mask.sum()), "lambda": lam, "mu": mu,
                                  "mass_per_particle": 1., "control_initial_std": .001},
              "input": {"initial_archive": args.initial_archive,
                        "source_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
                        "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
                        "used_archive_fields": ["src", "tgt"] if args.initial_archive else []},
              "terms": results, "summary": summary, "finite_differences": fd_records,
              "directional_finite_differences": directional,
              "F_control_bypass": bypass, "cpp_update_algebra": cpp_update_algebra()}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=int), encoding="utf-8")
    print(json.dumps({"out": str(output), "summary": summary,
                      "fd_pass": (all(r["two_of_three_below_8_percent"] for r in fd_records)
                                  if fd_records else None),
                      "bypass": bypass}, indent=2), flush=True)


if __name__ == "__main__":
    main()
