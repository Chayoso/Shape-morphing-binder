"""Causal one-step render intervention: identical initial state and shared gp/gr.

Run on hyde06. Only raw stress/state enter the physical comparison. Two forward
repeats measure observed variation, not a statistical upper bound on CUDA noise.
"""
from dataclasses import fields
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from physmorph.losses.volumetric import d_vol, target_mass_grid
from physmorph.mpm.function import RolloutSpec, warp_mpm_geometry
from physmorph.mpm.state import MPMParams
from physmorph.pipeline.gauss_loss import GaussViews
from physmorph.pipeline.geometric import GeometricConfig, forward_geometry, restrict_render_control, trajectory_health
from physmorph.pipeline.render_loss import make_views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if socket.gethostname() != "hyde06":
        raise SystemExit("All physical replays run on hyde06")
    torch.set_num_threads(4)
    ref = Path(args.reference)
    meta = json.loads((ref/"metadata.json").read_text())
    data = np.load(ref/"trajectory.npz", allow_pickle=False)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for name, digest in meta["provenance"]["source_sha256"].items():
        if sha(ROOT/name) != digest:
            raise ValueError("replay code differs from reference: "+name)
    desc, n = meta["discretization"], len(data["src"])
    prm = MPMParams(**{f.name: desc[f.name] for f in fields(MPMParams)})
    spec = RolloutSpec(data["src"], 1., desc["lam"], desc["mu"], prm, desc["T"],
                       device="cuda", vol0=data["vol0"])
    cfg = GeometricConfig(**meta["optimizer"])
    if meta["image_loss"] != "l2" or meta["children"] != 1:
        raise ValueError("this probe specifies the L2 parent-splat diagnostic")
    mask = torch.tensor(data["surface_mask"], device="cuda")
    target = torch.tensor(data["tgt"], device="cuda")
    tmask = torch.tensor(data["target_surface_mask"], device="cuda")
    mass = torch.ones(n, device="cuda")
    gmin = torch.tensor(prm.grid_min, device="cuda")
    grid = target_mass_grid(target, mass, gmin, desc["loss_dx"], tuple(desc["loss_dims"]))
    extent = float(np.abs(np.concatenate([data["src"], data["tgt"]])).max())*1.25
    bundle = GaussViews(make_views(2, (0., .35)), extent, meta["sigma0"], meta["loss_resolution"], "cuda")
    bundle.bake_targets(target, torch.eye(3, device="cuda").repeat(n, 1, 1), tmask)
    def losses(state, control):
        lp = (d_vol(state[0], mass, grid, gmin, desc["loss_dx"], tuple(desc["loss_dims"]))
              + .5*state[2].square().sum(1).mean() + .001*control.square().sum()/(n*spec.T))
        lr = torch.stack([.5*(bundle._render(state[0], cam, state[3], mask)-ti).square().mean()
                          for cam, ti in zip(bundle.cams, bundle.targets)]).mean()
        return lp, lr
    initial = torch.zeros(spec.T, n, 3, 3, device="cuda", requires_grad=True)
    state = warp_mpm_geometry(initial, spec)
    lp, lr = losses(state, initial)
    gp = torch.autograd.grad(lp, initial, retain_graph=True)[0]
    gr = torch.autograd.grad(lr, initial)[0]
    gs = restrict_render_control(gr, mask)
    def candidate(weight):
        # First-iteration RMS expression from optimize_geometric_window. Both
        # arms consume the SAME realized gradients and have zero initial moments.
        g = gp+weight*gs
        rms = (1-cfg.rms_decay)*g.square()
        denominator = (rms/(1-cfg.rms_decay)).sqrt()+cfg.rms_eps
        delta = -cfg.step_size*g/denominator
        return delta.detach(), float(((gp+weight*gr)*delta).sum())
    controls = [candidate(w) for w in (1., 0.)]
    def replay(control, weight, slope):
        with torch.no_grad():
            state, tr = forward_geometry(control, spec)
            health = trajectory_health(tr, cfg)
            if not health["valid"]:
                raise ValueError(f"invalid trial: {health}")
            np_, nr = losses(state, control)
            before, after = float(lp.detach()+weight*lr.detach()), float(np_+weight*nr)
            accepted = slope < 0 and after < before and after <= before+cfg.armijo*slope
        arrays = {"x": np.stack([a.numpy() for a in tr.x]),
                  "v": np.stack([a.numpy() for a in tr.v]),
                  "F_geom": np.stack([a.numpy() for a in tr.F_geom]),
                  "P_total": np.stack([a.numpy() for a in tr.P])}
        return arrays, {"accepted_at_full_step": accepted, "objective_before": before,
                        "objective_after": after, "render_after": float(nr), "health": health}
    runs = [[replay(c, w, slope) for _ in range(2)] for (c, slope), w in zip(controls, (1., 0.))]
    def norm(a):
        return float(np.linalg.norm(a.astype(np.float64)))
    def describe(a):
        return {"l2": norm(a), "max_component": float(np.abs(a).max())}
    effects = {}
    mask_np = data["surface_mask"]
    for key in runs[0][0][0]:
        difference = runs[0][0][0][key]-runs[1][0][0][key]
        repeat = [runs[i][0][0][key]-runs[i][1][0][key] for i in (0, 1)]
        effects[key] = {"on_minus_off": describe(difference),
            "joint_repeat": describe(repeat[0]), "physics_repeat": describe(repeat[1]),
            "effect_over_largest_observed_repeat_l2": norm(difference)/max(*(norm(a) for a in repeat), 1e-30),
            "surface_effect_l2": norm(difference[:, mask_np]), "interior_effect_l2": norm(difference[:, ~mask_np])}
    cd = (controls[0][0]-controls[1][0]).cpu().numpy()
    result = {"experiment": "same state and gp/gr; toggle render contribution in first joint RMS step",
        "discretization": desc, "windows": 1, "optimizer_steps": 1,
        "loss_resolution": bundle.res, "views": len(bundle.cams), "sigma0": meta["sigma0"],
        "surface_particles": int(mask_np.sum()), "image_loss": "l2", "children": 1,
        "initial_state_and_adjoints_shared": True, "replay_source_matches_reference": True,
        "control_delta": describe(cd), "interior_control_delta_max": float(np.abs(cd[:, ~mask_np]).max()),
        "interior_render_gradient_norm": float(gs[:, ~mask].norm()),
        "effects": effects, "trial_evaluations": [r[0][1] for r in runs],
        "limitations": ["Small one-step fixture; not evidence of useful full morph improvement.",
            "Two repeats measure observed forward variation, not a statistical noise bound.",
            "Directions share the same AD tensors; native raster branch accuracy remains a separate gate."],
        "provenance": {"script_sha256": sha(__file__), "archive_sha256": sha(ref/"trajectory.npz"),
                        "run_provenance": meta["provenance"]}}
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
