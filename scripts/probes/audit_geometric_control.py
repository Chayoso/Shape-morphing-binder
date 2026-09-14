"""Native CUDA geometry/control derivative audit. Run on hyde06 only."""
from dataclasses import asdict, replace
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from physmorph.mpm.function import RolloutSpec, warp_mpm_geometry
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import compute_rest_volumes
from physmorph.pipeline.gauss_loss import GaussViews, _gs
from physmorph.pipeline.geometric import forward_geometry
from physmorph.pipeline.render_loss import make_views
from physmorph.pipeline.runner import _surface_weights


def scalar(v):
    return float(v.detach())


def relative(a, b):
    return abs(a-b)/max(abs(a), abs(b), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--children", type=int, default=1, choices=(1, 4))
    args = ap.parse_args()
    if socket.gethostname() != "hyde06":
        raise SystemExit("Run derivative rollouts on hyde06 only")
    torch.set_num_threads(4)
    rng = np.random.default_rng(914)
    n, horizon = 256, 3
    source = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    target = source*np.array([1.13, .84, 1.04], np.float32)
    prm = MPMParams(dx=.75, dt=1/120, nx=16, ny=16, nz=16, grid_min=(-6.,)*3)
    spec = RolloutSpec(source, 1., 800., 400., prm, horizon, device="cuda",
                       vol0=compute_rest_volumes(source, 1., prm, "cuda"))
    mask = _surface_weights(source, 16, .4, 0.) > .5
    tmask = _surface_weights(target, 16, .4, 0.) > .5
    mt = torch.tensor(mask, device="cuda")
    bundle = GaussViews(make_views(2, (0., .35)), 1.8, .16, args.res, "cuda",
                        child_count=args.children)
    bundle.configure_source(source, mask)
    bundle.bake_targets(torch.tensor(target, device="cuda"),
                         F=torch.eye(3, device="cuda").repeat(n, 1, 1),
                         mask=torch.tensor(tmask, device="cuda"))
    def images(x, fg):
        return torch.stack([bundle._render(x.contiguous(), cam, fg, mt,
                                           bundle.source_offsets if args.children > 1 else None)
                            for cam in bundle.cams])
    target_img = torch.stack(bundle.targets)
    c = torch.tensor(rng.normal(0, .007, (horizon, n, 3, 3)).astype(np.float32),
                      device="cuda", requires_grad=True)
    state = warp_mpm_geometry(c, spec)
    picture = images(state[0], state[3])
    losses = {"l1": (picture-target_img).abs().mean(),
              "l2": .5*(picture-target_img).square().mean()}
    records = {}
    for name, loss in losses.items():
        seeds = torch.autograd.grad(loss, (state[0], state[3]), retain_graph=True)
        parts = [torch.autograd.grad(state[k], c, seed, retain_graph=True)[0]
                 for k, seed in zip((0, 3), seeds)]
        g = torch.autograd.grad(loss, c, retain_graph=True)[0]
        fixed_q = torch.autograd.grad(loss, picture, retain_graph=True)[0].detach()
        entry = {"loss": scalar(loss), "control_norm": scalar(g.norm()),
                 "x_path_norm": scalar(parts[0].norm()), "geom_path_norm": scalar(parts[1].norm()),
                 "chain_relative_error": scalar((g-parts[0]-parts[1]).norm()/g.norm()),
                 "endpoint_interior_norms": [scalar(s[~mt].norm()) for s in seeds],
                 "raw_control_interior_norm": scalar(g[:, ~mt].norm()), "checks": []}
        # Control direction gives separately: fixed endpoint covector -> MPM,
        # fixed pixel covector -> full pipeline, and the actual image objective.
        direction = g.detach()/g.norm()
        analytic = scalar((g*direction).sum())
        for eps in (.003, .01, .03):
            with torch.no_grad():
                sp, _ = forward_geometry(c+eps*direction, spec)
                sm, _ = forward_geometry(c-eps*direction, spec)
                ip, im = images(sp[0], sp[3]), images(sm[0], sm[3])
                mp = sum(((sp[k]-sm[k]).double()*q.double()).sum()
                         for k, q in zip((0, 3), seeds))/(2*eps)
                pix = ((ip-im).double()*fixed_q.double()).sum()/(2*eps)
                fn = (lambda i: (i-target_img).abs().double().mean()) if name == "l1" else (
                     lambda i: .5*(i-target_img).square().double().mean())
                actual = (fn(ip)-fn(im))/(2*eps)
            entry["checks"].append({"epsilon": eps, "analytic": analytic,
                "mpm_fixed_endpoint_fd": scalar(mp), "mpm_relative_error": relative(scalar(mp), analytic),
                "fixed_pixel_fd": scalar(pix), "fixed_pixel_relative_error": relative(scalar(pix), analytic),
                "actual_loss_fd": scalar(actual), "actual_relative_error": relative(scalar(actual), analytic)})
        # Renderer-only directional checks for x and F_geom, with frozen pixel q.
        entry["renderer_endpoint_checks"] = []
        for index, seed in zip((0, 3), seeds):
            d = seed/seed.norm()
            expected = scalar(seed.norm())
            for eps in (.001, .003, .01):
                endpoints = [state[0].detach(), state[3].detach()]
                j = 0 if index == 0 else 1
                with torch.no_grad():
                    plus, minus = list(endpoints), list(endpoints)
                    plus[j] = plus[j]+eps*d
                    minus[j] = minus[j]-eps*d
                    fd = ((images(*plus)-images(*minus)).double()*fixed_q.double()).sum()/(2*eps)
                entry["renderer_endpoint_checks"].append({"endpoint": "x" if index == 0 else "F_geom",
                    "epsilon": eps, "analytic": expected, "fd": scalar(fd),
                    "relative_error": relative(scalar(fd), expected)})
        records[name] = entry
        print(name, json.dumps(entry), flush=True)
    stopped = {}
    for name, stopped_spec in (("dt_zero", replace(spec, prm=replace(prm, dt=0.))),
                                ("stiffness_zero", replace(spec, lam=0., mu=0.))):
        fixed = warp_mpm_geometry(c, stopped_spec)
        loss = (images(fixed[0], fixed[3])-target_img).abs().mean()
        g = torch.autograd.grad(loss, c)[0]
        stopped[name] = {"max_displacement": scalar((fixed[0]-torch.tensor(source, device="cuda")).abs().max()),
                         "max_geom_change": scalar((fixed[3].reshape(-1, 3, 3)-torch.eye(3, device="cuda")).abs().max()),
                         "render_control_norm": scalar(g.norm())}
    mod = _gs()[0]
    ext = importlib.import_module(mod.__name__+"._C")
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = {"discretization": {**asdict(prm), "N": n, "T": horizon, "lam": 800., "mu": 400.,
               "sigma0": .16, "res_requested": args.res, "res_actual": bundle.res,
               "children": args.children, "views": len(bundle.cams), "surface_count": int(mask.sum())},
              "losses": records, "stopped": stopped,
              "provenance": {"torch": torch.__version__, "warp": wp.__version__,
                  "gpu": torch.cuda.get_device_name(), "raster_module": mod.__file__,
                  "raster_sha256": sha(ext.__file__), "script_sha256": sha(__file__),
                  "source_sha256": {str(p.relative_to(ROOT)): sha(p)
                                      for p in (ROOT/"physmorph").rglob("*.py")}}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved", out, flush=True)


if __name__ == "__main__":
    main()
