"""gradcheck.py --tgt MESH --n N — directional finite differences against autograd for each channel of the window
objective at the first window of a run (the source state): the physics core, the render term and the cleanup, on
each control leaf (dFc, u). Central differences through the no-grad rollout (the line-search path) against the
gradient through the tape adjoint (the gradient path); ratio analytic / FD per channel, leaf and step size. The
float32 rollout's replay noise sets the floor: a ratio near 1 at two step sizes is the pass.
"""
import physmorph  # noqa: F401  (before torch: CuPy's CUDA 12 NVRTC)
import argparse

import torch

from physmorph import gpu
from physmorph.mpm.traj import compute_rest_volumes
from physmorph.pipeline import PipelineConfig
from physmorph.pipeline.run.state import fragment_mask
from physmorph.pipeline.target import build_target, calibrate_units
from physmorph.pipeline.window.objective import Objective
from physmorph.pipeline.window.rollout import eval_terms, graph_terms
from physmorph.pipeline.window.setup import StartState, Window
from physmorph.prepare import prepare

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="assets/isosphere.obj"); ap.add_argument("--tgt", default="assets/bunny.obj")
ap.add_argument("--n", type=int, default=40000); ap.add_argument("--seed", type=int, default=97)
ap.add_argument("--scales", default="1e-3,3e-3", help="steps: relative change of the checked value")
ap.add_argument("--leaves", default="dFc,u")
ap.add_argument("--volume_exact", action="store_true", help="D129: the stress reads the tracked volume")
ap.add_argument("--J0", type=float, default=1.0,
                help="D129: the start state's tracked volume (a uniform value; 1 = the source's)")
a = ap.parse_args()
cfg0 = PipelineConfig()
prep = prepare(a.src, a.tgt, a.n, a.seed, 26.0, cfg0.young, cfg0.poisson, log=lambda s: None)
cfg = PipelineConfig(loss_res=prep.loss_res, unit_ref_res=prep.unit_ref_res, nn_berth_k=prep.nn_berth_k,
                     volume_exact=a.volume_exact)
prm, x = prep.prm, gpu.tensor(prep.src)
tgt = build_target(prep.tgt, prm, cfg)
calibrate_units(tgt, x, cfg)
N = len(x)
coh = gpu.knn(x, cfg.coh_k + 1)[1][:, 1:]
bonds = (coh, (x[coh] - x[:, None]).norm(dim=2), fragment_mask(x, prm).float())
J0 = torch.full((N,), a.J0, device="cuda") if a.volume_exact and a.J0 != 1.0 else None
win = Window(StartState(x=x, Fp=torch.eye(3, device="cuda").repeat(N, 1, 1), J=J0), prm, cfg, tgt,
             compute_rest_volumes(x, 1.0, prm), bonds)
obj = Objective(win)
dFc = torch.zeros(cfg.T, N, 3, 3, device="cuda", requires_grad=True)
u = torch.zeros(N, device="cuda", requires_grad=True)


def channels(e):
    return {"physics": obj.phys_core(e), "render": e.lr,
            "cleanup": obj.cleanup(e.xT)}


e = graph_terms(win, obj, dFc, u)
grads = {k: torch.autograd.grad(v, [dFc, u], retain_graph=True) for k, v in channels(e).items()}
print(f"N={N}  loss grid {cfg.loss_res}^3  unit ratio {tgt.unit_ratio:.4g}")
values = {k: float(v) for k, v in channels(e).items()}
total = [sum(g[i] for g in grads.values()) for i in range(2)]
for leaf_i, leaf in enumerate(("dFc", "u")):
    if leaf not in a.leaves.split(","):
        continue
    # directions: each channel's own gradient on this leaf, and the sum of the channels (the update direction);
    # steps that change the checked value by ~1e-3 and ~3e-3 of itself, far above the replay noise (~1e-6)
    dirs = {**{f"g_{k}": g[leaf_i] for k, g in grads.items()}, "g_total": total[leaf_i]}
    for dname, g_dir in dirs.items():
        d = g_dir / g_dir.norm().clamp_min(1e-30)
        for scale in [float(s) for s in a.scales.split(",")]:
            row = []
            for name, g in grads.items():
                an = float((g[leaf_i] * d).sum())
                if abs(an) < 1e-30:
                    continue
                h = scale * abs(values[name]) / abs(an)
                with torch.no_grad():
                    p = [dFc + h * d, u] if leaf_i == 0 else [dFc, u + h * d]
                    m = [dFc - h * d, u] if leaf_i == 0 else [dFc, u - h * d]
                    fd = float(channels(eval_terms(win, obj, *p))[name] - channels(eval_terms(win, obj, *m))[name]) / (2 * h)
                row.append(f"{name} {an / fd if fd else float('nan'):.3f}")
            print(f"{leaf:3s} along {dname:10s} rel step {scale:.0e}:  " + "   ".join(row), flush=True)
