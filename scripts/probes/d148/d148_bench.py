"""d148_bench.py FB_DUMP_NPZ FB_FRAMES12 -- D148 cost: where the split adds time at 300k. On D135's dragon state (FB's
window-20 dump: x and its transport image) the window's MPM rollout (40 steps, 20 driven) with and without the split:
(1) the graph-captured eval rollout and the tape's forward + adjoint (PersistentAdjoint), seconds each (synchronised,
median of 5); (2) per kernel, Warp's CUDA timing of one uncaptured taped forward + backward, summed by kernel name."""
import sys
import time

import physmorph  # noqa: F401  (before torch)
import numpy as np
import torch
import warp as wp

from physmorph.mpm.constitutive import lame
from physmorph.mpm.function import PersistentAdjoint, RolloutSpec
from physmorph.mpm.gap_split import GapPlanes
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory, compute_rest_volumes
from physmorph.pipeline.config import PipelineConfig

d = np.load(sys.argv[1])
tgt = np.load(sys.argv[2])["tgt"].astype(np.float64)
dx, n = 0.2948594285057975, 44
gmin = -dx * n / 2
cfg = PipelineConfig()
prm = MPMParams(dx=dx, dt=1.0 / 240.0, drag=0.9, smoothing=0.955, grid_min=(gmin,) * 3, nx=n, ny=n, nz=n)
x = torch.tensor(d["x"], device="cuda")
q = (x + torch.tensor(d["T_tgt"] - d["T_self"], device="cuda")).contiguous()
N, T = x.shape[0], 40
m = 40000.0 / N
lam, mu = lame(cfg.young, cfg.poisson)
vol0 = compute_rest_volumes(x, m, prm, "cuda")
gp = GapPlanes(tgt, dx, np.full(3, gmin), (n,) * 3, 2 * (-gmin) / 87)
pm, pn = torch.tensor(gp.m, device="cuda"), torch.tensor(gp.n, device="cuda")
print(f"N {N}, gap nodes {gp.n_gap}", flush=True)
gen = torch.Generator(device="cuda").manual_seed(0)
dc = torch.randn(T, N, 3, 3, device="cuda", generator=gen) * 1e-3
dc[T // 2:] = 0


def spec(gap):
    return RolloutSpec(x0=x, m=m, lam=lam, mu=mu, prm=prm, T=T, device="cuda", vol0=vol0, polar_adjoint=True,
                       track_geom=False, volume_exact="carried", control_steps=T // 2, gap=gap)


def sync_time(f, k=5):
    out = []
    for _ in range(k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        f()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return float(np.median(out))


res = {}
for name, gap in (("single", None), ("split", (pm, pn, q, 0.0))):
    s = spec(gap)
    seq = [wp.from_torch(dc[t].contiguous(), dtype=wp.mat33) for t in range(T)]
    tr = Trajectory(x, m, lam, mu, prm, T, dFc=seq, device="cuda", requires_grad=False, vol0=vol0, persistent=True,
                    polar_adjoint=True, volume_exact="carried", control_steps=T // 2, gap=gap)
    tr.capture()
    t_eval = sync_time(tr.run)
    adj = PersistentAdjoint(s)
    dcl = dc.clone().requires_grad_(True)

    def fb():
        xT, FT, vT, _, V = adj.apply(dcl, None)
        L = xT.square().sum() + V.square().sum()
        torch.autograd.grad(L, dcl)
    t_fb = sync_time(fb)
    print(f"[{name}] eval rollout {t_eval * 1e3:.1f} ms, tape forward + adjoint {t_fb * 1e3:.1f} ms", flush=True)
    res[name] = (t_eval, t_fb)
    # per kernel (uncaptured)
    tr2 = Trajectory(x, m, lam, mu, prm, T, dFc=[wp.from_torch(dc[t].contiguous(), dtype=wp.mat33, requires_grad=True)
                                                 for t in range(T)], device="cuda", requires_grad=True, vol0=vol0,
                     persistent=True, polar_adjoint=True, volume_exact="carried", control_steps=T // 2, gap=gap)
    for rep in range(2):
        tape = wp.Tape()
        if rep == 1:
            wp.timing_begin(cuda_filter=wp.TIMING_KERNEL)
        with tape:
            tr2.rollout()
        tape.backward(grads={tr2.x[T]: wp.ones(N, dtype=wp.vec3, device="cuda")})
        wp.synchronize()
        if rep == 1:
            rs = wp.timing_end()
        tape.zero()
    agg = {}
    for r in rs:
        agg[r.name] = agg.get(r.name, 0.0) + r.elapsed
    tot = sum(agg.values())
    print(f"[{name}] kernels, one taped forward + backward: total {tot:.1f} ms", flush=True)
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:18]:
        print(f"   {v:8.1f} ms  {100 * v / tot:5.1f} %  {k}", flush=True)
    del adj, tr, tr2
    torch.cuda.empty_cache()
print(f"split / single: eval x{res['split'][0] / res['single'][0]:.2f}, tape x{res['split'][1] / res['single'][1]:.2f}")
