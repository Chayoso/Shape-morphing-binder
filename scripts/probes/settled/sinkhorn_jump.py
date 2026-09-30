"""sinkhorn_jump.py ARCHIVE.npz — is the grid transport value a continuous function of the particle positions?

The grid Sinkhorn stops each blur level at the first 4-sweep block whose marginal error is below tol, so the number
of sweeps depends on the input. Near a block boundary a tiny position change moves the stop by one block and the value
jumps by the truncation error. At the archive's end state, positions x + delta * d (d a fixed random unit field,
delta from 1e-10 to 1e-4) are evaluated with (1) the solver as it is and (2) the same solver replaying the sweep
schedule of the unperturbed solve. Printed per delta: the value change of each, and the blocks used per level by (1).
A jump is a change that does not shrink with delta.
"""
import physmorph  # noqa: F401  (before torch: CuPy's CUDA 12 NVRTC)
import json
import sys

import numpy as np
import torch

from physmorph import gpu
from physmorph.losses.grid_ot import GridSinkhornLoss
from physmorph.losses.volumetric import rasterize_mass
from physmorph.mpm.state import MPMParams
from physmorph.pipeline import PipelineConfig
from physmorph.pipeline.target import build_target


class Recording(GridSinkhornLoss):
    """The solver with its sweep schedule recorded, or replayed (no early stop)."""
    schedule = None       # replay: {same: [blocks at each level]}
    last = None

    def _solve_cuda_blocks(self, a, b):
        same = a is b
        if same not in self._solve_graphs:
            super()._solve_cuda_blocks(a, b)                   # builds the graph
        graph, la, lb, f, g, temp, error, abuf, bbuf = self._solve_graphs[same]
        la.copy_(a.log()); lb.copy_(b.log()); abuf.copy_(a); bbuf.copy_(b)
        f.zero_(); g.zero_()
        level = max(0, int(torch.ceil(torch.log2(a.new_tensor(max(self.diameter2, self.eps) / self.eps)))))
        plan = None if self.schedule is None else list(self.schedule[same])
        used, count = [], 0
        for _ in range(0, self.iters, 4):
            temp.fill_(self.eps * 2. ** level)
            graph.replay()
            count += 1
            done = (count >= plan[0]) if plan is not None else float(error) < self.tol
            if done:
                used.append(count); count = 0
                if plan is not None:
                    plan.pop(0)
                if level == 0:
                    break
                level -= 1
        self.last = self.last or {}
        self.last[same] = used
        return f.clone(), g.clone()


def main(path):
    js = json.load(open(path.replace("_render_full_dt_iso_nn.npz", ".json")))
    arm = js["arms"]["render_full_dt_iso_nn"]; mpm = js["provenance"]["mpm"]; cfg0 = arm["config"]
    prm = MPMParams(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in mpm.items()
                       if k in MPMParams.__dataclass_fields__})
    cfg = PipelineConfig(loss_res=int(cfg0["loss_res"]), unit_ref_res=int(cfg0.get("unit_ref_res", 64)))
    z = np.load(path, allow_pickle=True)
    frames = z["frames"]
    x = gpu.tensor(np.asarray(frames[int(z["deliver_n"]) - 1 if "deliver_n" in z.files else -1], np.float32))
    pack = build_target(np.asarray(z["tgt"], np.float32), prm, cfg)
    kw = dict(eps=pack.ldx ** 2, iters=cfg.ot_iters, tol=min(cfg.ot_tol, 1e-3), mass_total=float(pack.m.sum()),
              cuda_blocks=True)
    ot = Recording(pack.grid, pack.lgmin, pack.ldx, pack.ldims, **kw)
    d = torch.randn(x.shape, device=x.device, generator=torch.Generator(device=x.device).manual_seed(5))
    d /= d.norm(dim=1, keepdim=True)

    def value(q, schedule=None):
        ot.schedule, ot.last = schedule, None
        with torch.no_grad():
            v = float(ot(rasterize_mass(q, pack.m, pack.lgmin, pack.ldx, pack.ldims)))
        return v, ot.last

    v0, sched0 = value(x)
    print(f"{path.split('/')[-1]}: loss grid {pack.ldims[0]}^3, value {v0:.6e}; blocks per level cross {sched0[False]} "
          f"self {sched0[True]}")
    print("   delta   adaptive dV     frozen dV    adaptive blocks at the last two levels (cross | self)")
    jumps = 0
    for delta in np.logspace(-10, -4, 19):
        va, sa = value(x + float(delta) * d)
        vf, _ = value(x + float(delta) * d, schedule=sched0)
        changed = sa != sched0
        jumps += changed
        print(f"{delta:8.1e} {va - v0:+12.3e} {vf - v0:+12.3e}    {sa[False][-2:]} | {sa[True][-2:]}"
              f"{'   <- schedule changed' if changed else ''}")
    print(f"schedule changed at {jumps} of 19 deltas")


if __name__ == "__main__":
    main(sys.argv[1])
