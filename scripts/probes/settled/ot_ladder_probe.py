"""ot_ladder_probe.py ARCHIVE.npz [WINDOW ...] — D29: what the Sinkhorn solve's sweeps cost and what they buy.

Three solvers at the same tolerance, all from zero duals, on the states of an archived run (a window's committed end
state and a second candidate two steps before it), each against a solve converged to 1e-5:
  before         the solver until D29: the parallel averaged sweep f, g <- (f + T(g)) / 2, (g + T(f)) / 2 for the
                 cross problem as for the self problem, both down the whole epsilon ladder;
  self at blur   the same cross solve, the self problem started at the blur (its plan is local);
  production     the cross problem with alternating sweeps (f <- T(g), then g <- T(f)), the self problem at the blur.
Printed: the error of the value, of the difference between the two candidates (what a line search compares) and of
the gradient; four-sweep blocks and seconds per value call (two solves).

Recorded in docs/experiments.md (D25, D28, D29): a fixed one or two blocks at each ladder level instead of
convergence costs more blocks in total and usually fails the sweep budget; the cross problem started at the blur does
not converge; a solve started from the potentials of the window's start state is 2-15 times cheaper late in a run but
reports less change than there is (a candidate difference off by 10-20 %), and was dropped."""
import physmorph  # noqa: F401  (before torch: CuPy's CUDA 12 NVRTC)
import json
import sys
import time

import numpy as np
import torch

from physmorph import gpu, prof
from physmorph.losses.grid_ot import GridSinkhornLoss
from physmorph.losses.volumetric import rasterize_mass
from physmorph.mpm.state import MPMParams
from physmorph.pipeline import PipelineConfig
from physmorph.pipeline.target import build_target


class Before(GridSinkhornLoss):
    """The solver until D29 (averaged = True: the cross problem with the parallel averaged sweep; self_ladder = True:
    the self problem down the ladder), or either half of it."""
    averaged = self_ladder = True
    _avg = None

    def _start_level(self, a, same):
        return super()._start_level(a, same and not self.self_ladder)

    def _solve_cuda_blocks(self, a, b):
        if a is b or not self.averaged or False not in self._solve_graphs:
            return super()._solve_cuda_blocks(a, b)
        if self._avg is None:
            la, lb = a.log().clone(), b.log().clone()
            f, g = torch.zeros_like(a), torch.zeros_like(b)
            temp, error = a.new_tensor(self.eps), a.new_zeros(())
            abuf, bbuf = a.clone(), b.clone()

            def block():
                for i in range(4):
                    fn = self.transform(g, lb, temp)
                    gn = self.transform(f, la, temp)
                    if i == 3:
                        ea = (torch.exp(la + (f - fn) / temp) - abuf).abs().sum()
                        eb = (torch.exp(lb + (g - gn) / temp) - bbuf).abs().sum()
                        error.copy_(torch.maximum(ea, eb))
                    f.copy_(.5 * (f + fn))
                    g.copy_(.5 * (g + gn))

            stream = torch.cuda.Stream(device=a.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                block()
                block()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                block()
            self._avg = (graph, la, lb, f, g, temp, error, abuf, bbuf)
        graph, la, lb, f, g, temp, error, abuf, bbuf = self._avg
        la.copy_(a.log()); lb.copy_(b.log()); abuf.copy_(a); bbuf.copy_(b)
        f.zero_(); g.zero_()
        level = self._start_level(a, False)
        for iteration in range(0, self.iters, 4):
            temperature = self.eps * 2. ** level
            temp.fill_(temperature)
            graph.replay()
            residual = float(error)
            if residual < self.tol:
                if level == 0:
                    break
                level -= 1
        prof.count("ot_blocks", iteration // 4 + 1)
        if level != 0 or not residual <= self.tol:
            raise ValueError(f'grid transport did not converge: marginal error {residual:g}')
        return f.clone(), g.clone()


def main(path, wins):
    js = json.load(open(path.replace("_render_full_dt_iso_nn.npz", ".json")))
    arm = js["arms"]["render_full_dt_iso_nn"]; mpm = js["provenance"]["mpm"]; cfg0 = arm["config"]
    prm = MPMParams(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in mpm.items()
                       if k in MPMParams.__dataclass_fields__})
    cfg = PipelineConfig(loss_res=int(cfg0["loss_res"]), unit_ref_res=int(cfg0.get("unit_ref_res", 64)),
                         ot_iters=int(cfg0["ot_iters"]), ot_tol=float(cfg0["ot_tol"]))
    z = np.load(path, allow_pickle=True)
    frames, T2 = z["frames"], 2 * int(cfg0["T"])
    n_win = (int(z["deliver_n"]) - 1) // T2
    pack = build_target(np.asarray(z["tgt"], np.float32), prm, cfg)
    prof.STATE["on"] = True
    X = lambda raw: gpu.tensor(np.asarray(frames[raw], np.float32))  # noqa: E731
    R = lambda x: rasterize_mass(x, pack.m, pack.lgmin, pack.ldx, pack.ldims)  # noqa: E731

    def solver(cls, tol, iters):
        ot = cls(pack.grid, pack.lgmin, pack.ldx, pack.ldims, eps=pack.ldx ** 2, iters=iters,
                 tol=tol, mass_total=float(pack.m.sum()), cuda_blocks=True)
        ot(R(X(0)))                                         # builds the graphs
        return ot

    def value(ot, x):
        prof.take()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            v = float(ot(R(x)))
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        return v, prof.take()["n"].get("ot_blocks", 0), dt

    def grad(ot, x):
        xg = x.clone().requires_grad_(True)
        return torch.autograd.grad(ot(R(xg)), xg)[0]

    tol = min(cfg.ot_tol, 1e-3)
    ref, old, new = solver(GridSinkhornLoss, 1e-5, 60000), solver(Before, tol, cfg.ot_iters), solver(GridSinkhornLoss, tol, cfg.ot_iters)
    print(f"{path.split('/')[-1]}: N {len(pack.m)}, loss grid {tuple(pack.ldims)}, tolerance {tol:g}, sweep budget {cfg.ot_iters}; {n_win} windows")
    print(" window | variant | value error (relative to the 1e-5 value) | candidate difference: true (relative to the value), this variant's error in it | "
          "gradient error |g - g*| / |g*| | blocks per value call | seconds per value call")
    # the gradient at the target itself against the gradient at the run's end state (the equilibrium property)
    xt, xend = gpu.tensor(np.asarray(z["tgt"], np.float32)), X(T2 * n_win)
    for name, ot in (("before", old), ("production", new)):
        gt_, ge_ = grad(ot, xt), grad(ot, xend)
        with torch.no_grad():
            vt_ = float(ot(R(xt)))
        print(f"   at the target sample, {name}: value {vt_:+.3e}; gradient rms {float(gt_.pow(2).sum(1).mean().sqrt()):.3e} against {float(ge_.pow(2).sum(1).mean().sqrt()):.3e} at the run's end state")
    for k in wins or (0, 4, 8, 12, 20, 28, 36):
        k = min(k, n_win - 1)
        xe, xn = X(T2 * (k + 1)), X(T2 * (k + 1) - 2)
        try:
            vt, bt, tt = value(ref, xe)
            vnt = value(ref, xn)[0]
            gt = grad(ref, xe)
        except Exception as e:
            print(f"  {k:4d} | the 1e-5 solve failed: {str(e)[:70]}")
            continue
        print(f"  {k:4d} | converged to 1e-5 | value {vt:.6e} | {(vt - vnt) / vt:+.3e} | | {bt} | {tt:.2f}")
        for name, ot, avg, lad in (("before      ", old, True, True), ("self at blur", old, True, False), ("production  ", new, None, None)):
            if avg is not None:
                ot.averaged, ot.self_ladder = avg, lad
            try:
                v, b, t = value(ot, xe)
                vn = value(ot, xn)[0]
                g = grad(ot, xe)
                print(f"       | {name} | {(v - vt) / vt:+.2e} | {(vt - vnt) / vt:+.3e}, {((v - vn) - (vt - vnt)) / vt:+.2e} | {float((g - gt).norm() / gt.norm()):.2e} | {b:4d} | {t:.3f}")
            except Exception as e:                          # a variant that does not converge in the budget
                print(f"       | {name} | failed: {str(e)[:80]}")


if __name__ == "__main__":
    main(sys.argv[1], [int(v) for v in sys.argv[2:]])
