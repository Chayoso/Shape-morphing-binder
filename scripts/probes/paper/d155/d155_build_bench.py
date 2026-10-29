"""d155_build_bench.py MESH [N] -- D155b: where a pack on the way spends its build (card). The run's prepare and final
target as pipeline_run builds them (defaults, render on the exterior), the moving target's D_Y, then build_target of Z_s
at s = 0.3 / 0.6 three times each, the parts timed (synchronised): the particle pictures (silhouettes, shading), the
exterior (field + lattice discs + pictures, per sample), the loss and gate grids, the cleanup's distance transform, the
surface proximity, the KD-tree, the rest. Measurement only."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from physmorph import gpu  # noqa: E402,F401
import torch  # noqa: E402

import physmorph.pipeline.target as T  # noqa: E402
from physmorph.pipeline.run.morph import MorphTarget  # noqa: E402
from scripts.pipeline_run import parse_args, prepared, run_config  # noqa: E402

mesh = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
args = parse_args(["--tgt", f"assets/{mesh}.obj", "--n", str(n), "--seed", "97", "--morph_ramp", "elastic",
                   "--morph_ref_n", "300000", "--morph_u_gate", "final"])
cfg0, prep = prepared(args, 97)
cfg = run_config(args, cfg0, prep)
acc = {}


def timed(name, fn):
    def w(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize()
        acc[name] = acc.get(name, 0.0) + time.perf_counter() - t0
        return r
    return w


for name in ("target_silhouettes", "shade_targets", "make_field", "exterior_targets", "target_mass_grid",
             "target_dt_grid", "SurfaceProximity", "density_units"):
    setattr(T, name, timed(name, getattr(T, name)))
T.Lattice.discs = timed("lattice_discs", T.Lattice.discs)
T.gpu.KNN = timed("KNN", T.gpu.KNN)
final = T.build_target(prep.tgt, prep.prm, cfg, draws=prep.tgt_draws)
T.calibrate_units(final, gpu.tensor(prep.src), cfg)
mt = MorphTarget(final, gpu.tensor(prep.src), prep.tgt_draws, prep.prm, cfg, log=print)
for skip in (True, False):
    import physmorph.pipeline.run.morph as M
    M.SKIP_UNUSED_PICTURES = skip
    for s_t in (0.3 * mt.t_m, 0.6 * mt.t_m):
        for rep in range(3):
            acc.clear()
            mt.s = None
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mt.at(s_t)
            total = time.perf_counter() - t0
            parts = {k: round(v, 3) for k, v in sorted(acc.items(), key=lambda kv: -kv[1])}
            print(f"skip={skip} t={s_t:.2f} rep {rep}: total {total:.3f} s; parts {parts}; rest "
                  f"{total - sum(acc.values()):.3f}", flush=True)
