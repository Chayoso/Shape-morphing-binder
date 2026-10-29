"""bench_start.py [pipeline_run args] -- D174: the start of a run timed by part (the prepare stage, the thin set's
parts, the target build's parts). Run from the repo: PYTHONPATH=$GPU_DEPS:$PWD $PY bench_start.py --tgt ... --n 300000."""
import os
import sys
import time

import physmorph  # noqa: F401
import torch

sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import pipeline_run as P  # noqa: E402
from physmorph import gpu, thin  # noqa: E402


def clk():
    torch.cuda.synchronize()
    return time.perf_counter()


T = {}
orig_edt, orig_knn, orig_fill = gpu.edt, gpu.KNN.query, gpu.fill_holes


def timed_fn(name, fn):
    def w(*a, **k):
        t0 = clk()
        out = fn(*a, **k)
        T[name] = T.get(name, [0.0, 0])
        T[name][0] += clk() - t0
        T[name][1] += 1
        return out
    return w


args = P.parse_args()
t0 = clk()
cfg0, prep = P.prepared(args, args.seed)
print(f"[bench] prepare {clk() - t0:.2f} s", flush=True)
cfg = P.run_config(args, cfg0, prep)
tgt, prm = prep.tgt, prep.prm
gpu.edt = timed_fn("edt", orig_edt)
gpu.fill_holes = timed_fn("fill_holes", orig_fill)
gpu.KNN.query = timed_fn("KNN.query", orig_knn)
from physmorph.render import knn_gpu  # noqa: E402
knn_gpu.knn_self_torch = timed_fn("knn_self_torch", knn_gpu.knn_self_torch)
thin.local_thickness = timed_fn("local_thickness", thin.local_thickness)
thin.outer_mask = timed_fn("outer_mask", thin.outer_mask)
t0 = clk()
ts = thin.thin_set(tgt, prep.cell_shape or prm.dx, cfg.mass_ref_n)
print(f"[bench] thin_set {clk() - t0:.2f} s: " + ", ".join(f"{k} {v[0]:.2f} s x{v[1]}" for k, v in T.items()), flush=True)
T.clear()
from physmorph.pipeline import target as TG  # noqa: E402
for name in [n for n in dir(TG) if not n.startswith("_") and callable(getattr(TG, n))]:
    f = getattr(TG, name)
    if getattr(f, "__module__", "") == TG.__name__ and name not in ("build_target", "TargetPack", "dataclass", "field"):
        setattr(TG, name, timed_fn("target." + name, f))
from physmorph.render import exterior as EX  # noqa: E402
for name in ("make_field", "tracked", "ZhuBridson"):
    if hasattr(EX, name):
        setattr(EX, name, timed_fn("exterior." + name, getattr(EX, name)))
t0 = clk()
pack = TG.build_target(tgt, prm, cfg, draws=prep.tgt_draws, mesh=prep.tgt_mesh, appearance=None, render_x=prep.render_x)
print(f"[bench] build_target {clk() - t0:.2f} s: " + ", ".join(f"{k} {v[0]:.2f} s x{v[1]}" for k, v in sorted(T.items(), key=lambda kv: -kv[1][0])), flush=True)
T.clear()
if prep.tgt_surface is not None:
    t0 = clk()
    pack.relief = TG.target_relief(pack.pts, prep.tgt_surface, cfg)
    print(f"[bench] target_relief {clk() - t0:.2f} s", flush=True)
t0 = clk()
src = gpu.tensor(prep.src)
TG.calibrate_units(pack, src, cfg)
print(f"[bench] calibrate_units {clk() - t0:.2f} s", flush=True)
