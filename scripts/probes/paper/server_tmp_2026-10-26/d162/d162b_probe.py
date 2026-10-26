"""d162b_probe.py -- D162b: the shape reading of the floor sample under the voxel rule, against the uniform true-mesh
level of D162's X step (band 2.06e-3, coin 3.39e-3); the run's TargetPack rebuilt as d162_arrange.py does. GPU."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "scripts"))
import physmorph  # noqa: F401
import numpy as np
import pipeline_run as PR
from physmorph.pipeline.appearance import make_appearance
from physmorph.pipeline.shape_read import ShapeRead
from physmorph.pipeline.target import build_target, render_floor
from physmorph.pipeline.window.setup import domain_bounds
A = "/data/relcfd/chayo/physmorph_v2/output/gpu/d161/assets"
for name, flags in (("coin", ["--tgt", f"{A}/d161coin.obj", "--render_tgt", f"{A}/d161L40coin.obj"]),
                    ("band", ["--tgt", "assets/bunny.obj", "--appearance", "band"])):
    args = PR.parse_args(flags + ["--n", "40000", "--seed", "97", "--render_weight_ref", "shape", "--out", "/tmp/x"])
    cfg0, prep = PR.prepared(args, 97)
    cfg = PR.run_config(args, cfg0, prep)
    app = make_appearance(args.appearance, prep.src, prep.tgt, prep.tgt_draws) if args.appearance else None
    tgt = build_target(prep.tgt, prep.prm, cfg, draws=prep.tgt_draws, mesh=prep.tgt_mesh, appearance=app, render_x=prep.render_x)
    fx = prep.render_x if prep.render_x is not None else prep.tgt
    alb = None if app is None else app.tgt
    t0 = time.time()
    sr = ShapeRead(tgt, cfg, fx, prep.tgt_mesh, prep.density["target_fill"], len(prep.tgt), domain_bounds(prep.prm), alb)
    t1 = time.time()
    b, se, n = sr.read(fx, alb)
    t2 = time.time()
    a = render_floor(tgt, cfg, fx, alb)
    print(f"PROBE {name}: a(floor sample) {a:.4e}; b_floor {sr.b_floor:.4e} +- {sr.b_floor_se:.1e} ({sr.n_floor} points; "
          f"N {len(fx)}); a second read {b:.4e} ({n} points); init {t1 - t0:.1f} s, one read {t2 - t1:.1f} s", flush=True)
