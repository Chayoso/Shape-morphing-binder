"""d155c_zdump.py MESH [N] -- D155c forensic (card, seconds): the moving target's own points, to read offline what volume
and what local density each target on the way asks. The run's prepare and final target as pipeline_run builds them
(the A3C flags), the moving target's D_Y (MorphTarget), then saves Y (sample 0), D_Y, the source sample and the body's
spacing sp(Y) to output/gpu/d155/gate/zdump_<mesh>40k.npz. Measurement only; no pack is built."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from physmorph import gpu  # noqa: E402,F401

import physmorph.pipeline.target as T  # noqa: E402
from physmorph.pipeline.run.morph import MorphTarget  # noqa: E402
from scripts.pipeline_run import parse_args, prepared, run_config  # noqa: E402

mesh = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
args = parse_args(["--tgt", f"assets/{mesh}.obj", "--n", str(n), "--seed", "97", "--morph_ramp", "elastic",
                   "--morph_ref_n", "300000", "--morph_u_gate", "final", "--morph_density_match"])
cfg0, prep = prepared(args, 97)
cfg = run_config(args, cfg0, prep)
final = T.build_target(prep.tgt, prep.prm, cfg, draws=prep.tgt_draws)
T.calibrate_units(final, gpu.tensor(prep.src), cfg)
mt = MorphTarget(final, gpu.tensor(prep.src), prep.tgt_draws, prep.prm, cfg, log=print)
out = "/data/relcfd/chayo/physmorph_v2/output/gpu/d155/gate/zdump_%s%dk.npz" % (mesh, n // 1000)
np.savez_compressed(out, Y=mt.samples[0].cpu().numpy(), D=mt.D[0].cpu().numpy(), src=np.asarray(prep.src),
                    sp_y=mt.sp_y, t_m=mt.t_m, gdx=float(final.gate[1] if final.gate is not None else final.ldx),
                    dx=float(prep.prm.dx) if hasattr(prep.prm, "dx") else -1.0)
print("saved", out, "sp_y", mt.sp_y, "t_m", mt.t_m)
