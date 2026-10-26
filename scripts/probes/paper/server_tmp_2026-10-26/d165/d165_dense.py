"""d165_dense.py SMOOTH_OBJ RELIEF_OBJ N_REF OUT_NPZ [SEED] -- D165: the relief mesh (D163's render target) sampled at
N_REF particles in the frame of the runs' 3D target (the smooth plaque at 300k, seed 98, scaled to the source's volume:
scripts/probes/settled/independent_sample.py's frame), written as tgt (and frames = [tgt], raws = [0]): the dense
samples whose pictures scripts/render_deliverable.py's display-only layer (+fit=1, D115) is fitted to."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from physmorph import gpu  # noqa: E402,F401  (before torch)
import numpy as np  # noqa: E402
from physmorph.sampling import load_normalized  # noqa: E402
from physmorph.sampling.mesh import load_mesh, sample_volume_stratified  # noqa: E402

smooth, relief, n_ref, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 98
src, v_src = load_normalized("assets/isosphere.obj", 300000, 97, return_volume=True, sample="stratified")
frame = {}
tgt = load_normalized(smooth, 300000, 98, match_volume=v_src, sample="stratified", frame=frame)
mesh = load_mesh(relief)
x = sample_volume_stratified(mesh, n_ref, seed=seed).astype(np.float64)
x = ((x - frame["offset"]) * frame["scale"]).astype(np.float32)
print(f"{relief}: {len(x)} particles; bbox {np.round(x.min(0), 3)} {np.round(x.max(0), 3)} against the smooth target's "
      f"{np.round(tgt.min(0), 3)} {np.round(tgt.max(0), 3)}", flush=True)
np.savez(out, frames=x[None], raws=np.array([0]), tgt=x)
print("wrote", out)
