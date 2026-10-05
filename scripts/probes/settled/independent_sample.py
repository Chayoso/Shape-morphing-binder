"""independent_sample.py MESH_OBJ N_REF OUT_NPZ [RUN_FRAMES_NPZ] [SEED] [N_FRAME] — D90: a volume sample of the target mesh at N_REF particles in
the exact frame of the pipeline's target at N_FRAME particles (default N_REF, at most 300k; prepare: the target sampled at seed 98 and scaled to the
source's volume; `frame` from load_normalized), written as tgt (and frames = [tgt], raws = [0]). With RUN_FRAMES_NPZ
the pipeline's target is checked against that run's own target (they must be the same sample)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402,F401  (before torch)
import numpy as np                                             # noqa: E402
from physmorph.sampling import load_normalized                 # noqa: E402
from physmorph.sampling.mesh import load_mesh, sample_volume_stratified  # noqa: E402
from physmorph.sampling.orientation import orient_name, rotation  # noqa: E402

path, n_ref, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 98            # another seed: an independent sample of the mesh
n_frame = int(sys.argv[6]) if len(sys.argv) > 6 else min(n_ref, 300000)    # the N of the pipeline run whose frame is used
src, v_src = load_normalized("assets/isosphere.obj", n_frame, 97, return_volume=True, sample="stratified")
frame = {}
tgt = load_normalized(path, n_frame, 98, match_volume=v_src, sample="stratified", frame=frame)
if len(sys.argv) > 4 and sys.argv[4] != "-":               # "-": no run to check against
    run_tgt = np.asarray(np.load(sys.argv[4], allow_pickle=True)["tgt"], np.float32)
    print(f"the pipeline's target against the run's: same shape {run_tgt.shape == tgt.shape}, largest difference {float(np.abs(run_tgt - tgt).max()):.2e} wu", flush=True)
mesh = load_mesh(path)
o = orient_name(path)
if o != "id":
    mesh.vertices = np.asarray(mesh.vertices, np.float64) @ rotation(o).T
x = sample_volume_stratified(mesh, n_ref, seed=seed).astype(np.float64)
x = ((x - frame["offset"]) * frame["scale"]).astype(np.float32)
print(f"{path}: {len(x)} particles; centroid {np.round(x.mean(0), 4)} against the 300k target's {np.round(tgt.mean(0), 4)}; "
      f"bbox {np.round(x.min(0), 3)} {np.round(x.max(0), 3)} against {np.round(tgt.min(0), 3)} {np.round(tgt.max(0), 3)}", flush=True)
np.savez(out, frames=x[None], raws=np.array([0]), tgt=x)
print("wrote", out)
