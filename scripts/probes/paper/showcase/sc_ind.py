"""sc_ind.py SRC_OBJ TGT_OBJ N_REF OUT_NPZ [RUN_FRAMES_NPZ|-] [SEED] [N_FRAME] -- SHOWCASE: independent_sample.py (D90)
with the run's source as an argument. A volume sample of the target mesh at N_REF particles, in the exact frame of the
pipeline's target of a run SRC -> TGT at N_FRAME particles (prepare: the source sampled at seed 97, the target at
seed 98 and scaled to the source's volume; `frame` from load_normalized; the frame does not depend on the density
match, which rescales the source only). With SRC = assets/isosphere.obj this is independent_sample.py's sample, call
for call. Written as tgt (and frames = [tgt], raws = [0]). With RUN_FRAMES_NPZ the pipeline's target is checked
against that run's own target sample (they must be the same). SEED 99 = the evaluation's independent sample (300k),
another seed for the renderer's dense sample (1.5M)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from physmorph import gpu                                      # noqa: E402,F401  (before torch)
import numpy as np                                             # noqa: E402
from physmorph.sampling import load_normalized                 # noqa: E402
from physmorph.sampling.mesh import load_mesh, sample_volume_stratified  # noqa: E402
from physmorph.sampling.orientation import orient_name, rotation  # noqa: E402

src_path, path, n_ref, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 98
n_frame = int(sys.argv[7]) if len(sys.argv) > 7 else min(n_ref, 300000)
src, v_src = load_normalized(src_path, n_frame, 97, return_volume=True, sample="stratified")
frame = {}
tgt = load_normalized(path, n_frame, 98, match_volume=v_src, sample="stratified", frame=frame)
if len(sys.argv) > 5 and sys.argv[5] != "-":
    run_tgt = np.asarray(np.load(sys.argv[5], allow_pickle=True)["tgt"], np.float32)
    print(f"the pipeline's target against the run's: same shape {run_tgt.shape == tgt.shape}, largest difference "
          f"{float(np.abs(run_tgt - tgt).max()):.2e} wu", flush=True)
mesh = load_mesh(path)
o = orient_name(path)
if o != "id":
    mesh.vertices = np.asarray(mesh.vertices, np.float64) @ rotation(o).T
x = sample_volume_stratified(mesh, n_ref, seed=seed).astype(np.float64)
x = ((x - frame["offset"]) * frame["scale"]).astype(np.float32)
print(f"{src_path} -> {path}: {len(x)} particles; centroid {np.round(x.mean(0), 4)} against the {n_frame} target's "
      f"{np.round(tgt.mean(0), 4)}; bbox {np.round(x.min(0), 3)} {np.round(x.max(0), 3)} against {np.round(tgt.min(0), 3)} "
      f"{np.round(tgt.max(0), 3)}; source volume {v_src:.3f} wu^3", flush=True)
np.savez(out, frames=x[None], raws=np.array([0]), tgt=x)
print("wrote", out)
