"""d163_ceiling_npz.py OUT_NPZ SMOOTH RELIEF N SEED -- a mesh sampled at N in the run's frame (seed + 1, sample_in_frame:
the reader's ceiling sample when MESH is the relief mesh; with the smooth mesh, the target sample's twin) written as a
one-frame kept-frames file (frames, raws, tgt), so that the renderers can draw what an N-particle body carrying the whole
relief looks like on the display surface."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "d161"))
from d161_lib import target_frame  # noqa: E402
from physmorph.sampling.mesh import fill_volume_in_frame, sample_in_frame  # noqa: E402

out, smooth, mesh, N, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
frame, x_t = target_frame("assets/isosphere.obj", smooth, N, seed)
n_r = int(round(N * fill_volume_in_frame(mesh, frame, N) / fill_volume_in_frame(smooth, frame, N)))
X = np.asarray(sample_in_frame(mesh, frame, n_r, seed + 1)[0], np.float32)
np.savez(out, frames=X[None], raws=np.array([0]), tgt=x_t.astype(np.float32))
print(len(X))
