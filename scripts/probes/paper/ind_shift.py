"""ind_shift.py IND_NPZ RUN_FRAMES_NPZ OUT_NPZ -- D145 B9: the independent sample moved onto the floor as the run's own
target was (prepare lifts the target along y so that it stands on the floor; x and z unchanged). The lift is read as the
difference of the y means of the run's target sample and the independent sample (two samples of the same mesh in the
same frame: their means agree to about 1e-4 wu at 300k), and written as the shifted sample (frames = [tgt], raws = [0])."""
import sys

import numpy as np

ind = np.asarray(np.load(sys.argv[1])["tgt"], np.float64)
run = np.asarray(np.load(sys.argv[2], allow_pickle=True)["tgt"], np.float64)
lift = float(run[:, 1].mean() - ind[:, 1].mean())
x = ind.copy()
x[:, 1] += lift
x = x.astype(np.float32)
np.savez(sys.argv[3], frames=x[None], raws=np.array([0]), tgt=x)
print(f"lift {lift:+.5f} wu; x/z mean differences {run[:, 0].mean() - ind[:, 0].mean():+.5f} {run[:, 2].mean() - ind[:, 2].mean():+.5f}")
