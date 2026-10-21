"""endarch.py RUNS_DIR TAG OUT_NPZ -- B9b (CPU): a standing scene run's last kept frame in its objective's frame (the
run's fixed_frames: B9's world, the target lifted onto the floor) as a one-frame archive (frames, raws, tgt, deliver_n),
the input of render_terms_probe.py (the yardstick) against the independent sample lifted as the run's target."""
import json
import sys
from pathlib import Path

import numpy as np

runs, tag, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
j = json.loads((runs / f"{tag}.json").read_text())
z = np.load(runs / f"{tag}.npz")
ff = j["scene"]["fixed_frames"][0]
c, R = np.asarray(ff["c"], np.float64), np.asarray(ff["R"], np.float64)
x = np.asarray(z["frames"][-1], np.float64)
np.savez(out, frames=((x - c) @ R).astype(np.float32)[None], raws=z["raws"][-1:], tgt=np.asarray(z["tgt0"], np.float32),
         deliver_n=np.int64(z["raws"][-1] + 1))
print(f"{tag}: end frame {int(z['raws'][-1])} in the objective's frame -> {out}")
