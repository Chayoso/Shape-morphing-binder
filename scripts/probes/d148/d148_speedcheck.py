"""d148_speedcheck.py OUT.npz -- D148's exact speed-up checked against the code before it: run in each repository (the
current directory on the path) on the CPU, the window case of tests/test_volume_exact.py with gap planes everywhere
and the sides by the sign of y (mixed in every stencil), hard sides; saves the tape's outputs (x, F, v, every step's v)
and the gradients on the control and u. Compare two files with d148_speedcheck.py A.npz B.npz."""
import dataclasses
import sys

import numpy as np

if len(sys.argv) == 3:
    a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
    for k in a.files:
        x, y = a[k], b[k]
        same = np.array_equal(x, y)
        rel = float(np.abs(x - y).max() / max(np.abs(x).max(), 1e-30))
        print(f"{k}: bit for bit {same}, max relative difference {rel:.2e}")
    sys.exit()
sys.path.insert(0, ".")
sys.path.insert(0, "tests")
import physmorph  # noqa: E402,F401
from test_runtime_exact import _tape_outputs  # noqa: E402
from test_volume_exact import _window_case  # noqa: E402

spec, dc, u = _window_case("cpu")
ng = spec.prm.nx * spec.prm.ny * spec.prm.nz
pm = np.zeros((ng, 3), np.float32)
pn = np.tile(np.asarray([0, 1.0, 0], np.float32), (ng, 1))
out = _tape_outputs(dataclasses.replace(spec, track_geom=False, gap=(pm, pn, np.asarray(spec.x0, np.float32), 0.0)),
                    dc, u)
np.savez(sys.argv[1], **{k: t.cpu().numpy() for k, t in zip(("x", "F", "v", "V", "g_dc", "g_u"), out)})
print("saved", sys.argv[1])
