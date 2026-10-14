"""d148_memprobe.py [pipeline_run args] -- D148 diagnosis: the 300k split runs ran out of memory (+7-14 GB a window).
Runs scripts/pipeline_run.py in-process and prints at every adjoint build (once a window): the device memory in use,
torch's allocated and reserved, and the live Trajectory / PersistentAdjoint objects (gc), so a retained window shows."""
import gc
import sys

import physmorph  # noqa: F401  (before torch: CuPy binds its CUDA 12 compiler first)
import torch

argv = sys.argv[1:]
sys.argv = ["pipeline_run.py"] + argv
import physmorph.mpm.function as FN  # noqa: E402
from physmorph.mpm.traj import Trajectory  # noqa: E402

orig = FN.PersistentAdjoint.__init__


def init(self, spec):
    free, tot = torch.cuda.mem_get_info()
    nt = sum(1 for o in gc.get_objects() if isinstance(o, Trajectory))
    na = sum(1 for o in gc.get_objects() if isinstance(o, FN.PersistentAdjoint))
    gc.collect()
    nt2 = sum(1 for o in gc.get_objects() if isinstance(o, Trajectory))
    na2 = sum(1 for o in gc.get_objects() if isinstance(o, FN.PersistentAdjoint))
    free2, _ = torch.cuda.mem_get_info()
    print(f"[mem] adjoint build: used {(tot - free) / 1e9:.2f} GB (after gc {(tot - free2) / 1e9:.2f}), torch allocated "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} reserved {torch.cuda.memory_reserved() / 1e9:.2f}; live "
          f"trajectories {nt} adjoints {na} (after gc {nt2} / {na2})", flush=True)
    orig(self, spec)


FN.PersistentAdjoint.__init__ = init
sys.path.insert(0, ".")
from scripts.pipeline_run import main  # noqa: E402
main()
