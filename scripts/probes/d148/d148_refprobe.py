"""d148_refprobe.py [pipeline_run args] -- D148 diagnosis: which object keeps a finished window's Trajectory alive
(the settled-base path retains one eval Trajectory a window, 7.5 GB at 300k). At every adjoint build after the first,
the referrers of the oldest live Trajectory, two levels deep (types, attribute names)."""
import gc
import sys
import types

import physmorph  # noqa: F401  (before torch)
import torch  # noqa: F401

sys.argv = ["pipeline_run.py"] + sys.argv[1:]
import physmorph.mpm.function as FN  # noqa: E402
from physmorph.mpm.traj import Trajectory  # noqa: E402

orig = FN.PersistentAdjoint.__init__
seen = []


def describe(o):
    if isinstance(o, dict):
        return f"dict(keys {list(o.keys())[:8]})"
    if isinstance(o, (list, tuple)):
        return f"{type(o).__name__}(len {len(o)})"
    if isinstance(o, types.FrameType):
        return f"frame {o.f_code.co_name} {o.f_code.co_filename.split('/')[-1]}:{o.f_lineno}"
    if isinstance(o, types.FunctionType):
        return f"function {o.__qualname__}"
    if isinstance(o, types.CellType):
        return "cell"
    return type(o).__module__ + "." + type(o).__qualname__


def init(self, spec):
    gc.collect()
    trs = [o for o in gc.get_objects() if isinstance(o, Trajectory)]
    print(f"[ref] live trajectories {len(trs)}", flush=True)
    if len(trs) >= 2:
        old = trs[0]
        me = sys._getframe()
        for r in gc.get_referrers(old):
            if r is trs or r is me:
                continue
            print(f"[ref]  L1 {describe(r)}", flush=True)
            for r2 in gc.get_referrers(r):
                if r2 is trs or isinstance(r2, types.FrameType) and r2 is me:
                    continue
                print(f"[ref]    L2 {describe(r2)}", flush=True)
                if isinstance(r2, dict):
                    for r3 in gc.get_referrers(r2):
                        if r3 is trs:
                            continue
                        print(f"[ref]      L3 {describe(r3)}", flush=True)
    orig(self, spec)


FN.PersistentAdjoint.__init__ = init
sys.path.insert(0, ".")
from scripts.pipeline_run import main  # noqa: E402
main()
