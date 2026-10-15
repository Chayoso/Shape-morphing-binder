"""eval_full.py [the repo's eval.py arguments] -- D145 B14: the authors' own eval.py (run from the repo root, ISD or
4Deform), with one change: their marching-cubes extractor keeps EVERY component of the zero level set
(connected=False) instead of only the largest-area one, so the frames are read like B5's (all components kept).
Nothing else in their code is touched; the largest component is recovered from the full mesh in the conversion."""
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())
from models import plot_manager as pm  # noqa: E402

_orig = pm.PlotManager.extract_mesh


def _full(self, *a, **k):
    k["connected"] = False
    return _orig(self, *a, **k)


pm.PlotManager.extract_mesh = _full
sys.argv = ["eval.py"] + sys.argv[1:]
runpy.run_path("eval.py", run_name="__main__")
