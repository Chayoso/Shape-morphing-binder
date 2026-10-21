"""run_pics.py SCRIPT [args] -- D155b's proof: SCRIPT as __main__ with run/morph.SKIP_UNUSED_PICTURES False (the packs on
the way draw the particle cloud's pictures as before), to compare the run with the skip bit for bit (under det_run.py)."""
import os
import runpy
import sys

import physmorph.pipeline.run.morph as morph  # noqa: E402

morph.SKIP_UNUSED_PICTURES = False
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
runpy.run_path(sys.argv[0], run_name="__main__")
