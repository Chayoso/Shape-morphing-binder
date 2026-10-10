"""--grad_dump and --term_dump, the diagnostic dumps the cleanup (D141) kept, are read-only: a run with the hook on is
the run without it, bit for bit.

Two CUDA runs of one control differ in the transfers' atomics (the replay noise), so the runs go in one subprocess
under Warp's run-to-run deterministic atomics and torch's deterministic algorithms (tests/dump_runs.py), and the test
first asks that two runs without a hook repeat there bit for bit: the comparison is then exact.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).parent


def test_the_dump_hooks_change_nothing(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(HERE.parent), os.environ.get("PYTHONPATH", "")]))
    out = subprocess.run([sys.executable, str(HERE / "dump_runs.py"), str(tmp_path)], capture_output=True,
                         text=True, timeout=3600, cwd=HERE.parent, env=env)
    assert out.returncode == 0, out.stderr[-4000:]
    rows = json.loads(out.stdout.strip().splitlines()[-1])
    assert rows["committed"] > 0, rows
    assert rows["repeat"], rows                     # the deterministic mode holds: two plain runs repeat
    assert rows["grad"] and rows["grad_files"] > 0, rows
    assert rows["term"] and rows["term_files"] > 0, rows
