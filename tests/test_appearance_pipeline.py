"""D160 at the pipeline level (tests/appearance_runs.py, CUDA, deterministic): (1) the reduction in a whole run: the
appearance "white" (one channel of ones against a white target) is the frozen path bit for bit, trajectory and every
history field; (2) the null space in a whole run: with the render weight at 0 (the twin) the trajectory is the same bit
for bit with no paint and with two different paints, so the physics never reads which material carries which colour;
(3) with the render weight at its rule the paint changes the trajectory (the colour acts through the render alone)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).parent


def test_the_appearance_reduces_to_the_frozen_path_and_the_twin_is_blind_to_the_paint():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(HERE.parent), os.environ.get("PYTHONPATH", "")]))
    out = subprocess.run([sys.executable, str(HERE / "appearance_runs.py")], capture_output=True, text=True,
                         timeout=5400, cwd=HERE.parent, env=env)
    assert out.returncode == 0, out.stderr[-4000:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert all(v["committed"] > 0 for v in r.values()), r
    assert r["plain"]["full"] == r["plain2"]["full"], r                    # the deterministic mode holds
    assert r["white"]["full"] == r["plain"]["full"], r                     # (1) bit for bit, history included
    assert r["twinA"]["traj"] == r["twin0"]["traj"] == r["twinB"]["traj"], r   # (2) the twin is blind to the paint
    assert r["fullA"]["traj"] != r["plain"]["traj"], r                     # (3) the colour acts through the render
