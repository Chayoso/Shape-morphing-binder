"""D164: the commit check's replay noise is measured where the window's rollouts are (the user's decision of 2026-10-22).

(1) The measure: the largest of the window's own replay pairs (the start pair, each accepted candidate against the next
    iteration's taped rollout of the same control), never below one float32 ulp of E; --replay_noise start is the tag's
    (the start pair alone).
(2) The pairs exist where the definition says: one for each accepted candidate that the next iteration's gradient
    starts from; finite and of the replay noise's size.
(3) They are replays: under Warp's run-to-run deterministic atomics every pair and the commit agree bit for bit
    (tests/d164_det_runs.py, a subprocess).
(4) What the check exists for still holds: a commit rollout that is not the accepted candidate's (here it forgets the
    controls, a state one of the two evaluations does not share) is nulled as commit_replay, under both measures.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import physmorph.pipeline.run.runner as runner_mod
import physmorph.pipeline.window.solve as S
from physmorph.mpm.state import MPMParams
from physmorph.pipeline import PipelineConfig, run_pipeline

HERE = Path(__file__).parent


def _stub(mode, replays, floor=1e-3):
    return SimpleNamespace(cfg=SimpleNamespace(replay_noise=mode), win=SimpleNamespace(loss_floor_eff=floor),
                           replays=list(replays))


def test_the_measure_is_the_largest_pair_and_never_below_one_ulp():
    E = 0.0078125                                                   # 2^-7: one float32 ulp is 2^-30 there
    ulp = 2.0 ** -30 / E
    noise, res = S.WindowOptimizer.window_replay_noise(_stub("window", []), 0.0, E)
    assert res == pytest.approx(ulp) and noise == pytest.approx(ulp)   # a zero reading: below one ulp, not zero
    noise, _ = S.WindowOptimizer.window_replay_noise(_stub("window", [3e-7, 1e-6]), 2e-8, E)
    assert noise == 1e-6                                           # the largest pair
    noise, _ = S.WindowOptimizer.window_replay_noise(_stub("window", [1e-8]), 4e-6, E)
    assert noise == 4e-6                                           # the start pair is one of them
    noise, res = S.WindowOptimizer.window_replay_noise(_stub("start", [1e-6]), 0.0, E)
    assert noise == 0.0 and res is None                            # the tag's measure
    noise, res = S.WindowOptimizer.window_replay_noise(_stub("window", [1e-6]), 0.0, None)
    assert noise == 0.0 and res is None                            # no accepted candidate: nothing to compare


@pytest.fixture(scope="module")
def prm():
    return MPMParams(dx=1.0, nx=32, ny=32, nz=32)


@pytest.fixture(scope="module")
def clouds():
    rng = np.random.default_rng(11)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    return src, tgt


def _cfg(**kw):
    base = dict(T=4, iters=3, animations=2, loss_res=12, render_views=2, render_elevs=(0.0, 0.5), render_res=24,
                dt_res=32, patience=3)
    base.update(kw)
    return PipelineConfig(**base)


def _windows(monkeypatch, prm, clouds, **kw):
    optimize, stats = runner_mod.optimize_window, []

    def window(*a, **k):
        res = optimize(*a, **k)
        stats.append(dict(res.stats))
        return res

    monkeypatch.setattr(runner_mod, "optimize_window", window)
    run_pipeline(*clouds, prm, _cfg(**kw), log=lambda *_: None)
    return stats


def test_each_accepted_candidate_the_next_gradient_starts_from_is_a_pair(monkeypatch, prm, clouds):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    seen = []
    run = S.WindowOptimizer.run

    def run_(self, *a, **k):
        res = run(self, *a, **k)
        seen.append((res.stats, len(self.replays), [r["iter"] for r in res.hist]))
        return res

    monkeypatch.setattr(S.WindowOptimizer, "run", run_)
    _windows(monkeypatch, prm, clouds)
    assert seen
    for st, n, iters in seen:
        if st.get("null_reason") in ("commit_replay", "commit_jt", "commit_invalid", "merit_nonfinite"):
            continue                                                # the history was dropped with the window
        assert n == st["replay_pairs"]
        assert n == sum(1 for i in iters if i < 2), (n, iters)      # iters = 3: accepted at 0 and 1 are followed
        if st["accepted"] > 0:
            assert st["replay_noise"] >= max(st["replay_rel"], st["replay_res"])
            assert np.isfinite(st["replay_noise"]) and st["replay_noise"] < 1e-3
            assert st["commit_tol"] >= 10.0 * st["replay_res"]


def test_the_pairs_and_the_commit_are_replays_bit_for_bit_under_deterministic_atomics():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(HERE.parent), os.environ.get("PYTHONPATH", "")]))
    out = subprocess.run([sys.executable, str(HERE / "d164_det_runs.py")], capture_output=True, text=True, timeout=3600,
                         cwd=HERE.parent, env=env)
    assert out.returncode == 0, out.stderr[-4000:]
    rows = json.loads(out.stdout.strip().splitlines()[-1])["windows"]
    paired = [r for r in rows if r["replay_pairs"]]
    assert paired, rows
    for r in rows:
        assert r["replay_rel"] == 0.0, r
        assert r["replay_taped_max"] in (None, 0.0), r
        if r["commit_mismatch"] is not None:
            assert r["commit_mismatch"] == 0.0, r


@pytest.mark.parametrize("mode", ["window", "start"])
def test_a_commit_that_is_not_the_accepted_candidate_is_still_nulled(monkeypatch, prm, clouds, mode):
    """The commit rollout forgets the controls (zero dFc and u: the free motion from the start, a state the line search's
    evaluation does not share): it lands the window's whole descent above the accepted value, far beyond ten times any
    replay pair, and the window is nulled."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    roll = S.commit_rollout
    monkeypatch.setattr(S, "commit_rollout",
                        lambda win, obj, leaf, u, lam: roll(win, obj, torch.zeros_like(leaf), torch.zeros_like(u), lam))
    stats = _windows(monkeypatch, prm, clouds, animations=1, replay_noise=mode)
    s = stats[0]
    assert s["E_accept"] is not None, s
    assert s["commit_mismatch"] > s["commit_tol"], s
    assert s["null_reason"] == "commit_replay" and s["accepted"] == 0, s
