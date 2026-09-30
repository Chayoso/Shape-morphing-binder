"""Contracts of settled transport on a small cloud (CUDA): the phase velocity cost, the
domain check of whole trajectories, window rejection on a non-finite or unsolved cost,
delivery without null padding, the calibrations across the coarse-to-fine switch, the
cost epochs of the acceptance, and the delivered slice."""
import numpy as np
import pytest
import torch

import physmorph.pipeline.run.runner as runner_mod
import physmorph.pipeline.target as target_mod
from physmorph.mpm.state import MPMParams
from physmorph.pipeline import PipelineConfig, run_pipeline
from physmorph.pipeline.run.selection import Selection, best_window
from physmorph.pipeline.window import WindowResult


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
    base = dict(T=4, iters=2, animations=2, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                render_res=24, dt_res=32, patience=2, c2f_at=0.0)
    base.update(kw)
    return PipelineConfig(**base)


def _unmoved(res, src):
    return all(np.array_equal(x, src) for x in res["frames"].x)


def test_phase_velocity_cost_allows_progress_but_penalizes_released_drift():
    from physmorph.pipeline.window.objective import velocity_variance
    v = torch.tensor([[[1., 0., 0.]], [[3., 0., 0.]], [[0., 0., 0.]], [[0., 0., 0.]]], requires_grad=True)
    value = velocity_variance(v, 2)
    assert float(value) == pytest.approx(.5)
    expected = torch.tensor([[[-.5, 0., 0.]], [[.5, 0., 0.]], [[0., 0., 0.]], [[0., 0., 0.]]])
    torch.testing.assert_close(torch.autograd.grad(value, v)[0], expected)
    assert float(velocity_variance(torch.cat((torch.ones(20, 2, 3), torch.zeros(20, 2, 3))), 20)) == 0.
    drift = torch.ones(4, 1, 3, requires_grad=True)                 # a constant released velocity
    cost = velocity_variance(drift, 2)                              # is drift, not rest
    assert float(cost) == pytest.approx(1.5)
    grad, = torch.autograd.grad(cost, drift)
    torch.testing.assert_close(grad[:2], torch.zeros(2, 1, 3))
    torch.testing.assert_close(grad[2:], torch.full((2, 1, 3), .5))


@pytest.mark.parametrize('bad', [float('nan'), 31.])
def test_trajectory_check_rejects_a_midpoint_escape_with_a_safe_endpoint(prm, bad):
    from physmorph.pipeline.window.setup import positions_in_domain
    x = torch.zeros(9, 3, 3, device="cuda")
    assert positions_in_domain(x, prm)
    x[4, 0, 0] = bad
    assert not positions_in_domain(x, prm)


def test_a_nonfinite_window_cost_cannot_commit(prm, clouds, monkeypatch):
    optimize = runner_mod.optimize_window

    def corrupted(*args, **kw):
        res = optimize(*args, **kw)
        assert res.commit is not None
        res.stats["selection_merit"] = float("nan")
        return res

    monkeypatch.setattr(runner_mod, "optimize_window", corrupted)
    res = run_pipeline(*clouds, prm, _cfg(animations=1), log=lambda *_: None)
    assert not any("frame_end" in h for h in res["history"])
    assert any(h.get("outer_rejected") for h in res["history"])
    assert _unmoved(res, clouds[0])


def test_an_unsolved_transport_cannot_initialize_an_accepted_commit(prm, clouds, monkeypatch):
    from physmorph.losses.grid_ot import GridSinkhornLoss
    optimize, energy = runner_mod.optimize_window, GridSinkhornLoss.state_energy
    outer = {"on": False}

    def window(*args, **kw):
        outer["on"] = False
        res = optimize(*args, **kw)
        outer["on"] = True
        return res

    def failed_outer_energy(self, x, *args, **kw):
        return x.new_tensor(float("inf")) if outer["on"] else energy(self, x, *args, **kw)

    monkeypatch.setattr(runner_mod, "optimize_window", window)
    monkeypatch.setattr(GridSinkhornLoss, "state_energy", failed_outer_energy)
    res = run_pipeline(*clouds, prm, _cfg(animations=1), log=lambda *_: None)
    assert any(h.get("outer_rejected") for h in res["history"])
    assert _unmoved(res, clouds[0])


def test_null_windows_add_no_simulated_time_and_are_not_delivered(prm, clouds, monkeypatch):
    optimize, first = runner_mod.optimize_window, []

    def one_commit(*args, **kw):
        if not first:
            res = optimize(*args, **kw)
            assert res.commit is not None
            first.append(res.stats)
            return res
        return WindowResult(commit=None, hist=[], stats=dict(first[0], accepted=0, grad_converged=False))

    monkeypatch.setattr(runner_mod, "optimize_window", one_commit)
    res = run_pipeline(*clouds, prm, _cfg(animations=3, patience=10), log=lambda *_: None)
    accepted = next(r for r in res["history"] if r.get("frame_end"))
    assert len(res["frames"]) == accepted["frame_end"] == res["deliver_n"]
    assert sum(1 for r in res["history"] if r.get("no_simulated_time")) == 2


def test_transport_calibration_survives_the_render_resolution_change(prm, clouds, monkeypatch):
    build, packs, starts = target_mod.build_target, [], []
    optimize = runner_mod.optimize_window

    def tracked(*args, **kw):
        packs.append(build(*args, **kw))
        return packs[-1]

    def window(*args, **kw):
        starts.append(args[3].settled_step)
        return optimize(*args, **kw)

    monkeypatch.setattr(target_mod, "build_target", tracked)
    monkeypatch.setattr(runner_mod, "build_target", tracked)
    monkeypatch.setattr(runner_mod, "optimize_window", window)
    run_pipeline(*clouds, prm, _cfg(c2f_at=.5, render_res_hi=28), log=lambda *_: None)
    assert len(packs) == 2
    assert packs[0].ot_scale == packs[1].ot_scale and packs[0].grid_ot is packs[1].grid_ot
    assert packs[0].unit_ratio == packs[1].unit_ratio
    assert all(p.settled_scale is not None for p in packs)
    assert packs[0].settled_scale is not packs[1].settled_scale   # the render weight: recalibrated
    assert starts == [None, None]                                 # a new resolution: a fresh search


def test_a_changed_render_weight_opens_a_new_cost_epoch():
    sel = Selection(PipelineConfig())
    rec = {"selection_merit": 1.0}
    sel.check_lambda(rec, 0.2)
    sel.best, sel.last, sel.stale = 1.0, 0, 3
    rec2 = {"selection_merit": 0.9}
    sel.check_lambda(rec2, 0.3)
    assert rec2["selection_epoch"] == 1 and sel.last is None and sel.stale == 0
    rec3 = {"selection_merit": float("nan")}                     # no finite cost: no new epoch
    sel.check_lambda(rec3, 0.4)
    assert rec3["selection_epoch"] == 1 and sel.lam == 0.3


def test_delivery_is_the_best_merit_window_of_the_last_epoch():
    hist = [{"animation": 0, "frame_end": 9, "d_vol": 1., "selection_merit": 1.0, "selection_epoch": 0},
            {"animation": 1, "frame_end": 17, "d_vol": 1., "selection_merit": 0.5, "selection_epoch": 1},
            {"animation": 2, "frame_end": 25, "d_vol": 1., "selection_merit": 0.501, "selection_epoch": 1},
            {"animation": 3, "null_commit": 1, "frame_end": 33, "d_vol": 1., "selection_merit": 0.1,
             "selection_epoch": 1}]
    n, trunc = best_window(hist, 26, tol=0.003)
    assert n == 17 and trunc["best_animation"] == 2 and trunc["frames_dropped"] == 9
    assert best_window(hist[:1], 9, tol=0.003) == (9, None)
