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
                render_res=24, dt_res=32, patience=2, c2f_event=False)
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
    # patience 0: the first committed window is the coarse epoch's stop event, which is what fires the switch
    res = run_pipeline(*clouds, prm, _cfg(c2f_event=True, render_res_hi=28, animations=3, patience=0), log=lambda *_: None)
    h = res["history"]
    switch = next(i for i, r in enumerate(h) if r.get("c2f_render_res") == 28)
    assert any(r.get("frame_end") for r in h[:switch])               # a stop event came first ...
    assert any("animation" in r for r in h[switch + 1:])              # ... and the run went on at 28 px
    assert len(packs) == 2
    assert packs[0].ot_scale == packs[1].ot_scale and packs[0].grid_ot is packs[1].grid_ot
    assert packs[0].unit_ratio == packs[1].unit_ratio
    assert starts == [None, None]                                 # a new resolution: a fresh search


def test_the_render_weight_is_calibrated_at_every_window(prm, clouds, monkeypatch):
    import physmorph.pipeline.render_loss as render_mod
    calls, update = [], render_mod.LambdaBalancer.update
    monkeypatch.setattr(render_mod.LambdaBalancer, "update",
                        lambda self, p, r: calls.append((p, r)) or update(self, p, r))
    res = run_pipeline(*clouds, prm, _cfg(animations=3), log=lambda *_: None)
    windows = [r for r in res["history"] if "animation" in r]
    assert 2 <= len(calls) <= len(windows)                        # held from the first window it was one call
    assert all(r["lambda"] > 0 for r in res["history"] if r.get("lambda") is not None)


def test_a_changed_render_weight_rescores_the_references():
    sel = Selection(PipelineConfig())
    rec = {"selection_merit": 1.0}
    sel.check_lambda(rec, 0.2, 2.0)
    sel.best, sel.last, sel.stale, sel.prev = 1.0, 0, 3, 1.0     # as an accepted window scored at 0.2 leaves them,
    sel.best_render = sel.prev_render = 2.0                      #   with a render term of 2.0
    rec2 = {"selection_merit": 0.9}
    sel.check_lambda(rec2, 0.3, 1.5)                             # the merit is linear in the weight: 1.2 at 0.3
    assert rec2["selection_epoch"] == 0 and sel.last == 0 and sel.stale == 3
    assert sel.best == pytest.approx(1.2) and sel.prev == pytest.approx(1.2) and sel.lam == 0.3
    rec3 = {"selection_merit": float("nan")}                     # no finite cost: nothing moves
    sel.check_lambda(rec3, 0.4, 1.0)
    assert sel.lam == 0.3 and sel.best == pytest.approx(1.2)


def test_delivery_is_the_best_merit_window_of_the_last_epoch():
    hist = [{"animation": 0, "frame_end": 9, "d_vol": 1., "selection_merit": 1.0, "selection_epoch": 0},
            {"animation": 1, "frame_end": 17, "d_vol": 1., "selection_merit": 0.5, "selection_epoch": 1},
            {"animation": 2, "frame_end": 25, "d_vol": 1., "selection_merit": 0.501, "selection_epoch": 1},
            {"animation": 3, "null_commit": 1, "frame_end": 33, "d_vol": 1., "selection_merit": 0.1,
             "selection_epoch": 1}]
    n, trunc = best_window(hist, 26, tol=0.003)
    assert n == 17 and trunc["best_animation"] == 2 and trunc["frames_dropped"] == 9
    assert best_window(hist[:1], 9, tol=0.003) == (9, None)
    # windows scored with different render weights are compared at the last one's: 1.0 at weight 0.5 with a render
    # term of 1.0 is 0.6 at weight 0.1, under the later window's 0.7
    moved = [{"animation": 0, "frame_end": 9, "d_vol": 1., "selection_merit": 1.0, "selection_epoch": 0,
              "lambda": 0.5, "d_render": 0.6, "d_pbr": 0.4},
             {"animation": 1, "frame_end": 17, "d_vol": 1., "selection_merit": 0.7, "selection_epoch": 0,
              "lambda": 0.1, "d_render": 0.6, "d_pbr": 0.4}]
    n, trunc = best_window(moved, 17, tol=0.003)
    assert n == 9 and trunc["best_animation"] == 1


def test_line_search_probe_is_diagnostic_only(prm, clouds, monkeypatch):
    """The first trial of the run fails its state check. With the probe on, that trial is re-run on
    dFc alone and on u alone and recorded; the accepted steps and losses equal the probe-off run."""
    import physmorph.pipeline.window.solve as solve_mod
    real = solve_mod.state_ok

    def run(probe):
        calls = []

        def first_trial_fails(e):
            calls.append(1)
            return False if len(calls) == 1 else real(e)
        monkeypatch.setattr(solve_mod, "state_ok", first_trial_fails)
        res = run_pipeline(*clouds, prm, _cfg(animations=1, ls_probe=probe), log=lambda *_: None)
        return next(r for r in res["history"] if r.get("frame_end"))

    off, on = run(False), run(True)
    assert off["ls_fail_state"] == on["ls_fail_state"] == 1 and off["ls_probe"] is None
    assert on["ls_trials"] == off["ls_trials"] and len(on["ls_probe"]) == on["ls_fail_state"] + on["ls_fail_merit"]
    row = on["ls_probe"][0]
    assert len(row) == 36 and row[6] == 0       # step, |du|max, 3 x (rel, d_vol, kin, render, ok), 3 x parts, cur parts,
    assert all(s >= 0. for s in row[32:])       # and the spread of three repeated evaluations of the current point
    assert on["iter_probe"] and all(len(r) == 14 for r in on["iter_probe"])   # one Adam-direction row per iteration
    assert on["start_ok"] == 1 and on["start_reason"] is None and on["commit_reason"] is None
    assert on["ls_fail_state_reason"] == {} or sum(on["ls_fail_state_reason"].values()) == on["ls_fail_state"]
    assert on["loss"] == pytest.approx(off["loss"], rel=1e-4)
    assert on["selection_merit"] == pytest.approx(off["selection_merit"], rel=1e-4)


def test_committed_windows_record_the_support_split(prm, clouds):
    """Each committed window records E, B, the support-gradient weight w (E / (E + w B))^2 and
    the per-particle penalty's quantiles; the ratio form keeps every particle at most radius^2."""
    from physmorph.thin import thin_set
    for form in ("log", "ratio"):
        cfg = _cfg(animations=3, support_form=form, work_telemetry=True)   # >1: a tiny cloud can null a window
        res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None, thin=thin_set(clouds[1], prm.dx, 300))
        rec = next(r for r in res["history"] if r.get("frame_end"))
        E, B, w = rec["sup_E"], rec["sup_B"], cfg.support_weight
        assert E > 0 and B >= 0 and rec["sup_w_eff"] == pytest.approx(w * (E / (E + w * B)) ** 2)
        assert rec["sup_pen_med"] <= rec["sup_pen_p99"] <= rec["sup_pen_max"]
        assert 0. <= rec["thin_uncovered"] <= 1.


def test_state_reason_names_the_failing_check():
    from physmorph.pipeline.window.rollout import Eval, state_ok, state_reason
    I = torch.eye(3).repeat(4, 1, 1).reshape(4, 9)
    z = torch.zeros(4, 3)
    e = Eval(z, I, z, *([None] * 9), jt=1.0)
    assert state_reason(e) is None and state_ok(e)
    assert state_reason(Eval(z, I, z, *([None] * 9), jt=1e-5)) == "jt"
    assert state_reason(Eval(z, I, z, *([None] * 9), jt=1.0, in_domain=False)) == "domain"
    F = I.clone(); F[0, 0] = -1.
    assert state_reason(Eval(z, F, z, *([None] * 9), jt=1.0)) == "det"
    xn = z.clone(); xn[0, 0] = float("nan")
    assert state_reason(Eval(xn, I, z, *([None] * 9), jt=1.0)) == "nonfinite"


def test_proximity_form_runs_and_records_the_gradient_ratio(prm, clouds):
    """support_form proximity: the target-surface proximity replaces the support; committed windows record its mean
    and its position-gradient norm against the transport's, and no bound weight."""
    res = run_pipeline(*clouds, prm, _cfg(animations=3, support_form="proximity", work_telemetry=True), log=lambda *_: None)
    rec = next(r for r in res["history"] if r.get("frame_end"))
    assert rec["sup_B"] >= 0. and rec["sup_w_eff"] is None and rec["sup_grad_ratio"] >= 0.
