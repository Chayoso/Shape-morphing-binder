"""Observer parity and deliberate old-adjoint reproduction on CPU."""
from contextlib import ExitStack
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from physmorph.mpm.function import PersistentAdjoint
from physmorph.mpm.traj import Trajectory
from scripts.probes.fragment_adjoint_compare import FragmentAdjointObserver
from test_fragment_activity_adjoint import changing_activity_case, observation, body_direction


@pytest.mark.parametrize('mode', ['legacy', 'retained'])
def test_observer_preserves_primal_and_records_real_masks_before_overwrite(mode):
    spec, dc, body = changing_activity_case()
    baseline = PersistentAdjoint(spec, position_sequence=True)
    wanted = baseline.apply_with_positions(dc, body_t=body)
    wanted_C = torch.stack([wp.to_torch(v).clone() for v in baseline.traj.C])
    wanted_gradient, = torch.autograd.grad(observation(wanted), body)
    runner = SimpleNamespace(optimize_window=lambda *_, **__: None)
    original = Trajectory.__init__
    observer = FragmentAdjointObserver(mode)
    with ExitStack() as stack:
        observer.install(stack, runner)
        adj = PersistentAdjoint(spec, position_sequence=True)
        actual = adj.apply_with_positions(dc, body_t=body)
        gradient, = torch.autograd.grad(observation(actual), body)
        assert len(observer.rows) == 1
        for actual_value, expected in zip(actual, wanted):
            torch.testing.assert_close(actual_value, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.stack([wp.to_torch(v) for v in adj.traj.C]), wanted_C, rtol=0, atol=0)
        if mode == 'retained':
            torch.testing.assert_close(gradient, wanted_gradient, rtol=0, atol=0)
        else:
            direction = body_direction(body)
            assert float((gradient*direction).sum()) * float((wanted_gradient*direction).sum()) < 0
        model = observer.models[0]
        assert model['retained_allocations'] == model['observation_buffers'] == spec.T
        assert model['reverse_unique_buffers'] == (spec.T if mode == 'retained' else 1)
        row = observer.rows[0]['cohorts']
        assert row['all']['count'] == 24 and row['layer_free']['count'] == 0
        assert row['all']['active'] == [1, 1, 1, 0, 0, 0, 0, 0]
        assert row['all']['different_from_final'] == [1, 1, 1, 0, 0, 0, 0, 0]
        assert row['all']['temporal_flips'] == [0, 0, 1, 0, 0, 0, 0]
        assert row['all']['dynamic_only'] == row['all']['active']
    assert Trajectory.__init__ is original


def test_observer_restores_attempt_and_patches_after_failure():
    def failing(**kwargs):
        assert observer.attempt == 17
        raise ValueError('deliberate failure')
    runner = SimpleNamespace(optimize_window=failing)
    observer = FragmentAdjointObserver('retained')
    original = Trajectory.__init__
    with pytest.raises(ValueError, match='deliberate'), ExitStack() as stack:
        observer.install(stack, runner)
        runner.optimize_window(win_index=17)
    assert observer.attempt is None and runner.optimize_window is failing
    assert Trajectory.__init__ is original
