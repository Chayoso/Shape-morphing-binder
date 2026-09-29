"""Actual CUDA graph masks and fixed-control observer parity."""
from contextlib import ExitStack
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from physmorph.compute import cuda_execution
from physmorph.mpm.function import PersistentAdjoint
from scripts.probes.fragment_adjoint_compare import FragmentAdjointObserver
from test_fragment_activity_adjoint import changing_activity_case, observation, body_direction
from test_withdrawal_adjoint_cuda import upload


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='hyde06 CUDA checks')


@pytest.mark.parametrize('mode', ['legacy', 'retained'])
def test_captured_observer_retains_actual_masks_and_changes_only_reverse(mode):
    spec, dc, body = changing_activity_case()
    with cuda_execution('cuda:0'):
        spec, (dc, body) = upload(spec, (dc, body))
        reference = PersistentAdjoint(spec, position_sequence=True)
        expected = reference.apply_with_positions(dc, body_t=body)
        expected_C = torch.stack([wp.to_torch(v).clone() for v in reference.traj.C])
        expected_gradient, = torch.autograd.grad(observation(expected), body)
        observer = FragmentAdjointObserver(mode)
        with ExitStack() as stack:
            observer.install(stack, SimpleNamespace(optimize_window=lambda **_: None))
            model = PersistentAdjoint(spec, position_sequence=True)
            assert model.g_fwd is not None and model.g_bwd is not None
            assert observer.rows == []  # Warmup and graph construction are excluded.
            actual = model.apply_with_positions(dc, body_t=body)
            gradient, = torch.autograd.grad(observation(actual), body)
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(torch.stack([wp.to_torch(v) for v in model.traj.C]), expected_C,
                                       rtol=1e-5, atol=1e-5)
            if mode == 'retained':
                torch.testing.assert_close(gradient, expected_gradient, rtol=2e-4, atol=2e-5)
            else:
                direction = body_direction(body)
                assert float((gradient*direction).sum())*float((expected_gradient*direction).sum()) < 0
            assert len(observer.rows) == 1
            data = observer.rows[0]['cohorts']['all']
            assert data['active'] == data['dynamic_only'] == data['different_from_final'] == [1, 1, 1, 0, 0, 0, 0, 0]
            assert data['temporal_flips'] == [0, 0, 1, 0, 0, 0, 0]
