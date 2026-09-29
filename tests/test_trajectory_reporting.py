"""Exact parity with the former final-trajectory reporting expressions."""
import math

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from physmorph.pipeline.trajectory_reporting import trajectory_health


def legacy_trajectory_health(states):
    inv_any = None
    jmin_traj = float('inf')
    with torch.no_grad():
        for state in states:
            Ft = state.reshape(-1, 3, 3).float()
            det_t = torch.linalg.det(Ft)
            bad = det_t <= 0.
            inv_any = bad if inv_any is None else (inv_any | bad)
            jmin_traj = min(jmin_traj, float(det_t.min().item()))
        n_inv_steps = int(inv_any.sum().item()) if inv_any is not None else 0
    return n_inv_steps, jmin_traj


def reporting_states(device='cpu', dtype=torch.float32):
    values = torch.linspace(-.3, .4, 63, dtype=dtype).reshape(7, 3, 3)
    base = torch.eye(3, dtype=dtype)[None] + values.sin()*.07
    states = [base.clone() for _ in range(4)]
    states[0][1, 0] *= -1
    states[1][2, 1] *= -1
    states[2][1, 0] *= -1  # Count a particle once, not every inverted step.
    states[3][4] = 0.
    return [state.to(device=device).requires_grad_() for state in states]


def assert_same(actual, expected):
    assert actual == expected
    assert type(actual[0]) is int and type(actual[1]) is float
    assert math.copysign(1., actual[1]) == math.copysign(1., expected[1])


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_real_determinants_legacy_parity_and_owned_inputs(dtype):
    states = reporting_states(dtype=dtype)
    before = [state.detach().clone() for state in states]
    actual = trajectory_health(iter(states))
    assert_same(actual, legacy_trajectory_health(states))
    assert actual[0] == 3 and actual[1] < 0.
    for state, old in zip(states, before):
        torch.testing.assert_close(state, old, rtol=0, atol=0)
        assert state.grad is None


def test_determinants_remain_float32_even_for_float64_input():
    state = torch.eye(3, dtype=torch.float64)[None]
    state[0, 0, 0] += 2.**-40
    assert_same(trajectory_health([state]), (0, 1.))
    assert float(torch.linalg.det(state)[0]) != 1.


def test_flat_particle_matrices_and_noncontiguous_inputs_keep_original_reshape():
    states = reporting_states()
    flattened = [state.reshape(7, 9) for state in states]
    assert_same(trajectory_health(flattened), legacy_trajectory_health(flattened))
    transposed = [state.transpose(1, 2) for state in states]
    assert not transposed[0].is_contiguous()
    assert_same(trajectory_health(transposed), legacy_trajectory_health(transposed))


@pytest.mark.parametrize('rows,expected_count,expected_min,negative_zero', [
    ([[float('nan'), -2.], [3., 4.]], 1, 3., False),
    ([[float('nan'), 2.], [float('nan'), -4.]], 1, float('inf'), False),
    ([[float('inf'), 3.], [float('-inf'), float('nan')]], 1, 3., False),
    ([[float('inf'), float('inf')], [float('-inf'), 2.]], 1, float('-inf'), False),
    ([[-0.], [0.]], 1, 0., True),
    ([[0.], [-0.]], 1, 0., False),
    ([[2., -3.], [-1., 4.], [5., -2.]], 2, -3., False),
])
def test_controlled_reduction_oracle_retains_nan_inf_or_and_signed_zero(rows, expected_count, expected_min,
                                                                      negative_zero, monkeypatch):
    # Test reporting semantics independently of how a particular LAPACK/CUDA
    # determinant routine represents singular/infinite input matrices.
    calls = []
    def determinant(state):
        assert state.dtype == torch.float32
        row = rows[len(calls) % len(rows)]
        calls.append(tuple(state.shape))
        return torch.tensor(row, dtype=torch.float32, device=state.device)
    monkeypatch.setattr(torch.linalg, 'det', determinant)
    states = [torch.eye(3, dtype=torch.float64).repeat(len(row), 1, 1) for row in rows]
    actual = trajectory_health(states)
    assert_same(actual, legacy_trajectory_health(states))
    assert actual[0] == expected_count and actual[1] == expected_min
    if actual[1] == 0.:
        assert (math.copysign(1., actual[1]) < 0) == negative_zero
    assert len(calls) == 2*len(rows)


class ScalarExtractionCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default:
            self.count += 1
        return func(*args, **(kwargs or {}))


def test_cpu_transfer_boundary_called_once_without_individual_scalar_extractions(monkeypatch):
    states = reporting_states()
    with ScalarExtractionCounter() as legacy:
        expected = legacy_trajectory_health(states)
    assert legacy.count == len(states)+1
    original_cpu, original_list = torch.Tensor.cpu, torch.Tensor.tolist
    copies, lists = [], []
    def cpu(value, *args, **kwargs):
        copies.append((tuple(value.shape), value.dtype))
        return original_cpu(value, *args, **kwargs)
    def tolist(value):
        lists.append(tuple(value.shape))
        return original_list(value)
    monkeypatch.setattr(torch.Tensor, 'cpu', cpu)
    monkeypatch.setattr(torch.Tensor, 'tolist', tolist)
    with ScalarExtractionCounter() as actual_counter:
        actual = trajectory_health(states)
    assert_same(actual, expected)
    assert actual_counter.count == 0
    assert copies == [((len(states)+1,), torch.float64)]
    assert lists == [(len(states)+1,)]


def test_empty_horizon_has_no_tensor_or_transfer_requirement(monkeypatch):
    def forbidden(*_a, **_k):
        pytest.fail('empty horizon should not create a host transfer')
    monkeypatch.setattr(torch.Tensor, 'cpu', forbidden)
    assert_same(trajectory_health(iter(())), legacy_trajectory_health(()))
    assert trajectory_health([]) == (0, float('inf'))


def test_empty_particle_step_preserves_original_reduction_error():
    states = [torch.empty((0, 3, 3))]
    with pytest.raises(RuntimeError, match='min'):
        legacy_trajectory_health(states)
    with pytest.raises(RuntimeError, match='min'):
        trajectory_health(states)
