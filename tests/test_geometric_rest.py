"""Cancellation and derivative checks for the terminal geometric-rest quantity."""
import pytest
import torch

from physmorph.mpm.endpoint_filter import FixedEndpointFilter
from physmorph.pipeline.geometric_rest import terminal_motion


def test_saved_transition_cancellation_is_not_rest():
    previous = torch.zeros(4, 3, dtype=torch.float64, requires_grad=True)
    raw = torch.tensor([[1., 2., 0.], [0., 0., 3.], [9., 8., 7.], [5., 4., 3.]],
                       dtype=torch.float64, requires_grad=True)
    promoted = torch.zeros_like(previous, requires_grad=True)
    cohort = torch.tensor([True, True, False, False])
    before = [x.detach().clone() for x in (raw, previous, promoted)]
    result = terminal_motion(raw, previous, promoted, cohort, .5)
    assert result['delivered_sq'] == 0
    assert result['total'] == 28
    assert result['cross'] == -28
    gp = torch.autograd.grad(result['total'], previous)[0]
    torch.testing.assert_close(gp[:2], -2 * raw[:2])
    assert gp[2:].count_nonzero() == 0
    for actual, expected in zip((raw, previous, promoted), before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_empty_cohort_stays_connected_with_zero_gradients():
    values = [torch.randn(5, 3, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    result = terminal_motion(*values, torch.zeros(5, dtype=torch.bool), .25)
    assert result['total'] == 0
    for grad in torch.autograd.grad(result['total'], values):
        assert grad.count_nonzero() == 0


def test_combined_previous_and_masked_xpic_derivative():
    torch.manual_seed(94)
    start = torch.rand(7, 3, dtype=torch.float64) * .5 + 1.
    masses = torch.linspace(.4, 2., 7, dtype=torch.float64)
    operator = FixedEndpointFilter(start, 1., (0., 0., 0.), (5, 5, 5), m=masses)
    pins = torch.tensor([True, False, False, False, False, False, False])
    cohort = torch.tensor([False, True, False, True, False, True, False])
    raw = (start + .1 * torch.randn_like(start)).requires_grad_()
    previous = (start + .1 * torch.randn_like(start)).requires_grad_()

    def loss(r, p):
        return terminal_motion(r, p, operator.endpoint(r, pins), cohort, .125)['total']

    assert torch.autograd.gradcheck(loss, (raw, previous), eps=1e-6, atol=1e-7, rtol=1e-5)
    parts = terminal_motion(raw, previous, operator.endpoint(raw, pins), cohort, .125)
    torch.testing.assert_close(parts['delivered_sq'], parts['total'] + parts['cross'])
    assert parts['delivered_sq'] <= 2 * parts['total']


@pytest.mark.parametrize('dt', [0., -1., float('nan'), float('inf')])
def test_invalid_timestep_rejected(dt):
    x = torch.zeros(2, 3)
    with pytest.raises(ValueError, match='positive dt'):
        terminal_motion(x, x, x, torch.ones(2, dtype=torch.bool), dt)
