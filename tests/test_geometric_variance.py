import pytest
import torch

from physmorph.pipeline.geometric_variance import path_rates, temporal_variance, path_telemetry


def test_uniform_drift_is_not_misreported_as_rest():
    start = torch.zeros(4, 3, dtype=torch.float64)
    positions = torch.arange(1., 5., dtype=torch.float64)[:, None, None].expand(4, 4, 3).clone()
    rates = path_rates(start, positions, positions[-1], .25)
    assert temporal_variance(rates) == 0
    report = path_telemetry(start, positions, positions[-1], rates, .25)
    assert report['net_rms_wu'] > 0 and report['saved_path_mean_wu'] > 0


def test_isolated_endpoint_jump_and_spreading_are_explicit():
    T, dt = 20, 1/240
    start = torch.zeros(5, 3, dtype=torch.float64)
    raw = start.expand(T, -1, -1).clone()
    end = start.clone(); end[:, 0] = .03
    variance = temporal_variance(path_rates(start, raw, end, dt))
    assert float(variance) == pytest.approx((T-1)/T**2*(.03/dt)**2)
    spread = torch.arange(1, T+1, dtype=raw.dtype)[:, None, None]*end[None]/T
    assert temporal_variance(path_rates(start, spread, end, dt)) < 1e-24
    assert path_telemetry(start, raw, end, raw, dt)['remap_rms_wu'] == pytest.approx(.03)


def test_gradient_uses_intermediate_positions_and_promoted_endpoint_only():
    torch.manual_seed(21)
    start = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    x = torch.randn(3, 4, 3, dtype=torch.float64, requires_grad=True)
    end = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    def loss(path, promoted):
        return temporal_variance(path_rates(start, path, promoted, .2))
    assert torch.autograd.gradcheck(loss, (x, end), eps=1e-6, atol=1e-7, rtol=1e-5)
    gx, ge, gs = torch.autograd.grad(loss(x, end), (x, end, start), allow_unused=True)
    assert gx[-1].count_nonzero() == 0 and gx[:-1].count_nonzero() > 0
    assert ge.count_nonzero() > 0 and gs is None


def test_T1_has_zero_variance_but_retains_terminal_motion():
    start = torch.zeros(2, 3)
    raw = start[None].clone().requires_grad_()
    end = torch.ones_like(start).requires_grad_()
    value = temporal_variance(path_rates(start, raw, end, .1))
    assert value == 0
    gr, ge = torch.autograd.grad(value, (raw, end))
    assert gr.count_nonzero() == 0 and ge.count_nonzero() == 0
    assert path_telemetry(start, raw, end, raw, .1)['remap_rms_wu'] > 0


@pytest.mark.parametrize('dt', [0., -1., float('nan'), float('inf')])
def test_bad_timestep_rejected(dt):
    x = torch.zeros(2, 3)
    with pytest.raises(ValueError, match='positive dt'):
        path_rates(x, x[None], x, dt)
