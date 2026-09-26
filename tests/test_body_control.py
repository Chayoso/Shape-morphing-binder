"""External-force conservation, adjoint and pin invariants on Warp CPU only."""
import dataclasses
import numpy as np
import pytest
import torch

from physmorph.mpm.function import RolloutSpec, PersistentAdjoint, warp_mpm_ext
from physmorph.mpm.state import MPMParams
from physmorph.pipeline.body_control import BodyControlBasis, rest_to_rest_pulse, terminal_velocity_pulse


def spec(lam=0.0, mu=0.0, pin=None):
    x = np.random.default_rng(2).uniform(-0.8, 0.8, (40, 3)).astype(np.float32)
    prm = MPMParams(dx=0.5, dt=0.005, nx=20, ny=20, nz=20,
                    grid_min=(-5., -5., -5.), drag=0., smoothing=1.)
    return RolloutSpec(x, 1., lam, mu, prm, 6, device="cpu", body_ctrl=True,
                       vol0=np.full(len(x), 0.02, np.float32), pin=pin)


def test_pulse_zero_impulse_and_unit_displacement():
    for T in (2, 3, 6, 20):
        a = rest_to_rest_pulse(T, 0.005)
        assert abs(float(a.sum())) < 1e-3
        assert np.dot(T - np.arange(T), a) * 0.005 ** 2 == pytest.approx(1., rel=1e-6)
    with pytest.raises(ValueError):
        rest_to_rest_pulse(1, 0.005)


def test_free_body_translation_stops_and_is_mass_independent():
    s = spec()
    dc = torch.zeros(s.T, len(s.x0), 3, 3)
    body = torch.tensor([0.025, -0.01, 0.015]).expand(len(s.x0), -1).contiguous()
    for mass in (1., 0.125):
        x, _, v, _, _ = warp_mpm_ext(dc, dataclasses.replace(s, m=mass), body_t=body)
        assert torch.allclose(x, torch.from_numpy(s.x0) + body, atol=3e-6)
        assert float(v.abs().max()) < 3e-5


@pytest.mark.parametrize('modes', [1, 2])
def test_body_adjoint_matches_fd_and_persistent_rollout(modes):
    s = dataclasses.replace(spec(400., 200.), body_modes=modes)
    basis = BodyControlBasis(s.x0, s.prm.grid_min, s.prm.dx)
    c = (0.01 * torch.randn(basis.n_nodes, 3 * modes, generator=torch.Generator().manual_seed(3))).requires_grad_()
    def field(coeff):
        return basis.expand(coeff).reshape(len(s.x0), modes, 3).permute(1, 0, 2).reshape(-1, 3).contiguous()
    dc = torch.zeros(s.T, len(s.x0), 3, 3)
    weight = torch.randn(len(s.x0), 3, generator=torch.Generator().manual_seed(4))
    def loss(coeff):
        x, _, v, _, _ = warp_mpm_ext(dc, s, body_t=field(coeff))
        return (x * weight).sum() + 0.01 * v.square().sum()
    g, = torch.autograd.grad(loss(c), c)
    d = torch.randn(c.shape, generator=torch.Generator().manual_seed(5)); d /= d.norm()
    eps = 2e-3
    fd = (loss(c.detach() + eps * d) - loss(c.detach() - eps * d)) / (2 * eps)
    assert float((g * d).sum()) == pytest.approx(float(fd), rel=0.025, abs=0.003)
    adj = PersistentAdjoint(s)
    x0 = warp_mpm_ext(dc, s, body_t=field(c))[0]
    xp = adj.apply(dc, body_t=field(c))[0]
    assert torch.allclose(xp, x0, atol=2e-6)
    gp, = torch.autograd.grad((xp * weight).sum(), c)
    gf, = torch.autograd.grad((warp_mpm_ext(dc, s, body_t=field(c))[0] * weight).sum(), c)
    assert torch.allclose(gp, gf, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize('modes', [1, 2])
def test_body_force_cannot_move_pinned_particles(modes):
    s = dataclasses.replace(spec(pin=np.ones(40, np.float32)), body_modes=modes)
    body = torch.full((modes * 40, 3), 0.02, requires_grad=True)
    x, _, v, _, _ = warp_mpm_ext(torch.zeros(s.T, 40, 3, 3), s, body_t=body)
    assert torch.equal(x, torch.from_numpy(s.x0))
    assert torch.count_nonzero(v) == 0
    g, = torch.autograd.grad(x.sum(), body)
    assert torch.count_nonzero(g) == 0


def test_terminal_mode_independently_brakes_a_moving_free_body():
    for T in (2, 3, 6, 20):
        a = terminal_velocity_pulse(T, 0.005)
        assert np.dot(T - np.arange(T), a) * 0.005 ** 2 == pytest.approx(0., abs=1e-6)
        assert a.sum() * 0.005 * (T * 0.005) == pytest.approx(1., rel=1e-6)
    v0 = np.tile([0.4, -0.2, 0.1], (40, 1)).astype(np.float32)
    s = dataclasses.replace(spec(), body_modes=2, v0=v0)
    desired = torch.tensor([0.01, 0.02, -0.01]).expand(40, 3)
    carried = torch.from_numpy(v0) * (s.T * s.prm.dt)
    body = torch.cat([desired - carried, -carried])
    for mass in (1., 0.125):
        x, _, v, _, _ = warp_mpm_ext(torch.zeros(s.T, 40, 3, 3), dataclasses.replace(s, m=mass), body_t=body)
        assert torch.allclose(x, torch.from_numpy(s.x0) + desired, atol=3e-6)
        assert float(v.abs().max()) < 3e-5


def test_body_mode_shape_mismatch_is_rejected():
    s = dataclasses.replace(spec(), body_modes=2)
    dc = torch.zeros(s.T, 40, 3, 3)
    for rollout in (lambda b: warp_mpm_ext(dc, s, body_t=b),
                    lambda b: PersistentAdjoint(s).apply(dc, body_t=b)):
        with pytest.raises(ValueError, match='shape'):
            rollout(torch.zeros(40, 3))
