"""External-force conservation, adjoint and pin invariants on Warp CPU only."""
import dataclasses
import numpy as np
import pytest
import torch

from physmorph.mpm.function import RolloutSpec, PersistentAdjoint, warp_mpm_ext
from physmorph.mpm.state import MPMParams
from physmorph.pipeline.body_control import BodyControlBasis, rest_to_rest_pulse
from physmorph.pipeline.settlement import accepted_arrivals


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


def test_body_adjoint_matches_fd_and_persistent_rollout():
    s = spec(400., 200.)
    basis = BodyControlBasis(s.x0, s.prm.grid_min, s.prm.dx)
    c = (0.01 * torch.randn(basis.n_nodes, 3, generator=torch.Generator().manual_seed(3))).requires_grad_()
    dc = torch.zeros(s.T, len(s.x0), 3, 3)
    weight = torch.randn(len(s.x0), 3, generator=torch.Generator().manual_seed(4))
    def loss(coeff):
        x, _, v, _, _ = warp_mpm_ext(dc, s, body_t=basis.expand(coeff))
        return (x * weight).sum() + 0.01 * v.square().sum()
    g, = torch.autograd.grad(loss(c), c)
    d = torch.randn(c.shape, generator=torch.Generator().manual_seed(5)); d /= d.norm()
    eps = 2e-3
    fd = (loss(c.detach() + eps * d) - loss(c.detach() - eps * d)) / (2 * eps)
    assert float((g * d).sum()) == pytest.approx(float(fd), rel=0.025, abs=0.003)
    adj = PersistentAdjoint(s)
    x0 = warp_mpm_ext(dc, s, body_t=basis.expand(c))[0]
    xp = adj.apply(dc, body_t=basis.expand(c))[0]
    assert torch.allclose(xp, x0, atol=2e-6)
    gp, = torch.autograd.grad((xp * weight).sum(), c)
    gf, = torch.autograd.grad((warp_mpm_ext(dc, s, body_t=basis.expand(c))[0] * weight).sum(), c)
    assert torch.allclose(gp, gf, atol=2e-5, rtol=2e-4)


def test_body_force_cannot_move_pinned_particles():
    s = spec(pin=np.ones(40, np.float32))
    body = torch.full((40, 3), 0.02, requires_grad=True)
    x, _, v, _, _ = warp_mpm_ext(torch.zeros(s.T, 40, 3, 3), s, body_t=body)
    assert torch.equal(x, torch.from_numpy(s.x0))
    assert torch.count_nonzero(v) == 0
    g, = torch.autograd.grad(x.sum(), body)
    assert torch.count_nonzero(g) == 0


def test_accepted_arrival_detects_departure_and_new_arrival():
    x = np.array([[0.2, 0, 0], [0.01, 0, 0]], np.float32)
    assert accepted_arrivals(x, np.zeros_like(x), 0.1, [True, False]).tolist() == [False, True]
    assert not accepted_arrivals(x, None, None).any()
