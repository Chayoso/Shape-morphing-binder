"""CPU correctness of the fixed endpoint map, including its non-symmetric adjoint."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from physmorph.mpm.endpoint_filter import FixedEndpointFilter
from physmorph.mpm.gridfilter import grid_project


def _cloud(dtype=torch.float64):
    # Truncated wall stencils, unequal/zero masses, and a wholly unsupported row.
    x = torch.tensor([[-.7, .2, .3], [.15, .45, .8], [1.2, 1.1, .7],
                      [2.6, 1.8, 1.5], [.6, .9, .4], [-4., .1, .2]], dtype=dtype)
    m = torch.tensor([.2, 3., .7, 2., 0., 1.4], dtype=dtype)
    return x, m


def _dense_pic(x, m, dx=1., origin=(0., 0., 0.), dims=(3, 3, 3)):
    """Independent small oracle: all particle/grid pairs, no truncated stencil loops."""
    nodes = torch.cartesian_prod(*(torch.arange(n, dtype=x.dtype) for n in dims))
    distance = ((x[:, None] - torch.tensor(origin, dtype=x.dtype)) / dx - nodes).abs()
    basis = torch.zeros_like(distance)
    inside = distance < 1
    shell = (distance >= 1) & (distance < 2)
    basis[inside] = 2/3 - distance[inside]**2 + distance[inside]**3/2
    basis[shell] = (2-distance[shell])**3/6
    W = basis.prod(-1)
    S = W.sum(1).clamp_min(1e-12)
    D = (W * m[:, None]).sum(0).clamp_min(1e-12)
    return (W/S[:, None]) @ ((W * m[:, None])/D).T


def _dense_h(x, m, order=5):
    P = _dense_pic(x, m)
    identity = torch.eye(len(x), dtype=x.dtype)
    return identity - torch.linalg.matrix_power(identity-P, order)


@pytest.mark.parametrize('order', [1, 5])
@pytest.mark.parametrize('mass_mode', ['unit', 'scalar', 'nonuniform'])
def test_legacy_forward_parity(order, mass_mode):
    x, mass = _cloud(torch.float32)
    m = None if mass_mode == 'unit' else (2.3 if mass_mode == 'scalar' else mass)
    generator = torch.Generator().manual_seed(15)
    field = torch.randn(len(x), 3, generator=generator)
    expected, _ = grid_project(field.numpy(), x.numpy(), 1., (0., 0., 0.), (3, 3, 3),
                               m=m.numpy() if torch.is_tensor(m) else m,
                               device='cpu', order=order)
    actual = FixedEndpointFilter(x, 1., (0., 0., 0.), (3, 3, 3), m=m, order=order).apply_H(field)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize('order', [1, 5])
@pytest.mark.parametrize('channels', [1, 7])
def test_dense_oracle_and_double_dot_product(order, channels):
    x, m = _cloud()
    op = FixedEndpointFilter(x, 1., (0., 0., 0.), (3, 3, 3), m=m, order=order)
    H = _dense_h(x, m, order)
    generator = torch.Generator().manual_seed(33)
    d = torch.randn(len(x), channels, dtype=x.dtype, generator=generator)
    g = torch.randn(len(x), channels, dtype=x.dtype, generator=generator)
    torch.testing.assert_close(op.apply_H(d), H @ d, atol=3e-14, rtol=3e-13)
    torch.testing.assert_close(op.apply_HT(g), H.T @ g, atol=3e-14, rtol=3e-13)
    torch.testing.assert_close((op.apply_H(d)*g).sum(), (d*op.apply_HT(g)).sum(),
                               atol=3e-14, rtol=3e-13)
    # This fixture would catch using H in place of H^T at a truncated wall.
    assert (H-H.T).abs().max() > .1
    assert torch.count_nonzero(op.apply_H(d)[-1]) == 0  # no retained grid support
    assert torch.count_nonzero(op.apply_HT(g)[4]) == 0  # zero particle mass


def test_endpoint_pins_and_gradient_are_HT_Q_not_Q_HT():
    x, m = _cloud()
    op = FixedEndpointFilter(x, 1., (0., 0., 0.), (3, 3, 3), m=m)
    H = _dense_h(x, m)
    pins = torch.tensor([True, False, False, False, False, False])
    d = torch.arange(len(x)*3, dtype=x.dtype).reshape(-1, 3) / 13
    raw = (x+d).requires_grad_()
    out = op.endpoint(raw, pins)
    expected = H @ d
    expected[pins] = 0
    torch.testing.assert_close(out, x+expected)
    assert torch.equal(out[pins], x[pins])
    assert torch.equal(out[-1], x[-1])
    g = torch.arange(1, len(x)*3+1, dtype=x.dtype).reshape(-1, 3)
    expected_g = g.clone()
    expected_g[pins] = 0
    gradient, = torch.autograd.grad((out*g).sum(), raw)
    torch.testing.assert_close(gradient, H.T @ expected_g, atol=3e-14, rtol=3e-13)
    assert gradient[pins].abs().max() > .1  # pinned input can influence free output
    zero_g, = torch.autograd.grad(op.endpoint(raw, pins)[pins].sum(), raw)
    assert torch.count_nonzero(zero_g) == 0


def test_gradcheck_fixed_stencil_and_transpose():
    x = torch.tensor([[-.4, .2, .3], [.3, .6, .7], [1.3, 1.1, .2]], dtype=torch.float64)
    m = torch.tensor([.3, 2., 1.], dtype=x.dtype)
    op = FixedEndpointFilter(x, 1., (0., 0., 0.), (3, 3, 3), m=m)
    raw = (x + .03).requires_grad_()
    pins = torch.tensor([False, True, False])
    assert torch.autograd.gradcheck(lambda r: op.endpoint(r, pins), (raw,),
                                    eps=1e-6, atol=2e-8, rtol=2e-6)
    field = torch.tensor([[.2], [-.4], [.1]], dtype=x.dtype, requires_grad=True)
    assert torch.autograd.gradcheck(op.apply_HT, (field,), eps=1e-6, atol=2e-8, rtol=2e-6)


def test_owned_stencil_no_input_mutation_or_reference_gradients():
    x, m = _cloud()
    x.requires_grad_()
    m.requires_grad_()
    origin = torch.zeros(3, dtype=x.dtype, requires_grad=True)
    originals = [value.detach().clone() for value in (x, m, origin)]
    op = FixedEndpointFilter(x, 1., origin, (3, 3, 3), m=m)
    raw = (x.detach()+.1).requires_grad_()
    pins = torch.tensor([0., 1., 0., 0., 0., 0.], dtype=x.dtype, requires_grad=True)
    before = raw.detach().clone()
    out = op.endpoint(raw, pins)
    gradients = torch.autograd.grad(out.square().sum(), (raw, x, m, origin, pins), allow_unused=True)
    assert gradients[0] is not None and all(g is None for g in gradients[1:])
    for value, expected in zip((x, m, origin), originals):
        assert torch.equal(value.detach(), expected)
    assert torch.equal(raw.detach(), before)
    expected = op.apply_H(raw.detach())
    with torch.no_grad():
        x.add_(20)
        m.mul_(7)
        origin.add_(3)
    op.x0.fill_(100)  # accessor owns a copy too
    torch.testing.assert_close(op.apply_H(raw.detach()), expected, rtol=0, atol=0)
    with pytest.raises(FrozenInstanceError):
        op._order = 1


def test_mask_ownership_across_backward():
    x, _ = _cloud()
    op = FixedEndpointFilter(x, 1., (0., 0., 0.), (3, 3, 3))
    raw = (x+.02).requires_grad_()
    pins = torch.tensor([True, False, False, False, False, False])
    expected_pins = pins.clone()
    result = op.endpoint(raw, pins)
    pins.logical_not_()
    gradient, = torch.autograd.grad(result.sum(), raw)
    g = torch.ones_like(raw)
    g[expected_pins] = 0
    torch.testing.assert_close(gradient, op.apply_HT(g))


def test_pinned_reference_is_bitwise_preserved_including_signed_zero():
    x = torch.tensor([[-0., 0., .2], [.2, .3, .4]], dtype=torch.float32)
    op = FixedEndpointFilter(x, 1., (0., 0., 0.), (3, 3, 3))
    result = op.endpoint(x+.1, torch.tensor([True, False]))
    assert torch.equal(result[0].view(torch.int32), x[0].view(torch.int32))


def test_linearity_independent_objects_and_repeated_cpu_calls():
    x, m = _cloud()
    a = FixedEndpointFilter(x, 1., (0., 0., 0.), (3, 3, 3), m=m)
    b = FixedEndpointFilter(x.clone(), 1., (0., 0., 0.), (3, 3, 3), m=m.clone())
    d = torch.arange(42, dtype=x.dtype).reshape(6, 7)/31
    g = d.flip(0) + .1
    expected = a.apply_H(d)
    torch.testing.assert_close(a.apply_H(.3*d-.7*g), .3*expected-.7*a.apply_H(g),
                               atol=2e-14, rtol=2e-13)
    assert a._x0.data_ptr() != b._x0.data_ptr()
    assert a._weights[0].data_ptr() != b._weights[0].data_ptr()
    for _ in range(3):
        assert torch.equal(a.apply_H(d), expected)
        assert torch.equal(b.apply_H(d), expected)
    changed = FixedEndpointFilter(x+.7, 1., (0., 0., 0.), (3, 3, 3), m=m)
    assert not torch.allclose(changed.apply_H(d), expected)
    assert torch.equal(a.apply_H(d), expected)


def test_zero_mass_support_has_zero_map_and_derivative():
    x, _ = _cloud()
    op = FixedEndpointFilter(x, 1., (0., 0., 0.), (3, 3, 3), m=0.)
    raw = (x+.5).requires_grad_()
    assert torch.count_nonzero(op.apply_H(raw)) == 0
    assert torch.count_nonzero(op.apply_HT(raw)) == 0
    assert torch.equal(op.endpoint(raw), x)
    gradient, = torch.autograd.grad(op.endpoint(raw).sum(), raw)
    assert torch.count_nonzero(gradient) == 0


def test_validation_and_noncontiguous_fields():
    x, m = _cloud()
    op = FixedEndpointFilter(x, 1., (0., 0., 0.), (3, 3, 3), m=m)
    field = torch.arange(30, dtype=x.dtype).reshape(5, 6).T
    assert not field.is_contiguous()
    torch.testing.assert_close(op.apply_H(field), op.apply_H(field.contiguous()))
    for kwargs in ({'dx':0}, {'order':0}, {'dims':(3, -1, 3)}, {'m':-1},
                   {'grid_min':(0, float('nan'), 0)}):
        args = dict(dx=1., grid_min=(0.,0.,0.), dims=(3,3,3))
        args.update(kwargs)
        with pytest.raises(ValueError):
            FixedEndpointFilter(x, **args)
    with pytest.raises(ValueError):
        op.apply_H(field.float())
    with pytest.raises(ValueError):
        op.endpoint(field)
    with pytest.raises(ValueError):
        op.endpoint(x, torch.zeros(len(x), 1))
