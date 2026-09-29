"""Independent forward and directional checks for the actual assimilation map."""
import numpy as np
import pytest
import torch

from physmorph.plasticity.assimilation import assimilate_elastic
from physmorph.plasticity.assimilation_adjoint import (
    assimilate_elastic_differentiable as assimilate, assimilate_handoff)


def _identity(n=1, dtype=torch.float64):
    return torch.eye(3, dtype=dtype).repeat(n, 1, 1)


def _rotation(angle=.43):
    c, s = np.cos(angle), np.sin(angle)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float64)


def _inputs():
    r, q = _rotation(), _rotation(-.72)
    F = torch.stack([r @ torch.diag(torch.tensor(v, dtype=torch.float64)) @ q
                     for v in [(1.3, .82, 1.05), (2., 2., .5), (12., 4., 1/48), (.2, .4, 1.2)]])
    P = _identity(4)
    P[0] = torch.tensor([[1.1, .15, .03], [0, .91, .08], [.05, 0, 1.04]])
    return F, P


def _fd(function, F, P, *, step=2e-6, atol=2e-7, rtol=2e-6, seed=5):
    gen = torch.Generator().manual_seed(seed)
    dF = torch.randn(F.shape, generator=gen, dtype=F.dtype)
    dP = torch.randn(P.shape, generator=gen, dtype=P.dtype)
    weight = torch.randn(F.shape, generator=gen, dtype=F.dtype)
    F, P = F.clone().requires_grad_(), P.clone().requires_grad_()
    value = function(F, P)
    grads = torch.autograd.grad((value * weight).sum(), (F, P), allow_unused=True)
    analytic = sum((g * d).sum() for g, d in zip(grads, (dF, dP)) if g is not None)
    with torch.no_grad():
        measured = ((function(F + step*dF, P + step*dP) - function(F - step*dF, P - step*dP)) * weight).sum() / (2*step)
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    torch.testing.assert_close(analytic, measured, atol=atol, rtol=rtol)


@pytest.mark.parametrize('iso', [False, True])
@pytest.mark.parametrize('eta', [0., .35, 1.])
def test_cpu_forward_matches_existing(iso, eta):
    F, P = (v.float() for v in _inputs())
    actual = assimilate(F, P, eta=eta, isochoric=iso)
    expected = assimilate_elastic(F.numpy(), P.numpy(), eta=eta, isochoric=iso)
    torch.testing.assert_close(actual, torch.from_numpy(expected), atol=4e-6, rtol=4e-6)


@pytest.mark.parametrize('iso', [False, True])
@pytest.mark.parametrize('case', ['identity', 'isotropic', 'rigid', 'twofold', 'noncommuting'])
def test_repeated_and_general_spectra_directional(iso, case):
    F, P = _identity(), _identity()
    if case == 'isotropic':
        F *= 1.7
        P *= .87
    elif case == 'rigid':
        F[0] = _rotation()
        P[0] = _rotation(-.8)
    elif case == 'twofold':
        F[0] = _rotation() @ torch.diag(torch.tensor([2., 2., .7], dtype=F.dtype)) @ _rotation(-.7)
    elif case == 'noncommuting':
        F, P = (v[:1] for v in _inputs())
    _fd(lambda f, p: assimilate(f, p, eta=.37, isochoric=iso), F, P)


@pytest.mark.parametrize('iso', [False, True])
def test_identity_full_linear_oracle_including_fp_rotation(iso):
    F, P = _identity().requires_grad_(), _identity().requires_grad_()
    w = torch.tensor([[[.2, .7, -.5], [-.3, 1.2, .4], [.8, -.6, .1]]], dtype=F.dtype)
    dF, dP = torch.autograd.grad((assimilate(F, P, eta=.37, isochoric=iso) * w).sum(), (F, P))
    symmetric = (w + w.transpose(1, 2))/2
    if iso:
        symmetric = symmetric - torch.diag_embed(symmetric.diagonal(dim1=1, dim2=2).sum(1)[:, None].expand(-1, 3)/3)
    skew = (w - w.transpose(1, 2))/2
    torch.testing.assert_close(dF, .37*symmetric, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(dP, .63*symmetric+skew, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('iso', [False, True])
def test_active_band_noncommuting_directional(iso):
    F, P = (v[2:3] for v in _inputs())
    _fd(lambda f, p: assimilate(f, p, eta=1., isochoric=iso), F, P, step=1e-7, atol=2e-6)


def test_iso_first_clamp_still_blocks_saturated_direction():
    F = torch.diag(torch.tensor([12., 4., 1/48], dtype=torch.float64))[None].requires_grad_()
    out = assimilate(F, _identity(), eta=1., isochoric=True)
    torch.testing.assert_close(out, torch.diag(torch.tensor([2.5, 2., .2], dtype=F.dtype))[None], atol=1e-12, rtol=1e-12)
    # Inspect the cumulative map in isolation: its initial clamp stays relevant
    # even though the final projected first axis is interior (2.5 < 5).
    from physmorph.plasticity.assimilation_adjoint import _CumulativeBand
    z = _CumulativeBand.apply(F, .2, 5., True)
    grad, = torch.autograd.grad(z[0, 0, 0], F)
    assert grad[0, 0, 0] == 0
    assert abs(grad[0, 1, 1]) > .1


@pytest.mark.parametrize('iso', [False, True])
def test_invalid_elastic_increment_keeps_cumulative_projection(iso):
    P = torch.diag(torch.tensor([7., 1.3, .1], dtype=torch.float64))[None]
    F = torch.diag(torch.tensor([-1.2, 1.1, .9], dtype=torch.float64))[None] @ P
    _fd(lambda f, p: assimilate(f, p, eta=.4, isochoric=iso), F, P, atol=2e-6)
    F.requires_grad_()
    dF, = torch.autograd.grad(assimilate(F, P, isochoric=iso).sum(), F)
    assert torch.equal(dF, torch.zeros_like(F))


def test_floor_and_singular_skipped_row_have_finite_gradients():
    # floor active, det=8e-6 > health gate; and a separate skipped singular row.
    F = torch.stack([torch.diag(torch.tensor([4., 4., 5e-7], dtype=torch.float64)),
                     torch.zeros(3, 3, dtype=torch.float64)])
    _fd(lambda f, p: assimilate(f, p, eta=.4, isochoric=True), F[:1], _identity(), step=1e-9, atol=5e-5)
    F.requires_grad_()
    P = _identity(2).requires_grad_()
    grads = torch.autograd.grad(assimilate(F, P, isochoric=True).sum(), (F, P))
    assert all(torch.isfinite(g).all() for g in grads)
    assert torch.equal(grads[0][1], torch.zeros_like(F[1]))


def test_eta_zero_bypasses_projection():
    F = _identity().requires_grad_()
    P = (7*_identity()).requires_grad_()
    value = assimilate(F, P, eta=0, isochoric=True)
    assert torch.equal(value, P)
    gF, gP = torch.autograd.grad(value.sum(), (F, P), allow_unused=True)
    assert gF is None
    assert torch.equal(gP, torch.ones_like(P))


@pytest.mark.parametrize('preserve', [False, True])
@pytest.mark.parametrize('eta', [0., .4])
def test_pin_composition_matches_runner_calls_and_directional(preserve, eta):
    F, P = _inputs()
    old, new = torch.tensor([True, False, False, False]), torch.tensor([False, True, True, False])
    fn = lambda f, p: assimilate_handoff(f, p, old, new, eta=eta, isochoric=True, settle_pin_assim=preserve)
    first = assimilate(F, P, eta=eta, isochoric=True)
    first = torch.where(old[:, None, None], P, first) if preserve else first
    second = assimilate(F, first, eta=1., isochoric=False)
    expected = torch.where(new[:, None, None], second, first) if preserve else first
    torch.testing.assert_close(fn(F, P), expected, atol=0, rtol=0)
    _fd(fn, F, P, step=1e-7, atol=3e-6)


def test_repeated_backward_and_missing_inputs():
    F, P = (v[:1].clone().requires_grad_() for v in _inputs())
    out = assimilate(F, P, isochoric=True)
    first = torch.autograd.grad(out.sum(), (F, P), retain_graph=True)
    again = torch.autograd.grad(out.sum(), (F, P))
    for a, b in zip(first, again):
        assert torch.equal(a, b)
    assert torch.isfinite(torch.autograd.grad(assimilate(F.detach(), P).sum(), P)[0]).all()


def test_invalid_metadata_and_overlap_rejected():
    with pytest.raises(ValueError, match='matching'):
        assimilate(_identity().float(), _identity())
    with pytest.raises(ValueError, match='finite'):
        assimilate(_identity(), _identity(), eta=float('nan'))
    with pytest.raises(RuntimeError, match='disjoint'):
        assimilate_handoff(_identity(), _identity(), torch.tensor([True]), torch.tensor([True]))


@pytest.mark.parametrize('spectrum,lo,hi,iso,step', [
    ((12., .03, .02), .2, 5., True, 1e-7),
    ((12., 12., .03), .2, 5., True, 1e-7),
    ((12., 12., .03), .2, 5., False, 1e-7),
    ((4., 3., 2.5), 2., 5., True, 1e-8),
    ((.4, .3, .25), .1, .5, True, 1e-8),
], ids=['one_free', 'repeated_capped_iso', 'repeated_capped_rotation',
        'target_clipped_above_one', 'target_clipped_below_one'])
def test_cumulative_active_sets_and_clipped_target_directional(spectrum, lo, hi, iso, step):
    """Rotated spectral cases test the complete matrix map, including its skew part.

    Bands wholly above/below one also exercise the production target clipping;
    these are forward-contract tests, not proposed material parameters.
    """
    from physmorph.plasticity.assimilation_adjoint import _CumulativeBand
    gen = torch.Generator().manual_seed(987)
    left, _ = torch.linalg.qr(torch.randn(3, 3, generator=gen, dtype=torch.float64))
    right, _ = torch.linalg.qr(torch.randn(3, 3, generator=gen, dtype=torch.float64))
    value = (left @ torch.diag(torch.tensor(spectrum, dtype=torch.float64)) @ right)[None]
    _fd(lambda f, p: _CumulativeBand.apply(f, lo, hi, iso), value, _identity(),
        step=step, atol=3e-6, rtol=1e-5)


def test_repeated_floor_values_with_healthy_elastic_determinant():
    """Two floored singular values need a finite, basis-independent pullback."""
    gen = torch.Generator().manual_seed(987)
    left, _ = torch.linalg.qr(torch.randn(3, 3, generator=gen, dtype=torch.float64))
    right, _ = torch.linalg.qr(torch.randn(3, 3, generator=gen, dtype=torch.float64))
    F = (left @ torch.diag(torch.tensor([20., 5e-4, 5e-4], dtype=torch.float64)) @ right)[None]
    assert torch.linalg.det(F).item() > 1e-6
    _fd(lambda f, p: assimilate(f, p, eta=.4, isochoric=True), F, _identity(),
        step=1e-9, atol=3e-5, rtol=1e-5)
