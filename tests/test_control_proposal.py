"""Independent dense update oracle, noncommuting operators and in-place ownership."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from physmorph.pipeline.control_proposal import apply_proposal, required_decrease


def test_mixed_proposal_matches_dense_oracle_and_preserves_storage():
    rng = np.random.default_rng(392)
    shapes = [(2, 3, 3, 3), (2, 3), (3, 6), (4,)]
    initial = [rng.normal(size=shape)*.25 for shape in shapes]
    initial[2][0] *= 10  # Exercise shared displacement/braking row-ball bound.
    gradient = [rng.normal(size=shape) for shape in shapes]
    old_m = [rng.normal(size=shape)*.3 for shape in shapes]
    old_v = [rng.uniform(.1, 1., size=shape) for shape in shapes]
    leaves = [torch.tensor(value, requires_grad=True) for value in initial]
    gradients = [torch.tensor(value) for value in gradient]
    mom, vel = ([torch.tensor(value) for value in values] for values in (old_m, old_v))
    pointers = [tensor.data_ptr() for group in (leaves, mom, vel) for tensor in group]
    control_scale = np.array([.2, .7, 1.]).reshape(1, 3, 1, 1)
    body_scale = np.tile([.2, .2, .2, 1., 1., 1.], (3, 1))
    projection = np.array([[.5, .5, 0., 0.], [0., .25, .75, 0.],
                           [0., 0., .5, .5], [.4, 0., 0., .6]])
    bounds = np.array([.7, .08, .04, .7])  # Keep two coordinates free to expose operator order.
    beta1, beta2, step, eps, alpha = .5, .25, 3, .07, .4
    scales = [1., .3, .8, 1.7]
    # Analytic bias-corrected Adam at this single time, then independent dense maps.
    new_m = [beta1*m+(1-beta1)*g for m, g in zip(old_m, gradient)]
    new_v = [beta2*v+(1-beta2)*g*g for v, g in zip(old_v, gradient)]
    updates = [m/(1-beta1**step)/(np.sqrt(v/(1-beta2**step))+eps)
               for m, v in zip(new_m, new_v)]
    updates[0] *= control_scale
    updates[2] *= body_scale
    updates[3] = projection@updates[3]
    unbounded = [x-alpha*scale*delta for x, scale, delta in zip(initial, scales, updates)]
    expected = deepcopy(unbounded)
    # Euclidean projections of the resulting coordinates, independently in NumPy.
    for t in range(2):
        for p in range(3):
            radius = np.linalg.norm(expected[0][t, p])
            if radius > .12:
                expected[0][t, p] *= .12/radius
    expected[1] = np.clip(expected[1], -.15, .15)
    for i in range(3):
        radius = np.linalg.norm(expected[2][i])
        if radius > 1:
            expected[2][i] /= radius
    expected[3] = np.clip(expected[3], -bounds, bounds)
    projection_inputs = []
    def project(value):
        projection_inputs.append(value.clone())
        return torch.tensor(projection)@value
    result = apply_proposal(leaves, gradients, mom, vel, step=step, alpha=alpha,
        lr_scale=scales, beta1=beta1, beta2=beta2, eps=eps, material_index=1,
        body_index=2, surface_index=3, ctrl_scale=torch.tensor(control_scale),
        body_scale=torch.tensor(body_scale), surface_project=project, dfc_clip=.12,
        material_clip=.15, surface_bound=torch.tensor(bounds))
    assert result is None
    for got, wanted in zip(leaves, expected):
        np.testing.assert_allclose(got.detach().numpy(), wanted, rtol=2e-14, atol=2e-14)
    for got, wanted in zip(mom+vel, new_m+new_v):
        np.testing.assert_allclose(got.numpy(), wanted, rtol=2e-14, atol=2e-14)
    assert pointers == [tensor.data_ptr() for group in (leaves, mom, vel) for tensor in group]
    assert len(projection_inputs) == 1
    np.testing.assert_allclose(projection_inputs[0].numpy(),
        new_m[3]/(1-beta1**step)/(np.sqrt(new_v[3]/(1-beta2**step))+eps), rtol=2e-14, atol=2e-14)
    assert all(value.grad is None and value.is_leaf for value in leaves)
    for got, wanted in zip(gradients, gradient):
        np.testing.assert_array_equal(got.numpy(), wanted)
    # Projection of parameters would be a different algorithm.
    wrong_u = np.clip(projection@(initial[3]-alpha*scales[3]*(
        new_m[3]/(1-beta1**step)/(np.sqrt(new_v[3]/(1-beta2**step))+eps))), -bounds, bounds)
    assert not np.allclose(wrong_u, expected[3])


def test_scaling_applies_after_adam_and_leaves_terminal_braking_unscaled():
    stress = torch.zeros(1, 1, 3, 3)
    body = torch.zeros(1, 6)
    leaves = [stress, body]
    gradients = [torch.full_like(stress, 2.), torch.full_like(body, 2.)]
    mom, vel = [[torch.zeros_like(x) for x in leaves] for _ in range(2)]
    apply_proposal(leaves, gradients, mom, vel, step=1, alpha=.1, lr_scale=[1., 1.],
        beta1=.5, beta2=.25, eps=0., body_index=1,
        ctrl_scale=torch.tensor(.25), body_scale=torch.tensor([[.25]*3+[1.]*3]))
    torch.testing.assert_close(stress, torch.full_like(stress, -.025), rtol=0, atol=1e-8)
    torch.testing.assert_close(body, torch.tensor([[-.025]*3+[-.1]*3]), rtol=0, atol=1e-8)
    # Scaling gradients before moment normalization would incorrectly cancel out.
    assert float(stress.flatten()[0]) != pytest.approx(-.1)
    assert torch.equal(mom[0], torch.ones_like(stress))


def test_diagnostic_clones_do_not_change_live_parameters_or_moments():
    live = [torch.full((1, 2, 3, 3), .02)]
    first, second = [torch.ones_like(live[0])], [torch.ones_like(live[0])]
    before = [x.clone() for x in live+first+second]
    copies = [[x.clone() for x in group] for group in (live, first, second)]
    apply_proposal(copies[0], [torch.ones_like(live[0])], copies[1], copies[2],
        step=7, alpha=.1, lr_scale=[1.], beta1=.9, beta2=.99, eps=1e-6)
    assert not torch.equal(copies[0][0], live[0])
    for actual, saved in zip(live+first+second, before):
        assert torch.equal(actual, saved)


def test_nonstandard_role_order_and_scalar_surface_bound():
    leaves = [torch.tensor([.4, -.4]), torch.ones(1, 1, 3, 3), torch.tensor([[2., 0., 0.]])]
    mom, vel = [[torch.zeros_like(x) for x in leaves] for _ in range(2)]
    apply_proposal(leaves, [torch.zeros_like(x) for x in leaves], mom, vel,
        step=1, alpha=.1, lr_scale=[1.]*3, beta1=.9, beta2=.99, eps=1e-8,
        stress_index=1, surface_index=0, body_index=2,
        surface_bound=torch.tensor(.1), dfc_clip=.3)
    torch.testing.assert_close(leaves[0], torch.tensor([.1, -.1]), rtol=0, atol=0)
    torch.testing.assert_close(leaves[1], torch.full_like(leaves[1], .1), rtol=0, atol=1e-8)
    torch.testing.assert_close(leaves[2], torch.tensor([[1., 0., 0.]]), rtol=0, atol=0)


@pytest.mark.parametrize('kwargs', [dict(surface_index=1), dict(body_index=0), dict(material_index=9)])
def test_invalid_role_contract_fails_before_mutation(kwargs):
    leaves = [torch.ones(1, 1, 3, 3), torch.ones(2)]
    mom, vel = [[torch.ones_like(x) for x in leaves] for _ in range(2)]
    before = [x.clone() for x in leaves+mom+vel]
    with pytest.raises(ValueError):
        apply_proposal(leaves, [torch.ones_like(x) for x in leaves], mom, vel,
            step=1, alpha=.1, lr_scale=[1., 1.], beta1=.9, beta2=.99, eps=1e-8, **kwargs)
    assert all(torch.equal(a, b) for a, b in zip(leaves+mom+vel, before))


@pytest.mark.parametrize('current,predicted,ratio,noise,armijo,expected', [
    (2., .3, 10., .01, .1, .03),  # positive model slope dominates
    (2., .01, 10., .01, .1, .02),  # noise dominates
    (-2., 0., 10., .01, .1, .02),
    (-2., -100., 10., .01, .1, .02),  # stale moment: never negative allowance
    (.001, .01, .02, .01, .1, .5),  # density unit floor, not an arbitrary 1
])
def test_required_decrease_preserves_armijo_and_loss_unit_noise(current, predicted, ratio, noise, armijo, expected):
    assert required_decrease(current, predicted, ratio, noise, armijo) == pytest.approx(expected)
