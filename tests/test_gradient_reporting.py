import math

import pytest
import torch

from physmorph.pipeline.gradient_reporting import raw_direction_observations


def legacy_raw_observations(physics, render):
    """Original optimizer expressions, including scalar conversion locations."""
    def norm(gs):
        return float(torch.sqrt(sum(g.pow(2).sum() for g in gs)).item())
    return norm(physics), norm(render), float(sum((a*b).sum() for a, b in zip(physics, render)))


def gradient_case(dtypes, device='cpu'):
    physics, render = [], []
    for i, dtype in enumerate(dtypes):
        base = torch.linspace(-.51, .37+i*.03, 9+i*4, dtype=torch.float64)
        physics.append((base.sin()+.07).to(device=device, dtype=dtype).requires_grad_())
        render.append((base.cos()*(.8 if i % 2 else -.6)).to(device=device, dtype=dtype).requires_grad_())
    return physics, render


@pytest.mark.parametrize('dtypes', [
    (torch.float32,), (torch.float32,)*3, (torch.float64,)*3,
    (torch.float16, torch.bfloat16, torch.float64),
    (torch.float32, torch.float32, torch.float64)])
def test_raw_observations_match_legacy_without_mutating_or_attaching_graph(dtypes):
    physics, render = gradient_case(dtypes)
    before = [g.detach().clone() for g in physics+render]
    actual = raw_direction_observations(physics, render)
    assert actual == legacy_raw_observations(physics, render)
    assert all(type(value) is float for value in actual)
    assert all(torch.equal(g, old) for g, old in zip(physics+render, before))
    assert all(g.grad is None for g in physics+render)


def test_raw_dot_retains_ordered_leaf_accumulation():
    physics = [torch.tensor([v], dtype=torch.float32) for v in (1e8, 1., -1e8)]
    render = [torch.ones_like(g) for g in physics]
    actual = raw_direction_observations(physics, render)
    assert actual == legacy_raw_observations(physics, render)
    assert actual[2] == 0.
    assert float((physics[0]+physics[2]+physics[1]).sum()) == 1.


def test_norm_does_not_promote_earlier_leaves_before_accumulation():
    physics = [torch.tensor([1e4]), torch.tensor([1.]), torch.tensor([1.], dtype=torch.float64)]
    render = [torch.ones_like(g) for g in physics]
    actual = raw_direction_observations(physics, render)
    assert actual == legacy_raw_observations(physics, render)
    assert actual[0] == math.sqrt(100000001.)
    assert actual[0] != float(torch.sqrt(sum(g.double().pow(2).sum() for g in physics)))


def test_packing_retains_float64_observations():
    value = 1.+2.**-40
    physics = [torch.tensor([value], dtype=torch.float64)]
    render = [torch.tensor([1.], dtype=torch.float32)]
    actual = raw_direction_observations(physics, render)
    assert actual == legacy_raw_observations(physics, render)
    assert actual == (value, 1., value)


@pytest.mark.parametrize('value', [0., float('inf'), float('nan')])
def test_reporting_does_not_add_finite_guards_or_change_zero_semantics(value):
    physics, render = [torch.tensor([value])], [torch.tensor([1.])]
    actual, expected = raw_direction_observations(physics, render), legacy_raw_observations(physics, render)
    assert all(a == b or (math.isnan(a) and math.isnan(b)) for a, b in zip(actual, expected))
