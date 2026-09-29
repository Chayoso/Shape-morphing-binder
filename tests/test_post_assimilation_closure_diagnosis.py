"""CPU arithmetic gate for the opt-in FP64 diagnostic, not an MPM rollout."""
from types import SimpleNamespace

import pytest
import torch

from physmorph.plasticity.assimilation_adjoint import assimilate_elastic_differentiable
from scripts.probes.post_assimilation_closure_diagnosis import fp64_handoff


@pytest.mark.parametrize('iso', [False, True])
@pytest.mark.parametrize('settle', [False, True])
@pytest.mark.parametrize('eta', [0., .5])
def test_double_counterfactual_matches_separate_ad_operator(iso, settle, eta):
    F = torch.diag_embed(torch.tensor([[1.2, .7, 1.1], [12., 4., 1/48],
        [.02, 2., 9.], [1., 1., 1.], [2., 2., .6], [-1., 1., 1.]], dtype=torch.float32))
    F[0, 0, 1] = .17
    F[4, 1, 2] = -.12
    P = torch.eye(3).repeat(len(F), 1, 1)
    P[0] = torch.tensor([[1.1, .13, .03], [0., .91, .07], [.02, 0., 1.04]])
    old = torch.tensor([True, False, False, False, False, False])
    new = torch.tensor([False, True, False, True, False, False])
    cfg = SimpleNamespace(assim=eta, assim_iso=iso, assim_smin=.2,
                          assim_smax=5., settle_pin_assim=settle)
    F_before, P_before = F.clone(), P.clone()
    actual = fp64_handoff(F, P, old, new, cfg)
    expected = assimilate_elastic_differentiable(F.double(), P.double(), eta=eta,
        isochoric=iso, smin=.2, smax=5.).float()
    if settle:
        expected[old] = P[old]
        expected[new] = assimilate_elastic_differentiable(F[new].double(),
            expected[new].double(), eta=1., isochoric=False, smin=.2, smax=5.).float()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    assert torch.equal(F, F_before) and torch.equal(P, P_before)
    if settle:
        assert torch.equal(actual[old], P[old])


def test_double_counterfactual_keeps_clamp_then_iso_projection():
    F = torch.diag(torch.tensor([12., 4., 1/48], dtype=torch.float64))[None]
    P = torch.eye(3, dtype=torch.float64)[None]
    pins = torch.zeros(1, dtype=torch.bool)
    cfg = SimpleNamespace(assim=1., assim_iso=True, assim_smin=.2,
                          assim_smax=5., settle_pin_assim=True)
    actual = fp64_handoff(F, P, pins, pins, cfg)
    expected = torch.diag(torch.tensor([2.5, 2., .2], dtype=torch.float32))[None]
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)


def test_double_counterfactual_rejects_overlapping_pin_masks():
    F = torch.eye(3)[None]
    pins = torch.ones(1, dtype=torch.bool)
    with pytest.raises(ValueError, match='pin masks'):
        fp64_handoff(F, F, pins, pins, SimpleNamespace())


def test_per_call_store_is_observably_different_from_one_final_cast():
    from physmorph.plasticity.assimilation_adjoint import assimilate_handoff
    generator = torch.Generator().manual_seed(17)
    F = torch.eye(3)[None] + .2*torch.randn(5, 3, 3, generator=generator)
    P = torch.eye(3)[None] + .1*torch.randn(5, 3, 3, generator=generator)
    old, new = torch.zeros(5, dtype=torch.bool), torch.ones(5, dtype=torch.bool)
    cfg = SimpleNamespace(assim=.5, assim_iso=True, assim_smin=.2,
                          assim_smax=5., settle_pin_assim=True)
    actual = fp64_handoff(F, P, old, new, cfg)
    once = assimilate_handoff(F.double(), P.double(), old, new, eta=.5,
                              isochoric=True, smin=.2, smax=5.).float()
    assert not torch.equal(actual, once)
