"""PIC component identities, cancellation and read-only ownership on CPU."""
import json

import numpy as np
import pytest
import torch

from physmorph.mpm.endpoint_filter import FixedEndpointFilter
from scripts.probes.pic_component_metrics import NAMES, decompose_pic


def _case(dtype=torch.float64):
    x = torch.tensor([[.05, .2, .3], [.22, .15, .6], [.9, .05, .1],
                      [1.2, .1, .2], [.1, 1.4, .3]], dtype=dtype)
    mass = torch.tensor([.2, 2., .5, 3., 1.], dtype=dtype)
    op = FixedEndpointFilter(x, 1., (0., 0., 0.), (4, 4, 4), m=mass)
    pin = torch.tensor([False, True, False, False, False])
    layer = torch.tensor([1., 1., 0., 1., 0.], dtype=dtype)
    sums = torch.randn((4, len(x), 3), dtype=dtype, generator=torch.Generator().manual_seed(18)) * .07
    sums[:, pin] = 0
    raw = x + sums.sum(0)
    return op, x, raw, op.endpoint(raw, pin), pin, sums, layer


def _dense_h(start, mass):
    # Independent dense construction with explicitly clipped wall support.
    x, m = start.numpy(), mass.numpy()
    W = np.zeros((len(x), 4**3))
    def cubic(a):
        a = abs(a)
        return .5*a**3-a*a+2/3 if a < 1 else (2-a)**3/6 if a < 2 else 0.
    for p in range(len(x)):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    W[p, (i*4+j)*4+k] = cubic(i-x[p, 0])*cubic(j-x[p, 1])*cubic(k-x[p, 2])
    P = ((W / W.sum(1)[:, None]) / np.maximum(W.T @ m, 1e-12)[None]) @ (W.T*m[None])
    return torch.from_numpy(np.eye(len(x)) - np.linalg.matrix_power(np.eye(len(x))-P, 5))


def test_components_use_pinned_output_mask_after_nonsymmetric_partial_stencil_filter():
    args = _case()
    op, start, raw, promoted, pin, sums, layer = args
    H = _dense_h(start, torch.tensor([.2, 2., .5, 3., 1.], dtype=start.dtype))
    assert not torch.allclose(H, H.T)  # H^T or reweighting pins would be wrong.
    ji = torch.einsum('ij,cjk->cik', H-torch.eye(len(start)), sums)
    assert torch.count_nonzero(ji[:, pin]) > 0
    ji[:, pin] = 0
    result = decompose_pic(*args, .035)
    report, endpoints = result['report'], result['endpoints']
    assert report['valid'] and report['gates']['pins_exact']
    torch.testing.assert_close(endpoints['advection_only'], raw + ji[0], atol=1e-13, rtol=1e-12)
    torch.testing.assert_close(endpoints['preserve_relaxation'], raw + ji[:3].sum(0), atol=1e-13, rtol=1e-12)
    for endpoint in endpoints.values():
        assert torch.equal(endpoint[pin], start[pin])
    assert report['cohorts']['all_free']['particles'] == 4
    assert report['cohorts']['layer_free']['particles'] == 2
    for name, mask in (('all_free', ~pin), ('layer_free', ~pin & (layer >= .5))):
        row = report['cohorts'][name]
        independent = ji[:, mask]
        gram = torch.einsum('inc,jnc->ij', independent, independent) / int(mask.sum())
        torch.testing.assert_close(torch.tensor(row['gram_mean_wu2']), gram.float())
        actual_energy = (promoted[mask]-raw[mask]).square().sum(1).mean()
        assert row['summed_component_energy_wu2'] == pytest.approx(float(actual_energy), abs=1e-14)
        assert sum(row['components'][c]['signed_fraction'] for c in NAMES) == pytest.approx(1.)
        assert row['component_diagonal_energy_wu2'] + row['pair_cross_energy_wu2'] == pytest.approx(float(actual_energy))
    json.dumps(report, allow_nan=False)


def test_advection_and_direct_correction_cancellation_can_make_candidate_jump_larger():
    op, start, _, _, pin, sums, layer = _case()
    sums[1:3] = 0
    sums[3] = -sums[0]
    raw = start.clone()
    result = decompose_pic(op, start, raw, op.endpoint(raw, pin), pin, sums, layer, .035)
    row = result['report']['cohorts']['all_free']
    assert result['report']['valid']
    assert row['actual_jump']['rms_wu'] == 0
    assert row['candidates']['advection_only']['rms_wu'] > .001
    assert not row['actual_jump_above_numerical_floor']
    assert all(row['components'][c]['signed_fraction'] is None for c in NAMES)
    assert row['advection_residual']['advection_to_actual_rms'] is None
    assert row['advection_residual']['cancellation_fraction'] == pytest.approx(-1.)
    assert row['advection_residual']['cross_energy_wu2'] < 0
    torch.testing.assert_close(result['endpoints']['preserve_relaxation'], result['endpoints']['advection_only'])


def test_large_pure_relaxation_is_preserved_but_is_not_a_quality_claim():
    op, start, _, _, pin, sums, layer = _case()
    sums[:3] = 0
    sums[3] *= 5
    raw = start + sums.sum(0)
    result = decompose_pic(op, start, raw, op.endpoint(raw, pin), pin, sums, layer, .035)
    assert result['report']['valid']
    row = result['report']['cohorts']['all_free']
    assert row['components']['layer_residual']['signed_fraction'] == pytest.approx(1.)
    assert row['actual_jump']['rms_sp'] > 1
    assert row['advection_residual']['advection_energy_wu2'] == 0
    assert row['candidates']['preserve_relaxation']['change_exceeds_10x_numerical_context']
    torch.testing.assert_close(result['endpoints']['advection_only'], raw)
    torch.testing.assert_close(result['endpoints']['preserve_relaxation'], raw)


def test_roundoff_repeat_context_and_nearzero_denominators_are_explicit():
    op, start, _, _, pin, sums, layer = _case()
    sums *= 1e-14
    raw = start + sums.sum(0)
    promoted = op.endpoint(raw, pin)
    class RepeatedOperator:
        x0 = start.clone()
        shapes = []
        def apply_H(self, value):
            self.shapes.append(tuple(value.shape))
            out = op.apply_H(value)
            if len(self.shapes) > 1:
                out = out + (1e-12 if len(self.shapes) == 2 else -1e-12)
            return out
    observed = RepeatedOperator()
    report = decompose_pic(observed, start, raw, promoted, pin, sums, layer, .035)['report']
    assert observed.shapes == [(len(start), 12), (len(start), 3), (len(start), 3)]
    assert report['valid']
    row = report['cohorts']['all_free']
    assert row['numerical_context']['repeat_error']['rms_wu'] > 1e-12
    assert row['numerical_context']['original_operator_error']['rms_wu'] > 1e-12
    assert not row['actual_jump_above_numerical_floor']
    assert all(row['components'][c]['signed_fraction'] is None for c in NAMES)


def test_bad_component_or_owned_map_closure_is_reported_as_invalid():
    args = list(_case())
    args[5] = args[5].clone(); args[5][0, 0, 0] += .01
    report = decompose_pic(*args, .035)['report']
    assert not report['valid']
    assert not report['gates']['component_sum_equals_displacement']['passed']
    args = list(_case())
    args[3] = args[3].clone(); args[3][0, 0] += .01
    report = decompose_pic(*args, .035)['report']
    assert not report['valid']
    assert not report['gates']['original_operator_equals_owned_endpoint']['passed']
    assert report['cohorts']['all_free']['candidates']['advection_only']['interpretation'] == 'closure_failed'
    args[2] = args[2].clone(); args[2][args[4]] += .1
    with pytest.raises(ValueError, match='raw endpoint changed'):
        decompose_pic(*args, .035)


def test_owned_endpoints_no_host_array_numerics_and_empty_cohort(monkeypatch):
    args = _case(dtype=torch.float32)
    before = [value.clone() if torch.is_tensor(value) else None for value in args]
    def forbidden(*_, **__):
        raise AssertionError('numeric host fallback')
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, 'cpu', forbidden)
        patch.setattr(torch.Tensor, 'numpy', forbidden)
        result = decompose_pic(*args, .035)
    assert result['report']['valid']
    snapshots = {key: value.clone() for key, value in result['endpoints'].items()}
    for actual, reference in zip(args, before):
        if reference is not None:
            assert torch.equal(actual, reference)
            actual.zero_()
    for key, reference in snapshots.items():
        assert torch.equal(result['endpoints'][key], reference)
    result['endpoints']['raw'].zero_()
    assert torch.equal(result['endpoints']['current'], snapshots['current'])
    op, start, _, _, pin, sums, layer = _case()
    pin[:] = True; sums.zero_()
    report = decompose_pic(op, start, start.clone(), start.clone(), pin, sums, layer, .035)['report']
    assert report['valid']
    assert all(row['particles'] == 0 for row in report['cohorts'].values())
