"""World-space geometry/population oracles; no renderer or simulation."""
import json
import math

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from physmorph.render.footprint_diagnostics import summarize_world_footprints


def rotation(dtype=torch.float64):
    # Fixed proper rotations keep the oracle independent of the helper's basis.
    a, b = .63, -.41
    rz = torch.tensor([[math.cos(a), -math.sin(a), 0.],
                       [math.sin(a), math.cos(a), 0.], [0., 0., 1.]], dtype=dtype)
    ry = torch.tensor([[math.cos(b), 0., math.sin(b)], [0., 1., 0.],
                       [-math.sin(b), 0., math.cos(b)]], dtype=dtype)
    return rz @ ry


def observe(covariance, normals=None, **kwargs):
    covariance = torch.as_tensor(covariance)
    if covariance.ndim == 2:
        covariance = covariance[None]
    if normals is None:
        normals = covariance.new_tensor([0., 0., 1.]).expand(len(covariance), -1)
    for key in ('support', 'opacity'):
        kwargs.setdefault(key, covariance.new_ones(len(covariance)))
    return summarize_world_footprints(covariance, normals, **kwargs)


def mean(report, name, population='all'):
    return report['populations'][population][name]['mean']


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_isotropic_material_covariance_scales_by_F_singular_values(dtype):
    u, v = rotation(dtype), rotation(dtype).T
    stretches = torch.tensor([3., 1.25, .2], dtype=dtype)
    deformation = u @ torch.diag(stretches) @ v.T
    spacing = .07
    covariance = spacing**2 * deformation @ deformation.T
    report = observe(covariance, u[:, 2][None])
    tolerance = 3e-6 if dtype == torch.float32 else 1e-13
    assert report['valid']
    assert mean(report, 'normal_std_wu') == pytest.approx(spacing*.2, rel=tolerance)
    assert mean(report, 'tangent_minor_std_wu') == pytest.approx(spacing*1.25, rel=tolerance)
    assert mean(report, 'tangent_major_std_wu') == pytest.approx(spacing*3., rel=tolerance)
    assert mean(report, 'tangent_anisotropy') == pytest.approx(3./1.25, rel=tolerance)
    assert mean(report, 'normal_tangent_covariance_wu2') < tolerance*spacing**2


def test_rotated_anisotropic_surface_covariance_and_normal_thickness_only_variant():
    r = rotation()
    sigma, reference = .14, .035
    for thickness in (sigma/4, reference/4):
        covariance = r @ torch.diag(torch.tensor([sigma**2, (.6*sigma)**2, thickness**2], dtype=r.dtype)) @ r.T
        report = observe(covariance, r[:, 2][None])
        assert report['valid']
        assert mean(report, 'normal_std_wu') == pytest.approx(thickness, rel=1e-7)
        assert mean(report, 'tangent_minor_std_wu') == pytest.approx(.6*sigma, rel=1e-7)
        assert mean(report, 'tangent_major_std_wu') == pytest.approx(sigma, rel=1e-7)
        assert mean(report, 'normal_tangent_covariance_wu2') < 1e-17


def test_tilted_ellipsoid_plane_oracle_and_coupling_are_distinct_from_full_eigenvalues():
    r = rotation()
    covariance = r @ torch.diag(torch.tensor([.09, .016, .001], dtype=torch.float64)) @ r.T
    n = torch.tensor([0., 0., 1.], dtype=torch.float64)
    report = observe(covariance, n[None])
    # e1/e2 span the supplied normal's plane, independent of helper basis choice.
    expected_tangent = torch.linalg.eigvalsh(covariance[:2, :2]).sqrt()
    coupling = torch.linalg.vector_norm(covariance[:2, 2]).item()
    assert report['valid']
    assert mean(report, 'normal_std_wu') == pytest.approx(covariance[2, 2].sqrt().item(), rel=1e-13)
    assert mean(report, 'tangent_minor_std_wu') == pytest.approx(expected_tangent[0].item(), rel=1e-13)
    assert mean(report, 'tangent_major_std_wu') == pytest.approx(expected_tangent[1].item(), rel=1e-13)
    assert mean(report, 'normal_tangent_covariance_wu2') == pytest.approx(coupling, rel=1e-13)
    assert coupling > .01
    assert abs(mean(report, 'tangent_major_std_wu')-.3) > .005


def test_population_membership_and_nonvisibility_are_explicit():
    radii = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
    covariance = radii[:, None, None].square()*torch.eye(3, dtype=torch.float64)
    report = observe(covariance, support=torch.tensor([1., 0., .5, 1.]),
                     opacity=torch.tensor([1., 1., 0., .25]),
                     populations={'active_free': torch.tensor([True, False, False, True]),
                                  'empty': torch.zeros(4, dtype=torch.bool)})
    rows = report['populations']
    assert rows['all']['selected_count'] == rows['all']['valid_count'] == 4
    assert rows['all']['zero_support_count'] == rows['all']['zero_opacity_count'] == 1
    assert rows['positive_support_and_opacity']['selected_count'] == 2
    assert rows['zero_opacity']['selected_count'] == 1
    assert mean(report, 'normal_std_wu') == 2.5
    assert rows['all']['normal_std_wu']['rms'] == pytest.approx(math.sqrt(7.5))
    assert rows['active_free']['normal_std_wu'] == rows['positive_support_and_opacity']['normal_std_wu']
    assert rows['empty']['normal_std_wu'] is None
    assert rows['empty']['tangent_anisotropy'] is None
    assert 'not camera visibility' in report['definitions']['populations']
    json.dumps(report, allow_nan=False)


def test_empty_cloud_and_zero_or_rank_deficient_covariance_have_no_fabricated_ratio():
    empty = observe(torch.empty((0, 3, 3), dtype=torch.float64))
    assert empty['valid'] and empty['input_count'] == 0
    assert empty['populations']['all']['normal_std_wu'] is None
    covariance = torch.stack((torch.diag(torch.tensor([4., 0., 0.])), torch.zeros((3, 3))))
    report = observe(covariance)
    assert report['valid']
    row = report['populations']['all']
    assert row['anisotropy_unbounded_count'] == row['anisotropy_undefined_count'] == 1
    assert row['tangent_anisotropy'] is None
    assert row['normal_std_wu']['maximum'] == 0.
    for item in (empty, report):
        json.dumps(item, allow_nan=False)


@pytest.mark.parametrize('bad,category', [
    ([[1., 0., 2.], [0., 1., 0.], [2., 0., 1.]], 'non_psd_covariance'),
    ([[-.01, 0., 0.], [0., 1., 0.], [0., 0., 1.]], 'non_psd_covariance'),
    ([[1., .1, 0.], [0., 1., 0.], [0., 0., 1.]], 'asymmetric_covariance'),
    ([[1., 0., 0.], [0., float('nan'), 0.], [0., 0., 1.]], 'nonfinite_covariance'),
])
def test_invalid_covariance_is_reported_and_excluded_without_repair(bad, category):
    covariance = torch.stack((torch.eye(3), torch.tensor(bad))).double()
    original = covariance.clone()
    report = observe(covariance)
    assert not report['valid'] and report['invalid_count'] == 1
    assert report['validation_counts'][category] == 1
    assert report['populations']['all']['normal_std_wu']['count'] == 1
    assert mean(report, 'normal_std_wu') == 1.
    torch.testing.assert_close(covariance, original, equal_nan=True, rtol=0, atol=0)
    json.dumps(report, allow_nan=False)


def test_numerical_input_validity_excludes_the_whole_required_row():
    covariance = torch.eye(3, dtype=torch.float64).repeat(6, 1, 1)
    normals = torch.tensor([[0., 0., 1.], [0., 0., 2.], [0., 0., float('inf')],
                            [0., 0., 1.], [0., 0., 1.], [0., 0., 1.]])
    report = observe(covariance, normals, support=torch.tensor([1., 1., 1., -1., 1., 1.]),
                     opacity=torch.tensor([1., 1., 1., 1., 1.1, float('nan')]))
    assert report['invalid_count'] == 5
    assert report['validation_counts']['nonfinite_normal'] == 1
    assert report['validation_counts']['invalid_support'] == 1
    assert report['validation_counts']['invalid_opacity'] == 2
    assert report['populations']['all']['valid_count'] == 1
    assert mean(report, 'normal_std_wu') == 1.
    json.dumps(report, allow_nan=False)


def test_only_recorded_roundoff_sized_negative_tangent_variance_is_clamped():
    delta = 8*torch.finfo(torch.float64).eps
    covariance = torch.tensor([[1., 1., 0.], [1., 1.-delta, 0.], [0., 0., 1.]], dtype=torch.float64)
    report = observe(covariance)
    assert report['valid'] and report['roundoff_clamped_rows'] == 1
    assert mean(report, 'tangent_minor_std_wu') == 0.
    assert report['numerics']['arithmetic_error_factor'] == 64*torch.finfo(torch.float64).eps
    covariance[1, 1] = 1.-1e-6
    invalid = observe(covariance)
    assert not invalid['valid'] and invalid['roundoff_clamped_rows'] == 0
    assert invalid['populations']['all']['tangent_minor_std_wu'] is None


def test_large_finite_statistics_do_not_overflow_the_scalar_packet():
    covariance = torch.diag(torch.tensor([1., 1e-308, 1.], dtype=torch.float64))[None].repeat(3, 1, 1)
    report = observe(covariance)
    assert report['valid']
    assert mean(report, 'tangent_anisotropy') == pytest.approx(1e154, rel=1e-14)
    assert report['populations']['all']['tangent_anisotropy']['rms'] == pytest.approx(1e154, rel=1e-14)
    json.dumps(report, allow_nan=False)


class NoScalarExtraction(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert func != torch.ops.aten._local_scalar_dense.default
        return func(*args, **(kwargs or {}))


def test_ownership_no_autograd_and_no_per_particle_scalar_extraction():
    covariance = torch.eye(3, dtype=torch.float64).repeat(3, 1, 1).requires_grad_()
    normals = torch.tensor([[0., 0., 1.]]*3, requires_grad=True)
    support = torch.ones(3, requires_grad=True)
    opacity = torch.ones(3, requires_grad=True)
    mask = torch.tensor([True, False, True])
    inputs = (covariance, normals, support, opacity, mask)
    originals = [v.detach().clone() for v in inputs]
    with NoScalarExtraction():
        report = summarize_world_footprints(covariance, normals, support=support, opacity=opacity,
                                             populations={'chosen': mask})
    for value, original in zip(inputs, originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
        assert value.grad is None
    assert report['valid'] and report['populations']['chosen']['valid_count'] == 2
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('change', [
    {'covariance': torch.ones(3, 3)},
    {'normals': torch.ones(2, 3)},
    {'support': torch.ones(1, dtype=torch.int64)},
    {'opacity': torch.ones(1, 1)},
    {'populations': []},
    {'populations': {'all': torch.ones(1, dtype=torch.bool)}},
    {'populations': {'bad': torch.ones(1)}},
])
def test_structural_contract_fails_closed(change):
    arguments = dict(covariance=torch.eye(3)[None], normals=torch.tensor([[0., 0., 1.]]),
                     support=torch.ones(1), opacity=torch.ones(1))
    arguments.update(change)
    with pytest.raises(ValueError):
        summarize_world_footprints(**arguments)
