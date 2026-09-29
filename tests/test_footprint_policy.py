import math

import pytest
import torch

from physmorph.render.footprint_policy import reference_thickness_covariance


def test_reference_thickness_preserves_tangent_restriction_and_bounds_normal():
    angle = .71
    rotation = torch.tensor([[math.cos(angle), 0., math.sin(angle)],
                             [0., 1., 0.], [-math.sin(angle), 0., math.cos(angle)]],
                            dtype=torch.float64).expand(3, -1, -1)
    sigma = torch.tensor([.1, .2, .4], dtype=torch.float64)
    cov = reference_thickness_covariance(rotation, sigma, .1)
    local = rotation.transpose(1, 2) @ cov @ rotation
    torch.testing.assert_close(local[:, :2, :2], torch.diag_embed(sigma.square().expand(2, -1).T))
    torch.testing.assert_close(local[:, 2, 2], torch.full_like(sigma, .025 ** 2))
    torch.testing.assert_close(local[:, :2, 2], torch.zeros((3, 2), dtype=sigma.dtype), atol=1e-17, rtol=0.)
    assert torch.all(torch.linalg.eigvalsh(cov) > 0)


def test_reference_thickness_unchanged_at_native_spacing_and_inputs_owned():
    rotation = torch.eye(3).expand(2, -1, -1).clone()
    sigma = torch.full((2,), .25)
    before = rotation.clone(), sigma.clone()
    expected = torch.diag(torch.tensor([.25 ** 2, .25 ** 2, (.25 / 4) ** 2])).expand(2, -1, -1)
    cov = reference_thickness_covariance(rotation, sigma, .25)
    assert torch.equal(cov, expected)
    cov.zero_()
    assert torch.equal(rotation, before[0]) and torch.equal(sigma, before[1])


@pytest.mark.parametrize('spacing', [0., -1., float('nan'), float('inf')])
def test_reference_thickness_rejects_invalid_reference(spacing):
    with pytest.raises(ValueError, match='Reference spacing'):
        reference_thickness_covariance(torch.eye(3)[None], torch.ones(1), spacing)
