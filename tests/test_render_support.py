import numpy as np

from physmorph.render.support import MaterialSupport


def _lattice(n=3):
    a = np.linspace(-0.2, 0.2, n, dtype=np.float32)
    return np.stack(np.meshgrid(a, a, a, indexing="ij"), -1).reshape(-1, 3)


def test_rigid_motion_and_uniform_scaling_keep_all_primitives_visible():
    rest = _lattice()
    graph = MaterialSupport.from_rest(rest)
    x = 1.7 * rest + np.array([3.0, -2.0, 0.5], np.float32)
    np.testing.assert_allclose(graph.opacity(x), 1.0)


def test_only_a_materially_disconnected_singleton_fades():
    rest = _lattice()
    graph = MaterialSupport.from_rest(rest)
    x = rest.copy()
    p = len(x) // 2
    x[p] += np.array([4.0, 0.0, 0.0], np.float32)
    alpha = graph.opacity(x)
    assert alpha[p] == 0.0
    assert int((alpha < 0.5).sum()) == 1


def test_coherent_material_patch_is_not_hidden_even_when_far_from_the_body():
    rest = _lattice(4)
    graph = MaterialSupport.from_rest(rest)
    x = rest.copy()
    patch = np.flatnonzero(rest[:, 0] > 0.15)
    x[patch] += np.array([4.0, 0.0, 0.0], np.float32)
    alpha = graph.opacity(x)
    retained = [p for p in patch if np.isin(graph.neighbor[p], patch).sum() >= 2]
    assert retained and np.all(alpha[retained] == 1.0)


def test_opacity_is_bounded_monotone_and_does_not_mutate_positions():
    rest = _lattice()
    graph = MaterialSupport.from_rest(rest)
    p = len(rest) // 2
    vals = []
    for d in np.linspace(0.0, 4.0, 41):
        x = rest.copy(); x[p, 0] += d
        before = x.copy()
        a = graph.opacity(x)
        assert np.all((0.0 <= a) & (a <= 1.0))
        np.testing.assert_array_equal(x, before)
        vals.append(float(a[p]))
    assert all(a + 1e-6 >= b for a, b in zip(vals, vals[1:]))
    assert vals[0] == 1.0 and vals[-1] == 0.0


"""CPU algebra checks for the Torch operations used with CUDA render tensors."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from physmorph.render.support import live_support, normal_filter_size, filter_normal_buffer
from physmorph.render.settled import SettledAppearance


def test_compact_support_never_increases_old_support_and_vanishes_outside():
    generator = torch.Generator().manual_seed(8)
    distances = torch.cat((torch.zeros(1000, 1), torch.rand(1000, 8, generator=generator)*2), 1)
    hard = live_support(distances, 1., .2)
    smooth = live_support(distances, 1., .2, smooth=True)
    assert torch.all((0 <= smooth) & (smooth <= hard))
    outside = torch.cat((torch.zeros(2, 1), torch.ones(2, 8)), 1)
    assert torch.equal(live_support(outside, 1., .2, smooth=True), torch.zeros(2))
    outside[:, 1:] += .01
    assert torch.equal(live_support(outside, 1., .2, smooth=True), torch.zeros(2))


def test_compact_boundary_crossing_is_continuous_and_monotone():
    # Only one neighbor crosses the old hard boundary; all others are unsupported.
    distances = torch.full((2001, 9), 2., dtype=torch.float64)
    distances[:, 0] = 0
    distances[:, 1] = torch.linspace(.79, 1.01, len(distances), dtype=torch.float64)
    soft = live_support(distances, 1., .2, smooth=True)
    hard = live_support(distances, 1., .2)
    assert (soft[1:]-soft[:-1]).abs().max() < .001
    assert (hard[1:]-hard[:-1]).abs().max() == .25
    assert torch.all(soft[1:] <= soft[:-1])
    assert soft[0] == .25 and soft[-1] == 0


def test_pins_keep_normals_and_radius_while_current_support_can_drop():
    lock = SettledAppearance(np.array([0, 99]), 'cpu')
    x = torch.zeros(2, 3)
    normal = torch.tensor([[0., 1., 0.], [0., 0., 1.]])
    radius = torch.tensor([.2, .3])
    first = torch.cat((torch.zeros(2, 1), torch.full((2, 8), .5)), 1)
    support = live_support(first, 1., .2, smooth=True)
    lock.apply(0, x, normal, radius, support)
    first[:, 1:] = 1.1
    current = live_support(first, 1., .2, smooth=True)
    n, r, s = lock.apply(1, x, -normal, 2*radius, current)
    assert torch.equal(n[0], normal[0]) and torch.equal(r[0], radius[0])
    assert torch.equal(s, torch.zeros(2))
    x[0, 0] = .01
    with pytest.raises(ValueError, match='active pin moved'):
        lock.apply(2, x, normal, radius, current)


def test_normal_filter_changes_only_normal_estimate_and_preserves_old_default():
    generator = torch.Generator().manual_seed(2)
    coverage = torch.rand(9, 11, generator=generator)
    normal = torch.rand(9, 11, 3, generator=generator)*coverage[..., None]
    saved = coverage.clone()
    def old_smooth(image):
        return F.avg_pool2d(image.permute(2, 0, 1)[None], 3, 1, 1)[0].permute(1, 2, 0)
    old = F.normalize(2*old_smooth(normal)/old_smooth(coverage[..., None]).clamp_min(1e-3)-1,
                      dim=-1, eps=1e-6)
    assert torch.equal(filter_normal_buffer(normal, coverage), old)
    assert not torch.equal(filter_normal_buffer(normal, coverage, 5), old)
    assert torch.equal(coverage, saved)
    assert normal_filter_size(1080, True) == 3
    assert normal_filter_size(2160, True) == 5
    assert normal_filter_size(2160) == 3


@pytest.mark.parametrize('radius,spacing', [(0, 1), (1, 0), (float('nan'), 1), (1, float('inf'))])
def test_invalid_support_scales_fail(radius, spacing):
    with pytest.raises(ValueError, match='finite and positive'):
        live_support(torch.zeros(1, 9), radius, spacing, smooth=True)


def test_studio_diagnostic_buffers_have_a_shared_pixel_mask(monkeypatch):
    from physmorph.render import studio as renderer
    monkeypatch.setattr(renderer, 'decompose_cov_torch', lambda cov: (torch.ones(1, 3), torch.ones(1, 4)))
    studio = renderer.StudioRaster.__new__(renderer.StudioRaster)
    studio.direct_covariance = False
    studio.raster = lambda *args, **kw: (kw['colors_precomp'][0, :, None, None].expand(3, 4, 6),)
    studio.to_camera = torch.tensor([0., 0., 1.]).expand(4, 6, 3)
    studio.background = torch.zeros(4, 6, 3)
    studio.albedo = torch.ones(3)
    studio.lights = []
    image, coverage, normal = studio(torch.zeros(1, 3), torch.tensor([[0., 1., 0.]]),
                                     torch.eye(3)[None], torch.ones(1), return_buffers=True)
    mask = coverage >= .5
    assert coverage.shape == (4, 6) and image.shape == normal.shape == (4, 6, 3)
    assert image.mean(-1)[mask].shape == normal.norm(dim=-1)[mask].shape == (24,)


@pytest.mark.parametrize('transform', [
    torch.eye(3),
    torch.tensor([[-.5, -.8660254, 0.], [.8660254, -.5, 0.], [0., 0., 1.]]),
    torch.tensor([[1., .7, .2], [0., 1., .3], [0., 0., 1.]])])
def test_full_affine_shading_preserves_identity_rotation_and_shear(transform):
    from physmorph.render.support import rest_affine_inverse, transport_affine_normal
    generator = torch.Generator().manual_seed(52)
    rest = torch.randn(8, 32, 3, generator=generator)
    normal = F.normalize(torch.randn(8, 3, generator=generator), dim=1)
    inverse, rank_valid = rest_affine_inverse(rest)
    current = rest @ transform.T
    carried, valid = transport_affine_normal(rest, current, inverse, normal)
    expected = F.normalize(normal @ torch.linalg.inv(transform), dim=1)
    assert rank_valid.all() and valid.all()
    torch.testing.assert_close(carried, expected, atol=2e-6, rtol=2e-6)


def test_degenerate_or_invalid_material_fit_uses_current_refit_not_cached_normal():
    from physmorph.render.support import MaterialShadingNormals, rest_affine_inverse
    generator = torch.Generator().manual_seed(24)
    points = torch.randn(48, 3, generator=generator)
    neighbors = torch.cdist(points, points).argsort(1)[:, :33]
    normals = F.normalize(points, dim=1)
    exposed = torch.ones(48, dtype=torch.bool)
    locked = torch.zeros(48, dtype=torch.bool)
    state = MaterialShadingNormals(48, 'cpu')
    state.update(points, normals, neighbors, exposed, locked)
    same, status = state.update(points, normals, neighbors, exposed, locked)
    assert status['transported'].all() and not status['invalid_fit'].any()
    torch.testing.assert_close(same, normals, atol=2e-6, rtol=2e-6)
    # Reflection is a rejected material fit; result must be the new current refit.
    reflected = points.clone(); reflected[:, 0] *= -1
    refit = -normals
    result, status = state.update(reflected, refit, neighbors, exposed, locked)
    assert status['invalid_fit'].all()
    assert torch.equal(result, refit)
    flat = torch.randn(3, 32, 3, generator=generator); flat[:, :, 2] = 0
    _, valid = rest_affine_inverse(flat)
    assert not valid.any()


def test_material_shading_pin_latch_survives_changing_neighbors():
    from physmorph.render.support import MaterialShadingNormals
    generator = torch.Generator().manual_seed(14)
    x = torch.randn(48, 3, generator=generator)
    neighbors = torch.cdist(x, x).argsort(1)[:, :33]
    n = F.normalize(x, dim=1)
    material = MaterialShadingNormals(48, 'cpu')
    latch = SettledAppearance(np.r_[0, np.full(47, 99)], 'cpu')
    sigma, support = torch.ones(48), torch.ones(48)
    first, _ = material.update(x, n, neighbors, torch.ones(48, dtype=torch.bool), latch.anchored)
    first, _, _ = latch.apply(0, x, first, sigma, support)
    changed = x.clone(); changed[1:] += .3
    current, status = material.update(changed, -n, neighbors, torch.ones(48, dtype=torch.bool), latch.anchored)
    current, _, _ = latch.apply(1, changed, current, sigma, support)
    assert status['pinned_frozen'][0] and torch.equal(current[0], first[0])
