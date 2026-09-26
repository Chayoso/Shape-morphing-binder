"""Actual CPU silhouette calculations for the promoted-state outer merit."""
from types import SimpleNamespace

import pytest
import torch

from physmorph.losses.silhouette import soft_silhouette
from physmorph.pipeline.config import PipelineConfig
from physmorph.pipeline.outer_merit import fixed_outer_render
from physmorph.pipeline.render_loss import d_render, target_silhouettes


@pytest.fixture
def scene():
    axis = torch.linspace(-.4, .4, 4)
    target = torch.cartesian_prod(axis, axis * .75, axis * .5)
    cfg = PipelineConfig(lambda_auto=.8, render_res=20, sil_k=1.9,
                         w_hole=3.25, w_spray=.75, w_pbr=7.)
    views = [(0., 0.), (.8, .35), (1.9, -.2)]
    tgt = SimpleNamespace(views=views, extent=1.5,
                          sils=target_silhouettes(target, views, cfg.render_res,
                                                 1.5, cfg.sil_k))
    return target, cfg, tgt


def test_paced_inner_values_do_not_change_fixed_outer_merit(scene):
    target, cfg, tgt = scene
    promoted = target + target.new_tensor((-.3, .04, 0.))
    inner = []
    for lead in (.05, .2):
        paced = promoted + promoted.new_tensor((lead, 0., 0.))
        paced_sils = target_silhouettes(paced, tgt.views, cfg.render_res,
                                        tgt.extent, cfg.sil_k)
        inner.append(d_render(promoted, paced_sils, tgt.views, cfg.render_res,
                              tgt.extent, cfg.sil_k, cfg.w_hole, cfg.w_spray))
    assert not torch.isclose(inner[0], inner[1])
    first = fixed_outer_render(promoted, cfg, tgt, inner[0], inner[0] + .5)
    second = fixed_outer_render(promoted, cfg, tgt, inner[1], inner[1] + 10.)
    explicit = d_render(promoted, tgt.sils, tgt.views, cfg.render_res,
                        tgt.extent, cfg.sil_k, cfg.w_hole, cfg.w_spray)
    torch.testing.assert_close(first, second, rtol=0., atol=0.)
    torch.testing.assert_close(first, explicit, rtol=0., atol=0.)


def test_same_inner_endpoint_different_promoted_positions_change_merit(scene):
    target, cfg, tgt = scene
    inner_endpoint = target + target.new_tensor((-.2, .02, 0.))
    inner = d_render(inner_endpoint, tgt.sils, tgt.views, cfg.render_res,
                     tgt.extent, cfg.sil_k, cfg.w_hole, cfg.w_spray)
    corrected = fixed_outer_render(target, cfg, tgt, inner, inner)
    displaced = fixed_outer_render(target + .35, cfg, tgt, inner, inner)
    assert corrected < 1e-8
    assert displaced > corrected + 1e-4


def test_explicit_fixed_target_formula_device_and_no_autograd(scene):
    target, cfg, tgt = scene
    promoted = (target * .8 + .13).requires_grad_()
    original = promoted.detach().clone()
    targets_before = [alpha.clone() for alpha in tgt.sils]
    value = fixed_outer_render(promoted, cfg, tgt, .5, 100.)
    penalties = []
    for alpha, (theta, phi) in zip(tgt.sils, tgt.views):
        rendered = soft_silhouette(promoted.detach(), theta, cfg.render_res,
                                   tgt.extent, cfg.sil_k, phi)
        penalties.append((cfg.w_hole * (alpha-rendered).clamp_min(0).square()
                          + cfg.w_spray * (rendered-alpha).clamp_min(0).square()).mean())
    torch.testing.assert_close(value, torch.stack(penalties).mean(), rtol=1e-5, atol=1e-8)
    assert value.ndim == 0 and value.device == promoted.device
    assert not value.requires_grad and value.grad_fn is None
    torch.testing.assert_close(promoted.detach(), original, rtol=0., atol=0.)
    for before, after in zip(targets_before, tgt.sils):
        torch.testing.assert_close(before, after, rtol=0., atol=0.)


def test_inactive_channels_do_not_access_inputs_or_create_merit():
    # No target access, rasterization or device allocation is needed when off.
    assert fixed_outer_render(None, None, None, None, None) is None


def test_inactive_channels_stay_off_with_cached_fixed_target(scene):
    target, cfg, tgt = scene
    assert fixed_outer_render(target, cfg, tgt, None, None) is None


@pytest.mark.parametrize('inner_sil,inner_render', [(0., None), (None, 0.), (0., 0.)])
def test_zero_valued_inner_channel_remains_eligible(scene, inner_sil, inner_render):
    target, cfg, tgt = scene
    value = fixed_outer_render(target + .2, cfg, tgt, inner_sil, inner_render)
    assert isinstance(value, torch.Tensor) and value > 0


@pytest.mark.parametrize('sils', [None, []])
def test_active_channel_requires_fixed_target(scene, sils):
    target, cfg, tgt = scene
    tgt.sils = sils
    with pytest.raises(ValueError, match='requires fixed tgt.sils'):
        fixed_outer_render(target, cfg, tgt, None, 0.)


def test_active_channel_rejects_mismatched_target_views(scene):
    target, cfg, tgt = scene
    tgt.sils = tgt.sils[:1]
    with pytest.raises(ValueError, match='one image per target view'):
        fixed_outer_render(target, cfg, tgt, 0., None)


def test_fixed_target_is_not_implicitly_moved_or_resized(scene):
    target, cfg, tgt = scene
    with pytest.raises(ValueError, match='share a device'):
        fixed_outer_render(target.to('meta'), cfg, tgt, 0., None)
    cfg.render_res += 1
    with pytest.raises(ValueError, match='resolution'):
        fixed_outer_render(target, cfg, tgt, 0., None)
