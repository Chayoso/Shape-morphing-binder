"""Gradient-combination rules (docs/render_controls_physics.md §5)."""
import math

import torch

from physmorph.pipeline.grad_combine import (blend_anchored, cagrad_two, combine, pcgrad)


def _cos(a, b):
    return float(sum((x * y).sum() for x, y in zip(a, b))
                 / (math.sqrt(float(sum(x.pow(2).sum() for x in a)))
                    * math.sqrt(float(sum(y.pow(2).sum() for y in b))) + 1e-30))


def test_pcgrad_both_sides():
    gp = [torch.tensor([1.0, 0.0, 0.0])]
    gr = [torch.tensor([-2.0, 1.0, 0.0])]
    out, c = pcgrad(gp, gr)
    assert c and torch.allclose(out[0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    out2, c2 = pcgrad(gr, gp)                     # strip physics along render
    assert c2 and abs(float((out2[0] * gr[0]).sum())) < 1e-6
    out3, c3 = pcgrad(gp, [torch.tensor([0.5, 3.0, 0.0])])
    assert not c3


def test_cagrad_is_descent_for_both_tasks_when_they_conflict_mildly():
    torch.manual_seed(0)
    ga = [torch.randn(50)]
    gb = [ga[0] * 0.3 + torch.randn(50) * 0.8]   # positively correlated but noisy
    d = cagrad_two(ga, gb, c=0.5)
    assert float((d[0] * ga[0]).sum()) > 0 and float((d[0] * gb[0]).sum()) > 0
    # c=0 reduces to the mean gradient
    d0 = cagrad_two(ga, gb, c=0.0)
    assert torch.allclose(d0[0], 0.5 * (ga[0] + gb[0]), atol=1e-5)


def test_cagrad_guards_the_worse_task_under_conflict():
    ga = [torch.tensor([1.0, 0.0])]
    gb = [torch.tensor([-0.6, 0.8])]              # cos = -0.6
    d = cagrad_two(ga, gb, c=0.9)
    # with strong conservatism the direction must not hurt either task by much
    assert float((d[0] * gb[0]).sum()) > -1e-3
    assert float((d[0] * ga[0]).sum()) > -1e-3


def test_blend_keeps_physics_magnitude_and_steers_toward_render():
    gp = [torch.tensor([3.0, 0.0, 0.0])]
    gr = [torch.tensor([0.0, 1.0, 0.0])]
    d = blend_anchored(gp, gr, beta=1.0)
    assert abs(float(d[0].norm()) - 3.0) < 1e-6
    assert abs(_cos(d, gp) - math.cos(math.pi / 4)) < 1e-5


def test_combine_modes_dispatch_and_conflict_flag():
    gp = [torch.tensor([1.0, 0.0])]
    gr = [torch.tensor([-1.0, 0.5])]
    for mode in ("render", "phys", "cagrad", "blend", "off"):
        g, info = combine(mode, gp, gr, lam_r=2.0, cagrad_c=0.5)
        assert len(g) == 1 and torch.isfinite(g[0]).all()
        if mode != "off":
            assert info["conflicted"]
    g_off, _ = combine("off", gp, gr, 2.0)
    assert torch.allclose(g_off[0], torch.tensor([-1.0, 1.0]))
    g_phys, _ = combine("phys", gp, gr, 2.0)     # physics component along render removed
    assert float((g_phys[0] * gr[0]).sum()) > float((g_off[0] * gr[0]).sum()) - 1e-6
    try:
        combine("bogus", gp, gr, 1.0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_blend_via_combine_differs_from_sum_and_uses_fixed_beta():
    """REFUTE F3: with beta derived from lambda the blend collapsed onto the plain sum."""
    torch.manual_seed(3)
    gp = [torch.randn(40) * 5.0]
    gr = [torch.randn(40) * 0.01]
    lam = 7.3                                            # an EMA'd / capped balancer value
    g_sum, _ = combine("off", gp, gr, lam)
    g_blend, info = combine("blend", gp, gr, lam, blend_beta=0.5)
    assert info["beta"] == 0.5
    assert abs(float(g_blend[0].norm()) - float(gp[0].norm())) < 1e-4
    assert _cos(g_blend, g_sum) < 0.999                  # a distinct direction
    expect = blend_anchored(gp, gr, 0.5)
    assert torch.allclose(g_blend[0], expect[0], atol=1e-6)
