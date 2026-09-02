"""D_render math (docs/pipeline_v2.md §3.4): v1 equivalence, asymmetry, views, balancer.
Torch CPU only."""
import numpy as np
import pytest
import torch

from physmorph.losses.silhouette import _project, d_img, ring_thetas, target_silhouettes
from physmorph.pipeline.render_loss import (LambdaBalancer, d_pbr, d_render, field_normals,
                                            make_views, shade_targets,
                                            target_silhouettes as v2_targets)

RES, EXTENT = 32, 1.5


@pytest.fixture
def clouds():
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.uniform(-1, 1, (500, 3)).astype(np.float32))
    t = torch.tensor(rng.uniform(-1, 1, (500, 3)).astype(np.float32))
    return x, t


def test_equal_weights_reproduce_v1_d_img(clouds):
    """relu(a_t-a)^2 + relu(a-a_t)^2 == (a-a_t)^2, so w_hole=w_spray=1 must equal v1."""
    x, t = clouds
    thetas = ring_thetas(4)
    views = [(float(th), 0.0) for th in thetas]
    sils_v1 = target_silhouettes(t, thetas, RES, EXTENT)
    sils_v2 = v2_targets(t, views, RES, EXTENT)
    l_v1 = d_img(x, sils_v1, thetas, RES, EXTENT)
    l_v2 = d_render(x, sils_v2, views, RES, EXTENT, w_hole=1.0, w_spray=1.0)
    assert torch.allclose(l_v1, l_v2, rtol=1e-5, atol=1e-7)


def test_hole_weight_scales_deficit_only(clouds):
    """Current cloud = half the target -> pure deficit; w_hole must scale the loss,
    w_spray must not."""
    _, t = clouds
    views = make_views(4, (0.0,))
    sils = v2_targets(t, views, RES, EXTENT)
    half = t[:250]
    l1 = d_render(half, sils, views, RES, EXTENT, w_hole=1.0, w_spray=1.0)
    l2 = d_render(half, sils, views, RES, EXTENT, w_hole=2.0, w_spray=1.0)
    l3 = d_render(half, sils, views, RES, EXTENT, w_hole=1.0, w_spray=5.0)
    assert l2 > l1 * 1.5                      # deficit-dominated: w_hole bites
    assert abs(l3 - l1) < l1 * 0.2            # nearly no excess anywhere


def test_spray_weight_scales_excess(clouds):
    """Current = target + far spray inside the viewport -> w_spray bites."""
    _, t = clouds
    views = make_views(4, (0.0,))
    sils = v2_targets(t, views, RES, EXTENT)
    spray = torch.cat([t, t * 0.0 + torch.tensor([1.3, 1.3, 0.0])])   # off-body clump
    l1 = d_render(spray, sils, views, RES, EXTENT, w_hole=1.0, w_spray=1.0)
    l2 = d_render(spray, sils, views, RES, EXTENT, w_hole=1.0, w_spray=4.0)
    assert l2 > l1 * 1.5


def test_gradient_flows_and_is_finite(clouds):
    x, t = clouds
    views = make_views(3, (0.0, 0.5))
    sils = v2_targets(t, views, RES, EXTENT)
    xg = x.clone().requires_grad_(True)
    L = d_render(xg, sils, views, RES, EXTENT)
    (g,) = torch.autograd.grad(L, xg)
    assert torch.isfinite(g).all() and float(g.abs().sum()) > 0


def test_make_views_count_and_distinct_azimuths():
    elevs = (0.0, 0.5, -0.5)
    views = make_views(6, elevs)
    assert len(views) == 18
    # every ring must have a distinct azimuth offset (odd ring counts re-stacked in v1)
    rings = [sorted(th % (2 * np.pi) for th, phi in views if phi == e) for e in elevs]
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            gap = min(abs(a - b) for a, b in zip(rings[i], rings[j]))
            assert gap > 1e-3


def test_projection_phi_zero_matches_v1():
    x = torch.tensor(np.random.default_rng(2).uniform(-1, 1, (100, 3)).astype(np.float32))
    for th in (0.0, 1.1, 4.0):
        assert torch.allclose(_project(x, th), _project(x, th, 0.0))


GMIN = torch.tensor([-2.0, -2.0, -2.0])
PDX, PDIMS = 0.25, (16, 16, 16)


def _ball(n=4000, seed=3, squash=1.0):
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, (n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = rng.uniform(0, 1, (n, 1)) ** (1 / 3)
    x = (v * r * 1.2).astype(np.float32)
    x[:, 1] *= squash
    return torch.tensor(x)


def test_field_normals_outward_and_surface_weighted():
    x = _ball()
    n, sw = field_normals(x, GMIN, PDX, PDIMS)
    r = x.norm(dim=1)
    shell, deep = r > 1.05, r < 0.5                 # true outer band vs interior
    cos = torch.nn.functional.cosine_similarity(n[shell], x[shell] / r[shell, None])
    assert float(cos.mean()) > 0.8                  # outward normals on the shell
    # the surface weight must discriminate shell from interior (the adversarial round
    # showed the NORMALISED field alone does not, on solid clouds)
    assert float(sw[shell].mean()) > 1.5 * float(sw[deep].mean())


def test_shading_is_headlit_not_backlit():
    """Adversarial BLOCKER regression: the camera-facing hemisphere must be brighter
    than the away-facing one (previous sign lit the scene from behind, l·d = -1)."""
    from physmorph.pipeline.render_loss import shaded_view
    x = _ball(6000)
    n, sw = field_normals(x, GMIN, PDX, PDIMS)
    img, alpha = shaded_view(x, 0.0, 0.0, 48, 1.5, n, sw)   # camera on +z
    m = alpha > 0.5
    assert float(img[m].mean()) > 0.45              # lit well above pure ambient (0.25)


def test_d_pbr_zero_on_identical_and_sees_curvature():
    x = _ball()
    views = make_views(3, (0.0, 0.5))
    tgts = shade_targets(x, views, 32, 1.5, GMIN, PDX, PDIMS)
    l_same = d_pbr(x, tgts, views, 32, 1.5, GMIN, PDX, PDIMS)
    assert float(l_same) < 1e-8                     # identical cloud -> zero
    # ABSOLUTE threshold (the previous relative one collapsed to 5e-11, vacuous)
    l_squash = d_pbr(_ball(squash=0.6), tgts, views, 32, 1.5, GMIN, PDX, PDIMS)
    assert float(l_squash) > 1e-4


def test_d_pbr_gradient_flows():
    x = _ball(1500)
    views = make_views(2, (0.0,))
    tgts = shade_targets(_ball(1500, squash=0.7), views, 24, 1.5, GMIN, PDX, PDIMS)
    xg = x.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(d_pbr(xg, tgts, views, 24, 1.5, GMIN, PDX, PDIMS), xg)
    assert torch.isfinite(g).all() and float(g.abs().sum()) > 0


def test_balancer_cap():
    b = LambdaBalancer(0.5, ema=1.0, cap=100.0)
    assert b.update(1e6, 1e-3) == 100.0             # runaway ratio capped


def test_balancer_ema_and_activity():
    b = LambdaBalancer(0.0)
    assert not b.active
    b = LambdaBalancer(0.5, ema=0.3)
    assert b.active
    l1 = b.update(100.0, 1.0)                # first estimate taken as-is
    assert abs(l1 - 50.0) < 1e-9
    l2 = b.update(100.0, 100.0)              # target drops to 0.5; EMA damps the jump
    assert 0.5 < l2 < 50.0
    assert abs(l2 - (0.7 * 50.0 + 0.3 * 0.5)) < 1e-9


# ---- pointwise-W1 spray (fringe tranche, rationale.md §7) ----

def _dt_setup():
    """Target = dense clump at the origin; one view down +z."""
    from physmorph.pipeline.render_loss import target_dts
    rng = np.random.default_rng(3)
    t = torch.tensor(rng.uniform(-0.4, 0.4, (800, 3)).astype(np.float32))
    views = [(0.0, 0.0)]
    sils = v2_targets(t, views, RES, EXTENT)
    dts = target_dts(sils, EXTENT)
    return t, views, sils, dts


def test_dt_zero_inside_support_and_grows_outside():
    t, views, sils, dts = _dt_setup()
    from physmorph.pipeline.render_loss import d_spray_dt
    inside = t[:50].clone().requires_grad_(True)
    l_in = d_spray_dt(inside, dts, views, RES, EXTENT)
    assert float(l_in) < 1e-6                        # support is a zero set of the DT
    far = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    near = torch.tensor([[0.7, 0.0, 0.0]], requires_grad=True)
    assert float(d_spray_dt(far, dts, views, RES, EXTENT)) > \
        float(d_spray_dt(near, dts, views, RES, EXTENT)) > 0


def test_dt_gradient_sparsity_invariant_where_saturated_loss_fades():
    """THE mechanism claim: per-particle DT pull is identical for a lone stray and a
    co-located clump, while the saturated w_spray per-particle gradient decays."""
    t, views, sils, dts = _dt_setup()
    from physmorph.pipeline.render_loss import d_spray_dt

    def dt_grad(n_stray):
        x = torch.cat([t, torch.tensor([[0.72, 0.0, 0.0]]).repeat(n_stray, 1)])
        x = x.clone().requires_grad_(True)
        # sum (not mean) isolates the per-particle contribution from the 1/N factor
        l = d_spray_dt(x, dts, views, RES, EXTENT) * len(x)
        l.backward()
        return float(x.grad[len(t):].norm(dim=1).mean())

    g1, g8 = dt_grad(1), dt_grad(8)
    assert g1 > 1e-4
    assert abs(g1 - g8) < 0.05 * g1                  # linear term: no saturation

    def sil_grad(n_stray):
        x = torch.cat([t, torch.tensor([[0.72, 0.0, 0.0]]).repeat(n_stray, 1)])
        x = x.clone().requires_grad_(True)
        l = d_render(x, sils, views, RES, EXTENT, w_hole=0.0, w_spray=1.0)
        l.backward()
        return float(x.grad[len(t):].norm(dim=1).mean())

    # saturated loss: 8 co-located strays get LESS pull per particle than the ratio
    # a linear loss would give (alpha flattens) — the blindness being fixed
    assert sil_grad(8) < 8.0 * sil_grad(1)


def test_dt_pull_points_toward_support():
    t, views, sils, dts = _dt_setup()
    from physmorph.pipeline.render_loss import d_spray_dt
    x = torch.tensor([[0.72, 0.2, 0.0]], requires_grad=True)   # stray inside DT basin
    d_spray_dt(x, dts, views, RES, EXTENT).backward()
    step = -x.grad[0]
    assert step[0] < 0                               # descent moves toward the clump
