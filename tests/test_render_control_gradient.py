"""End-to-end render-loss gradient gate through a deterministic Warp CPU rollout."""
import numpy as np
import pytest
import torch

from physmorph.mpm.function import RolloutSpec, warp_mpm_full
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import compute_rest_volumes
from physmorph.pipeline.render_loss import d_render, make_views, target_silhouettes


DEV = "cpu"


@pytest.fixture(scope="module")
def render_case():
    # Explicit discretisation: cubic MLS-MPM, dx=.75, dt=1/120, T=3;
    # orthographic CIC render loss, 4 views at 20 px.
    rng = np.random.default_rng(20260902)
    source = rng.uniform(-1.0, 1.0, (48, 3)).astype(np.float32)
    target = (source * np.array([1.15, 0.82, 1.05], np.float32)
              + np.array([0.16, 0.05, -0.08], np.float32))
    prm = MPMParams(dx=0.75, dt=1.0 / 120.0, drag=0.9, smoothing=0.955,
                    grid_min=(-6.0, -6.0, -6.0), nx=16, ny=16, nz=16)
    vol0 = compute_rest_volumes(source, 1.0, prm, DEV)
    spec = RolloutSpec(x0=source, m=1.0, lam=800.0, mu=400.0, prm=prm, T=3,
                       device=DEV, vol0=vol0)
    views = make_views(2, (0.0, 0.35))
    extent, res, sil_k = 1.8, 20, 1.2
    silhouettes = target_silhouettes(torch.as_tensor(target), views, res, extent, sil_k)
    return source, spec, views, silhouettes, extent, res, sil_k


def _render_loss(xT, views, silhouettes, extent, res, sil_k):
    return d_render(xT, silhouettes, views, res, extent, sil_k,
                    w_hole=2.0, w_spray=1.0)


def test_terminal_render_loss_dfc_gradient_matches_central_difference(render_case):
    """Selected dFc -> MPM -> xT -> image loss derivative is live and correct."""
    source, spec, views, silhouettes, extent, res, sil_k = render_case
    dfc = torch.zeros(3, len(source), 3, 3, requires_grad=True)
    xT, _, _ = warp_mpm_full(dfc, spec)
    loss = _render_loss(xT, views, silhouettes, extent, res, sil_k)
    (grad,) = torch.autograd.grad(loss, dfc)

    # Fixed high-signal entry for this deterministic case; avoids an aggregate
    # directional test hiding a dead or mis-indexed control layer.
    idx = (0, 8, 1, 1)
    analytic = float(grad[idx])
    assert np.isfinite(analytic) and abs(analytic) > 1.0e-6

    eps = 5.0e-3
    with torch.no_grad():
        plus = torch.zeros_like(dfc)
        minus = torch.zeros_like(dfc)
        plus[idx] = eps
        minus[idx] = -eps
        xp, _, _ = warp_mpm_full(plus, spec)
        xm, _, _ = warp_mpm_full(minus, spec)
        lp = _render_loss(xp, views, silhouettes, extent, res, sil_k)
        lm = _render_loss(xm, views, silhouettes, extent, res, sil_k)
        finite_difference = float((lp - lm) / (2.0 * eps))

    rel = abs(finite_difference - analytic) / max(
        abs(finite_difference), abs(analytic), 1.0e-12)
    assert rel < 0.03


def test_render_channel_does_not_change_fixed_control_forward_rollout(render_case):
    """Render is feedback through optimisation, never a hidden forward force."""
    source, spec, views, silhouettes, extent, res, sil_k = render_case
    dfc = torch.zeros(3, len(source), 3, 3)
    dfc[0, 8, 1, 1] = 5.0e-3
    with torch.no_grad():
        x_off, F_off, v_off = warp_mpm_full(dfc, spec)
        x_on, F_on, v_on = warp_mpm_full(dfc, spec)
        _ = _render_loss(x_on, views, silhouettes, extent, res, sil_k)

    assert torch.allclose(x_on, x_off, rtol=0.0, atol=1.0e-7)
    assert torch.allclose(F_on, F_off, rtol=0.0, atol=1.0e-7)
    assert torch.allclose(v_on, v_off, rtol=0.0, atol=1.0e-7)
