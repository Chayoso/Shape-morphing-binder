"""Autograd-bridge gates on the warp CPU device — the G1a/G1b checks runnable locally.

Slower than the pure-numpy tests (warp compiles its CPU kernels once, then cached).
"""
import numpy as np
import pytest
import torch

from physmorph.mpm.constitutive import lame
from physmorph.mpm.function import RolloutSpec, warp_mpm, warp_mpm_full
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory, compute_rest_volumes

DEV = "cpu"


@pytest.fixture(scope="module")
def prm():
    return MPMParams(dx=1.0, nx=32, ny=32, nz=32)


@pytest.fixture(scope="module")
def src():
    rng = np.random.default_rng(3)
    return rng.uniform(-2, 2, (128, 3)).astype(np.float32)


def test_g1a_constant_sequence_equals_shared(prm, src):
    T = 4
    spec = RolloutSpec(x0=src, m=1.0, lam=1.0e5, mu=5.0e4, prm=prm, T=T, device=DEV)
    torch.manual_seed(0)
    c = torch.randn(len(src), 3, 3) * 1e-3
    with torch.no_grad():
        x_shared, _ = warp_mpm(c, spec)
        x_seq, _ = warp_mpm(c.unsqueeze(0).repeat(T, 1, 1, 1).contiguous(), spec)
    tol = 1e-6 * max(1.0, float(x_shared.abs().max()))
    assert float((x_shared - x_seq).abs().max()) <= tol


def test_g1b_material_leaf_matches_finite_difference(prm, src):
    """dL/ds must reach the material leaves through the tape (channel 2)."""
    lam0, mu0 = lame(1.4e5, 0.2)
    T, n = 4, len(src)
    spec = RolloutSpec(x0=src, m=1.0, lam=lam0, mu=mu0, prm=prm, T=T, device=DEV)
    torch.manual_seed(1)
    dfc = torch.randn(T, n, 3, 3) * 5e-2
    s = torch.zeros(2, n, requires_grad=True)

    def L_of(shift):
        lam_t = lam0 * torch.exp(s[0] + shift)
        mu_t = mu0 * torch.exp(s[1] + shift)
        xT, _, vT = warp_mpm_full(dfc, spec, lam_t, mu_t)
        return xT.pow(2).sum() * 1e-3 + vT.pow(2).sum() * 1e-3

    L = L_of(0.0)
    (g,) = torch.autograd.grad(L, s)
    assert torch.isfinite(g).all()
    an = float(g.sum())
    assert abs(an) > 1e-12                    # channel is alive, not silently dead
    eps = 1e-2
    with torch.no_grad():
        fd = (float(L_of(+eps)) - float(L_of(-eps))) / (2 * eps)
    assert abs(fd - an) / max(abs(fd), abs(an)) < 0.25


def test_v_t_output_and_adjoint(prm, src):
    """v_T must be returned and differentiable (channel 4's loss term)."""
    T, n = 3, len(src)
    spec = RolloutSpec(x0=src, m=1.0, lam=1.0e5, mu=5.0e4, prm=prm, T=T, device=DEV)
    torch.manual_seed(2)
    dfc = (torch.randn(T, n, 3, 3) * 5e-2).requires_grad_(True)
    xT, FT, vT = warp_mpm_full(dfc, spec)
    assert vT.shape == (n, 3)
    L = vT.pow(2).sum()
    (g,) = torch.autograd.grad(L, dfc)
    assert torch.isfinite(g).all() and float(g.abs().sum()) > 0


def test_source_rest_volume_is_reused_without_recomputation(prm, src, monkeypatch):
    """Vp0 supplied by the source must bypass per-trajectory density fitting."""
    vol0 = compute_rest_volumes(src, 1.0, prm, DEV)
    assert vol0.shape == (len(src),)
    assert np.isfinite(vol0).all() and (vol0 > 0.0).all()
    fallback = Trajectory(src, 1.0, 1.0e5, 5.0e4, prm, 1,
                          device=DEV, requires_grad=False)
    assert np.allclose(fallback.vol.numpy(), vol0, rtol=1e-6, atol=1e-7)

    def forbidden_recompute(_self):
        raise AssertionError("a supplied source-rest Vp0 must not be recomputed")

    monkeypatch.setattr(Trajectory, "_compute_volumes", forbidden_recompute)
    # A different x0 models a later outer window.  Its trajectory must retain the
    # source volumes verbatim rather than fitting volumes to the deformed density.
    x_later = np.ascontiguousarray(src * np.array([1.2, 0.8, 1.0], np.float32))
    tr = Trajectory(x_later, 1.0, 1.0e5, 5.0e4, prm, 1,
                    device=DEV, requires_grad=False, vol0=vol0)
    assert np.array_equal(tr.vol.numpy(), vol0)


def test_rollout_spec_forwards_source_rest_volume(prm, src, monkeypatch):
    """The differentiable torch bridge must also bypass trajectory recomputation."""
    vol0 = compute_rest_volumes(src, 1.0, prm, DEV)

    def forbidden_recompute(_self):
        raise AssertionError("RolloutSpec.vol0 was not forwarded to Trajectory")

    monkeypatch.setattr(Trajectory, "_compute_volumes", forbidden_recompute)
    spec = RolloutSpec(x0=src, m=1.0, lam=1.0e5, mu=5.0e4, prm=prm, T=2,
                       vol0=vol0, device=DEV)
    dfc = (torch.randn(2, len(src), 3, 3) * 1e-3).requires_grad_(True)
    xT, _, vT = warp_mpm_full(dfc, spec)
    (g,) = torch.autograd.grad(xT.pow(2).mean() + vT.pow(2).mean(), dfc)
    assert torch.isfinite(g).all()


def test_trajectory_rejects_invalid_source_rest_volume(prm, src):
    with pytest.raises(ValueError, match="vol0 must have shape"):
        Trajectory(src, 1.0, 1.0e5, 5.0e4, prm, 1,
                   device=DEV, requires_grad=False,
                   vol0=np.ones(len(src) - 1, np.float32))
