"""Autograd-bridge gates on the warp CPU device — the G1a/G1b checks runnable locally.

Slower than the pure-numpy tests (warp compiles its CPU kernels once, then cached).
"""
import numpy as np
import pytest
import torch

from physmorph.mpm.constitutive import lame
from physmorph.mpm.function import RolloutSpec, warp_mpm, warp_mpm_full
from physmorph.mpm.state import MPMParams

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
