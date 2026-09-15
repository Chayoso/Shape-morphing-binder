"""Support-gated APIC (Yao-Zhao 2026, arXiv 2603.03860), warp CPU.

omega_p = smoothstep((n_p/n0 - r_lo)/(r_hi - r_lo)) scales the affine term m*C in P2G, with
n_p the particle count in the 3^3 cells around p. Checks: (1) gate off, or saturated, is plain
APIC bit-for-bit; (2) an isolated particle carrying an affine field keeps it under plain APIC
(APIC reproduces an affine field exactly on a lone particle - the positional trap) and loses
it under the gate, while its translation is unchanged either way (a lone particle cannot
accelerate itself); (3) the gated adjoint matches central finite differences.
"""
import dataclasses

import numpy as np
import pytest
import torch

from physmorph.mpm.function import RolloutSpec, warp_mpm_ext
from physmorph.mpm.state import MPMParams
from physmorph.mpm.step import nominal_support
from physmorph.mpm.traj import Trajectory, compute_rest_volumes

DEV = "cpu"


def _cloud(n=200):
    rng = np.random.default_rng(3)
    x = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)   # a dense blob ...
    x[0] = (4.0, 4.0, 4.0)                                    # ... and one isolated particle
    return x


def _prm(**kw):
    return MPMParams(dx=0.5, dt=1.0 / 240.0, drag=0.0, smoothing=1.0,
                     grid_min=(-6.0, -6.0, -6.0), nx=24, ny=24, nz=24, **kw)


def _run(x0, prm, C0, T=2):
    vol0 = compute_rest_volumes(x0, 1.0, prm, DEV)
    tr = Trajectory(x0, 1.0, 0.0, 0.0, prm, T, C0=C0, device=DEV, requires_grad=False, vol0=vol0)
    tr.rollout()
    return tr


def test_gate_off_or_saturated_is_plain_apic():
    x0 = _cloud()
    C0 = np.tile(np.diag([0.3, -0.2, 0.1]).astype(np.float32), (len(x0), 1, 1))
    a = _run(x0, _prm(), C0)
    assert not a.gate
    b = _run(x0, _prm(gate_r_lo=0.0, gate_r_hi=1e-6, gate_n0=1.0), C0)   # every omega == 1
    assert b.gate and float(b.omega[0].numpy().min()) == 1.0
    assert np.array_equal(a.x[-1].numpy(), b.x[-1].numpy())
    assert np.array_equal(a.C[-1].numpy(), b.C[-1].numpy())


def test_isolated_particle_loses_its_affine_field_under_the_gate():
    x0 = _cloud()
    C0 = np.zeros((len(x0), 3, 3), np.float32)
    C0[0] = np.diag([0.3, -0.2, 0.1])
    plain = _run(x0, _prm(), C0, T=1)
    n0 = nominal_support(x0, _prm(), DEV)
    assert n0 > 10.0                                          # interior-dominated median
    gated = _run(x0, _prm(gate_r_lo=0.1, gate_r_hi=0.5, gate_n0=n0), C0, T=1)
    om = gated.omega[0].numpy()
    assert om[0] == 0.0 and om.max() == 1.0 and ((om > 0.0) & (om < 1.0)).any()
    assert np.abs(plain.C[1].numpy()[0]).max() > 0.1         # APIC keeps the lone affine field
    assert np.abs(gated.C[1].numpy()[0]).max() < 1e-6         # gate: PIC transfer, no affine memory
    assert np.allclose(plain.v[1].numpy()[0], gated.v[1].numpy()[0], atol=1e-7)  # translation same
    assert np.allclose(plain.x[1].numpy()[0], gated.x[1].numpy()[0], atol=1e-7)


def _loss(dfc, spec, kind):
    xT, FT, vT, FgT, V = warp_mpm_ext(dfc, spec)
    if kind == "x":
        return (xT * torch.linspace(0.5, 1.5, 3)).pow(2).sum() * 1e2
    return V.pow(2).sum(2).mean() * 1e2


@pytest.mark.parametrize("kind", ["x", "vall"])
def test_gated_adjoint_matches_finite_difference(kind):
    rng = np.random.default_rng(20260915)
    x0 = rng.uniform(-1.0, 1.0, (40, 3)).astype(np.float32)
    prm = MPMParams(dx=0.75, dt=1.0 / 120.0, drag=0.9, smoothing=0.955,
                    grid_min=(-6.0, -6.0, -6.0), nx=16, ny=16, nz=16,
                    gate_r_lo=0.2, gate_r_hi=1.5)             # a spread of omega in (0, 1)
    prm = dataclasses.replace(prm, gate_n0=nominal_support(x0, prm, DEV))
    vol0 = compute_rest_volumes(x0, 1.0, prm, DEV)
    C0 = (rng.standard_normal((40, 3, 3)) * 0.3).astype(np.float32)   # the gated term is live
    spec = RolloutSpec(x0=x0, m=1.0, lam=800.0, mu=400.0, prm=prm, T=3, device=DEV,
                       vol0=vol0, C0=C0)
    torch.manual_seed(2)
    dfc = (torch.randn(3, len(x0), 3, 3) * 2e-2).requires_grad_(True)
    L = _loss(dfc, spec, kind)
    (g,) = torch.autograd.grad(L, dfc)
    assert torch.isfinite(g).all() and float(g.abs().max()) > 0
    idx = tuple(int(i) for i in np.unravel_index(int(g.abs().argmax()), g.shape))
    eps = 2e-3
    with torch.no_grad():
        dp, dm = dfc.clone(), dfc.clone()
        dp[idx] += eps
        dm[idx] -= eps
        fd = float((_loss(dp, spec, kind) - _loss(dm, spec, kind)) / (2 * eps))
    an = float(g[idx])
    assert abs(fd - an) / max(abs(fd), abs(an), 1e-9) < 0.05, (fd, an)
