"""Chebyshev-accelerated grid Gauss-Seidel (Wang 2015) on the screened-diffusion
preconditioner, multi-channel fields (x AND F covectors). Torch CPU."""
import numpy as np
import torch

from physmorph.pipeline.grid_smooth import chebyshev_rho, smooth_particle_field

GMIN = torch.tensor([-2.0, -2.0, -2.0])
DX, DIMS = 0.25, (16, 16, 16)


def _cloud(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.uniform(-1.5, 1.5, (n, 3)).astype(np.float32))


def test_multichannel_field_is_supported_and_norm_preserved():
    x = _cloud()
    g = torch.randn(len(x), 9)
    gs = smooth_particle_field(x, g, GMIN, DX, DIMS, iters=8)
    assert gs.shape == g.shape
    assert abs(gs.norm().item() - g.norm().item()) < 1e-4 * g.norm().item()


def test_chebyshev_converges_faster_than_plain_sweeps():
    """Error to the CONVERGED solution after k sweeps (the residual re-scattered from
    particles floors at the gather/scatter round-trip error and cannot see this).
    Measured 2026-09-14 (6k particles, 16^3): kappa 20 / 20 sweeps: plain 1.5e-2,
    Chebyshev 4.9e-4 (30x); kappa 4 / 12 sweeps: 2.0e-4 vs 7.9e-6 (25x); kappa 50 /
    40 sweeps: 2.1e-2 vs 1.0e-4 (200x). The first ~4 sweeps are WORSE (warm-up +
    extrapolation transient), as in Wang 2015 — which is why the optimizer never
    uses fewer than 8 sweeps with cheb on."""
    torch.manual_seed(0)
    x = _cloud(6000)
    g = torch.randn(len(x), 3)
    kappa, iters = 20.0, 20
    ref = smooth_particle_field(x, g, GMIN, DX, DIMS, iters=2000, kappa=kappa, rescale=False)
    plain = smooth_particle_field(x, g, GMIN, DX, DIMS, iters=iters, kappa=kappa,
                                  rescale=False)
    cheb = smooth_particle_field(x, g, GMIN, DX, DIMS, iters=iters, kappa=kappa,
                                 cheb_rho=chebyshev_rho(kappa), rescale=False)
    e_plain = float((plain - ref).norm() / ref.norm())
    e_cheb = float((cheb - ref).norm() / ref.norm())
    assert e_cheb < 0.1 * e_plain, (e_plain, e_cheb)


def test_chebyshev_limit_matches_plain_limit():
    """Both iterations solve the SAME linear system: at convergence they agree."""
    torch.manual_seed(1)
    x = _cloud(3000)
    g = torch.randn(len(x), 3)
    kappa = 2.0
    a = smooth_particle_field(x, g, GMIN, DX, DIMS, iters=200, kappa=kappa, rescale=False)
    b = smooth_particle_field(x, g, GMIN, DX, DIMS, iters=60, kappa=kappa,
                              cheb_rho=chebyshev_rho(kappa), rescale=False)
    assert float((a - b).norm() / a.norm()) < 0.02


def test_rho_estimate_in_unit_interval():
    for k in (0.5, 4.0, 50.0):
        assert 0.0 < chebyshev_rho(k) < 1.0


def test_chebyshev_is_never_worse_than_plain_below_the_minimum_sweep_count():
    """REFUTE F6: at 3-6 sweeps the accelerated iterate was worse; the function now
    falls back to plain sweeps below cheb_min_iters, so the two coincide there."""
    torch.manual_seed(2)
    x = _cloud(3000)
    g = torch.randn(len(x), 3)
    kappa = 20.0
    for iters in (3, 6):
        a = smooth_particle_field(x, g, GMIN, DX, DIMS, iters=iters, kappa=kappa, rescale=False)
        b = smooth_particle_field(x, g, GMIN, DX, DIMS, iters=iters, kappa=kappa,
                                  cheb_rho=chebyshev_rho(kappa), rescale=False)
        assert torch.allclose(a, b)
