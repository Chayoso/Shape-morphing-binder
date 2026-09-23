"""Discretisation contract: dx from (N, volume, ppc); CFL and hole-risk telemetry."""
import numpy as np
import pytest

from physmorph.mpm.discretisation import derive, measure_ppc, report


def test_dx_gives_requested_ppc_on_a_uniform_cloud():
    rng = np.random.default_rng(0)
    N, side = 20000, 4.0
    x = rng.uniform(-side / 2, side / 2, (N, 3)).astype(np.float32)
    d = derive(N, volume=side ** 3, extent=side * np.sqrt(3), dt=1 / 240, young=1.4e5,
               poisson=0.2, ppc=8.0)
    m = measure_ppc(x, d.dx, d.grid_min)
    # by construction N dx^3 / V == ppc up to the domain-tiling rounding of dx
    assert N * d.dx ** 3 / side ** 3 == pytest.approx(8.0, rel=0.15)
    # measured occupancy (boundary cells are partially filled, so the median sits
    # below the interior value of ~8)
    assert 5.0 <= m["ppc_median"] <= 10.5, m
    assert d.grid_n * d.dx == pytest.approx(32.0)
    assert d.kernel_support_ratio > 3.0


def test_stability_numbers_and_warnings():
    d = derive(5000, volume=30.0, extent=8.0, dt=1 / 240, young=1.4e5, poisson=0.2, ppc=8)
    assert d.rho == pytest.approx(5000 / 30.0)
    lam = 1.4e5 * 0.2 / (1.2 * 0.6)
    mu = 1.4e5 / 2.4
    assert d.sound_speed == pytest.approx(np.sqrt((lam + 2 * mu) / d.rho))
    assert d.cfl == pytest.approx(d.sound_speed / 240 / d.dx)
    txt = report(d)
    assert "CFL" in txt
    stiff = derive(5000, volume=30.0, extent=8.0, dt=1 / 240, young=1.4e7, poisson=0.2, ppc=8)
    assert "WARNING: CFL" in report(stiff)


def test_rejects_inconsistent_requests():
    with pytest.raises(ValueError):
        derive(4, volume=1.0, extent=1.0, dt=1e-3, young=1e5, poisson=0.2)
    with pytest.raises(ValueError):
        derive(100, volume=0.0, extent=1.0, dt=1e-3, young=1e5, poisson=0.2)


def test_hole_risk_visible_in_measurement():
    rng = np.random.default_rng(1)
    x = rng.uniform(-2, 2, (300, 3)).astype(np.float32)     # sparse: ppc ~ 0.1 at dx 0.5
    m = measure_ppc(x, 0.5, -16.0)
    assert m["frac_cells_below_4ppc"] > 0.9
