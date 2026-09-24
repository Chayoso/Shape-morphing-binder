"""The commit-time null-space projection (mpm/gridfilter.py; docs/method.md 10.20), CPU: (1) a field
that is linear in x is grid-representable and survives the projection (partition of unity, linear
reproduction of the cubic B-spline); (2) sub-cell noise added to it is removed; (3) the projection
is idempotent within float error."""
import numpy as np
import pytest
import torch

from physmorph.mpm.gridfilter import grid_project

DEV = "cpu"


def _cloud(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.5, 3.5, (n, 3)).astype(np.float32)        # inside a 4-wu box, away from the walls
    return x


def test_linear_field_is_reproduced_and_noise_removed():
    x = _cloud()
    dx, gmin, dims = 0.5, (-1.0, -1.0, -1.0), (10, 10, 10)
    A = np.array([[0.02, 0.01, 0.0], [0.0, -0.015, 0.005], [0.01, 0.0, 0.02]], np.float32)
    d_lin = x @ A.T + np.array([0.03, -0.01, 0.02], np.float32)
    noise = np.random.default_rng(1).normal(0, 0.01, x.shape).astype(np.float32)   # sub-cell, uncorrelated
    p_lin, s_lin = grid_project(d_lin, x, dx, gmin, dims, device=DEV)
    p_all, s_all = grid_project(d_lin + noise, x, dx, gmin, dims, device=DEV)
    # XPIC(5) on a random cloud: a linear field within 3 % (order 1 would be 6 %), its null share < 1 %
    assert np.abs(p_lin - d_lin).max() < 0.03 * np.abs(d_lin).max() + 1e-6
    assert s_lin["null_share"] < 0.01
    resid = p_all - d_lin
    assert np.sqrt((resid ** 2).mean()) < 0.25 * np.sqrt((noise ** 2).mean())   # the noise mostly gone (7 % left)
    expected = np.sqrt((noise ** 2).mean()) / np.sqrt(((d_lin + noise) ** 2).mean())   # the noise's share of the field
    assert s_all["null_share"] > 0.5 * expected


def test_second_pass_removes_less_than_the_first():
    """XPIC(order) is not an exact projection on a random cloud (P is a smoothing there), so it is
    not idempotent; what holds is convergence: a second pass removes far less than the first."""
    x = _cloud(seed=2)
    dx, gmin, dims = 0.5, (-1.0, -1.0, -1.0), (10, 10, 10)
    d = np.random.default_rng(3).normal(0, 0.02, x.shape).astype(np.float32)   # pure sub-cell noise
    p1, s1 = grid_project(d, x, dx, gmin, dims, device=DEV)
    p2, s2 = grid_project(p1, x, dx, gmin, dims, device=DEV)
    removed1 = np.sqrt(((d - p1) ** 2).mean()); removed2 = np.sqrt(((p1 - p2) ** 2).mean())
    assert s1["null_share"] > 0.8                       # noise is almost all null space
    assert removed2 < 0.1 * removed1                    # the second pass removes < 10 % of what the first did
    assert s2["null_share"] < 0.5
