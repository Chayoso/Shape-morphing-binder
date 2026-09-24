"""The support-preserving projection of a paced step (physmorph/losses/projection.py).

A translation and a rotation of a ball are divergence-free and pass through unchanged; a radial
expansion is a pure gradient inside the ball and is removed there (p = 0 on the free surface
leaves the interior step near zero); the discrete divergence after the projection is a small
fraction of the one before.
"""
import numpy as np
import torch

from physmorph.losses.projection import project_step

DX = 0.3
DIMS = (28, 28, 28)


def _ball(n=30000, r=2.0, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(-r, r, (3 * n, 3)).astype(np.float32)
    p = p[np.linalg.norm(p, axis=1) <= r][:n]
    x = torch.as_tensor(p) + 0.5 * DX * DIMS[0]        # the ball at the grid's centre
    gmin = torch.zeros(3)
    return x, gmin


def _rel_change(d, d2):
    return float(((d2 - d).norm(dim=1) / d.norm(dim=1).clamp_min(1e-9)).median())


def test_translation_unchanged():
    x, gmin = _ball()
    d = torch.zeros_like(x)
    d[:, 0] = DX
    d2, st = project_step(x, d, 1.0, gmin, DX, DIMS)
    assert st["body"] > 1000
    assert _rel_change(d, d2) < 0.03, st


def test_rotation_unchanged():
    x, gmin = _ball()
    c = x.mean(0)
    d = torch.zeros_like(x)
    d[:, 0] = -0.1 * (x[:, 1] - c[1])
    d[:, 1] = 0.1 * (x[:, 0] - c[0])
    d2, st = project_step(x, d, 1.0, gmin, DX, DIMS)
    assert _rel_change(d, d2) < 0.05, st


def test_radial_expansion_removed_inside():
    x, gmin = _ball()
    c = x.mean(0)
    d = 0.1 * (x - c)
    d2, st = project_step(x, d, 1.0, gmin, DX, DIMS)
    inner = (x - c).norm(dim=1) < 1.2
    ratio = float(d2[inner].norm(dim=1).median() / d[inner].norm(dim=1).median())
    assert ratio < 0.2, (ratio, st)
    assert st["div1"] < 0.15 * st["div0"], st
    assert st["cg_res"] < 1e-3, st
