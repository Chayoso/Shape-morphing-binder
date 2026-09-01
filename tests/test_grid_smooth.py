"""Sobolev/grid-GS smoothing of per-particle fields (docs/method.md §6). Torch CPU."""
import numpy as np
import torch

from physmorph.pipeline.grid_smooth import smooth_particle_field

GMIN = torch.tensor([-2.0, -2.0, -2.0])
DX, DIMS = 0.25, (16, 16, 16)


def _cloud(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.uniform(-1.5, 1.5, (n, 3)).astype(np.float32))


def test_iters_zero_is_identity():
    x = _cloud()
    g = torch.randn(len(x), 3)
    assert smooth_particle_field(x, g, GMIN, DX, DIMS, iters=0) is g


def test_constant_field_direction_preserved():
    x = _cloud()
    g = torch.tensor([1.0, 0.0, 0.0]).expand(len(x), 3).clone()
    gs = smooth_particle_field(x, g, GMIN, DX, DIMS, iters=15)
    cos = torch.nn.functional.cosine_similarity(gs.mean(0, keepdim=True),
                                                g.mean(0, keepdim=True)).item()
    assert cos > 0.99


def test_norm_preserved_exactly():
    x = _cloud()
    g = torch.randn(len(x), 3)
    gs = smooth_particle_field(x, g, GMIN, DX, DIMS, iters=10)
    assert abs(gs.norm().item() - g.norm().item()) < 1e-4 * g.norm().item()


def test_zero_pocket_is_filled():
    """The point of the preconditioner: particles with zero raw gradient inside a
    uniformly-pulled body must inherit a consistent direction from their neighbours."""
    x = _cloud(4000)
    g = torch.tensor([0.0, 1.0, 0.0]).expand(len(x), 3).clone()
    pocket = x.norm(dim=1) < 0.4
    assert pocket.sum() > 20
    g[pocket] = 0.0
    gs = smooth_particle_field(x, g, GMIN, DX, DIMS, iters=25)
    filled = gs[pocket]
    assert float(filled.norm(dim=1).mean()) > 0.05          # no longer zero
    cos = torch.nn.functional.cosine_similarity(
        filled, torch.tensor([0.0, 1.0, 0.0]).expand(len(filled), 3)).mean().item()
    assert cos > 0.9                                        # aligned with the body pull
