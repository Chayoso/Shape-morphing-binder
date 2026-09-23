"""Sobolev (H1) descent direction on the material kNN graph: converges, preserves the norm,
and removes sub-stencil disagreement between material neighbours (a lone spike is spread
over its neighbourhood) while leaving a graph-smooth field unchanged."""
import numpy as np
import torch
from scipy.spatial import cKDTree

from physmorph.pipeline.optimizer import _sobolev_direction


def _graph(n=3000, k=8, seed=2):
    x = np.random.default_rng(seed).uniform(-1, 1, (n, 3)).astype(np.float32)
    knn = cKDTree(x).query(x, k=k + 1)[1][:, 1:]
    return x, torch.as_tensor(np.ascontiguousarray(knn))


def test_spike_is_spread_and_norm_kept():
    x, knn = _graph()
    T, N = 2, len(x)
    g = torch.zeros(T, N, 3, 3)
    g[:, 0, 0, 0] = 1.0                                      # one particle pushed alone
    u = _sobolev_direction(g, knn, kappa=2.0)
    assert abs(float(u.norm()) - float(g.norm())) < 1e-4
    assert float(u[0, 0, 0, 0]) < 0.9 * float(g.norm())      # the spike is shared out
    nb = knn[0]
    assert float(u[0, nb, 0, 0].abs().mean()) > 0.0          # its neighbours now move too
    far = torch.ones(N, dtype=torch.bool); far[0] = False; far[nb] = False
    assert float(u[0, far, 0, 0].abs().max()) < float(u[0, 0, 0, 0])   # decays with graph distance


def test_smooth_field_is_a_fixed_point():
    x, knn = _graph()
    N = len(x)
    g = torch.as_tensor(x[:, :1] * 0.3 + 1.0).view(1, N, 1, 1).expand(1, N, 3, 3).contiguous()  # linear in x
    u = _sobolev_direction(g, knn, kappa=2.0)
    rel = float((u - g).norm() / g.norm())
    assert rel < 0.05                                        # a graph-smooth field passes through
