"""The GPU k-NN (render/knn_gpu.knn_self) returns the rows of cKDTree.query(x, k): self first,
distances ascending, the same neighbour sets up to float32 ties — on a uniform cloud, on a
surface-heavy cloud (a thin shell plus a sparse interior: the layer's case) and for the k values
the pipeline uses (9, 25, 33, 64). Skipped without CUDA."""
import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from physmorph.render.knn_gpu import knn_self

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _check(x, k):
    d_ref, i_ref = cKDTree(x).query(x, k=k, workers=-1)
    d, i = knn_self(x, k)
    assert d.shape == (len(x), k) and i.shape == (len(x), k)
    assert np.all(i[:, 0] == np.arange(len(x))), "self is column 0"
    assert np.all(np.diff(d, axis=1) >= -1e-6), "ascending distances"
    assert np.allclose(d, d_ref, rtol=1e-5, atol=1e-6), float(np.abs(d - d_ref).max())
    # the neighbour SETS agree except at float32 ties (the same distance to 1e-6)
    same = (i == i_ref).mean()
    assert same > 0.999, same


def test_uniform_cloud():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, (30000, 3)).astype(np.float32)
    for k in (9, 25, 33):
        _check(x, k)


def test_shell_and_sparse_interior():
    rng = np.random.default_rng(1)
    n_shell, n_in = 24000, 6000
    u = rng.normal(size=(n_shell, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    shell = u * (1.0 + 0.02 * rng.normal(size=(n_shell, 1)))
    inner = rng.uniform(-0.6, 0.6, (n_in, 3))
    x = np.concatenate([shell, inner]).astype(np.float32)
    for k in (9, 33, 64):
        _check(x, k)


def test_small_cloud_falls_back_to_scipy():
    x = np.random.default_rng(2).uniform(size=(500, 3)).astype(np.float32)
    d, i = knn_self(x, 9)
    d_ref, i_ref = cKDTree(x).query(x, k=9)
    assert np.allclose(d, d_ref) and np.all(i == i_ref)
