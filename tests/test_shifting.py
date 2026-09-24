"""Fickian particle shifting (mpm/shifting.py; docs/method.md 10.18), CPU path: (1) a perfect lattice is
a fixed point in the bulk up to the kNN tie-break bias; (2) one step lowers the spacing disorder of a
jittered lattice and of a uniform random cloud; (3) on a slab the free-surface rule keeps the top
layer's shift tangential; (4) no shift exceeds half a spacing."""
import numpy as np

from physmorph.mpm.shifting import fickian_shift, native_spacing


def _lattice(n=12, s=0.2, jitter=0.0, seed=0, layers=None):
    g = np.arange(n) * s
    ny = n if layers is None else layers
    X, Y, Z = np.meshgrid(g, np.arange(ny) * s, g, indexing="ij")
    x = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
    if jitter > 0:
        x += np.random.default_rng(seed).uniform(-jitter, jitter, x.shape).astype(np.float32) * s
    return x


def _nn_cv(x, k=1):
    from scipy.spatial import cKDTree
    d = cKDTree(x).query(x, k=k + 1, workers=-1)[0][:, 1:]
    return float(d.std() / d.mean())


def test_lattice_is_a_fixed_point_in_the_bulk():
    s = 0.2
    x = _lattice(12, s)
    dx, st = fickian_shift(x, s)
    inner = (np.abs(x - x.mean(0)) < 3.5 * s).all(1)      # away from the faces
    assert np.abs(dx[inner]).max() < 0.02 * s              # the kNN tie-break bias on exact ties
    assert st["max_sp"] <= 0.5 + 1e-6


def test_one_step_orders_a_jittered_lattice_and_a_random_cloud():
    s = 0.2
    for x in (_lattice(12, s, jitter=0.3, seed=1),
              np.random.default_rng(3).uniform(0, 12 * s, (12 ** 3, 3)).astype(np.float32)):
        sp = native_spacing(x)
        before1, before8 = _nn_cv(x, 1), _nn_cv(x, 8)
        dx, st = fickian_shift(x, sp)
        y = x + dx
        assert _nn_cv(y, 1) < 0.8 * before1 and _nn_cv(y, 8) < before8
        assert 0.0 < st["median_sp"] <= 0.5 and st["max_sp"] <= 0.5 + 1e-6
        _, st2 = fickian_shift(y, sp)
        assert st2["disorder"] < st["disorder"]


def test_free_surface_rule_keeps_the_top_layer_tangential():
    s = 0.2
    x = _lattice(12, s, jitter=0.2, seed=2, layers=6)
    dx, st = fickian_shift(x, s)
    top = x[:, 1] > x[:, 1].max() - 0.4 * s
    assert top.sum() > 50 and st["n_surface"] >= top.sum()
    normal = np.abs(dx[top][:, 1])
    assert np.median(normal) < 0.05 * s
    assert normal.max() < 0.25 * s
