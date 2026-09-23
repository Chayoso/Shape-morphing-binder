import numpy as np

from physmorph.render.support import MaterialSupport


def _lattice(n=3):
    a = np.linspace(-0.2, 0.2, n, dtype=np.float32)
    return np.stack(np.meshgrid(a, a, a, indexing="ij"), -1).reshape(-1, 3)


def test_rigid_motion_and_uniform_scaling_keep_all_primitives_visible():
    rest = _lattice()
    graph = MaterialSupport.from_rest(rest)
    x = 1.7 * rest + np.array([3.0, -2.0, 0.5], np.float32)
    np.testing.assert_allclose(graph.opacity(x), 1.0)


def test_only_a_materially_disconnected_singleton_fades():
    rest = _lattice()
    graph = MaterialSupport.from_rest(rest)
    x = rest.copy()
    p = len(x) // 2
    x[p] += np.array([4.0, 0.0, 0.0], np.float32)
    alpha = graph.opacity(x)
    assert alpha[p] == 0.0
    assert int((alpha < 0.5).sum()) == 1


def test_coherent_material_patch_is_not_hidden_even_when_far_from_the_body():
    rest = _lattice(4)
    graph = MaterialSupport.from_rest(rest)
    x = rest.copy()
    patch = np.flatnonzero(rest[:, 0] > 0.15)
    x[patch] += np.array([4.0, 0.0, 0.0], np.float32)
    alpha = graph.opacity(x)
    retained = [p for p in patch if np.isin(graph.neighbor[p], patch).sum() >= 2]
    assert retained and np.all(alpha[retained] == 1.0)


def test_opacity_is_bounded_monotone_and_does_not_mutate_positions():
    rest = _lattice()
    graph = MaterialSupport.from_rest(rest)
    p = len(rest) // 2
    vals = []
    for d in np.linspace(0.0, 4.0, 41):
        x = rest.copy(); x[p, 0] += d
        before = x.copy()
        a = graph.opacity(x)
        assert np.all((0.0 <= a) & (a <= 1.0))
        np.testing.assert_array_equal(x, before)
        vals.append(float(a[p]))
    assert all(a + 1e-6 >= b for a, b in zip(vals, vals[1:]))
    assert vals[0] == 1.0 and vals[-1] == 0.0
