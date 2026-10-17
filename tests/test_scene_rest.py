"""B9b: one body at rest on the floor (scene/spec.SceneSpec.rest) with the gravity given as the single-body runs'
acceleration (SceneSpec.g): no throw, the body about the origin, the floor's plane half a sample pitch below its lowest
particle (where the floor's position constraint holds the centres), the world's f_ext the given g. CPU only."""
import numpy as np
import pytest

from physmorph.pipeline.scene.spec import SceneSpec, layout


def _sphere(n=4000, r=2.4, seed=3):
    rng = np.random.default_rng(seed)
    p = rng.uniform(-r, r, (4 * n, 3))
    p = p[np.linalg.norm(p, axis=1) <= r][:n]
    return p - p.mean(0)


def test_a_body_at_rest_stands_on_the_floor_under_the_given_gravity():
    X = _sphere()
    vp = (4.0 / 3.0) * np.pi * 2.4 ** 3 / len(X)
    spec = SceneSpec(bodies=1, rest=True, g=0.65, mu=0.0, yaw=(0.0, 0.0), seconds=8.33)
    x0, v0, R0, prm, info = layout(spec, [X], [vp], 1.4e5, 0.3, 40000, 0.3, 1.0 / 240.0, 20, radius=2.9)
    assert np.allclose(x0, X, atol=1e-5)                       # about the origin, unturned
    assert not v0.any()                                       # at rest
    assert prm.f_ext == (0.0, -0.65, 0.0) and info["g"] == 0.65
    assert prm.floor_y == pytest.approx(float(x0[:, 1].min()) - 0.5 * vp ** (1.0 / 3.0), abs=1e-6)
    assert prm.floor_friction == 0.0 and prm.drag == 0.0
    lo = np.asarray(prm.grid_min)
    hi = lo + 0.3 * np.array([prm.nx, prm.ny, prm.nz])
    assert (lo < x0.min(0) - 0.29).all() and (hi > x0.max(0) + 0.29).all()   # the body inside, with the margin
    assert lo[1] < prm.floor_y                                # the floor inside the domain
    assert info["windows"] == 50


def test_rest_needs_one_body_and_g_zero_keeps_G():
    X = _sphere(800)
    with pytest.raises(ValueError):
        layout(SceneSpec(bodies=2, rest=True), [X, X], [0.1, 0.1], 1.4e5, 0.3, 40000, 0.3, 1.0 / 240.0, 20)
    _, _, _, prm, info = layout(SceneSpec(bodies=1, G=0.02, yaw=(0.0, 0.0)), [X], [0.1], 1.4e5, 0.3, 40000, 0.3,
                                1.0 / 240.0, 20)
    rho = (40000 / len(X)) / 0.1
    H = float(X[:, 1].max() - X[:, 1].min())
    assert info["g"] == pytest.approx(0.02 * 1.4e5 / (rho * H)) and info["G_of_g"] == pytest.approx(0.02)
