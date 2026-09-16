"""Conservative particle resampling (runner.reattach_fragments): a cluster detached from the
body (its cells not connected to the body on the dilated occupancy) is merged onto the nearest
body particles — afterwards the fragment mask is empty, the particle count is unchanged, the
merged particles carry their neighbour's state, and a connected cloud is left untouched."""
import numpy as np

from physmorph.mpm.state import MPMParams
from physmorph.pipeline.runner import fragment_mask, reattach_fragments


def _prm():
    return MPMParams(dx=0.5, dt=1.0 / 240.0, drag=0.0, smoothing=1.0,
                     grid_min=(-6.0, -6.0, -6.0), nx=24, ny=24, nz=24)


def _state(x, seed=0):
    rng = np.random.default_rng(seed)
    N = len(x)
    v = rng.normal(size=(N, 3)).astype(np.float32)
    C = rng.normal(size=(N, 3, 3)).astype(np.float32)
    F = (np.eye(3, dtype=np.float32) + 0.1 * rng.normal(size=(N, 3, 3))).astype(np.float32)
    Fp = np.repeat(np.eye(3, dtype=np.float32)[None], N, 0)
    Fg = F.copy()
    return v, C, F, Fp, Fg


def test_detached_cluster_is_merged_and_mass_conserved():
    prm = _prm()
    body = np.random.default_rng(1).uniform(-1.0, 1.0, (600, 3)).astype(np.float32)
    debris = body[:5].copy(); debris[:, 0] += 5.0            # far cluster (> 2 cells past the body)
    x = np.concatenate([body, debris]).astype(np.float32)
    v, C, F, Fp, Fg = _state(x)
    F_body_before = F[:600].copy()
    assert fragment_mask(x, prm)[600:].all()
    from scipy.spatial import cKDTree
    _, j = cKDTree(x[:600]).query(x[600:], k=1)                # merge targets (before the merge)
    n = reattach_fragments(x, v, C, F, Fp, Fg, prm, spacing=0.2, seed=3)
    assert n == 5 and len(x) == 605
    assert not fragment_mask(x, prm).any()
    # merged particles sit half a spacing from their target and carry its state
    assert np.allclose(np.linalg.norm(x[600:] - x[:600][j], axis=1), 0.1, atol=1e-5)
    assert np.allclose(F[600:], F[:600][j]) and np.allclose(v[600:], v[:600][j])
    assert np.array_equal(F[:600], F_body_before)              # the body is untouched


def test_connected_cloud_is_untouched():
    prm = _prm()
    x = np.random.default_rng(2).uniform(-1.0, 1.0, (800, 3)).astype(np.float32)
    x0 = x.copy()
    v, C, F, Fp, Fg = _state(x)
    assert reattach_fragments(x, v, C, F, Fp, Fg, prm, spacing=0.2) == 0
    assert np.array_equal(x, x0)
