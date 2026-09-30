"""The renderers' choice of the archived deformation gradient per frame."""
import numpy as np


def test_select_archive_F_prefers_geometric_when_present():
    from physmorph.render.covariance import select_archive_F
    N = 5
    eye = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
    Fp = np.stack([eye * (1 + 0.1 * k) for k in range(3)])          # physics samples
    Fg = np.stack([eye * (2 + k) for k in range(2)])                # geometric commits
    d = {"F_samples": Fp, "F_sample_idx": np.array([0, 10, 20]),
         "Fg_commits": Fg, "Fg_commit_idx": np.array([11, 21])}     # states at frames 10, 20
    F, kind = select_archive_F(d, 15, prefer_geom=False)
    assert kind == "physics" and np.allclose(F, eye * 1.1)
    F, kind = select_archive_F(d, 15, prefer_geom=True)
    assert kind == "geom" and np.allclose(F, eye * 2.0)
    F, kind = select_archive_F(d, 25, prefer_geom=True)
    assert kind == "geom" and np.allclose(F, eye * 3.0)
    F, kind = select_archive_F(d, 5, prefer_geom=True)              # before the first commit
    assert kind == "physics" and np.allclose(F, eye * 1.0)
    d2 = {"F_samples": Fp, "F_sample_idx": np.array([0, 10, 20])}   # legacy archive
    F, kind = select_archive_F(d2, 25, prefer_geom=True)
    assert kind == "physics" and np.allclose(F, eye * 1.2)
