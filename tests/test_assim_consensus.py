"""Neighbourhood-consensus plastic assimilation: (1) a uniformly stretched cloud is assimilated
exactly as per-particle; (2) one particle stretched 3x inside an unstretched cloud keeps its
stretch ELASTIC (Fp stays ~I) where per-particle assimilation would forgive half of it; (3) a
particle with no neighbour in its stencil gets its own F_e (nothing to compare with)."""
import numpy as np

from physmorph.plasticity.assimilation import assimilate_elastic, consensus_elastic

GMIN, DX, DIMS = (-6.0, -6.0, -6.0), 0.5, (24, 24, 24)


def _cloud(n=2000, seed=1):
    return np.random.default_rng(seed).uniform(-1.0, 1.0, (n, 3)).astype(np.float32)


def _stretch(s):
    return np.diag([s, 1.0, 1.0]).astype(np.float32)


def test_uniform_stretch_matches_per_particle():
    x = _cloud(); N = len(x)
    F = np.repeat(_stretch(1.8)[None], N, 0); Fp = np.repeat(np.eye(3, dtype=np.float32)[None], N, 0)
    Fbar = consensus_elastic(x, F, Fp, GMIN, DX, DIMS, device="cpu")
    assert np.allclose(Fbar, F, atol=1e-5)
    a = assimilate_elastic(F, Fp, eta=0.5, isochoric=False)
    b = assimilate_elastic(F, Fp, eta=0.5, isochoric=False, Fe=Fbar)
    assert np.allclose(a, b, atol=1e-5)


def test_lone_stretched_particle_keeps_its_stretch_elastic():
    x = _cloud(); N = len(x)
    F = np.repeat(np.eye(3, dtype=np.float32)[None], N, 0); F[0] = _stretch(3.0)
    Fp = np.repeat(np.eye(3, dtype=np.float32)[None], N, 0)
    Fbar = consensus_elastic(x, F, Fp, GMIN, DX, DIMS, device="cpu")
    assert abs(Fbar[0, 0, 0] - 1.0) < 0.05                     # its neighbours are not stretched
    per = assimilate_elastic(F, Fp, eta=0.5, isochoric=False)
    con = assimilate_elastic(F, Fp, eta=0.5, isochoric=False, Fe=Fbar)
    assert abs(per[0, 0, 0] - np.sqrt(3.0)) < 1e-3               # per-particle: half forgiven
    assert abs(con[0, 0, 0] - 1.0) < 0.05                        # consensus: stays elastic
    Fe_con = F[0] @ np.linalg.inv(con[0])
    assert Fe_con[0, 0] > 2.8                                    # the 3x stretch is still elastic
    assert np.allclose(con[1:], per[1:], atol=2e-2)              # the others barely change


def test_isolated_particle_uses_its_own_strain():
    x = _cloud(); x[0] = np.array([4.5, 4.5, 4.5], np.float32)  # far from everyone
    N = len(x)
    F = np.repeat(np.eye(3, dtype=np.float32)[None], N, 0); F[0] = _stretch(2.0)
    Fp = np.repeat(np.eye(3, dtype=np.float32)[None], N, 0)
    Fbar = consensus_elastic(x, F, Fp, GMIN, DX, DIMS, device="cpu")
    assert np.allclose(Fbar[0], F[0], atol=1e-6)
