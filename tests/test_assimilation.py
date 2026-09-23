"""Objectivity + exactness of commit-time elastic-stretch assimilation (§3.5).

The rejected variants (OT update_fp: rotation fabricated strain, dilation invisible;
displacement-polar: mismatched the dFc-inflated F) live in git history with their probes.
"""
import numpy as np

from physmorph.plasticity import assimilate_elastic


def _rot_y(th):
    return np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0],
                     [-np.sin(th), 0, np.cos(th)]], np.float32)


def _id(n):
    return np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))


def test_exact_stretch_power():
    """F_e_new = R_e S_e^{1-eta} EXACTLY (the property the runner relies on)."""
    R = _rot_y(0.7)
    S = np.diag([1.6, 1.0, 0.7]).astype(np.float32)
    F = np.tile((R @ S).astype(np.float32), (8, 1, 1))
    Fp = assimilate_elastic(F, _id(8), eta=0.4)
    Fe_new = np.einsum("nij,njk->nik", F, np.linalg.inv(Fp))
    sv = np.sort(np.linalg.svd(Fe_new, compute_uv=False), axis=1)
    assert np.allclose(sv, np.sort(np.diag(S) ** 0.6), atol=1e-4)


def test_rigid_rotation_is_a_strict_noop():
    F = np.tile(_rot_y(1.2), (6, 1, 1))
    Fp = assimilate_elastic(F, _id(6), eta=0.7)
    assert np.abs(Fp - np.eye(3)).max() < 1e-5


def test_dilation_is_assimilated():
    """Volumetric deformation must NOT be invisible (the v1 isochoric no-op)."""
    F = np.tile(np.diag([1.3, 1.3, 1.3]).astype(np.float32), (4, 1, 1))
    Fp = assimilate_elastic(F, _id(4), eta=1.0)
    assert np.allclose(Fp[:, [0, 1, 2], [0, 1, 2]], 1.3, atol=1e-4)


def test_elastic_energy_never_increases():
    """Fixed-corotated psi must not increase at a commit (the anti-spring-back claim)."""
    def psi(Fe, lam=1.0, mu=1.0):
        S = np.linalg.svd(Fe, compute_uv=False)
        J = np.linalg.det(Fe)
        return mu * ((S - 1) ** 2).sum(1) + 0.5 * lam * (J - 1) ** 2
    rng = np.random.default_rng(6)
    F = (rng.normal(0, 0.25, (50, 3, 3)) + np.eye(3)).astype(np.float32)
    F = F[np.linalg.det(F) > 1e-3]
    Fp = assimilate_elastic(F, _id(len(F)), eta=0.5)
    Fe_new = np.einsum("nij,njk->nik", F, np.linalg.inv(Fp))
    assert (psi(Fe_new) <= psi(F) + 1e-5).all()


def test_inverted_fe_rows_are_skipped():
    """det(F_e) <= 0 must surface in the F guards, not be absorbed into Fp."""
    F = np.stack([np.diag([-1.0, 1.0, 1.0]).astype(np.float32),
                  np.diag([1.5, 1.0, 1.0]).astype(np.float32)])
    Fp = assimilate_elastic(F, _id(2), eta=1.0)
    assert np.allclose(Fp[0], np.eye(3))              # inverted row untouched
    assert abs(Fp[1, 0, 0] - 1.5) < 1e-4              # healthy row assimilated


def test_cumulative_band_honoured():
    F = np.tile(np.diag([9.0, 1.0, 0.05]).astype(np.float32), (3, 1, 1))
    Fp = assimilate_elastic(F, _id(3), eta=1.0, smin=0.2, smax=5.0)
    S = np.linalg.svd(Fp, compute_uv=False)
    assert S.min() >= 0.199 and S.max() <= 5.001


def test_eta_zero_is_identity():
    Fp0 = _id(5)
    F = np.tile(np.diag([1.4, 1.0, 0.8]).astype(np.float32), (5, 1, 1))
    assert np.array_equal(assimilate_elastic(F, Fp0, eta=0.0), Fp0)


def test_isochoric_assimilation_preserves_fp_volume():
    """assim_iso: the plastic increment is det-free, so det(Fp) stays 1 no matter how
    compressed F_e is — the volume ratchet (|J-1|>0.3 on 34% by 120 commits) cannot
    form. The band clamp may still alter det slightly; assert near-1 over many rounds."""
    import numpy as np
    from physmorph.plasticity import assimilate_elastic
    rng = np.random.default_rng(4)
    n = 200
    Fp = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    for _ in range(12):                          # 12 commits of heavy squeezing
        A = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
        A += 0.2 * rng.normal(size=(n, 3, 3)).astype(np.float32)
        A[:, 0, 0] *= 0.7                        # persistent compression channel
        F = np.einsum("nij,njk->nik", A, Fp)
        Fp = assimilate_elastic(F, Fp, eta=0.5, isochoric=True)
    dets = np.linalg.det(Fp)
    assert np.abs(dets - 1.0).max() < 0.05
    # sanity: the unnormalised form DOES ratchet volume under the same forcing
    Fp2 = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    rng = np.random.default_rng(4)
    for _ in range(12):
        A = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
        A += 0.2 * rng.normal(size=(n, 3, 3)).astype(np.float32)
        A[:, 0, 0] *= 0.7
        F = np.einsum("nij,njk->nik", A, Fp2)
        Fp2 = assimilate_elastic(F, Fp2, eta=0.5, isochoric=False)
    assert np.abs(np.linalg.det(Fp2) - 1.0).max() > 0.3


def test_growth_channel_demand_driven_and_governed():
    """Growth (morphoelastic det command): demand grows rest volume per commit, zero
    demand leaves det untouched, and the cumulative governor caps at grow_band."""
    import numpy as np
    from physmorph.plasticity.assimilation import assimilate_growth
    n = 8
    Fp = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    F = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    grow = np.ones(n, np.float32)
    grow[:4] = 1.05                                # 5% commanded growth on half
    for _ in range(20):
        Fp = assimilate_growth(F, Fp, eta=0.5, isochoric=True,
                               grow=grow, grow_band=1.5)
        F = Fp.copy()                              # keep F_e = I (pure rest tracking)
    det = np.linalg.det(Fp)
    assert np.all(det[4:] < 1.02) and np.all(det[4:] > 0.98)   # no demand: no growth
    assert np.all(det[:4] > 1.3) and np.all(det[:4] < 1.55)    # grown, governor-capped
