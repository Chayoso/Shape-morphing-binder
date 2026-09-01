"""Objectivity of commit-time plastic assimilation (docs/pipeline_v2.md §3.5).

These are the adversarial round-1 probes made permanent: the v1 primitive fabricated
strain from rigid rotation and ignored dilation entirely.
"""
import numpy as np
import pytest

from physmorph.plasticity import assimilate_elastic, assimilate_fp
from physmorph.plasticity.sinkhorn import displacement_jacobian


@pytest.fixture
def cloud():
    rng = np.random.default_rng(0)
    return rng.uniform(-1, 1, (400, 3)).astype(np.float32)


def _rot_y(th):
    return np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0],
                     [-np.sin(th), 0, np.cos(th)]], np.float32)


def _id(n):
    return np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))


def test_rigid_rotation_leaves_fp_unchanged(cloud):
    """A stress-free motion must not create plastic strain (objectivity)."""
    R = _rot_y(np.pi / 2)
    d = cloud @ R.T - cloud
    Fp = assimilate_fp(_id(len(cloud)), cloud, d, eta=0.5, k=12)
    assert np.abs(Fp - np.eye(3)).max() < 0.05


def test_dilation_is_assimilated(cloud):
    """The volumetric mode D_vol drives must NOT be invisible (v1 isochoric no-op)."""
    Fp = assimilate_fp(_id(len(cloud)), cloud, 0.3 * cloud, eta=1.0, k=12)
    diag = Fp[:, [0, 1, 2], [0, 1, 2]].mean()
    assert abs(diag - 1.3) < 0.05


def test_rotation_plus_stretch_assimilates_stretch_only(cloud):
    A = _rot_y(np.pi / 2) @ np.diag([1.2, 1.0, 0.9]).astype(np.float32)
    d = cloud @ A.T - cloud
    Fp = assimilate_fp(_id(len(cloud)), cloud, d, eta=1.0, k=12)
    sv = np.sort(np.linalg.svd(Fp, compute_uv=False).mean(0))
    assert np.allclose(sv, [0.9, 1.0, 1.2], atol=0.05)


def test_partial_eta_is_stretch_power(cloud):
    A = _rot_y(np.pi / 2) @ np.diag([1.2, 1.0, 0.9]).astype(np.float32)
    d = cloud @ A.T - cloud
    Fp = assimilate_fp(_id(len(cloud)), cloud, d, eta=0.5, k=12)
    sv = np.sort(np.linalg.svd(Fp, compute_uv=False).mean(0))
    assert np.allclose(sv, [np.sqrt(0.9), 1.0, np.sqrt(1.2)], atol=0.05)


def test_band_clamp_applied_last(cloud):
    """v1 clamped BEFORE the isochoric renorm, so the result violated the band."""
    d = cloud * np.array([9.0, 9.0, -0.95], np.float32)
    Fp = assimilate_fp(_id(len(cloud)), cloud, d, eta=1.0, k=12, smin=0.2, smax=5.0)
    S = np.linalg.svd(Fp, compute_uv=False)
    assert S.min() >= 0.199 and S.max() <= 5.001


def test_inverted_local_map_is_skipped(cloud):
    """det(A) <= 0 rows must surface in the F guards, not be absorbed into Fp."""
    Fp = assimilate_fp(_id(len(cloud)), cloud, -1.5 * cloud, eta=1.0, k=12)
    assert np.abs(Fp - np.eye(3)).max() < 1e-5


def test_small_cloud_does_not_crash():
    """k > N-2 crashed v1 (kNN sentinel index). Must degrade gracefully."""
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, (10, 3)).astype(np.float32)
    Fp = assimilate_fp(_id(10), x, 0.1 * x, eta=0.5, k=12)
    assert Fp.shape == (10, 3, 3) and np.isfinite(Fp).all()


def test_eta_zero_is_identity(cloud):
    Fp0 = _id(len(cloud))
    Fp = assimilate_fp(Fp0, cloud, 0.3 * cloud, eta=0.0, k=12)
    assert np.array_equal(Fp, Fp0)


def test_elastic_assim_is_exact_stretch_power():
    """F_e_new = R_e S_e^{1-eta} EXACTLY (the property the runner relies on)."""
    rng = np.random.default_rng(5)
    R = _rot_y(0.7)
    S = np.diag([1.6, 1.0, 0.7]).astype(np.float32)
    F = np.tile((R @ S).astype(np.float32), (8, 1, 1))
    Fp = assimilate_elastic(F, _id(8), eta=0.4)
    Fe_new = np.einsum("nij,njk->nik", F, np.linalg.inv(Fp))
    sv = np.sort(np.linalg.svd(Fe_new, compute_uv=False), axis=1)
    expect = np.sort(np.diag(S) ** (1 - 0.4))
    assert np.allclose(sv, expect, atol=1e-4)


def test_elastic_assim_ignores_rotation():
    F = np.tile(_rot_y(1.2), (6, 1, 1))
    Fp = assimilate_elastic(F, _id(6), eta=0.7)
    assert np.abs(Fp - np.eye(3)).max() < 1e-5


def test_elastic_assim_reduces_elastic_energy():
    """Fixed-corotated psi must not increase at a commit (the anti-spring-back claim)."""
    def psi(Fe, lam=1.0, mu=1.0):
        S = np.linalg.svd(Fe, compute_uv=False)
        J = np.linalg.det(Fe)
        return (mu * ((S - 1) ** 2).sum(1) + 0.5 * lam * (J - 1) ** 2)
    rng = np.random.default_rng(6)
    F = (rng.normal(0, 0.25, (50, 3, 3)) + np.eye(3)).astype(np.float32)
    ok = np.linalg.det(F) > 1e-3
    F = F[ok]
    Fp = assimilate_elastic(F, _id(len(F)), eta=0.5)
    Fe_new = np.einsum("nij,njk->nik", F, np.linalg.inv(Fp))
    assert (psi(Fe_new) <= psi(F) + 1e-5).all()


def test_unsymmetrised_jacobian_recovers_affine_field(cloud):
    """The LSQ (no value-space smoothing) must recover a linear field near-exactly;
    the smoothing distorted a 90-degree rotation to |J - J_true| = 0.90."""
    R = _rot_y(np.pi / 2)
    d = cloud @ R.T - cloud
    J = displacement_jacobian(cloud, d, k=12, diffusion_iters=0, symmetrize=False)
    assert np.abs(J - (R - np.eye(3))).max() < 0.05
