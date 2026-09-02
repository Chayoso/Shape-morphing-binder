"""Plastic stress-chain and angular-momentum regressions on the Warp CPU device."""
import numpy as np
import warp as wp

from physmorph.mpm import kernels as K


DEV = "cpu"
LAM, MU = 2.0, 3.0


def _proper_rotation(F):
    U, _, Vt = np.linalg.svd(F)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def _pk1_elastic(Fe):
    J = np.linalg.det(Fe)
    return (2.0 * MU * (Fe - _proper_rotation(Fe))
            + LAM * (J - 1.0) * J * np.linalg.inv(Fe).T)


def _psi(Fe):
    sig = np.linalg.svd(Fe, compute_uv=False)
    J = np.linalg.det(Fe)
    return MU * np.square(sig - 1.0).sum() + 0.5 * LAM * (J - 1.0) ** 2


def _kernel_total_pk1(Fbar, Fp):
    F = wp.array(np.ascontiguousarray(Fbar[None], np.float32), dtype=wp.mat33, device=DEV)
    dFc = wp.zeros(1, dtype=wp.mat33, device=DEV)
    Fp_wp = wp.array(np.ascontiguousarray(Fp[None], np.float32), dtype=wp.mat33,
                     device=DEV)
    lam = wp.array(np.array([LAM], np.float32), dtype=wp.float32, device=DEV)
    mu = wp.array(np.array([MU], np.float32), dtype=wp.float32, device=DEV)
    P = wp.zeros(1, dtype=wp.mat33, device=DEV)
    wp.launch(K.k_stress, dim=1, inputs=[F, dFc, Fp_wp, lam, mu, P], device=DEV)
    return P.numpy()[0]


def _b3(q):
    a = abs(float(q))
    if a < 1.0:
        return 0.5 * a ** 3 - a ** 2 + 2.0 / 3.0
    if a < 2.0:
        return (2.0 - a) ** 3 / 6.0
    return 0.0


def _grid_force_moments(tau, x):
    """Cubic 4^3 P2G moments with dx=1, C0=3, dt*Vp factored to one."""
    force = np.zeros(3)
    torque = np.zeros(3)
    base = np.floor(x).astype(int) - 1
    for oi in range(4):
        for oj in range(4):
            for ok in range(4):
                node = base + np.array([oi, oj, ok])
                dgp = node - x
                w = _b3(dgp[0]) * _b3(dgp[1]) * _b3(dgp[2])
                fg = -3.0 * w * (tau @ dgp)
                force += fg
                torque += np.cross(dgp, fg)
    return force, torque


def test_fp_identity_preserves_legacy_stress():
    Fbar = np.array([[1.16, 0.08, -0.03],
                     [0.02, 0.91, 0.04],
                     [0.01, -0.05, 1.07]])
    got = _kernel_total_pk1(Fbar, np.eye(3))
    assert np.allclose(got, _pk1_elastic(Fbar), rtol=2e-5, atol=2e-5)


def test_total_pk1_matches_elastoplastic_energy_directional_derivative():
    Fe = np.diag([1.2, 0.9, 1.1])
    Fp = np.array([[1.0, 0.55, 0.0],
                   [0.0, 1.0, 0.25],
                   [0.0, 0.0, 1.0]])
    Fbar = Fe @ Fp
    H = np.array([[0.2, -0.3, 0.1],
                  [0.4, 0.05, -0.2],
                  [0.1, 0.25, -0.15]])
    Ptotal = _kernel_total_pk1(Fbar, Fp).astype(np.float64)
    Fpi = np.linalg.inv(Fp)
    eps = 1.0e-4
    fd = (_psi((Fbar + eps * H) @ Fpi) - _psi((Fbar - eps * H) @ Fpi)) / (2 * eps)
    assert np.isclose(np.sum(Ptotal * H), fd, rtol=2e-5, atol=2e-6)


def test_plastic_stress_is_symmetric_and_has_zero_discrete_internal_torque():
    Fe = np.diag([1.2, 0.9, 1.1])
    Fp = np.array([[1.0, 0.55, 0.0],
                   [0.0, 1.0, 0.25],
                   [0.0, 0.0, 1.0]])
    Fbar = Fe @ Fp
    Pe = _pk1_elastic(Fe)
    tau = _kernel_total_pk1(Fbar, Fp).astype(np.float64) @ Fbar.T
    tau_wrong = Pe @ Fbar.T

    assert np.linalg.norm(tau - tau.T) < 2e-5
    assert np.linalg.norm(tau_wrong - tau_wrong.T) > 1e-2  # nontrivial regression case

    force, torque = _grid_force_moments(tau, np.array([4.2, 4.35, 4.1]))
    _, torque_wrong = _grid_force_moments(tau_wrong, np.array([4.2, 4.35, 4.1]))
    assert np.linalg.norm(force) < 1e-12
    assert np.linalg.norm(torque) < 2e-5
    assert np.linalg.norm(torque_wrong) > 1e-2
