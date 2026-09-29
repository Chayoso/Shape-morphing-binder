"""Constitutive model + interpolation kernels (Warp device functions).

All `@wp.func` here are called from the MPM kernels. Equations refer to docs/SPEC.md.
Cubic B-spline matches DiffMPMLib3D/Interpolation.h exactly (oracle).
PK1 fixed-corotated matches DiffMPMLib3D/Elasticity.cpp:54 — eq (3).
"""
from __future__ import annotations

import warp as wp


# ── Cubic B-spline (oracle: Interpolation.h) ────────────────────────────────
@wp.func
def bspline_w(x: float) -> float:
    ax = wp.abs(x)
    if ax < 1.0:
        return 0.5 * ax * ax * ax - ax * ax + 2.0 / 3.0
    if ax < 2.0:
        t = 2.0 - ax
        return t * t * t / 6.0
    return 0.0


@wp.func
def bspline_dw(x: float) -> float:
    """d/dx of cubic B-spline (oracle: CubicBSplineSlope)."""
    ax = wp.abs(x)
    if ax < 1.0:
        return 1.5 * x * ax - 2.0 * x
    if ax < 2.0:
        return -x * ax / 2.0 + 2.0 * x - 2.0 * x / ax
    return 0.0


@wp.func
def weight(dgp: wp.vec3, inv_dx: float) -> float:
    """Tensor-product cubic B-spline weight w_gp."""
    return (
        bspline_w(dgp[0] * inv_dx)
        * bspline_w(dgp[1] * inv_dx)
        * bspline_w(dgp[2] * inv_dx)
    )


# ── Fixed-corotated elasticity — eq (1)(2)(3) ───────────────────────────────
def lame(young: float, poisson: float) -> tuple[float, float]:
    """Lamé parameters — eq (1). (host-side; Elasticity.cpp:151)"""
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    return float(lam), float(mu)


@wp.func
def corotated_R(F: wp.mat33) -> wp.mat33:
    """Rotation from signed SVD (proper, det=+1). Handles inversion."""
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    wp.svd3(F, U, sig, V)
    R = U @ wp.transpose(V)
    # ensure proper rotation (flip if reflection)
    if wp.determinant(R) < 0.0:
        U2 = wp.mat33(
            U[0, 0], U[0, 1], -U[0, 2],
            U[1, 0], U[1, 1], -U[1, 2],
            U[2, 0], U[2, 1], -U[2, 2],
        )
        R = U2 @ wp.transpose(V)
    return R


@wp.func
def pk1_fixed_corotated(F: wp.mat33, lam: float, mu: float) -> wp.mat33:
    """1st Piola–Kirchhoff stress — eq (3). P = 2mu(F-R) + lam(J-1)J F^{-T}."""
    R = corotated_R(F)
    J = wp.determinant(F)
    Jc = wp.max(J, 1.0e-6)
    Fit = wp.transpose(wp.inverse(F))
    return 2.0 * mu * (F - R) + lam * (Jc - 1.0) * Jc * Fit


@wp.func
def polar_R(F: wp.mat33) -> wp.mat33:
    """Same forward rotation; a polar-factor VJP for nonsingular det(F)>0."""
    return corotated_R(F)


@wp.func_grad(polar_R)
def adj_polar_R(F: wp.mat33, adj_R: wp.mat33):
    R = corotated_R(F)
    S0 = wp.transpose(R) @ F
    S = 0.5 * (S0 + wp.transpose(S0))
    H = wp.transpose(R) @ adj_R - wp.transpose(adj_R) @ R
    # S*Omega + Omega*S = H. In 3D its axial form is a 3x3
    # solve with eigenvalues sigma_i + sigma_j, never their differences.
    tr = wp.trace(S)
    A = wp.mat33(tr, 0.0, 0.0, 0.0, tr, 0.0, 0.0, 0.0, tr) - S
    w = wp.inverse(A) @ wp.vec3(H[2, 1], H[0, 2], H[1, 0])
    Omega = wp.mat33(0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0)
    wp.adjoint[F] += R @ Omega


@wp.func
def pk1_fixed_corotated_polar(F: wp.mat33, lam: float, mu: float) -> wp.mat33:
    # Inverted trials retain the signed-SVD branch and are rejected by the
    # trajectory guard; the positive-definite polar derivative is not used there.
    R = wp.mat33()
    J = wp.determinant(F)
    if J > 0.0:
        R = polar_R(F)
    else:
        R = corotated_R(F)
    Jc = wp.max(J, 1.0e-6)
    Fit = wp.transpose(wp.inverse(F))
    return 2.0 * mu * (F - R) + lam * (Jc - 1.0) * Jc * Fit


@wp.func
def psi_fixed_corotated(F: wp.mat33, lam: float, mu: float) -> float:
    """Elastic energy density — eq (2). mu*sum(sig-1)^2 + 0.5 lam (J-1)^2."""
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    wp.svd3(F, U, sig, V)
    J = wp.determinant(F)
    s = (sig[0] - 1.0) * (sig[0] - 1.0) + (sig[1] - 1.0) * (sig[1] - 1.0) + (sig[2] - 1.0) * (sig[2] - 1.0)
    return mu * s + 0.5 * lam * (J - 1.0) * (J - 1.0)
