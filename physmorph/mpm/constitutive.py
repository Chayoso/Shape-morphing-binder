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
def psi_fixed_corotated(F: wp.mat33, lam: float, mu: float) -> float:
    """Elastic energy density — eq (2). mu*sum(sig-1)^2 + 0.5 lam (J-1)^2."""
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    wp.svd3(F, U, sig, V)
    J = wp.determinant(F)
    s = (sig[0] - 1.0) * (sig[0] - 1.0) + (sig[1] - 1.0) * (sig[1] - 1.0) + (sig[2] - 1.0) * (sig[2] - 1.0)
    return mu * s + 0.5 * lam * (J - 1.0) * (J - 1.0)
