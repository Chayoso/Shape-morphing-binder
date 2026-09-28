"""Constitutive adjoints; CPU default, explicit CUDA option for server tests only."""
import os
import numpy as np
import pytest
import warp as wp

from physmorph.mpm.constitutive import corotated_R, pk1_fixed_corotated
from physmorph.mpm.kernels import k_stress

DEVICE = os.environ.get('PHYSMORPH_CONSTITUTIVE_TEST_DEVICE', 'cpu')
if DEVICE not in ('cpu', 'cuda') or (DEVICE == 'cuda' and os.name == 'nt'):
    raise RuntimeError('Constitutive tests use cpu locally; cuda is an explicit server-only option')


@wp.func
def legacy_rotation(F: wp.mat33) -> wp.mat33:
    # Exact pre-change forward operations, deliberately without the new adjoint.
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    wp.svd3(F, U, sig, V)
    R = U @ wp.transpose(V)
    if wp.determinant(R) < 0.0:
        U2 = wp.mat33(U[0, 0], U[0, 1], -U[0, 2],
                     U[1, 0], U[1, 1], -U[1, 2],
                     U[2, 0], U[2, 1], -U[2, 2])
        R = U2 @ wp.transpose(V)
    return R


@wp.kernel
def maps(F: wp.array(dtype=wp.mat33), R: wp.array(dtype=wp.mat33),
         P: wp.array(dtype=wp.mat33), old_R: wp.array(dtype=wp.mat33),
         old_P: wp.array(dtype=wp.mat33)):
    i = wp.tid()
    R[i] = corotated_R(F[i])
    P[i] = pk1_fixed_corotated(F[i], 80.0, 40.0)
    old_R[i] = legacy_rotation(F[i])
    J = wp.determinant(F[i])
    Jc = wp.max(J, 1.0e-6)
    old_P[i] = 80.0 * (F[i]-old_R[i]) + 80.0*(Jc-1.0)*Jc*wp.transpose(wp.inverse(F[i]))


def rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q[:, -1] *= np.linalg.det(q)
    return q


def reference(F, B, lam=80., mu=40.):
    """Independent diagonal Sylvester solution, including signed proper branch."""
    U, s, Vh = np.linalg.svd(F)
    if np.linalg.det(U@Vh) < 0:
        U[:, -1] *= -1
        s[-1] *= -1
    denominator = s[:, None]+s[None, :]
    assert min(abs(denominator[i, j]) for i in range(3) for j in range(i)) > 1e-7
    local = U.T@B@Vh.T
    gR = U@((local-local.T)/denominator)@Vh
    R = U@Vh
    J, A = np.linalg.det(F), np.linalg.inv(F).T
    if J > 1e-6:
        gVol = lam*((2*J-1)*J*np.sum(B*A)*A-J*(J-1)*(A@B.T@A))
    else:
        # The retained PK1 forward clamps Jc; on the distinct inverted branch it
        # is constant and only F^{-T} contributes to this term's derivative.
        gVol = -lam*(1e-6-1)*1e-6*(A@B.T@A)
    return R, gR, 2*mu*(B-gR)+gVol


def evaluate(Fs, seeds=None, kind='rotation'):
    Fs = np.asarray(Fs, np.float32)
    f = wp.array(Fs, dtype=wp.mat33, device=DEVICE, requires_grad=True)
    outputs = [wp.zeros(len(Fs), dtype=wp.mat33, device=DEVICE, requires_grad=True) for _ in range(4)]
    with wp.Tape() as tape:
        wp.launch(maps, len(Fs), inputs=[f, *outputs], device=DEVICE)
    values = [out.numpy().astype(np.float64) for out in outputs]
    if seeds is None:
        return values
    tape.backward(grads={outputs[0 if kind == 'rotation' else 1]:
                         wp.array(np.asarray(seeds, np.float32), dtype=wp.mat33, device=DEVICE)})
    return values, f.grad.numpy().astype(np.float64)


SPECTRA = [(1., 1., 1.), (.7, .7, .7), (1.3, 1.3, 1.3), (1., 1., 1.4),
           (1., 1.00001, 1.00002), (1., 1.001, 1.002), (.7, 1.2, 1.8),
           (-.7, 1.2, 1.8)]


@pytest.mark.parametrize('rotated', [False, True])
@pytest.mark.parametrize('kind', ['rotation', 'stress'])
def test_signed_sylvester_vjp_and_exact_unchanged_forward(rotated, kind):
    rng = np.random.default_rng(300)
    L, R = rotation(rng), rotation(rng)
    Fs = [L@np.diag(s)@R.T if rotated else np.diag(s) for s in SPECTRA]
    Fs = np.asarray(Fs, np.float32).astype(np.float64)
    seeds = rng.normal(size=Fs.shape).astype(np.float32).astype(np.float64)
    values, got = evaluate(Fs, seeds, kind)
    # Forward values are bitwise identical, including the proper negative-det branch.
    np.testing.assert_array_equal(values[0], values[2])
    np.testing.assert_array_equal(values[1], values[3])
    assert np.isfinite(got).all()
    for F, B, actual in zip(Fs, seeds, got):
        _, rot, stress = reference(F, B)
        wanted = rot if kind == 'rotation' else stress
        # Fixed float32 allowance scales with the constitutive coefficients and
        # cotangent; no singular-value-gap regularizer or observed-error fit.
        scale = max(np.linalg.norm(wanted), np.linalg.norm(B)*(1 if kind == 'rotation' else 160), 1.)
        assert np.linalg.norm(actual-wanted) <= 64*np.finfo(np.float32).eps*scale


def test_identity_linear_elastic_tangent_has_no_skew_stiffness():
    skew = np.array([[0., 1., .2], [-1., 0., -.3], [-.2, .3, 0.]])
    symmetric = np.array([[.7, .2, -.1], [.2, -.3, .4], [-.1, .4, .9]])
    seeds = np.asarray([skew, symmetric, symmetric+skew], np.float32).astype(np.float64)
    _, got = evaluate(np.tile(np.eye(3), (3, 1, 1)), seeds, 'stress')
    wanted = np.array([40*(B+B.T)+80*np.trace(B)*np.eye(3) for B in seeds])
    np.testing.assert_allclose(got, wanted, rtol=2e-6, atol=1e-4)
    assert np.linalg.norm(got[0]) < 1e-4  # Prior generic-SVD adjoint error was ~94.


@pytest.mark.parametrize('kind', ['rotation', 'stress'])
def test_centered_forward_differences_include_positive_ties_and_distinct_inversion(kind):
    rng = np.random.default_rng(301)
    L, R = rotation(rng), rotation(rng)
    Fs = np.asarray([np.eye(3), np.diag([1., 1., 1.4]),
                     L@np.diag([1., 1., 1.])@R.T,
                     L@np.diag([-.7, 1.2, 1.8])@R.T], np.float32).astype(np.float64)
    seed = np.array([[0., 1., .2], [-1., 0., -.3], [-.2, .3, 0.]])
    seeds = np.asarray([seed, seed, L@seed@R.T, L@seed@R.T], np.float32).astype(np.float64)
    directions = seeds/np.linalg.norm(seeds, axis=(1, 2))[:, None, None]
    _, gradients = evaluate(Fs, seeds, kind)
    index = 0 if kind == 'rotation' else 1
    for eps in (.01, .005):
        plus, minus = evaluate(Fs+eps*directions)[index], evaluate(Fs-eps*directions)[index]
        fd = np.sum((plus-minus)*seeds, axis=(1, 2))/(2*eps)
        ad = np.sum(gradients*directions, axis=(1, 2))
        # FD includes O(eps^2) truncation and float32 forward SVD error. These
        # local constitutive tests do not alter P297's separate strict tolerances.
        np.testing.assert_allclose(ad, fd, rtol=1e-3, atol=.03 if kind == 'stress' else 5e-4)


def test_material_and_plastic_inverse_chain_still_use_autodiff():
    rng = np.random.default_rng(302)
    F = np.diag([1.1, .9, 1.2]).astype(np.float32)
    Fp = np.array([[1.03, .02, 0.], [0., .97, -.01], [.01, 0., 1.01]], np.float32)
    dc = rng.normal(0., .01, (3, 3)).astype(np.float32)
    seed = rng.normal(size=(3, 3)).astype(np.float32)
    base = [F, dc, Fp, 80., 40.]
    arrays = [wp.array(value[None], dtype=wp.mat33, device=DEVICE, requires_grad=True)
              for value in base[:3]]
    scalars = [wp.array(np.array([value], np.float32), device=DEVICE, requires_grad=True)
               for value in base[3:]]
    p = wp.zeros(1, dtype=wp.mat33, device=DEVICE, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(k_stress, 1, inputs=[*arrays, *scalars, p], device=DEVICE)
    tape.backward(grads={p:wp.array(seed[None], dtype=wp.mat33, device=DEVICE)})
    gradients = [x.grad.numpy()[0].astype(np.float64) for x in arrays+scalars]
    np.testing.assert_array_equal(gradients[0], gradients[1])

    def exact_forward(values):
        f, d, fp, lam, mu = [np.asarray(value, np.float64) for value in values]
        G = np.linalg.inv(fp)
        E = (f+d)@G
        U, _, Vh = np.linalg.svd(E)
        assert np.linalg.det(E) > 1e-6
        R = U@Vh
        J = np.linalg.det(E)
        Pe = 2*mu*(E-R)+lam*(J-1)*J*np.linalg.inv(E).T
        return Pe@G.T

    f, d, fp, lam, mu = [np.asarray(value, np.float64) for value in base]
    B = seed.astype(np.float64)
    G = np.linalg.inv(fp)
    A, E = f+d, (f+d)@G
    barPe = B@G
    R, _, barE = reference(E, barPe, lam, mu)
    J = np.linalg.det(E)
    volume = (J-1)*J*np.linalg.inv(E).T
    Pe = 2*mu*(E-R)+lam*volume
    barA = barE@G.T
    # Fp affects both E=(F+dFc)G and the right factor in P=Pe G^T.
    barG = A.T@barE+B.T@Pe
    expected = [barA, barA.copy(), -G.T@barG@G.T,
                np.asarray(np.sum(barPe*volume)), np.asarray(np.sum(barPe*2*(E-R)))]
    eps32 = np.finfo(np.float32).eps
    forward = exact_forward(base)
    assert np.linalg.norm(p.numpy()[0].astype(np.float64)-forward) <= 64*eps32*max(
        np.linalg.norm(forward), abs(lam)+2*abs(mu), 1.)
    for actual, wanted, coefficient in zip(gradients, expected, (160., 160., 160., 1., 2.)):
        assert np.isfinite(actual).all()
        # Same fixed 64-epsilon coefficient/cotangent allowance as the spectral
        # test above, now applied to every entry of all five chain gradients.
        scale = max(np.linalg.norm(wanted), np.linalg.norm(B)*coefficient, 1.)
        assert np.linalg.norm(actual-wanted) <= 64*eps32*scale

    for channel in range(5):
        direction = rng.normal(size=(3, 3)) if channel < 3 else np.array(1.)
        direction = direction/np.linalg.norm(direction)
        eps = .003 if channel < 3 else .1
        plus, minus = list(base), list(base)
        plus[channel], minus[channel] = base[channel]+eps*direction, base[channel]-eps*direction
        # Validate the analytic chain independently in float64 with the original
        # steps and criterion. The preserved P300 CUDA failure shows that the
        # old float32 forward subtraction at eps=.003 is not this oracle.
        fd = float(np.sum((exact_forward(plus)-exact_forward(minus))*B)/(2*eps))
        ad = float(np.sum(expected[channel]*direction))
        assert ad == pytest.approx(fd, rel=2e-3, abs=.02)


def test_inverted_minimum_tie_is_excluded_because_limits_select_different_rotations():
    # At diag(-1,1,2), the two minimum singular values tie. Distinct perturbations
    # select different proper rotations, so no single derivative is asserted.
    a = np.diag([-1.001, 1., 2.])
    b = np.diag([-.999, 1., 2.])
    rotations = evaluate([a, b])[0]
    assert np.linalg.norm(rotations[0]-rotations[1]) > 2.


def test_exact_singular_sylvester_system_is_not_reported_as_a_zero_derivative():
    # F=0 gives S=0 exactly, regardless of the SVD's arbitrary rotation. This
    # tests the computed-singular guard, not detection of every inverted tie.
    _, gradient = evaluate(np.zeros((1, 3, 3)), np.ones((1, 3, 3)), 'rotation')
    assert np.isnan(gradient).all()
