"""condition_F: repair-only semantics in the blessed path (docs/pipeline_v2.md §3.7)."""
import numpy as np

from physmorph.mpm.conditioning import condition_F


def _healthy(n=5, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 0.3, (n, 3, 3)) + np.eye(3)).astype(np.float32)


def test_healthy_f_is_bit_exact_without_clamp():
    F = _healthy()
    out, nb, nf, nc = condition_F(F, clamp=False)
    assert np.array_equal(out, F.reshape(-1, 3, 3))
    assert (nb, nf, nc) == (0, 0, 0)


def test_reflection_is_repaired_and_counted():
    F = _healthy()
    F[0] = np.diag([-3.0, 1.0, 1.0]).astype(np.float32)      # det = -3 (inverted)
    out, nb, nf, nc = condition_F(F, clamp=False)
    assert nf == 1 and nb == 0 and nc == 0
    assert np.linalg.det(out[0]) > 0                          # genuinely repaired, not rescaled


def test_nonfinite_rows_reset_and_counted():
    F = _healthy()
    F[2, 1, 1] = np.nan
    out, nb, nf, nc = condition_F(F, clamp=False)
    assert nb == 1
    assert np.allclose(out[2], np.eye(3))
    assert np.isfinite(out).all()


def test_legacy_clamp_counts_sv_projection():
    """The silent SV rewrite was an adversarial blocker: when clamping is requested it
    must at least be COUNTED; the blessed path uses clamp=False."""
    F = _healthy()
    F[1] = np.diag([3.0, 1.0, 1.0]).astype(np.float32)
    out, nb, nf, nc = condition_F(F, smin=0.5, smax=2.0, clamp=True)
    assert nc >= 1
    S = np.linalg.svd(out[1], compute_uv=False)
    assert S.max() <= 2.0 + 1e-5


def test_no_clamp_preserves_large_stretch():
    F = np.stack([np.diag([3.0, 1.0, 0.3]).astype(np.float32)])
    out, nb, nf, nc = condition_F(F, clamp=False)
    S = np.linalg.svd(out[0], compute_uv=False)
    assert abs(S.max() - 3.0) < 1e-5 and abs(S.min() - 0.3) < 1e-5 and nc == 0
