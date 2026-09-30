"""repair_F: repair-only semantics at window ends (non-finite rows and reflections, counted)."""
import numpy as np
import torch

from physmorph.mpm.conditioning import repair_F


def _healthy(n=5, seed=0):
    rng = np.random.default_rng(seed)
    return torch.as_tensor((rng.normal(0, 0.3, (n, 3, 3)) + np.eye(3)).astype(np.float32), device="cuda")


def test_healthy_f_is_returned_unchanged():
    F = _healthy()
    out, nb, nf = repair_F(F)
    assert torch.equal(out, F.reshape(-1, 3, 3))
    assert (nb, nf) == (0, 0)


def test_reflection_is_repaired_and_counted():
    F = _healthy()
    F[0] = torch.diag(torch.tensor([-3.0, 1.0, 1.0], device="cuda"))   # det = -3 (inverted)
    out, nb, nf = repair_F(F)
    assert nf == 1 and nb == 0
    assert float(torch.linalg.det(out[0])) > 0                          # repaired, not rescaled


def test_nonfinite_rows_reset_and_counted():
    F = _healthy()
    F[2, 1, 1] = float("nan")
    out, nb, nf = repair_F(F)
    assert nb == 1
    assert torch.allclose(out[2], torch.eye(3, device="cuda"))
    assert bool(torch.isfinite(out).all())


def test_large_stretch_is_kept():
    F = torch.diag(torch.tensor([3.0, 1.0, 0.3], device="cuda"))[None]
    out, _, _ = repair_F(F)
    S = torch.linalg.svdvals(out[0])
    assert abs(float(S.max()) - 3.0) < 1e-5 and abs(float(S.min()) - 0.3) < 1e-5
