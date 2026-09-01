"""Metric validity (docs/pipeline_v2.md §5): fixed extent, loss independence, held-aware
jitter. The autoscale variants were adversarial blockers (one stray closed 43% of holes)."""
import numpy as np
import pytest

from physmorph import metrics


@pytest.fixture
def shell():
    rng = np.random.default_rng(0)
    n = 6000
    ang = rng.uniform(0, 2 * np.pi, n)
    r = rng.uniform(0.7, 1.0, n)
    return np.stack([r * np.cos(ang), rng.uniform(-0.05, 0.05, n),
                     r * np.sin(ang)], 1).astype(np.float32)


@pytest.fixture
def blob():
    rng = np.random.default_rng(1)
    return rng.uniform(-1, 1, (6000, 3)).astype(np.float32)


def test_hole_frac_invariant_to_stray(shell, blob):
    e = metrics.target_extent(blob)
    h0 = metrics.hole_frac(shell, e)
    stray = np.vstack([shell, [[9.0, 9.0, 9.0]]]).astype(np.float32)
    assert abs(metrics.hole_frac(stray, e) - h0) < 1e-3
    assert h0 > 0.02                                  # the annulus genuinely has holes


def test_sil_iou_invariant_to_stray(shell, blob):
    e = metrics.target_extent(blob)
    i0 = metrics.sil_iou(shell, blob, extent=e)
    stray = np.vstack([shell, [[9.0, 9.0, 9.0]]]).astype(np.float32)
    assert abs(metrics.sil_iou(stray, blob, extent=e) - i0) < 5e-3


def test_sil_iou_identity_is_one(blob):
    assert metrics.sil_iou(blob, blob) > 0.999


def test_outside_frac_flags_ejecta(shell, blob):
    e = metrics.target_extent(blob)
    assert metrics.outside_frac(shell, e) == 0.0
    stray = np.vstack([shell, [[9.0, 9.0, 9.0]]]).astype(np.float32)
    assert metrics.outside_frac(stray, e) > 0


def test_jitter_excludes_held_frames(shell):
    frames = [shell + 0.01 * k for k in range(5)] + [shell + 0.04] * 3
    j_honest = metrics.jitter(frames, tail=10, n_held=3)
    j_naive = metrics.jitter(frames, tail=10, n_held=0)
    assert j_honest["jitter_rel"] > j_naive["jitter_rel"]   # padding hid real motion
    assert j_honest["jitter_abs"] > 0


def test_jitter_short_run_has_all_keys(shell):
    j = metrics.jitter([shell], tail=10, n_held=0)
    assert set(j) == {"jitter_abs", "jitter_rel", "jitter_max_abs"}


def test_summarize_keys(shell, blob):
    out = metrics.summarize([shell, shell + 0.01], blob, n_held=0)
    for k in ("chamfer", "sil_iou", "hole_frac", "hole_frac_tgt", "outside_frac",
              "extent", "bbox_diag", "jitter_rel", "n_held"):
        assert k in out


def test_chamfer_zero_on_identical(blob):
    assert metrics.chamfer(blob, blob) == 0.0
