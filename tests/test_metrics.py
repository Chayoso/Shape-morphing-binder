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


def test_layer_breathing_separates_drift_from_alternation():
    """metrics.layer_breathing (docs/oscillation.md Addendum 9): a slab whose top face drifts
    outward every window reads flips ~0 and net/summed ~1; one whose top face goes out and back
    every window reads flips ~1 and net/summed ~0."""
    import numpy as np
    from physmorph.metrics import layer_breathing
    s = 0.2; g = np.arange(16) * s
    X, Y, Z = np.meshgrid(g, np.arange(6) * s, g, indexing="ij")
    x0 = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
    x0 += np.random.default_rng(0).uniform(-0.03, 0.03, x0.shape).astype(np.float32) * s
    top = x0[:, 1] > x0[:, 1].max() - 0.5 * s
    T = 5; nwin = 14
    def frames_of(kind):
        fr = []
        for w in range(nwin * T + 1):
            x = x0.copy()
            k = w // T
            if kind == "drift":
                x[top, 1] += 0.05 * s * k
            else:
                x[top, 1] += 0.05 * s * (k % 2)
            fr.append(x)
        return fr
    d = layer_breathing(frames_of("drift"), window=T, k_windows=10)
    a = layer_breathing(frames_of("alt"), window=T, k_windows=10)
    assert d["layer_flip_frac"] < 0.2 and d["layer_net_ratio"] > 0.8
    assert a["layer_flip_frac"] > 0.8 and a["layer_net_ratio"] < 0.2
    assert 0.0 < a["layer_step_sp"] < 0.2
