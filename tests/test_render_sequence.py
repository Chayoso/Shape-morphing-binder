import numpy as np
import pytest

from scripts import render_sequence
from scripts.render_sequence import load_render_archive


def _archive(tmp_path, **replace):
    n, m = 4, 5
    src = np.arange(n * 3, dtype=np.float32).reshape(n, 3) / 10
    tgt = np.arange(m * 3, dtype=np.float32).reshape(m, 3) / 8
    values = {
        "src": src,
        "tgt": tgt,
        "frames": np.stack([src, src + 0.1, src + 0.2]),
        "F_samples": np.tile(np.eye(3, dtype=np.float32), (2, n, 1, 1)),
        "F_sample_idx": np.array([0, 2], np.int64),
        "render_mask": np.array([1, 0, 1, 0], bool),
        "target_render_mask": np.array([1, 1, 0, 0, 0], bool),
        "sigma0": np.float32(0.123),
    }
    values.update(replace)
    path = tmp_path / "run.npz"
    np.savez(path, **values)
    return path


def test_saved_masks_and_sigma_are_reused_exactly(tmp_path):
    data = load_render_archive(_archive(tmp_path), surface_frac=0.9, sigma_scale=9.0)
    assert np.array_equal(data["render_mask"], [True, False, True, False])
    assert np.array_equal(data["target_render_mask"], [True, True, False, False, False])
    assert data["sigma0"] == pytest.approx(0.123)
    assert data["frames"].shape == (3, 4, 3)


def test_single_child_archive_keeps_the_parent_sigma(tmp_path):
    source_offsets = np.zeros((4, 1, 3), np.float32)
    target_offsets = np.zeros((5, 1, 3), np.float32)
    data = load_render_archive(_archive(
        tmp_path, source_child_offsets=source_offsets,
        target_child_offsets=target_offsets,
        gauss_child_sigma_scale=np.float32(0.55)))
    assert data["source_child_offsets"].shape[1] == 1
    assert data["gauss_child_sigma_scale"] == pytest.approx(1.0)


@pytest.mark.parametrize("replace,match", [
    ({"F_sample_idx": np.array([], np.int64),
      "F_samples": np.empty((0, 4, 3, 3), np.float32)}, "non-empty"),
    ({"F_sample_idx": np.array([2, 1], np.int64)}, "strictly increasing"),
    ({"render_mask": np.zeros(4, bool)}, "too few"),
    ({"sigma0": np.float32(np.nan)}, "sigma0"),
])
def test_invalid_archive_is_rejected_before_cuda_render(tmp_path, replace, match):
    with pytest.raises(ValueError, match=match):
        load_render_archive(_archive(tmp_path, **replace))


def test_nonfinite_selected_frame_is_rejected(tmp_path):
    src = np.arange(12, dtype=np.float32).reshape(4, 3) / 10
    frames = np.stack([src, src + 0.1, src + 0.2])
    frames[2, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite selected state"):
        load_render_archive(_archive(tmp_path, frames=frames))


def test_archive_renderer_expands_colors_F_and_support_opacity_per_child(tmp_path,
                                                                        monkeypatch):
    source_offsets = np.zeros((4, 2, 3), np.float32)
    source_offsets[:, 0, 0], source_offsets[:, 1, 0] = 0.01, -0.01
    target_offsets = np.zeros((5, 2, 3), np.float32)
    target_offsets[:, 0, 0], target_offsets[:, 1, 0] = 0.01, -0.01
    archive = _archive(tmp_path, source_child_offsets=source_offsets,
                       target_child_offsets=target_offsets,
                       gauss_child_sigma_scale=np.float32(0.55))
    calls = []

    def fake_render(x, color, **kw):
        calls.append((x.copy(), color.copy(), kw))
        return np.zeros((8, 8, 3), np.float32)

    class FakeSupport:
        @classmethod
        def from_rest(cls, _x, _k):
            return cls()

        def opacity(self, _x):
            return np.array([1.0, 0.75, 0.5, 0.25], np.float32)

    monkeypatch.setattr(render_sequence, "render_3dgs", fake_render)
    monkeypatch.setattr(render_sequence, "MaterialSupport", FakeSupport)
    out = tmp_path / "children.gif"
    monkeypatch.setattr("sys.argv", ["render_sequence.py", "--npz", str(archive),
                                     "--out", str(out), "--res", "8", "--views", "0"])
    render_sequence.main()
    x, color, kw = calls[0]
    assert x.shape == (4, 3) and kw["F"].shape == (4, 3, 3)
    assert np.array_equal(color[0], color[1]) and np.array_equal(color[2], color[3])
    np.testing.assert_allclose(kw["opacity"], [0.92, 0.92, 0.46, 0.46])
    assert kw["sigma0"] == pytest.approx(0.123 * 0.55)
    assert out.exists()
