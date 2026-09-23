"""Sampler fill forensic (2026-09-16): the target voxel fill must contain no axis-fill
streaks (1-voxel interior columns), must be independent of the particle count n (20k and 40k
targets come from the same voxel set), and the filled volume must be the sampled volume."""
import numpy as np
import pytest

from physmorph.sampling import mesh as sm

BUNNY = "assets/bunny.obj"


@pytest.fixture(scope="module")
def bunny():
    return sm.load_mesh(BUNNY)


def test_bunny_fill_has_no_streaks(bunny):
    ext = float(bunny.extents.max())
    c = sm._fill_centers(bunny, ext / 110)
    assert len(c) > 200_000 and sm.STREAK_REPORT["method"] in ("orthographic", "ortho_reliable")
    # re-check independently: no interior voxel with <= 2 filled 6-neighbours
    vg = bunny.voxelized(pitch=ext / 110)
    f = vg.copy().fill(method="orthographic")
    _, n_streak = sm._strip_streaks(f.matrix.copy(), vg.matrix.copy())
    assert n_streak == 0
    # and the 'base' fill DOES have them (the artefact this test guards against)
    fb = vg.copy().fill(method="base")
    _, n_base = sm._strip_streaks(fb.matrix.copy(), vg.matrix.copy())
    assert n_base > 100


def test_sample_is_n_independent_and_volume_consistent(bunny):
    x20 = sm.load_normalized(BUNNY, 20000, seed=2)
    x40 = sm.load_normalized(BUNNY, 40000, seed=2)
    # same fill, same normalisation: bounding boxes agree to a jitter
    assert np.allclose(x20.min(0), x40.min(0), atol=0.15) and np.allclose(x20.max(0), x40.max(0), atol=0.15)
    ext = float(bunny.extents.max())
    c = sm._fill_centers(bunny, ext / 110)
    assert abs(sm.filled_volume(bunny) - len(c) * (ext / 110) ** 3) < 1e-6
