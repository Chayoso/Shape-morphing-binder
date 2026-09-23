import numpy as np
import pytest
import trimesh

from physmorph.sampling.mesh import filled_volume, load_mesh, load_normalized, sample_volume


def _interior_fraction(mesh, x, k=0.03):
    _, dist, _ = trimesh.proximity.closest_point(mesh, x[:1500])
    return float((dist > k * float(np.linalg.norm(mesh.extents))).mean())


@pytest.mark.parametrize("path", ["assets/bunny.obj", "assets/isosphere.obj"])
def test_sample_volume_is_volumetric(path):
    # 2026-09-03 forensic: the default trimesh fill left bunny.obj (non-watertight)
    # with zero interior voxels and the pipeline morphed into a SHELL for days.
    m = load_mesh(path)
    x = sample_volume(m, 6000, seed=1)
    assert _interior_fraction(m, x) > 0.3


def test_target_volume_matched_to_source():
    src, vs = load_normalized("assets/isosphere.obj", 4000, 1, return_volume=True)
    tgt, vt = load_normalized("assets/bunny.obj", 4000, 2, match_volume=vs,
                              return_volume=True)
    assert abs(vt - vs) / vs < 0.02
    assert vs > 10.0                      # a solid at bbox-diag 8, not a shell
