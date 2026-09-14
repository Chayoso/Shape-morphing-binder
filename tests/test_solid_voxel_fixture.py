import numpy as np
from physmorph.sampling.mesh import seal_internal_voxel_voids


def test_solid_fixture_seals_internal_void_but_preserves_open_concavity():
    occupied = np.ones((7, 7, 7), bool)
    occupied[3, 3, 3] = False  # sealed interior
    occupied[:2, 2:5, 2:5] = False  # exterior notch must remain
    points = np.argwhere(occupied).astype(np.float32)*.25
    result, stats = seal_internal_voxel_voids(points, .25)
    keys = set(map(tuple, np.rint(result/.25).astype(int)))
    assert (3, 3, 3) in keys and (1, 3, 3) not in keys
    assert stats['sealed_internal_voxels'] == 1
    assert len(result) == len(points)+1
