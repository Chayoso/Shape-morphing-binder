import numpy as np
import pytest

from physmorph.pipeline.surface_validity import triangle_pair_is_separated


@pytest.mark.parametrize("scale,offset", [(1., 0.), (1e-5, 0.), (1e5, 0.), (1., 1e6)])
def test_observed_near_coplanar_open3d_false_positive_has_a_separating_axis(scale, offset):
    a = np.array([[-.939180076, -.201466948, -.163946897],
                  [-.946374476, -.162210569, -.107133448],
                  [-.939180076, -.122954190, -.163946897]])
    b = np.array([[-.946374476, -.240723327, -.107133448],
                  [-.953568876, -.279979706, -.050319966],
                  [-.953568876, -.201466948, -.050319992]])
    assert triangle_pair_is_separated(a*scale+offset, b*scale+offset)
    assert triangle_pair_is_separated(b*scale+offset, a*scale+offset)


@pytest.mark.parametrize("second", [
    [[.2, .2, 0], [.7, .2, 0], [.2, .7, 0]],  # Coplanar overlap.
    [[.2, .2, -1], [.2, .2, 1], [.7, .2, 0]],  # Transverse penetration.
    [[1, 0, 0], [2, 0, 0], [1, 1, 0]],  # Point contact.
    [[0, 0, 0], [1, 0, 0], [0, -1, 0]],  # Edge contact.
    [[2, 2, 2], [2, 2, 2], [2, 2, 2]],  # Degenerate point must not certify.
    [[2, 2, 2], [3, 3, 3], [4, 4, 4]],  # Degenerate line.
])
def test_intersection_contact_and_degeneracy_never_certify_separation(second):
    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    assert not triangle_pair_is_separated(a, second)
    assert not triangle_pair_is_separated(second, a)


def test_nonfinite_and_roundoff_sized_gap_are_not_certified():
    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    assert not triangle_pair_is_separated(a, a + [0, 0, 1e-12])
    assert not triangle_pair_is_separated(a, a*np.nan)
    assert triangle_pair_is_separated(a, a + [0, 0, .1])
