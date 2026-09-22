"""The reliable-axis fill of a NON-WATERTIGHT mesh (sampling.mesh._fill_ortho_reliable, method.md 10.13):
a box with a hole in its bottom face is filled above the hole (the plain three-axis intersection leaves
that column empty); a closed box and a torus (a hole enclosed by two axes only) are filled exactly as
the plain intersection fills them."""
import numpy as np
import trimesh

from physmorph.sampling import mesh as sm


def _plain(surf):
    enc = []
    for ax in range(3):
        fwd = np.maximum.accumulate(surf, axis=ax)
        bwd = np.flip(np.maximum.accumulate(np.flip(surf, axis=ax), axis=ax), axis=ax)
        enc.append(fwd & bwd)
    return (enc[0] & enc[1] & enc[2]) | surf


def test_box_with_a_bottom_hole_is_filled_above_the_hole():
    box = trimesh.creation.box(extents=(4.0, 2.0, 4.0))
    box = box.subdivide().subdivide().subdivide()
    F = np.asarray(box.faces); V = np.asarray(box.vertices)
    cen = V[F].mean(1)
    # remove the bottom faces near the centre: a hole of ~1.6 x 1.6 in a 4 x 4 bottom (y = -1)
    hole = (np.abs(cen[:, 1] + 1.0) < 1e-6) & (np.abs(cen[:, 0]) < 0.8) & (np.abs(cen[:, 2]) < 0.8)
    open_box = trimesh.Trimesh(vertices=V, faces=F[~hole], process=False)
    assert not open_box.is_watertight
    pitch = 0.2
    vg = open_box.voxelized(pitch=pitch)
    surf = vg.matrix.copy()
    plain = _plain(surf)
    fps = sm._hole_footprints(open_box, vg)
    assert fps[1] is not None                       # the hole faces -y: its footprint along y has area
    # along x and z the loop is seen edge-on: at most a thin strip at the bottom plane (no interior area)
    if fps[0] is not None:                          # footprint over (y, z): a line at y = -1, dilated -> <= 3 rows
        assert fps[0].any(axis=1).sum() <= 3, fps[0].any(axis=1).sum()
    if fps[2] is not None:                          # footprint over (x, y): the same line -> <= 3 columns
        assert fps[2].any(axis=0).sum() <= 3, fps[2].any(axis=0).sum()
    rel = sm._fill_ortho_reliable(surf, fps)
    # the column above the hole: interior voxels at the box centre
    idx = vg.points_to_indices(np.array([[0.0, 0.0, 0.0], [0.3, 0.2, -0.3]]))
    for i in idx:
        assert not plain[tuple(i)], "the plain intersection leaves the column above the hole empty"
        assert rel[tuple(i)], "the reliable-axis fill closes it from the side projections"
    # nothing is filled outside the box
    out = vg.points_to_indices(np.array([[0.0, 1.6, 0.0]]))
    assert not rel[tuple(out[0])]
    # and the interior count is now close to the closed box's
    closed = trimesh.creation.box(extents=(4.0, 2.0, 4.0)).voxelized(pitch=pitch)
    n_closed = int(_plain(closed.matrix.copy()).sum())
    assert abs(int(rel.sum()) - n_closed) < 0.05 * n_closed, (int(rel.sum()), n_closed)


def test_closed_meshes_are_unchanged():
    for mesh in (trimesh.creation.box(extents=(3.0, 2.0, 2.5)), trimesh.creation.torus(major_radius=2.0, minor_radius=0.6)):
        vg = mesh.voxelized(pitch=0.2)
        surf = vg.matrix.copy()
        fps = sm._hole_footprints(mesh, vg)
        assert all(f is None for f in fps)
        assert np.array_equal(sm._fill_ortho_reliable(surf, fps), _plain(surf))
    torus = trimesh.creation.torus(major_radius=2.0, minor_radius=0.6)
    vg = torus.voxelized(pitch=0.2)
    filled = sm._fill_ortho_reliable(vg.matrix.copy(), sm._hole_footprints(torus, vg))
    centre = vg.points_to_indices(np.array([[0.0, 0.0, 0.0]]))[0]
    assert not filled[tuple(centre)], "the torus hole stays open"
