import numpy as np
import trimesh
from physmorph.sampling.closed_mesh import weld_geometry, center_target_by_quadrature


def test_obj_face_normal_seams_do_not_split_the_material_surface():
    sphere = trimesh.creation.icosphere(subdivisions=1)
    packed = np.asarray(sphere.vertices)[sphere.faces].reshape(-1, 3)
    disconnected = trimesh.Trimesh(packed, np.arange(len(packed)).reshape(-1, 3), process=False)
    assert not disconnected.is_watertight
    fixed = weld_geometry(disconnected)
    assert fixed.is_watertight and fixed.is_winding_consistent and fixed.euler_number == 2
    assert len(fixed.vertices) == len(sphere.vertices)
    assert np.isclose(fixed.volume, sphere.volume)


def test_target_translation_is_set_by_mass_quadrature_and_shared_by_all_arrays():
    rng = np.random.default_rng(14)
    dense = rng.uniform(-1, 1, (350, 3))+[3, 5, 7]
    target = rng.uniform(-1, 1, (17, 3))+[3, 5, 7]
    vertices = trimesh.creation.icosphere().vertices+[3, 5, 7]
    source_com = np.array([.01, -.02, .03])
    (t, v, d), center = center_target_by_quadrature(target, vertices, dense, source_com)
    assert np.allclose(d.mean(0, dtype=np.float64), source_com, atol=1e-8)
    for before, after in ((target, t), (vertices, v), (dense, d)):
        assert after.dtype == np.float32
        assert np.allclose(after-before, source_com-center, atol=1e-7)
    # Do not independently recenter the random evaluation sample.
    assert np.linalg.norm(t.mean(0)-source_com) > .01
