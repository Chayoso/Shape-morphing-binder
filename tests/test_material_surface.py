import numpy as np
import torch

from physmorph.render.material_surface import source_surface, triangle_gaussians


def test_source_surface_refines_connected_geometry_and_contains_original_points():
    from scipy.spatial import ConvexHull
    import trimesh
    rng = np.random.default_rng(31)
    x = rng.normal(size=(100, 3))
    vertices, faces = source_surface(x, subdivisions=1)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert len(vertices) > len(ConvexHull(x).vertices)
    hull = ConvexHull(vertices)
    assert (x@hull.equations[:, :3].T+hull.equations[:, 3]).max() < 2e-6


def test_triangle_gaussians_reproduce_rigid_motion_and_differentiate_vertex_motion():
    x = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=torch.double, requires_grad=True)
    f = torch.tensor([[0, 1, 2]])
    assert torch.autograd.gradcheck(lambda v: triangle_gaussians(v, f)[:2], (x,))
    center, c6, _ = triangle_gaussians(x, f)
    # Cyclic axis permutation is a rigid rotation, including normal thickness.
    perm = [2, 0, 1]
    moved, cm6, _ = triangle_gaussians(x[:, perm]+2., f)
    assert torch.allclose(moved, center[:, perm]+2.)
    def full(c):
        return c[:, [0, 1, 2, 1, 3, 4, 2, 4, 5]].reshape(-1, 3, 3)
    assert torch.allclose(full(cm6), full(c6)[:, perm][:, :, perm])
