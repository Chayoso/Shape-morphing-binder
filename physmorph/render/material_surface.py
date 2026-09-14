"""Connected material-surface quadrature; no independently fitted Gaussian state.

Mesh-face parameterization is related to GaMeS (Waczynska et al., 2024).
Here all weights, footprints, opacity and color are fixed. Only MPM-advected
vertices change. The footprint is the vertex population covariance, deliberately
four times the uniform-triangle area covariance to overlap neighboring faces.
"""
import numpy as np
import torch


def source_surface(source, subdivisions=2):
    """A convex source's enclosing hull, refined BEFORE material advection.

    Each new vertex is transported independently by the MPM grid on every step;
    subdivision therefore increases geometric resolution, not just splat count.
    """
    from scipy.spatial import ConvexHull
    import trimesh

    x = np.asarray(source, np.float64)
    hull = ConvexHull(x)
    faces = hull.simplices.copy()
    tri = x[faces]
    backwards = (np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])*hull.equations[:, :3]).sum(1) < 0
    faces[backwards] = faces[backwards][:, [0, 2, 1]]
    mesh = trimesh.Trimesh(vertices=x, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    for _ in range(subdivisions):
        mesh = mesh.subdivide()
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError("initial material surface must be closed and oriented")
    return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int64)


def triangle_gaussians(vertices, faces, thickness=.001):
    """One Gaussian on each moving triangle; exact first-order Torch derivatives."""
    tri = vertices[faces]
    center = tri.mean(1)
    delta = tri-center[:, None, :]
    normal = torch.linalg.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    normal = normal/normal.norm(dim=1, keepdim=True).clamp_min(1e-12)
    covariance = torch.einsum("nki,nkj->nij", delta, delta)/3
    covariance = covariance+thickness**2*normal[:, :, None]*normal[:, None, :]
    covariance6 = torch.stack([covariance[:, i, j] for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))], 1)
    return center, covariance6.contiguous(), normal


class MaterialSurfaceViews:
    def __init__(self, views, camera_radius, source_faces, target_vertices, target_faces,
                 resolution=1024, device="cuda", thickness=.001):
        from physmorph.pipeline.gauss_loss import _camera
        self.res, self.thickness = resolution, thickness
        self.faces = torch.as_tensor(source_faces, dtype=torch.long, device=device)
        self.cams = [_camera(az, el, camera_radius, resolution, .7, device) for az, el in views]
        target = torch.as_tensor(target_vertices, dtype=torch.float32, device=device)
        target_faces = torch.as_tensor(target_faces, dtype=torch.long, device=device)
        with torch.no_grad():
            self.targets = [self.render(target, c, target_faces).detach() for c in self.cams]

    def render(self, vertices, camera, faces=None):
        from physmorph.pipeline.gauss_loss import _gs
        mod, has_norm = _gs()
        centers, covariance, normals = triangle_gaussians(vertices, self.faces if faces is None else faces, self.thickness)
        kw = dict(means3D=centers, means2D=torch.zeros_like(centers),
                  opacities=torch.full((len(centers), 1), .9, dtype=vertices.dtype, device=vertices.device),
                  colors_precomp=torch.full((len(centers), 3), .35, dtype=vertices.dtype, device=vertices.device))
        if has_norm:
            kw.update(cov3Ds_precomp=covariance, norm3Ds_precomp=normals)
        else:
            kw.update(cov3D_precomp=covariance)
        out = mod.GaussianRasterizer(raster_settings=camera)(**kw)
        return out[0] if isinstance(out, tuple) else out
