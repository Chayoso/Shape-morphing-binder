"""Global material-surface checks on raw simulation vertices, without rendering."""
import numpy as np


def triangle_pair_is_separated(a, b):
    """Conservative float64 separation certificate for an Open3D candidate.

    Any axis can prove separation of convex triangles. In-plane edge normals
    also handle nearly coplanar triangles, where plane-line tests are fragile.
    Contact, degeneracy, and uncertain numerical gaps do not prove separation.
    See https://www.geometrictools.com/Documentation/MethodOfSeparatingAxes.pdf
    """
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != (3, 3) or b.shape != (3, 3) or not np.isfinite([a, b]).all():
        return False
    # Use a common local origin; retain an error allowance for the subtraction.
    absolute_scale = max(float(np.abs(a).max()), float(np.abs(b).max()))
    origin = a[0].copy()
    a, b = a-origin, b-origin
    ea, eb = np.roll(a, -1, axis=0)-a, np.roll(b, -1, axis=0)-b
    local_scale = max(float(np.linalg.norm(ea, axis=1).max()), float(np.linalg.norm(eb, axis=1).max()))
    if local_scale == 0 or not np.isfinite(local_scale):
        return False
    ea, eb = ea/local_scale, eb/local_scale
    na, nb = np.cross(ea[0], ea[1]), np.cross(eb[0], eb[1])
    eps = np.finfo(np.float64).eps
    if min(np.linalg.norm(na), np.linalg.norm(nb)) <= 256*eps:
        return False
    axes = np.concatenate([np.array([na, nb]), np.cross(ea[:, None], eb[None]).reshape(-1, 3),
                           np.cross(na, ea), np.cross(nb, eb)])
    lengths = np.linalg.norm(axes, axis=1)
    axes = axes[np.isfinite(lengths) & (lengths > 256*eps)]
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    pa, pb = a@axes.T, b@axes.T
    gap = np.maximum(pa.min(0)-pb.max(0), pb.min(0)-pa.max(0))
    error = 256*eps*(absolute_scale+local_scale+max(float(np.abs(a).max()), float(np.abs(b).max())))
    return bool(np.any(gap > max(1e-10*local_scale, error)))


def surface_intersection_pairs(vertices, faces):
    """Open3D candidates minus independently certified disjoint pairs.

    This removes observed near-coplanar false positives. It is not an
    independent completeness guarantee for Open3D's candidate generation.
    """
    import open3d as o3d
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                    o3d.utility.Vector3iVector(faces))
    candidates = np.asarray(mesh.get_self_intersecting_triangles()).reshape(-1, 2)
    retained = [pair for pair in candidates
                if not triangle_pair_is_separated(vertices[faces[pair[0]]], vertices[faces[pair[1]]])]
    return np.asarray(retained, np.int64).reshape(-1, 2), len(candidates)


def no_surface_intersections(trajectory):
    if trajectory.surface_x is None or trajectory.surface_faces is None:
        raise ValueError("global surface checks require connected material geometry")
    removed = 0
    for t, state in enumerate(trajectory.surface_x):
        pairs, candidates = surface_intersection_pairs(state.numpy(), trajectory.surface_faces)
        removed += candidates-len(pairs)
        if len(pairs):
            return {"valid": False, "reason": "surface_self_intersection", "substep": t,
                    "intersection_pairs": len(pairs)}
    return {"valid": True, "global_surface_intersections_checked": trajectory.T+1,
            "certified_disjoint_candidates": removed}
