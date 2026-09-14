"""Direct interior queries on a validated closed mesh; no axis-fill bridges."""
import numpy as np


def center_target_by_quadrature(target, vertices, dense, source_center):
    """One translation for all target representations, set by the mass objective."""
    center = np.asarray(dense).mean(0, dtype=np.float64)
    shift = np.asarray(source_center, np.float64)-center
    arrays = tuple(np.asarray(np.asarray(a, np.float64)+shift, np.float32) for a in (target, vertices, dense))
    return arrays, center


def validate_surface_arrays(vertices, faces):
    """Inspect exact final arrays without welding, repairing, or dropping faces."""
    import trimesh
    from ..pipeline.surface_validity import surface_intersection_pairs
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    pairs, candidates = surface_intersection_pairs(vertices, faces)
    components = len(trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(faces))))
    report = dict(watertight=bool(mesh.is_watertight), winding=bool(mesh.is_winding_consistent),
        euler=int(mesh.euler_number), components=components, volume=float(mesh.volume),
        min_triangle_area=float(mesh.area_faces.min()), intersection_pairs=len(pairs),
        certified_disjoint_candidates=candidates-len(pairs))
    report["valid"] = (np.isfinite(vertices).all() and report["watertight"] and report["winding"]
        and components == 1 and report["volume"] > 0 and report["min_triangle_area"] > 0 and not len(pairs))
    report["valid"] = bool(report["valid"])
    if not report["valid"]:
        raise ValueError(f"final surface geometry failed validation: {report}")
    return report


def weld_geometry(mesh):
    """OBJ normal/UV seams must not split the physical boundary into face islands."""
    import trimesh
    result = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces),
                             process=True, validate=True)
    if result.volume < 0:
        result.invert()
    return result


class ClosedMeshSampler:
    def __init__(self, mesh):
        import open3d as o3d
        mesh = weld_geometry(mesh)
        self.o3d, self.mesh = o3d, mesh
        legacy = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                          o3d.utility.Vector3iVector(mesh.faces))
        if (not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0
                or legacy.is_self_intersecting()):
            raise ValueError("sampling requires a closed oriented nonintersecting mesh")
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))

    def contains(self, points):
        query = self.o3d.core.Tensor(np.ascontiguousarray(points, np.float32))
        return self.scene.compute_occupancy(query, nthreads=4, nsamples=3).numpy() > .5

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        accepted, remaining = [], n
        for _ in range(100):
            candidates = rng.uniform(self.mesh.bounds[0], self.mesh.bounds[1], (max(2048, 3*remaining), 3))
            inside = candidates[self.contains(candidates)][:remaining]
            accepted.append(inside)
            remaining -= len(inside)
            if not remaining:
                return np.asarray(np.concatenate(accepted), np.float32)
        raise RuntimeError("closed-mesh rejection sampler did not fill its budget")

    def quadrature(self, resolution=110):
        pitch = float(self.mesh.extents.max())/resolution
        axes = [np.arange(np.floor(lo/pitch)-1, np.ceil(hi/pitch)+2)*pitch
                for lo, hi in self.mesh.bounds.T]
        candidates = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        inside = self.contains(candidates)
        return np.asarray(candidates[inside], np.float32), pitch
