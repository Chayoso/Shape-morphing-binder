"""particle_depth_probe.py TARGET_OBJ LABEL=FRAMES_NPZ[:STATE] [...] — D71: where the particles themselves stand against the
target mesh, apart from how the display's field reads them. The signed distance of every particle to the mesh
(exterior_offset_probe.py's fitted mesh: the plane of the nearest of three million surface samples), in display
pitches: the outermost layer's mean and median, the share of all particles outside the mesh, and the share of the
particles in shells by signed distance. STATE: `target`, or `end` (the last kept frame, default)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import trimesh                                                 # noqa: E402
from physmorph.pipeline.window.layer import layer_by_asymmetry, layer_spacing  # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.sampling.mesh import load_mesh                  # noqa: E402
from physmorph.sampling.orientation import orient_name, rotation  # noqa: E402

dev = torch.device("cuda")
surface_tree = None
edges = [-3., -2., -1., -.5, 0., .5, 1., 2.]
for spec in sys.argv[2:]:
    label, path = spec.split("=")
    path, state = (path.split(":") + ["end"])[:2]
    z = np.load(path, allow_pickle=True)
    tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
    if surface_tree is None:
        a = .708 * float(knn_self_torch(tgt, 9)[0][:, 8].median())
        mesh = load_mesh(sys.argv[1])
        mesh.merge_vertices()
        o = orient_name(sys.argv[1])
        if o != "id":
            mesh.vertices = np.asarray(mesh.vertices, np.float64) @ rotation(o).T
        t_np = tgt.cpu().numpy().astype(np.float64)
        vb = np.asarray(mesh.bounds, np.float64)
        step = float(knn_self_torch(tgt, 7)[0][:, 6].median())
        mesh.vertices = (np.asarray(mesh.vertices, np.float64) - vb.mean(0)) * float(np.mean((t_np.max(0) - t_np.min(0) + step) / (vb[1] - vb[0]))) \
            + 0.5 * (t_np.max(0) + t_np.min(0))
        points, face = trimesh.sample.sample_surface(mesh, 3000000, seed=0)
        surface = torch.as_tensor(np.asarray(points, np.float32), device=dev)
        surface_normal = torch.as_tensor(np.asarray(mesh.face_normals[face], np.float32), device=dev)
        surface_tree = gpu.KNN(surface)
    x = tgt if state == "target" else torch.as_tensor(np.asarray(z["frames"][-1], np.float32), device=dev)
    with torch.no_grad():
        at = surface_tree.query(x, 1)[1].reshape(-1)
        s = ((x - surface[at]) * surface_normal[at]).sum(1) / a
        layer = layer_by_asymmetry(x, layer_spacing(x))[0]
        sl = s[layer]
        sl = sl[sl.abs() < 3.]                                  # the mesh's open base and closed sets inside are left out
        shells = " ".join(f"[{lo:+.1f},{hi:+.1f}) {100 * float(((s >= lo) & (s < hi)).float().mean()):.2f}" for lo, hi in zip(edges[:-1], edges[1:]))
        print(f"{label}: layer {100 * float(layer.float().mean()):.2f} % of the particles, its signed distance mean {float(sl.mean()):+.3f} median {float(sl.median()):+.3f} | "
              f"outside the mesh {100 * float((s > 0).float().mean()):.2f} % of all, beyond +0.5 {100 * float((s > .5).float().mean()):.2f} %, beyond +1 {100 * float((s > 1.).float().mean()):.3f} % | "
              f"99th percentile {float(torch.quantile(s[::3], .99)):+.3f} | shells, % of all: {shells}", flush=True)
