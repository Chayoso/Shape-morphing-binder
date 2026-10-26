"""d161_canvas.py OUT_DIR -- D161's smooth canvases (CPU, trimesh), the 3D targets of the two scenes and the bases of their
relief meshes:
  d161sph.obj           a sphere (icosphere, 5 subdivisions: 10242 vertices, near-equilateral faces), radius 7.25 about
                        the isosphere asset's centre (the run's frame normalises size and position, so only the shape
                        counts); the asset itself is an 80-face polyhedron whose facets would be relief of their own;
  d161smooth_bunny.obj  the bunny with its scan bumps taken out by Taubin's smoothing (lambda 0.5, mu -0.51, 300
                        iterations: pass band k_PB = 0.04, wavelengths below about 30 edges (1.8 wu) go, no shrinkage), so that the letters are the only relief beyond the
                        smooth 3D target and the bumps (D159's subject) do not mix in; the name ends in _bunny so the
                        orientation table turns it as the bunny.
  d161coin.obj          a coin (the points within 0.9 of a flat disc of radius 1.85: flat faces, a round rim; axis z),
                        marching cubes of its exact distance at h = 0.022: a real morph from the sphere with a flat canvas.
Prints each mesh's counts, watertightness and its volume against the original's."""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rows = {}
iso = trimesh.load("assets/isosphere.obj", process=True, force="mesh")
iso.merge_vertices(merge_tex=True, merge_norm=True)
c = np.asarray(iso.vertices).mean(0)
s = trimesh.creation.icosphere(subdivisions=5, radius=7.25)
s.apply_translation(c)
s.export(out / "d161sph.obj")
rows["d161sph"] = dict(verts=len(s.vertices), faces=len(s.faces), watertight=bool(s.is_watertight), volume=float(s.volume),
                       iso_volume=float(iso.convex_hull.volume), center=c.tolist())
b = trimesh.load("assets/bunny.obj", process=True, force="mesh")
b.merge_vertices()
v0 = float(abs(b.volume)) if b.is_watertight else float(b.convex_hull.volume)
bs = b.copy()
trimesh.smoothing.filter_taubin(bs, lamb=0.5, nu=0.51, iterations=300)
bs.export(out / "d161smooth_bunny.obj")
d = np.linalg.norm(np.asarray(bs.vertices) - np.asarray(b.vertices), axis=1)
rows["d161smooth_bunny"] = dict(verts=len(bs.vertices), faces=len(bs.faces), watertight=bool(bs.is_watertight),
                                hull_volume_ratio=float(bs.convex_hull.volume / b.convex_hull.volume),
                                move_raw_median=float(np.median(d)), move_raw_p99=float(np.quantile(d, 0.99)),
                                edge_raw=float(np.median(b.edges_unique_length)))
print(json.dumps(rows, indent=1))
json.dump(rows, open(out / "canvas.json", "w"), indent=1)
# the coin: the points within r_e of a flat disc of radius R0 (a disc grown by a ball: flat faces, a round rim, smooth),
# axis z, overall diameter 2 (R0 + r_e) = 5.5 and thickness 2 r_e = 1.8 in raw units (the frame rescales); marching
# cubes of the exact distance at h = 0.022 (skimage), the sampler's watertight input as D156's hulls
from skimage.measure import marching_cubes  # noqa: E402
R0, re_, h = 1.85, 0.9, 0.022
lo = np.array([-(R0 + re_) - 3 * h, -(R0 + re_) - 3 * h, -re_ - 3 * h])
n = np.ceil((-2 * lo) / h).astype(int) + 1
gx, gy, gz = (lo[i] + h * np.arange(n[i]) for i in range(3))
X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
rho = np.hypot(X, Y)
sdf = np.sqrt(np.maximum(rho - R0, 0.0) ** 2 + Z ** 2) - re_
V, F, _, _ = marching_cubes(sdf, level=0.0, spacing=(h, h, h), allow_degenerate=False)
coin = trimesh.Trimesh(V + lo, F, process=True)
coin.update_faces(coin.nondegenerate_faces())          # zero-area faces (as D156's hull_carve.py)
coin.merge_vertices()
coin.remove_unreferenced_vertices()
if not coin.is_watertight:
    trimesh.repair.fill_holes(coin)
assert coin.is_watertight, "the coin is not watertight"
if coin.volume < 0:
    coin.invert()
coin.export(out / "d161coin.obj")
rows["d161coin"] = dict(verts=len(coin.vertices), faces=len(coin.faces), watertight=bool(coin.is_watertight),
                        volume=float(coin.volume), exact_volume=float(np.pi * R0 ** 2 * 2 * re_ + np.pi ** 2 * re_ ** 2 * R0
                                                                     + 4 / 3 * np.pi * re_ ** 3),
                        min_edge=float(coin.edges_unique_length.min()), median_edge=float(np.median(coin.edges_unique_length)))
print(json.dumps(rows["d161coin"], indent=1))
json.dump(rows, open(out / "canvas.json", "w"), indent=1)
