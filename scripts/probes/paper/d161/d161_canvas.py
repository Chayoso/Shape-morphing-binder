"""d161_canvas.py OUT_DIR -- D161's smooth canvases (CPU, trimesh), the 3D targets of the two scenes and the bases of their
relief meshes:
  d161sph.obj           a sphere (icosphere, 5 subdivisions: 10242 vertices, near-equilateral faces), radius 7.25 about
                        the isosphere asset's centre (the run's frame normalises size and position, so only the shape
                        counts); the asset itself is an 80-face polyhedron whose facets would be relief of their own;
  d161smooth_bunny.obj  the bunny with its scan bumps taken out by Taubin's smoothing (lambda 0.5, mu -0.51, 300
                        iterations: pass band k_PB = 0.04, wavelengths below about 30 edges (1.8 wu) go, no shrinkage), so that the letters are the only relief beyond the
                        smooth 3D target and the bumps (D159's subject) do not mix in; the name ends in _bunny so the
                        orientation table turns it as the bunny.
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
