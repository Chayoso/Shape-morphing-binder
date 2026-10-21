"""d161_mesh.py OUT_OBJ SPEC_JSON -- D161: a relief target mesh, the smooth mesh carrying an embossed pattern along its
normals, written in the smooth mesh's own raw coordinates (the run puts it in the target's frame by sample_in_frame, as
it puts the smooth mesh). CPU only.

SPEC_JSON (one dict): base (assets/isosphere.obj | assets/bunny.obj), n (the run's N: the frame and the pitch a, d161_lib
target_frame, seed 97), text (the letters, DejaVu Sans Bold) or rings (a list of [r_in, r_out] in pitches: a logo of
annuli), height (the letters' em size or the logo's scale, pitches), amp (delta, pitches), edge (the Gaussian edge's
sigma, pitches), map (sphere: equirectangular about the vertical axis through the sphere's centre, the text centred
at longitude `lon0` deg; flank: orthographic along the body's narrower horizontal axis onto its +side, centred on the
flank point, the faces that see that side (n . lat > 0.25) and lie in front).
The mesh is subdivided (trimesh subdivide_to_size: every face split alike, no slivers) until its longest edge is below
min(edge, stroke) / 3, its vertex normals taken (area-weighted, outward); the pattern P in [0, 1] (the raster of the
letters, blurred by `edge`) is read at each vertex through the map; the vertex moves by amp (P - Pbar) n, Pbar P's area
mean over the whole surface, so the volume is the smooth mesh's to first order (and the pictures' mean brightness, Alexa
and Matusik 2010). Also written: OUT_OBJ + '.pattern.npz' (the raster, its texel, the map's constants: d161_pattern.py
reads P at any point of the frame) and OUT_OBJ + '.json' (the spec, the pitch, the volumes, the mesh's counts)."""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d161_lib import target_frame  # noqa: E402
from d161_pattern import Pattern  # noqa: E402


def raster_text(text, em, texel, pad):
    """(mask, u0, v0): the even-odd fill of the text's outline (em size `em`), centred on (0, 0), texel `texel`."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path as MPath
    from matplotlib.textpath import TextPath
    tp = TextPath((0, 0), text, size=em, prop=FontProperties(family="DejaVu Sans", weight="bold"))
    polys = tp.to_polygons()
    allv = np.concatenate(polys)
    lo, hi = allv.min(0), allv.max(0)
    c = 0.5 * (lo + hi)
    us = np.arange(lo[0] - pad, hi[0] + pad, texel) - c[0]
    vs = np.arange(lo[1] - pad, hi[1] + pad, texel) - c[1]
    U, Vv = np.meshgrid(us, vs, indexing="ij")
    pts = np.stack([U.ravel() + c[0], Vv.ravel() + c[1]], 1)
    inside = np.zeros(len(pts), bool)
    for p in polys:
        inside ^= MPath(p).contains_points(pts)
    return inside.reshape(U.shape).astype(np.float64), float(us[0]), float(vs[0])


def raster_rings(rings, texel, pad):
    rmax = max(r[1] for r in rings)
    us = np.arange(-rmax - pad, rmax + pad, texel)
    U, Vv = np.meshgrid(us, us, indexing="ij")
    r = np.hypot(U, Vv)
    m = np.zeros(U.shape)
    for a, b in rings:
        m[(r >= a) & (r < b)] = 1.0
    return m, float(us[0]), float(us[0])


def main():
    out, spec = sys.argv[1], json.load(open(sys.argv[2]))
    base, n = spec["base"], int(spec["n"])
    frame, x_t = target_frame("assets/isosphere.obj", base, n, 97)
    from scipy.spatial import cKDTree
    sub = x_t[np.random.default_rng(0).choice(len(x_t), min(20000, len(x_t)), replace=False)]
    a = 0.708 * float(np.median(cKDTree(sub).query(sub, 9)[0][:, 8])) * (len(sub) / len(x_t)) ** (1 / 3)
    texel = 0.2 * a
    edge = spec["edge"] * a
    pad = 4 * edge + 4 * texel                # the blur stays inside the raster (no step at its border)
    if "text" in spec:
        mask, u0, v0 = raster_text(spec["text"], spec["height"] * a, texel, pad)
    else:
        mask, u0, v0 = raster_rings([[r0 * a, r1 * a] for r0, r1 in spec["rings"]], texel, pad)
    tex = gaussian_filter(mask, edge / texel, mode="constant")
    from physmorph.sampling.mesh import _oriented
    from physmorph.sampling.orientation import orient_name, rotation
    raw = trimesh.load(base, process=True, force="mesh")
    raw.merge_vertices(merge_tex=True, merge_norm=True)
    o = orient_name(base)
    R = rotation(o)
    om = _oriented(base)
    # the frame: world = (R raw - offset) x scale
    Vw = (np.asarray(raw.vertices, np.float64) @ R.T - frame["offset"]) * frame["scale"]
    maxe_w = min(edge, spec.get("stroke_pitch", spec["edge"]) * a) / 3.0
    V, F = trimesh.remesh.subdivide_to_size(Vw, np.asarray(raw.faces), max_edge=maxe_w, max_iter=12)
    tm = trimesh.Trimesh(V, F, process=True)
    tm.merge_vertices()
    V, F = np.asarray(tm.vertices, np.float64), np.asarray(tm.faces)
    vn = np.asarray(tm.vertex_normals, np.float64)
    cen = x_t.mean(0).astype(np.float64)
    if np.sign(((V - cen) * vn).sum(1)).mean() < 0:
        vn = -vn
    pat = Pattern.build(spec, frame, x_t, a, tex, texel, u0, v0)
    P = pat.at(V, vn, F)
    area_v = np.zeros(len(V))
    np.add.at(area_v, F.ravel(), np.repeat(tm.area_faces / 3.0, 3))
    Pbar = float((P * area_v).sum() / area_v.sum())
    h = spec["amp"] * a * (P - Pbar)
    V2 = V + h[:, None] * vn
    # back to the raw coordinates: raw = R^T (world / scale + offset)
    Vraw = (V2 / frame["scale"] + frame["offset"]) @ R
    out_m = trimesh.Trimesh(Vraw, F, process=False)
    out_m.export(out)
    pat.save(out + ".pattern.npz")
    smooth_vol = trimesh.Trimesh(V, F, process=False).volume
    info = dict(spec=spec, pitch=a, texel=texel, Pbar=Pbar, verts=len(V), faces=len(F),
                max_edge_pitch=float(trimesh.Trimesh(V, F, process=False).edges_unique_length.max() / a),
                volume_smooth_world=float(smooth_vol), volume_relief_world=float(trimesh.Trimesh(V2, F, process=False).volume),
                watertight=bool(out_m.is_watertight), relief_area_share=float((area_v * (P > 0.5)).sum() / area_v.sum()),
                h_min_pitch=float(h.min() / a), h_max_pitch=float(h.max() / a))
    json.dump(info, open(out + ".json", "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
