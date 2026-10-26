"""d163_mesh.py -- D163: the plaque canvas and its lettered relief meshes (CPU: trimesh, scikit-image, matplotlib).

  canvas OUT_OBJ SPEC_JSON   the smooth 3D target: a plaque, the points within r of a flat rectangle L0 x W0 (axis z: flat
                             faces at +-r, a round rim of radius r), marching cubes of its exact distance (watertight).
                             Designed in raw units of one nominal pitch at N = spec n_ref: the flat face holds the longer
                             word at em `height` with `margin` em on every side, and the thickness 2 r is what makes the
                             volume c_fill x n_ref pitches^3 (c_fill = 1.27: the coin's measured V / (N a^3), D161, the
                             stratified sampler's number density against the pitch 0.708 x the 8th-neighbour distance).
  relief OUT_OBJ SPEC_JSON   D161's builder (d161_mesh.py) with the plaque map (d163_pattern.py): the front word on +z, the
                             back word on -z, each reading correctly from outside in the true geometry; embossed along the
                             smooth mesh's normals by amp (P - Pbar) x the pitch, volume-neutral; sizes (em, amp, edge) in
                             pitches of the run at N = spec pitch_n (default n), the frame at N = spec n (the run's).
Writes OUT_OBJ, OUT_OBJ + '.json' and (relief) OUT_OBJ + '.pattern.npz'."""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "d161"))
from d161_lib import target_frame  # noqa: E402
from d163_pattern import Pattern  # noqa: E402

C_FILL = 1.27


def text_polys(text, em):
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    tp = TextPath((0, 0), text, size=em, prop=FontProperties(family="DejaVu Sans", weight="bold"))
    return tp.to_polygons()


def text_box(text, em):
    allv = np.concatenate(text_polys(text, em))
    return allv.min(0), allv.max(0)


def raster_text(text, em, texel, pad):
    """(mask, u0, v0): the even-odd fill of the text's outline (em size `em`), its ink box centred on (0, 0)."""
    from matplotlib.path import Path as MPath
    polys = text_polys(text, em)
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


def stem_width(em):
    """The measured stem of 'I' at this em (its ink width)."""
    lo, hi = text_box("I", em)
    return float(hi[0] - lo[0])


def canvas(out, spec):
    E, m = float(spec["height"]), float(spec["margin"]) * float(spec["height"])
    boxes = {w: text_box(w, E) for w in (spec["front"], spec["back"])}
    wmax = max(hi[0] - lo[0] for lo, hi in boxes.values())
    hmax = max(hi[1] - lo[1] for lo, hi in boxes.values())
    L0, W0 = wmax + 2 * m, hmax + 2 * m
    A, P = L0 * W0, 2 * (L0 + W0)
    Vt = C_FILL * float(spec["n_ref"])
    # Steiner: V(r) = 2 A r + (pi / 2) P r^2 + (4 / 3) pi r^3 for the flat rectangle grown by a ball of radius r
    r = 1.0
    for _ in range(200):
        f = 2 * A * r + 0.5 * np.pi * P * r * r + 4 / 3 * np.pi * r ** 3 - Vt
        df = 2 * A + np.pi * P * r + 4 * np.pi * r * r
        r -= f / df
    h = float(spec.get("mc_h", 0.3))
    lo = np.array([-(L0 / 2 + r) - 3 * h, -(W0 / 2 + r) - 3 * h, -r - 3 * h])
    n = np.ceil((-2 * lo) / h).astype(int) + 1
    from skimage.measure import marching_cubes
    gx, gy, gz = (lo[i] + h * np.arange(n[i]) for i in range(3))
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    dxy2 = np.maximum(np.abs(X) - L0 / 2, 0.0) ** 2 + np.maximum(np.abs(Y) - W0 / 2, 0.0) ** 2
    sdf = (np.sqrt(dxy2[..., None] + gz[None, None, :] ** 2) - r).astype(np.float32)
    V, F, _, _ = marching_cubes(sdf, level=0.0, spacing=(h, h, h), allow_degenerate=False)
    del sdf
    t = trimesh.Trimesh(V + lo, F, process=True)
    t.update_faces(t.nondegenerate_faces())
    t.merge_vertices()
    t.remove_unreferenced_vertices()
    if not t.is_watertight:
        trimesh.repair.fill_holes(t)
    assert t.is_watertight, "the plaque is not watertight"
    if t.volume < 0:
        t.invert()
    t.export(out)
    exact = 2 * A * r + 0.5 * np.pi * P * r * r + 4 / 3 * np.pi * r ** 3
    info = dict(spec=spec, L0=L0, W0=W0, r=r, thickness=2 * r, length=L0 + 2 * r, height_total=W0 + 2 * r,
                text_boxes={k: [list(map(float, lo_)), list(map(float, hi_))] for k, (lo_, hi_) in boxes.items()},
                stem=stem_width(E), volume=float(t.volume), exact_volume=float(exact), verts=len(t.vertices),
                faces=len(t.faces), watertight=bool(t.is_watertight), max_edge=float(t.edges_unique_length.max()))
    json.dump(info, open(out + ".json", "w"), indent=1)
    print(json.dumps(info, indent=1))


def relief(out, spec):
    base, n = spec["base"], int(spec["n"])
    pn = int(spec.get("pitch_n", n))

    def pitch_of(x_t):
        sub = x_t[np.random.default_rng(0).choice(len(x_t), min(20000, len(x_t)), replace=False)]
        return 0.708 * float(np.median(cKDTree(sub).query(sub, 9)[0][:, 8])) * (len(sub) / len(x_t)) ** (1 / 3)

    frame, x_t = target_frame("assets/isosphere.obj", base, n, 97)
    a_run = pitch_of(x_t)
    a = a_run if pn == n else pitch_of(target_frame("assets/isosphere.obj", base, pn, 97)[1])
    texel = 0.2 * a
    edge = spec["edge"] * a
    pad = 4 * edge + 4 * texel
    rasters = {}
    for side in ("front", "back"):
        mask, u0, v0 = raster_text(spec[side], spec["height"] * a, texel, pad)
        rasters[side] = (gaussian_filter(mask, edge / texel, mode="constant"), u0, v0)
    raw = trimesh.load(base, process=True, force="mesh")
    raw.merge_vertices()
    Vw = (np.asarray(raw.vertices, np.float64) - frame["offset"]) * frame["scale"]    # orientation 'id' (d163 names)
    stroke = stem_width(spec["height"] * a)
    maxe_w = min(edge, stroke) / 3.0
    V, F = Vw, np.asarray(raw.faces)
    while trimesh.Trimesh(V, F, process=False).edges_unique_length.max() > maxe_w:
        V, F = trimesh.remesh.subdivide(V, F)
    tm = trimesh.Trimesh(V, F, process=True)
    tm.merge_vertices()
    V, F = np.asarray(tm.vertices, np.float64), np.asarray(tm.faces)
    vn = np.asarray(tm.vertex_normals, np.float64)
    cen = x_t.mean(0).astype(np.float64)
    if np.sign(((V - cen) * vn).sum(1)).mean() < 0:
        vn = -vn
    (tf, uf, vf), (tb, ub, vb) = rasters["front"], rasters["back"]
    pat = Pattern.build_plaque(x_t, a, tf, uf, vf, tb, ub, vb, texel, spec.get("offset_uv", (0.0, 0.0)))
    P = pat.at(V, vn)
    area_v = np.zeros(len(V))
    np.add.at(area_v, F.ravel(), np.repeat(tm.area_faces / 3.0, 3))
    Pbar = float((P * area_v).sum() / area_v.sum())
    h = spec["amp"] * a * (P - Pbar)
    V2 = V + h[:, None] * vn
    Vraw = V2 / frame["scale"] + frame["offset"]
    out_m = trimesh.Trimesh(Vraw, F, process=False)
    out_m.export(out)
    pat.save(out + ".pattern.npz")
    info = dict(spec=spec, pitch=a, pitch_run=a_run, texel=texel, stroke_pitch=stroke / a, stroke_pitch_run=stroke / a_run,
                Pbar=Pbar, verts=len(V), faces=len(F),
                max_edge_pitch=float(trimesh.Trimesh(V, F, process=False).edges_unique_length.max() / a),
                volume_smooth_world=float(trimesh.Trimesh(V, F, process=False).volume),
                volume_relief_world=float(trimesh.Trimesh(V2, F, process=False).volume),
                watertight=bool(out_m.is_watertight), relief_area_share=float((area_v * (P > 0.5)).sum() / area_v.sum()),
                h_min_pitch=float(h.min() / a), h_max_pitch=float(h.max() / a),
                extent_world=float(np.abs(x_t).max() * 1.25), extent_pitch=float(np.abs(x_t).max() * 1.25 / a_run),
                bbox_pitch=list(map(float, (x_t.max(0) - x_t.min(0)) / a_run)), n_sample=len(x_t))
    json.dump(info, open(out + ".json", "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    mode, out, spec = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
    {"canvas": canvas, "relief": relief}[mode](out, spec)
