"""d172_mesh.py OUT_OBJ SPEC_JSON -- D172 (D): a stylized scale relief on the smoothed dragon, built as D161's relief meshes
are (tmp/d161/d161_mesh.py: the smooth mesh in the run's frame, subdivided, each vertex moved along its normal by
amp (P - Pbar), the pattern P in [0, 1] read through D161's flank map; written back in the raw coordinates with
OUT_OBJ.pattern.npz and OUT_OBJ.json). CPU only.

SPEC_JSON: base (the smoothed dragon), n (the run's N: the frame and the pitch a, seed 97), scale (the scales' spacing, pitches:
a hexagonal lattice of domes, each of diameter 0.95 x scale, profile sqrt(1 - (r / R)^2)), amp (pitches), edge (the
Gaussian blur, pitches), map "flank" (orthographic along the body's narrower horizontal axis onto its +side, the faces that
see that side), height (= scale: D161's reader takes s_bg = height / 2). The raster covers the whole body's projection on
that side. Only the faces that can carry the pattern (some vertex with n . lat > 0.1) are subdivided until their longest
edge is below edge / 3: the pattern is 0 where n . lat < 0.25, so the coarse edges at the border carry no displacement and
the T-junctions there leave no gap."""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d161")
from d161_lib import target_frame  # noqa: E402
from d161_pattern import Pattern  # noqa: E402


def raster_scales(spacing, u_lo, u_hi, v_lo, v_hi, texel, rfrac=0.475, power=0.5):
    us = np.arange(u_lo, u_hi, texel)
    vs = np.arange(v_lo, v_hi, texel)
    U, V = np.meshgrid(us, vs, indexing="ij")
    R = rfrac * spacing
    dv = spacing * np.sqrt(3) / 2
    P = np.zeros(U.shape)
    j0, j1 = int(np.floor(v_lo / dv)) - 1, int(np.ceil(v_hi / dv)) + 1
    for j in range(j0, j1 + 1):
        vc = j * dv
        sh = 0.5 * spacing * (j % 2)
        i0, i1 = int(np.floor((u_lo - sh) / spacing)) - 1, int(np.ceil((u_hi - sh) / spacing)) + 1
        rows = np.abs(vs - vc) < R
        if not rows.any():
            continue
        for i in range(i0, i1 + 1):
            uc = i * spacing + sh
            cols = np.abs(us - uc) < R
            if not cols.any():
                continue
            sub = (slice(np.flatnonzero(cols)[0], np.flatnonzero(cols)[-1] + 1), slice(np.flatnonzero(rows)[0], np.flatnonzero(rows)[-1] + 1))
            r2 = ((U[sub] - uc) ** 2 + (V[sub] - vc) ** 2) / (R * R)
            P[sub] = np.maximum(P[sub], np.clip(1 - r2, 0, None) ** power)
    return P, float(us[0]), float(vs[0])


def main():
    out, spec = sys.argv[1], json.load(open(sys.argv[2]))
    base, n = spec["base"], int(spec["n"])
    frame, x_t = target_frame("assets/isosphere.obj", base, n, 97)
    from scipy.spatial import cKDTree
    sub = x_t[np.random.default_rng(0).choice(len(x_t), min(20000, len(x_t)), replace=False)]
    a = 0.708 * float(np.median(cKDTree(sub).query(sub, 9)[0][:, 8])) * (len(sub) / len(x_t)) ** (1 / 3)
    texel = 0.2 * a
    edge = spec["edge"] * a
    pat = Pattern.build(dict(spec, map="flank"), frame, x_t, a, np.zeros((2, 2)), texel, 0.0, 0.0)
    xt = x_t.astype(np.float64)
    u, v = pat.uv(xt)
    pad = 4 * edge + 4 * texel + spec["scale"] * a
    mask, u0, v0 = raster_scales(spec["scale"] * a, u.min() - pad, u.max() + pad, v.min() - pad, v.max() + pad, texel,
                                 spec.get("rfrac", 0.475), spec.get("power", 0.5))
    pat.tex, pat.u0, pat.v0 = gaussian_filter(mask, edge / texel, mode="constant"), u0, v0
    from physmorph.sampling.orientation import orient_name, rotation
    raw = trimesh.load(base, process=True, force="mesh")
    raw.merge_vertices(merge_tex=True, merge_norm=True)
    R = rotation(orient_name(base))
    V = (np.asarray(raw.vertices, np.float64) @ R.T - frame["offset"]) * frame["scale"]
    F = np.asarray(raw.faces)
    lat = np.asarray(pat.c["lat"], np.float64)
    cen = xt.mean(0)
    maxe = edge / 3.0
    for it in range(12):
        tm = trimesh.Trimesh(V, F, process=False)
        vn = np.asarray(tm.vertex_normals, np.float64)
        if np.sign(((V - cen) * vn).sum(1)).mean() < 0:
            vn = -vn
        el = np.linalg.norm(V[F[:, [1, 2, 0]]] - V[F], axis=2).max(1)
        sel = np.flatnonzero((el > maxe) & ((vn[F] @ lat).max(1) > 0.1))
        print(f"pass {it}: {len(V)} verts, {len(F)} faces, {len(sel)} to split", flush=True)
        if not len(sel):
            break
        V, F = trimesh.remesh.subdivide(V, F, face_index=sel)
    tm = trimesh.Trimesh(V, F, process=True)
    tm.merge_vertices()
    V, F = np.asarray(tm.vertices, np.float64), np.asarray(tm.faces)
    vn = np.asarray(tm.vertex_normals, np.float64)
    if np.sign(((V - cen) * vn).sum(1)).mean() < 0:
        vn = -vn
    P = pat.at(V, vn, F)
    tex = pat.tex
    pat.tex = np.ones_like(tex)
    m = pat.at(V, vn, F)                                # the map's own weight (facing ramp x front-most), the pattern's support
    pat.tex = tex
    area_v = np.zeros(len(V))
    np.add.at(area_v, F.ravel(), np.repeat(tm.area_faces / 3.0, 3))
    Pbar = float((P * area_v).sum() / max((m * area_v).sum(), 1e-30))
    h = spec["amp"] * a * (P - Pbar * m)                # zero mean over the support (the volume kept), 0 off it, no step
    V2 = V + h[:, None] * vn
    Vraw = (V2 / frame["scale"] + frame["offset"]) @ R
    trimesh.Trimesh(Vraw, F, process=False).export(out)
    pat.save(out + ".pattern.npz")
    sup = m > 0.5
    info = dict(spec=spec, pitch=a, texel=texel, Pbar=Pbar, verts=len(V), faces=len(F),
                max_edge_pitch_pattern=float(np.linalg.norm(V[F[:, [1, 2, 0]]] - V[F], axis=2).max(1)[(P[F] > 0).any(1)].max() / a),
                volume_smooth_world=float(trimesh.Trimesh(V, F, process=False).volume),
                volume_relief_world=float(trimesh.Trimesh(V2, F, process=False).volume),
                pattern_area_share=float((area_v * sup).sum() / area_v.sum()), h_min_pitch=float(h.min() / a),
                h_max_pitch=float(h.max() / a), h_rms_pitch_on_pattern=float(np.sqrt((h[sup] ** 2).mean()) / a),
                raster=list(mask.shape))
    json.dump(info, open(out + ".json", "w"), indent=1)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
