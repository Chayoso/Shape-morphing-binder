"""d173_magscale.py OUT_OBJ K N -- D173 (option C): the dragon's OWN scale relief, magnified K times, on the smoothed dragon.

Construction (CPU):
  1. The real relief: H = the detailed dragon's signed offset from the smoothed dragon along the smoothed surface's normal
     (assets/dragon.obj against output/gpu/d170/assets/d170smooth_dragon.obj, the Taubin canvas; D170's size probe's H: the
     nearer of the outward and the inward ray hit, Embree), at 1M area-uniform points of the smoothed surface, in the frame
     of N (seed 97).
  2. A local surface parameterisation: D161's flank map (d161_pattern, kind "flank": orthographic along the body's narrower
     horizontal axis onto its +side, the faces that see that side, front-most within one pitch). H is rastered on the map
     (texel 0.2 pitch): each texel the mean H of the visible, facing points that fall in it (facing weight >= 0.5), gaps
     filled by linear interpolation of the filled texels.
  3. Magnified K times about the map's origin (the flank point): H_K(u, v) = K x H(u / K, v / K) -- the lateral size and the
     height both K times, so the scales keep their shape and aspect ratio.
  4. The relief mesh: the smoothed dragon, its faces that see the flank subdivided until their edges are below a third of a
     pitch (d172_mesh.py's rule), each vertex moved along its normal by m (H_K - mean_m H_K), m the map's own weight (facing
     ramp x front-most): the volume kept, nothing off the flank, no step at its border. Written in the raw coordinates with
     OUT_OBJ.pattern.npz (the map with the support raster: D172's detail reader takes its region from it) and OUT_OBJ.json."""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.interpolate import griddata

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d161")
import d157_lib as D7  # noqa: E402
from d161_lib import Surface, target_frame  # noqa: E402
from d161_pattern import Pattern, bilinear  # noqa: E402

B = "/data/relcfd/chayo/physmorph_v2"
SMOOTH = f"{B}/output/gpu/d170/assets/d170smooth_dragon.obj"
DETAIL = f"{B}/repo_d173/assets/dragon.obj"


def main():
    out, K, N = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    frame, x_t = target_frame("assets/isosphere.obj", SMOOTH, N, 97)
    xt = x_t.astype(np.float64)
    a = .708 * D7.median_kth_spacing(xt, 8, 20000)
    S = Surface(SMOOTH, frame, 1_000_000, xt)
    from physmorph.sampling.mesh import mesh_in_frame
    V, F = mesh_in_frame(DETAIL, frame)
    t, _ = D7.close_holes(V, F)
    if t.volume < 0:
        t.invert()
    det = D7.Surf(np.asarray(t.vertices, np.float64), np.asarray(t.faces, np.int64))
    H, best = np.full(S.n, np.nan), np.full(S.n, np.inf)
    for sgn in (1.0, -1.0):
        hit, _, _, tt = det.cast(np.concatenate([S.P, sgn * S.N], 1))
        tt = np.where(hit, tt, np.inf)
        better = tt < best
        H[better] = sgn * tt[better]
        best = np.minimum(best, tt)
    H[np.abs(H) > 3 * a] = np.nan
    texel = 0.2 * a
    pat = Pattern.build(dict(map="flank"), frame, x_t, a, np.ones((2, 2)), texel, 0.0, 0.0)
    u, v = pat.uv(S.P)
    lo_u, hi_u, lo_v, hi_v = u.min() - 2 * a, u.max() + 2 * a, v.min() - 2 * a, v.max() + 2 * a
    nu, nv = int(np.ceil((hi_u - lo_u) / texel)), int(np.ceil((hi_v - lo_v) / texel))
    pat.tex, pat.u0, pat.v0 = np.ones((nu, nv)), lo_u, lo_v
    m = pat.at(S.P, S.N, zpts=S.P)                       # facing x front-most, with a raster of ones
    ok = (m >= 0.5) & np.isfinite(H)
    iu = np.clip(((u[ok] - lo_u) / texel).astype(int), 0, nu - 1)
    iv = np.clip(((v[ok] - lo_v) / texel).astype(int), 0, nv - 1)
    num = np.zeros((nu, nv))
    cnt = np.zeros((nu, nv))
    np.add.at(num, (iu, iv), H[ok])
    np.add.at(cnt, (iu, iv), 1)
    filled = cnt > 0
    R = np.zeros((nu, nv))
    R[filled] = num[filled] / cnt[filled]
    gi, gj = np.nonzero(filled)
    ei, ej = np.nonzero(~filled)
    if len(ei):
        R[ei, ej] = griddata(np.stack([gi, gj], 1), R[gi, gj], np.stack([ei, ej], 1), method="linear", fill_value=0.0)
    # the magnified raster on the same grid: H_K(u, v) = K H(u / K, v / K), the origin the flank point (u = v = 0)
    U = lo_u + (np.arange(nu) + 0.5) * texel
    Vv = lo_v + (np.arange(nv) + 0.5) * texel
    UU, VV = np.meshgrid(U / K, Vv / K, indexing="ij")
    RK = K * bilinear(R, ((UU - lo_u) / texel - 0.5).ravel(), ((VV - lo_v) / texel - 0.5).ravel()).reshape(nu, nv)
    # the relief mesh (d172_mesh.py's subdivision of the faces that can carry the pattern)
    from physmorph.sampling.orientation import orient_name, rotation
    raw = trimesh.load(SMOOTH, process=True, force="mesh")
    raw.merge_vertices(merge_tex=True, merge_norm=True)
    Rm = rotation(orient_name(SMOOTH))
    Vw = (np.asarray(raw.vertices, np.float64) @ Rm.T - frame["offset"]) * frame["scale"]
    Fw = np.asarray(raw.faces)
    lat = np.asarray(pat.c["lat"], np.float64)
    cen = xt.mean(0)
    for it in range(12):
        tm = trimesh.Trimesh(Vw, Fw, process=False)
        vn = np.asarray(tm.vertex_normals, np.float64)
        if np.sign(((Vw - cen) * vn).sum(1)).mean() < 0:
            vn = -vn
        el = np.linalg.norm(Vw[Fw[:, [1, 2, 0]]] - Vw[Fw], axis=2).max(1)
        sel = np.flatnonzero((el > a / 3.0) & ((vn[Fw] @ lat).max(1) > 0.1))
        if not len(sel):
            break
        Vw, Fw = trimesh.remesh.subdivide(Vw, Fw, face_index=sel)
    tm = trimesh.Trimesh(Vw, Fw, process=True)
    tm.merge_vertices()
    Vw, Fw = np.asarray(tm.vertices, np.float64), np.asarray(tm.faces)
    vn = np.asarray(tm.vertex_normals, np.float64)
    if np.sign(((Vw - cen) * vn).sum(1)).mean() < 0:
        vn = -vn
    mv = pat.at(Vw, vn, Fw)                              # the support weight at the vertices
    uw, vw = pat.uv(Vw)
    hk = bilinear(RK, (uw - lo_u) / texel - 0.5, (vw - lo_v) / texel - 0.5)
    area_v = np.zeros(len(Vw))
    np.add.at(area_v, Fw.ravel(), np.repeat(tm.area_faces / 3.0, 3))
    mean = float((mv * hk * area_v).sum() / max((mv * area_v).sum(), 1e-30))
    h = mv * (hk - mean)
    V2 = Vw + h[:, None] * vn
    Vraw = (V2 / frame["scale"] + frame["offset"]) @ Rm
    trimesh.Trimesh(Vraw, Fw, process=False).export(out)
    pat.save(out + ".pattern.npz")                       # the support raster (ones): the reader's region
    sup = mv > 0.5
    info = dict(K=K, N=N, pitch=a, texel=texel, raster=[nu, nv], verts=len(Vw), faces=len(Fw),
                real_H_rms_pitch_on_flank=float(np.sqrt(np.nanmean(H[ok] ** 2)) / a),
                h_rms_pitch_on_support=float(np.sqrt((h[sup] ** 2).mean()) / a), h_min_pitch=float(h.min() / a),
                h_max_pitch=float(h.max() / a), support_area_share=float((area_v * sup).sum() / area_v.sum()),
                volume_smooth=float(trimesh.Trimesh(Vw, Fw, process=False).volume),
                volume_relief=float(trimesh.Trimesh(V2, Fw, process=False).volume), filled_texels=float(filled.mean()))
    json.dump(dict(info, spec=dict(K=K, N=N, base=SMOOTH, detail=DETAIL, map="flank", height=4.0 * K)),
              open(out + ".json", "w"), indent=1)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
