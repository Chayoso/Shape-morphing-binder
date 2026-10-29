# d157_lib.py -- D156: a snapshot of scripts/probes/paper/d157/d157_read.py at 14a419b (D157's reader, its helpers imported by d156_read.py; main() not used).
"""d157_read.py OUT_JSONL JOBS_FILE [FIRST LAST] -- D157: standard metrics of a state's surfaces against the exact mesh.

Measurement only, CPU (open3d's Embree ray caster, skimage, lpips on the CPU). One JSON line per job.
JOBS_FILE lines `MESH SEED N LABEL NPZ RAW` (N = 300k | 40k; RAW = `end`: the archive's last frame, `tgt`: its target
sample, the perfect placement). A mesh-seed's jobs must be consecutive (its context is built once).

The referee is the target mesh in the run's own frame (D154's mesh_frames.py: d154/meshes, d154/meshes40); an open mesh
(bunny, dragon, beast, maxplanck, ogre) has its boundary loops closed by a fan from each loop's centroid, the volume the
sampler fills; the closed mesh is checked against the run's target sample (the share of its points inside). Two
referees: raw (the mesh) and corrected (the mesh grown by c: the iso-surface at distance c outside it, marching cubes at
a / 2 of exact distances, the inner sheet removed; c = the own sample's exterior's mean signed offset from the mesh).

A state's surfaces: `zb` (Zhu and Bridson's zero set, radius 3, offset 0.8 a: the exterior every metric and the loss
read) and `zbs` (with Solenthaler et al.'s correction: the display), marching cubes at a / 2 with exact field values in
the band of the zero set (D154's zb_mesh), normals the field's gradient. Constants from the run's own target sample:
pitch a = 0.708 x its median 8th-neighbour distance, box 1.25 x its largest |x|.

Views: the loss's 18 training views and 46 held-out ones (a 64-point Fibonacci sphere minus 12 deg caps around the
training directions and their antipodes; `far`: > 25 deg from every training line).
(1) `sil`: orthographic silhouettes ray-cast through pixel centres at 256 / 512 / 1024 px; per view 1 - IoU, Boundary
    IoU (Cheng et al. 2021: mask minus its k-fold 3x3 erosion, k = round(0.02 x diagonal), and k = one pitch), the
    boundary F-score (DAVIS's seg2bmap boundaries matched within ceil(0.008 x diagonal) px, and within 0.5 / 1 pitch),
    the contour Chamfer (half the sum of the two directed mean distances), Hausdorff and HD95 (the 95th percentile of
    the pooled directed distances), in pitches; the means over the training, held-out and far views.
(2) `d3`: the visible exterior of each surface (faces hit first in any of the 64 views at 1024 px), NPTS area-uniform
    points, exact distances to the other triangle mesh: F-score at TAUS pitches, Chamfer-L1, normal consistency; the
    recall within one pitch on the mesh's thin part (shape diameter along -n below two MPM cells).
(4a) `region`: the signed offset of the state's points from the raw mesh (+ outside, pitches) by the region of their
    nearest mesh point: crevice (a ray out along the normal meets the mesh within 6 pitches), rim (within one render
    pixel of the run's own resolution of the mesh's outline in a training view, |n . d| < 0.5 there), head-on.
(3) `img`: the held-out views as studio cameras (render/studio.py's camera, 3.6 radii, 30 deg; its shading: ambient,
    three GGX lights fixed to the camera, the backdrop, the tone map, sRGB) on ray-cast normals, 2 x 2 supersampled to
    IMG px: the display surface (zbs) against each referee (the mesh with crease-aware corner normals, 30 deg; the
    grown mesh with its distance gradient); shaded images and camera-space normal maps: PSNR, SSIM, LPIPS (AlexNet,
    v0.1), and the mean angle between the normals where both cover."""
import json
import os
import sys
import time

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as nnf
import trimesh
from scipy.ndimage import binary_dilation, distance_transform_cdt, distance_transform_edt
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from skimage.metrics import structural_similarity

from physmorph.pipeline.render_loss import make_views
from physmorph.render.exterior import ZhuBridson, ZhuBridsonSolenthaler

torch.set_grad_enabled(False)
NT = int(os.environ.get("D157_THREADS", "6"))
torch.set_num_threads(NT)
G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
RES = (256, 512, 1024)
TAUS = (0.25, 0.5, 1.0, 2.0)
NPTS = int(os.environ.get("D157_NPTS", "200000"))
IMG = int(os.environ.get("D157_IMG", "512"))
RUN_RES = {"300k": 188, "40k": 96}                              # the runs' own render resolution (pipeline_run.py)
MESHDIR = {"300k": f"{G}/d154/meshes", "40k": f"{G}/d154/meshes40"}
SURFS = ("zb", "zbs")


def dvec(th, ph):
    return np.array([np.sin(th) * np.cos(ph), np.sin(ph), np.cos(th) * np.cos(ph)])


def basis(th, ph):
    right = np.array([np.cos(th), 0.0, -np.sin(th)])
    up = np.array([-np.sin(ph) * np.sin(th), np.cos(ph), -np.sin(ph) * np.cos(th)])
    return right, up, np.cross(right, up)                      # d: toward the camera (= dvec)


TRAIN = make_views(6, (0.0, 0.5, -0.5))


def heldout(n=64, cap=12.0, far=25.0):
    i = np.arange(n) + 0.5
    pol = np.arccos(1 - 2 * i / n)
    az = np.pi * (1 + 5 ** 0.5) * i
    P = np.stack([np.cos(az) * np.sin(pol), np.cos(pol), np.sin(az) * np.sin(pol)], 1)
    tr = np.array([dvec(t, p) for t, p in TRAIN])
    ang = np.degrees(np.arccos(np.clip(np.abs(P @ tr.T), 0, 1))).min(1)
    keep = ang > cap
    return ([(float(np.arctan2(p[0], p[2])), float(np.arcsin(np.clip(p[1], -1, 1)))) for p in P[keep]],
            ang[keep] > far)


HELD, FAR = heldout()
VIEWS = TRAIN + HELD
NTR = len(TRAIN)
SETS = {"train": np.arange(NTR), "held": NTR + np.arange(len(HELD)), "far": NTR + np.flatnonzero(FAR)}


def median_kth_spacing(x, kth, subsample, seed=0):
    """physmorph.gpu.median_kth_spacing on the CPU (D154's port)."""
    n = len(x)
    sub, scale = x, 1.0
    if subsample is not None and n > subsample:
        sub = x[np.random.default_rng(seed).choice(n, subsample, replace=False)]
        scale = (subsample / n) ** (1.0 / 3.0)
    d = cKDTree(sub).query(sub, kth + 1)[0][:, kth].astype(np.float32)
    return float(np.median(d.astype(np.float64))) * scale


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


class Surf:
    """A triangle mesh in an Embree scene; per-hit normals from per-vertex (vn) or per-corner (cn) normals, else the
    face's."""

    def __init__(self, V, F, vn=None, cn=None):
        self.V, self.F = np.asarray(V, np.float64), np.asarray(F, np.int64)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(self.V.astype(np.float32)), o3d.core.Tensor(self.F.astype(np.uint32)))
        fn = np.cross(self.V[self.F[:, 1]] - self.V[self.F[:, 0]], self.V[self.F[:, 2]] - self.V[self.F[:, 0]])
        self.area = 0.5 * np.linalg.norm(fn, axis=1)
        self.fn = unit(fn)
        self.vn, self.cn = vn, cn

    def normals_at(self, pid, uv):
        w = np.concatenate([1 - uv[:, :1] - uv[:, 1:], uv], 1)
        if self.cn is not None:
            g = (w[:, :, None] * self.cn[pid]).sum(1)
        elif self.vn is not None:
            g = (w[:, :, None] * self.vn[self.F[pid]]).sum(1)
        else:
            return self.fn[pid]
        return unit(g)

    def cast(self, rays):
        ans = self.scene.cast_rays(o3d.core.Tensor(np.ascontiguousarray(rays, np.float32)), nthreads=NT)
        t = ans["t_hit"].numpy()
        return (np.isfinite(t), ans["primitive_ids"].numpy().astype(np.int64),
                ans["primitive_uvs"].numpy().astype(np.float64), t)

    def occluded(self, rays):
        return self.scene.test_occlusions(o3d.core.Tensor(np.ascontiguousarray(rays, np.float32)), nthreads=NT).numpy()

    def closest(self, P):
        ans = self.scene.compute_closest_points(o3d.core.Tensor(np.ascontiguousarray(P, np.float32)), nthreads=NT)
        cp = ans["points"].numpy().astype(np.float64)
        return cp, ans["primitive_ids"].numpy().astype(np.int64), ans["primitive_uvs"].numpy().astype(np.float64)

    def signed(self, P):
        return self.scene.compute_signed_distance(o3d.core.Tensor(np.ascontiguousarray(P, np.float32)), nthreads=NT,
                                                  nsamples=3).numpy().astype(np.float64)

    def udist(self, P):
        return self.scene.compute_distance(o3d.core.Tensor(np.ascontiguousarray(P, np.float32)),
                                           nthreads=NT).numpy().astype(np.float64)

    def inside(self, P):
        return self.scene.compute_occupancy(o3d.core.Tensor(np.ascontiguousarray(P, np.float32)), nthreads=NT,
                                            nsamples=3).numpy() > 0.5


def ortho_rays(view, extent, res):
    r, u, d = basis(*view)
    c = -extent + (np.arange(res) + 0.5) * (2.0 * extent / res)
    cu, cv = np.meshgrid(c, c, indexing="ij")                 # [i, j]: i along `right`, j along `up`
    o = cu[..., None] * r + cv[..., None] * u + 3.0 * extent * d
    return np.concatenate([o.reshape(-1, 3), np.broadcast_to(-d, (res * res, 3))], 1).astype(np.float32)


def silhouettes(surf, extent):
    """Per resolution the 64 views' binary masks; the faces hit first at 1024 px."""
    masks, seen = {r: [] for r in RES}, []
    for v in VIEWS:
        for r in RES:
            rays = ortho_rays(v, extent, r)
            if r == 1024:
                hit, pid, _, _ = surf.cast(rays)
                seen.append(np.unique(pid[hit]))
                masks[r].append(hit.reshape(r, r))
            else:
                masks[r].append(surf.occluded(rays).reshape(r, r))
    return masks, np.unique(np.concatenate(seen))


def bmap(m):
    """DAVIS's seg2bmap (Perazzi et al. 2016), same size."""
    e = np.zeros_like(m); s = np.zeros_like(m); se = np.zeros_like(m)
    e[:, :-1] = m[:, 1:]; s[:-1, :] = m[1:, :]; se[:-1, :-1] = m[1:, 1:]
    b = m ^ e | m ^ s | m ^ se
    b[-1, :] = m[-1, :] ^ e[-1, :]
    b[:, -1] = m[:, -1] ^ s[:, -1]
    b[-1, -1] = 0
    return b


class MaskInfo:
    """What the metrics need of one mask: the mask, its chessboard depth (Cheng's erosions), its DAVIS boundary and the
    Euclidean distance of every pixel to that boundary."""

    def __init__(self, m):
        self.m = m
        self.cd = distance_transform_cdt(np.pad(m, 1), metric="chessboard")[1:-1, 1:-1]
        self.b = bmap(m)
        self.dt = distance_transform_edt(~self.b).astype(np.float32)


def sil_metrics(S, T, res, extent, a):
    """S, T: MaskInfo of the state and the referee at res px."""
    px = 2.0 * extent / res
    diag = np.sqrt(2.0) * res
    out = {}
    inter, uni = int((S.m & T.m).sum()), int((S.m | T.m).sum())
    out["iou_err"] = 1.0 - inter / max(uni, 1)
    out["inter_px"], out["union_px"] = inter, uni              # for the pooled 1 - IoU (D154's: sum xor / sum union)
    for name, k in (("biou2pct", max(1, int(round(0.02 * diag)))), ("biou1a", max(1, int(round(a / px))))):
        bs, bt = S.m & (S.cd <= k), T.m & (T.cd <= k)
        out[name + "_err"] = 1.0 - int((bs & bt).sum()) / max(int((bs | bt).sum()), 1)
    ds, dt = T.dt[S.b], S.dt[T.b]                              # px: the state's boundary to the referee's, converse
    for name, r in (("bfdavis", np.ceil(0.008 * diag)), ("bf05a", 0.5 * a / px), ("bf1a", a / px)):
        p, rc = float((ds <= r).mean()) if len(ds) else 0.0, float((dt <= r).mean()) if len(dt) else 0.0
        out[name + "_err"] = 1.0 - (2 * p * rc / (p + rc) if p + rc > 0 else 0.0)
    k = px / a
    out["chamfer_a"] = 0.5 * (ds.mean() + dt.mean()) * k
    out["hd_a"] = max(ds.max(), dt.max()) * k
    out["hd95_a"] = float(np.percentile(np.concatenate([ds, dt]), 95)) * k
    return out


def banded_mc(read, lo, nc, hc, level, coarse):
    """Marching cubes of a scalar field at `level`: the coarse grid lo + hc * idx (nc nodes per axis) with values
    `coarse`; the fine nodes (hc / 2) of every coarse cell the level crosses and the ring around those read exactly by
    `read`, the rest trilinear from the coarse values (only their side of the level reaches the surface)."""
    s = coarse > level
    crossed = np.zeros(nc - 1, bool)
    base = tuple(slice(0, k - 1) for k in nc)
    for o in np.ndindex(2, 2, 2):
        crossed |= s[tuple(slice(oi, oi + k - 1) for oi, k in zip(o, nc))] != s[base]
    band = binary_dilation(crossed, structure=np.ones((3, 3, 3), bool))
    nf = 2 * (nc - 1) + 1
    ff = nnf.interpolate(torch.as_tensor(coarse, dtype=torch.float32)[None, None], size=tuple(int(v) for v in nf),
                         mode="trilinear", align_corners=True)[0, 0].numpy().astype(np.float64)
    cells = np.argwhere(band)
    off = np.stack(np.meshgrid(*[np.arange(3)] * 3, indexing="ij"), -1).reshape(1, -1, 3)
    fine = np.unique((2 * cells[:, None, :] + off).reshape(-1, 3), axis=0)
    hm = 0.5 * hc
    ff[fine[:, 0], fine[:, 1], fine[:, 2]] = read(lo + hm * fine)
    verts, faces, _, _ = marching_cubes(ff, level=level, spacing=(hm, hm, hm))
    return verts + lo, faces, len(fine)


def field_mesh(kind, x, a):
    """The zero set of the state's field (`zb` or `zbs`) as a triangle mesh, vertex normals its gradient."""
    xt = torch.as_tensor(x, dtype=torch.float32)
    field = (ZhuBridson if kind == "zb" else ZhuBridsonSolenthaler)(xt, a)
    R = field.radius
    hc = a
    lo = x.min(0) - (R + 2 * hc)
    nc = np.ceil((x.max(0) + (R + 2 * hc) - lo) / hc).astype(int) + 1
    nodes = np.stack(np.meshgrid(*[np.arange(k) for k in nc], indexing="ij"), -1).reshape(-1, 3)
    q = lo + hc * nodes
    near = cKDTree(x).query(q, 1, distance_upper_bound=R, workers=NT)[0] < R
    fc = np.full(len(q), R, np.float64)

    def read(p):
        f, _, _ = field(torch.as_tensor(p, dtype=torch.float32), grad=False)
        return np.nan_to_num(f.numpy().astype(np.float64), posinf=R)

    fc[near] = read(q[near])
    del q, nodes
    verts, faces, nfine = banded_mc(read, lo, nc, hc, 0.0, fc.reshape(*nc))
    _, g, _ = field(torch.as_tensor(verts, dtype=torch.float32))
    vn = unit(g.numpy().astype(np.float64))
    return Surf(verts, faces, vn=vn), dict(verts=len(verts), faces=len(faces), fine_read=nfine)


def close_holes(V, F):
    """The mesh, its boundary loops closed by a fan from each loop's centroid (a boundary edge u->v of a face gives
    the fan triangle (v, u, centroid), the winding of the face beside it)."""
    t = trimesh.Trimesh(V, F, process=True)
    if t.is_watertight:
        return t, 0
    E = np.concatenate([t.faces[:, [0, 1]], t.faces[:, [1, 2]], t.faces[:, [2, 0]]])
    _, inv, cnt = np.unique(np.sort(E, 1), axis=0, return_inverse=True, return_counts=True)
    bd = E[cnt[inv.ravel()] == 1]
    comps = trimesh.graph.connected_components(bd, min_len=1)
    Vs, Fs, k = [np.asarray(t.vertices)], [np.asarray(t.faces)], len(t.vertices)
    for comp in comps:
        e = bd[np.isin(bd[:, 0], comp)]
        Vs.append(np.asarray(t.vertices)[np.unique(e)].mean(0)[None])
        Fs.append(np.stack([e[:, 1], e[:, 0], np.full(len(e), k)], 1))
        k += 1
    return trimesh.Trimesh(np.concatenate(Vs), np.concatenate(Fs), process=True), len(comps)


def corner_normals(V, F, crease=30.0, chunk=20000):
    """Per face corner: the area-weighted mean of the normals of the faces at that vertex within `crease` degrees of
    the face's own (a crease-aware vertex normal)."""
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    A = np.linalg.norm(fn, axis=1)
    fn = fn / np.maximum(A[:, None], 1e-30)
    vf = trimesh.Trimesh(V, F, process=False).vertex_faces
    cosT = np.cos(np.radians(crease))
    out = np.zeros(F.shape + (3,))
    for s in range(0, len(F), chunk):
        sl = slice(s, s + chunk)
        for k in range(3):
            inc = vf[F[sl, k]]
            ok = inc >= 0
            ni = fn[np.where(ok, inc, 0)]
            w = A[np.where(ok, inc, 0)] * ok * ((ni * fn[sl][:, None, :]).sum(-1) >= cosT)
            out[sl, k] = unit((w[..., None] * ni).sum(1))
    return out


def grown_mesh(raw, c, a):
    """The iso-surface at distance c outside the closed mesh `raw`: marching cubes at a / 2 of exact unsigned distances,
    the inner sheet (faces whose centroid is inside the mesh) removed; vertex normals the distance's gradient."""
    hc = a
    lo = raw.V.min(0) - (c + 3 * hc)
    nc = np.ceil((raw.V.max(0) + (c + 3 * hc) - lo) / hc).astype(int) + 1
    nodes = np.stack(np.meshgrid(*[np.arange(k) for k in nc], indexing="ij"), -1).reshape(-1, 3)
    coarse = np.concatenate([raw.udist(lo + hc * nodes[i:i + 4000000]) for i in range(0, len(nodes), 4000000)])
    del nodes
    verts, faces, nfine = banded_mc(raw.udist, lo, nc, hc, c, coarse.reshape(*nc))
    cen = verts[faces].mean(1)
    keep = ~raw.inside(cen)
    t = trimesh.Trimesh(verts, faces[keep], process=True)
    t.remove_unreferenced_vertices()
    V, F = np.asarray(t.vertices, np.float64), np.asarray(t.faces, np.int64)
    cp, _, _ = raw.closest(V)
    return Surf(V, F, vn=unit(V - cp)), dict(verts=len(V), faces=len(F), inner_removed=int((~keep).sum()),
                                              fine_read=nfine, watertight=bool(t.is_watertight))


def area_sample(surf, faces, n, rng):
    p = surf.area[faces] / surf.area[faces].sum()
    f = rng.choice(faces, n, p=p)
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    uv = np.stack([s * (1 - r2), s * r2], 1)
    w = np.concatenate([1 - uv[:, :1] - uv[:, 1:], uv], 1)
    P = (w[:, :, None] * surf.V[surf.F[f]]).sum(1)
    return P, surf.normals_at(f, uv), f


def shape_diameter(raw, P, n, a):
    """The ray from P (a point of the mesh, n its outward normal) inward to the next hit: the local thickness."""
    rays = np.concatenate([P - 1e-3 * a * n, -n], 1)
    _, _, _, t = raw.cast(rays)
    return np.where(np.isfinite(t), t + 1e-3 * a, np.inf)


def outward_gap(raw, P, n, a):
    rays = np.concatenate([P + 1e-3 * a * n, n], 1)
    _, _, _, t = raw.cast(rays)
    return np.where(np.isfinite(t), t + 1e-3 * a, np.inf)


def signed_dt(mask):
    return np.where(mask, .5 - distance_transform_edt(mask), distance_transform_edt(~mask) - .5)


# ---- the studio look on ray-cast normals (render/studio.py's model) ----
LIGHTS = (((-.65, .8, -1.), (1., .91, .81), 2.6), ((.8, .15, -.7), (.73, .85, 1.), .85),
          ((.4, .75, .8), (.85, .92, 1.), 1.35))
ALBEDO = np.array((.64, .69, .72))


def studio_cam(center, radius, view, res):
    cam = center + 3.6 * radius * dvec(*view)
    fwd = unit(center - cam)
    right = unit(np.cross(np.array([0.0, 1.0, 0.0]), fwd))
    up = np.cross(fwd, right)
    tan = np.tan(np.radians(30) / 2)
    yy, xx = np.meshgrid(np.arange(res, dtype=np.float64), np.arange(res, dtype=np.float64), indexing="ij")
    rays = ((2 * (xx + .5) / res - 1)[..., None] * tan * right + (1 - 2 * (yy + .5) / res)[..., None] * tan * up
            + fwd)
    return cam, unit(rays), right, up, fwd


def render(surf, center, radius, view, res=IMG, ss=2):
    """(shaded sRGB image, normal map, coverage, world normals) of one held-out view, ss x ss supersampled."""
    cam, dirs, right, up, fwd = studio_cam(center, radius, view, res * ss)
    rays = np.concatenate([np.broadcast_to(cam, dirs.shape).reshape(-1, 3), dirs.reshape(-1, 3)], 1)
    hit, pid, uv, _ = surf.cast(rays)
    n = np.zeros((len(rays), 3))
    n[hit] = surf.normals_at(pid[hit], uv[hit])
    R = res * ss
    hit, n = hit.reshape(R, R), n.reshape(R, R, 3)
    cov = hit.reshape(res, ss, res, ss).mean((1, 3))
    ns = unit((n * hit[..., None]).reshape(res, ss, res, ss, 3).sum((1, 3)))
    _, dirs1, _, _, _ = studio_cam(center, radius, view, res)
    v = -dirs1
    nv = np.clip((ns * v).sum(-1), 1e-4, None)
    amb = .20 + .12 * (.5 + .5 * ns[..., 1])
    rad = ALBEDO * amb[..., None]
    rough = .36
    a2 = rough ** 4
    k = (rough + 1) ** 2 / 8
    vis_v = nv / (nv * (1 - k) + k)
    for d, col, inten in LIGHTS:
        l = unit(d[0] * right + d[1] * up + d[2] * fwd)
        h = unit(v + l)
        nl = np.clip((ns * l).sum(-1), 0, 1)
        nh = np.clip((ns * h).sum(-1), 0, 1)
        vh = np.clip((v * h).sum(-1), 0, 1)
        D = a2 / (np.pi * np.maximum((nh * nh * (a2 - 1) + 1) ** 2, 1e-8))
        Fr = .04 + .96 * (1 - vh) ** 5
        vis_l = nl / (nl * (1 - k) + k)
        spec = D * Fr * vis_v * vis_l / np.maximum(4 * nv * nl, 1e-5)
        diff = ALBEDO * (1 - Fr[..., None]) / np.pi
        rad = rad + (diff + spec[..., None]) * (np.array(col) * inten) * nl[..., None]
    blend = (np.arange(res) / max(1, res - 1))[:, None, None]
    bg = (1 - blend) * np.array((.115, .137, .165)) + blend * np.array((.215, .237, .265))
    lin = rad * cov[..., None] + bg * (1 - cov[..., None])
    mapped = lin / (1 + lin)
    srgb = np.clip(np.where(mapped <= .0031308, 12.92 * mapped, 1.055 * np.power(mapped, 1 / 2.4) - .055), 0, 1)
    ncam = np.stack([(ns * right).sum(-1), (ns * up).sum(-1), -(ns * fwd).sum(-1)], -1)
    nmap = (0.5 * ncam + 0.5) * cov[..., None]
    return srgb.astype(np.float32), nmap.astype(np.float32), cov, ns


_LP = None


def lpips_batch(A, B):
    global _LP
    import lpips
    if _LP is None:
        _LP = lpips.LPIPS(net="alex", version="0.1", verbose=False).eval()
    out = []
    for i in range(0, len(A), 8):
        x = torch.as_tensor(np.stack(A[i:i + 8])).permute(0, 3, 1, 2) * 2 - 1
        y = torch.as_tensor(np.stack(B[i:i + 8])).permute(0, 3, 1, 2) * 2 - 1
        out.append(_LP(x, y).flatten().numpy())
    return np.concatenate(out).astype(np.float64)


def img_metrics(S, T):
    """S, T: lists over the held-out views of (shaded, nmap, cov, ns)."""
    out = {}
    for j, kind in ((0, "shaded"), (1, "normal")):
        mse = np.array([float(((s[j] - t[j]) ** 2).mean()) for s, t in zip(S, T)])
        ssim = np.array([structural_similarity(s[j], t[j], channel_axis=2, data_range=1.0, gaussian_weights=True,
                                               sigma=1.5, use_sample_covariance=False) for s, t in zip(S, T)])
        lp = lpips_batch([s[j] for s in S], [t[j] for t in T])
        out[kind] = dict(mse=mse.tolist(), psnr=(10 * np.log10(1.0 / np.maximum(mse, 1e-12))).tolist(),
                         ssim=ssim.tolist(), lpips=lp.tolist())
    ang = []
    for s, t in zip(S, T):
        both = (s[2] >= .75) & (t[2] >= .75)
        c = np.clip((s[3][both] * t[3][both]).sum(-1), -1, 1)
        ang.append(float(np.degrees(np.arccos(c)).mean()) if both.any() else float("nan"))
    out["normal_angle_deg"] = ang
    return out


def summarize_views(per):
    """per: list over views of dicts -> {set: {metric: mean}} plus the 1024-px per-view lists kept by the caller."""
    keys = per[0].keys()
    return {sn: {k: float(np.mean([per[i][k] for i in idx])) for k in keys} for sn, idx in SETS.items()}


def d3_metrics(st, sP, sN, ref, rP, rN, a, thin_ref=None):
    """F-score, Chamfer-L1, normal consistency between the state's sample (sP, sN on st) and the referee's (rP, rN on
    ref). Signed distances of the state's points to the referee (+ outside)."""
    e = ref.signed(sP)
    _, pid, uv = ref.closest(sP)
    nr = ref.normals_at(pid, uv)                              # raw: the face's normal; grown: its distance gradient
    cp2, pid2, uv2 = st.closest(rP)
    d = np.linalg.norm(cp2 - rP, axis=1)
    ns = st.normals_at(pid2, uv2)
    out = {}
    ae = np.abs(e)
    for t in TAUS:
        p, r = float((ae <= t * a).mean()), float((d <= t * a).mean())
        out[f"f{t:g}"] = 2 * p * r / (p + r) if p + r > 0 else 0.0
        out[f"prec{t:g}"], out[f"rec{t:g}"] = p, r
    out["acc_a"], out["comp_a"] = float(ae.mean() / a), float(d.mean() / a)
    out["chamfer_l1_a"] = 0.5 * (out["acc_a"] + out["comp_a"])
    out["nc"] = 0.5 * (float(np.abs((sN * nr).sum(1)).mean()) + float(np.abs((rN * ns).sum(1)).mean()))
    out["signed_mean_a"] = float(e.mean() / a)
    if thin_ref is not None and thin_ref.any():
        out["thin_rec1"] = float((d[thin_ref] <= a).mean())
        out["thin_rec05"] = float((d[thin_ref] <= 0.5 * a).mean())
        out["thin_n"] = int(thin_ref.sum())
    return out, e


class Ctx:
    """One mesh-seed: the referees, the constants, the referees' masks, samples and pictures."""
    cur = None

    def __init__(self, mesh, seed, N, npz):
        t0 = time.time()
        self.key = (mesh, seed, N)
        z = np.load(npz, allow_pickle=True)
        tgt = np.asarray(z["tgt"], np.float32)
        self.tgt = tgt
        self.a = a = .708 * median_kth_spacing(tgt, 8, 20000)
        self.extent = float(np.abs(tgt).max()) * 1.25
        self.center = tgt.astype(np.float64).mean(0)
        self.radius = float(np.linalg.norm(tgt - self.center, axis=1).max())
        m = np.load(f"{MESHDIR[N]}/{mesh}_{seed}.npz")
        t, loops = close_holes(np.asarray(m["V"], np.float64), np.asarray(m["F"], np.int64))
        if t.volume < 0:
            t.invert()
        V, F = np.asarray(t.vertices, np.float64), np.asarray(t.faces, np.int64)
        self.raw = Surf(V, F, cn=corner_normals(V, F))
        self.raw_fn_only = Surf(V, F)
        ins = float(self.raw.inside(tgt[np.random.default_rng(0).choice(len(tgt), min(len(tgt), 50000),
                                                                         replace=False)]).mean())
        self.info = dict(loops_closed=loops, watertight=bool(t.is_watertight), winding=bool(t.is_winding_consistent),
                         faces=len(F), tgt_inside_share=ins, pitch=a, extent=self.extent)
        js = json.load(open(npz.replace("_end1.npz", ".json").replace("_frames12.npz", ".json")))
        self.dx = float(js["provenance"]["mpm"]["dx"])
        self.run_res = RUN_RES[N]
        rng = np.random.default_rng(seed)
        # own sample's exterior: the offset c
        own_zb, info_own = field_mesh("zb", tgt, a)
        own_masks, own_seen = silhouettes(own_zb, self.extent)
        P, _, _ = area_sample(own_zb, own_seen, NPTS, rng)
        self.c = float(self.raw.signed(P).mean())
        self.own_cache = ("zb", own_zb, own_masks, own_seen, info_own)
        self.info["c_a"] = self.c / a
        self.grown, gi = grown_mesh(self.raw_fn_only, self.c, a)
        self.info["grown"] = gi
        self.refs = {"raw": self.raw, "cor": self.grown}
        self.rmask, self.rseen, self.rsamp, self.rimg = {}, {}, {}, {}
        for rn, rs in self.refs.items():
            geo = self.raw_fn_only if rn == "raw" else rs           # raw: face normals (exact) for the 3D reading
            masks, seen = silhouettes(geo, self.extent)
            self.rmask[rn] = {r: [MaskInfo(mm) for mm in masks[r]] for r in RES}
            self.rseen[rn] = seen
            P, Nn, _ = area_sample(geo, seen, NPTS, rng)
            if rn == "raw":
                sd = shape_diameter(self.raw_fn_only, P, Nn, a)
            else:
                cp, pid, _ = self.raw.closest(P)
                sd = shape_diameter(self.raw_fn_only, cp, self.raw.fn[pid], a)
            self.rsamp[rn] = (P, Nn, sd < 2 * self.dx)
            self.rimg[rn] = [render(rs, self.center, self.radius, v) for v in HELD]   # raw: crease-aware normals
        gs = self.raw.signed(self.rsamp["cor"][0]) / a
        self.info["grown_offset_check_a"] = float(gs.mean())
        self.info["grown_offset_sd_a"] = float(gs.std())
        self.info["thin_share_raw"] = float(self.rsamp["raw"][2].mean())
        self.info["grown_vs_dilated_iou1024"] = float(np.mean([
            1 - ((g.m & (signed_dt(r.m) <= self.c / (2 * self.extent / 1024))).sum()
                 / max(((g.m | (signed_dt(r.m) <= self.c / (2 * self.extent / 1024))).sum()), 1))
            for g, r in zip(self.rmask["cor"][1024], self.rmask["raw"][1024])]))
        # rim maps: the raw mesh's signed outline distance in the 18 training views at 1024 px
        self.rim_sdt = [signed_dt(self.rmask["raw"][1024][i].m) for i in range(NTR)]
        self.info["ctx_seconds"] = time.time() - t0

    @classmethod
    def get(cls, mesh, seed, N, npz):
        if cls.cur is None or cls.cur.key != (mesh, seed, N):
            cls.cur = None
            cls.cur = Ctx(mesh, seed, N, npz)
        return cls.cur

    def regions(self, P, a):
        """Per point of a state: crevice / rim / head-on by its nearest point of the raw mesh."""
        cp, pid, _ = self.raw.closest(P)
        n = self.raw.fn[pid]
        crev = outward_gap(self.raw_fn_only, cp, n, a) < 6 * a
        px = 2 * self.extent / 1024
        rp = 1024 / self.run_res                                # one render pixel of the run, in 1024-px pixels
        near = np.full(len(P), np.inf)                          # the nearest training outline, render pixels
        for i, v in enumerate(TRAIN):
            r, u, d = basis(*v)
            ii = np.clip(((cp @ r + self.extent) / px).astype(int), 0, 1023)
            jj = np.clip(((cp @ u + self.extent) / px).astype(int), 0, 1023)
            dd = np.where(np.abs(n @ d) < 0.5, np.abs(self.rim_sdt[i][ii, jj]) / rp, np.inf)
            near = np.minimum(near, dd)
        rim = (near <= 1.0) & ~crev
        out = dict(crevice=crev, rim=rim, headon=~(crev | rim))
        # graded by the distance to the nearest training outline in the run's render pixels (added after the reader's
        # test showed the one-pixel band covers ~90 % of a 40k surface: 18 views, 15 distinct lines)
        edges = (0, .1, .25, .5, 1., 2., np.inf)
        for k in range(len(edges) - 1):
            out[f"bin{k}"] = (near >= edges[k]) & (near < edges[k + 1]) & ~crev
        return out


def state_x(npz, raw):
    z = np.load(npz, allow_pickle=True)
    if raw == "tgt":
        return np.asarray(z["tgt"], np.float32)
    return np.asarray(z["frames"][-1], np.float32)


def main():
    args = sys.argv[1:]
    out_path, jobs_path = args[0], args[1]
    jobs = [ln.split() for ln in open(jobs_path) if ln.strip() and not ln.startswith("#")]
    if len(args) >= 4:
        jobs = jobs[int(args[2]):int(args[3])]
    with open(out_path, "a") as out:
        for mesh, seed, N, label, npz, raw in jobs:
            t0 = time.time()
            seed = int(seed)
            ctx = Ctx.get(mesh, seed, N, npz)
            a, ext = ctx.a, ctx.extent
            x = state_x(npz, raw)
            row = dict(mesh=mesh, seed=seed, n=N, label=label, raw=raw, npz=npz.split("/")[-1], ctx=ctx.info,
                       sil={}, d3={}, region={}, img={}, sil_views={})
            rng = np.random.default_rng(seed * 7 + {"L": 1, "P": 2, "V": 1, "W": 2, "own": 3}.get(label, 4))
            for kind in SURFS:
                t1 = time.time()
                if raw == "tgt" and kind == "zb":
                    _, st, masks, seen, info = ctx.own_cache
                else:
                    st, info = field_mesh(kind, x, a)
                    masks, seen = silhouettes(st, ext)
                row[f"{kind}_mesh"] = info
                sil = {}
                for r in RES:
                    Ms = [MaskInfo(mm) for mm in masks[r]]
                    for rn in ("raw", "cor"):
                        per = [sil_metrics(Ms[i], ctx.rmask[rn][r][i], r, ext, a) for i in range(len(VIEWS))]
                        sil.setdefault(rn, {})[str(r)] = summarize_views(per)
                        if r == 1024 and kind == "zb":
                            row["sil_views"][rn] = {k: [p[k] for p in per] for k in per[0]}
                    del Ms
                row["sil"][kind] = sil
                sP, sN, _ = area_sample(st, seen, NPTS, rng)
                d3 = {}
                for rn in ("raw", "cor"):
                    rP, rN, thin = ctx.rsamp[rn]
                    ref = ctx.raw_fn_only if rn == "raw" else ctx.grown
                    d3[rn], e = d3_metrics(st, sP, sN, ref, rP, rN, a, thin)
                    if rn == "raw":
                        regs = ctx.regions(sP, a)
                        row["region"][kind] = {k: dict(share=float(m.mean()),
                                                        offset_a=float(e[m].mean() / a) if m.any() else None,
                                                        offset_abs_a=float(np.abs(e[m]).mean() / a) if m.any() else None)
                                               for k, m in regs.items()}
                row["d3"][kind] = d3
                if kind == "zbs":
                    S = [render(st, ctx.center, ctx.radius, v) for v in HELD]
                    row["img"] = {rn: img_metrics(S, ctx.rimg[rn]) for rn in ("raw", "cor")}
                    del S
                row[f"t_{kind}"] = time.time() - t1
                del st, masks
            row["seconds"] = time.time() - t0
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps(dict(mesh=mesh, seed=seed, n=N, label=label, seconds=row["seconds"],
                                  biou=row["sil"]["zb"]["cor"]["1024"]["held"]["biou2pct_err"],
                                  f05=row["d3"]["zb"]["cor"]["f0.5"], psnr=float(np.mean(row["img"]["cor"]["shaded"]["psnr"])))),
                  flush=True)


if __name__ == "__main__":
    main()
