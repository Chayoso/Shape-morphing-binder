"""d154_mesh_read.py OUT_JSONL JOBS_FILE [FIRST LAST] [yard] -- D154 (1): a floor-free reading of a state's exterior.

The referee is the target MESH in the run's own frame (mesh_frames.py: the frame that reproduces the run's own target
sample bit for bit), not a sample of it, so the reference has no sampling noise. CPU only (torch, open3d's Embree ray
caster, skimage); measurement only. One JSON line per job.

JOBS_FILE lines: `MESH SEED LABEL NPZ RAW`; RAW = an integer raw frame index of NPZ's kept frames, `end` (NPZ's last
frame), `tgt` (NPZ's target sample: the run's own, a perfect placement of the particles) or `ind` (D90's independent
sample of the mesh, output/gpu/d90/MESH_ind300k.npz).

The state's exterior is the yardstick's (render_terms_probe.py, repo_r77, the code imported here): the zero set of Zhu
and Bridson's field of the particles (radius 3, offset 0.8 pitches; pitch = 0.708 x the independent sample's median
8th-neighbour distance, as the yardstick's), on the yardstick's views (6 azimuths x elevations 0, +-0.5), its
orthographic box (1.25 x the independent sample's largest |x|).

(A) `yard_*`: the yardstick's own definitions at 96 px: the exterior discs (one per lattice cell of pitch h the zero set
    crosses) drawn by the soft operators (d_exterior: the asymmetric silhouette w_hole 2 / w_spray 1, the headlight
    shading of the discs) against the target pictures drawn by the same operators from discs of
    - `ind`: the independent sample's zero set (the yardstick as it is: read against a sample), and
    - `mesh`: the mesh itself, on the same lattice: one disc per lattice cell whose corners the mesh's signed distance
      puts on both sides, the cell's centre moved to the nearest point of the mesh (kept where it stays within half a
      cell along every axis, Lattice.discs' rule), the normal that face's outward normal. No sample is drawn.
(B) `exact_R`: both surfaces ray-cast through the centre of every pixel at R px (96 = the yardstick's, and finer): the
    state's zero set as a triangle mesh (marching cubes of the field at half a pitch, exact field values in a band of
    one pitch around the zero set; vertex normals = the field's gradient there), the target mesh as it is. Per state,
    pooled over the 18 views: 1 - IoU of the two binary silhouettes, the missing / extra parts (shares of the mesh's
    silhouette), the mean contour offset (the two silhouettes' symmetric difference over the mesh's contour length, in
    pitches), the mean angle between the two surfaces' normals where both are hit, the headlight shading error (mean
    square over the union; shade = ambient 0.25 + 0.75 max(n . view, 0), 0 off the surface) and the depth offset where
    both are hit (state minus mesh along the ray, pitches; > 0 = the state's surface behind the mesh's).
(C) `dist`: from the 384-px casts, the visible surface points of each against the other surface: the state's to the
    mesh (signed by the mesh's inside, five-ray majority), the mesh's to the state's triangle mesh, in pitches.
`yard` as the last argument: (A) only."""
import json
import sys
import time

import numpy as np

B = "/data/relcfd/chayo/physmorph_v2"
G = f"{B}/output/gpu"
sys.path.insert(0, f"{B}/repo_r77")                            # the yardstick's code, as D145's read300.sh
import open3d as o3d                                           # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from scipy.spatial import cKDTree                              # noqa: E402
from skimage.measure import marching_cubes, perimeter          # noqa: E402
from scipy.ndimage import binary_erosion, distance_transform_edt  # noqa: E402
from physmorph.losses.silhouette import set_kernel             # noqa: E402
from physmorph.pipeline.config import PipelineConfig           # noqa: E402
from physmorph.pipeline.render_loss import d_exterior, exterior_targets, make_views  # noqa: E402
from physmorph.render.exterior import Lattice, ZhuBridson      # noqa: E402

torch.set_grad_enabled(False)
import os                                                      # noqa: E402
NT = int(os.environ.get("D154_THREADS", "6"))                 # threads of every CPU kernel here (a shared server)
torch.set_num_threads(NT)
MESHES = f"{G}/d154/meshes"
RES = (96, 384, 1024)
cfg = PipelineConfig()
res_y = cfg.render_res_hi                                      # the yardstick's 96 px (repo_r77)
set_kernel("cic")
views = make_views(cfg.render_views, cfg.render_elevs)
AMB = cfg.pbr_ambient
args = sys.argv[1:]
yard_only = args[-1] == "yard"
if yard_only:
    args = args[:-1]
out_path, jobs_path = args[0], args[1]
jobs = [ln.split() for ln in open(jobs_path) if ln.strip() and not ln.startswith("#")]
if len(args) >= 4:
    jobs = jobs[int(args[2]):int(args[3])]


def median_kth_spacing(x, kth, subsample, seed=0):
    """physmorph.gpu.median_kth_spacing on the CPU: the median kth-neighbour distance of a seeded subset, rescaled."""
    n = len(x)
    sub, scale = x, 1.0
    if subsample is not None and n > subsample:
        sub = x[np.random.default_rng(seed).choice(n, subsample, replace=False)]
        scale = (subsample / n) ** (1.0 / 3.0)
    d = cKDTree(sub).query(sub, kth + 1)[0][:, kth].astype(np.float32)
    return float(np.median(d.astype(np.float64))) * scale


def basis(theta, phi):
    right = np.array([np.cos(theta), 0.0, -np.sin(theta)])
    up = np.array([-np.sin(phi) * np.sin(theta), np.cos(phi), -np.sin(phi) * np.cos(theta)])
    return right, up, np.cross(right, up)                      # d: toward the camera


class Surface:
    """A triangle mesh ray-cast through pixel centres; normals per hit (a function of the hit)."""

    def __init__(self, V, F, vnormals=None):
        self.V, self.F = np.asarray(V, np.float64), np.asarray(F, np.int64)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(self.V.astype(np.float32)), o3d.core.Tensor(self.F.astype(np.uint32)))
        fn = np.cross(self.V[self.F[:, 1]] - self.V[self.F[:, 0]], self.V[self.F[:, 2]] - self.V[self.F[:, 0]])
        self.fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-30)
        self.vn = vnormals

    @property
    def facing_share(self):
        """The share of the first hits whose face normal (by the winding) faces the camera: ~1 = wound outward."""
        f = getattr(self, "_facing", [0, 0])
        return f[0] / max(f[1], 1)

    def cast(self, extent, res):
        """Per view: (hit mask, unit normal facing the camera (state: the field's gradient), depth along the ray
        from the camera plane, hit points)."""
        px = 2.0 * extent / res
        c = -extent + (np.arange(res) + 0.5) * px
        cu, cv = np.meshgrid(c, c, indexing="ij")              # [i, j]: i along `right`, j along `up`
        outs = []
        for th, ph in views:
            r, u, d = basis(th, ph)
            o = cu[..., None] * r + cv[..., None] * u + 3.0 * extent * d
            rays = np.concatenate([o.reshape(-1, 3), np.broadcast_to(-d, (res * res, 3))], 1).astype(np.float32)
            ans = self.scene.cast_rays(o3d.core.Tensor(rays), nthreads=NT)
            t = ans["t_hit"].numpy().astype(np.float64)
            hit = np.isfinite(t)
            pid = ans["primitive_ids"].numpy().astype(np.int64)
            n = np.zeros((res * res, 3))
            if self.vn is None:                                # the mesh: its faces' normals, turned to the camera
                n[hit] = self.fn[pid[hit]]
                fd = n[hit] @ d
                self._facing = getattr(self, "_facing", [0, 0])
                self._facing[0] += int((fd > 0).sum())
                self._facing[1] += int(hit.sum())
                n[hit] *= np.sign(fd)[:, None] + (fd == 0)[:, None]
            pts = rays[:, :3].astype(np.float64) - t[:, None] * d
            if self.vn is not None:                            # the state: the field's gradient, barycentric
                uv = ans["primitive_uvs"].numpy().astype(np.float64)[hit]
                tri = self.F[pid[hit]]
                w = np.concatenate([1 - uv[:, :1] - uv[:, 1:], uv], 1)
                g = (w[:, :, None] * self.vn[tri]).sum(1)
                n[hit] = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-30)
                if hit.any():                                  # the barycentric convention, checked on the hits
                    self.uv_err = max(getattr(self, "uv_err", 0.0),
                                      float(np.abs((w[:, :, None] * self.V[tri]).sum(1) - pts[hit]).max()))
            outs.append((hit.reshape(res, res), n.astype(np.float32).reshape(res, res, 3),
                         t.astype(np.float32).reshape(res, res), pts.reshape(res, res, 3) if res == 384 else None, d,
                         np.unique(pid[hit]) if self.vn is None else None))
        return outs


def shade(hit, n, d):
    return np.where(hit, AMB + (1 - AMB) * np.clip(n @ d, 0, None), 0.0)


def signed_dt(mask):
    """Signed distance (pixels) of each pixel centre to the mask's outline (half a pixel beyond its outermost pixel
    centres): > 0 outside the mask, < 0 inside."""
    return np.where(mask, .5 - distance_transform_edt(mask), distance_transform_edt(~mask) - .5)


def contour(mask):
    return mask & ~binary_erosion(mask)


def compare(S, T, res, extent, a):
    """(B) between the state's casts S and the mesh's T at one resolution, pooled over the views. The contour offsets:
    each silhouette's boundary pixels read on the other's signed distance (> 0 = outside it), in pitches; their mean
    is the silhouette's uniform dilation, their spread what remains of the outline's shape."""
    px = 2.0 * extent / res
    xor = uni = inter = miss = extra = tgt_px = 0
    per = 0.0
    ang, dep, sh, cs, ct = [], [], 0.0, [], []
    for (hs, ns, ts, _, d, _), (ht, nt, tt, _, _, _) in zip(S, T):
        xor += int((hs ^ ht).sum()); uni += int((hs | ht).sum()); both = hs & ht; inter += int(both.sum())
        miss += int((ht & ~hs).sum()); extra += int((hs & ~ht).sum()); tgt_px += int(ht.sum())
        per += perimeter(ht, neighborhood=4)
        c = np.clip((ns[both] * nt[both]).sum(-1), -1, 1)
        ang.append(np.degrees(np.arccos(c)))
        dep.append(ts[both] - tt[both])
        sh += float(((shade(hs, ns, d) - shade(ht, nt, d))[hs | ht] ** 2).sum())
        cs.append(signed_dt(ht)[contour(hs)] + .5)             # the state's outline (half a pixel beyond its outer
        ct.append(signed_dt(hs)[contour(ht)] + .5)             #   pixels) against the mesh's, and the converse
    ang, dep = np.concatenate(ang), np.concatenate(dep) / a
    cs, ct = np.concatenate(cs) * px / a, np.concatenate(ct) * px / a
    return dict(one_minus_iou=xor / max(uni, 1), missing=miss / max(tgt_px, 1), extra=extra / max(tgt_px, 1),
                contour_offset_a=(xor * px * px) / max(per * px, 1e-30) / a,
                outline_state_mean_a=float(cs.mean()), outline_state_sd_a=float(cs.std()),
                outline_mesh_mean_a=float(ct.mean()), outline_mesh_sd_a=float(ct.std()),
                outline_abs_two_sided_a=float(.5 * (np.abs(cs).mean() + np.abs(ct).mean())),
                normal_deg=float(ang.mean()), normal_deg_median=float(np.median(ang)), shading_mse=sh / max(uni, 1),
                depth_mean_a=float(dep.mean()), depth_absmed_a=float(np.median(np.abs(dep))),
                depth_far_share=float((np.abs(dep) > 1.0).mean()), px_over_a=px / a)


def zb_mesh(field, x, a):
    """The field's zero set as a triangle mesh: marching cubes at half a pitch; the field read exactly at the coarse
    nodes (a pitch) within its radius of a particle and at the fine nodes of every coarse cell the zero set crosses and
    the ring around those; elsewhere the coarse values, trilinear (only their sign reaches the zero set)."""
    hm, hc = 0.5 * a, a
    R = field.radius
    lo = x.min(0) - (R + 2 * hc)
    nc = np.ceil((x.max(0) + (R + 2 * hc) - lo) / hc).astype(int) + 1
    gi = [np.arange(k) for k in nc]
    nodes = np.stack(np.meshgrid(*gi, indexing="ij"), -1).reshape(-1, 3)
    q = lo + hc * nodes
    near = cKDTree(x).query(q, 1, distance_upper_bound=R, workers=NT)[0] < R
    fc = np.full(len(q), R, np.float64)
    f, _, _ = field(torch.as_tensor(q[near], dtype=torch.float32))
    fc[near] = np.nan_to_num(f.numpy().astype(np.float64), posinf=R)
    fc = fc.reshape(*nc)
    s = fc > 0
    crossed = np.zeros(nc - 1, bool)
    for o in np.ndindex(2, 2, 2):
        sl = tuple(slice(oi, oi + k - 1) for oi, k in zip(o, nc))
        crossed |= s[sl] != s[tuple(slice(0, k - 1) for k in nc)]
    from scipy.ndimage import binary_dilation
    band = binary_dilation(crossed, structure=np.ones((3, 3, 3), bool))
    nf = 2 * (nc - 1) + 1
    ff = nnf.interpolate(torch.as_tensor(fc, dtype=torch.float32)[None, None], size=tuple(int(v) for v in nf),
                         mode="trilinear", align_corners=True)[0, 0].numpy().astype(np.float64)
    cells = np.argwhere(band)
    fine = np.unique((2 * cells[:, None, :] + np.stack(np.meshgrid(*[np.arange(3)] * 3, indexing="ij"), -1)
                      .reshape(1, -1, 3)).reshape(-1, 3), axis=0)
    f, _, _ = field(torch.as_tensor(lo + hm * fine, dtype=torch.float32))
    ff[fine[:, 0], fine[:, 1], fine[:, 2]] = np.nan_to_num(f.numpy().astype(np.float64), posinf=R)
    verts, faces, _, _ = marching_cubes(ff, level=0.0, spacing=(hm, hm, hm))
    verts = verts + lo
    _, g, _ = field(torch.as_tensor(verts, dtype=torch.float32))
    g = g.numpy().astype(np.float64)
    vn = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-30)
    return verts, faces, vn, dict(coarse_read=int(near.sum()), fine_read=len(fine), verts=len(verts), faces=len(faces))


class Ref:
    """Per mesh: the yardstick's constants from the independent sample and its pictures; per mesh-seed: the mesh,
    its discs on the yardstick's lattice and pictures, its casts."""
    _ind, _seed = {}, {}

    @classmethod
    def mesh_const(cls, mesh):
        if mesh not in cls._ind:
            ind = np.asarray(np.load(f"{G}/d90/{mesh}_ind300k.npz")["tgt"], np.float32)
            tgt = torch.as_tensor(ind)
            extent = float(tgt.abs().max()) * 1.25
            sp_t = median_kth_spacing(ind, 8, 20000)
            pitch = .708 * sp_t
            center = tgt.mean(0)
            lattice = Lattice(center, 2.8 * float((tgt - center).norm(dim=1).max()))
            h = min(extent / res_y, .92 * pitch)
            c = dict(ind=ind, extent=extent, pitch=pitch, lattice=lattice, h=h)
            p, n = discs(c, tgt)
            c["e_ind"] = exterior_targets(p, n, views, res_y, extent, cfg.sil_k, AMB)
            cls._ind[mesh] = c
        return cls._ind[mesh]

    @classmethod
    def seed_const(cls, mesh, seed):
        key = (mesh, seed)
        if key not in cls._seed:
            cls._seed.clear()                                  # one mesh-seed at a time (the casts are large)
            c = cls.mesh_const(mesh)
            z = np.load(f"{MESHES}/{mesh}_{seed}.npz")
            surf = Surface(z["V"], z["F"])
            casts = {r: surf.cast(c["extent"], r) for r in ((1024,) if yard_only else RES)}
            seen = np.unique(np.concatenate([o[5] for o in casts[1024]]))
            p, n, info = mesh_discs(c, surf, seen)
            e_mesh = exterior_targets(torch.as_tensor(p, dtype=torch.float32), torch.as_tensor(n, dtype=torch.float32),
                                      views, res_y, c["extent"], cfg.sil_k, AMB)
            info.update(faces=len(surf.F), faces_seen=len(seen), facing_share=surf.facing_share)
            cls._seed[key] = dict(surf=surf, e_mesh=e_mesh, casts=casts, mesh_discs=info)
        return cls._seed[key]


def discs(c, x):
    p, g, _, _ = c["lattice"].discs(ZhuBridson(x, c["pitch"]), c["h"], refine=False)
    return p, nnf.normalize(g, dim=1)


def mesh_discs(c, surf, seen):
    """The mesh's discs on the yardstick's lattice (refine=False): the lattice cells of pitch h near the independent
    sample's cells whose corners the mesh's signed distance (five-ray majority) puts on both sides, the centre moved to
    the mesh's nearest point, kept where it stays within half a cell along every axis (Lattice.discs' rule) and lies on
    a face that one of the 18 views sees first at 1024 px (the outer surface: the teapot's closed cavity, which its
    sample fills, and the faces inside the bunny's and the dragon's open bases are not the target's exterior); the
    normal is that face's, its winding oriented once for the mesh by the first hits (facing_share)."""
    lat, h = c["lattice"], c["h"]
    from physmorph.render.exterior import key
    x = torch.as_tensor(c["ind"])
    nodes = lat.near_nodes(x, h, 2)
    q = lat.at(nodes, h).numpy().astype(np.float32)
    sd = surf.scene.compute_signed_distance(o3d.core.Tensor(q), nthreads=NT, nsamples=5).numpy().astype(np.float64)
    keys = key(nodes)
    order = torch.argsort(keys)
    cells = lat.crossed(nodes, keys[order], torch.as_tensor(sd)[order])
    cen = lat.at(cells.float() + .5, h).numpy().astype(np.float32)
    ans = surf.scene.compute_closest_points(o3d.core.Tensor(cen), nthreads=NT)
    cp = ans["points"].numpy().astype(np.float64)
    pid = ans["primitive_ids"].numpy().astype(np.int64)
    stay = np.abs(cp - cen).max(1) <= .5 * h
    ok = stay & np.isin(pid, seen)
    n = surf.fn[pid] * (1.0 if surf.facing_share >= .5 else -1.0)
    return cp[ok], n[ok], dict(nodes=len(q), crossed=len(cen), stay=int(stay.sum()), kept=int(ok.sum()))


def state_x(npz, raw, mesh):
    if raw == "ind":
        return np.asarray(np.load(f"{G}/d90/{mesh}_ind300k.npz")["tgt"], np.float32)
    z = np.load(npz, allow_pickle=True)
    if raw == "tgt":
        return np.asarray(z["tgt"], np.float32)
    if raw == "end":
        return np.asarray(z["frames"][-1], np.float32)
    raws = [int(r) for r in z["raws"]]
    return frame_of(npz, raws.index(int(raw)))


def frame_of(npz, i):
    """Kept frame i of an (uncompressed) archive, read from its place in the file: the archive is not loaded."""
    import zipfile
    from numpy.lib import format as npf
    with zipfile.ZipFile(npz) as zf, zf.open("frames.npy") as f:
        version = npf.read_magic(f)
        shape, fortran, dtype = npf._read_array_header(f, version)
        assert not fortran and len(shape) == 3
        row = int(np.prod(shape[1:])) * dtype.itemsize
        f.seek(f.tell() + i * row)
        return np.frombuffer(f.read(row), dtype).reshape(shape[1:]).astype(np.float32)


with open(out_path, "a") as out:
    for mesh, seed, label, npz, raw in jobs:
        t0 = time.time()
        seed = int(seed)
        c, s = Ref.mesh_const(mesh), Ref.seed_const(mesh, seed)
        xs = state_x(npz, raw, mesh)
        x = torch.as_tensor(xs)
        a = c["pitch"]
        field = ZhuBridson(x, a)
        p, g, _, _ = c["lattice"].discs(field, c["h"], refine=False)
        n = nnf.normalize(g, dim=1)
        yi = d_exterior(p, n, *c["e_ind"], views, res_y, c["extent"], cfg.sil_k, cfg.w_hole, cfg.w_spray, AMB)
        ym = d_exterior(p, n, *s["e_mesh"], views, res_y, c["extent"], cfg.sil_k, cfg.w_hole, cfg.w_spray, AMB)
        row = dict(mesh=mesh, seed=seed, label=label, raw=raw, npz=npz.split("/")[-1], pitch=a, discs=len(p),
                   mesh_discs=s["mesh_discs"],
                   yard_ind=dict(sil=float(yi[0]), pbr=float(yi[1])), yard_mesh=dict(sil=float(ym[0]), pbr=float(ym[1])))
        row["t_yard"], t1 = time.time() - t0, time.time()
        if not yard_only:
            V, F, vn, info = zb_mesh(field, xs.astype(np.float64), a)
            st = Surface(V, F, vn)
            row["zb_mesh"] = info
            S = {r: st.cast(c["extent"], r) for r in RES}
            for r in RES:
                row[f"exact_{r}"] = compare(S[r], s["casts"][r], r, c["extent"], a)
            ps = np.concatenate([o[3][o[0]] for o in S[384]])                  # the state's visible surface
            ns = np.concatenate([o[1][o[0]] for o in S[384]]).astype(np.float64)
            pt = np.concatenate([o[3][o[0]] for o in s["casts"][384]])         # the mesh's visible surface
            ms = s["surf"].scene
            P32 = o3d.core.Tensor(ps.astype(np.float32))
            cl = ms.compute_closest_points(P32, nthreads=NT)
            du = np.linalg.norm(cl["points"].numpy().astype(np.float64) - ps, axis=1)
            occ = ms.compute_occupancy(P32, nthreads=NT, nsamples=5).numpy() > 0.5
            dsg = np.where(occ, -du, du) / a                                     # > 0: outside the mesh
            nm = s["surf"].fn[cl["primitive_ids"].numpy().astype(np.int64)]
            ang = np.degrees(np.arccos(np.clip(np.abs((ns * nm).sum(1)), 0, 1)))  # unsigned: the winding plays no part
            dt = st.scene.compute_distance(o3d.core.Tensor(pt.astype(np.float32)), nthreads=NT).numpy().astype(np.float64) / a
            fpt, _, _ = field(torch.as_tensor(pt, dtype=torch.float32))
            dts = np.where(fpt.numpy() < 0, -dt, dt)                             # > 0: the mesh outside the state
            row["dist"] = dict(state_to_mesh_signed_mean=float(dsg.mean()), state_to_mesh_signed_sd=float(dsg.std()),
                               state_to_mesh_abs_mean=float(np.abs(dsg).mean()),
                               state_to_mesh_rms=float(np.sqrt((dsg ** 2).mean())),
                               state_to_mesh_p95=float(np.quantile(np.abs(dsg), .95)),
                               mesh_to_state_signed_mean=float(dts.mean()), mesh_to_state_signed_sd=float(dts.std()),
                               mesh_to_state_abs_mean=float(dt.mean()), mesh_to_state_rms=float(np.sqrt((dt ** 2).mean())),
                               mesh_to_state_p95=float(np.quantile(dt, .95)),
                               two_sided_mean=float(.5 * (np.abs(dsg).mean() + dt.mean())),
                               normal_closest_deg=float(ang.mean()), normal_closest_deg_median=float(np.median(ang)),
                               n_state=len(ps), n_mesh=len(pt))
            row["uv_err_a"] = getattr(st, "uv_err", 0.0) / a
            row["t_exact"] = time.time() - t1
        row["seconds"] = time.time() - t0
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row), flush=True)
