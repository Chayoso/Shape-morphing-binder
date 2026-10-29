"""d156_read.py OUT_JSONL JOBS_FILE [FIRST LAST] -- D156: what a state recovers of the true mesh beyond its visual hull.

Measurement only, CPU (open3d's Embree, skimage, torch on the CPU; lpips from deps/d157_lpips). One JSON line per job.
Run with PYTHONPATH = repo_d156 (the sampler, sample_in_frame, the exterior operators) + deps/d157_lpips + this folder
(d157_lib.py: D157's reader at 14a419b, its helpers). JOBS_FILE lines `MESH SEED FRAME LABEL NPZ RAW`:
  FRAME  hull: a D156 run (its target is the hull, its pictures the true mesh's) | mesh: a tag run (its target the mesh)
  LABEL  L P (D156 arms) | V W (the tag's) | H M Mp (perfect samples: the hull's, the mesh's, an independent draw of the
         mesh at seed + 2); NPZ is then any run of that context, read for the frame check only
  RAW    end (the archive's last frame) | ref (for H / M / Mp)
A context (mesh, seed, frame) is built once; its jobs must be consecutive.

The frame: the run's own. load_normalized of the frame's mesh (the hull for FRAME hull, the mesh for FRAME mesh) at N,
seed + 1, matched to the source sample's volume (isosphere, N, seed), as prepare.prepare; the body's sample B must equal
the run's own target (archive `tgt`) to 1e-6 wu. The other mesh is put in that frame by sample_in_frame at B's number
density (n x its fill volume / B's, as prepare's render_tgt). H and M are B and that sample (by FRAME); Mp = the mesh at
seed + 2, the same count as M. The true mesh and the hull as triangle meshes in the frame: (oriented vertices - offset)
x scale; the true mesh's boundary loops closed by a fan for its occupancy (D157's close_holes).
Constants (the run's render operator, target.build_target): pitch a = 0.708 x M's median 8th-neighbour distance
(D154's CPU port), box = 1.25 x M's largest |x|, the exterior lattice centred on M's mean, radius 2.8 x M's largest
distance from it, h = min(box / 96, 0.92 a).
Measures of a state X (particles):
(1) `vol`: the shares of X's particles inside R = hull minus mesh (m), outside the hull, inside the mesh; the number
    density inside the mesh over the hull sample's (compression).
(2) `surf`: the context's points of the true mesh's surface (area-uniform, NS of them) on faces that a training view sees
    first at 1024 px; at each the signed distance to X's exterior (Zhu and Bridson's zero set at pitch a, D157's
    field_mesh; < 0 = inside it), summed over all points and per bin of the point's depth below the hull's surface
    (unsigned distance to the hull, pitches: 0-0.5 / 0.5-1 / 1-2 / >= 2): the share is formed by d156_table.py,
    (S_X - S_H) / (S_M - S_H); and per bin the mean angle between the exterior's normal at its closest point and the
    mesh's face normal (unsigned).
(3) `sil`: X's exterior against M's exterior, orthographic, ray-cast at pixel centres at 512 px: 1 - IoU per view over
    the 18 training and D157's 46 held-out views (means per set).
(4) `img`: X's display surface (the field with Solenthaler et al.'s correction, zbs) in D157's studio cameras on the 46
    held-out views (512 px, 2 x 2 supersampled): shaded image and normal map against the true mesh's (crease-aware
    corner normals) and against M's display surface: PSNR / SSIM / LPIPS (D157's img_metrics).
(5) `pic`: the run's own render terms: X's exterior discs (the operator's pitch and lattice) against the eight-draw
    pictures of the true mesh at 96 px (M and draws_in_frame at seeds seed + 1 + 100 k, k = 1..7, the counts of M:
    prepare's), silhouette (w_hole 2, w_spray 1) and shading (d_exterior).
(6) `pieces` (L P V W): particles outside the largest component at radius 2 x the median nearest distance."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as nnf
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d157_lib as D7                                                       # noqa: E402
from physmorph.losses.silhouette import set_kernel                          # noqa: E402
from physmorph.pipeline.config import PipelineConfig                        # noqa: E402
from physmorph.pipeline.render_loss import d_exterior, exterior_targets     # noqa: E402
from physmorph.render.exterior import Lattice, ZhuBridson                   # noqa: E402
from physmorph.sampling.mesh import (_oriented, draws_in_frame, fill_volume_in_frame, load_normalized,  # noqa: E402
                                     sample_in_frame)

torch.set_grad_enabled(False)
NT = D7.NT
N = int(os.environ.get("D156_N", "40000"))
RES_RUN = 96 if N <= 40000 else 188                                         # the run's render resolution
HULLS = os.environ.get("D156_HULLS", "/data/relcfd/chayo/physmorph_v2/output/gpu/d156/assets_hull")
NS = int(os.environ.get("D156_NS", "200000"))
SIL_RES = 512
BINS = (0.0, 0.5, 1.0, 2.0, np.inf)
cfg = PipelineConfig()
set_kernel("cic")
TRAIN, HELD = D7.TRAIN, D7.HELD


def hull_path(mesh):
    return f"{HULLS}/{mesh}_hull18_r{RES_RUN}.obj"


def to_frame(path, frame):
    m = _oriented(path)
    return (np.asarray(m.vertices, np.float64) - frame["offset"]) * frame["scale"], np.asarray(m.faces, np.int64)


def masks(surf, box, views, res):
    return [surf.occluded(D7.ortho_rays(v, box, res)).reshape(res, res) for v in views]


def iou_err(A, B):
    return [1.0 - float((a & b).sum()) / max(float((a | b).sum()), 1.0) for a, b in zip(A, B)]


class Ctx:
    cur = None

    def __init__(self, mesh, seed, kind, npz):
        t0 = time.time()
        self.key = (mesh, seed, kind)
        true_path, hp = f"assets/{mesh}.obj", hull_path(mesh)
        # the frame's mesh: the run's target (the visual hull, the mesh, or D156d's solid, kind "sfs")
        fpath = {"hull": hp, "mesh": true_path, "sfs": f"{HULLS}/{mesh}_sfs18_r{RES_RUN}.obj"}[kind]
        _, v_src = load_normalized("assets/isosphere.obj", N, seed, return_volume=True, sample="stratified")
        frame = {}
        B = load_normalized(fpath, N, seed + 1, match_volume=v_src, sample="stratified", frame=frame)
        own = np.asarray(np.load(npz, allow_pickle=True)["tgt"], np.float32)
        self.frame_maxdiff = float(np.abs(B - own).max())
        assert self.frame_maxdiff <= 1e-6, (mesh, seed, kind, self.frame_maxdiff)
        f_B = fill_volume_in_frame(fpath, frame, N)

        def other(path):                                                 # a mesh's sample in this frame at B's density
            if path == fpath:
                return B
            n_o = int(round(N * fill_volume_in_frame(path, frame, N) / f_B))
            return sample_in_frame(path, frame, n_o, seed + 1)[0]
        self.H, self.M = other(hp), other(true_path)
        self.S = B if kind == "sfs" else None                            # D156d's solid (its run's own target)
        self.n_true = len(self.M)
        self.Mp, _ = sample_in_frame(true_path, frame, self.n_true, seed + 2)
        self.frame = frame
        # the meshes in the frame
        Vt, Ft = to_frame(true_path, frame)
        tc, loops = D7.close_holes(Vt, Ft)
        if tc.volume < 0:
            tc.invert()
        self.true = D7.Surf(np.asarray(tc.vertices), np.asarray(tc.faces))
        self.true_cn = D7.Surf(np.asarray(tc.vertices), np.asarray(tc.faces),
                               cn=D7.corner_normals(np.asarray(tc.vertices, np.float64), np.asarray(tc.faces, np.int64)))
        self.vol_true = float(tc.volume)
        Vh, Fh = to_frame(hp, frame)
        th = trimesh.Trimesh(Vh, Fh, process=True)
        if th.volume < 0:
            th.invert()
        self.hull = D7.Surf(np.asarray(th.vertices), np.asarray(th.faces))
        self.vol_hull = float(th.volume)
        # the run's render operator's constants
        M = self.M
        self.a = .708 * D7.median_kth_spacing(M, 8, 20000)
        self.box = float(np.abs(M).max()) * 1.25
        Mt = torch.as_tensor(M)
        center = Mt.mean(0)
        self.lattice = Lattice(center, 2.8 * float((Mt - center).norm(dim=1).max()))
        self.h = min(self.box / RES_RUN, .92 * self.a)
        self.center = M.astype(np.float64).mean(0)
        self.radius = float(np.linalg.norm(M - self.center, axis=1).max())
        # (5) the run's pictures: M and its seven further draws (prepare's seeds and counts)
        samples = [M] + draws_in_frame(true_path, frame, self.n_true, [seed + 1 + 100 * k for k in range(1, 8)])
        per = [exterior_targets(*self.discs(s), D7.TRAIN, RES_RUN, self.box, cfg.sil_k, cfg.pbr_ambient) for s in samples]
        self.pic = ([torch.stack(v).mean(0) for v in zip(*[e[0] for e in per])],
                    [torch.stack(v).mean(0) for v in zip(*[e[1] for e in per])])
        # (2) the image-visible surface of the true mesh, its depth below the hull
        seen = []
        for v in TRAIN:
            hit, pid, _, _ = self.true.cast(D7.ortho_rays(v, self.box, 1024))
            seen.append(np.unique(pid[hit]))
        seen = np.unique(np.concatenate(seen))
        rng = np.random.default_rng(seed)
        P, Nn, _ = D7.area_sample(self.true, seen, NS, rng)
        self.P, self.Pn = P, Nn
        self.depth = self.hull.udist(P) / self.a
        self.bin = np.digitize(self.depth, BINS[1:-1])
        # (3) and (4): M's exterior masks and its display renders, the true mesh's renders
        Mzb, _ = D7.field_mesh("zb", M, self.a)
        self.Mmask = {"train": masks(Mzb, self.box, TRAIN, SIL_RES), "held": masks(Mzb, self.box, HELD, SIL_RES)}
        Mzbs, _ = D7.field_mesh("zbs", M, self.a)
        self.img_M = [D7.render(Mzbs, self.center, self.radius, v) for v in HELD]
        self.img_T = [D7.render(self.true_cn, self.center, self.radius, v) for v in HELD]
        self.info = dict(frame_maxdiff=self.frame_maxdiff, n_body=N, n_true=self.n_true, loops_closed=loops,
                         vol_true=self.vol_true, vol_hull=self.vol_hull, pitch=self.a, box=self.box, h=self.h,
                         surf_points=len(P), surf_faces_seen=len(seen),
                         depth_share=[float((self.bin == k).mean()) for k in range(len(BINS) - 1)],
                         ctx_seconds=time.time() - t0)

    def discs(self, x):
        p, g, _, _ = self.lattice.discs(ZhuBridson(torch.as_tensor(x, dtype=torch.float32), self.a), self.h, refine=False)
        return p, nnf.normalize(g, dim=1)

    @classmethod
    def get(cls, mesh, seed, kind, npz):
        if cls.cur is None or cls.cur.key != (mesh, seed, kind):
            cls.cur = None
            cls.cur = Ctx(mesh, seed, kind, npz)
        return cls.cur


def pieces(x):
    d = cKDTree(x).query(x, 2)[0][:, 1]
    r = 2.0 * float(np.median(d))
    pr = cKDTree(x).query_pairs(r, output_type="ndarray")
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(x), len(x)))
    nc, lab = connected_components(g, directed=False)
    cnt = np.bincount(lab)
    return dict(components=int(nc), outside_largest=int(len(x) - cnt.max()), radius_over_nn=2.0)


def read(ctx, x, label):
    row = {}
    # (1) volume
    in_t, in_h = ctx.true.inside(x), ctx.hull.inside(x)
    row["vol"] = dict(in_R=float((in_h & ~in_t).mean()), out_hull=float((~in_h).mean()), in_true=float(in_t.mean()),
                      compression=float(in_t.sum() / ctx.vol_true / (len(x) / ctx.vol_hull)) if label != "M" else None)
    # (2) the image-visible concave surface
    zb, zinfo = D7.field_mesh("zb", x, ctx.a)
    row["zb_mesh"] = zinfo
    du = zb.udist(ctx.P)
    s = np.where(zb.inside(ctx.P), -du, du) / ctx.a
    cp, pid, uv = zb.closest(ctx.P)
    nz = zb.normals_at(pid, uv)
    ang = np.degrees(np.arccos(np.clip(np.abs((nz * ctx.Pn).sum(1)), 0, 1)))
    row["surf"] = dict(S=float(s.sum()), n=len(s), mean=float(s.mean()),
                       bins=[dict(S=float(s[ctx.bin == k].sum()), n=int((ctx.bin == k).sum()),
                                  mean=float(s[ctx.bin == k].mean()) if (ctx.bin == k).any() else None,
                                  normal_deg=float(ang[ctx.bin == k].mean()) if (ctx.bin == k).any() else None)
                             for k in range(len(BINS) - 1)])
    # (3) silhouettes against M's exterior
    row["sil"] = {k: float(np.mean(iou_err(masks(zb, ctx.box, views, SIL_RES), ctx.Mmask[k])))
                  for k, views in (("train", TRAIN), ("held", HELD))}
    del zb
    # (4) the display surface in the held-out studio views
    zbs, _ = D7.field_mesh("zbs", x, ctx.a)
    S = [D7.render(zbs, ctx.center, ctx.radius, v) for v in HELD]
    row["img"] = {}
    for rn, T in (("true", ctx.img_T), ("M", ctx.img_M)):
        mt = D7.img_metrics(S, T)
        row["img"][rn] = {k: {q: float(np.mean(v)) for q, v in mt[k].items()} for k in ("shaded", "normal")}
        row["img"][rn]["normal_angle_deg"] = float(np.nanmean(mt["normal_angle_deg"]))
    del zbs, S
    # (5) the run's render terms
    p, n = ctx.discs(x)
    sil, pbr = d_exterior(p, n, *ctx.pic, TRAIN, RES_RUN, ctx.box, cfg.sil_k, cfg.w_hole, cfg.w_spray, cfg.pbr_ambient)
    row["pic"] = dict(sil=float(sil), pbr=float(pbr), discs=len(p))
    if label in ("L", "P", "V", "W"):
        row["pieces"] = pieces(x)
    return row


def main():
    args = sys.argv[1:]
    out_path, jobs_path = args[0], args[1]
    jobs = [ln.split() for ln in open(jobs_path) if ln.strip() and not ln.startswith("#")]
    if len(args) >= 4:
        jobs = jobs[int(args[2]):int(args[3])]
    with open(out_path, "a") as out:
        for mesh, seed, kind, label, npz, raw in jobs:
            t0 = time.time()
            seed = int(seed)
            ctx = Ctx.get(mesh, seed, kind, npz)
            if label in ("H", "M", "Mp", "S"):
                x = {"H": ctx.H, "M": ctx.M, "Mp": ctx.Mp, "S": ctx.S}[label]
            else:
                x = np.asarray(np.load(npz, allow_pickle=True)["frames"][-1], np.float32)
            row = dict(mesh=mesh, seed=seed, frame=kind, label=label, npz=npz.split("/")[-1], n=len(x), ctx=ctx.info,
                       **read(ctx, x, label))
            row["seconds"] = time.time() - t0
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps(dict(mesh=mesh, seed=seed, frame=kind, label=label, seconds=row["seconds"],
                                  in_R=row["vol"]["in_R"], S=row["surf"]["S"], held=row["sil"]["held"])), flush=True)


if __name__ == "__main__":
    main()
