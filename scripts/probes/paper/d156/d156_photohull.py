"""d156_photohull.py OUT_DIR N RES MESH [MESH ...] -- D156b: the photo hull of a target mesh from the render's own pictures.

Kutulakos and Seitz's space carving (IJCV 2000) with a model-based consistency check under the render's own shading
(render_loss.shaded_discs: shade = a + (1 - a) max(n . l, 0), a = cfg.pbr_ambient, l toward the camera), started from
D156's visual hull; the consistency test's tolerance is the noise of the pictures themselves, estimated from their eight
draws (Seitz and Dyer 1999's estimate of sensor noise from repeated measurements). CPU only (repo_d156 on the path, this
folder for d157_lib and hull_carve's helpers). For each mesh:
- the frame and the photographs: the mesh's own 40k run frame on the frozen path (seed 97: the source sample, the mesh
  at seed 98 matched to its volume), the box 1.25 x the sample's largest |x|, the 18 training views; the eight samples
  of the mesh the render's pictures are drawn from (seed 98 and draws_in_frame at 98 + 100 k, k = 1..7, prepare's),
  each drawn by the exterior operators (Zhu and Bridson's zero set at the sample's pitch, the lattice, d discs,
  exterior_targets at RES px): per draw d and view v the alpha A and the shaded picture S, the shade c = S / A, read
  bilinearly on the pictures' own lattice (losses/silhouette.py's nodes); a view is usable for a point if every draw
  has alpha > 0 there (the shade is defined in all of them);
- the start: the visual hull's voxels (hull_carve.py's field on its grid at the pictures' pitch, field > 0);
- visibility: a surface voxel (occupied, with an empty face neighbour) is seen in view v if the ray from it toward the
  camera, marched in half-voxel steps from sqrt(3)/2 voxel, meets no occupied voxel; recomputed every iteration;
- the exterior offset c: the pictures show the exterior, which lies c outside a solid; c = the mean signed distance of
  the seed-98 sample's exterior (its zero-set mesh, d157_lib.field_mesh) from the mesh (fan-closed); a voxel X is read
  at X + c n_hat, n_hat its fitted normal (fit at X, shift, fit again);
- the statistic: T(c) = min over unit n of sum over the usable visible views of (pred_v(n) - c_v)^2 / s_v^2, c the
  eight draws' mean shade (what the render loss compares with), s_v^2 the variance over the draws / 8 (the mean's
  noise); the unit-norm least squares solved exactly (eigen decomposition, the secular equation by bisection);
- the test: the eight draws give eight noise realisations at the mean's scale, e_d = (c^d - mean) / sqrt(7); a
  consistent point would show T_d = T(pred(n_hat) + e_d); X is inconsistent iff T(mean) > max_d T_d. A voxel seen in
  fewer than three usable views is kept (a unit normal has two degrees of freedom);
- A0, the gate of the carve: the first iteration's decisions on the visual hull's surface voxels, split by their signed
  distance to the mesh: on-mesh (|d| <= half a voxel) and phantom (d >= one voxel outside). The mesh is carved only if
  the phantom rejection rate exceeds the on-mesh one; otherwise it is reported and left uncarved;
- the carve: every iteration tests the surface voxels whose visible view set changed since their last test and removes
  all inconsistent ones (a batch Space Carving step); stop when an iteration removes none;
- the output: the hull field at the kept voxels capped at half a pixel, -1/2 pixel at the carved ones, marching cubes
  at zero, in the mesh's oriented coordinates: OUT_DIR/MESH_photo18_rRES.obj (orientation 'id', checked); one JSON line
  per mesh (A0's rates, iterations, voxels carved, the volumes, the over-carve: the share of the mesh's sample outside
  the photo hull against the visual hull's)."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import d157_lib as D7                                                                  # noqa: E402
import open3d as o3d                                                                   # noqa: E402
import torch                                                                           # noqa: E402
import torch.nn.functional as nnf                                                      # noqa: E402
import trimesh                                                                         # noqa: E402
from scipy.ndimage import distance_transform_edt, map_coordinates                      # noqa: E402
from skimage.measure import marching_cubes                                             # noqa: E402

from physmorph.losses.silhouette import set_kernel                                     # noqa: E402
from physmorph.pipeline.config import PipelineConfig                                   # noqa: E402
from physmorph.pipeline.render_loss import exterior_targets, make_views                # noqa: E402
from physmorph.render.exterior import Lattice, ZhuBridson                              # noqa: E402
from physmorph.sampling.mesh import _oriented, draws_in_frame, load_normalized          # noqa: E402
from physmorph.sampling.orientation import orient_name                                 # noqa: E402

torch.set_grad_enabled(False)
NT = D7.NT
cfg = PipelineConfig()
set_kernel("cic")
views = make_views(cfg.render_views, cfg.render_elevs)
AMB = cfg.pbr_ambient
SEED = 97
V = len(views)


def basis(theta, phi):
    right = np.array([np.cos(theta), 0.0, -np.sin(theta)])
    up = np.array([-np.sin(phi) * np.sin(theta), np.cos(phi), -np.sin(phi) * np.cos(theta)])
    return right, up, np.cross(right, up)


R_, U_, L_ = (np.array(v) for v in zip(*[basis(th, ph) for th, ph in views]))       # (V, 3) each


def silhouette(scene, box, res, theta, phi):
    px = 2.0 * box / res
    c = -box + np.arange(res) * px
    cu, cv = np.meshgrid(c, c, indexing="ij")
    r, u, d = basis(theta, phi)
    o = cu[..., None] * r + cv[..., None] * u + 3.0 * box * d
    rays = np.concatenate([o.reshape(-1, 3), np.broadcast_to(-d, (res * res, 3))], 1).astype(np.float32)
    t = scene.cast_rays(o3d.core.Tensor(rays), nthreads=NT)["t_hit"].numpy()
    return np.isfinite(t).reshape(res, res)


def inside_sd(mask):
    return np.where(mask, distance_transform_edt(mask) - 0.5, 0.5 - distance_transform_edt(~mask))


def lookup(img, P, box, px):
    """img (V, res, res) on the pictures' nodes, P (M, 3) world -> (M, V) bilinear values."""
    out = np.empty((len(P), V))
    for v in range(V):
        a, b = (P @ R_[v] + box) / px, (P @ U_[v] + box) / px
        out[:, v] = map_coordinates(img[v], [a, b], order=1, mode="constant", cval=0.0)
    return out


def unit_lsq(Aw, bw):
    """argmin over unit n of n^T A n - 2 b^T n for a batch (M,3,3), (M,3): the secular equation by bisection."""
    lam, Q = np.linalg.eigh(Aw)
    beta = np.einsum("mji,mj->mi", Q, bw)
    lo = lam[:, 0] - np.linalg.norm(bw, axis=1) - 1e-12
    hi = lam[:, 0].copy()
    for _ in range(80):
        mu = 0.5 * (lo + hi)
        nrm = ((beta / np.where(np.abs(lam - mu[:, None]) < 1e-300, 1e-300, lam - mu[:, None])) ** 2).sum(1)
        big = nrm > 1.0
        hi = np.where(big, mu, hi)
        lo = np.where(big, lo, mu)
    mu = lo
    n = np.einsum("mij,mj->mi", Q, beta / (lam - mu[:, None]))
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def stat(cobs, use, s2):
    """(T, n_hat) for observed shades cobs (M, V), usable-visible mask use (M, V), noise variances s2 (M, V)."""
    w = use * (1 - AMB) ** 2 / s2
    y = (cobs - AMB) / (1 - AMB)
    Aw = np.einsum("mv,vi,vj->mij", w, L_, L_)
    bw = np.einsum("mv,mv,vi->mi", w, y, L_)
    n = unit_lsq(Aw, bw)
    pred = AMB + (1 - AMB) * np.clip(n @ L_.T, 0, None)
    T = (use * (pred - cobs) ** 2 / s2).sum(1)
    return T, n, pred


def visibility(occ, idx, g0, px):
    """(M, V) bool: surface voxels idx (M, 3) seen in each view."""
    M = len(idx)
    P = g0 + idx * px
    vis = np.ones((M, V), bool)
    shape = np.array(occ.shape)
    kmax = int(np.ceil(np.linalg.norm(shape) * 2)) + 2
    for v in range(V):
        alive = np.ones(M, bool)
        for k in range(kmax):
            if not alive.any():
                break
            s = (np.sqrt(3) / 2 + 0.5 * k) * px
            q = np.rint((P[alive] + s * L_[v] - g0) / px).astype(np.int64)
            inb = np.all((q >= 0) & (q < shape), 1)
            hit = np.zeros(len(q), bool)
            hit[inb] = occ[q[inb, 0], q[inb, 1], q[inb, 2]]
            ia = np.flatnonzero(alive)
            vis[ia[hit], v] = False
            alive[ia[hit | ~inb]] = False
    return vis


def surface(occ):
    pad = np.pad(occ, 1)
    emp = np.zeros_like(occ)
    for ax in range(3):
        for sh in (-1, 1):
            emp |= ~np.roll(pad, sh, ax)[1:-1, 1:-1, 1:-1]
    return np.argwhere(occ & emp)


def main():
    out_dir, N, RES, meshes = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
    out_dir.mkdir(parents=True, exist_ok=True)
    for mesh in meshes:
        carve_one(out_dir, N, RES, mesh)


def carve_one(out_dir, N, RES, mesh):
    t0 = time.time()
    path = f"assets/{mesh}.obj"
    _, v_src = load_normalized("assets/isosphere.obj", N, SEED, return_volume=True, sample="stratified")
    frame = {}
    tgt = load_normalized(path, N, SEED + 1, match_volume=v_src, sample="stratified", frame=frame)
    box = float(np.abs(tgt).max()) * 1.25
    px = 2.0 * box / RES
    m = _oriented(path)
    Vw = (np.asarray(m.vertices, np.float64) - frame["offset"]) * frame["scale"]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(Vw.astype(np.float32)), o3d.core.Tensor(np.asarray(m.faces, np.uint32)))
    tc, _ = D7.close_holes(Vw, np.asarray(m.faces, np.int64))
    if tc.volume < 0:
        tc.invert()
    true = D7.Surf(np.asarray(tc.vertices), np.asarray(tc.faces))
    # the photographs: the eight samples' exterior pictures (the run's own operators and constants)
    samples = [tgt] + draws_in_frame(path, frame, N, [SEED + 1 + 100 * k for k in range(1, 8)])
    pitch = 0.708 * D7.median_kth_spacing(tgt, 8, 20000)
    T0 = torch.as_tensor(tgt)
    center = T0.mean(0)
    lattice = Lattice(center, 2.8 * float((T0 - center).norm(dim=1).max()))
    h = min(box / RES, 0.92 * pitch)
    Al, Sl = [], []
    for s in samples:
        p, g, _, _ = lattice.discs(ZhuBridson(torch.as_tensor(s), pitch), h, refine=False)
        sl, sh = exterior_targets(p, nnf.normalize(g, dim=1), views, RES, box, cfg.sil_k, AMB)
        Al.append(np.stack([x.numpy() for x in sl]).astype(np.float64))
        Sl.append(np.stack([x.numpy() for x in sh]).astype(np.float64))
    Al, Sl = np.stack(Al), np.stack(Sl)                                    # (8, V, res, res)
    # the exterior offset c of a sample of the mesh (seed 98), world units
    zb, _ = D7.field_mesh("zb", tgt, pitch)
    c_off = float(true.signed(zb.V).mean())
    # the visual hull's field on its grid (hull_carve.py's), occupancy at the nodes
    sds = [inside_sd(silhouette(scene, box, RES, th, ph)) for th, ph in views]
    g = -box - px + px * np.arange(RES + 3)
    G = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    f = np.full(len(G), np.inf)
    for (th, ph), sd in zip(views, sds):
        r, u, _ = basis(th, ph)
        f = np.minimum(f, map_coordinates(sd, [(G @ r + box) / px, (G @ u + box) / px], order=1, mode="constant",
                                          cval=-1e9))
    f = f.reshape(len(g), len(g), len(g))
    f[[0, -1], :, :] = f[:, [0, -1], :] = f[:, :, [0, -1]] = -1.0
    occ0 = f > 0
    occ = occ0.copy()
    g0 = np.array([g[0]] * 3)
    last = {}                                                              # voxel -> the view mask it was last tested with
    it, carved, a0 = 0, 0, None

    def observe(P):
        """(use (M,V), mean shade (M,V), draws (8,M,V), s2 (M,V)) at points P."""
        Am = np.stack([lookup(Al[d], P, box, px) for d in range(8)])
        Sm = np.stack([lookup(Sl[d], P, box, px) for d in range(8)])
        ok = (Am > 0).all(0)
        cd = Sm / np.where(Am > 0, Am, 1.0)
        mean = cd.mean(0)
        s2 = cd.var(0, ddof=1) / 8.0
        return ok, mean, cd, np.maximum(s2, 1e-12)

    while True:
        idx = surface(occ)
        if len(idx) == 0:                                                  # everything carved
            break
        vis = visibility(occ, idx, g0, px)
        keys = [tuple(k) for k in idx]
        masks = np.packbits(vis, axis=1, bitorder="little").view(np.uint8)
        mk = [bytes(r) for r in masks]
        need = np.array([last.get(k) != b for k, b in zip(keys, mk)])
        X = g0 + idx[need] * px
        vis_n = vis[need]
        ok, mean, cd, s2 = observe(X)
        use = vis_n & ok
        Tm, n1, _ = stat(mean, use, s2)
        Xe = X + c_off * n1                                                 # read at the exterior's place
        ok, mean, cd, s2 = observe(Xe)
        use = vis_n & ok
        Tm, nh, pred = stat(mean, use, s2)
        Tn = np.stack([stat(pred + (cd[d] - mean) / np.sqrt(7.0), use, s2)[0] for d in range(8)])
        testable = use.sum(1) >= 3
        bad = testable & (Tm > Tn.max(0))
        if it == 0:
            d = true.signed(X)
            on, ph = np.abs(d) <= 0.5 * px, d >= px
            nu = np.maximum(use.sum(1) - 2, 1)
            q = lambda z, mm: [float(v) for v in np.quantile(z[mm & testable], [.25, .5, .75])] if (mm & testable).any() else None  # noqa: E731
            a0 = dict(n_surface=int(len(X)), testable=float(testable.mean()),
                      on_mesh=dict(n=int(on.sum()), rejected=float(bad[on].mean()) if on.any() else None,
                                   T_per_dof_q=q(Tm / nu, on), Tnull_max_per_dof_q=q(Tn.max(0) / nu, on),
                                   views_q=q(use.sum(1).astype(float), on)),
                      phantom=dict(n=int(ph.sum()), rejected=float(bad[ph].mean()) if ph.any() else None,
                                   T_per_dof_q=q(Tm / nu, ph), Tnull_max_per_dof_q=q(Tn.max(0) / nu, ph),
                                   views_q=q(use.sum(1).astype(float), ph)))
            sep = (a0["phantom"]["rejected"] or 0.0) > (a0["on_mesh"]["rejected"] or 0.0)
            a0["separates"] = bool(sep)
            print(json.dumps(dict(mesh=mesh, A0=a0)), flush=True)
            if not sep:
                break
        for j in np.flatnonzero(need)[~bad]:                              # tested and kept: remember its view set
            last[keys[j]] = mk[j]
        rm = idx[need][bad]
        occ[rm[:, 0], rm[:, 1], rm[:, 2]] = False
        carved += len(rm)
        it += 1
        print(json.dumps(dict(mesh=mesh, iteration=it, tested=int(need.sum()), removed=int(len(rm)), carved=carved,
                              seconds=round(time.time() - t0, 1))), flush=True)
        if len(rm) == 0:
            break
    # the photo hull's mesh
    F = np.where(occ, np.minimum(f, 0.5), np.where(occ0, -0.5, f))
    verts, faces, _, _ = marching_cubes(F, level=0.0, spacing=(px, px, px), allow_degenerate=False)
    pw = trimesh.Trimesh(verts + g[0], faces, process=True)
    pw.update_faces(pw.nondegenerate_faces())
    pw.merge_vertices()
    if not pw.is_watertight:
        trimesh.repair.fill_holes(pw)
    if pw.volume < 0:
        pw.invert()
    pm = trimesh.Trimesh(pw.vertices / frame["scale"] + frame["offset"], pw.faces, process=False)
    out = out_dir / f"{mesh}_photo18_r{RES}.obj"
    pm.export(str(out))
    assert orient_name(str(out)) == "id", out
    hull = trimesh.load(str(out_dir / f"{mesh}_hull18_r{RES}.obj"), process=False, force="mesh")
    hw = (np.asarray(hull.vertices) - frame["offset"]) * frame["scale"]

    def outside(Vs, Fs, pts):
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.core.Tensor(np.asarray(Vs, np.float32)), o3d.core.Tensor(np.asarray(Fs, np.uint32)))
        return float((sc.compute_occupancy(o3d.core.Tensor(tgt), nthreads=NT, nsamples=3).numpy() < 0.5).mean())
    row = dict(mesh=mesh, n=N, res=RES, px=px, pitch=pitch, c_offset_pitch=c_off / pitch, A0=a0, iterations=it,
               voxels_hull=int(occ0.sum()), voxels_carved=int(carved), watertight=bool(pw.is_watertight),
               vol_mesh=float(tc.volume), vol_hull=float(trimesh.Trimesh(hw, hull.faces).volume),
               vol_photo=float(pw.volume), mesh_sample_outside_hull=outside(hw, hull.faces, tgt),
               mesh_sample_outside_photo=outside(pw.vertices, pw.faces, tgt), out=str(out),
               seconds=time.time() - t0)
    print(json.dumps(row), flush=True)
    with open(out_dir / f"photo_r{RES}.jsonl", "a") as fo:
        fo.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
