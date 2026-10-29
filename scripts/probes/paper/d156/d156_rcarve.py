"""d156_rcarve.py OUT_DIR N RES MESH [MESH ...] -- D156c: the visual hull carved by the render loss itself.

Broadhurst, Drummond and Cipolla's comparison (ICCV 2001) of the data with a voxel and without it, with the render loss
as the data term: the pipeline's own exterior operator against the eight-draw pictures, so no point model stands
between the pictures and the decision (D156b's per-point test failed A0 on that model's mismatch). CPU only
(repo_d156 on the path; d156_photohull.py and d157_lib.py from this folder). For each mesh:
- frame, photographs and operator: the true mesh's 40k run frame at seed 97; its eight samples (seed 98 and the draws
  98 + 100 k) drawn by the exterior operators (pitch 0.708 x the seed-98 sample's median 8th-neighbour distance, the
  lattice centred on it, h = min(box / RES, 0.92 pitch)); the pictures their mean (target.build_target's); the loss
  sil + w_pbr pbr of d_exterior (w_hole 2, w_spray 1, ambient 0.25): the run's render term;
- the carried sample: the visual hull's sample in that frame at the true sample's number density (sample_in_frame,
  seed 98, n = N x fill(hull) / fill(mesh)); each particle in the node of D156's hull grid (the pictures' pitch) it
  rounds to; carving a node deletes its particles, the others stay: the loss changes only through the carve;
- the order: the surface voxels (occupied nodes with an empty face neighbour, among the hull's) seen in at least three
  views, ranked by D156b's statistic T per degree of freedom (descending);
- the step: try removing the top k of the ranking, k = all of them first, halved after each refusal; the first
  removal that lowers the loss is accepted; stop when none down to k = 1 does;
- the output: the kept nodes meshed (the hull field at kept nodes, -1/2 pixel at carved ones, marching
  cubes at zero) in the mesh's oriented coordinates, OUT_DIR/MESH_rcarve18_rRES.obj; one JSON line per mesh: the loss
  at the start, the end and of an independent sample of the mesh at the same density (seed 99, the floor), the
  iterations and the tries, voxels and particles carved, the volumes, the over-carve (the seed-98 sample's share
  outside the carved solid against the visual hull's)."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import d156_photohull as PH                                                           # noqa: E402
from d156_photohull import D7, o3d, torch, nnf, trimesh, marching_cubes, map_coordinates  # noqa: E402
from physmorph.pipeline.render_loss import d_exterior, exterior_targets                # noqa: E402
from physmorph.render.exterior import Lattice, ZhuBridson                              # noqa: E402
from physmorph.sampling.mesh import (_oriented, draws_in_frame, fill_volume_in_frame, load_normalized,  # noqa: E402
                                     sample_in_frame)
from physmorph.sampling.orientation import orient_name                                 # noqa: E402

cfg, AMB, views, SEED, NT = PH.cfg, PH.AMB, PH.views, PH.SEED, PH.NT


def carve_one(out_dir, N, RES, mesh):
    t0 = time.time()
    path, hpath = f"assets/{mesh}.obj", str(out_dir / f"{mesh}_hull18_r{RES}.obj")
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
    # the operator and the pictures (the run's render term)
    samples = [tgt] + draws_in_frame(path, frame, N, [SEED + 1 + 100 * k for k in range(1, 8)])
    pitch = 0.708 * D7.median_kth_spacing(tgt, 8, 20000)
    T0 = torch.as_tensor(tgt)
    center = T0.mean(0)
    lattice = Lattice(center, 2.8 * float((T0 - center).norm(dim=1).max()))
    h = min(box / RES, 0.92 * pitch)

    def discs(x):
        p, g, _, _ = lattice.discs(ZhuBridson(torch.as_tensor(np.asarray(x, np.float32)), pitch), h, refine=False)
        return p, nnf.normalize(g, dim=1)

    per = [exterior_targets(*discs(s), views, RES, box, cfg.sil_k, AMB) for s in samples]
    sils = [torch.stack(v).mean(0) for v in zip(*[e[0] for e in per])]
    shade = [torch.stack(v).mean(0) for v in zip(*[e[1] for e in per])]
    Al = np.stack([np.stack([x.numpy() for x in e[0]]) for e in per]).astype(np.float64)
    Sl = np.stack([np.stack([x.numpy() for x in e[1]]) for e in per]).astype(np.float64)

    def loss(x):
        p, n = discs(x)
        s, b = d_exterior(p, n, sils, shade, views, RES, box, cfg.sil_k, cfg.w_hole, cfg.w_spray, AMB)
        return float(s + cfg.w_pbr * b)

    zb, _ = D7.field_mesh("zb", tgt, pitch)
    c_off = float(true.signed(zb.V).mean())
    # the visual hull's grid and field (hull_carve.py's)
    sds = [PH.inside_sd(PH.silhouette(scene, box, RES, th, ph)) for th, ph in views]
    g = -box - px + px * np.arange(RES + 3)
    G = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    f = np.full(len(G), np.inf)
    for (th, ph), sd in zip(views, sds):
        r, u, _ = PH.basis(th, ph)
        f = np.minimum(f, map_coordinates(sd, [(G @ r + box) / px, (G @ u + box) / px], order=1, mode="constant",
                                          cval=-1e9))
    f = f.reshape(len(g), len(g), len(g))
    f[[0, -1], :, :] = f[:, [0, -1], :] = f[:, :, [0, -1]] = -1.0
    occ0 = f > 0
    occ = occ0.copy()
    g0 = np.array([g[0]] * 3)
    shape = np.array(occ.shape)
    # the carried sample and its nodes
    n_vh = int(round(N * fill_volume_in_frame(hpath, frame, N) / fill_volume_in_frame(path, frame, N)))
    xs, _ = sample_in_frame(hpath, frame, n_vh, SEED + 1)
    node = np.clip(np.rint((xs - g0) / px).astype(np.int64), 0, shape - 1)
    # a particle of the sampler's outer shell (up to a voxel outside the hull) rounds to an empty node: it belongs to the
    # nearest occupied node, so carving that node removes it too (else a detached skin of particles keeps the exterior)
    from scipy.ndimage import distance_transform_edt
    _, near = distance_transform_edt(~occ0, return_indices=True)
    node = np.stack([near[a][node[:, 0], node[:, 1], node[:, 2]] for a in range(3)], 1)
    alive = np.ones(len(xs), bool)
    floor_x, _ = sample_in_frame(path, frame, N, SEED + 2)
    L0 = L = loss(xs)
    L_floor = loss(floor_x)

    def observe(P):
        Am = np.stack([PH.lookup(Al[d], P, box, px) for d in range(8)])
        Sm = np.stack([PH.lookup(Sl[d], P, box, px) for d in range(8)])
        ok = (Am > 0).all(0)
        cd = Sm / np.where(Am > 0, Am, 1.0)
        return ok, cd.mean(0), np.maximum(cd.var(0, ddof=1) / 8.0, 1e-12)

    it = tries = carved = 0
    log = []
    while True:
        idx = PH.surface(occ)
        if len(idx) == 0:
            break
        vis = PH.visibility(occ, idx, g0, px)
        X = g0 + idx * px
        ok, mean, s2 = observe(X)
        use = vis & ok
        _, n1, _ = PH.stat(mean, use, s2)
        ok, mean, s2 = observe(X + c_off * n1)
        use = vis & ok
        T, _, _ = PH.stat(mean, use, s2)
        cnt = use.sum(1)
        cand = np.flatnonzero(cnt >= 3)
        cand = cand[np.argsort(-(T[cand] / np.maximum(cnt[cand] - 2, 1)))]
        k, acc = len(cand), None
        while k >= 1:
            rm = idx[cand[:k]]
            kill = np.zeros(shape, bool)
            kill[rm[:, 0], rm[:, 1], rm[:, 2]] = True
            gone = alive & kill[node[:, 0], node[:, 1], node[:, 2]]
            tries += 1
            if gone.any():
                L_try = loss(xs[alive & ~gone])
                if L_try < L:
                    acc = (rm, gone, L_try)
                    break
            k //= 2
        if acc is None:
            break
        rm, gone, L = acc
        occ[rm[:, 0], rm[:, 1], rm[:, 2]] = False
        alive &= ~gone
        carved += len(rm)
        it += 1
        log.append(dict(iteration=it, k=len(rm), loss=L, particles_left=int(alive.sum())))
        print(json.dumps(dict(mesh=mesh, iteration=it, removed=len(rm), carved=carved, loss=L, loss0=L0,
                              tries=tries, seconds=round(time.time() - t0, 1))), flush=True)
    # kept nodes keep the hull's field (the uncarved surface is the visual hull's), carved nodes read -1/2 pixel
    F = np.where(occ, f, np.where(occ0, -0.5, f))
    np.savez_compressed(out_dir / f"{mesh}_rcarve18_r{RES}_occ.npz", occ0=np.packbits(occ0), occ=np.packbits(occ),
                        shape=np.array(occ.shape), g0=g[0], px=px)
    verts, faces, _, _ = marching_cubes(F, level=0.0, spacing=(px, px, px), allow_degenerate=False)
    pw = trimesh.Trimesh(verts + g[0], faces, process=True)
    pw.update_faces(pw.nondegenerate_faces())
    pw.merge_vertices()
    if not pw.is_watertight:
        trimesh.repair.fill_holes(pw)
    if pw.volume < 0:
        pw.invert()
    pm = trimesh.Trimesh(pw.vertices / frame["scale"] + frame["offset"], pw.faces, process=False)
    out = out_dir / f"{mesh}_rcarve18_r{RES}.obj"
    pm.export(str(out))
    assert orient_name(str(out)) == "id", out
    hull = trimesh.load(hpath, process=False, force="mesh")
    hw = (np.asarray(hull.vertices) - frame["offset"]) * frame["scale"]

    def outside(Vs, Fs):
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.core.Tensor(np.asarray(Vs, np.float32)), o3d.core.Tensor(np.asarray(Fs, np.uint32)))
        return float((sc.compute_occupancy(o3d.core.Tensor(tgt), nthreads=NT, nsamples=3).numpy() < 0.5).mean())

    row = dict(mesh=mesh, n=N, res=RES, px=px, pitch=pitch, c_offset_pitch=c_off / pitch, loss_hull=L0, loss_end=L,
               loss_floor=L_floor, iterations=it, tries=tries, voxels_hull=int(occ0.sum()), voxels_carved=int(carved),
               particles=int(len(xs)), particles_carved=int((~alive).sum()), watertight=bool(pw.is_watertight),
               vol_mesh=float(tc.volume), vol_hull=float(trimesh.Trimesh(hw, hull.faces).volume),
               vol_rcarve=float(pw.volume), mesh_sample_outside_hull=outside(hw, hull.faces),
               mesh_sample_outside_rcarve=outside(pw.vertices, pw.faces), out=str(out), log=log,
               seconds=time.time() - t0)
    print(json.dumps({k: v for k, v in row.items() if k != "log"}), flush=True)
    with open(out_dir / f"rcarve_r{RES}.jsonl", "a") as fo:
        fo.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    od, n, res = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    for ms in sys.argv[4:]:
        carve_one(od, n, res, ms)
