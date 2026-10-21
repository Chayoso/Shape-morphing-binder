"""d156_rcarve_gpu.py OUT_DIR N RES MESH [MESH ...] -- D156c's render-loss carve (d156_rcarve.py) on a card.

The same carve, step for step: the frame, the eight-draw pictures and the operator of the run's render term; the
visual hull's carried sample, each particle in its nearest occupied node; the surface voxels seen in at least three
views ranked by D156b's statistic T per degree of freedom; a removal of the top k accepted iff the render loss falls.
Ported to torch on the card (the loss through the pipeline's own exterior operators on the device; visibility,
observation and the statistic batched), because on the CPU one loss evaluation takes about 8 s and a mesh needs
hundreds of accepted steps. The search over k: the first step tries all candidates, a later one starts at twice the
last accepted k; k is halved after each refusal; the carve stops when no k >= 1 lowers the loss. Run from repo_d156
with a card (CUDA_VISIBLE_DEVICES set by the caller)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import physmorph  # noqa: E402,F401  (before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as nnf  # noqa: E402
import d156_photohull as PH  # noqa: E402
from d156_photohull import D7, o3d, trimesh, marching_cubes, map_coordinates  # noqa: E402
from physmorph.pipeline.render_loss import d_exterior, exterior_targets  # noqa: E402
from physmorph.render.exterior import Lattice, make_field  # noqa: E402
from physmorph.sampling.mesh import (_oriented, draws_in_frame, fill_volume_in_frame, load_normalized,  # noqa: E402
                                     sample_in_frame)
from physmorph.sampling.orientation import orient_name  # noqa: E402

torch.set_grad_enabled(False)
import os  # noqa: E402
dev = torch.device(os.environ.get("D156_DEV", "cuda"))          # "cpu" only for the port check (d156_port_check.py)
cfg, AMB, views, SEED, NT = PH.cfg, PH.AMB, PH.views, PH.SEED, PH.NT
V = len(views)
R_t, U_t, L_t = (torch.as_tensor(a, dtype=torch.float64, device=dev) for a in (PH.R_, PH.U_, PH.L_))
FACE = torch.zeros(1, 1, 3, 3, 3, device=dev)
for (a, b, c) in ((0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)):
    FACE[0, 0, a, b, c] = 1.0


def surface_t(occ):
    emp = nnf.pad((~occ).float()[None, None], (1,) * 6, value=1.0)
    nb = nnf.conv3d(emp, FACE)[0, 0]
    return torch.nonzero(occ & (nb > 0))


def visibility_t(occ, idx, g0, px):
    shape = torch.as_tensor(occ.shape, device=dev)
    P = (g0 + idx.double() * px)[:, None, :]                            # (M,1,3)
    M = len(idx)
    vis = torch.ones(M, V, dtype=torch.bool, device=dev)
    alive = torch.ones(M, V, dtype=torch.bool, device=dev)
    flat = occ.reshape(-1)
    kmax = int(np.ceil(float(shape.double().norm()) * 2)) + 2
    for k in range(kmax):
        s = (np.sqrt(3) / 2 + 0.5 * k) * px
        q = torch.round((P + s * L_t[None] - g0) / px).long()            # (M,V,3)
        inb = ((q >= 0) & (q < shape)).all(-1)
        qc = torch.minimum(torch.clamp(q, min=0), shape - 1)
        hit = flat[(qc[..., 0] * shape[1] + qc[..., 1]) * shape[2] + qc[..., 2]] & inb
        vis &= ~(hit & alive)
        alive &= ~hit & inb
        if k % 16 == 0 and not bool(alive.any()):
            break
    return vis


def lookup_t(img, P, box, px):
    """img (D,V,R,R) on the pictures' nodes, P (M,3) -> (D,M,V) bilinear, 0 outside."""
    D, _, Rr, _ = img.shape
    a = ((P @ R_t.T) + box) / px                                         # (M,V)
    b = ((P @ U_t.T) + box) / px
    i0, j0 = torch.floor(a).long(), torch.floor(b).long()
    fa, fb = a - i0, b - j0
    vv = torch.arange(V, device=dev)[None, :].expand_as(i0)
    out = torch.zeros(D, *a.shape, dtype=torch.float64, device=dev)
    flat = img.reshape(D, -1).double()
    for di, wi in ((0, 1 - fa), (1, fa)):
        for dj, wj in ((0, 1 - fb), (1, fb)):
            ii, jj = i0 + di, j0 + dj
            ok = (ii >= 0) & (ii < Rr) & (jj >= 0) & (jj < Rr)
            ix = ((vv * Rr + ii.clamp(0, Rr - 1)) * Rr + jj.clamp(0, Rr - 1)).reshape(-1)
            out += (flat[:, ix].reshape(D, *a.shape) * (wi * wj * ok)[None])
    return out


def unit_lsq_t(Aw, bw):
    lam, Q = torch.linalg.eigh(Aw)
    beta = torch.einsum("mji,mj->mi", Q, bw)
    lo = lam[:, 0] - bw.norm(dim=1) - 1e-12
    hi = lam[:, 0].clone()
    for _ in range(80):
        mu = 0.5 * (lo + hi)
        den = lam - mu[:, None]
        nrm = (beta / torch.where(den.abs() < 1e-300, torch.full_like(den, 1e-300), den)).square().sum(1)
        big = nrm > 1.0
        hi = torch.where(big, mu, hi)
        lo = torch.where(big, lo, mu)
    n = torch.einsum("mij,mj->mi", Q, beta / (lam - lo[:, None]))
    return n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)


def stat_t(cobs, use, s2):
    w = use * (1 - AMB) ** 2 / s2
    y = (cobs - AMB) / (1 - AMB)
    Aw = torch.einsum("mv,vi,vj->mij", w, L_t, L_t)
    bw = torch.einsum("mv,mv,vi->mi", w, y, L_t)
    n = unit_lsq_t(Aw, bw)
    pred = AMB + (1 - AMB) * (n @ L_t.T).clamp_min(0)
    return (use * (pred - cobs).square() / s2).sum(1), n, pred


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
    samples = [tgt] + draws_in_frame(path, frame, N, [SEED + 1 + 100 * k for k in range(1, 8)])
    pitch = 0.708 * D7.median_kth_spacing(tgt, 8, 20000)
    T0 = torch.as_tensor(tgt, device=dev)
    center = T0.mean(0)
    lattice = Lattice(center, 2.8 * float((T0 - center).norm(dim=1).max()))
    h = min(box / RES, 0.92 * pitch)

    def discs(x):
        x = torch.as_tensor(np.asarray(x, np.float32) if isinstance(x, np.ndarray) else x, device=dev)
        p, g, _, _ = lattice.discs(make_field("zb", x, pitch, device_field=dev.type == "cuda"), h, refine=False)
        return p, nnf.normalize(g, dim=1)

    per = [exterior_targets(*discs(s), views, RES, box, cfg.sil_k, AMB) for s in samples]
    sils = [torch.stack(v).mean(0) for v in zip(*[e[0] for e in per])]
    shade = [torch.stack(v).mean(0) for v in zip(*[e[1] for e in per])]
    Al = torch.stack([torch.stack(list(e[0])) for e in per]).double()       # (8,V,R,R)
    Sl = torch.stack([torch.stack(list(e[1])) for e in per]).double()

    def loss(x):
        p, n = discs(x)
        s, b = d_exterior(p, n, sils, shade, views, RES, box, cfg.sil_k, cfg.w_hole, cfg.w_spray, AMB)
        return float(s + cfg.w_pbr * b)

    zb, _ = D7.field_mesh("zb", tgt, pitch)
    c_off = float(true.signed(zb.V).mean())
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
    occ = torch.as_tensor(occ0, device=dev)
    g0 = torch.full((3,), float(g[0]), dtype=torch.float64, device=dev)
    shape = np.array(occ0.shape)
    n_vh = int(round(N * fill_volume_in_frame(hpath, frame, N) / fill_volume_in_frame(path, frame, N)))
    xs_np, _ = sample_in_frame(hpath, frame, n_vh, SEED + 1)
    node = np.clip(np.rint((xs_np - g[0]) / px).astype(np.int64), 0, shape - 1)
    from scipy.ndimage import distance_transform_edt
    _, near = distance_transform_edt(~occ0, return_indices=True)            # shell particles: the nearest occupied node
    node = np.stack([near[a][node[:, 0], node[:, 1], node[:, 2]] for a in range(3)], 1)
    node_flat = torch.as_tensor((node[:, 0] * shape[1] + node[:, 1]) * shape[2] + node[:, 2], device=dev)
    xs = torch.as_tensor(xs_np, device=dev)
    alive = torch.ones(len(xs), dtype=torch.bool, device=dev)
    floor_x, _ = sample_in_frame(path, frame, N, SEED + 2)
    L0 = L = loss(xs)
    L_floor = loss(floor_x)

    def observe(P):
        Am, Sm = lookup_t(Al, P, box, px), lookup_t(Sl, P, box, px)
        ok = (Am > 0).all(0)
        cd = Sm / torch.where(Am > 0, Am, torch.ones_like(Am))
        return ok, cd.mean(0), (cd.var(0) / 8.0).clamp_min(1e-12)

    it = tries = carved = 0
    k_last = None
    log = []
    while True:
        idx = surface_t(occ)
        if len(idx) == 0:
            break
        vis = visibility_t(occ, idx, g0, px)
        X = g0 + idx.double() * px
        ok, mean, s2 = observe(X)
        use = (vis & ok).double()
        _, n1, _ = stat_t(mean, use, s2)
        ok, mean, s2 = observe(X + c_off * n1)
        use = (vis & ok).double()
        T, _, _ = stat_t(mean, use, s2)
        cnt = use.sum(1)
        cand = torch.nonzero(cnt >= 3).squeeze(1)
        cand = cand[torch.argsort(-(T[cand] / (cnt[cand] - 2).clamp_min(1)))]
        k = len(cand) if k_last is None else min(len(cand), 2 * k_last)
        acc = None
        while k >= 1:
            rm = idx[cand[:k]]
            kill = torch.zeros(occ.numel(), dtype=torch.bool, device=dev)
            kill[(rm[:, 0] * shape[1] + rm[:, 1]) * shape[2] + rm[:, 2]] = True
            gone = alive & kill[node_flat]
            tries += 1
            if bool(gone.any()):
                L_try = loss(xs[alive & ~gone])
                if L_try < L:
                    acc = (rm, gone, L_try, k)
                    break
            k //= 2
        if acc is None:
            break
        rm, gone, L, k_last = acc
        occ[rm[:, 0], rm[:, 1], rm[:, 2]] = False
        alive &= ~gone
        carved += len(rm)
        it += 1
        log.append(dict(iteration=it, k=int(len(rm)), loss=L, particles_left=int(alive.sum())))
        if it % 10 == 1:
            print(json.dumps(dict(mesh=mesh, iteration=it, removed=int(len(rm)), carved=carved, loss=L, loss0=L0,
                                  tries=tries, seconds=round(time.time() - t0, 1))), flush=True)
    occ_np = occ.cpu().numpy()
    F = np.where(occ_np, f, np.where(occ0, -0.5, f))
    np.savez_compressed(out_dir / f"{mesh}_rcarve18_r{RES}_occ.npz", occ0=np.packbits(occ0), occ=np.packbits(occ_np),
                        shape=np.array(occ0.shape), g0=g[0], px=px)
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
