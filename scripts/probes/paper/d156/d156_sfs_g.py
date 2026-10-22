"""d156_sfs_g.py OUT_DIR HULL_DIR N RES MESH -- D156g (docs entry): the carve whose state is a uniform solid. The visual
hull minus a carve depth d >= 0 on the hull mesh's vertices (read at a point through the barycentric weights of its
closest hull point; capped by the hull's thickness under the vertex). A state's uniform sample: the run's sampler's draw
of the hull (stratified, n_hull particles) restricted by the sampler's own voxel-centre rule (a particle is dropped iff
its voxel centre c lies in the carved layer: d(pi(c)) > 0 and depth(c) < d(pi(c))), so two states are compared on the
same draws. Each step: four fresh seeds; the gradient on draw 0 through the true operator (the Zhu-Bridson field, tracked
discs found at the sample, read differentiably), g_i = -(dL/dx_i . nu(pi(x_i))) scattered to the vertices and divided by
their areas, smoothed by (I + l^2 L) on the hull's vertex graph (4 / sum |dx|^2), coarse to fine l = box / 2^k down to
one pixel; the trial accepted iff the mean over the four paired draws falls (a trial that moves no voxel centre equals
the current state: refused without evaluation, the halving goes on). The score: the marching-cubes solid of depth -
d(pi) at half a pitch, resampled by the run's sampler, the mean over the eight held-out seeds 95 + 100 k. CPU."""
import json
import os
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
from d156_photohull import D7, o3d, trimesh, marching_cubes  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from physmorph.pipeline.render_loss import d_exterior, exterior_targets  # noqa: E402
from physmorph.render.exterior import Lattice, make_field, tracked  # noqa: E402
from physmorph.sampling.mesh import (_oriented, _stratified_fill, draws_in_frame, fill_volume_in_frame,  # noqa: E402
                                     load_normalized)
from physmorph.sampling.orientation import orient_name  # noqa: E402

torch.set_grad_enabled(True)
BUDGET = int(os.environ.get("D156_BUDGET", "300"))
WALL = float(os.environ.get("D156_WALL", str(6 * 3600)))
J = 4
RATIO = 2.0
cfg, AMB, views, SEED, NT = PH.cfg, PH.AMB, PH.views, PH.SEED, PH.NT
FINAL = [95 + 100 * k for k in range(8)]


def cg(Aop, b, tol=1e-6, iters=300):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rr = (r * r).sum()
    b2 = (b * b).sum().clamp_min(1e-300)
    for _ in range(iters):
        Ap = Aop(p)
        a = rr / (p * Ap).sum().clamp_min(1e-300)
        x = x + a * p
        r = r - a * Ap
        rn = (r * r).sum()
        if float(rn / b2) < tol * tol:
            break
        p = r + (rn / rr) * p
        rr = rn
    return x


def bary(uv):
    return np.concatenate([1 - uv[:, :1] - uv[:, 1:], uv], 1)


def solve(out_dir, hull_dir, N, RES, mesh):
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    path, hpath = f"assets/{mesh}.obj", str(hull_dir / f"{mesh}_hull18_r{RES}.obj")
    _, v_src = load_normalized("assets/isosphere.obj", N, SEED, return_volume=True, sample="stratified")
    frame = {}
    tgt = load_normalized(path, N, SEED + 1, match_volume=v_src, sample="stratified", frame=frame)
    box = float(np.abs(tgt).max()) * 1.25
    px = 2.0 * box / RES
    m = _oriented(path)
    Vw = (np.asarray(m.vertices, np.float64) - frame["offset"]) * frame["scale"]
    tc, _ = D7.close_holes(Vw, np.asarray(m.faces, np.int64))
    if tc.volume < 0:
        tc.invert()
    true = D7.Surf(np.asarray(tc.vertices), np.asarray(tc.faces))
    hm = _oriented(hpath)
    Hw = (np.asarray(hm.vertices, np.float64) - frame["offset"]) * frame["scale"]
    hullt = trimesh.Trimesh(Hw, np.asarray(hm.faces), process=True)
    if hullt.volume < 0:
        hullt.invert()
    hull = D7.Surf(np.asarray(hullt.vertices), np.asarray(hullt.faces))
    pitch = 0.708 * D7.median_kth_spacing(tgt, 8, 20000)
    T0 = torch.as_tensor(tgt)
    center = T0.mean(0)
    lattice = Lattice(center, 2.8 * float((T0 - center).norm(dim=1).max()))
    h = min(box / RES, 0.92 * pitch)

    def fresh(x):
        with torch.no_grad():
            p, g, _, _ = lattice.discs(make_field("zb", torch.as_tensor(np.asarray(x, np.float32)), pitch), h,
                                       refine=False)
        return p, nnf.normalize(g, dim=1)

    with torch.no_grad():
        samples = [tgt] + draws_in_frame(path, frame, N, [SEED + 1 + 100 * k for k in range(1, 8)])
        per = [exterior_targets(*fresh(s), views, RES, box, cfg.sil_k, AMB) for s in samples]
        sils = [torch.stack(v).mean(0) for v in zip(*[e[0] for e in per])]
        shade = [torch.stack(v).mean(0) for v in zip(*[e[1] for e in per])]

    def loss_pn(p, n):
        s, b = d_exterior(p, n, sils, shade, views, RES, box, cfg.sil_k, cfg.w_hole, cfg.w_spray, AMB)
        return s + cfg.w_pbr * b

    def loss(x):
        with torch.no_grad():
            return float(loss_pn(*fresh(x)))

    def mean_loss(xs):
        v = [loss(x) for x in xs]
        return float(np.mean(v)), [round(x, 7) for x in v]

    # the hull's draws by the run's sampler (stratified_draws' operations), with each particle's voxel centre
    fill_h = fill_volume_in_frame(hpath, frame, N)
    n_vh = int(round(N * fill_h / fill_volume_in_frame(path, frame, N)))
    hi, centers, vp = _stratified_fill(hm, n_vh)
    print(f"[d156g] {mesh}: hull fill {hi}^3 = {len(centers)} voxels for n = {n_vh}", flush=True)

    def hull_draws(seeds):
        out = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            keep = rng.choice(len(centers), n_vh, replace=False) if len(centers) > n_vh else np.arange(n_vh)
            jitter = (rng.uniform(-0.5, 0.5, (n_vh, 3)) * vp).astype(np.float32)
            x = (centers[keep] + jitter).astype(np.float32)
            out.append((((x.astype(np.float64) - frame["offset"]) * frame["scale"]).astype(np.float32), keep))
        return out

    chk = hull_draws([SEED + 1])[0][0]
    same = bool(np.array_equal(chk, draws_in_frame(hpath, frame, n_vh, [SEED + 1])[0]))
    print(f"[d156g] the restricted sampler's d = 0 draw is the run's sampler's draw bit for bit: {same}", flush=True)
    assert same
    Cf = (np.asarray(centers, np.float64) - frame["offset"]) * frame["scale"]
    depth_c = -hull.signed(Cf)
    _, fid_c, uv_c = hull.closest(Cf)
    HF = np.asarray(hullt.faces, np.int64)
    vid_c, W_c = HF[fid_c], bary(uv_c)
    # the extraction grid (half a pitch over the hull's box + 3 pitches): depth and pi fixed
    hh = 0.5 * pitch
    lo = Hw.min(0) - 3 * pitch
    nc = np.ceil((Hw.max(0) + 3 * pitch - lo) / hh).astype(int) + 1
    G = np.stack(np.meshgrid(*[lo[i] + hh * np.arange(nc[i]) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    depth_G = -hull.signed(G)
    in_G = np.flatnonzero(depth_G > 0)
    _, fid_G, uv_G = hull.closest(G[in_G])
    vid_G, W_G = HF[fid_G], bary(uv_G)
    del G
    # the vertices: normals, areas, thickness, the graph, the true excess
    V = np.asarray(hullt.vertices, np.float64)
    nV = len(V)
    vn = np.asarray(hullt.vertex_normals, np.float64)
    A_v = np.zeros(nV)
    np.add.at(A_v, HF.ravel(), np.repeat(hullt.area_faces / 3.0, 3))
    eps = 1e-4 * pitch
    ok, _, _, tv = hull.cast(np.concatenate([V - eps * vn, -vn], 1))
    thick_np = np.where(ok, tv + eps, 0.0)
    E = np.asarray(hullt.edges_unique, np.int64)
    I = torch.as_tensor(np.concatenate([E[:, 0], E[:, 1]]))
    Jn = torch.as_tensor(np.concatenate([E[:, 1], E[:, 0]]))
    Vt = torch.as_tensor(V)
    dxe = Vt[Jn] - Vt[I]
    s2 = torch.zeros(nV, dtype=torch.float64).index_add_(0, I, (dxe * dxe).sum(1))
    c = torch.where(s2 > 0, 4.0 / s2.clamp_min(1e-30), torch.zeros_like(s2))
    w = 0.5 * (c[I] + c[Jn])
    wsum = torch.zeros(nV, dtype=torch.float64).index_add_(0, I, w)
    ell = {"v": px}

    def H1(f):
        return f + ell["v"] ** 2 * (wsum * f - torch.zeros(nV, dtype=torch.float64).index_add_(0, I, w * f[Jn]))

    e_v = true.udist(V)

    def removed(d):
        dc = (W_c * d[vid_c]).sum(1)
        return (dc > 0) & (depth_c < dc)

    def restrict(draws, rem):
        return [x[~rem[k]] for x, k in draws]

    def grad_d(x):
        xt = torch.as_tensor(x)
        tr = tracked(make_field("zb", xt, pitch), lattice, h, pitch)
        xr = xt.clone().requires_grad_(True)
        p, n, _ = tr.read(xr)
        L = loss_pn(p, n)
        gx = torch.autograd.grad(L, xr)[0].numpy().astype(np.float64)
        _, fid, uv = hull.closest(x)
        gi = -(gx * hull.fn[fid]).sum(1)
        g = np.zeros(nV)
        np.add.at(g, HF[fid].ravel(), (bary(uv) * gi[:, None]).ravel())
        return float(L), torch.as_tensor(g / np.maximum(A_v, 1e-30))

    d = np.zeros(nV)
    levels = []
    lk = box
    while lk > px:
        levels.append(lk)
        lk = lk / RATIO
    levels.append(px)
    steps, evals, t_seed, skipped, refused, log = 0, 0, 0, 0, 0, []
    stop = "levels"
    for li, lv in enumerate(levels):
        ell["v"] = lv
        step_px = 1.0
        first = True
        while True:
            if steps >= BUDGET:
                stop = "budget"
                break
            if time.time() - t0 > WALL:
                stop = "wall"
                break
            draws = hull_draws([100000 + J * t_seed + j for j in range(J)])
            t_seed += 1
            rem = removed(d)
            cur = restrict(draws, rem)
            Lcur = float(np.mean([loss(x) for x in cur]))
            evals += J
            _, g = grad_d(cur[0])
            evals += 1
            gS = cg(H1, g).numpy()
            gmax = float(np.abs(gS).max())
            if not gmax > 0:
                break
            s = 1.0 if first else 2.0 * step_px
            acc = None
            while True:
                dt = np.minimum(np.maximum(d - (s * px / gmax) * gS, 0.0), thick_np)
                if float(np.abs(dt - d).max()) < 1e-3 * px:
                    break
                remt = removed(dt)
                if not any(bool((remt[k] != rem[k]).any()) for _, k in draws):
                    skipped += 1
                    s *= 0.5
                    continue
                Lt = float(np.mean([loss(x) for x in restrict(draws, remt)]))
                evals += J
                if Lt < Lcur:
                    acc = (dt, Lt, s)
                    break
                refused += 1
                s *= 0.5
            if acc is None:
                break
            d, Lacc, step_px = acc
            first = False
            steps += 1
            row = dict(level=li, ell_px=lv / px, step=steps, loss_cur=Lcur, loss_trial=Lacc, s=step_px,
                       removed=int(removed(d).sum()), d_max_px=float(d.max()) / px, evals=evals, skipped=skipped,
                       refused=refused, seconds=round(time.time() - t0, 1))
            log.append(row)
            print(json.dumps(dict(mesh=mesh, **row)), flush=True)
        if stop != "levels":
            break
    np.savez_compressed(out_dir / f"{mesh}_sfsg_g_r{RES}_d.npz", V=V.astype(np.float32), d=d, thick=thick_np)

    # the score: the marching-cubes solid, resampled by the run's sampler, the held-out seeds
    def solid(dv, name):
        F = depth_G.copy()
        dg = (W_G * dv[vid_G]).sum(1)
        F[in_G] = np.where(dg > 0, depth_G[in_G] - dg, depth_G[in_G])
        v, f, _, _ = marching_cubes(F.reshape(*nc), level=0.0, spacing=(hh, hh, hh), allow_degenerate=False)
        S = trimesh.Trimesh(v + lo, f, process=True)
        S.update_faces(S.nondegenerate_faces())
        S.merge_vertices()
        if not S.is_watertight:
            trimesh.repair.fill_holes(S)
        if S.volume < 0:
            S.invert()
        o = out_dir / f"{mesh}_{name}18_r{RES}.obj"
        trimesh.Trimesh(S.vertices / frame["scale"] + frame["offset"], S.faces, process=False).export(str(o))
        assert orient_name(str(o)) == "id", o
        n_s = int(round(n_vh * fill_volume_in_frame(str(o), frame, N) / fill_h))
        Lm, Lall = mean_loss(draws_in_frame(str(o), frame, n_s, FINAL))
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.core.Tensor(S.vertices.astype(np.float32)), o3d.core.Tensor(np.asarray(S.faces, np.uint32)))
        outside = float((sc.compute_occupancy(o3d.core.Tensor(tgt), nthreads=NT, nsamples=3).numpy() < 0.5).mean())
        return dict(volume=float(S.volume), loss=Lm, loss_all=Lall, mesh_outside=outside, watertight=bool(S.is_watertight),
                    n=n_s, out=str(o))

    base = solid(np.zeros(nV), "sfsgg0")
    sol = solid(d, "sfsgg")
    fin = hull_draws(FINAL)
    L_hullfile, L_hullfile_all = mean_loss([x for x, _ in fin])
    L_restr_end, L_restr_end_all = mean_loss(restrict(fin, removed(d)))
    L_floor, L_floor_all = mean_loss(draws_in_frame(path, frame, N, [SEED + 2 + 100 * k for k in range(8)]))
    vm = float(tc.volume)
    row = dict(mesh=mesh, mode="g", levels=[lv / px for lv in levels], n=N, res=RES, px=px, pitch=pitch, n_hull=n_vh,
               hull_vertices=nV, J=J, loss_floor=L_floor, loss_floor_all=L_floor_all, loss_hull_file=L_hullfile,
               loss_hull_file_all=L_hullfile_all, loss_restricted_end=L_restr_end,
               loss_restricted_end_all=L_restr_end_all, base=base, solid=sol,
               way=(base["loss"] - sol["loss"]) / (base["loss"] - L_floor),
               way_from_hull_file=(L_hullfile - sol["loss"]) / (L_hullfile - L_floor),
               way_restricted=(L_hullfile - L_restr_end) / (L_hullfile - L_floor),
               excess_removed=(base["volume"] - sol["volume"]) / (base["volume"] - vm), vol_mesh=vm,
               spearman_d_excess=float(spearmanr(d, e_v).correlation) if d.max() > 0 else None,
               d_q_px=[float(v) for v in np.quantile(d / px, [.5, .9, .99, .999])], carved_vertices=int((d > 0).sum()),
               steps=steps, evals=evals, skipped=skipped, refused=refused, stop=stop, log=log,
               seconds=time.time() - t0)
    row["passes"] = bool(row["way"] >= 0.5 and row["excess_removed"] >= 0.25
                         and sol["mesh_outside"] <= base["mesh_outside"] + 0.01)
    print(json.dumps({k: v for k, v in row.items() if k != "log"}), flush=True)
    with open(out_dir / f"sfsgg_r{RES}.jsonl", "a") as fo:
        fo.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    solve(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
