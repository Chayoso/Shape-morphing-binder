"""d156_sfs_f.py OUT_DIR N RES MESH [MESH ...] -- D156f: the shape-from-shading carve read through the run's own render
path (the docs entry D156f): the visual hull's carried sample moved inward particle by particle (x_i - delta_i nu_i,
0 <= delta_i <= the hull's thickness along -nu_i), the pictures of a state those of its Zhu-Bridson field's tracked discs
(Objective.render_terms: found at a state, read differentiably at the particles, found again when a tenth have moved more
than half a lattice pitch), the descent in delta smoothed by (I + l^2 L) on the particles' graph (pairs within one picture
pixel), coarse to fine l = box / 2^k down to one pixel, accept iff the loss falls. The carved solid: the moved
particles' exterior (field_mesh) taken back by the mesh sample's exterior offset c, resampled uniformly by the run's
sampler; every score the mean over eight seeds of that resample; the same extraction of the unmoved sample is the
baseline. CPU (D156_DEV=cpu) or a card."""
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
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from physmorph.pipeline.render_loss import d_exterior, exterior_targets  # noqa: E402
from physmorph.render.exterior import Lattice, make_field, tracked  # noqa: E402
from physmorph.sampling.mesh import (_oriented, draws_in_frame, fill_volume_in_frame, load_normalized,  # noqa: E402
                                     sample_in_frame)
from physmorph.sampling.orientation import orient_name  # noqa: E402

torch.set_grad_enabled(True)
dev = torch.device(os.environ.get("D156_DEV", "cpu"))
BUDGET = int(os.environ.get("D156_BUDGET", "500"))
RATIO = 2.0
cfg, AMB, views, SEED, NT = PH.cfg, PH.AMB, PH.views, PH.SEED, PH.NT


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


def solve(out_dir, N, RES, mesh):
    t0 = time.time()
    path, hpath = f"assets/{mesh}.obj", str(out_dir / f"{mesh}_hull18_r{RES}.obj")
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
    T0 = torch.as_tensor(tgt, device=dev)
    center = T0.mean(0)
    lattice = Lattice(center, 2.8 * float((T0 - center).norm(dim=1).max()))
    h = min(box / RES, 0.92 * pitch)
    cuda = dev.type == "cuda"

    def fresh(x):
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(x, np.float32), device=dev) if isinstance(x, np.ndarray) else x.detach()
            p, g, _, _ = lattice.discs(make_field("zb", x, pitch, device_field=cuda), h, refine=False)
        return p, nnf.normalize(g, dim=1)

    with torch.no_grad():
        samples = [tgt] + draws_in_frame(path, frame, N, [SEED + 1 + 100 * k for k in range(1, 8)])
        per = [exterior_targets(*fresh(s), views, RES, box, cfg.sil_k, AMB) for s in samples]
        sils = [torch.stack(v).mean(0) for v in zip(*[e[0] for e in per])]
        shade = [torch.stack(v).mean(0) for v in zip(*[e[1] for e in per])]

    def loss_pn(p, n):
        s, b = d_exterior(p, n, sils, shade, views, RES, box, cfg.sil_k, cfg.w_hole, cfg.w_spray, AMB)
        return s + cfg.w_pbr * b

    def mean_loss(xs_list):
        with torch.no_grad():
            v = [float(loss_pn(*fresh(x))) for x in xs_list]
        return float(np.mean(v)), [round(x, 6) for x in v]

    n_vh = int(round(N * fill_volume_in_frame(hpath, frame, N) / fill_volume_in_frame(path, frame, N)))
    X0np, _ = sample_in_frame(hpath, frame, n_vh, SEED + 1)
    X0np = X0np.astype(np.float64)
    seeds = [SEED + 1 + 100 * k for k in range(8)]
    L_floor, L_floor_all = mean_loss(draws_in_frame(path, frame, N, [SEED + 2 + 100 * k for k in range(8)]))
    # nu: the hull's outward normal at each particle's nearest hull point; t: the thickness along -nu
    cp, pid, _ = hull.closest(X0np)
    nu = hull.fn[pid]
    inside = hull.inside(X0np)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(hullt.vertices.astype(np.float32)), o3d.core.Tensor(np.asarray(hullt.faces, np.uint32)))
    cast = lambda o, dr: sc.cast_rays(o3d.core.Tensor(np.concatenate([o, dr], 1).astype(np.float32)), nthreads=NT)["t_hit"].numpy()  # noqa: E731
    t1 = cast(X0np, -nu)
    eps = 1e-4 * pitch
    t2 = np.full(len(X0np), np.inf)
    out_ = ~inside & np.isfinite(t1)
    t2[out_] = cast(X0np[out_] - (t1[out_] + eps)[:, None] * nu[out_], -nu[out_]) + t1[out_] + eps
    thick_np = np.where(inside, np.where(np.isfinite(t1), t1, 0.0), np.where(np.isfinite(t2), t2, 0.0))
    # the particles' graph and its Laplacian
    pairs = cKDTree(X0np).query_pairs(px, output_type="ndarray")
    I = torch.as_tensor(np.concatenate([pairs[:, 0], pairs[:, 1]]), device=dev)
    J = torch.as_tensor(np.concatenate([pairs[:, 1], pairs[:, 0]]), device=dev)
    X0 = torch.as_tensor(X0np, dtype=torch.float32, device=dev)
    nu_t = torch.as_tensor(nu, dtype=torch.float32, device=dev)
    thick = torch.as_tensor(thick_np, dtype=torch.float32, device=dev)
    M = len(X0)
    dx = X0[J] - X0[I]
    s2 = torch.zeros(M, device=dev).index_add_(0, I, (dx * dx).sum(1))
    c = torch.where(s2 > 0, 6.0 / s2.clamp_min(1e-30), torch.zeros_like(s2))
    w = 0.5 * (c[I] + c[J])
    wsum = torch.zeros(M, device=dev).index_add_(0, I, w)
    ell = {"v": px}

    def H1(f):
        return f + ell["v"] ** 2 * (wsum * f - torch.zeros(M, device=dev).index_add_(0, I, w * f[J]))

    def pos(dl):
        return X0 - dl[:, None] * nu_t

    st = {}

    def track(x):                                         # the run's discs: found at the state
        st["tr"] = tracked(make_field("zb", x.detach(), pitch, device_field=cuda), lattice, h, pitch)

    def loss_at(dl, grad=False):
        x = pos(dl)
        if grad:
            x = x.detach().requires_grad_(True)
        p, n, move = st["tr"].read(x)
        if float(torch.quantile(move.detach().abs()[::max(1, len(move) // 50000)], .9)) > .5 * h:
            track(x)                                      # moved past half a lattice pitch: found again (the run's rule)
            p, n, move = st["tr"].read(x)
        L = loss_pn(p, n)
        if grad:
            g = torch.autograd.grad(L, x)[0]
            return float(L), -(g * nu_t).sum(1)
        return float(L)

    dl = torch.zeros(M, device=dev)
    track(pos(dl))
    with torch.no_grad():
        L0 = L = loss_at(dl)
    levels = []
    lk = box
    while lk > px:
        levels.append(lk)
        lk = lk / RATIO
    levels.append(px)
    steps, evals, log = 0, 1, []
    for li, lv in enumerate(levels):
        ell["v"] = lv
        step_px = 1.0
        first = True
        while steps < BUDGET:
            track(pos(dl))
            _, g = loss_at(dl, grad=True)
            gS = cg(H1, g)
            gmax = float(gS.abs().max())
            if not gmax > 0:
                break
            s = 1.0 if first else 2.0 * step_px
            acc = None
            while True:
                dt = torch.minimum((dl - (s * px / gmax) * gS).clamp_min(0), thick)
                if float((dt - dl).abs().max()) < 1e-3 * px:
                    break
                with torch.no_grad():
                    Lt = loss_at(dt)
                evals += 1
                if Lt < L:
                    acc = (dt, Lt, s)
                    break
                s *= 0.5
            if acc is None:
                break
            dl, L, step_px = acc
            first = False
            steps += 1
            log.append(dict(level=li, ell_px=lv / px, step=steps, loss=L, delta_max_px=float(dl.max()) / px))
            if steps % 10 == 1:
                print(json.dumps(dict(mesh=mesh, level=li, step=steps, loss=L, loss0=L0, floor=L_floor,
                                      delta_max_px=float(dl.max()) / px, evals=evals,
                                      seconds=round(time.time() - t0, 1))), flush=True)
    Xe = pos(dl).detach().cpu().numpy().astype(np.float64)
    with torch.no_grad():
        L_moved = float(loss_pn(*fresh(Xe.astype(np.float32))))
    # the carved solid: the moved particles' exterior taken back by the mesh sample's own offset c, resampled
    zbt, _ = D7.field_mesh("zb", tgt, pitch)
    c_off = float(true.signed(zbt.V).mean())

    def extract(x, name):
        ext, _ = D7.field_mesh("zb", x, pitch)
        hh = 0.5 * pitch
        lo = x.min(0) - 3 * pitch
        nc = np.ceil((x.max(0) + 3 * pitch - lo) / hh).astype(int) + 1
        G = np.stack(np.meshgrid(*[lo[i] + hh * np.arange(nc[i]) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
        F = (-ext.signed(G) - c_off).reshape(*nc)
        v, f, _, _ = marching_cubes(F, level=0.0, spacing=(hh, hh, hh), allow_degenerate=False)
        S = trimesh.Trimesh(v + lo, f, process=True)
        S.update_faces(S.nondegenerate_faces())
        S.merge_vertices()
        if not S.is_watertight:
            trimesh.repair.fill_holes(S)
        if S.volume < 0:
            S.invert()
        Sm = trimesh.Trimesh(S.vertices / frame["scale"] + frame["offset"], S.faces, process=False)
        o = out_dir / f"{mesh}_{name}18_r{RES}.obj"
        Sm.export(str(o))
        assert orient_name(str(o)) == "id", o
        n_s = int(round(n_vh * fill_volume_in_frame(str(o), frame, N) / fill_volume_in_frame(hpath, frame, N)))
        Lm, Lall = mean_loss(draws_in_frame(str(o), frame, n_s, seeds))
        osc = o3d.t.geometry.RaycastingScene()
        osc.add_triangles(o3d.core.Tensor(S.vertices.astype(np.float32)), o3d.core.Tensor(np.asarray(S.faces, np.uint32)))
        outside = float((osc.compute_occupancy(o3d.core.Tensor(tgt), nthreads=NT, nsamples=3).numpy() < 0.5).mean())
        return dict(volume=float(S.volume), loss=Lm, loss_all=Lall, mesh_outside=outside, watertight=bool(S.is_watertight),
                    n=n_s, out=str(o))

    base = extract(X0np, "sfsf0")
    sol = extract(Xe, "sfsf")
    L_hullfile, _ = mean_loss(draws_in_frame(hpath, frame, n_vh, seeds))
    vm = float(tc.volume)
    surf = np.abs(hull.signed(X0np)) < pitch
    e = true.udist(X0np)
    dpx = dl.cpu().numpy() / px
    rho = float(spearmanr(dpx[surf], e[surf]).correlation) if surf.sum() > 10 else None
    row = dict(mesh=mesh, mode="f", levels=[lv / px for lv in levels], n=N, res=RES, px=px, pitch=pitch, particles=M,
               pairs=int(len(pairs)), c_offset_pitch=c_off / pitch, loss_start=L0, loss_end=L, loss_moved_fresh=L_moved,
               loss_floor=L_floor, loss_floor_all=L_floor_all, loss_hull_file=L_hullfile, base=base, solid=sol,
               way=(base["loss"] - sol["loss"]) / (base["loss"] - L_floor),
               excess_removed=(base["volume"] - sol["volume"]) / (base["volume"] - vm), vol_mesh=vm,
               spearman_delta_excess_surface=rho, delta_q_px=[float(v) for v in np.quantile(dpx, [.5, .9, .99, .999])],
               steps=steps, evals=evals, budget_hit=steps >= BUDGET, log=log, seconds=time.time() - t0)
    row["passes"] = bool(row["way"] >= 0.5 and row["excess_removed"] >= 0.25
                         and sol["mesh_outside"] <= base["mesh_outside"] + 0.01)
    print(json.dumps({k: v for k, v in row.items() if k != "log"}), flush=True)
    np.savez_compressed(out_dir / f"{mesh}_sfsf18_r{RES}_delta.npz", x0=X0np.astype(np.float32), nu=nu.astype(np.float32),
                        delta=dl.cpu().numpy(), thick=thick_np)
    with open(out_dir / f"sfsf_r{RES}.jsonl", "a") as fo:
        fo.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    od, n, res = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    for ms in sys.argv[4:]:
        solve(od, n, res, ms)
