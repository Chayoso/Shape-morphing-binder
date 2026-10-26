"""d156_sfs_e.py OUT_DIR N RES MESH [MESH ...] -- D156e (D156d made coarse to fine, bounded by the hull's thickness, scored on
the pipeline's own samples; D156_RATIO the level ratio, 2 by default; the docs entry D156e has the changes). D156d: multiview shape from shading of the target, the visual hull's
surface carved as a whole by the render's own operator (Hernández, Vogiatzis and Cipolla, PAMI 2008, with our
operator; the descent in Neuberger's Sobolev metric, Sundaramoorthi, Yezzi and Mennucci, IJCV 2007).

Frame, photographs, operator and loss are D156c's: the true mesh's run frame at seed 97, the mean of its eight samples'
exterior pictures at RES px, the exterior operator (pitch, lattice, h of the seed-98 sample), sil + w_pbr pbr of
d_exterior. The surface: the exterior discs of the visual hull's carried sample (at the mesh sample's density), found
once (p0, n0). The unknown d >= 0 per disc: discs p = p0 - d n0, normals normalize(n0 + grad_s d), grad_s d the
least-squares tangential gradient over the lattice neighbours (the 26 cells around each disc's cell). The descent: the
loss gradient in d smoothed by (I + l^2 L) g_S = g (L the disc graph's Laplacian, scaled as a surface Laplacian,
symmetrised; l = one picture pixel), conjugate gradients; the step d <- max(0, d - a g_S), the first trial moving the
largest disc one pixel, accepted iff the loss falls, halved on a refusal, the next step starting at twice the last;
stop when no trial down to a largest move of 1e-3 pixel lowers the loss (or D156_BUDGET accepted steps). The solid:
{x : depth below the visual hull's surface > d of the nearest disc}, marching cubes on a half-pitch grid, written in the
mesh's oriented coordinates as OUT_DIR/MESH_sfs18_rRES.obj. One JSON line per mesh: L0, L_end, the floor (an
independent sample of the mesh), L_solid (a fresh sample of the solid at the hull's density), the excess volume removed,
the over-carve, the steps. D156_DEV=cpu runs it on the CPU (a check of the code)."""
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
from physmorph.pipeline.render_loss import d_exterior, exterior_targets  # noqa: E402
from physmorph.render.exterior import Lattice, key, make_field  # noqa: E402
from physmorph.sampling.mesh import (_oriented, draws_in_frame, fill_volume_in_frame, load_normalized,  # noqa: E402
                                     sample_in_frame, stratified_draws)
from physmorph.sampling.orientation import orient_name  # noqa: E402

torch.set_grad_enabled(True)                                  # d157_lib turns it off at import
dev = torch.device(os.environ.get("D156_DEV", "cuda"))
BUDGET = int(os.environ.get("D156_BUDGET", "500"))
RATIO = float(os.environ.get("D156_RATIO", "2"))
TAG = "" if RATIO == 2.0 else f"_r{RATIO:g}".replace(".", "p")
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
    hm = _oriented(hpath)
    Hw = (np.asarray(hm.vertices, np.float64) - frame["offset"]) * frame["scale"]
    hullt = trimesh.Trimesh(Hw, np.asarray(hm.faces), process=True)
    if hullt.volume < 0:
        hullt.invert()
    pitch = 0.708 * D7.median_kth_spacing(tgt, 8, 20000)
    T0 = torch.as_tensor(tgt, device=dev)
    center = T0.mean(0)
    lattice = Lattice(center, 2.8 * float((T0 - center).norm(dim=1).max()))
    h = min(box / RES, 0.92 * pitch)

    def discs_of(x):
        x = torch.as_tensor(np.asarray(x, np.float32), device=dev)
        with torch.no_grad():
            p, g, _, _ = lattice.discs(make_field("zb", x, pitch, device_field=dev.type == "cuda"), h, refine=False)
        return p, nnf.normalize(g, dim=1)

    with torch.no_grad():
        samples = [tgt] + draws_in_frame(path, frame, N, [SEED + 1 + 100 * k for k in range(1, 8)])
        per = [exterior_targets(*discs_of(s), views, RES, box, cfg.sil_k, AMB) for s in samples]
        sils = [torch.stack(v).mean(0) for v in zip(*[e[0] for e in per])]
        shade = [torch.stack(v).mean(0) for v in zip(*[e[1] for e in per])]

    def loss_pn(p, n):
        s, b = d_exterior(p, n, sils, shade, views, RES, box, cfg.sil_k, cfg.w_hole, cfg.w_spray, AMB)
        return s + cfg.w_pbr * b

    n_vh = int(round(N * fill_volume_in_frame(hpath, frame, N) / fill_volume_in_frame(path, frame, N)))
    xs, _ = sample_in_frame(hpath, frame, n_vh, SEED + 1)
    p0, n0 = discs_of(xs)
    floor_x, _ = sample_in_frame(path, frame, N, SEED + 2)
    with torch.no_grad():
        L_floor = float(loss_pn(*discs_of(floor_x)))
    M = len(p0)
    # the disc graph: discs in the 26 lattice cells around each disc's cell
    cell = torch.floor((p0 - lattice.origin) / h).long()
    ck = key(cell)
    order = torch.argsort(ck)
    sk = ck[order]
    I, J = [], []
    for off in torch.stack(torch.meshgrid(*[torch.arange(-1, 2, device=dev)] * 3, indexing="ij"), -1).reshape(-1, 3):
        if bool((off == 0).all()):
            continue
        nk = key(cell + off)
        at = torch.searchsorted(sk, nk).clamp(max=M - 1)
        hit = sk[at] == nk
        I.append(torch.nonzero(hit).squeeze(1))
        J.append(order[at[hit]])
    I, J = torch.cat(I), torch.cat(J)
    dp = p0[J] - p0[I]
    dpt = dp - (dp * n0[I]).sum(1, keepdim=True) * n0[I]                # tangential offsets
    A = torch.zeros(M, 3, 3, device=dev).index_add_(0, I, dpt[:, :, None] * dpt[:, None, :])
    A = A + n0[:, :, None] * n0[:, None, :] * (A.diagonal(dim1=1, dim2=2).sum(1)[:, None, None] + 1e-12)
    Minv = torch.linalg.inv(A.double()).float()
    s2 = torch.zeros(M, device=dev).index_add_(0, I, (dp * dp).sum(1))
    deg = torch.zeros(M, device=dev).index_add_(0, I, torch.ones_like(I, dtype=torch.float32))
    c = torch.where(s2 > 0, 4.0 / s2.clamp_min(1e-30), torch.zeros_like(s2))
    w = 0.5 * (c[I] + c[J])                                              # symmetrised surface-Laplacian weights
    wsum = torch.zeros(M, device=dev).index_add_(0, I, w)
    # the hull's thickness under each disc: the ray along -n0 enters the hull (t1) and leaves it (t1 + thick)
    hsc = o3d.t.geometry.RaycastingScene()
    hsc.add_triangles(o3d.core.Tensor(hullt.vertices.astype(np.float32)), o3d.core.Tensor(np.asarray(hullt.faces, np.uint32)))
    P0n, N0n = p0.cpu().numpy().astype(np.float64), n0.cpu().numpy().astype(np.float64)
    t1 = hsc.cast_rays(o3d.core.Tensor(np.concatenate([P0n, -N0n], 1).astype(np.float32)), nthreads=NT)["t_hit"].numpy()
    eps = 1e-4 * pitch                                                   # numerical: past the entry face
    ok1 = np.isfinite(t1)
    t2 = np.full(M, np.inf)
    t2[ok1] = hsc.cast_rays(o3d.core.Tensor(np.concatenate([P0n[ok1] - (t1[ok1] + eps)[:, None] * N0n[ok1], -N0n[ok1]], 1)
                                            .astype(np.float32)), nthreads=NT)["t_hit"].numpy()
    thick_np = np.where(ok1 & np.isfinite(t2), t2 + eps, 0.0)
    thick = torch.as_tensor(thick_np, dtype=torch.float32, device=dev)
    ell = {"v": px}

    def H1(f):                                                           # (I + l^2 L) f at the current level
        Lf = wsum * f - torch.zeros(M, device=dev).index_add_(0, I, w * f[J])
        return f + ell["v"] ** 2 * Lf

    def surface(d):
        S = torch.zeros(M, 3, device=dev).index_add(0, I, dpt * (d[J] - d[I])[:, None])
        gs = torch.einsum("mij,mj->mi", Minv, S)
        gs = gs - (gs * n0).sum(1, keepdim=True) * n0
        return p0 - d[:, None] * n0, nnf.normalize(n0 + gs, dim=1)

    def loss_d(d):
        return loss_pn(*surface(d))

    d = torch.zeros(M, device=dev)
    with torch.no_grad():
        L0 = L = float(loss_d(d))
    steps, evals, log = 0, 1, []
    levels = []
    lk = box
    while lk > px:
        levels.append(lk)
        lk = lk / RATIO
    levels.append(px)
    for li, lv in enumerate(levels):
      ell["v"] = lv
      step_px, lsteps = 1.0, 0
      while steps < BUDGET:
          dg = d.clone().requires_grad_(True)
          g = torch.autograd.grad(loss_d(dg), dg)[0]
          gS = cg(H1, g)
          gmax = float(gS.abs().max())
          if not gmax > 0:
              break
          acc = None
          s = 2.0 * step_px if steps else 1.0
          while True:
              dt = torch.minimum((d - (s * px / gmax) * gS).clamp_min(0), thick)
              move = float((dt - d).abs().max())
              if move < 1e-3 * px:
                  break
              with torch.no_grad():
                  Lt = float(loss_d(dt))
              evals += 1
              if Lt < L:
                  acc = (dt, Lt, s)
                  break
              s *= 0.5
          if acc is None:
              break
          d, L, step_px = acc
          steps += 1
          log.append(dict(level=li, ell_px=lv / px, step=steps, loss=L, step_px=step_px, d_max_px=float(d.max()) / px, d_mean_px=float(d.mean()) / px))
          if steps % 10 == 1:
              print(json.dumps(dict(mesh=mesh, step=steps, loss=L, loss0=L0, floor=L_floor, d_max_px=float(d.max()) / px,
                                    evals=evals, seconds=round(time.time() - t0, 1))), flush=True)
    # the solid: depth below the hull's surface > the H1-smoothed d (one pixel) of the nearest disc, within the thickness
    ell["v"] = px
    with torch.no_grad():
        ds = torch.minimum(cg(H1, d).clamp_min(0), thick)
    dn = ds.detach().cpu().numpy().astype(np.float64)
    draw_d = d.detach().cpu().numpy().astype(np.float64)
    P0 = p0.detach().cpu().numpy().astype(np.float64)
    hh = 0.5 * pitch
    lo = Hw.min(0) - 3 * pitch
    nc = np.ceil((Hw.max(0) + 3 * pitch - lo) / hh).astype(int) + 1
    G = np.stack(np.meshgrid(*[lo[i] + hh * np.arange(nc[i]) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    sdh = hsc.compute_signed_distance(o3d.core.Tensor(G.astype(np.float32)), nthreads=NT, nsamples=3).numpy().astype(np.float64)
    dnn = dn[cKDTree(P0).query(G, workers=NT)[1]]
    F = (-sdh - dnn).reshape(*nc)
    verts, faces, _, _ = marching_cubes(F, level=0.0, spacing=(hh, hh, hh), allow_degenerate=False)
    Sw = trimesh.Trimesh(verts + lo, faces, process=True)
    Sw.update_faces(Sw.nondegenerate_faces())
    Sw.merge_vertices()
    if not Sw.is_watertight:
        trimesh.repair.fill_holes(Sw)
    if Sw.volume < 0:
        Sw.invert()
    Sm = trimesh.Trimesh(Sw.vertices / frame["scale"] + frame["offset"], Sw.faces, process=False)
    out = out_dir / f"{mesh}_sfse{TAG}18_r{RES}.obj"
    Sm.export(str(out))
    assert orient_name(str(out)) == "id", out
    # scores on the pipeline's own samples (drawn in mesh coordinates), the means over eight seeds
    seeds = [SEED + 1 + 100 * k for k in range(8)]
    n_s = int(round(n_vh * fill_volume_in_frame(str(out), frame, N) / fill_volume_in_frame(hpath, frame, N)))

    def mean_loss(xs_list):
        with torch.no_grad():
            v = [float(loss_pn(*discs_of(x))) for x in xs_list]
        return float(np.mean(v)), [round(x, 6) for x in v]

    L_solid, L_solid_all = mean_loss(draws_in_frame(str(out), frame, n_s, seeds))
    L_hullm, L_hull_all = mean_loss(draws_in_frame(hpath, frame, n_vh, seeds))
    L_floorm, L_floor_all = mean_loss(draws_in_frame(path, frame, N, [SEED + 2 + 100 * k for k in range(8)]))
    # where it carved: the smoothed d against the true excess depth under each disc
    from scipy.stats import spearmanr
    tsurf = D7.Surf(np.asarray(tc.vertices), np.asarray(tc.faces))
    e = tsurf.udist(P0)
    ep = (e - np.median(e)) / px
    dp = dn / px
    diag = dict(spearman_d_excess=float(spearmanr(dp, ep).correlation), mean_d_deep_px=float(dp[ep > 1].mean()),
                mean_d_touch_px=float(dp[ep < .25].mean()), share_deep=float((ep > 1).mean()),
                d_q_px=[float(v) for v in np.quantile(dp, [.5, .9, .99, .999])],
                thick_zero_share=float((thick_np <= 0).mean()), draw_d_max_px=float(draw_d.max()) / px)
    def outside(Vs, Fs):
        s = o3d.t.geometry.RaycastingScene()
        s.add_triangles(o3d.core.Tensor(np.asarray(Vs, np.float32)), o3d.core.Tensor(np.asarray(Fs, np.uint32)))
        return float((s.compute_occupancy(o3d.core.Tensor(tgt), nthreads=NT, nsamples=3).numpy() < 0.5).mean())

    vm, vh, vs = float(tc.volume), float(hullt.volume), float(Sw.volume)
    row = dict(mesh=mesh, mode="e", ratio=RATIO, levels=[lv / px for lv in levels], n=N, res=RES, px=px, pitch=pitch,
               discs=M, loss_discs_hull=L0, loss_discs_end=L, loss_floor_single=L_floor,
               loss_solid=L_solid, loss_hull=L_hullm, loss_floor=L_floorm,
               loss_solid_all=L_solid_all, loss_hull_all=L_hull_all, loss_floor_all=L_floor_all,
               way_solid=(L_hullm - L_solid) / (L_hullm - L_floorm), way_discs=(L0 - L) / (L0 - L_floor),
               steps=steps, evals=evals, budget_hit=steps >= BUDGET, d_max_px=float(dn.max()) / px,
               d_mean_px=float(dn.mean()) / px, vol_mesh=vm, vol_hull=vh, vol_solid=vs,
               excess_removed=(vh - vs) / (vh - vm), mesh_outside_hull=outside(hullt.vertices, hullt.faces),
               mesh_outside_solid=outside(Sw.vertices, Sw.faces), watertight=bool(Sw.is_watertight), out=str(out),
               **diag, log=log, seconds=time.time() - t0)
    row["passes"] = bool(row["way_solid"] >= 0.5 and row["excess_removed"] >= 0.25
                         and row["mesh_outside_solid"] <= row["mesh_outside_hull"] + 0.01)
    print(json.dumps({k: v for k, v in row.items() if k != "log"}), flush=True)
    np.savez_compressed(out_dir / f"{mesh}_sfse{TAG}18_r{RES}_d.npz", p0=P0.astype(np.float32), n0=n0.cpu().numpy(),
                        d=draw_d, d_smooth=dn, thick=thick_np)
    with open(out_dir / f"sfse_r{RES}.jsonl", "a") as fo:
        fo.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    od, n, res = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    for ms in sys.argv[4:]:
        solve(od, n, res, ms)
