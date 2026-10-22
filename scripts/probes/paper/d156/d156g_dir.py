"""d156g_dir.py OUT_DIR HULL_DIR N RES MESH -- D156g's direction diagnostic (after its reading): at d = 0, does the
proposal gradient point where the hull is in excess, before and after the coarse-to-fine H1 smoothing? D156g's setup
exactly (the frame, the pictures, the hull's restricted draws, the vertex graph and its Laplacian, the scatter); the
gradient averaged over the four draws of D156g's first step (seeds 100 000 + j); per vertex the true excess e (the hull
vertex's distance to the mesh) and its region (excess: e >= one picture pixel; on the mesh: e <= a quarter pixel).
Reported per smoothing length (D156g's levels): the share of each region the direction would carve (g_S < 0), the
area-weighted mean of g_S per region, Spearman of -g_S with e; for (a) the full gradient smoothed (D156g's direction) and
(b) its feasible part at d = 0 smoothed, min(g, 0) (the cone's projection before the Riesz map); the silhouette and the
shading parts apart, and the shading's feasible part. One JSON line per mesh
in OUT_DIR/d156g_dir2.jsonl. CPU."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import physmorph  # noqa: E402,F401  (before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as nnf  # noqa: E402
import d156_photohull as PH  # noqa: E402
from d156_photohull import D7, trimesh  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from physmorph.pipeline.render_loss import d_exterior, exterior_targets  # noqa: E402
from physmorph.render.exterior import Lattice, make_field, tracked  # noqa: E402
from physmorph.sampling.mesh import _oriented, _stratified_fill, draws_in_frame, fill_volume_in_frame, load_normalized  # noqa: E402
from d156_sfs_g import cg, bary  # noqa: E402

torch.set_grad_enabled(True)
cfg, AMB, views, SEED = PH.cfg, PH.AMB, PH.views, PH.SEED
J = 4


def main(out_dir, hull_dir, N, RES, mesh):
    path, hpath = f"assets/{mesh}.obj", str(hull_dir / f"{mesh}_hull18_r{RES}.obj")
    _, v_src = load_normalized("assets/isosphere.obj", N, SEED, return_volume=True, sample="stratified")
    frame = {}
    tgt = load_normalized(path, N, SEED + 1, match_volume=v_src, sample="stratified", frame=frame)
    box = float(np.abs(tgt).max()) * 1.25
    px = 2.0 * box / RES
    m = _oriented(path)
    Vw = (np.asarray(m.vertices, np.float64) - frame["offset"]) * frame["scale"]
    tc, _ = D7.close_holes(Vw, np.asarray(m.faces, np.int64))
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
    with torch.no_grad():
        samples = [tgt] + draws_in_frame(path, frame, N, [SEED + 1 + 100 * k for k in range(1, 8)])
        per = []
        for s in samples:
            p, g, _, _ = lattice.discs(make_field("zb", torch.as_tensor(s), pitch), h, refine=False)
            per.append(exterior_targets(p, nnf.normalize(g, dim=1), views, RES, box, cfg.sil_k, AMB))
        sils = [torch.stack(v).mean(0) for v in zip(*[e[0] for e in per])]
        shade = [torch.stack(v).mean(0) for v in zip(*[e[1] for e in per])]
    n_vh = int(round(N * fill_volume_in_frame(hpath, frame, N) / fill_volume_in_frame(path, frame, N)))
    hi, centers, vp = _stratified_fill(hm, n_vh)
    HF = np.asarray(hullt.faces, np.int64)
    V = np.asarray(hullt.vertices, np.float64)
    nV = len(V)
    A_v = np.zeros(nV)
    np.add.at(A_v, HF.ravel(), np.repeat(hullt.area_faces / 3.0, 3))
    E = np.asarray(hullt.edges_unique, np.int64)
    I = torch.as_tensor(np.concatenate([E[:, 0], E[:, 1]]))
    Jn = torch.as_tensor(np.concatenate([E[:, 1], E[:, 0]]))
    Vt = torch.as_tensor(V)
    dxe = Vt[Jn] - Vt[I]
    s2 = torch.zeros(nV, dtype=torch.float64).index_add_(0, I, (dxe * dxe).sum(1))
    c = torch.where(s2 > 0, 4.0 / s2.clamp_min(1e-30), torch.zeros_like(s2))
    w = 0.5 * (c[I] + c[Jn])
    wsum = torch.zeros(nV, dtype=torch.float64).index_add_(0, I, w)
    gsum = np.zeros(nV)
    gsum_s, gsum_b = np.zeros(nV), np.zeros(nV)
    for j in range(J):
        rng = np.random.default_rng(100000 + j)
        keep = rng.choice(len(centers), n_vh, replace=False) if len(centers) > n_vh else np.arange(n_vh)
        jitter = (rng.uniform(-0.5, 0.5, (n_vh, 3)) * vp).astype(np.float32)
        x = (centers[keep] + jitter).astype(np.float32)
        x = ((x.astype(np.float64) - frame["offset"]) * frame["scale"]).astype(np.float32)
        xt = torch.as_tensor(x)
        tr = tracked(make_field("zb", xt, pitch), lattice, h, pitch)
        xr = xt.clone().requires_grad_(True)
        p, n, _ = tr.read(xr)
        s, b = d_exterior(p, n, sils, shade, views, RES, box, cfg.sil_k, cfg.w_hole, cfg.w_spray, AMB)
        gxs = torch.autograd.grad(s, xr, retain_graph=True)[0].numpy().astype(np.float64)
        gxb = torch.autograd.grad(b, xr)[0].numpy().astype(np.float64)
        gx = gxs + cfg.w_pbr * gxb
        _, fid, uv = hull.closest(x)
        gi = -(gx * hull.fn[fid]).sum(1)
        np.add.at(gsum, HF[fid].ravel(), (bary(uv) * gi[:, None]).ravel())
        for gg, acc in ((gxs, gsum_s), (gxb, gsum_b)):
            np.add.at(acc, HF[fid].ravel(), (bary(uv) * (-(gg * hull.fn[fid]).sum(1))[:, None]).ravel())
    g = gsum / J / np.maximum(A_v, 1e-30)
    g_s, g_b = gsum_s / J / np.maximum(A_v, 1e-30), cfg.w_pbr * gsum_b / J / np.maximum(A_v, 1e-30)
    e = true.udist(V) / px
    exc, onm = e >= 1.0, e <= 0.25
    levels = []
    lk = box
    while lk > px:
        levels.append(lk)
        lk = lk / 2.0
    levels.append(px)

    def stats(gS):
        return dict(carve_share_excess=float((gS[exc] < 0).mean()), carve_share_on_mesh=float((gS[onm] < 0).mean()),
                    mean_excess=float((gS[exc] * A_v[exc]).sum() / A_v[exc].sum()),
                    mean_on_mesh=float((gS[onm] * A_v[onm]).sum() / A_v[onm].sum()),
                    spearman=float(spearmanr(-gS, e).correlation))

    rows = []
    for lv in [0.0] + levels:
        def H1(f, lv=lv):
            return f + lv ** 2 * (wsum * f - torch.zeros(nV, dtype=torch.float64).index_add_(0, I, w * f[Jn]))
        full = g if lv == 0 else cg(H1, torch.as_tensor(g)).numpy()
        feas = np.minimum(g, 0.0) if lv == 0 else cg(H1, torch.as_tensor(np.minimum(g, 0.0))).numpy()
        sm = lambda v: v if lv == 0 else cg(H1, torch.as_tensor(v)).numpy()  # noqa: E731
        rows.append(dict(ell_px=lv / px, full=stats(full), feasible=stats(feas), sil=stats(sm(g_s)), pbr=stats(sm(g_b)),
                         pbr_feasible=stats(sm(np.minimum(g_b, 0.0)))))
    row = dict(mesh=mesh, vertices=nV, excess_vertices=int(exc.sum()), on_mesh_vertices=int(onm.sum()),
               excess_area_share=float(A_v[exc].sum() / A_v.sum()), levels=rows)
    print(json.dumps(row), flush=True)
    with open(out_dir / "d156g_dir2.jsonl", "a") as fo:
        fo.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
