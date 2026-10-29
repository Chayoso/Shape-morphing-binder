import sys, json, numpy as np
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/repo_d156"); sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d156")
import d157_lib as D7
from physmorph.sampling.mesh import _oriented, load_normalized
from scipy.stats import spearmanr
A = "/data/relcfd/chayo/physmorph_v2/output/gpu/d156/assets_hull"
for mesh in sys.argv[1:]:
    z = np.load(f"{A}/{mesh}_sfs18_r96_d.npz"); p0, d = z["p0"].astype(np.float64), z["d"]
    _, v_src = load_normalized("assets/isosphere.obj", 40000, 97, return_volume=True, sample="stratified")
    fr = {}; tgt = load_normalized(f"assets/{mesh}.obj", 40000, 98, match_volume=v_src, sample="stratified", frame=fr)
    px = 2 * 1.25 * float(np.abs(tgt).max()) / 96
    m = _oriented(f"assets/{mesh}.obj"); Vw = (np.asarray(m.vertices, np.float64) - fr["offset"]) * fr["scale"]
    tc, _ = D7.close_holes(Vw, np.asarray(m.faces, np.int64)); true = D7.Surf(np.asarray(tc.vertices), np.asarray(tc.faces))
    e = true.udist(p0)                      # distance of the hull's exterior disc to the mesh: offset + the excess depth there
    e0 = np.median(e)                       # the typical offset where the hull touches the mesh
    dp = d / px; ep = (e - e0) / px
    rho = spearmanr(dp, ep).correlation
    deep, flat = ep > 1, ep < 0.25
    print(json.dumps(dict(mesh=mesh, discs=len(d), share_d_over_half_px=float((dp > .5).mean()), share_d_over_2px=float((dp > 2).mean()),
          d_q=[float(x) for x in np.quantile(dp, [.5, .9, .99, .999])], spearman_d_excess=float(rho),
          mean_d_deep_px=float(dp[deep].mean()) if deep.any() else None, mean_d_flat_px=float(dp[flat].mean()),
          share_deep=float(deep.mean()), excess_q=[float(x) for x in np.quantile(ep, [.5, .9, .99])],
          d_over_2px_on_deep=float((deep & (dp > 2)).sum() / max((dp > 2).sum(), 1)))), flush=True)
