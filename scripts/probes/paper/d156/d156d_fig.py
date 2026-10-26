import sys, numpy as np
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/repo_d156"); sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d156")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import d157_lib as D7
from physmorph.sampling.mesh import _oriented, load_normalized
A = "/data/relcfd/chayo/physmorph_v2/output/gpu/d156/assets_hull"
meshes = sys.argv[2:]; out = sys.argv[1]
fig, ax = plt.subplots(len(meshes), 2, figsize=(10, 4.6 * len(meshes)), squeeze=False)
for i, mesh in enumerate(meshes):
    z = np.load(f"{A}/{mesh}_sfs18_r96_d.npz"); p0, d = z["p0"].astype(np.float64), z["d"]
    _, v_src = load_normalized("assets/isosphere.obj", 40000, 97, return_volume=True, sample="stratified")
    fr = {}; tgt = load_normalized(f"assets/{mesh}.obj", 40000, 98, match_volume=v_src, sample="stratified", frame=fr)
    px = 2 * 1.25 * float(np.abs(tgt).max()) / 96
    m = _oriented(f"assets/{mesh}.obj"); Vw = (np.asarray(m.vertices, np.float64) - fr["offset"]) * fr["scale"]
    tc, _ = D7.close_holes(Vw, np.asarray(m.faces, np.int64)); true = D7.Surf(np.asarray(tc.vertices), np.asarray(tc.faces))
    e = true.udist(p0); ep = np.clip((e - np.median(e)) / px, 0, None); dp = d / px
    th, ph = np.radians(30.0), np.radians(35.0)            # a view between the training rings (held out)
    r = np.array([np.cos(th), 0, -np.sin(th)]); u = np.array([-np.sin(ph) * np.sin(th), np.cos(ph), -np.sin(ph) * np.cos(th)])
    dv = np.cross(r, u); front = np.argsort(p0 @ dv)          # far to near, the near drawn last
    for j, (val, title) in enumerate(((ep, "true excess: hull minus mesh"),
                                      (dp, "carved by the render shading (d)"))):
        a = ax[i, j]
        sc = a.scatter((p0 @ r)[front], (p0 @ u)[front], c=val[front], s=1.2, cmap="Blues", vmin=0, vmax=2.0, linewidths=0)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{mesh}: {title}", fontsize=9)
    fig.colorbar(sc, ax=ax[i, :], shrink=0.8, label="picture pixels (one pixel ≈ one pitch)")
fig.suptitle("D156d check: the shading descent carves where the hull is too big, but only part of the depth", fontsize=11)
fig.savefig(out, dpi=120, bbox_inches="tight"); print(out)
