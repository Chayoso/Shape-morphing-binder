"""d156_views.py OUT_PNG MESH SEED [VIEWS] -- D156: side-by-side studio views of the hull target, P, L and the true mesh.

CPU (D157's studio renderer, d157_lib.render: the camera at 3.6 radii, 30 deg, three GGX lights, the backdrop, the tone
map). Columns: the hull (its perfect sample H's display surface), P's end, L's end, the true mesh's perfect sample M's
display surface, the true mesh itself (crease-aware normals). Rows: the views, given as THETA:PHI in degrees (default:
the three held-out views where the hull's display differs most from M's, by the shaded image's MSE). The display
surface is the field with Solenthaler et al.'s correction (zbs) at the run's pitch, as d156_read.py's (4)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d157_lib as D7                                                    # noqa: E402
import d156_read as R                                                    # noqa: E402

out, mesh, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
G = "/data/relcfd/chayo/physmorph_v2/output/gpu/d156/runs40"
L, P = f"{G}/{mesh}40k_L_{seed}_frames12.npz", f"{G}/{mesh}40k_P_{seed}_frames12.npz"
ctx = R.Ctx(mesh, seed, "hull", L)
end = lambda p: np.asarray(np.load(p, allow_pickle=True)["frames"][-1], np.float32)   # noqa: E731
surfs = {"hull (H)": D7.field_mesh("zbs", ctx.H, ctx.a)[0], "P (render off)": D7.field_mesh("zbs", end(P), ctx.a)[0],
         "L (render on)": D7.field_mesh("zbs", end(L), ctx.a)[0], "true mesh (M)": D7.field_mesh("zbs", ctx.M, ctx.a)[0],
         "true mesh (exact)": ctx.true_cn}
if len(sys.argv) > 4:
    views = [tuple(np.radians(float(v)) for v in s.split(":")) for s in sys.argv[4].split(",")]
else:
    hv = [D7.render(surfs["hull (H)"], ctx.center, ctx.radius, v)[0] for v in R.HELD]
    mse = [float(((h - m[0]) ** 2).mean()) for h, m in zip(hv, ctx.img_M)]
    views = [R.HELD[i] for i in np.argsort(mse)[::-1][:3]]
import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
fig, ax = plt.subplots(len(views), len(surfs), figsize=(3.2 * len(surfs), 3.2 * len(views)), squeeze=False)
for i, v in enumerate(views):
    for j, (name, s) in enumerate(surfs.items()):
        ax[i, j].imshow(D7.render(s, ctx.center, ctx.radius, v)[0])
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        if i == 0:
            ax[i, j].set_title(name, fontsize=11)
        if j == 0:
            ax[i, j].set_ylabel(f"view {np.degrees(v[0]):.0f}, {np.degrees(v[1]):.0f} deg", fontsize=9)
fig.suptitle(f"{mesh} 40k seed {seed}: held-out views (not used by the render loss)", fontsize=12)
fig.tight_layout()
fig.savefig(out, dpi=110)
print(out, [tuple(round(float(np.degrees(x)), 1) for x in v) for v in views])
