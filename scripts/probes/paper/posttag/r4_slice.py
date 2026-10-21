"""r4_slice.py OUT_PREFIX MESH AXIS VALUE U0 U1 V0 V1 LABEL=ARCHIVE[:ind|:tgt] [...] -- D145 reading 4 (2), observed, not
gated: the display surface cut by one plane, to see where an end state's loop closes (the teapot's handle).

The yardstick's exterior field (Zhu and Bridson, radius 3, offset 0.8 pitches, repo_r77) of each state evaluated on a
grid in the plane AXIS = VALUE (a third of a pitch; box U0..U1 x V0..V1 in the other two axes, in order), the outside
(field > 0) split into connected regions: the regions that do not reach the box's border are holes enclosed in the
slice (a loop's opening, or an internal void). Per state one JSON line (each enclosed region's area in pitches^2 and
its centroid) and one panel of OUT_PREFIX.png (the zero contour over the state's particles within half a pitch of the
plane). ARCHIVE's last finite frame; with `:ind` D90's independent sample, with `:tgt` the archive's own target
sample (the floor). CPU."""
import json
import sys

import numpy as np
import torch
from scipy.ndimage import label as cc_label
from scipy.spatial import cKDTree

B = "/data/relcfd/chayo/physmorph_v2"
sys.path.insert(0, f"{B}/repo_r77")
from physmorph.render.exterior import ZhuBridson  # noqa: E402

torch.set_grad_enabled(False)
torch.set_num_threads(6)
prefix, mesh, axis, value = sys.argv[1], sys.argv[2], "xyz".index(sys.argv[3]), float(sys.argv[4])
u0, u1, v0, v1 = (float(s) for s in sys.argv[5:9])
I = np.asarray(np.load(f"{B}/output/gpu/d90/{mesh}_ind300k.npz")["tgt"], np.float64)
a = 0.708 * float(np.median(cKDTree(I).query(I[::20], 9, workers=6)[0][:, 8]))
ua, va = [k for k in range(3) if k != axis]
h = a / 3
us, vs = np.arange(u0, u1, h), np.arange(v0, v1, h)
U, V = np.meshgrid(us, vs, indexing="ij")
q = np.zeros((U.size, 3))
q[:, axis], q[:, ua], q[:, va] = value, U.ravel(), V.ravel()
states = []
for spec in sys.argv[9:]:
    lab, path = spec.split("=", 1)
    kind = path.rsplit(":", 1)[1] if path.endswith((":ind", ":tgt")) else "end"
    path = path[:-4] if kind != "end" else path
    z = np.load(path, allow_pickle=True)
    if kind == "ind":
        x = I
    elif kind == "tgt":
        x = np.asarray(z["tgt"], np.float64)
    else:
        F = z["frames"]
        k = len(F) - 1
        while k > 0 and not np.isfinite(np.asarray(F[k][:16])).all():
            k -= 1
        x = np.asarray(F[k], np.float64)
    field = ZhuBridson(torch.as_tensor(x, dtype=torch.float32), a)
    near = cKDTree(x).query(q, 1, distance_upper_bound=field.radius, workers=6)[0] < field.radius
    f = np.full(len(q), np.inf)
    out = []
    for s in range(0, int(near.sum()), 200000):
        out.append(field(torch.as_tensor(q[near][s:s + 200000], dtype=torch.float32))[0].numpy())
    f[near] = np.concatenate(out) if out else []
    f = f.reshape(U.shape)
    reg, n = cc_label(f > 0)
    border = set(np.unique(np.concatenate([reg[0], reg[-1], reg[:, 0], reg[:, -1]]))) - {0}
    holes = []
    for r in range(1, n + 1):
        if r in border:
            continue
        m = reg == r
        holes.append(dict(area_a2=round(float(m.sum() * h * h / a / a), 2),
                          centroid=[round(float(U[m].mean()), 3), round(float(V[m].mean()), 3)]))
    holes.sort(key=lambda d: -d["area_a2"])
    row = dict(mesh=mesh, label=lab, plane=f"{sys.argv[3]}={value}", pitch=a, enclosed_holes=len(holes),
               holes=holes[:12])
    print(json.dumps(row), flush=True)
    with open(f"{prefix}.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")
    slab = np.abs(x[:, axis] - value) < 0.5 * a
    states.append((lab, f, x[slab][:, [ua, va]], holes))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(1, len(states), figsize=(5 * len(states), 5.5))
for j, (lab, f, p, holes) in enumerate(states):
    ax[j].scatter(p[:, 0], p[:, 1], s=1, c="0.6", lw=0)
    ax[j].contour(U, V, np.where(np.isfinite(f), f, 10 * a), levels=[0.0], colors="tab:red", linewidths=0.8)
    ax[j].set_aspect("equal")
    ax[j].set_title(f"{lab}: {len(holes)} enclosed, largest {holes[0]['area_a2'] if holes else 0} a^2", fontsize=9)
fig.suptitle(f"{mesh}, the display surface's zero set in the plane {sys.argv[3]} = {value}")
fig.tight_layout()
fig.savefig(f"{prefix}.png", dpi=80)
