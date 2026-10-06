"""d105_where.py TARGET_DISCS NAME=DISCS [NAME=DISCS ...] — where a displayed surface stands off the mesh, by the target's
local thickness: the discs kept by exterior_offset_probe.py (points, mesh normals, signed offset s in pitches, its Gaussian
means g1 / g25). The thickness at a target disc is the distance to the nearest target disc facing the other way (mesh
normals at more than 120 degrees) among its 256 nearest, in pitches (none found: thick). Every arm's disc takes the
thickness of its nearest target disc. Per arm and thickness bin: the share of the arm's discs, the mean offset, the
offset's spread about it, the fine part (s - g1), and the shares standing more than one pitch out and in."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402

dev = torch.device("cuda")
T = np.load(sys.argv[1])
a = float(T["a"])
tp = torch.as_tensor(T["points"], device=dev)
tm = torch.as_tensor(T["mesh_normals"], device=dev)
d, nb = gpu.KNN(tp).query(tp, 256)
d = torch.as_tensor(d, device=dev).float()
nb = torch.as_tensor(nb, device=dev).long()
opp = (tm[nb] * tm[:, None]).sum(-1) < -0.5
thick = torch.where(opp, d, torch.full_like(d, float("inf"))).min(1).values / a
bins = [(0, 3), (3, 6), (6, 12), (12, float("inf"))]
print(f"target {Path(sys.argv[1]).name}: {len(tp)} discs, pitch {a:.4f}; share by thickness "
      + ", ".join(f"{lo:g}-{hi:g}: {float(((thick >= lo) & (thick < hi)).float().mean()):.3f}" for lo, hi in bins))
knn_t = gpu.KNN(tp)
print(f"{'arm':10s} {'bin':8s} {'share':>6s} {'mean s':>7s} {'spread':>7s} {'fine':>6s} {'out>1':>6s} {'in<-1':>6s}")
for arg in sys.argv[2:]:
    name, path = arg.split("=", 1)
    Z = np.load(path)
    p = torch.as_tensor(Z["points"], device=dev)
    s = torch.as_tensor(Z["s"], device=dev)
    g1 = torch.as_tensor(Z["g1"], device=dev)
    _, j = knn_t.query(p, 1)
    th = thick[torch.as_tensor(j, device=dev).long()[:, 0]]
    rows = [("all", torch.ones_like(s, dtype=torch.bool))] + [(f"{lo:g}-{hi:g}", (th >= lo) & (th < hi)) for lo, hi in bins]
    for label, m in rows:
        if int(m.sum()) == 0:
            continue
        sm = s[m]
        print(f"{name:10s} {label:8s} {float(m.float().mean()):6.3f} {float(sm.mean()):+7.3f} {float(sm.std()):7.3f} "
              f"{float((sm - g1[m]).pow(2).mean().sqrt()):6.3f} {float((sm > 1).float().mean()):6.3f} {float((sm < -1).float().mean()):6.3f}",
              flush=True)
