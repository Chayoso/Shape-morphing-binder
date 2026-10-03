"""thin_fill_probe.py ARCHIVE_NPZ RUN_LOG [RAW ...] — D41: how much material the thin part of the target holds.

The thin share counts a thin target point as covered when one particle is within 1.5 target spacings: a horn drawn by
a few beads counts as much as a solid one (D38). The body has as many particles as the target sample has points and
every particle stands for the same mass, so a part of the target that holds M sample points is full when M particles
sit in it.

Per frame (by default the start of every fifth window and the end of the run), by the target's local feature thickness
(thin.local_thickness: below 2 MPM cells, 2 to 4, 4 and more):
  fill: the particles whose nearest target point is of the class and within 2 target spacings, over the class's points;
  local fill at each of the class's points: particles within 2.5 target spacings over the other target points within
    2.5 target spacings (1 for an independent sample of the target): its median, the share of the points below 0.5
    (a sparse cover) and at 0 (no particle);
  for reference the thin share's own measure: the share of the points with no particle within 1.5 target spacings."""
import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.thin import local_thickness                     # noqa: E402

dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
dx = float(re.search(r"\| dx=([0-9.]+) dt=", open(sys.argv[2]).read()).group(1))
frames, n_del = z["frames"], int(z["deliver_n"])
raws = [int(v) for v in sys.argv[3:]] or sorted(set(list(range(200, n_del - 1, 200)) + [n_del - 1]))
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
sp = float(knn_self_torch(tgt, 2)[0][:, 1].median())
K, R = 128, 2.5 * sp
with torch.inference_mode():
    h = (local_thickness(tgt, sp) / dx).float()
    tknn = gpu.KNN(tgt)
    own = torch.cat([(tknn.query(q, K)[0] <= R).sum(1) - 1 for q in tgt.split(50000)]).clamp_min(1).float()
classes = (("below 2 cells", h < 2), ("2 to 4 cells", (h >= 2) & (h < 4)), ("4 cells and more", h >= 4))
print(f"N {len(tgt)}; target spacing {sp:.4f} wu; MPM cell {dx:.4f} wu = {dx / sp:.1f} spacings; delivered frames {n_del}")
print("class: target points, their other target points within 2.5 sp (median) | " + "; ".join(
    f"{name}: {int(m.sum())}, {float(own[m].median()):.0f}" for name, m in classes))
print("   raw (window) | class | fill | local fill median | points with local fill below 0.5 % | with no particle within 2.5 sp % | with no particle within 1.5 sp %")
for raw in raws:
    with torch.inference_mode():
        x = torch.as_tensor(np.asarray(frames[raw], np.float32), device=dev)
        d, nearest = tknn.query(x, 1)
        arrived = d[:, 0] <= 2 * sp
        held = torch.bincount(nearest[:, 0][arrived], minlength=len(tgt))
        xknn = gpu.KNN(x)
        dq = torch.cat([xknn.query(q, K)[0] for q in tgt.split(50000)])
        local = (dq <= R).sum(1).float() / own
        bare = dq[:, 0] > 1.5 * sp
    for name, m in classes:
        print(f"   {raw:5d} ({raw / 40:5.1f}) | {name:16s} | {float(held[m].sum()) / float(m.sum()):.3f} | {float(local[m].median()):.2f} | "
              f"{100 * float((local[m] < .5).float().mean()):5.1f} | {100 * float((local[m] == 0).float().mean()):5.2f} | {100 * float(bare[m].float().mean()):5.2f}")
