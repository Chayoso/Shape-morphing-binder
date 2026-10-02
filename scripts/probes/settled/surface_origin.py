"""surface_origin.py ARCHIVE_NPZ — who makes the final surface: for the outer layer of the last delivered frame, how
deep each particle was in the source (in lattice steps of the sample), and where the source's own outer layer ends.
The outer layer is the probes' rule (the centroid of 33 neighbours lies off the particle by more than 0.35 of the
33rd-neighbour distance). Also the share of all particles that are outer layer at the start and at the end."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402

dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
frames = z["frames"]
n_del = int(z["deliver_n"]) if "deliver_n" in z.files else len(frames)
x0 = torch.as_tensor(np.asarray(frames[0], np.float32), device=dev)
xe = torch.as_tensor(np.asarray(frames[n_del - 1], np.float32), device=dev)
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)


def outer(x):
    d, i = knn_self_torch(x, 34)
    return (x[i[:, 1:]].mean(1) - x).norm(dim=1) > 0.35 * d[:, -1]


step = float(knn_self_torch(x0, 7)[0][:, 6].median())          # about one lattice step of the jittered lattice
o0, oe, ot = outer(x0), outer(xe), outer(tgt)
N = len(x0)
depth0 = gpu.KNN(x0[o0]).query(x0, 1)[0][:, 0] / step           # depth below the source's outer layer, lattice steps
depthe = gpu.KNN(xe[oe]).query(xe, 1)[0][:, 0] / step
print(f"N {N}; lattice step {step:.4f} wu; outer layer: source {int(o0.sum())} ({100 * float(o0.float().mean()):.1f} %), "
      f"last frame {int(oe.sum())} ({100 * float(oe.float().mean()):.1f} %), target sample {int(ot.sum())} ({100 * float(ot.float().mean()):.1f} %)")
print(f"the final outer layer has {float(oe.sum() / o0.sum()):.2f} times the particles of the source's outer layer (the area grew by about that)")
d = depth0[oe]
print("where the final outer layer was in the source (depth below the source surface, lattice steps): "
      + ", ".join(f"< {k}: {100 * float((d < k).float().mean()):.0f} %" for k in (0.5, 1, 2, 4, 8)) + f"; median {float(d.median()):.1f}, p90 {float(torch.quantile(d, .9)):.1f}")
d = depthe[o0]
print("where the source's outer layer is at the end (depth below the final surface, lattice steps): "
      + ", ".join(f"< {k}: {100 * float((d < k).float().mean()):.0f} %" for k in (0.5, 1, 2, 4, 8)) + f"; median {float(d.median()):.1f}, p90 {float(torch.quantile(d, .9)):.1f}")
for k in (1, 2, 3):
    print(f"particles within {k} lattice step(s) of the surface: source {100 * float((depth0 < k).float().mean()):.1f} %, last frame {100 * float((depthe < k).float().mean()):.1f} %")
