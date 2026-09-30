"""floor_probe.py LABEL=ARCHIVE.npz ... — the support floor at a run's end state under the current definition (the
target's leave-one-out density at the particle's nearest target point, piecewise constant) and under the same-location
form A'': f(x) = 1/2 (t(x) - K(0)), with t(x) the target's kernel sum at the particle's own position over its 33 nearest
target points and K(0) = 1 (Kelsall & Diggle 1995; Monaghan 2005). Same kernel, h and k as physmorph/losses/support.py.

Per archive: (1) A'' at the target points equals the current floor (max abs difference); (2) the share of body particles
with f'' <= 0, where they are (distance to the target in spacings; outer; near the thin set) and how many of them sit
within one spacing of the target; (3) the ratio f'' / f_NN for particles within one spacing and near the thin set;
(4) the support penalty (ratio form) under both floors: mean and the share of paying particles; (5) how many particles
sit at a floor jump under the current definition: their nearest and second-nearest target points' floors differ by
more than 10 %, i.e. crossing the Voronoi boundary would change f by that much."""
import sys

sys.path.insert(0, __file__.rsplit("scripts", 1)[0])
import physmorph.gpu as gpu  # noqa: E402  (before torch: CuPy binds its CUDA compiler first)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch  # noqa: E402
from physmorph.thin import outer_mask  # noqa: E402


def kernel_sum(tree, pts, queries, h, k):
    d = tree.query(queries, k)[0]
    return torch.exp(-d * d / (2 * h * h)).sum(1)


for arg in sys.argv[1:]:
    label, fn = arg.split("=", 1)
    z = np.load(fn, allow_pickle=True)
    X = gpu.tensor(np.asarray(z["frames"][int(z["deliver_n"]) - 1], np.float64), torch.float64)
    Y = gpu.tensor(np.asarray(z["tgt"], np.float64), torch.float64)
    tree = gpu.KNN(Y)
    k = 32
    dY = tree.query(Y, k + 1)[0][:, 1:]
    r8 = gpu.median(dY[:, 7])
    h = .5 * r8
    rho = torch.exp(-dY * dY / (2 * h * h)).sum(1)               # leave-one-out, the current per-point floor / 0.5
    f_nn_pt = .5 * rho
    sp = gpu.median(tree.query(Y, 2)[0][:, 1])
    # (1) A'' at the target points
    t_y = kernel_sum(tree, Y, Y, h, k + 1)                        # self included among the 33
    f2_y = .5 * (t_y - 1.0)
    print(f"{label}: N {len(X)}, target spacing {sp:.4f}, r8 {r8:.4f}, h {h:.4f}")
    print(f"  (1) A'' at target points vs current floor: max |diff| {float((f2_y - f_nn_pt).abs().max()):.2e} "
          f"(floor median {float(f_nn_pt.median()):.3f}, p5 {float(torch.quantile(f_nn_pt, .05)):.3f})")
    # body particles
    dnn, inn = tree.query(X, 2)
    f_nn = f_nn_pt[inn[:, 0]]
    t_x = kernel_sum(tree, Y, X, h, k + 1)
    f2 = .5 * (t_x - 1.0)
    dist_sp = dnn[:, 0] / sp
    neg = f2 <= 0
    outer = outer_mask(X.float(), float(gpu.median(gpu.knn(X.float(), 2)[0][:, 1])))
    near = dist_sp <= 1.0
    print(f"  (2) f'' <= 0: {100 * float(neg.double().mean()):.2f} % of particles; of those, distance to the target "
          f"median {float(dist_sp[neg].median()) if bool(neg.any()) else 0:.2f} sp, p10 {float(torch.quantile(dist_sp[neg], .1)) if bool(neg.any()) else 0:.2f} sp; "
          f"outer share {float(outer[neg].double().mean()) if bool(neg.any()) else 0:.2f}; "
          f"within one spacing of the target: {int((neg & near).sum())} particles of {int(near.sum())} near ones")
    ratio = f2 / f_nn
    for name, m in (("within 1 sp", near), ("1-2 sp", (dist_sp > 1) & (dist_sp <= 2)), ("beyond 2 sp", dist_sp > 2)):
        if bool(m.any()):
            q = torch.quantile(ratio[m], torch.tensor([.1, .5, .9], dtype=ratio.dtype, device=ratio.device))
            print(f"  (3) f''/f_NN {name:12s}: n {int(m.sum()):6d}, p10 {float(q[0]):.3f} median {float(q[1]):.3f} p90 {float(q[2]):.3f}")
    # (4) the penalty under both floors
    _, idx = knn_self_torch(X.float(), k + 1)
    is_self = idx == torch.arange(len(X), device=X.device)[:, None]
    order = torch.argsort(is_self.to(torch.int8), dim=1, stable=True)
    nb = idx.gather(1, order)[:, :k]
    s = torch.exp(-(X[nb] - X[:, None]).square().sum(2) / (2 * h * h)).sum(1)
    for name, f in (("current", f_nn), ("A''", f2)):
        pen = torch.where(f > 0, torch.relu(1 - s / f.clamp_min(1e-300)), torch.zeros_like(f)).square()
        print(f"  (4) penalty ({name:7s}): mean {float(pen.mean()):.3e}, paying {100 * float((pen > 0).double().mean()):.2f} %, "
              f"max {float(pen.max()):.3f}")
    # (5) particles at a floor jump under the current definition
    jump = (f_nn_pt[inn[:, 1]] / f_nn - 1).abs() > .10
    print(f"  (5) current floor: {100 * float(jump.double().mean()):.1f} % of particles have a second-nearest target "
          f"point whose floor differs by > 10 % (a Voronoi crossing changes f by that much); median |ratio - 1| "
          f"{float((f_nn_pt[inn[:, 1]] / f_nn - 1).abs().median()):.3f}")
