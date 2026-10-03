"""layer_groups_probe.py TERMS_DIR ARCHIVE_NPZ [WINDOW ...] — the layer data of a window's start state, with a detached
group measured against itself (the definition until D26; `own` below) and as the layer defines it now (layer_relax_data:
not relaxed, the normal taken against the material around it; under D24's form, relaxed against that material, the
"against the surroundings" rows showed the residual that form removed).

Per window (the start state of the term dump's channel record): detached groups and their members; for the members
under each definition: the share on the layer, the cosine of the layer normal with the direction to the nearest target
point, the relaxation residual d - dbar along the normal (target spacings; what one window removes 87 % of), the share
of the relaxation weight on fellow members, rows without any weight; and the same for the body's layer particles
(which the change must leave as they are). With the seconds each definition takes."""
import glob, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from physmorph.pipeline.window.layer import detached_groups, layer_by_asymmetry, layer_relax_data, layer_spacing  # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402

dev = torch.device("cuda")
files = sorted(glob.glob(os.path.join(sys.argv[1], "terms_*.npz")))
tgt = torch.as_tensor(np.asarray(np.load(sys.argv[2], allow_pickle=True)["tgt"], np.float32), device=dev)
sp = float(knn_self_torch(tgt, 2)[0][:, 1].median())
tknn = gpu.KNN(tgt)
wins = [int(v) for v in sys.argv[3:]] or [4, 8, 12, 20, 30, len(files) - 1]


def own(x0, spacing, k=24, h_sp=2.0, thr_sp=0.5):
    """The definition until D24: every particle against its nearest neighbours, whoever they are."""
    N = x0.shape[0]
    mask, nrm = layer_by_asymmetry(x0, spacing, thr_sp=thr_sp)
    idx = torch.nonzero(mask).squeeze(1)
    nbr = torch.zeros(N, k, dtype=torch.long, device=x0.device)
    w = torch.zeros(N, k, device=x0.device)
    P, R = x0[idx].contiguous(), nrm[idx]
    d, nb = knn_self_torch(P, k + 1)
    d, nb = d[:, 1:], nb[:, 1:]
    ww = torch.exp(-(d / (h_sp * spacing)) ** 2) * torch.clamp((R[nb] * R[:, None, :]).sum(-1), min=0.0)
    ww = ww / torch.clamp(ww.sum(1, keepdim=True), min=1e-12)
    nbr[idx] = idx[nb]
    w[idx] = ww.float()
    return mask.float(), nrm.float(), nbr, w


def timed(f, *a):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = f(*a)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def row(name, x, m, data, group, n_t):
    mask, nrm, nbr, w = data
    on = m & (mask > 0.5)
    if not bool(m.any()):
        return f"      {name}: none"
    cen = (w[:, :, None] * x[nbr]).sum(1)
    res = (nrm * (x - cen)).sum(1) * mask
    res = (res - (w * res[nbr]).sum(1)) * mask
    fel = (w * (group[nbr] == group[:, None]).float()).sum(1)
    empty = on & (w.sum(1) <= 0)
    s = lambda v: f"{float(v[on].median()):+.2f} (p10 {float(v[on].quantile(.1)):+.2f}, p90 {float(v[on].quantile(.9)):+.2f})" if bool(on.any()) else "-"  # noqa: E731
    return (f"      {name}: {int(m.sum())} particles, on the layer {100 * float(on.float().sum() / m.sum()):3.0f} % | cos(normal, to target) {s((nrm * n_t).sum(1))} | "
            f"residual sp {s(res / sp)} | weight on fellows {100 * float(fel[on].mean()) if bool(on.any()) else 0:3.0f} % | rows without weight {int(empty.sum())}")


for k in wins:
    x = torch.as_tensor(np.load(files[k])["c_x0"], device=dev)
    spacing = layer_spacing(x)
    d_a, nb_a = knn_self_torch(x, 33)
    group, size = detached_groups(d_a, nb_a, spacing)
    q, it = tknn.query(x, 1)
    n_t = torch.nn.functional.normalize(tgt[it[:, 0]] - x, dim=1)
    dist = q[:, 0].float() / sp
    old, t_old = timed(own, x, spacing)
    new, t_new = timed(layer_relax_data, x, spacing)
    det = group > 0
    sizes = size[det]
    print(f"\n== window {k}: layer spacing {spacing:.4f} wu = {spacing / sp:.2f} sp; detached {int(det.sum())} particles in {int(group.max())} groups "
          f"(singles {int((sizes == 1).sum())}, in groups of 2-24 {int(((sizes > 1) & (sizes <= 24)).sum())}, 25-512 {int(((sizes > 24) & (sizes <= 512)).sum())}, larger {int((sizes > 512).sum())}); "
          f"seconds: against itself {t_old:.2f}, against the surroundings {t_new:.2f}")
    same = (old[0] == new[0]).all() and bool((old[1][~det] - new[1][~det]).abs().max() == 0) and bool((old[2][~det] == new[2][~det]).all())
    print(f"   the body's rows (mask, normal, neighbours) identical: {bool(same)}; the body's weights differ by at most {float((old[3][~det] - new[3][~det]).abs().max()):.1e}")
    for name, m in (("group members beyond 2 sp of the target", det & (size > 1) & (size <= 512) & (dist > 2)),
                    ("group members within 2 sp of the target", det & (size > 1) & (size <= 512) & (dist <= 2)),
                    ("detached singles beyond 2 sp", det & (size == 1) & (dist > 2))):
        print(f"   {name}")
        print(row("against itself         ", x, m, old, group, n_t))
        print(row("against the surroundings", x, m, new, group, n_t))
