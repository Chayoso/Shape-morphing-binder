"""zero_row_probe.py TERMS_DIR ARCHIVE_NPZ — D39: layer particles whose relaxation row has no weight.

layer_relax_data weights each layer particle's k nearest layer particles by a Gaussian of the distance times the
normal agreement (clamped at 0) and divides the row by its sum, clamped at 1e-12. A row whose weights are all zero
(every neighbour on the other side, or all of them farther than about ten layer spacings) stays zero. The kernels
then read its neighbour centroid as the origin: k_layer_resid gives d = n . (x - 0) and k_layer_project moves the
particle by -(1/T) d n every step, toward the plane through the world's origin.

Per window (the start state and the channel record of a term dump): the layer particles whose row sums to less than
one, how many of them are rows of exactly zero, their relaxation displacement as recorded (the `rest` channel, in
target spacings) against what the origin plane predicts (-(1 - (1 - 1/T)^(2T)) (n . x) n) and against the other layer
particles' displacement; and over the run how many distinct particles are hit and how far they are carried in all."""
import glob, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402,F401  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from physmorph.pipeline.window.layer import detached_groups, layer_relax_data, layer_spacing  # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402

dev = torch.device("cuda")
files = sorted(glob.glob(os.path.join(sys.argv[1], "terms_*.npz")))
tgt = torch.as_tensor(np.asarray(np.load(sys.argv[2], allow_pickle=True)["tgt"], np.float32), device=dev)
sp = float(knn_self_torch(tgt, 2)[0][:, 1].median())
T = 20
decay = 1 - (1 - 1 / T) ** (2 * T)
seen, carried = {}, 0.0
print(f"{len(files)} windows; target spacing {sp:.4f} wu; a zero row is carried {decay:.2f} of n . x toward the plane through the origin in one window")
print(" window | layer particles | rows summing to less than one | of them exactly zero | recorded relaxation displacement of those rows, sp (median, max) | "
      "the origin plane's prediction, sp (median, max) | cosine of the two | the other layer particles' relaxation displacement, sp (median, p99) | detached among them")
for k, fp in enumerate(files):
    f = np.load(fp)
    x = torch.as_tensor(f["c_x0"], device=dev)
    rest = torch.as_tensor(f["c_d_rest"], device=dev)
    mask, nrm, nbr, w = layer_relax_data(x, layer_spacing(x))
    on = mask > .5
    s = w.sum(1)
    bad = on & (s < .999)
    ids = torch.nonzero(bad).squeeze(1)
    good = on & ~bad
    line = f"   {k:4d} | {int(on.sum()):6d} | {len(ids):4d} | {int((on & (s == 0)).sum()):4d}"
    if len(ids):
        pred = -decay * (nrm[ids] * x[ids]).sum(1, keepdim=True) * nrm[ids]
        rec = rest[ids]
        cos = torch.nn.functional.cosine_similarity(rec, pred, dim=1)
        d, nb = knn_self_torch(x, 33)
        group, _ = detached_groups(d, nb, layer_spacing(x))
        line += (f" | {float(rec.norm(dim=1).median()) / sp:7.1f}, {float(rec.norm(dim=1).max()) / sp:7.1f} | {float(pred.norm(dim=1).median()) / sp:7.1f}, {float(pred.norm(dim=1).max()) / sp:7.1f} | "
                 f"{float(cos.median()):+.3f} | {float(rest[good].norm(dim=1).median()) / sp:.3f}, {float(rest[good].norm(dim=1).quantile(.99)) / sp:.2f} | {int((group[ids] > 0).sum())}")
        for i, r in zip(ids.tolist(), rec.norm(dim=1).tolist()):
            seen[i] = seen.get(i, 0) + 1
            carried += r
    print(line)
print(f"over the run: {sum(seen.values())} zero-weight rows on {len(seen)} distinct particles; carried {carried / sp:.0f} target spacings in all "
      f"({carried / max(sum(seen.values()), 1) / sp:.1f} a row)")
# where they are at the end, and how much of the end's far material they are
z = np.load(sys.argv[2], allow_pickle=True)
xe = torch.as_tensor(np.asarray(z["frames"][int(z["deliver_n"]) - 1], np.float32), device=dev)
tknn = gpu.KNN(tgt)
de = tknn.query(xe, 1)[0][:, 0].float() / sp
hit = torch.zeros(len(xe), dtype=torch.bool, device=dev)
hit[torch.as_tensor(sorted(seen), dtype=torch.long, device=dev)] = True
src = torch.as_tensor(np.asarray(z["frames"][0], np.float32), device=dev)
inside = tknn.query(xe[hit], 1)[0][:, 0].float() / sp
print(f"at the end of the run the {int(hit.sum())} particles are {float(de[hit].median()):.1f} sp from the target in the median (within 1 sp: {100 * float((de[hit] <= 1).float().mean()):.0f} %, "
      f"beyond 3 sp: {int((de[hit] > 3).sum())}, beyond 4.4 sp: {int((de[hit] > 4.4).sum())}); all particles beyond 3 sp at the end: {int((de > 3).sum())}, beyond 4.4 sp: {int((de > 4.4).sum())}")
