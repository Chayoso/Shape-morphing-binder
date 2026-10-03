"""detached_probe.py ARCHIVE_NPZ RUN_JSON TERMS_DIR [LINK_A] — the floating Gaussians as DETACHED material: particles
not connected to the body. (TERMS_DIR without a term dump, e.g. `-`: parts A and B only.)

Connectivity: single linkage at LINK_A (default 2) particle spacings a (a = the median nearest-neighbour distance of
the source frame); the body is the largest component, a detached particle is a rendered one (support > 0) in any other
component.
A. Per window end: detached particles, components, the largest, and where they are against the two local terms'
   definitions (inside the berth / in the near band / beyond one loss cell; denser than the isolation gate's lower edge).
B. The end state's components: size, gap to the body, distance to the target, distance to the target on the opposite
   side (a component in mid-gap has the two about equal), the coherence of its particles' directions to their nearest
   target points (1 = one side, 0 = split between two sides), density, 4K pixel, world centre.
C. The largest end components traced through the windows (term dump): distance, gap to everything else, density, the
   direction coherence, each term's gradient summed over the component against the sum of its magnitudes (what a pull
   that the grid averages over a sub-cell blob keeps), the net pull in units of the all-particle rms, and what the
   outer layer does to it: share on the layer, share of its relaxation weights on fellow members, the relaxation
   residual along the normal, the cosine of the layer normal with the direction to the target."""
import glob, json, math, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from physmorph.pipeline.window.layer import layer_relax_data, layer_spacing  # noqa: E402
from physmorph.render.covariance_torch import world_to_view_torch  # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.support import live_support              # noqa: E402

W, H, AZ, EL = 3840, 2160, 35.0, 18.0
dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
run = json.load(open(sys.argv[2]))
link_a = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
cfg = run["arms"]["render_full_dt_iso_nn"]["config"]
frames = z["frames"]
n_del = int(z["deliver_n"])
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
td = knn_self_torch(tgt, 9)[0]
sp, cov = float(td[:, 1].median()), float(td[:, 8].median())
ldx = 2 * (1.25 * float(max(np.abs(np.asarray(z["src"])).max(), np.abs(np.asarray(z["tgt"])).max())) + 2 * float(run["provenance"]["mpm"]["dx"])) / int(cfg["loss_res"])
berth = float(cfg.get("nn_berth_k", 1.97)) * sp
tknn = gpu.KNN(tgt)
center = tgt.mean(0)
radius = float((tgt - center).norm(dim=1).max())
az, el = math.radians(AZ), math.radians(EL)
cam = center + 3.6 * radius * center.new_tensor((math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)))
view = world_to_view_torch(cam, center)
tan_y = math.tan(math.radians(30) / 2); tan_x = tan_y * W / H


def X(raw):
    return torch.as_tensor(np.asarray(frames[raw], np.float32), device=dev)


def pix(c):
    p = view[:3, :3] @ c + view[:3, 3]
    return float((p[0] / (p[2] * tan_x) + 1) / 2 * W), float((1 - p[1] / (p[2] * tan_y)) / 2 * H)


x0 = X(0)
a = float(knn_self_torch(x0, 2)[0][:, 1].median())
link = link_a * a


def components(x, k=33):
    """(label (N,), knn distances (N, k)): single linkage at `link` over each particle's k - 1 nearest neighbours."""
    d, nb = knn_self_torch(x, k)
    N = len(x)
    ok = d[:, 1:] < link
    i = torch.arange(N, device=dev)[:, None].expand(-1, k - 1)[ok]
    j = nb[:, 1:][ok]
    label = torch.arange(N, device=dev)
    while True:
        new = label.clone()
        new.scatter_reduce_(0, i, label[j], "amin")
        new.scatter_reduce_(0, j, label[i], "amin")
        new = new[new]
        if bool((new == label).all()):
            return label, d
        label = new


def detached(x):
    """(ids of detached rendered particles, their labels, knn distances, distance to the target in sp)."""
    label, d = components(x)
    u, cnt = label.unique(return_counts=True)
    body = u[cnt.argmax()]
    sup = live_support(d, cov, sp)
    ids = torch.nonzero((label != body) & (sup > 0)).squeeze(1)
    return ids, label[ids], d, tknn.query(x, 1)[0][:, 0].float(), int(((label != body) & (sup == 0)).sum())


print(f"N {len(x0)}; particle spacing a {a:.4f} wu, linkage {link_a:g} a = {link:.4f} wu = {link / sp:.1f} sp; target spacing sp {sp:.4f} wu; berth {berth / sp:.2f} sp; "
      f"loss cell {ldx:.4f} wu = {ldx / sp:.1f} sp; MPM cell {float(run['provenance']['mpm']['dx']):.3f} wu = {float(run['provenance']['mpm']['dx']) / sp:.1f} sp")
print("\nA. per window end: raw frame | detached rendered particles | components | largest | inside the berth / near band / beyond one loss cell | denser than the gate's lower edge (8NN/coverage < 1.2) | beyond a loss cell AND dense | detached unrendered")
for raw in range(40, n_del, 40):
    x = X(raw)
    ids, lab, d, dt, unr = detached(x)
    if len(ids) == 0:
        print(f"   {raw:5d} | 0")
        continue
    cnt = lab.unique(return_counts=True)[1]
    dd = dt[ids]
    dense = d[ids, 8] / cov < 1.2
    far = dd > ldx
    print(f"   {raw:5d} | {len(ids):6d} | {len(cnt):5d} | {int(cnt.max()):5d} | {int((dd <= berth).sum()):5d} / {int(((dd > berth) & ~far).sum()):5d} / {int(far.sum()):5d} | "
          f"{int(dense.sum()):5d} | {int((far & dense).sum()):5d} | {unr}")

xe = X(n_del - 1)
ids, lab, d, dt, _ = detached(xe)
is_det = torch.zeros(len(xe), dtype=torch.bool, device=dev); is_det[ids] = True
body_pts = xe[~is_det]
groups = sorted(((int((lab == l).sum()), int(l)) for l in lab.unique().tolist()), reverse=True)
print(f"\nB. the end state (raw {n_del - 1}): {len(ids)} detached rendered particles in {len(groups)} components; in components of 5 or more: {sum(g[0] for g in groups if g[0] >= 5)}")
print("   size | gap to the body (a) | distance to the target sp (median) = loss cells | to the target on the opposite side sp | direction coherence | 8NN/coverage | 4K pixel | world centre")
big = []


def describe(size, l):
    g = ids[lab == l]
    c = xe[g].mean(0)
    gap = float(torch.cdist(xe[g], body_pts).min()) / a
    q, it = tknn.query(xe[g], 1)
    n = torch.nn.functional.normalize(tgt[it[:, 0]] - xe[g], dim=1)
    n1 = torch.nn.functional.normalize(n.mean(0), dim=0)
    v = tgt - c
    opp = (v @ n1) < -0.5 * v.norm(dim=1)
    d2 = float(v[opp].norm(dim=1).min()) / sp if bool(opp.any()) else float("nan")
    u_, v_ = pix(c)
    print(f"   {size:4d} | {gap:5.1f} | {float(dt[g].median()) / sp:5.1f} = {float(dt[g].median()) / ldx:.2f} | {d2:5.1f} | {float(n.mean(0).norm()):.2f} | {float((d[g, 8] / cov).median()):.2f} | "
          f"({u_:5.0f}, {v_:5.0f}) | {[round(float(t), 2) for t in c]}")
    return g


for size, l in groups[:14]:
    big.append(describe(size, l))
out_groups = [(s, l) for s, l in groups if float(dt[ids[lab == l]].median()) > berth]
print(f"   the same for the components whose median distance is beyond the berth ({len(out_groups)} components, {sum(s for s, _ in out_groups)} particles):")
for size, l in out_groups[:10]:
    describe(size, l)

files = sorted(glob.glob(os.path.join(sys.argv[3], "terms_*.npz")))
if not files:                                                  # a run without a term dump: A and B only
    raise SystemExit(0)
T = ("ot", "surf", "near", "spray", "rend")
lk, lh = int(cfg.get("layer_k", 24)), float(cfg.get("layer_h_sp", 2.0))
rows = sorted({min(k, len(files) - 1) for k in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, len(files) - 1)})
out = {gi: [] for gi in range(min(4, len(big)))}
for k in rows:
    f = np.load(files[k])
    x = torch.as_tensor(f["x"], device=dev)
    g = {t: torch.as_tensor(f["g_" + t], device=dev) for t in T}
    rms_all = float(sum(g.values()).pow(2).sum(1).mean().sqrt())
    dd = knn_self_torch(x, 9)[0]
    mask, nrm, nbr, w = layer_relax_data(x, layer_spacing(x), k=lk, h_sp=lh)
    cen = (w[:, :, None] * x[nbr]).sum(1)
    res = (nrm * (x - cen)).sum(1) * mask
    res = (res - (w * res[nbr]).sum(1)) * mask
    for gi in out:
        m = big[gi]
        member = torch.zeros(len(x), dtype=torch.bool, device=dev); member[m] = True
        q, it = tknn.query(x[m], 1)
        n = torch.nn.functional.normalize(tgt[it[:, 0]] - x[m], dim=1)
        gap = float(torch.cdist(x[m], x[~member]).min(1).values.median()) / a
        on = mask[m] > 0.5
        coh = lambda t: float(g[t][m].sum(0).norm()) / max(float(g[t][m].norm(dim=1).sum()), 1e-30)
        net = lambda t: float(g[t][m].mean(0).norm()) / rms_all
        lay = (f"{100 * float(on.float().mean()):3.0f} | {100 * float((w[m][on] * member[nbr[m][on]].float()).sum(1).mean()) if bool(on.any()) else 0:3.0f} | "
               f"{float(res[m][on].mean()) / sp if bool(on.any()) else 0:+5.2f} | {float((nrm[m][on] * n[on]).sum(1).mean()) if bool(on.any()) else 0:+.2f}")
        out[gi].append(f"   {k:4d} | {float(q[:, 0].float().median()) / sp:5.1f} | {gap:5.1f} | {float((dd[m, 8] / cov).median()):.2f} | {float(n.mean(0).norm()):.2f} | "
                       + " ".join(f"{coh(t):.2f}/{net(t):5.2f}" for t in T) + " | " + lay)
for gi, lines in out.items():
    print(f"\nC. component {gi} ({len(big[gi])} particles)")
    print("   window | distance sp | gap to everything else (a) | 8NN/coverage | direction coherence | per term (ot surf near spray rend): coherence / net pull in all-particle rms | "
          "on the layer % | relaxation weight on fellow members % | relaxation residual along the normal sp | cos(layer normal, direction to the target)")
    print("\n".join(lines))
