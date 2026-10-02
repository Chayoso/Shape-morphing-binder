"""web_probe.py NPZ DX LDX — who the ear-region floaters are, where they come from, where they end, and how they move.

Sets (particle indices): web = floaters (> 3 target spacings from the target) above the ear base at raw 240; tuft = the
same at raw 480; fringe = particles > 1.5 sp from the target above the ear base at the last delivered frame; feet =
floaters in the bottom fifth at the last frame. For each: origin in the source (depth below the source surface, height),
destination (which ear, distance), a window-by-window series, and the direction of the early motion. Then: the gap
between the two ears in simulation cells, the banding along the upright ear, and the within-window path over net motion
of the ear material (the breathing)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.support import live_support              # noqa: E402

dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
DX, LDX = float(sys.argv[2]), float(sys.argv[3])
frames = z["frames"]
n_del = int(z["deliver_n"])
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
td, _ = knn_self_torch(tgt, 9)
sp, cov_r = float(td[:, 1].median()), float(td[:, 8].median())
tknn = gpu.KNN(tgt)
y0, y1 = float(tgt[:, 1].min()), float(tgt[:, 1].max())
y_base = y0 + 0.72 * (y1 - y0)
last = n_del - 1
print(f"N {frames.shape[1]}, delivered {n_del}; target nn spacing {sp:.4f} wu; sim cell {DX:.4f} wu = {DX / sp:.1f} sp; loss cell {LDX:.4f} wu = {LDX / sp:.1f} sp; "
      f"target height {y0:+.2f}..{y1:+.2f}, ear base y {y_base:+.2f}")


def X(raw):
    return torch.as_tensor(np.asarray(frames[raw], np.float32), device=dev)


def state(raw):
    x = X(raw)
    d, _ = knn_self_torch(x, 33)
    dt, it = tknn.query(x, 1)
    return x, d[:, 8] / cov_r, live_support(d, cov_r, sp), dt[:, 0] / sp, it[:, 0]


# the two ears of the target: 2-means on the points above the ear base, in the horizontal plane
ear_t = torch.nonzero(tgt[:, 1] > y_base + 0.25 * (y1 - y_base)).squeeze(1)
pts = tgt[ear_t][:, [0, 2]]
c = torch.stack([pts[pts[:, 0].argmin()], pts[pts[:, 0].argmax()]])
for _ in range(30):
    lab = torch.cdist(pts, c).argmin(1)
    c = torch.stack([pts[lab == k].mean(0) for k in (0, 1)])
tip = [float(tgt[ear_t][lab == k][:, 1].max()) for k in (0, 1)]
up = 0 if tip[0] >= tip[1] else 1                              # the upright ear reaches higher
region = torch.zeros(len(tgt), dtype=torch.long, device=dev)   # 0 body/head, 1 upright ear, 2 slanted ear
region[ear_t[lab == up]] = 1
region[ear_t[lab != up]] = 2
RN = ("body/head", "upright ear", "slanted ear")
print(f"target regions: upright ear {int((region == 1).sum())} points (tip y {max(tip):+.2f}), slanted ear {int((region == 2).sum())} (tip y {min(tip):+.2f})")

# the gap between the ears by height, in simulation cells
print("\n== the gap between the two ears (nearest points of the two ears in a height slab), and the ear thickness")
a_all, b_all = tgt[region == 1], tgt[region == 2]
for yy in np.linspace(y_base + 0.25 * (y1 - y_base), min(tip), 7):
    a, b = a_all[(a_all[:, 1] - yy).abs() < 2 * sp], b_all[(b_all[:, 1] - yy).abs() < 2 * sp]
    if len(a) == 0 or len(b) == 0:
        continue
    gap = float(torch.cdist(a, b).min())
    th = []
    for e in (a, b):                                            # slab cross-section: smallest principal extent
        q = e[:, [0, 2]] - e[:, [0, 2]].mean(0)
        ev = torch.linalg.eigvalsh(q.T @ q / len(q))
        th.append(float(4 * ev[0].sqrt()))
    print(f"   y {yy:+.2f}: gap {gap:.3f} wu = {gap / DX:.2f} sim cells = {gap / LDX:.2f} loss cells = {gap / sp:.1f} sp | thickness upright {th[0]:.3f} wu ({th[0] / DX:.2f} cells), slanted {th[1]:.3f} wu ({th[1] / DX:.2f} cells)")

x0, r0, s0, dt0, it0 = state(0)
xe, re, se, dte, ite = state(last)
# the source's surface: distance of each particle to the source's outer layer
dk, ik = knn_self_torch(x0, 34)
outer_s = (x0[ik[:, 1:]].mean(1) - x0).norm(dim=1) > 0.35 * dk[:, -1]
depth0 = gpu.KNN(x0[outer_s]).query(x0, 1)[0][:, 0] / sp
c0 = x0.mean(0)
print(f"\nsource: centre {[round(float(v), 2) for v in c0]}, radius {float((x0 - c0).norm(dim=1).max()):.2f} wu, top y {float(x0[:, 1].max()):+.2f}; outer layer {int(outer_s.sum())} particles")

x240, r240, s240, dt240, it240 = state(240)
x480, r480, s480, dt480, it480 = state(480)
sets = {
    "web (raw 240)": torch.nonzero((dt240 > 3) & (s240 > 0) & (x240[:, 1] > y_base)).squeeze(1),
    "tuft (raw 480)": torch.nonzero((dt480 > 3) & (s480 > 0) & (x480[:, 1] > y_base)).squeeze(1),
    "ear fringe (end, > 1.5 sp)": torch.nonzero((dte > 1.5) & (xe[:, 1] > y_base)).squeeze(1),
    "feet floaters (end)": torch.nonzero((dte > 3) & (xe[:, 1] < y0 + 0.2 * (y1 - y0))).squeeze(1),
}
wins = list(range(0, 21)) + list(range(24, n_del // 40, 6))
series = {}
for w in wins:
    raw = min(40 * w, last)
    series[w] = state(raw)


def q(v, p):
    return float(torch.quantile(v.float(), p)) if len(v) else float("nan")


for name, ids in sets.items():
    print(f"\n== {name}: {len(ids)} particles")
    if len(ids) == 0:
        continue
    print(f"   origin (raw 0): depth below the source surface median {q(depth0[ids], .5):.1f} sp (p10 {q(depth0[ids], .1):.1f}, p90 {q(depth0[ids], .9):.1f}); "
          f"within 2 sp of the source surface {100 * float((depth0[ids] < 2).float().mean()):.0f} %; height y {q(x0[ids, 1], .1):+.2f}..{q(x0[ids, 1], .9):+.2f}; "
          f"distance to the target {q(dt0[ids], .5):.1f} sp; share of the source's own outer 2 sp among all particles {100 * float((depth0 < 2).float().mean()):.0f} %")
    reg = region[ite[ids]]
    print("   destination (end): nearest target point in " + ", ".join(f"{RN[k]} {100 * float((reg == k).float().mean()):.0f} %" for k in range(3))
          + f"; distance to the target median {q(dte[ids], .5):.2f} sp, p90 {q(dte[ids], .9):.2f}; still > 1.5 sp {100 * float((dte[ids] > 1.5).float().mean()):.0f} %, > 3 sp {100 * float((dte[ids] > 3).float().mean()):.0f} %")
    print("   window | raw | dist to target sp (median p90) | height y median | move in the window sp | toward the target (cos) | upward (cos) | 8NN/coverage median | partial support % | region of nearest target")
    prev = None
    for w in wins:
        x, r, s, dt, it = series[w]
        mv = cs = cu = float("nan")
        if prev is not None and w - prev[0] == 1:
            d = x[ids] - prev[1][ids]
            to_t = tgt[prev[2][ids]] - prev[1][ids]
            mv = q(d.norm(dim=1) / sp, .5)
            cs = float(torch.nn.functional.cosine_similarity(d, to_t, dim=1).median())
            cu = float((d[:, 1] / d.norm(dim=1).clamp_min(1e-12)).median())
        rg = region[it[ids]]
        print(f"   {w:4d} | {min(40 * w, last):5d} | {q(dt[ids], .5):6.1f} {q(dt[ids], .9):6.1f} | {q(x[ids, 1], .5):+.2f} | {mv:6.2f} | {cs:+.2f} | {cu:+.2f} | {q(r[ids], .5):.2f} | "
              f"{100 * float(((s[ids] > 0) & (s[ids] < 1)).float().mean()):5.1f} | " + "/".join(f"{100 * float((rg == k).float().mean()):.0f}" for k in range(3)))
        prev = (w, x, it)

# overlaps
web, tuft, fringe = (set(sets[k].tolist()) for k in ("web (raw 240)", "tuft (raw 480)", "ear fringe (end, > 1.5 sp)"))
print(f"\noverlap: tuft in web {len(tuft & web)} of {len(tuft)}; end fringe in web {len(fringe & web)} of {len(fringe)}; end fringe in tuft {len(fringe & tuft)} of {len(fringe)}")

# banding along the upright ear: particle count along the ear axis, autocorrelation of the detrended profile
print("\n== banding along the upright ear (count of surface-side particles per 0.1 sp along the ear axis; first autocorrelation peak)")
ea = tgt[region == 1]
mu = ea.mean(0)
axis = torch.linalg.eigh((ea - mu).T @ (ea - mu))[1][:, -1]


def band(pts, label):
    s = ((pts - mu) @ axis) / sp
    lo, hi = float(s.min()), float(s.max())
    h = torch.histc(s, bins=int((hi - lo) / 0.1), min=lo, max=hi)
    k = 61
    sm = torch.nn.functional.avg_pool1d(h[None, None], k, 1, k // 2, count_include_pad=False)[0, 0]
    r = (h - sm)[k:-k]
    r = r - r.mean()
    ac = torch.stack([(r[:-l] * r[l:]).mean() for l in range(1, 150)]) / (r * r).mean()
    pk = [(l + 1, float(ac[l])) for l in range(1, 148) if ac[l] > ac[l - 1] and ac[l] > ac[l + 1] and ac[l] > 0.1]
    rel = float(r.std() / sm[k:-k].mean())
    top = sorted(pk, key=lambda t: -t[1])[:3]
    print(f"   {label}: {len(pts)} points; relative ripple {rel:.3f}; strongest autocorrelation peaks " + (", ".join(f"{l * 0.1:.1f} sp = {l * 0.1 * sp:.3f} wu = {l * 0.1 * sp / DX:.2f} sim cells = {l * 0.1 * sp / LDX:.2f} loss cells (r {v:.2f})" for l, v in top) or "none above 0.1"))


band(ea, "target (reference)")
for raw in (120, 200, 300, 480, 1032, last):
    x, r, s, dt, it = state(raw)
    m = (region[it] == 1) & (dt > 0.3)
    band(x[m], f"raw {raw:5d}, particles off the target cloud by > 0.3 sp")
    m = (region[it] == 1)
    band(x[m], f"raw {raw:5d}, all particles nearest the upright ear  ")

# the breathing: within each window, the path over the net motion of the ear material
print("\n== breathing of the ear material (particles whose end position is nearest an ear): per window, median net motion, median path, and the largest excursion from the start-to-end chord (sp; 1 sp = about 12 px at 4K)")
ear_ids = torch.nonzero(region[ite] > 0).squeeze(1).cpu().numpy()
prev_net = None
for w in range(0, n_del // 40):
    if not (w < 20 or w % 6 == 0):
        continue
    blk = torch.as_tensor(np.asarray(frames[40 * w: 40 * w + 41][:, ear_ids], np.float32), device=dev)
    net = blk[-1] - blk[0]
    path = (blk[1:] - blk[:-1]).norm(dim=2).sum(0)
    tt = torch.linspace(0, 1, len(blk), device=dev)[:, None, None]
    exc = (blk - (blk[0] + tt * net)).norm(dim=2).max(0).values
    cosr = float((net * prev_net).sum() / (net.norm() * prev_net.norm()).clamp_min(1e-30)) if prev_net is not None and prev_net.shape == net.shape else float("nan")
    # driven half against released half
    a, b = blk[20] - blk[0], blk[-1] - blk[20]
    chalf = float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))
    print(f"   window {w:3d}: net {q(net.norm(dim=1) / sp, .5):7.3f}  path {q(path / sp, .5):7.3f}  path/net {q(path / net.norm(dim=1).clamp_min(1e-9), .5):6.2f}  excursion median {q(exc / sp, .5):6.3f} p99 {q(exc / sp, .99):6.3f}"
          f"  released vs driven half cos {chalf:+.2f}  vs previous window cos {cosr:+.2f}")
    prev_net = net if (w < 19 or w % 6 == 0) else None
