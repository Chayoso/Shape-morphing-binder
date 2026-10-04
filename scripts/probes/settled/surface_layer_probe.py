"""surface_layer_probe.py KEYFRAMES_NPZ OUT_DIR x0,y0,x1,y1 [M [RBAR [RFAC [STATE ...]]]] — D59, stage 1a: a displayed
exterior (surface discs without mass) built from the particles of one state, beside the base display of the same
particles. Display only.

The surface: the zero set of f(q) = |q - xbar(q)| - RBAR * a (Zhu and Bridson 2005), xbar the mean of the particles
within R = RFAC * a of q weighted by (1 - (d / R)^2)^3, a = (V / N)^(1/3) the volume sample's pitch (0.708 of the
target's median 8th-neighbour distance).
The discs: the nodes of a grid of half a pitch that lie within a cell of the zero set (they cover it wherever it is),
copied with a jitter of one cell, projected onto f = 0 by Newton steps along the field's gradient, thinned to a
Poisson-disk set of about M points (no two nearer than r; parallel random-priority selection), drawn as discs of
tangential sigma r (the set's covering radius), opacity 0.92. Two drawings: with the field's gradient as
the normal, and with the base display's treatment of its normals (the mean over the neighbours within the reach of its
32 nearest particles, twice).

STATE: `target` or a raw frame index kept in the key-frame file (default: target and every kept frame but the first).
Per state: OUT_DIR/STATE.jpg (the crop: base display | exterior, field normals | exterior, display normals; the discs
of connected sets other than the largest are tinted red) and STATE_whole.jpg, the layer's points (STATE_layer.npz), and
a line of numbers: the layer's count, spacing, roughness and connected sets, the share of the zero set without a disc
and the share that lets more than a fifth through (the discs as one layer, at sigma r and at 0.65 of the neighbour
distance), solid and soft pixels of the drawings (whole and crop), the intersection over union of the solid regions."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from PIL import Image, ImageDraw                               # noqa: E402
from scipy.sparse import coo_matrix                            # noqa: E402
from scipy.sparse.csgraph import connected_components          # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.studio import DensityNormals, StudioRaster  # noqa: E402
from physmorph.render.support import live_support, normal_filter_size  # noqa: E402

dev = torch.device("cuda")
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
box = [int(v) for v in sys.argv[3].split(",")]
M = int(sys.argv[4]) if len(sys.argv) > 4 else 300000
RBAR = float(sys.argv[5]) if len(sys.argv) > 5 else .8
RFAC = float(sys.argv[6]) if len(sys.argv) > 6 else 3.          # the field's kernel radius R, in pitches
z = np.load(sys.argv[1], allow_pickle=True)
raws = [int(v) for v in z["raws"]]
states = sys.argv[7:] or ["target"] + [str(r) for r in raws[1:]]
W, H = 3840, 2160
target = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
center = target.mean(0)
radius = float((target - center).norm(dim=1).max())
td = knn_self_torch(target, 9)[0]
sp, cov_r = float(td[:, 1].median()), float(td[:, 8].median())
a = .708 * cov_r                                               # the volume sample's pitch
R, rbar = RFAC * a, RBAR * a
K = int(min(256, 16 + 4.2 * RFAC ** 3))                        # neighbours asked for: the particles within R, with room
density_normals = DensityNormals(center, radius, sp)
studio = StudioRaster(center, radius, W, H, 35., 18.)
kernel = normal_filter_size(H, False)
gen = torch.Generator(device=dev).manual_seed(0)
cube = lambda lo, hi: torch.stack(torch.meshgrid(*[torch.arange(lo, hi, device=dev)] * 3, indexing="ij"), -1).reshape(-1, 3)  # noqa: E731


def field(q, x, tree):
    """f(q), its gradient (autograd through the weights; the neighbours of q are fixed), and the summed weight."""
    f, g, s = [], [], []
    for qc in q.split(max(50000, int(25600000 / K))):
        d, idx = tree.query(qc, K)
        qc = qc.detach().requires_grad_(True)
        with torch.enable_grad():
            w = (1. - ((qc[:, None, :] - x[idx]).square().sum(-1) / (R * R))).clamp_min(0.) ** 3
            w = w * (d.float() < R)
            sw = w.sum(1)
            xbar = (w[..., None] * x[idx]).sum(1) / sw.clamp_min(1e-12)[:, None]
            fc = (qc - xbar).norm(dim=1) - rbar
            gc, = torch.autograd.grad(fc.sum(), qc)
        f.append(fc.detach()); g.append(gc); s.append(sw.detach())
    return torch.cat(f), torch.cat(g), torch.cat(s)


def zero_set_nodes(x, tree, h):
    """The nodes of a grid of pitch h = a / 2, within two pitches of a particle's cell, that lie within h of the zero set."""
    lo = x.min(0).values - 3 * a
    cells = torch.unique(((x - lo) / a).long(), dim=0)
    key = lambda c: (c[..., 0] * 4096 + c[..., 1]) * 4096 + c[..., 2]  # noqa: E731
    keys = torch.unique(key(cells[:, None, :] + cube(-2, 3)[None]).reshape(-1))
    cells = torch.stack((keys // (4096 * 4096), (keys // 4096) % 4096, keys % 4096), -1)
    nodes = lo + h * (2 * cells[:, None, :] + cube(0, 2)[None]).reshape(-1, 3).float()
    f, _, s = field(nodes, x, tree)
    return nodes[(f.abs() < h) & (s > 1e-6)]


def project(q, x, tree, steps=6):
    """Newton steps onto f = 0, a step no longer than half a pitch; the points that end on the surface."""
    for _ in range(steps):
        f, g, s = field(q, x, tree)
        step = f[:, None] * g / g.square().sum(1, keepdim=True).clamp_min(1e-12)
        n = step.norm(dim=1, keepdim=True)
        q = q - step * (.5 * a / n.clamp_min(1e-12)).clamp(max=1.)
    f, g, s = field(q, x, tree)
    ok = (f.abs() < .02 * a) & (s > 1e-6) & torch.isfinite(q).all(1)
    return q[ok], nnf.normalize(g[ok], dim=1)


def poisson_disk(p, r, k=64):
    """A maximal set of the pool p with no two points nearer than r (random priorities, accepted in parallel rounds)."""
    cell = ((p - p.min(0).values) / (.25 * r)).long()                      # at most one pool point per quarter-r cell
    key = (cell[:, 0] * 2000003 + cell[:, 1]) * 2000003 + cell[:, 2]
    order = torch.randperm(len(p), device=dev, generator=gen)
    _, first = torch.unique(key[order], return_inverse=True)
    keep = torch.zeros(int(first.max()) + 1, dtype=torch.long, device=dev).scatter_(0, first, order)
    p = p[keep]
    d, nb = gpu.KNN(p).query(p, k)
    near = (d.float() < r)
    near[:, 0] = False                                                     # itself
    prio = torch.randperm(len(p), device=dev, generator=gen)               # no two equal: a tie would never be decided
    undecided = torch.ones(len(p), dtype=torch.bool, device=dev)
    accepted = torch.zeros_like(undecided)
    while bool(undecided.any()):
        rival = torch.where(near & undecided[nb], prio[nb], prio.new_full((), -1)).max(1).values
        win = undecided & (prio > rival)
        accepted |= win
        undecided &= ~win & ~(near & win[nb]).any(1)
    return keep[accepted], p[accepted]


def base_drawing(x):
    """The base display's primitives (render_splat_photoreal.py), its picture and coverage, and the reach of the
    neighbourhood its normals are averaged over at the surface."""
    d, nb = knn_self_torch(x, 33)
    support = live_support(d, cov_r, sp)
    normals, magnitude = density_normals(x)
    strong = magnitude >= torch.quantile(magnitude[::max(1, len(x) // 100000)], .6)
    nearest = nb[:, 1:33]
    sn = strong[nearest]
    chosen = nearest[torch.arange(len(x), device=dev), sn.float().argmax(1)]
    normals = torch.where((~strong & sn.any(1))[:, None], normals[chosen], normals)
    for _ in range(2):
        normals = nnf.normalize(normals[nb].mean(1), dim=1, eps=1e-9)
    sigma = sp * (d[:, 8] / cov_r).clamp(1., 4.)
    surface = (x - x[nb[:, 1:]].mean(1)).norm(dim=1) >= .5 * cov_r
    return discs(x, normals, sigma, .92 * support), float(d[surface, 32].median())


def discs(x, normals, sigma, opacity, mark=None):
    """The picture and coverage of the discs; with a mark, the marked discs' visible share of each pixel tinted red."""
    reference = torch.where(normals[:, :1].abs() < .9, x.new_tensor((1., 0., 0.)), x.new_tensor((0., 1., 0.))).expand_as(x)
    tangent = nnf.normalize(torch.linalg.cross(normals, reference), dim=1, eps=1e-9)
    rotation = torch.stack((tangent, torch.linalg.cross(normals, tangent), normals), dim=2)
    variance = torch.stack((sigma ** 2, sigma ** 2, (sigma / 4) ** 2), dim=1)
    covariance = (rotation * variance[:, None]) @ rotation.transpose(1, 2)
    image, coverage, _ = studio(x, normals, covariance, opacity, normal_kernel=kernel, return_buffers=True)
    if mark is None:
        return image, coverage
    seen = studio.raster_colors(x, normals, covariance, opacity, mark[:, None].float().expand(-1, 3))[..., :1]
    return image * (1. - .6 * seen) + .6 * seen * x.new_tensor((1., .1, .1)), coverage, seen[..., 0]


def layer(x):
    """About M surface discs on the zero set of the field of the particles x."""
    tree = gpu.KNN(x)
    h = .5 * a
    seeds = zero_set_nodes(x, tree, h)
    copies = max(2, int(12 * M / max(len(seeds), 1)))
    q = seeds.repeat_interleave(copies, 0)
    q = q + h * (torch.rand(q.shape, device=dev, generator=gen) - .5)
    pool, normals = project(q, x, tree)
    r = .27 * a
    for _ in range(4):
        keep, pts = poisson_disk(pool, r)
        if abs(len(pts) - M) <= .03 * M:
            break
        r *= (len(pts) / M) ** .5
    check = project(seeds, x, tree)[0]                                      # the zero set, apart from the pool
    d = gpu.KNN(pts).query(check, 16)[0].float()
    through = lambda sigma: float(((1. - .92 * torch.exp(-.5 * (d / sigma) ** 2)).prod(1) > .2).float().mean())  # noqa: E731
    spacing = float(knn_self_torch(pts, 2)[0][:, 1].median())
    built = dict(seeds=len(seeds), pool=len(pool), uncovered=float((d[:, 0] > 1.5 * r).float().mean()),
                 through_sigma_065_spacing=through(.65 * spacing), through_sigma_r=through(r))
    return pts, normals[keep], r, spacing, built


def display_normals(pts, normals, reach, spacing):
    """The base display's treatment of its normals on the discs: twice the mean over the neighbours within its reach."""
    k = int(min(256, 3.63 * (reach / spacing) ** 2)) + 1                    # the discs within the reach (hexagonal count)
    d, nb = gpu.KNN(pts).query(pts, k)
    w = (d.float() <= reach)[..., None]
    for _ in range(2):
        normals = nnf.normalize((normals[nb] * w).sum(1), dim=1, eps=1e-9)
    return normals


def roughness(pts, normals):
    """Each disc against its 48 nearest: the angle of its normal to their mean (degrees), its height above their mean plane."""
    d, nb = knn_self_torch(pts, 49)
    mean = nnf.normalize(normals[nb[:, 1:]].mean(1), dim=1, eps=1e-9)
    angle = torch.rad2deg(torch.acos((normals * mean).sum(1).clamp(-1., 1.)))
    height = ((pts - pts[nb[:, 1:]].mean(1)) * mean).sum(1)
    return dict(reach_over_a=float(d[:, -1].median()) / a, angle_rms=float(angle.square().mean().sqrt()), angle_99=float(torch.quantile(angle[::7], .99)),
                opposed=int((angle > 90.).sum()), height_rms_over_a=float(height.square().mean().sqrt()) / a)


def connected_sets(pts, r):
    """The layer's connected sets (discs linked within 2.2 r): their number, and the discs outside the largest."""
    d, nb = knn_self_torch(pts, 9)
    link = d[:, 1:] < 2.2 * r
    i = torch.arange(len(pts), device=dev)[:, None].expand_as(link)[link].cpu().numpy()
    n, label = connected_components(coo_matrix((np.ones(len(i), bool), (i, nb[:, 1:][link].cpu().numpy())), shape=(len(pts),) * 2), directed=False)
    return n, torch.as_tensor(label != np.bincount(label).argmax(), device=dev)


def stats(coverage):
    c = coverage[box[1]:box[3], box[0]:box[2]]
    return dict(solid=int((coverage >= .5).sum()), soft=int(((coverage >= .02) & (coverage < .5)).sum()),
                crop_solid=int((c >= .5).sum()), crop_soft=int(((c >= .02) & (c < .5)).sum()))


def sheet(pictures, labels, path, crop):
    tiles = []
    for image, label in zip(pictures, labels):
        im = image[box[1]:box[3], box[0]:box[2]] if crop else image[:, 900:3000]
        im = Image.fromarray((im * 255 + .5).byte().cpu().numpy())
        if not crop:
            im = im.resize((im.width // 2, im.height // 2))
        ImageDraw.Draw(im).text((8, 6), label, fill=(255, 255, 0))
        tiles.append(im)
    w, h = tiles[0].size
    s = Image.new("RGB", (len(tiles) * w + 4 * (len(tiles) - 1), h), (20, 20, 20))
    for k, t in enumerate(tiles):
        s.paste(t, (k * (w + 4), 0))
    s.save(path, quality=90)


print(f"N {len(target)}; target spacing {sp:.4f} wu, pitch a {a:.4f} wu = {a / sp:.2f} spacings; R {R / a:.1f} a, rbar {RBAR:.2f} a; M {M}")
with torch.no_grad():                                          # the field's gradient re-enables autograd where it needs it
    for name in states:
        x = target if name == "target" else torch.as_tensor(np.asarray(z["frames"][raws.index(int(name))], np.float32), device=dev)
        (base, base_cover), reach = base_drawing(x)
        pts, normals, r, spacing, built = layer(x)
        sigma = torch.full((len(pts),), r, device=dev)          # the set's covering radius: no surface point is farther from a disc
        opacity = torch.full((len(pts),), .92, device=dev)
        shown = display_normals(pts, normals, reach, spacing)
        n_sets, apart = connected_sets(pts, r)
        ext, cover, seen = discs(pts, normals, sigma, opacity, apart)
        ext_shown = discs(pts, shown, sigma, opacity, apart)[0]
        apart_pixels = dict(whole=int((seen >= .5).sum()), crop=int((seen[box[1]:box[3], box[0]:box[2]] >= .5).sum()))
        sb, se = stats(base_cover), stats(cover)
        both = ((base_cover >= .5) & (cover >= .5)).sum()
        either = ((base_cover >= .5) | (cover >= .5)).sum()
        label = "target sample" if name == "target" else f"raw {name} (window {int(name) / 40:.0f})"
        labels = (f"{label}: base display, {len(x)} particles", f"{label}: exterior, {len(pts)} discs, the field's normals", "exterior, the base display's normal treatment")
        sheet((base, ext, ext_shown), labels, out / f"{name}.jpg", True)
        sheet((base, ext, ext_shown), labels, out / f"{name}_whole.jpg", False)
        np.savez(out / f"{name}_layer.npz", points=pts.cpu().numpy(), normals=normals.cpu().numpy(), shown=shown.cpu().numpy(), apart=apart.cpu().numpy(),
                 r=r, sigma=float(sigma[0]))
        row = dict(state=name, **built, discs=len(pts), r_over_a=r / a, spacing_median_over_a=spacing / a,
                   sigma_over_base=r / sp, normal_reach_over_a=reach / a,
                   field_normals=roughness(pts, normals), display_normals=roughness(pts, shown),
                   sets=n_sets, apart=int(apart.sum()), apart_seen_pixels=apart_pixels,
                   base=sb, exterior=se, iou_solid=float(both) / float(either))
        print(json.dumps(row), flush=True)
