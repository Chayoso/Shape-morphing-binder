"""surface_layer_probe.py FRAMES_NPZ OUT_DIR x0,y0,x1,y1 KIND [STATE ...] — D59: a displayed exterior
(surface discs without mass, physmorph/render/exterior.py) built from the particles of one state, beside the base
display of the same particles. Display only.

The surface: the zero set of Zhu and Bridson's field of the particles, kernel radius 3 pitches (the volume sample's
pitch a = (V / N)^(1/3), 0.708 of the target's median 8th-neighbour distance), offset 0.8 a.
The discs, KIND `poisson` (stage 1a): the nodes of a grid of half a pitch that lie within a cell of the zero set (they
cover it wherever it is), copied with a jitter of one cell, projected onto the zero set, thinned to a Poisson-disk set
of about M points (no two nearer than r; parallel random-priority selection), sigma r (the set's covering radius).
The discs, KIND `lattice` (stage 1b, every frame of a video): one disc for each cell of a lattice fixed in space that
the zero set crosses, the cell's centre projected onto it and kept where it stays in its cell; the lattice's pitch h
is set once so that the target sample's surface takes M discs; sigma h. Nothing is carried from frame to frame.
Two drawings of the discs (opacity 0.92): with the field's gradient as the normal, and with the base display's
treatment of its normals (the mean over the neighbours within the reach of its 32 nearest particles, at most 256
discs, twice). The discs of connected sets other than the largest are tinted red where they are seen.

STATE: `target` or a raw frame index kept in FRAMES_NPZ (default: target and every kept frame).
`poisson` writes OUT_DIR/STATE.jpg (the crop: base display | exterior, field normals | exterior, display normals),
STATE_whole.jpg and the layer's points (STATE_layer.npz); `lattice` writes OUT_DIR/crop/NNNN.jpg and whole/NNNN.jpg in
the order of the states (target.jpg for the target). Per state a line of numbers: the layer's count, spacing, roughness
and connected sets, the share of the zero set without a disc and the share that lets more than a fifth through (the
discs as one layer; `lattice`: on the target and every 40th state), solid and soft pixels of the drawings (whole and
crop), the intersection over union of the solid regions, and against the target sample as the exterior draws both
(front camera, its crop, a camera on the far side: the solid regions' intersection over union and the pictures' mean
absolute difference over their union); `lattice` also the pixels of the state before that changed their solid state
for that one frame only (blips), and the seconds the layer took."""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from PIL import Image, ImageDraw                               # noqa: E402
from scipy.sparse import coo_matrix                            # noqa: E402
from scipy.sparse.csgraph import connected_components          # noqa: E402
from physmorph.render.exterior import Lattice, ZhuBridson      # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.studio import DensityNormals, StudioRaster  # noqa: E402
from physmorph.render.support import live_support, normal_filter_size  # noqa: E402

dev = torch.device("cuda")
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
box = [int(v) for v in sys.argv[3].split(",")]
kind = sys.argv[4]
M = 300000                                                     # the discs' budget
z = np.load(sys.argv[1], allow_pickle=True)
raws, frames = [int(v) for v in z["raws"]], z["frames"]
states = sys.argv[5:] or ["target"] + [str(r) for r in raws]
W, H = 3840, 2160
target = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
center = target.mean(0)
radius = float((target - center).norm(dim=1).max())
td = knn_self_torch(target, 9)[0]
sp, cov_r = float(td[:, 1].median()), float(td[:, 8].median())
a = .708 * cov_r                                               # the volume sample's pitch
density_normals = DensityNormals(center, radius, sp)
studio = StudioRaster(center, radius, W, H, 35., 18.)
back = StudioRaster(center, radius, W, H, 215., 18.)           # the far side, for the measures against the target only
kernel = normal_filter_size(H, False)
gen = torch.Generator(device=dev).manual_seed(0)
lat = Lattice(center, 2.8 * radius)


def field_of(x):
    return ZhuBridson(x, a)


def poisson_disk(p, r, k=64):
    """A maximal set of the pool p with no two points nearer than r (random priorities, accepted in parallel rounds)."""
    cell = ((p - p.min(0).values) / (.25 * r)).long()                      # at most one pool point per quarter-r cell
    cell = (cell[:, 0] * 2000003 + cell[:, 1]) * 2000003 + cell[:, 2]
    order = torch.randperm(len(p), device=dev, generator=gen)
    _, first = torch.unique(cell[order], return_inverse=True)
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


def discs(x, normals, sigma, opacity, mark=None, cam=None):
    """The picture and coverage of the discs (the front camera unless another is given); with a mark, also the
    marked discs' visible share of each pixel."""
    cam = cam or studio
    reference = torch.where(normals[:, :1].abs() < .9, x.new_tensor((1., 0., 0.)), x.new_tensor((0., 1., 0.))).expand_as(x)
    tangent = nnf.normalize(torch.linalg.cross(normals, reference), dim=1, eps=1e-9)
    rotation = torch.stack((tangent, torch.linalg.cross(normals, tangent), normals), dim=2)
    variance = torch.stack((sigma ** 2, sigma ** 2, (sigma / 4) ** 2), dim=1)
    covariance = (rotation * variance[:, None]) @ rotation.transpose(1, 2)
    image, coverage, _ = cam(x, normals, covariance, opacity, normal_kernel=kernel, return_buffers=True)
    if mark is None:
        return image, coverage
    return image, coverage, cam.raster_colors(x, normals, covariance, opacity, mark[:, None].float().expand(-1, 3))[..., 0]


def cover(pts, sigmas, field):
    """Against the zero set found apart from the discs: the share of it farther than 1.5 of the first sigma from a
    disc, and the share that lets more than a fifth through, the discs taken as one layer, at each sigma."""
    check = field.project(lat.zero_set_nodes(field, .5 * a))[0]
    d = gpu.KNN(pts).query(check, 16)[0].float()
    through = [float(((1. - .92 * torch.exp(-.5 * (d / s) ** 2)).prod(1) > .2).float().mean()) for s in sigmas]
    return dict(uncovered=float((d[:, 0] > 1.5 * sigmas[0]).float().mean()), through=through)


def poisson(field):
    """About M surface discs on the zero set of the field, no two nearer than r."""
    h = .5 * a
    seeds = lat.zero_set_nodes(field, h)
    copies = max(2, int(12 * M / max(len(seeds), 1)))
    q = seeds.repeat_interleave(copies, 0)
    q = q + h * (torch.rand(q.shape, device=dev, generator=gen) - .5)
    pool, g = field.project(q)
    r = .27 * a
    for _ in range(4):
        keep, pts = poisson_disk(pool, r)
        if abs(len(pts) - M) <= .03 * M:
            break
        r *= (len(pts) / M) ** .5
    return pts, nnf.normalize(g[keep], dim=1), r, dict(seeds=len(seeds), pool=len(pool))


def lattice(field, h):
    """One disc for each lattice cell of pitch h that the zero set crosses."""
    pts, g, cells, nodes = lat.discs(field, h)
    return pts, nnf.normalize(g, dim=1), h, dict(nodes=nodes, crossed=cells)


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


def in_box(image):
    return image[box[1]:box[3], box[0]:box[2]]


def stats(coverage):
    c = in_box(coverage)
    return dict(solid=int((coverage >= .5).sum()), soft=int(((coverage >= .02) & (coverage < .5)).sum()),
                crop_solid=int((c >= .5).sum()), crop_soft=int(((c >= .02) & (c < .5)).sum()))


def sheet(pictures, labels, path, crop):
    tiles = []
    for image, label in zip(pictures, labels):
        im = in_box(image) if crop else image[:, 900:3000]
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


def against(image, cover_, ref):
    """Against the target sample's exterior drawing from the same camera: the intersection over union of the solid
    regions, and the mean absolute difference of the pictures over their union."""
    a, b = cover_ >= .5, ref[1] >= .5
    return float((a & b).sum()) / float((a | b).sum()), float((image - ref[0]).abs().mean(-1)[a | b].mean())


print(f"N {len(target)}; target spacing {sp:.4f} wu, pitch a {a:.4f} wu = {a / sp:.2f} spacings; M {M}; {kind}")
tint = lambda image, seen: image * (1. - .6 * seen[..., None]) + .6 * seen[..., None] * image.new_tensor((1., .1, .1))  # noqa: E731
with torch.no_grad():                                          # the field's gradient re-enables autograd where it needs it
    if kind == "lattice":
        h = .4 * a
        h *= (len(lattice(field_of(target), h)[0]) / M) ** .5            # the pitch at which the target sample's surface takes M discs
        print(f"lattice pitch {h / a:.3f} a")
        for d in ("crop", "whole"):
            (out / d).mkdir(exist_ok=True)
    solid, order, ref = [], 0, None                             # the solid regions of the last frames (base, exterior); the target's drawings
    for name in states:
        x = target if name == "target" else torch.as_tensor(np.asarray(frames[raws.index(int(name))], np.float32), device=dev)
        (base, base_cover), reach = base_drawing(x)
        t0 = time.time()
        field = field_of(x)
        pts, normals, r, built = poisson(field) if kind == "poisson" else lattice(field, h)
        built["seconds"] = time.time() - t0
        spacing = float(knn_self_torch(pts, 2)[0][:, 1].median())
        if kind == "poisson":
            built.update(cover(pts, (r, .65 * spacing), field))
        elif name == "target" or order % 40 == 0:
            built.update(cover(pts, (r,), field))
        sigma = torch.full((len(pts),), r, device=dev)          # the set's covering radius
        opacity = torch.full((len(pts),), .92, device=dev)
        shown = display_normals(pts, normals, reach, spacing)
        n_sets, apart = connected_sets(pts, r)
        ext, ext_cover, seen = discs(pts, normals, sigma, opacity, apart)
        ext_shown = discs(pts, shown, sigma, opacity)[0]
        behind = discs(pts, shown, sigma, opacity, cam=back)
        sb, se = stats(base_cover), stats(ext_cover)
        both = ((base_cover >= .5) & (ext_cover >= .5)).sum()
        either = ((base_cover >= .5) | (ext_cover >= .5)).sum()
        label = "target sample" if name == "target" else f"raw {name} (window {int(name) / 40:.0f})"
        labels = (f"{label}: base display, {len(x)} particles", f"{label}: exterior, {len(pts)} discs, the field's normals", "exterior, the base display's normal treatment")
        video = kind == "lattice" and name != "target"
        pictures = (base, tint(ext, seen), tint(ext_shown, seen))
        sheet(pictures, labels, out / (f"crop/{order:04d}.jpg" if video else f"{name}.jpg"), True)
        sheet(pictures, labels, out / (f"whole/{order:04d}.jpg" if video else f"{name}_whole.jpg"), False)
        row = dict(state=name, **built, discs=len(pts), r_over_a=r / a, spacing_median_over_a=spacing / a,
                   sigma_over_base=r / sp, normal_reach_over_a=reach / a,
                   field_normals=roughness(pts, normals), display_normals=roughness(pts, shown),
                   sets=n_sets, apart=int(apart.sum()), apart_seen_pixels=dict(whole=int((seen >= .5).sum()), crop=int((in_box(seen) >= .5).sum())),
                   base=sb, exterior=se, iou_solid=float(both) / float(either))
        if name == "target":
            ref = ((ext_shown, ext_cover), behind)
        elif ref is not None:                                   # the state against the target sample, as the exterior draws both
            views = dict(front=against(ext_shown, ext_cover, ref[0]), crop=against(in_box(ext_shown), in_box(ext_cover), [in_box(v) for v in ref[0]]),
                         back=against(*behind, ref[1]))
            row["to_target"] = {k: dict(iou=v[0], difference=v[1]) for k, v in views.items()}
        if video:
            solid = solid[-2:] + [(base_cover >= .5, ext_cover >= .5)]
            if len(solid) == 3:                                 # the frame before: solid for that frame only, or not solid for that frame only
                blip = [(s0 == s2) & (s1 != s0) for s0, s1, s2 in zip(*solid)]
                row["blips_before"] = dict(base=int(blip[0].sum()), exterior=int(blip[1].sum()), base_crop=int(in_box(blip[0]).sum()), exterior_crop=int(in_box(blip[1]).sum()))
            order += 1
        else:
            np.savez(out / f"{name}_layer.npz", points=pts.cpu().numpy(), normals=normals.cpu().numpy(), shown=shown.cpu().numpy(), apart=apart.cpu().numpy(), sigma=r)
        print(json.dumps(row), flush=True)
