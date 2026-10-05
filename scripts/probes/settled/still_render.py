"""still_render.py FRAMES_NPZ OUT_DIR [RAW] [focus=top|face] [zoom=A,B] [discs=N] — stills of one state on a white background (the
last kept frame unless RAW; the close cameras on the top of the body or on the point facing the camera, A and B times
nearer).

The surface is the display's (surface_layer_probe.py, lattice): flat Gaussian discs, one for each cell of a lattice
fixed in space that the particles' Zhu–Bridson zero set crosses (about 300k at 300k particles), sigma the lattice
pitch, opacity 0.92, normals averaged over the base display's reach. Written to OUT_DIR:
  shaded_<colour>.png   the discs under the studio's lights (GGX), each material colour in turn, on white;
  gaussians_whole.png   the same discs one by one: each in its own colour, shrunk to 30 % of its sigma (opaque), lit by the key
                        light alone, only the discs facing the camera, so the gaps show every disc's elliptical footprint;
  gaussians_head.png, gaussians_close.png  close cameras on the head (8 and 18 times nearer): each disc tens of
                        pixels across;
  sheet.png             the shaded colours side by side."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from PIL import Image                                          # noqa: E402
from physmorph.render.exterior import Lattice, ZhuBridson      # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.studio import StudioRaster               # noqa: E402
from physmorph.render.support import normal_filter_size        # noqa: E402

COLOURS = {"terracotta": (.45, .13, .06), "slate_blue": (.08, .17, .40), "jade": (.05, .30, .22),
           "ochre": (.55, .30, .04), "graphite": (.09, .09, .11)}       # linear albedos; the studio's lights and tone map lift them
dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
raws = [int(v) for v in z["raws"]]
opts = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
raw = [a for a in sys.argv[3:] if "=" not in a]
k = raws.index(int(raw[0])) if raw else len(raws) - 1
FOCUS = opts.get("focus", "top")                                # close cameras on the top (the head) or on the point facing the camera
ZOOMS = [float(v) for v in opts.get("zoom", "8,18").split(",")]
DISCS = int(opts.get("discs", "300000"))                     # the discs the surface takes (the display: 300k)
x = torch.as_tensor(np.asarray(z["frames"][k], np.float32), device=dev)
target = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
center, radius = target.mean(0), float((target - target.mean(0)).norm(dim=1).max())
td = knn_self_torch(target, 9)[0]
cov_r = float(td[:, 8].median())
a = .708 * cov_r                                               # the volume sample's pitch
W, H = 3840, 2160
cam = StudioRaster(center, radius, W, H, 35., 18.)
lat = Lattice(center, 2.8 * radius)

with torch.no_grad():
    field = ZhuBridson(x, a)
    h = .4 * a                                                 # the pitch at which the surface takes about DISCS discs
    h *= (len(lat.discs(field, h)[0]) / DISCS) ** .5
    pts, g, _, _ = lat.discs(field, h)
    normals = nnf.normalize(g, dim=1)
    # the base display's treatment of the normals: twice the mean over the neighbours within its reach
    d, nb = knn_self_torch(x, 33)
    reach = float(d[(x - x[nb[:, 1:]].mean(1)).norm(dim=1) >= .5 * cov_r, 32].median())
    spacing = float(knn_self_torch(pts, 2)[0][:, 1].median())
    kk = int(min(256, 3.63 * (reach / spacing) ** 2)) + 1
    dd, nn = gpu.KNN(pts).query(pts, kk)
    w = (dd.float() <= reach)[..., None]
    shown = normals
    for _ in range(2):
        shown = nnf.normalize((shown[nn] * w).sum(1), dim=1, eps=1e-9)
print(f"state raw {raws[k]}: {len(x)} particles, {len(pts)} discs, lattice pitch {h / a:.3f} a, sigma {h:.4f} wu", flush=True)


def covariance(n, sigma):
    ref = torch.where(n[:, :1].abs() < .9, n.new_tensor((1., 0., 0.)), n.new_tensor((0., 1., 0.))).expand_as(n)
    t = nnf.normalize(torch.linalg.cross(n, ref), dim=1, eps=1e-9)
    rot = torch.stack((t, torch.linalg.cross(n, t), n), dim=2)
    var = torch.stack((sigma ** 2, sigma ** 2, (sigma / 4) ** 2), dim=1)
    return (rot * var[:, None]) @ rot.transpose(1, 2)


def save(img, name):
    Image.fromarray((img.clamp(0, 1) * 255).round().byte().cpu().numpy()).save(out / name)


def frame_box(cover, margin=.06):
    """The body's box on screen with a margin, 16:9."""
    ys, xs = torch.nonzero(cover > .5, as_tuple=True)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    hgt = int((y1 - y0) * (1 + 2 * margin)); wid = max(int((x1 - x0) * (1 + 2 * margin)), hgt * 16 // 9)
    hgt = max(hgt, wid * 9 // 16)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    top, left = max(0, min(cover.shape[0] - hgt, cy - hgt // 2)), max(0, min(cover.shape[1] - wid, cx - wid // 2))
    return slice(top, top + hgt), slice(left, left + wid)


opacity = torch.full((len(pts),), .92, device=dev)
sigma = torch.full((len(pts),), h, device=dev)
with torch.no_grad():
    cov = covariance(shown, sigma)
    cam.background = torch.zeros_like(cam.background)          # composited on white below
    shots, box = [], None
    for name, rgb in COLOURS.items():
        cam.albedo = torch.tensor(rgb, device=dev)
        img, cover, _ = cam(pts, shown, cov, opacity, normal_kernel=normal_filter_size(H, False), return_buffers=True)
        img = img + (1. - cover[..., None])                    # white behind, in display space
        box = box or frame_box(cover)
        save(img[box], f"shaded_{name}.png")
        shots.append(img[box])
    sheet = torch.cat([s[::2, ::2] for s in shots], 1)
    save(sheet, "sheet.png")

    # the discs one by one: random hues, each disc shrunk to 30 % of its sigma (opaque), lit by the key light alone
    gen = torch.Generator(device=dev).manual_seed(7)
    hue = torch.rand(len(pts), generator=gen, device=dev)
    hsv = torch.stack((hue, torch.full_like(hue, .55), torch.full_like(hue, .95)), 1)
    i = (hsv[:, 0] * 6).floor() % 6
    f = hsv[:, 0] * 6 - (hsv[:, 0] * 6).floor()
    v, s = hsv[:, 2], hsv[:, 1]
    p_, q_, t_ = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    rgb = torch.stack([torch.stack(c, 1) for c in ((v, t_, p_), (q_, v, p_), (p_, v, t_), (p_, q_, v), (t_, p_, v), (v, p_, q_))], 0)
    colour = rgb[i.long(), torch.arange(len(pts), device=dev)]
    key = cam.lights[0][0]
    lit = (.35 + .65 * (shown @ key).clamp(0, 1))[:, None] * colour
    small = covariance(shown, .3 * sigma)
    full = torch.ones_like(opacity)
    az, el = np.radians(35.), np.radians(18.)
    toward = pts.new_tensor((np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)))

    def drawn(c, look_at, r):
        """The discs facing the camera (looking at look_at from 3.6 r along the display direction), one by one."""
        eye = look_at + 3.6 * r * toward
        f = ((eye - pts) * shown).sum(1) > 0
        img = c.raster_colors(pts[f], shown[f], small[f], full[f], lit[f])
        cover = c.raster_colors(pts[f], shown[f], small[f], full[f], torch.ones_like(lit[f]))[..., :1]
        return img + (1. - cover)

    save(drawn(cam, center, radius)[box], "gaussians_whole.png")
    if FOCUS == "face":                                        # the surface point nearest the display camera: seen face on
        head = pts[int(((pts - center) @ toward).argmax())]
    else:                                                      # the top 3 % of the discs (the head)
        head = pts[pts[:, 1] > torch.quantile(pts[::7, 1], .97)].mean(0)
    for zoom, name in zip(ZOOMS, ("gaussians_head.png", "gaussians_close.png")):
        save(drawn(StudioRaster(head, radius / zoom, 1920, 1080, 35., 18.), head, radius / zoom), name)
print("wrote", out)
