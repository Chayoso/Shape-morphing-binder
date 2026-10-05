"""camera_ring.py FRAMES_NPZ OUT_DIR [RAW] — the body in the middle of its cameras, seen from a camera further out and
above (the display's azimuth, 30 degrees up), on white. The body is the display's surface (still_render.py's Gaussian
discs, slate blue); each camera a triangle, its apex at the camera and its base toward the body; cameras behind the
body are drawn under it. Written to OUT_DIR:
  cameras_8.png             eight cameras evenly on one ring at 15 degrees up (an illustration);
  cameras_render_views.png  the render loss's own views (render_loss.make_views: six azimuths on each of the rings at
                            0 and +-0.5 rad, every ring turned by its share of a step), orthographic in the loss.
The ring's radius is drawn at 1.8 times the body's radius (the loss's cameras are orthographic: their distance does
not matter). Screen positions follow the rasteriser's own projection; the axes' orientation is taken from the body's
discs landing on the body's own pixels."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from PIL import Image, ImageDraw                               # noqa: E402
from physmorph.pipeline.render_loss import make_views          # noqa: E402
from physmorph.render.covariance_torch import world_to_view_torch  # noqa: E402
from physmorph.render.exterior import Lattice, ZhuBridson      # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.studio import StudioRaster               # noqa: E402
from physmorph.render.support import normal_filter_size        # noqa: E402

dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
raws = [int(v) for v in z["raws"]]
k = raws.index(int(sys.argv[3])) if len(sys.argv) > 3 else len(raws) - 1
x = torch.as_tensor(np.asarray(z["frames"][k], np.float32), device=dev)
target = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
center, radius = target.mean(0), float((target - target.mean(0)).norm(dim=1).max())
cov_r = float(knn_self_torch(target, 9)[0][:, 8].median())
a = .708 * cov_r
W, H, AZ, EL, SCALE, RING = 3000, 2000, 35., 30., 2.2, 1.8
cam = StudioRaster(center, SCALE * radius, W, H, AZ, EL)     # further out: the ring fits in the frame
lat = Lattice(center, 2.8 * radius)

with torch.no_grad():                                          # the display's discs (still_render.py)
    field = ZhuBridson(x, a)
    h = .4 * a
    h *= (len(lat.discs(field, h)[0]) / 300000) ** .5
    pts, g, _, _ = lat.discs(field, h)
    normals = nnf.normalize(g, dim=1)
    d, nb = knn_self_torch(x, 33)
    reach = float(d[(x - x[nb[:, 1:]].mean(1)).norm(dim=1) >= .5 * cov_r, 32].median())
    spacing = float(knn_self_torch(pts, 2)[0][:, 1].median())
    dd, nn = gpu.KNN(pts).query(pts, int(min(256, 3.63 * (reach / spacing) ** 2)) + 1)
    w = (dd.float() <= reach)[..., None]
    for _ in range(2):
        normals = nnf.normalize((normals[nn] * w).sum(1), dim=1, eps=1e-9)
    ref = torch.where(normals[:, :1].abs() < .9, normals.new_tensor((1., 0., 0.)), normals.new_tensor((0., 1., 0.))).expand_as(normals)
    t = nnf.normalize(torch.linalg.cross(normals, ref), dim=1, eps=1e-9)
    rot = torch.stack((t, torch.linalg.cross(normals, t), normals), dim=2)
    sig = torch.full((len(pts),), h, device=dev)
    cov = (rot * torch.stack((sig ** 2, sig ** 2, (sig / 4) ** 2), 1)[:, None]) @ rot.transpose(1, 2)
    cam.background = torch.zeros_like(cam.background)
    cam.albedo = torch.tensor((.08, .17, .40), device=dev)    # slate blue
    body, cover, _ = cam(pts, normals, cov, torch.full((len(pts),), .92, device=dev),
                         normal_kernel=normal_filter_size(H, False), return_buffers=True)

# the rasteriser's projection (StudioRaster: Graphdeco view, 30 degree vertical field, then the rows flipped)
az, el = math.radians(AZ), math.radians(EL)
eye = center + 3.6 * SCALE * radius * center.new_tensor((math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)))
view = world_to_view_torch(eye, center)
tan_y = math.tan(math.radians(15)); tan_x = tan_y * W / H


def raw_project(p):
    q = (view[:3, :3] @ p.T).T + view[:3, 3]
    return q[:, 0] / (q[:, 2] * tan_x), q[:, 1] / (q[:, 2] * tan_y), q[:, 2]


def orient(nx, ny, fx, fy):
    u = ((fx * nx + 1) * W - 1) / 2
    v = ((fy * ny + 1) * H - 1) / 2
    return u, v


nx, ny, _ = raw_project(pts[::50])
best, sign = -1., (1, 1)
for fx in (1, -1):
    for fy in (1, -1):
        u, v = orient(nx, ny, fx, fy)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        hit = float(cover[v[ok].long(), u[ok].long()].gt(.5).float().mean()) if ok.any() else 0.
        if hit > best:
            best, sign = hit, (fx, fy)
print(f"{len(pts)} discs; projection axes {sign}: {100 * best:.1f} % of the body's discs land on its own pixels", flush=True)


def project(p):
    px, py, depth = raw_project(p)
    u, v = orient(px, py, *sign)
    return torch.stack((u, v), 1).cpu().numpy(), depth.cpu().numpy()


def figure(rings, name, label):
    """rings: list of (azimuths, elevation) in radians."""
    cams = [(th, ph) for ths, ph in rings for th in ths]
    pos = torch.stack([center + RING * radius * center.new_tensor((math.cos(ph) * math.sin(th), math.sin(ph), math.cos(ph) * math.cos(th)))
                       for th, ph in cams])
    uv, depth = project(pos)
    cuv, cdepth = project(center[None])
    aim, _ = project(pos + .28 * (center - pos))
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    under = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    # the rings as faint ellipses (a dashed look: every other segment)
    for ths, ph in rings:
        ring = torch.stack([center + RING * radius * center.new_tensor((math.cos(ph) * math.sin(tt), math.sin(ph), math.cos(ph) * math.cos(tt)))
                            for tt in np.linspace(0, 2 * np.pi, 241)])
        ruv, rdepth = project(ring)
        for i in range(0, 240, 2):
            layer = over if rdepth[i] < cdepth[0] else under
            ImageDraw.Draw(layer).line([tuple(ruv[i]), tuple(ruv[i + 1])], fill=(120, 130, 145, 200), width=4)
    near, far = depth.min(), depth.max()
    for (u, v), dep, (au, av) in sorted(zip(uv, depth, aim), key=lambda e: -e[1]):
        layer = over if dep < cdepth[0] else under
        size = 62 * (1.2 - .4 * (dep - near) / max(far - near, 1e-9))     # nearer cameras drawn larger
        n = math.hypot(au - u, av - v) + 1e-9
        dx, dy = (au - u) / n, (av - v) / n                                # toward the body on screen
        # the camera: a triangle, its apex at the camera, its base toward the body
        base = (u + 1.4 * size * dx, v + 1.4 * size * dy)
        p1 = (base[0] - .7 * size * dy, base[1] + .7 * size * dx)
        p2 = (base[0] + .7 * size * dy, base[1] - .7 * size * dx)
        ImageDraw.Draw(layer).polygon([(u, v), p1, p2], fill=(43, 47, 54, 235), outline=(20, 22, 26, 255), width=3)
    canvas.paste(under, (0, 0), under)
    img = (body + (1. - cover[..., None])).clamp(0, 1)
    obj = Image.fromarray((img * 255).round().byte().cpu().numpy())
    mask = Image.fromarray((cover.clamp(0, 1) * 255).round().byte().cpu().numpy())
    canvas.paste(obj, (0, 0), mask)
    canvas.paste(over, (0, 0), over)
    # crop to the drawn content with a margin
    box = Image.eval(canvas.convert("L"), lambda q: 255 - q).getbbox()
    if box:
        m = 60
        canvas = canvas.crop((max(0, box[0] - m), max(0, box[1] - m), min(W, box[2] + m), min(H, box[3] + m)))
    canvas.save(out / name)
    print("wrote", out / name, canvas.size, flush=True)


figure([(np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8, math.radians(15))], "cameras_8.png",
       "8 cameras on a ring around the morphed dragon")
views = make_views(6, (0.0, 0.5, -0.5))
rings = {}
for th, ph in views:
    rings.setdefault(ph, []).append(th)
figure([(np.array(v), ph) for ph, v in rings.items()], "cameras_render_views.png",
       "the render loss's 18 views (6 azimuths x 3 rings at 0, +-28.6 degrees)")
