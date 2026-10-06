"""detail_compare.py OUT_DIR FOCUS ZOOM LABEL=NPZ:STATE [...] — the display's detail side by side: each state drawn as
still_render.py draws it (the Zhu–Bridson exterior's Gaussian discs, about 300k, normals averaged over the base
display's reach, slate blue under the studio's lights, on white), whole from the display camera and close (a camera
ZOOM times nearer, looking at the target's top 3 % (FOCUS=top) or its point facing the camera (FOCUS=face)).
STATE: `last` (the last kept frame of a TAG_frames12.npz, drawn at the pitch of the run's target), `tgt` (the file's
target sample at the same pitch), `own` (the file's `tgt` at its own pitch: a denser sample, finer). The first state's
file fixes the camera and the run's pitch. `+smooth=N`: the normals averaged N times over the reach (default 2, the
base display; 0 shows the field's own normals). Written to OUT_DIR: whole_<i>.png, close_<i>.png and sheet.png (the
columns in order, whole above close, each labelled)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from PIL import Image, ImageDraw, ImageFont                    # noqa: E402
from physmorph.render.exterior import Lattice, ZhuBridson      # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.studio import StudioRaster               # noqa: E402
from physmorph.render.support import normal_filter_size        # noqa: E402

dev = torch.device("cuda")
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
focus, zoom = sys.argv[2], float(sys.argv[3])
opts = dict(s[1:].split("=", 1) for s in sys.argv[4:] if s.startswith("+"))
PASSES = int(opts.get("smooth", 2))                             # the base display averages the normals twice over its reach
specs = [(s.split("=", 1)[0], *s.split("=", 1)[1].rsplit(":", 1)) for s in sys.argv[4:] if not s.startswith("+")]
first = np.load(specs[0][1], allow_pickle=True)
target = torch.as_tensor(np.asarray(first["tgt"], np.float32), device=dev)
center, radius = target.mean(0), float((target - target.mean(0)).norm(dim=1).max())
pitch = lambda s: .708 * float(knn_self_torch(s, 9)[0][:, 8].median())   # noqa: E731  (the volume sample's pitch)
a_run = pitch(target)
az, el = np.radians(35.), np.radians(18.)
toward = target.new_tensor((np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)))
if focus == "face":
    look = target[int(((target - center) @ toward).argmax())]
else:
    look = target[target[:, 1] > torch.quantile(target[::7, 1], .97)].mean(0)
W, H = 1920, 1080


def discs(x, a):
    """(points, shown normals, sigma) of the display's exterior of x at field pitch a (still_render.py)."""
    lat = Lattice(center, 2.8 * radius)
    with torch.no_grad():
        field = ZhuBridson(x, a)
        h = .4 * a
        h *= (len(lat.discs(field, h)[0]) / 300000) ** .5
        pts, g, _, _ = lat.discs(field, h)
        shown = nnf.normalize(g, dim=1)
        cov_r = a / .708
        d, nb = knn_self_torch(x, 33)
        reach = float(d[(x - x[nb[:, 1:]].mean(1)).norm(dim=1) >= .5 * cov_r, 32].median())
        spacing = float(knn_self_torch(pts, 2)[0][:, 1].median())
        dd, nn = gpu.KNN(pts).query(pts, int(min(256, 3.63 * (reach / spacing) ** 2)) + 1)
        w = (dd.float() <= reach)[..., None]
        for _ in range(PASSES):
            shown = nnf.normalize((shown[nn] * w).sum(1), dim=1, eps=1e-9)
    return pts, shown, h


def shade(cam, pts, n, h):
    ref = torch.where(n[:, :1].abs() < .9, n.new_tensor((1., 0., 0.)), n.new_tensor((0., 1., 0.))).expand_as(n)
    t = nnf.normalize(torch.linalg.cross(n, ref), dim=1, eps=1e-9)
    rot = torch.stack((t, torch.linalg.cross(n, t), n), dim=2)
    sig = torch.full((len(pts),), h, device=dev)
    cov = (rot * torch.stack((sig ** 2, sig ** 2, (sig / 4) ** 2), 1)[:, None]) @ rot.transpose(1, 2)
    cam.background = torch.zeros_like(cam.background)
    cam.albedo = torch.tensor((.08, .17, .40), device=dev)    # slate blue
    with torch.no_grad():
        img, cover, _ = cam(pts, n, cov, torch.full((len(pts),), .92, device=dev),
                            normal_kernel=normal_filter_size(H, False), return_buffers=True)
    return Image.fromarray(((img + (1. - cover[..., None])).clamp(0, 1) * 255).round().byte().cpu().numpy())


tiles = []
for i, (label, path, state) in enumerate(specs):
    z = np.load(path, allow_pickle=True)
    if state == "last":
        x, a = torch.as_tensor(np.asarray(z["frames"][-1], np.float32), device=dev), a_run
    else:
        x = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
        a = a_run if state == "tgt" else pitch(x)
    pts, n, h = discs(x, a)
    whole = shade(StudioRaster(center, radius, W, H, 35., 18.), pts, n, h)
    close = shade(StudioRaster(look, radius / zoom, W, H, 35., 18.), pts, n, h)
    whole.save(out / f"whole_{i}.png"); close.save(out / f"close_{i}.png")
    tiles.append((label, whole, close))
    print(f"{label}: {len(x)} particles, pitch {a:.4f} wu, {len(pts)} discs", flush=True)

font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
tw, th = W // 2, H // 2
sheet = Image.new("RGB", (len(tiles) * tw, 2 * th + 50), (255, 255, 255))
dr = ImageDraw.Draw(sheet)
for i, (label, whole, close) in enumerate(tiles):
    sheet.paste(whole.resize((tw, th), Image.LANCZOS), (i * tw, 50))
    sheet.paste(close.resize((tw, th), Image.LANCZOS), (i * tw, 50 + th))
    dr.text((i * tw + 16, 10), label, fill=(30, 30, 30), font=font)
    if i:
        dr.line([(i * tw, 0), (i * tw, sheet.height)], fill=(200, 200, 200), width=2)
sheet.save(out / "sheet.png")
print("wrote", out, flush=True)
