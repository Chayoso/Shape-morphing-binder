"""inflation_probe.py ARCHIVE_NPZ OUT_DIR RAW:x0,y0,x1,y1 [RAW:x0,y0,x1,y1 ...] — D34: what the display's disc inflation
does to a frame: how much it blurs, and what it closes.

The 4K display draws a particle as a disc of sigma = target spacing x clamp(8th-neighbour distance / coverage radius,
1, CAP) with CAP = 4: where the particles are sparser than the target sample the discs grow. Each frame is drawn with
CAP = 4 (the rule as it is), 2, 1.5 and 1 (no inflation), everything else unchanged (render_axes.py's primitives).
Printed per frame: the rendered particles whose disc is inflated (by 1.1, 1.5, 2 times or more) and the share of the
summed disc area that the inflation adds; per CAP against the rule as it is: the covered pixels lost (coverage >= 0.5
before, < 0.5 now: what the inflation was closing), the width of the silhouette edge, the shading detail inside
(mean luminance gradient at full coverage), and in the box the share of object pixels with a strong luminance
gradient. A sheet of the box under each CAP is saved."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402,F401  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from PIL import Image, ImageDraw                               # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.studio import DensityNormals, StudioRaster  # noqa: E402
from physmorph.render.support import live_support              # noqa: E402

W, H, AZ, EL = 3840, 2160, 35.0, 18.0
dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
td = knn_self_torch(tgt, 9)[0]
spacing, cov_r = float(td[:, 1].median()), float(td[:, 8].median())
center = tgt.mean(0)
radius = float((tgt - center).norm(dim=1).max())
studio = StudioRaster(center, radius, W, H, AZ, EL)
CAPS = (4.0, 2.0, 1.5, 1.0)


def discs(normals, sigma):
    ref = torch.where(normals[:, :1].abs() < .9, normals.new_tensor((1., 0., 0.)), normals.new_tensor((0., 1., 0.))).expand_as(normals)
    tangent = nnf.normalize(torch.linalg.cross(normals, ref), dim=1, eps=1e-9)
    rot = torch.stack((tangent, torch.linalg.cross(normals, tangent), normals), dim=2)
    var = torch.stack((sigma ** 2, sigma ** 2, (sigma / 4) ** 2), dim=1)
    return (rot * var[:, None]) @ rot.transpose(1, 2)


def draw(x, cap):
    """The display renderer's primitives (render_splat_photoreal.py, the unpinned path) with the inflation's cap."""
    with torch.inference_mode():
        d, nb = knn_self_torch(x, 33)
        support = live_support(d, cov_r, spacing)
        sigma = spacing * (d[:, 8] / cov_r).clamp(1., cap)
        normals, mag = DensityNormals(center, radius, spacing, blur=3.0)(x)
        strong = mag >= torch.quantile(mag[::max(1, len(x) // 100000)], .6)
        near = nb[:, 1:33]
        sn = strong[near]
        chosen = near[torch.arange(len(x), device=dev), sn.float().argmax(1)]
        normals = torch.where((~strong & sn.any(1))[:, None], normals[chosen], normals)
        for _ in range(2):
            normals = nnf.normalize(normals[nb].mean(1), dim=1, eps=1e-9)
        return studio(x, normals, discs(normals, sigma), .92 * support, normal_kernel=3, return_buffers=True)


def grad_mag(f):
    gy, gx = torch.gradient(f)
    return (gx ** 2 + gy ** 2).sqrt()


lum = lambda img: img @ img.new_tensor((.2126, .7152, .0722))  # noqa: E731
print(f"target spacing {spacing:.4f} wu, coverage radius {cov_r / spacing:.2f} spacings; caps {CAPS}")
for arg in sys.argv[3:]:
    raw, b = arg.split(":")
    raw, box = int(raw), [int(v) for v in b.split(",")]
    x = torch.as_tensor(np.asarray(z["frames"][raw], np.float32), device=dev)
    d = knn_self_torch(x, 33)[0]
    infl = (d[:, 8] / cov_r).clamp(1., 4.)
    ren = live_support(d, cov_r, spacing) > 0
    area = infl[ren] ** 2
    print(f"\n== raw {raw}: {int(ren.sum())} rendered particles; disc inflated by 1.1x or more on {100 * float((infl[ren] >= 1.1).float().mean()):.1f} %, 1.5x on {100 * float((infl[ren] >= 1.5).float().mean()):.1f} %, "
          f"2x on {100 * float((infl[ren] >= 2).float().mean()):.2f} %; the inflation adds {100 * float((area - 1).sum() / area.sum()):.0f} % of the summed disc area")
    print("   cap | covered pixels lost against the rule as it is % (whole frame, box) | silhouette edge width px | shading detail against the rule as it is | box: strong-gradient share % of object pixels")
    base, tiles = None, []
    for cap in CAPS:
        img, cov, _ = draw(x, cap)
        gc = grad_mag(cov)
        edge = (cov > .4) & (cov < .6) & (gc > 1e-4)
        det = float(grad_mag(lum(img))[cov > .99].mean())
        if base is None:
            base = (cov >= .5, det)
        x0, y0, x1, y1 = box
        cb, bb = cov[y0:y1, x0:x1] >= .5, base[0][y0:y1, x0:x1]
        strong = float((grad_mag(255 * lum(img[y0:y1, x0:x1]))[cb] > 4).float().mean()) if bool(cb.any()) else 0.0
        lost = 100 * float((base[0] & ~(cov >= .5)).sum() / base[0].sum())
        lost_b = 100 * float((bb & ~cb).sum() / bb.sum().clamp_min(1))
        print(f"   {cap:3.1f} | {lost:5.2f}, {lost_b:5.2f} | {float((0.8 / gc[edge]).median()):.1f} | {det / base[1]:.2f} | {100 * strong:.1f}")
        t = Image.fromarray((img[y0:y1, x0:x1].clamp(0, 1) * 255).byte().cpu().numpy())
        ImageDraw.Draw(t).text((8, 6), f"raw {raw}, inflation cap {cap:g}" + (" (the rule as it is)" if cap == 4 else " (no inflation)" if cap == 1 else ""), fill=(255, 255, 0))
        tiles.append(t)
    w, h = tiles[0].size
    sheet = Image.new("RGB", (len(tiles) * w + 6 * (len(tiles) - 1), h), (20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * (w + 6), 0))
    name = out / f"inflation_{raw}_{box[0]}_{box[1]}.jpg"
    sheet.save(name, quality=92)
    print(f"   wrote {name}")
