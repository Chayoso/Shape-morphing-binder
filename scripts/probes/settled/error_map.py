"""error_map.py PHYS_NPZ RENDER_NPZ REF_NPZ OUT_PNG RAW BOX TITLE — where each arm's picture misses the target, side by
side: the physics-only twin and the render arm at kept frame RAW (TAG_frames12.npz), against an independent sample
of the target (REF_NPZ's `tgt`), all drawn as the display draws them (the exterior's Gaussian discs, still_render.py)
from the display camera. Grey: both cover; blue: the target covers and the arm does not (missing); orange: the arm
covers and the target does not (extra). Top row the whole picture, bottom row the crop BOX (x0,y0,x1,y1 at 3840 x
2160) enlarged; each panel with its 1 - IoU and its missing and extra pixels."""
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

dev = torch.device("cuda")
phys, rend, ref = (np.load(p, allow_pickle=True) for p in sys.argv[1:4])
out, raw, title = sys.argv[4], int(sys.argv[5]), sys.argv[7]
box = [int(v) for v in sys.argv[6].split(",")]
target = torch.as_tensor(np.asarray(rend["tgt"], np.float32), device=dev)
reference = torch.as_tensor(np.asarray(ref["tgt"], np.float32), device=dev)
center, radius = reference.mean(0), float((reference - reference.mean(0)).norm(dim=1).max())
W, H = 3840, 2160
cam = StudioRaster(center, radius, W, H, 35., 18.)


def pitch(sample):
    return .708 * float(knn_self_torch(sample, 9)[0][:, 8].median())         # the volume sample's pitch


def cover_of(x, a):
    """The display's exterior of a particle set at the field pitch a, and its coverage from the display camera
    (still_render.py's discs)."""
    lat = Lattice(center, 2.8 * radius)
    with torch.no_grad():
        field = ZhuBridson(x, a)
        h = .4 * a
        h *= (len(lat.discs(field, h)[0]) / 300000) ** .5
        pts, g, _, _ = lat.discs(field, h)
        n = nnf.normalize(g, dim=1)
        ref_v = torch.where(n[:, :1].abs() < .9, n.new_tensor((1., 0., 0.)), n.new_tensor((0., 1., 0.))).expand_as(n)
        t = nnf.normalize(torch.linalg.cross(n, ref_v), dim=1, eps=1e-9)
        rot = torch.stack((t, torch.linalg.cross(n, t), n), dim=2)
        sig = torch.full((len(pts),), h, device=dev)
        cov = (rot * torch.stack((sig ** 2, sig ** 2, (sig / 4) ** 2), 1)[:, None]) @ rot.transpose(1, 2)
        return cam.raster_colors(pts, n, cov, torch.full((len(pts),), .92, device=dev), torch.ones_like(pts))[..., 0] >= .5


def frame(z):
    raws = [int(v) for v in z["raws"]]
    k = min(range(len(raws)), key=lambda i: abs(raws[i] - raw))
    return torch.as_tensor(np.asarray(z["frames"][k], np.float32), device=dev), raws[k]


tc = cover_of(reference, pitch(reference))
a_run = pitch(target)
panels, stats = [], []
for name, z in (("physics only", phys), ("with render", rend)):
    x, r_used = frame(z)
    c = cover_of(x, a_run)
    img = np.full((H, W, 3), 255, np.uint8)
    both, miss, extra = (tc & c).cpu().numpy(), (tc & ~c).cpu().numpy(), (~tc & c).cpu().numpy()
    img[both] = (205, 207, 212)
    img[miss] = (42, 120, 214)
    img[extra] = (232, 104, 34)
    panels.append(img)
    for part, sl in (("whole", (slice(None), slice(None))), ("crop", (slice(box[1], box[3]), slice(box[0], box[2])))):
        b, m, e = both[sl].sum(), miss[sl].sum(), extra[sl].sum()
        stats.append((name, part, r_used, 1 - b / max(b + m + e, 1), m, e))

font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 44)
small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
cw, ch = 1600, int(1600 * (box[3] - box[1]) / (box[2] - box[0]))
sheet = Image.new("RGB", (2 * 1920, 1080 + ch + 180), (255, 255, 255))
dr = ImageDraw.Draw(sheet)
for i, img in enumerate(panels):
    whole = Image.fromarray(img).resize((1920, 1080), Image.LANCZOS)
    sheet.paste(whole, (i * 1920, 70))
    crop = Image.fromarray(img[box[1]:box[3], box[0]:box[2]]).resize((cw, ch), Image.NEAREST)
    sheet.paste(crop, (i * 1920 + (1920 - cw) // 2, 1080 + 110))
    sw, sc = stats[2 * i], stats[2 * i + 1]
    dr.text((i * 1920 + 30, 12), f"{sw[0]}  (raw frame {sw[2]})", fill=(20, 20, 20), font=font)
    dr.text((i * 1920 + 30, 1080 + 76 - 40), f"whole: 1-IoU {sw[3]:.4f}, missing {sw[4]} px, extra {sw[5]} px", fill=(70, 70, 70), font=small)
    dr.text((i * 1920 + 30, 1080 + 110 + ch + 2), f"crop: 1-IoU {sc[3]:.4f}, missing {sc[4]} px, extra {sc[5]} px", fill=(70, 70, 70), font=small)
dr.line([(1920, 0), (1920, sheet.height)], fill=(180, 180, 180), width=4)
sheet.save(out)
print(title, "| " + " | ".join(f"{s[0]} {s[1]}: 1-IoU {s[3]:.4f} missing {s[4]} extra {s[5]}" for s in stats), flush=True)
