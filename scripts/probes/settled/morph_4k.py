"""morph_4k.py FRAMES_NPZ OUT_DIR [STRIDE] — the morph in 4K as the display draws it, in the studio's own look (a light
body on its grey gradient, StudioRaster's defaults): every STRIDE-th kept frame of TAG_frames12.npz (default 1) and the
last, each the exterior's Gaussian discs (surface_layer_probe.py's lattice: one disc per cell of a lattice fixed in
space that the particles' Zhu–Bridson zero set crosses, the lattice's pitch set once so that the target sample takes
about 300k discs; normals the base display's, averaged twice over its reach; opacity 0.92) from the display camera
(azimuth 35, elevation 18) at 3840 x 2160. Written to OUT_DIR: frames/NNNN.jpg, last.png (the last frame) and
target.png (the target sample drawn the same way); the caller joins the frames (ffmpeg)."""
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

dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
out = Path(sys.argv[2]); (out / "frames").mkdir(parents=True, exist_ok=True)
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1
target = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
center, radius = target.mean(0), float((target - target.mean(0)).norm(dim=1).max())
cov_r = float(knn_self_torch(target, 9)[0][:, 8].median())
a = .708 * cov_r                                               # the volume sample's pitch
W, H = 3840, 2160
cam = StudioRaster(center, radius, W, H, 35., 18.)            # the studio's grey gradient and light body
lat = Lattice(center, 2.8 * radius)
with torch.no_grad():                                          # the lattice's pitch, once: the target takes about 300k discs
    h = .4 * a
    h *= (len(lat.discs(ZhuBridson(target, a), h)[0]) / 300000) ** .5


def draw(x):
    with torch.no_grad():
        pts, g, _, _ = lat.discs(ZhuBridson(x, a), h)
        shown = nnf.normalize(g, dim=1)
        d, nb = knn_self_torch(x, 33)
        reach = float(d[(x - x[nb[:, 1:]].mean(1)).norm(dim=1) >= .5 * cov_r, 32].median())
        spacing = float(knn_self_torch(pts, 2)[0][:, 1].median())
        dd, nn = gpu.KNN(pts).query(pts, int(min(256, 3.63 * (reach / spacing) ** 2)) + 1)
        w = (dd.float() <= reach)[..., None]
        for _ in range(2):
            shown = nnf.normalize((shown[nn] * w).sum(1), dim=1, eps=1e-9)
        ref = torch.where(shown[:, :1].abs() < .9, shown.new_tensor((1., 0., 0.)), shown.new_tensor((0., 1., 0.))).expand_as(shown)
        t = nnf.normalize(torch.linalg.cross(shown, ref), dim=1, eps=1e-9)
        rot = torch.stack((t, torch.linalg.cross(shown, t), shown), dim=2)
        sig = torch.full((len(pts),), h, device=dev)
        cov = (rot * torch.stack((sig ** 2, sig ** 2, (sig / 4) ** 2), 1)[:, None]) @ rot.transpose(1, 2)
        img = cam(pts, shown, cov, torch.full((len(pts),), .92, device=dev), normal_kernel=normal_filter_size(H, False))
    return Image.fromarray((img.clamp(0, 1) * 255).round().byte().cpu().numpy())


n = len(z["raws"])
keep = list(range(0, n, stride)) + ([n - 1] if (n - 1) % stride else [])
for i, k in enumerate(keep):
    im = draw(torch.as_tensor(np.asarray(z["frames"][k], np.float32), device=dev))
    im.save(out / "frames" / f"{i:04d}.jpg", quality=93)
    if k == n - 1:
        im.save(out / "last.png")
    if i % 25 == 0:
        print(f"frame {i + 1} / {len(keep)} (raw {int(z['raws'][k])})", flush=True)
draw(target).save(out / "target.png")
print("wrote", out, len(keep), "frames", flush=True)
