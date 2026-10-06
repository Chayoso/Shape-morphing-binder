"""sim_camera.py FRAMES_NPZ OUT_DIR — the simulation seen by one camera: the run's first frame (the sphere) and its last
(the morphed body) side by side with three dots between them (the simulation), drawn as the display draws them
(still_render.py's Gaussian discs, slate blue, from the display's azimuth), and below them one camera looking up at the
row: a body, a lens ring and a hood opening toward the row (camera_ring.py's spotlight shape, unlit). The dots and the
camera stand on the picture's middle. Both bodies are drawn at one scale (the target's radius), each about its own
centre. Written to OUT_DIR:
  sim_camera.png          the row and the camera;
  sim_camera_frustum.png  the same with the camera's field of view as two thin dashed lines (no light)."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from PIL import Image, ImageDraw                               # noqa: E402
from physmorph.render.exterior import Lattice, ZhuBridson      # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.studio import StudioRaster               # noqa: E402
from physmorph.render.support import normal_filter_size        # noqa: E402

dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
target = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
radius = float((target - target.mean(0)).norm(dim=1).max())
cov_r = float(knn_self_torch(target, 9)[0][:, 8].median())
a = .708 * cov_r
S, AZ, EL = 1800, 35., 18.


def draw(x):
    """(colour (S,S,3) in [0,1] where covered, cover (S,S)) of particle set x: the display's discs about x's centre."""
    center = x.mean(0)
    lat = Lattice(center, 2.8 * radius)
    with torch.no_grad():
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
        cam = StudioRaster(center, 1.75 * radius, S, S, AZ, EL)
        cam.background = torch.zeros_like(cam.background)
        cam.albedo = torch.tensor((.08, .17, .40), device=dev)    # slate blue
        img, cover, _ = cam(pts, normals, cov, torch.full((len(pts),), .92, device=dev),
                            normal_kernel=normal_filter_size(S, False), return_buffers=True)
    cover = cover.clamp(0, 1).cpu().numpy()
    return np.clip(img.cpu().numpy() / np.maximum(cover[..., None], 1e-3), 0, 1), cover


panels = [draw(torch.as_tensor(np.asarray(z["frames"][k], np.float32), device=dev)) for k in (0, len(z["raws"]) - 1)]
print("raws drawn:", int(z["raws"][0]), int(z["raws"][-1]), flush=True)


def bbox(cover):
    ys, xs = np.nonzero(cover > .5)
    return xs.min(), ys.min(), xs.max(), ys.max()


boxes = [bbox(c) for _, c in panels]
gap = int(.42 * S)                                             # the dots' room between the bodies
widths = [b[2] - b[0] + 1 for b in boxes]
row_h = max(b[3] - b[1] + 1 for b in boxes)
m = int(.08 * S)
cam_room = int(.34 * S)                                        # below the row: the camera and its distance to the row
slot = max(widths)                                             # each body centred in a slot of the wider one's width,
W = m + slot + gap + slot + m                                  # so the dots and the camera stand on the picture's middle
H = m + row_h + cam_room + m
bodies = np.zeros((H, W, 3), np.float32)
alpha = np.zeros((H, W, 1), np.float32)
mid_y = m + row_h / 2                                          # the row's centre line
lefts = [m + (slot - widths[0]) // 2, m + slot + gap + (slot - widths[1]) // 2]
for (img, cover), (x0, y0, x1, y1), left in zip(panels, boxes, lefts):
    top = int(round(mid_y - (y1 - y0 + 1) / 2))                # each body centred on the row's line
    bodies[top:top + y1 - y0 + 1, left:left + x1 - x0 + 1] = img[y0:y1 + 1, x0:x1 + 1]
    alpha[top:top + y1 - y0 + 1, left:left + x1 - x0 + 1] = cover[y0:y1 + 1, x0:x1 + 1, None]
dots_x = W / 2


def camera(dr, cx, bottom, size):
    """A camera pointing up, its foot at `bottom`: a rounded body, a lens ring, and a hood that opens toward the row
    (the spotlight's shape, unlit). Returns the hood's two upper corners."""
    ink, edge, ring = (43, 47, 54), (20, 22, 26), (92, 99, 112)
    bw, bh = 1.25 * size, 1.15 * size                          # the body
    top = bottom - bh
    dr.rounded_rectangle([cx - bw / 2, top, cx + bw / 2, bottom], radius=.16 * size, fill=ink, outline=edge, width=4)
    rw, rh = .62 * size, .16 * size                            # the lens ring
    dr.rectangle([cx - rw / 2, top - rh, cx + rw / 2, top], fill=ring, outline=edge, width=3)
    hb, ht, hh = .62 * size, 1.45 * size, .62 * size           # the hood, widening upward
    y0, y1 = top - rh, top - rh - hh
    dr.polygon([(cx - hb / 2, y0), (cx + hb / 2, y0), (cx + ht / 2, y1), (cx - ht / 2, y1)], fill=ink, outline=edge, width=4)
    return (cx - ht / 2, y1), (cx + ht / 2, y1)


def finish(frustum, name):
    pic = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(pic)
    size = .075 * S
    cx, bottom = W / 2, H - m                                  # the camera, under the dots, looking up
    corners = camera(ImageDraw.Draw(Image.new("RGB", (1, 1))), cx, bottom, size)   # where the hood's corners fall
    if frustum:                                                 # its field of view: two thin dashed lines behind the bodies, no light
        top_y = m * .6
        for side, corner in zip((-1, 1), corners):
            x_end = cx + side * (W / 2 - m * .6)
            p0, p1 = np.array(corner), np.array((x_end, top_y))
            L = float(np.linalg.norm(p1 - p0)); u = (p1 - p0) / L
            dash, s = .018 * S, 0.
            while s < L:
                q0, q1 = p0 + u * s, p0 + u * min(s + dash, L)
                dr.line([tuple(q0), tuple(q1)], fill=(150, 158, 170), width=4)
                s += 2 * dash
    under = np.asarray(pic, np.float32) / 255
    pic = Image.fromarray(((under * (1 - alpha) + bodies * alpha) * 255).round().astype(np.uint8))
    dr = ImageDraw.Draw(pic)
    r = .012 * S
    for k in (-1, 0, 1):                                        # the simulation: three dots
        dx = dots_x + k * 4.2 * r
        dr.ellipse([dx - r, mid_y - r, dx + r, mid_y + r], fill=(110, 118, 132))
    camera(dr, cx, bottom, size)
    pic.save(out / name)
    print("wrote", out / name, pic.size, flush=True)


finish(False, "sim_camera.png")
finish(True, "sim_camera_frustum.png")
