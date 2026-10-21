"""sc_views.py OUT_PNG NPZ [AZ,AZ,...] [EL] [KEY=frames|tgt] -- SHOWCASE: a preview strip for choosing a render camera (CPU only).
The sample (the npz's `tgt`, or its last frame with KEY=frames) seen by render_deliverable.py's studio camera (eye at
centre + 3.6 radii (cos el sin az, sin el, cos el cos az), 30 degree vertical field of view, +y up): a z-buffer of the
points at 640 x 360 per azimuth, shaded by the depth map's normals. A preview, not the renderer."""
import sys

import numpy as np
from PIL import Image, ImageDraw

out, path = sys.argv[1], sys.argv[2]
azs = [float(a) for a in (sys.argv[3] if len(sys.argv) > 3 else "0,35,90,145,180,270").split(",")]
el = float(sys.argv[4]) if len(sys.argv) > 4 else 18.0
key = sys.argv[5] if len(sys.argv) > 5 else "tgt"
z = np.load(path, allow_pickle=True)
x = np.asarray(z["frames"][-1] if key == "frames" else z["tgt"], np.float64)
tg = np.asarray(z["tgt"], np.float64)
c = tg.mean(0)
r = float(np.linalg.norm(tg - c, axis=1).max())
W, H = 640, 360
tiles = []
for az in azs:
    a, e = np.radians(az), np.radians(el)
    cam = c + 3.6 * r * np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])
    f = (c - cam) / np.linalg.norm(c - cam)
    right = np.cross(f, [0.0, 1.0, 0.0]); right /= np.linalg.norm(right)
    up = np.cross(right, f)
    p = x - cam
    zc, xc, yc = p @ f, p @ right, p @ up
    ty = np.tan(np.radians(15.0)); tx = ty * W / H
    px = ((xc / zc / tx) + 1) * W / 2
    py = (1 - (yc / zc / ty)) * H / 2
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    idx = py[ok].astype(int) * W + px[ok].astype(int)
    depth = np.full(W * H, np.inf)
    np.minimum.at(depth, idx, zc[ok])
    d = depth.reshape(H, W)
    m = np.isfinite(d)
    dd = np.where(m, d, np.nanmax(np.where(m, d, np.nan)) if m.any() else 0)
    gy, gx = np.gradient(dd)
    n = np.stack([-gx * W / (2 * tx * dd.clip(1e-6)), gy * H / (2 * ty * dd.clip(1e-6)), np.ones_like(dd)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    light = np.array([-0.5, 0.6, 0.62]); light /= np.linalg.norm(light)
    shade = np.clip(n @ light, 0, 1) * 0.75 + 0.25
    img = np.where(m, shade * 220, 40).astype(np.uint8)
    im = Image.fromarray(img, "L").convert("RGB")
    ImageDraw.Draw(im).text((8, 8), f"az {az:g} el {el:g}", fill=(255, 200, 80))
    tiles.append(im)
strip = Image.new("RGB", (W * len(tiles), H))
for k, t in enumerate(tiles):
    strip.paste(t, (k * W, 0))
strip.save(out)
print("wrote", out)
