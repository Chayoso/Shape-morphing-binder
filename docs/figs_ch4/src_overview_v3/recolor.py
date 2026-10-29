"""recolor.py OUT_DIR SHELL(0/1) NAME=STILL.jpg[:flip] ... -- FIGS_V2 (CPU): the overview's matte blue bodies from the
studio stills: the body segmented from the grey backdrop (difference to the per-row backdrop, holes filled, erode 2 px,
feathered), recoloured by a luminance-preserving matte blue ramp close to the user's reference, a soft elliptical contact
shadow, optionally a thin translucent light-blue shell (the exterior's discs), cropped to the body, written as RGBA PNG."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
shell = sys.argv[2] == "1"
DARK, MID, LIGHT = np.array([12, 34, 66.]), np.array([36, 92, 146.]), np.array([118, 168, 214.])
for spec in sys.argv[3:]:
    name, path = spec.split("=", 1)
    flip = path.endswith(":flip"); path = path.removesuffix(":flip")
    im = Image.open(path).convert("RGB")
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    a = np.asarray(im).astype(float)
    bg = np.median(np.concatenate([a[:, :150], a[:, -150:]], 1), axis=1)
    m = np.abs(a - bg[:, None, :]).max(-1) > 16
    m = ndimage.binary_opening(m, iterations=2)
    lab, n = ndimage.label(m)
    keep = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    m = ndimage.binary_fill_holes(lab == keep)
    m = ndimage.binary_erosion(m, iterations=2)
    alpha = ndimage.gaussian_filter(m.astype(float), 1.2)
    L = a.mean(-1) / 255.0
    t = np.clip((L - 0.30) / 0.55, 0, 1) ** 1.1
    col = np.where(t[..., None] < .5, DARK + (MID - DARK) * (t[..., None] / .5), MID + (LIGHT - MID) * ((t[..., None] - .5) / .5))
    H, W = m.shape
    ys, xs = np.where(m)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    canvas = np.zeros((H, W, 4))
    # contact shadow
    sh = Image.new("L", (W, H), 0)
    w = x1 - x0
    ImageDraw.Draw(sh).ellipse((x0 + .08 * w, y1 - .03 * (y1 - y0), x1 - .08 * w, y1 + .07 * (y1 - y0)), fill=120)
    sh = np.asarray(sh.filter(ImageFilter.GaussianBlur(0.035 * w))).astype(float) / 255
    canvas[..., :3] = np.array([30, 36, 46.]); canvas[..., 3] = sh
    if shell:                                               # the exterior's discs: a thin translucent shell
        ring = ndimage.binary_dilation(m, iterations=max(4, int(.012 * w))) & ~m
        ra = ndimage.gaussian_filter(ring.astype(float), 1.0) * .55
        canvas[..., :3] = np.where(ra[..., None] > canvas[..., 3:4] * 0, (canvas[..., :3] * canvas[..., 3:4] * (1 - ra[..., None]) + np.array([120, 175, 235.]) * ra[..., None]) / np.maximum(canvas[..., 3:4] * (1 - ra[..., None]) + ra[..., None], 1e-6), canvas[..., :3])
        canvas[..., 3] = canvas[..., 3] * (1 - ra) + ra
    # body over
    A = alpha[..., None]
    rgb = (col * A + canvas[..., :3] * canvas[..., 3:4] * (1 - A)) / np.maximum(A + canvas[..., 3:4] * (1 - A), 1e-6)
    alp = A[..., 0] + canvas[..., 3] * (1 - A[..., 0])
    res = np.dstack([rgb, alp * 255]).clip(0, 255).astype(np.uint8)
    pad = int(.06 * max(w, y1 - y0))
    box = (max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(int(y1 + .12 * (y1 - y0)) + pad, H))
    Image.fromarray(res, "RGBA").crop(box).save(out / f"{name}.png")
    print(name, box)
