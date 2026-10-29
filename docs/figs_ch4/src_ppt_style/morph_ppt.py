"""morph_ppt.py OUT KS LABEL=FRAMES_DIR[:noflip] ... -- FIGS_PPT (2026-10-23): Figure 4.2 in the user's PowerPoint style.
The same frames and crops as compose_morph.py (one crop per row over its frames, the studio raster's images flipped
back), each body cut from the studio backdrop and set on a white strip with a drop shadow that is shorter than the
bodies (pop-out rows); bold serif time headers, italic rotated mesh labels, the end time in small italics."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import ppt

out, ks_s = sys.argv[1], sys.argv[2].split(",")
rows = [s.split("=", 1) for s in sys.argv[3:]]
TW, TH, GAP, LEFT, TOP, RG = 640, 480, 10, 120, 110, 26
ASPECT = TW / TH


def body_box(im):
    a = np.asarray(im.convert("RGB")).astype(np.int16)
    bg = np.median(np.concatenate([a[:, :120], a[:, -120:]], 1), axis=1)
    m = np.abs(a - bg[:, None, :]).max(-1) > 18
    ys, xs = np.where(m)
    return xs.min(), ys.min(), xs.max(), ys.max()


def frames_of(d):
    fs = sorted(Path(d).glob("*.jpg"))
    return [fs[-1] if k == "last" else Path(d) / f"{int(k):04d}.jpg" for k in ks_s]


ncol = len(ks_s)
W = LEFT + ncol * TW + (ncol - 1) * GAP + 50
H = TOP + len(rows) * (TH + RG) + 10
img = ppt.canvas(W, H)
d = ImageDraw.Draw(img)
for j, k in enumerate(ks_s):
    cx = LEFT + j * (TW + GAP) + TW / 2
    txt = "end" if k == "last" else f"t = {int(k) * 0.05:g} s"
    ppt.rich(d, (cx, TOP - 40), [txt], 50, fill=ppt.INK, anchor="m", bold=True)
for i, (label, dd) in enumerate(rows):
    flip = not dd.endswith(":noflip")
    dd = dd.removesuffix(":noflip")
    paths = frames_of(dd)
    ims = [Image.open(p).transpose(Image.FLIP_LEFT_RIGHT) if flip else Image.open(p) for p in paths]
    boxes = np.array([body_box(im) for im in ims])
    x0, y0, x1, y1 = boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()
    w, h = (x1 - x0) * 1.10, (y1 - y0) * 1.10
    w, h = (h * ASPECT, h) if w / h < ASPECT else (w, w / ASPECT)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    W0, H0 = ims[0].size
    w, h = min(w, W0), min(h, H0)
    w, h = (h * ASPECT, h) if w / h > ASPECT else (w, w / ASPECT)
    l = int(round(min(max(cx - w / 2, 0), W0 - w))); t = int(round(min(max(cy - h / 2, 0), H0 - h)))
    box = (l, t, l + int(round(w)), t + int(round(h)))
    y = TOP + i * (TH + RG)
    x_end = LEFT + ncol * TW + (ncol - 1) * GAP
    ppt.shadow_card(img, (LEFT - 14, y + .30 * TH, x_end + 14, y + .84 * TH), radius=10)
    ppt.vlabel(img, (LEFT - 52, y + TH / 2), label, 46)
    for j, (im, p) in enumerate(zip(ims, paths)):
        tile = im.crop(box).resize((TW, TH), Image.LANCZOS)
        body = ppt.cut_body(tile)
        x = LEFT + j * (TW + GAP)
        if body is not None:
            ppt.contact_shadow(img, (x, y, x + TW, y + TH), body.split()[3])
            img.alpha_composite(body, (x, y))
        if ks_s[j] == "last":
            ppt.rich(d, (x + TW - 12, y + .84 * TH - 14), [f"{int(p.stem) * 0.05:.1f} s"], 34, fill=ppt.GREY,
                     anchor="r", italic=True)
ppt.save(img, out)
