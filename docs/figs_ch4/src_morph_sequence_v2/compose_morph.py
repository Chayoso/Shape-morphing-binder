"""compose_morph.py OUT_JPG KS LABEL=FRAMES_DIR [...] -- FIGS_V2 (CPU): the dissertation's Figure 4.2 v2 from the stills of
fig_render.py (FRAMES_DIR/NNNN.jpg, the studio raster's own mirrored images, flipped back left-right here as SHOWCASE
does at its encoding). One row per LABEL, one column per kept frame of KS ("last" = the row's last kept frame; the same
indices for every row, a kept frame = 0.05 s). Per row one crop for all its columns (the union of the body over the
row's frames, found against the backdrop, padded, at one aspect for the whole figure), so a row's frames share their
scale. Column heads give the time; the end tile carries its own time."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/fonts")
import tnr  # noqa: E402  (Times New Roman, the user 2026-10-22)

out, ks_s = sys.argv[1], sys.argv[2].split(",")
rows = [s.split("=", 1) for s in sys.argv[3:]]          # DIR ending in ":noflip": stills already mirrored back (SHOWCASE)
TW, TH, GAP, LABW, HEADH = 640, 480, 6, 345, 92
ASPECT = TW / TH
try:
    font = tnr.pil(64)
    small = tnr.pil(44)
except OSError:
    font = small = ImageFont.load_default()


def body_box(im):
    a = np.asarray(im.convert("RGB")).astype(np.int16)
    bg = np.median(np.concatenate([a[:, :120], a[:, -120:]], 1), axis=1)       # the backdrop: a vertical gradient
    m = np.abs(a - bg[:, None, :]).max(-1) > 18
    ys, xs = np.where(m)
    return xs.min(), ys.min(), xs.max(), ys.max()


def frames_of(d):
    fs = sorted(Path(d).glob("*.jpg"))
    last = fs[-1]
    pick = []
    for k in ks_s:
        p = last if k == "last" else Path(d) / f"{int(k):04d}.jpg"
        pick.append(p)
    return pick


ncol = len(ks_s)
W = LABW + ncol * (TW + GAP)
H = HEADH + len(rows) * (TH + GAP)
sheet = Image.new("RGB", (W, H), (255, 255, 255))
dr = ImageDraw.Draw(sheet)
for j, k in enumerate(ks_s):
    txt = "end" if k == "last" else f"t = {int(k) * 0.05:g} s"
    x = LABW + j * (TW + GAP) + TW // 2
    dr.text((x, HEADH // 2), txt, fill=(20, 20, 20), font=font, anchor="mm")
for i, (label, d) in enumerate(rows):
    flip = not d.endswith(":noflip")
    d = d.removesuffix(":noflip")
    paths = frames_of(d)
    ims = [Image.open(p).transpose(Image.FLIP_LEFT_RIGHT) if flip else Image.open(p) for p in paths]   # mirrored back
    boxes = np.array([body_box(im) for im in ims])
    x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
    x1, y1 = boxes[:, 2].max(), boxes[:, 3].max()
    w, h = (x1 - x0) * 1.10, (y1 - y0) * 1.10
    if w / h < ASPECT:
        w = h * ASPECT
    else:
        h = w / ASPECT
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    W0, H0 = ims[0].size
    w, h = min(w, W0), min(h, H0)
    if w / h < ASPECT - 1e-6:
        h = w / ASPECT
    elif w / h > ASPECT + 1e-6:
        w = h * ASPECT
    l = int(round(min(max(cx - w / 2, 0), W0 - w)))
    t = int(round(min(max(cy - h / 2, 0), H0 - h)))
    box = (l, t, l + int(round(w)), t + int(round(h)))
    y = HEADH + i * (TH + GAP)
    dr.text((LABW - 14, y + TH // 2), label, fill=(20, 20, 20), font=font, anchor="rm")
    for j, (im, p) in enumerate(zip(ims, paths)):
        tile = im.crop(box).resize((TW, TH), Image.LANCZOS)
        if ks_s[j] == "last":
            ImageDraw.Draw(tile).text((TW - 14, TH - 12), f"{int(p.stem) * 0.05:.1f} s", fill=(235, 235, 235),
                                      font=small, anchor="rb")
        sheet.paste(tile, (LABW + j * (TW + GAP), y))
    print(label, [p.name for p in paths], "crop", box, flush=True)
sheet.save(out, quality=92)
print("wrote", out, sheet.size)
