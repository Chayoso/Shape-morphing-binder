"""d155_sbs.py LEFT_DIR RIGHT_DIR OUT_DIR -- D155 demo: the side-by-side frames (1920 x 1080) and the contact sheet from two
render_deliverable.py frame folders (4K studio frames NNNN.jpg, 20 kept frames a second): the central half of each
frame scaled to 960 x 1080, left "frozen", right "elastic-period ramp", a time label; frames 0..120 (6 s) and a 1 s hold
of the last; the contact sheet at t = 0.25 / 0.5 / 0.75 / 1 / 1.5 / 2 / 3 s and each run's end (its last kept frame)."""
import glob
import os
import sys

import matplotlib
from PIL import Image, ImageDraw, ImageFont

L, R, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(f"{OUT}/sbs", exist_ok=True)
FONT = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
big, small = ImageFont.truetype(FONT, 40), ImageFont.truetype(FONT, 30)
INK = (40, 40, 40)


def frames(d):
    return {int(os.path.basename(p)[:-4]): p for p in glob.glob(f"{d}/frames/*.jpg")}


fl, fr = frames(L), frames(R)


def half(p, w=960, h=1080):
    im = Image.open(p).convert("RGB")
    W, H = im.size
    return im.crop((W // 4, 0, W - W // 4, H)).resize((w, h), Image.LANCZOS)


def label(img, text, xy, font):
    ImageDraw.Draw(img).text(xy, text, fill=INK, font=font)


k_max = min(120, max(k for k in fl if k <= 120), max(k for k in fr if k <= 120))
seq = list(range(0, k_max + 1)) + [k_max] * 20
for i, k in enumerate(seq):
    c = Image.new("RGB", (1920, 1080), (255, 255, 255))
    c.paste(half(fl[k]), (0, 0))
    c.paste(half(fr[k]), (960, 0))
    ImageDraw.Draw(c).line([(960, 0), (960, 1080)], fill=(200, 200, 200), width=2)
    label(c, "frozen", (40, 30), big)
    label(c, "elastic-period ramp", (1000, 30), big)
    label(c, f"t = {k / 20:.2f} s", (40, 1020), small)
    c.save(f"{OUT}/sbs/{i:04d}.jpg", quality=92)
times = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, None]
cw, ch = 480, 540
sheet = Image.new("RGB", (cw * len(times), ch * 2 + 60), (255, 255, 255))
for row, (fs, name) in enumerate(((fl, "frozen"), (fr, "elastic-period ramp"))):
    for col, t in enumerate(times):
        k = max(fs) if t is None else int(round(t * 20))
        k = min(fs, key=lambda q: abs(q - k))
        sheet.paste(half(fs[k], cw, ch), (col * cw, 60 + row * ch))
        label(sheet, ("end" if t is None else f"t = {t:g} s") + ("" if t is not None else f" ({k / 20:.2f} s)"),
              (col * cw + 12, 60 + row * ch + 8), small)
    label(sheet, name, (12, 60 + row * ch + ch - 44), small)
label(sheet, "left to right: t = 0.25 / 0.5 / 0.75 / 1 / 1.5 / 2 / 3 s and the end; top frozen, bottom elastic-period ramp",
      (12, 12), small)
sheet.save(f"{OUT}/contact_sheet.jpg", quality=92)
print("written", OUT, len(seq), "frames")
