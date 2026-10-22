"""d159_sbs.py LEFT_DIR RIGHT_DIR OUT_DIR TITLE -- D159: side-by-side frames (1920 x 1080) and a contact sheet from two
render_deliverable.py frame folders (4K studio frames NNNN.jpg, 20 kept frames a second; D155's d155_sbs.py with D159's
labels and the whole run): the central half of each frame scaled to 960 x 1080; left the tag's render arm, right D159's
(both "dF_c + layer u + render loss (full)"; D159's render target pictures carry the mesh's relief); a run that has ended
holds its last frame; a 1 s hold at the end. Contact sheet at t = 0.5 / 1 / 2 / 4 s and each run's end."""
import glob
import os
import sys

import matplotlib
from PIL import Image, ImageDraw, ImageFont

L, R, OUT, TITLE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(f"{OUT}/sbs", exist_ok=True)
FONT = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
big, small = ImageFont.truetype(FONT, 34), ImageFont.truetype(FONT, 26)
INK = (40, 40, 40)
NAMES = ("tag: render target = 300k samples", "D159: render target = mesh relief")


def frames(d):
    return {int(os.path.basename(p)[:-4]): p for p in glob.glob(f"{d}/frames/*.jpg")}


fl, fr = frames(L), frames(R)


def half(p, w=960, h=1080):
    im = Image.open(p).convert("RGB")
    W, H = im.size
    return im.crop((W // 4, 0, W - W // 4, H)).resize((w, h), Image.LANCZOS)


def at(fs, k):
    return fs[min(fs, key=lambda q: (abs(q - k), q))] if k <= max(fs) else fs[max(fs)]


def label(img, text, xy, font):
    ImageDraw.Draw(img).text(xy, text, fill=INK, font=font)


k_max = max(max(fl), max(fr))
seq = list(range(0, k_max + 1)) + [k_max] * 20
for i, k in enumerate(seq):
    c = Image.new("RGB", (1920, 1080), (255, 255, 255))
    c.paste(half(at(fl, k)), (0, 0))
    c.paste(half(at(fr, k)), (960, 0))
    ImageDraw.Draw(c).line([(960, 0), (960, 1080)], fill=(200, 200, 200), width=2)
    label(c, NAMES[0], (30, 24), big)
    label(c, NAMES[1], (990, 24), big)
    label(c, f"{TITLE}   t = {k / 20:.2f} s" + ("  (left ended)" if k > max(fl) else "") + ("  (right ended)" if k > max(fr) else ""),
          (30, 1030), small)
    c.save(f"{OUT}/sbs/{i:04d}.jpg", quality=92)
times = [0.5, 1.0, 2.0, 4.0, None]
cw, ch = 480, 540
sheet = Image.new("RGB", (cw * len(times), ch * 2 + 60), (255, 255, 255))
for row, (fs, name) in enumerate(((fl, NAMES[0]), (fr, NAMES[1]))):
    for col, t in enumerate(times):
        k = max(fs) if t is None else int(round(t * 20))
        sheet.paste(half(at(fs, k), cw, ch), (col * cw, 60 + row * ch))
        label(sheet, "end" + f" ({max(fs) / 20:.1f} s)" if t is None else f"t = {t:g} s", (col * cw + 12, 60 + row * ch + 8), small)
    label(sheet, name, (12, 60 + row * ch + ch - 40), small)
label(sheet, f"{TITLE}: t = 0.5 / 1 / 2 / 4 s and the end; top the tag, bottom D159 (both the full arm)", (12, 12), small)
sheet.save(f"{OUT}/contact_sheet.jpg", quality=92)
print("written", OUT, len(seq), "frames")
