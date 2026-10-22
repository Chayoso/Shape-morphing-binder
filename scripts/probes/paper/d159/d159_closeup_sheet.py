"""d159_closeup_sheet.py DIR -- D159: one sheet per close-up (bunny ears, bunny wrinkles, dragon head, dragon scales): the
tag's render arm above, D159's below, at t = 1 / 2 / 4 s and the end (kept frames 20, 40, 80, last), the central part of
each 4K studio frame. Both rows are "dF_c + layer u + render loss (full)"; D159's render target pictures carry the mesh's
relief."""
import glob
import os
import sys

import matplotlib
from PIL import Image, ImageDraw, ImageFont

D = sys.argv[1]
FONT = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
big, small = ImageFont.truetype(FONT, 30), ImageFont.truetype(FONT, 24)
INK = (40, 40, 40)
TITLE = {"bunny_ears": "bunny 300k: the ears", "bunny_wrinkles": "bunny 300k: the wrinkled haunch",
         "dragon_head": "dragon 300k: the head and horns", "dragon_scales": "dragon 300k: the belly scutes",
         "bunny_wrinkles25": "DISPLAY READING: bunny wrinkles drawn through the r = 2.5 field (D123), not the r = 3 the loss reads",
         "dragon_scales25": "DISPLAY READING: dragon scutes drawn through the r = 2.5 field (D123), not the r = 3 the loss reads"}
ROWS = (("tag", "tag: render target = 300k samples"), ("d159", "D159: render target = mesh relief"))
for part, title in TITLE.items():
    cells = {}
    for arm, _ in ROWS:
        for d in glob.glob(f"{D}/{arm}_{part}_*"):
            if os.path.isdir(d):
                k = int(d.rsplit("_", 1)[1])
                f = glob.glob(f"{d}/frames/*.jpg")
                if f:
                    cells[(arm, k)] = f[0]
    ks = sorted({k for _, k in cells})
    if not ks:
        continue
    cw, ch = 720, 540
    sheet = Image.new("RGB", (cw * len(ks), ch * 2 + 70), (255, 255, 255))
    for r, (arm, name) in enumerate(ROWS):
        arm_ks = sorted(k for a, k in cells if a == arm)
        for c, k in enumerate(arm_ks):
            im = Image.open(cells[(arm, k)]).convert("RGB")
            W, H = im.size
            im = im.crop((W // 2 - 1080, H // 2 - 810, W // 2 + 1080, H // 2 + 810)).resize((cw, ch), Image.LANCZOS)
            sheet.paste(im, (c * cw, 70 + r * ch))
            lab = f"t = {k / 20:g} s" if k in (20, 40, 80) else f"end ({k / 20:.1f} s)"
            ImageDraw.Draw(sheet).text((c * cw + 10, 70 + r * ch + 8), lab, fill=INK, font=small)
        ImageDraw.Draw(sheet).text((10, 70 + r * ch + ch - 36), name, fill=INK, font=small)
    ImageDraw.Draw(sheet).text((10, 18), f"{title}; top the tag, bottom D159 (both the full arm), studio look", fill=INK, font=big)
    sheet.save(f"{D}/{part}.jpg", quality=92)
    print("wrote", part, len(cells))
