"""d159_closeup_sheet.py DIR -- D159: one sheet per close-up (bunny ears, bunny wrinkles, dragon head, dragon scales, and
the two r = 2.5 display readings): one row per arm present (the tag, D159, D159c), at t = 1 / 2 / 4 s and the end (kept
frames 20, 40, 80, last), the central part of each 4K studio frame. Every row is "dF_c + layer u + render loss (full)";
D159's render target pictures carry the mesh's relief, D159c also holds the render weight at its first-window value."""
import glob
import os
import sys

import matplotlib
from PIL import Image, ImageDraw, ImageFont

D = sys.argv[1]
FONT = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
big, small = ImageFont.truetype(FONT, 30), ImageFont.truetype(FONT, 24)
INK = (40, 40, 40)
TITLE = {"bunny_ears": ("bunny 300k: the ears", ""), "bunny_wrinkles": ("bunny 300k: the wrinkled haunch", ""),
         "dragon_head": ("dragon 300k: the head and horns", ""), "dragon_scales": ("dragon 300k: the belly scutes", ""),
         "bunny_wrinkles25": ("DISPLAY READING: bunny wrinkles, r = 2.5 field (D123)", "drawn through the D123 field, not the r = 3 field the loss reads"),
         "dragon_scales25": ("DISPLAY READING: dragon scutes, r = 2.5 field (D123)", "drawn through the D123 field, not the r = 3 field the loss reads")}
ARMS = (("tag", "tag: render target = 300k samples"), ("d159", "D159: render target = mesh relief"),
        ("d159c", "D159c: mesh relief + render weight held"))
for part, (title, note) in TITLE.items():
    cells = {}
    for arm, _ in ARMS:
        for d in glob.glob(f"{D}/{arm}_{part}_*"):
            if os.path.isdir(d) and d.rsplit("_", 1)[1].isdigit():
                k = int(d.rsplit("_", 1)[1])
                f = glob.glob(f"{d}/frames/*.jpg")
                if f:
                    cells[(arm, k)] = f[0]
    rows = [(a, n) for a, n in ARMS if any(x == a for x, _ in cells)]
    if not rows:
        continue
    ncol = max(sum(1 for x, _ in cells if x == a) for a, _ in rows)
    cw, ch, top = 720, 540, 140
    sheet = Image.new("RGB", (cw * ncol, ch * len(rows) + top), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for r, (arm, name) in enumerate(rows):
        for c, k in enumerate(sorted(k for a, k in cells if a == arm)):
            im = Image.open(cells[(arm, k)]).convert("RGB")
            W, H = im.size
            im = im.crop((W // 2 - 1080, H // 2 - 810, W // 2 + 1080, H // 2 + 810)).resize((cw, ch), Image.LANCZOS)
            sheet.paste(im, (c * cw, top + r * ch))
            lab = f"t = {k / 20:g} s" if k in (20, 40, 80) else f"end ({k / 20:.1f} s)"
            dr.text((c * cw + 10, top + r * ch + 8), lab, fill=INK, font=small)
        dr.text((10, top + r * ch + ch - 36), name, fill=INK, font=small)
    order = ", ".join(n.split(":")[0] for _, n in rows)
    dr.text((10, 12), title, fill=INK, font=big)
    dr.text((10, 54), f"rows: {order} (every row the full arm), studio look", fill=INK, font=small)
    if note:
        dr.text((10, 90), note, fill=INK, font=small)
    sheet.save(f"{D}/{part}.jpg", quality=92)
    print("wrote", part, len(cells), "rows", len(rows), "cols", ncol)
