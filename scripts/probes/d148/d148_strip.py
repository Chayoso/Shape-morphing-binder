"""d148_strip.py OUT.jpg LABEL=FRAMES_DIR[,LABEL=FRAMES_DIR...] I0,I1,... [crop=x0,y0,x1,y1] [w=W] -- a contact sheet:
one row per run, one column per kept frame index (20 a second), each tile the frame (cropped, scaled to width W), its
run label and time written in the corner."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

out = sys.argv[1]
runs = [s.split("=", 1) for s in sys.argv[2].split(",")]
idx = [int(v) for v in sys.argv[3].split(",")]
opt = dict(s.split("=", 1) for s in sys.argv[4:])
crop = tuple(int(v) for v in opt["crop"].split(",")) if "crop" in opt else None
W = int(opt.get("w", 640))
tiles = []
for lab, d in runs:
    row = []
    for i in idx:
        p = Path(d) / f"{i:04d}.jpg"
        if not p.exists():
            row.append(None)
            continue
        im = Image.open(p).convert("RGB")
        if crop:
            im = im.crop(crop)
        im = im.resize((W, int(im.height * W / im.width)))
        ImageDraw.Draw(im).text((8, 8), f"{lab} {i / 20:.2f} s", fill=(0, 0, 0))
        row.append(im)
    tiles.append(row)
H = max(t.height for r in tiles for t in r if t is not None)
sheet = Image.new("RGB", (W * len(idx), H * len(runs)), (255, 255, 255))
for r, row in enumerate(tiles):
    for c, t in enumerate(row):
        if t is not None:
            sheet.paste(t, (c * W, r * H))
sheet.save(out, quality=90)
print(out, sheet.size)
