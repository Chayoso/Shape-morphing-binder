"""d148_tracks.py TRACKS_JSON FRAMES_DIR OUT.jpg T0 T1 [pad=80] [w=320] -- D148 item (1): the detector's (d140_image.py)
pieces seen apart from the body whose track overlaps T0-T1 s, each cropped around its box at its largest frame, labelled
with the track, its time span, its largest area (px) and whether particles apart from the body project into it (cls i);
a contact sheet to check every candidate by eye. Prints one line per track."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

tr = json.load(open(sys.argv[1]))
fdir, out = Path(sys.argv[2]), sys.argv[3]
t0, t1 = float(sys.argv[4]), float(sys.argv[5])
opt = dict(s.split("=", 1) for s in sys.argv[6:])
pad, W = int(opt.get("pad", 80)), int(opt.get("w", 320))
tiles = []
for k, t in sorted(tr.items(), key=lambda kv: kv[1]["t_first"]):
    if t["t_last"] < t0 or t["t_first"] > t1:
        continue
    x0, y0, x1, y1 = t["bbox"]
    if y1 > float(opt.get("ymax", 1e9)) or t["max_px"] < float(opt.get("minpx", 0)):
        continue
    print(f"track {k}: {t['t_first']:.2f}-{t['t_last']:.2f} s, {t['frames']} frames, max {t['max_px']} px at frame "
          f"{t['k_max']}, cls_i frames {t['cls_i']}, box {t['bbox']}")
    p = fdir / f"{t['k_max']:04d}.jpg"
    if not p.exists():
        continue
    im = Image.open(p).convert("RGB")
    c = (max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad))
    im = im.crop(c)
    s = W / max(im.width, im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
    tile = Image.new("RGB", (W, W + 30), (255, 255, 255))
    tile.paste(im, (0, 30))
    ImageDraw.Draw(tile).text((4, 4), f"#{k} {t['t_first']:.2f}-{t['t_last']:.2f}s {t['max_px']}px i{t['cls_i']}",
                              fill=(0, 0, 0))
    tiles.append(tile)
if tiles:
    n = len(tiles)
    cols = min(n, 8)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * W, rows * (W + 30)), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * W, (i // cols) * (W + 30)))
    sheet.save(out, quality=90)
print(f"{len(tiles)} tracks in {t0}-{t1} s -> {out if tiles else 'none'}")
