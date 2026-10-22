"""sc_montage.py OUT_JPG COLS TILE NAME[=LABEL] [...] -- SHOWCASE: the gallery montage (CPU only). Per example its
start, middle (the first kept frame with progress >= 0.5, mid/NAME.json) and end, from the render's stills
(output/gpu/showcase/stills/NAME_main/NNNN.jpg), each cropped to the centred 4:3 part of the 16:9 frame (the studio
camera frames the body's bounding sphere to the picture's height) and scaled to TILE pixels; the three side by side
under the example's label, COLS examples a row."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

D = Path("/data/relcfd/chayo/physmorph_v2/output/gpu/showcase")
out, cols, tile = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
specs = [(s.split("=", 1) + [None])[:2] for s in sys.argv[4:]]
try:
    font = ImageFont.truetype("DejaVuSans.ttf", max(14, tile // 11))
    small = ImageFont.truetype("DejaVuSans.ttf", max(12, tile // 15))
except OSError:
    font = small = ImageFont.load_default()
lab_h, gap = tile // 6, tile // 12
tw = tile * 4 // 3                      # a 4:3 crop of each 16:9 frame (the wide bodies, cow and fandisk, fit)
cell_w, cell_h = 3 * tw + gap, tile + lab_h + gap
rows = (len(specs) + cols - 1) // cols
sheet = Image.new("RGB", (cols * cell_w + gap, rows * cell_h + gap), (22, 24, 28))
dr = ImageDraw.Draw(sheet)


def square(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    c = h * 4 // 3
    return im.crop(((w - c) // 2, 0, (w + c) // 2, h)).resize((tw, tile), Image.LANCZOS)


for i, (name, label) in enumerate(specs):
    sd = D / "stills" / f"{name}_main"
    ks = sorted(int(p.stem) for p in sd.glob("*.jpg"))
    mid = json.load(open(D / "mid" / f"{name}.json"))["mid"]
    t_mid = mid / 20.0
    picks = [(0, "start"), (mid, f"t = {t_mid:.2f} s"), (ks[-1], f"end, {ks[-1] / 20:.1f} s")]
    x0 = gap + (i % cols) * cell_w
    y0 = gap + (i // cols) * cell_h
    dr.text((x0, y0), label or (D / f"title_{name}").read_text().strip(), fill=(235, 235, 235), font=font)
    for j, (k, cap) in enumerate(picks):
        sheet.paste(square(sd / f"{k:04d}.jpg"), (x0 + j * tw, y0 + lab_h))
        dr.text((x0 + j * tw + 8, y0 + lab_h + 6), cap, fill=(255, 255, 255), font=small)
sheet.save(out, quality=92)
print("wrote", out, sheet.size)
