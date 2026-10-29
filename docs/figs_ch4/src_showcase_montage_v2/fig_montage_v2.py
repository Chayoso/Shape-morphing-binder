"""fig_montage_v2.py OUT_JPG COLS TILE NAME[=LABEL] [...] -- FIGS_V2 (CPU): SHOWCASE's gallery montage (sc_montage.py,
unchanged layout and look) with examples from two places: SHOWCASE's (output/gpu/showcase/stills/NAME_main/NNNN.jpg and
mid/NAME.json) and FIGS_V2's new runs (output/gpu/figs_v2/render/NAME/stills/NNNN.jpg and figs_v2/mid/NAME.json). Both
sets are hflipped stills of the same renderer (render_deliverable.py's base display, studio look, 3840 x 2160). Per
example its start, middle (the first kept frame with progress >= 0.5) and end, each cropped to the centred 4:3 part."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/fonts")
import tnr  # noqa: E402  (Times New Roman, the user 2026-10-22)

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
SC, FV = G / "showcase", G / "figs_v2"
out, cols, tile = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
specs = [(s.split("=", 1) + [None])[:2] for s in sys.argv[4:]]
try:
    font = tnr.pil(max(15, tile // 10))
    small = tnr.pil(max(13, tile // 13))
except OSError:
    font = small = ImageFont.load_default()
lab_h, gap = tile // 6, tile // 12
tw = tile * 4 // 3
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
    if (SC / "stills" / f"{name}_main").is_dir():
        sd, mj = SC / "stills" / f"{name}_main", SC / "mid" / f"{name}.json"
    else:
        sd, mj = FV / "render" / name / "stills", FV / "mid" / f"{name}.json"
    ks = sorted(int(p.stem) for p in sd.glob("*.jpg"))
    mid = json.load(open(mj))["mid"]
    picks = [(0, "start"), (mid, f"t = {mid / 20.0:.2f} s"), (ks[-1], f"end, {ks[-1] / 20:.1f} s")]
    x0 = gap + (i % cols) * cell_w
    y0 = gap + (i // cols) * cell_h
    dr.text((x0, y0), label or name, fill=(235, 235, 235), font=font)
    for j, (k, cap) in enumerate(picks):
        sheet.paste(square(sd / f"{k:04d}.jpg"), (x0 + j * tw, y0 + lab_h))
        dr.text((x0 + j * tw + 8, y0 + lab_h + 6), cap, fill=(255, 255, 255), font=small)
    print(name, sd, "frames", [k for k, _ in picks], flush=True)
sheet.save(out, quality=92)
print("wrote", out, sheet.size)
