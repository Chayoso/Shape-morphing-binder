"""montage_ppt.py OUT_JPG COLS TILE NAME[=LABEL] [...] -- FIGS_PPT (2026-10-23): the 300k gallery montage (Figure 4.7) in
the user's PowerPoint style: per example its start, middle (the first kept frame with progress >= 0.5) and end, the
bodies cut from the studio stills (SHOWCASE's or FIGS_V2's, hflipped) and set on one white strip with a drop shadow,
shorter than the bodies (pop-out); the example's name in bold serif above, the frame times in small italics."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

import ppt

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
SC, FV = G / "showcase", G / "figs_v2"
out, cols, tile = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
specs = [(s.split("=", 1) + [None])[:2] for s in sys.argv[4:]]
tw = tile * 4 // 3
lab_h, gap, cgap = int(tile * .30), int(tile * .16), int(tile * .22)
cell_w, cell_h = 3 * tw + cgap, tile + lab_h + gap
rows = (len(specs) + cols - 1) // cols
img = ppt.canvas(cols * cell_w + cgap, rows * cell_h + gap)
d = ImageDraw.Draw(img)


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
    x0 = cgap + (i % cols) * cell_w
    y0 = gap + (i // cols) * cell_h
    wt = ppt.rich(d, (x0 + 4, y0 + lab_h * .62), [label or name], tile * .118, fill=ppt.INK, bold=True)
    ppt.rich(d, (x0 + 4 + wt + tile * .06, y0 + lab_h * .62), [" · ".join(c for _, c in picks)], tile * .095,
             fill=ppt.GREY, italic=True)
    ty = y0 + lab_h
    ppt.shadow_card(img, (x0, ty + .30 * tile, x0 + 3 * tw, ty + .86 * tile), radius=8, blur=10, offset=4)
    for j, (k, cap) in enumerate(picks):
        body = ppt.cut_body(square(sd / f"{k:04d}.jpg"))
        x = x0 + j * tw
        if body is not None:
            ppt.contact_shadow(img, (x, ty, x + tw, ty + tile), body.split()[3], strength=.3)
            img.alpha_composite(body, (x, ty))
    print(name, sd, [k for k, _ in picks], flush=True)
ppt.save(img, out, quality=92)
