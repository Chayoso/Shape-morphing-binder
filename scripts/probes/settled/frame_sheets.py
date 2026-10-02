"""frame_sheets.py NPZ RENDER_JSON FRAMES_DIR OUT_DIR [TOP_FRAC=0.72] — contact sheets of a 4K render for frame-by-frame
viewing: (1) the whole frame, every rendered frame in order, 12 per sheet (640 px wide each, labelled with the rendered
index and the raw frame); (2) the top region of the target (the ears on the bunny: target points above TOP_FRAC of the
height, projected with the renderer's camera) cropped at full 4K resolution, 12 per sheet."""
import json, math, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph.render.covariance_torch import world_to_view_torch  # noqa: E402

W, H, AZ, EL = 3840, 2160, 35.0, 18.0
z = np.load(sys.argv[1], allow_pickle=True)
rj = json.loads(Path(sys.argv[2]).read_text())
fdir, out = Path(sys.argv[3]), Path(sys.argv[4])
top = float(sys.argv[5]) if len(sys.argv) > 5 else 0.72
out.mkdir(parents=True, exist_ok=True)
raws = rj["raw_frame_indices"]
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32))
center = tgt.mean(0)
radius = float((tgt - center).norm(dim=1).max())
az, el = math.radians(AZ), math.radians(EL)
cam = center + 3.6 * radius * center.new_tensor((math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)))
view = world_to_view_torch(cam, center)
tan_y = math.tan(math.radians(30) / 2); tan_x = tan_y * W / H
y0, y1 = float(tgt[:, 1].min()), float(tgt[:, 1].max())
ears = tgt[tgt[:, 1] > y0 + top * (y1 - y0)]
p = ears @ view[:3, :3].T + view[:3, 3]
px = (p[:, 0] / (p[:, 2] * tan_x) + 1) / 2 * W
py = (1 - p[:, 1] / (p[:, 2] * tan_y)) / 2 * H
m = 90
box = [max(0, int(px.min()) - m), max(0, int(py.min()) - m), min(W, int(px.max()) + m), min(H, int(py.max()) + m)]
print(f"{len(raws)} rendered frames; top-region crop box (4K pixels) {box}, {box[2] - box[0]} x {box[3] - box[1]}")


def sheet(tiles, cols, name):
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    s = Image.new("RGB", (cols * w, rows * h), (20, 20, 20))
    for i, t in enumerate(tiles):
        s.paste(t, ((i % cols) * w, (i // cols) * h))
    s.save(out / name, quality=92)


full, crop = [], []
n_full = n_crop = 0
for i, raw in enumerate(raws):
    img = Image.open(fdir / f"{i:04d}.png").convert("RGB")
    t = img.resize((640, 360), Image.LANCZOS)
    ImageDraw.Draw(t).text((6, 4), f"#{i} raw {raw}", fill=(255, 255, 0))
    full.append(t)
    c = img.crop(box)
    if c.width > 960:                                   # keep the crop at full resolution unless it is very wide
        c = c.resize((960, int(c.height * 960 / c.width)), Image.LANCZOS)
    ImageDraw.Draw(c).text((6, 4), f"#{i} raw {raw}", fill=(255, 255, 0))
    crop.append(c)
    if len(full) == 12 or i == len(raws) - 1:
        sheet(full, 3, f"full_{n_full:02d}.jpg"); n_full += 1; full = []
    if len(crop) == 6 or i == len(raws) - 1:
        sheet(crop, 3, f"top_{n_crop:02d}.jpg"); n_crop += 1; crop = []
print(f"wrote {n_full} full sheets and {n_crop} top-region sheets to {out}")
