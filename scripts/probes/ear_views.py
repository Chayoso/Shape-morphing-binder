"""Ear/head notch check with the GIF's own splat renderer: target vs mid vs end frame at three
views (az 0.6 = the gallery view where the two ears overlap in projection; az 1.4 = the notch
between the ears and the head is visible; top-down). Also prints the over-mass beyond absolute
distances from the target surface (0.15 / 0.25 / 0.4 wu) at mid and end.
Usage: python ear_views.py OUT_DIR run mid_fraction"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.expanduser("~/physmorph_v2"))
from scripts.make_gif import splat, shade  # noqa: E402

OUT, name, MID = sys.argv[1], sys.argv[2], float(sys.argv[3])
z = np.load(os.path.join(OUT, name + ".npz"))
fr, tgt = z["frames"], z["tgt"]
dn = int(z["deliver_n"]) if "deliver_n" in z.files else len(fr)
fr = fr[:dn]
ttree = cKDTree(tgt)
ext = float(np.abs(tgt).max()) * 1.08
RES = 300
views = [("az 0.6 (gallery)", 0.6, 0.18), ("az 1.4 (notch visible)", 1.4, 0.18), ("top-down", 0.6, 1.25)]
frames = [("target", tgt), (f"mid frame {int(MID*(dn-1))}", fr[int(MID * (dn - 1))]), (f"end frame {dn-1}", fr[dn - 1])]

for label, x in frames[1:]:
    d = ttree.query(x, k=1, workers=-1)[0]
    print(f"{label}: particles beyond 0.15 / 0.25 / 0.40 wu of the target surface: "
          f"{(d > 0.15).mean():.3%} / {(d > 0.25).mean():.3%} / {(d > 0.40).mean():.4%}  (N={len(x)})")


def render(x, az, el):
    cov, dep = splat(x.astype(np.float32), RES, az, el=el, extent=ext)
    tcov, _ = splat(tgt.astype(np.float32), RES, az, el=el, extent=ext)
    img = shade(cov, dep, outline=tcov > 0)          # same renderer as the gallery GIFs
    img = np.ascontiguousarray(img.transpose(1, 0, 2)[::-1])   # (i=horizontal, j=vertical) -> rows top-down
    return Image.fromarray(img)


tile = None
sheet = Image.new("RGB", (RES * len(views), RES * len(frames)), "white")
for r, (fl, x) in enumerate(frames):
    for c, (vl, az, el) in enumerate(views):
        im = render(x, az, el).convert("RGB")
        ImageDraw.Draw(im).text((4, 4), f"{fl} — {vl}", fill=(20, 20, 20))
        sheet.paste(im, (c * RES, r * RES))
sheet.save(os.path.join(OUT, f"ear_views_{name}.png"))
print("saved")
