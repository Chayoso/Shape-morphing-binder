"""Projected coverage difference morph-vs-target with the gallery's own splat (fp=1, RES 300):
EXTRA = pixels the morph covers and the target does not (a web / bridge shows up here),
MISSING = target pixels the morph does not cover. Reported as a share of the target's
projected area, per view, for the mid and end frames; diff images saved (extra red, missing
blue, both grey). Also the same in 3D: particles with no target point within 0.15 wu and
target points with no particle within 0.15 wu, split into the ear region and the rest.
Usage: python cover_diff.py OUT_DIR run mid_fraction"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.expanduser("~/physmorph_v2"))
from scripts.make_gif import splat  # noqa: E402

OUT, name, MID = sys.argv[1], sys.argv[2], float(sys.argv[3])
z = np.load(os.path.join(OUT, name + ".npz"))
fr, tgt = z["frames"], z["tgt"]
dn = int(z["deliver_n"]) if "deliver_n" in z.files else len(fr)
fr = fr[:dn]
ext = float(np.abs(tgt).max()) * 1.08
RES = 300
views = [("az 0.6", 0.6, 0.18), ("az 1.4", 1.4, 0.18), ("az 2.2", 2.2, 0.18), ("top", 0.6, 1.25)]
frames = [(f"mid {int(MID*(dn-1))}", fr[int(MID * (dn - 1))]), (f"end {dn-1}", fr[dn - 1])]


def thin_mask(t, radius=0.9, q=0.22):
    cnt = np.array([len(c) for c in cKDTree(t).query_ball_point(t, radius, workers=-1)])
    r = np.linalg.norm(t - t.mean(0), axis=1)
    return (cnt <= np.quantile(cnt, q)) & (r > np.quantile(r, 0.5))


tm = thin_mask(tgt)
ttree, thin_tree = cKDTree(tgt), cKDTree(tgt[tm])
print(f"{name}: N={len(tgt)} target, ear/thin points {tm.mean():.1%}; 1 px = {2*ext/RES:.3f} wu")
for fl, x in frames:
    d = ttree.query(x, k=1, workers=-1)[0]
    d_thin = thin_tree.query(x, k=1, workers=-1)[0]
    near_ear = d_thin < 0.6
    extra3 = d > 0.15
    d_rev = cKDTree(x).query(tgt, k=1, workers=-1)[0]
    miss3 = d_rev > 0.15
    print(f"  {fl}: 3D extra (>0.15 wu from target) {extra3.mean():.3%} [near ear {extra3[near_ear].mean():.3%}] | "
          f"3D missing (target pts >0.15 wu from any particle) {miss3.mean():.2%} [ear {miss3[tm].mean():.2%}, body {miss3[~tm].mean():.2%}]")

sheet = Image.new("RGB", (RES * len(views), RES * len(frames)), "white")
for r, (fl, x) in enumerate(frames):
    for c, (vl, az, el) in enumerate(views):
        cm, _ = splat(x.astype(np.float32), RES, az, el=el, extent=ext)
        ct, _ = splat(tgt.astype(np.float32), RES, az, el=el, extent=ext)
        m, t = cm > 0, ct > 0
        t_fill = ndimage.binary_closing(t, iterations=2)          # close the target's sampling holes
        m_fill = ndimage.binary_closing(m, iterations=2)
        extra, miss = m_fill & ~t_fill, t_fill & ~m_fill
        # ear region in 2D: pixels within 12 px of a projected ear/thin point
        cthin, _ = splat(tgt[tm].astype(np.float32), RES, az, el=el, extent=ext)
        ear2d = ndimage.binary_dilation(cthin > 0, iterations=12)
        ta = t_fill.sum()
        print(f"  {fl} {vl}: EXTRA {extra.sum()/ta:.2%} of target area (in ear region {(extra & ear2d).sum()/ta:.2%}) | "
              f"MISSING {miss.sum()/ta:.2%} (ear region {(miss & ear2d).sum()/ta:.2%})")
        img = np.ones((RES, RES, 3), np.float32)
        img[t_fill & m_fill] = (0.80, 0.82, 0.86)
        img[extra] = (0.85, 0.15, 0.15)
        img[miss] = (0.20, 0.35, 0.85)
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8).transpose(1, 0, 2)[::-1].copy())
        ImageDraw.Draw(im).text((4, 4), f"{fl} — {vl}: red = morph only, blue = target only", fill=(20, 20, 20))
        sheet.paste(im, (c * RES, r * RES))
sheet.save(os.path.join(OUT, f"cover_diff_{name}.png"))
print("saved")
