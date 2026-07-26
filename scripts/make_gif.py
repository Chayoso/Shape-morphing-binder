"""Animate a saved morph: orthographic splat + depth shading -> GIF. CPU only.

Two panels (two azimuths) so the shape is readable, target silhouette drawn as a faint outline so
convergence is visible, and a frame counter. This is the raw particle state, not a 3DGS render.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def splat(x, res, az, el=0.18, extent=1.0, fp=1):
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    right = np.array([ca, 0.0, -sa], np.float32)
    up = np.array([-sa * se, ce, -ca * se], np.float32)
    fwd = np.cross(right, up)
    p = np.stack([x @ right, x @ up], 1)
    z = x @ fwd
    rel = (p + extent) / (2 * extent) * res
    ij = np.floor(rel).astype(np.int64)
    ok = (ij >= 0).all(1) & (ij < res).all(1)
    ij, zz = ij[ok], z[ok]
    cov = np.zeros((res, res), np.float32)
    dep = np.full((res, res), -1e9, np.float32)
    for ox in range(-fp, fp + 1):
        for oy in range(-fp, fp + 1):
            i2 = np.clip(ij[:, 0] + ox, 0, res - 1)
            j2 = np.clip(ij[:, 1] + oy, 0, res - 1)
            f = i2 * res + j2
            np.add.at(cov.reshape(-1), f, 1.0)
            np.maximum.at(dep.reshape(-1), f, zz)
    return cov, dep


def shade(cov, dep, base=(0.36, 0.55, 0.78), outline=None):
    m = cov > 0
    img = np.ones(cov.shape + (3,), np.float32)
    if m.any():
        d = np.where(m, dep, dep[m].min())
        d = ndimage.uniform_filter(d, 3)
        gy, gx = np.gradient(d)
        n = np.stack([-gx, -gy, np.full_like(d, 2.2)], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
        lit = 0.28 + 0.72 * np.clip(n @ np.array([0.35, 0.5, 0.79], np.float32), 0, 1) ** 0.8
        for c in range(3):
            img[..., c] = np.where(m, np.clip(base[c] * lit, 0, 1), 1.0)
    if outline is not None:                      # faint target contour for reference
        edge = outline ^ ndimage.binary_erosion(outline, iterations=2)
        for c, v in enumerate((0.85, 0.42, 0.30)):
            img[..., c] = np.where(edge & ~m, v, img[..., c])
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--res", type=int, default=190)
    ap.add_argument("--views", default="0.6,2.2")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--hold", type=int, default=12, help="extra final frames")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.npz)
    fr, tgt = d["frames"], d["tgt"]
    azs = [float(s) for s in args.views.split(",")]
    ext = float(np.abs(np.stack([fr[0], fr[-1], tgt])).max()) * 1.15

    tgt_masks = []
    for az in azs:
        c, _ = splat(tgt, args.res, az, extent=ext)
        tgt_masks.append(ndimage.binary_fill_holes(c > 0))

    idx = list(range(0, len(fr), args.stride))
    if idx[-1] != len(fr) - 1:
        idx.append(len(fr) - 1)
    imgs = []
    for k in idx:
        panels = []
        for az, tm in zip(azs, tgt_masks):
            cov, dep = splat(fr[k], args.res, az, extent=ext)
            panels.append(np.transpose(shade(cov, dep, outline=tm), (1, 0, 2))[::-1])
        im = Image.fromarray(np.concatenate(panels, 1))
        dr = ImageDraw.Draw(im)
        dr.text((6, 4), f"{args.label}  frame {k:3d}/{len(fr) - 1}", fill=(40, 40, 40))
        imgs.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
    imgs.extend([imgs[-1]] * args.hold)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(args.out, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"saved {args.out}  ({len(imgs)} frames, {Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
