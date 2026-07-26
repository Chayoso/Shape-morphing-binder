"""CPU-only look at a saved morph: orthographic splat + depth shading, no rasteriser needed.

Deliberately not the 3DGS renderer — this is a diagnostic view of the raw particle state, so what
you see is the simulation, not a reconstruction. Also reports the hole fraction (background visible
inside the silhouette), which is the number that told us the render-guided PhysForm arm was tearing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from scipy import ndimage                # noqa: E402


def splat(x, res=220, az=0.0, el=0.18, pad=1.15, extent=None):
    """Orthographic projection -> (coverage, nearest-depth). Returns float images."""
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    right = np.array([ca, 0.0, -sa], np.float32)
    up = np.array([-sa * se, ce, -ca * se], np.float32)
    fwd = np.cross(right, up)
    p = np.stack([x @ right, x @ up], 1)
    z = x @ fwd
    e = extent if extent is not None else float(np.abs(p).max()) * pad
    rel = (p + e) / (2 * e) * res
    ij = np.floor(rel).astype(np.int64)
    ok = (ij >= 0).all(1) & (ij < res).all(1)
    ij, zz = ij[ok], z[ok]
    cov = np.zeros((res, res), np.float32)
    dep = np.full((res, res), -1e9, np.float32)
    # FOOTPRINT: one pixel per particle leaves ~0.4 hits/px at this N and resolution, so the body
    # renders as a point cloud and "holes" measure the splat, not the physics. Give each particle a
    # 3x3 footprint so coverage saturates and the silhouette is the object's, not the sampling's.
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            i2 = np.clip(ij[:, 0] + ox, 0, res - 1)
            j2 = np.clip(ij[:, 1] + oy, 0, res - 1)
            flat = i2 * res + j2
            np.add.at(cov.reshape(-1), flat, 1.0)
            np.maximum.at(dep.reshape(-1), flat, zz)
    return cov, dep, e


def shade(cov, dep, base=(0.36, 0.55, 0.78)):
    """Solid look: mask by coverage, brightness from the depth field's local slope."""
    m = cov > 0
    img = np.ones(cov.shape + (3,), np.float32)
    if not m.any():
        return img
    d = dep.copy()
    d[~m] = np.nan
    d = ndimage.generic_filter(np.nan_to_num(d, nan=float(np.nanmin(d))), np.mean, size=3)
    gy, gx = np.gradient(d)
    n = np.stack([-gx, -gy, np.ones_like(d) * 2.2], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    lit = np.clip(n @ np.array([0.35, 0.5, 0.79], np.float32), 0.0, 1.0) ** 0.8
    lit = 0.28 + 0.72 * lit
    for c in range(3):
        img[..., c] = np.where(m, np.clip(base[c] * lit, 0, 1), 1.0)
    return img


def hole_fraction(cov):
    body = cov > 0
    filled = ndimage.binary_fill_holes(body)
    n = filled.sum()
    return float((filled & ~body).sum() / n) if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--frames", default="0,6,12,18,24,30")
    ap.add_argument("--views", default="0.6,2.2")
    ap.add_argument("--res", type=int, default=130)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.npz)
    fr, tgt = d["frames"], d["tgt"]
    idx = [min(int(s), len(fr) - 1) for s in args.frames.split(",")]
    azs = [float(s) for s in args.views.split(",")]

    # one shared extent so the strip is spatially comparable frame to frame
    ext = max(float(np.abs(np.stack([fr[0], fr[-1], tgt])).max()) * 1.15, 1e-6)

    ncol = len(idx) + 1
    fig, axes = plt.subplots(len(azs), ncol, figsize=(2.05 * ncol, 2.12 * len(azs)))
    axes = np.atleast_2d(axes)
    holes = []
    for r, az in enumerate(azs):
        for c, t in enumerate(idx):
            cov, dep, _ = splat(fr[t], args.res, az, extent=ext)
            axes[r, c].imshow(np.transpose(shade(cov, dep), (1, 0, 2))[::-1])
            if r == 0:
                axes[r, c].set_title(f"frame {t}", fontsize=9)
                holes.append(hole_fraction(cov))
        cov, dep, _ = splat(tgt, args.res, az, extent=ext)
        axes[r, ncol - 1].imshow(np.transpose(shade(cov, dep, base=(0.82, 0.45, 0.32)),
                                              (1, 0, 2))[::-1])
        if r == 0:
            axes[r, ncol - 1].set_title("TARGET", fontsize=9, color="#A5473B", fontweight="bold")
        axes[r, 0].set_ylabel(f"view az={az:.1f}", fontsize=8)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    ttl = args.title or Path(args.npz).stem
    fig.suptitle(f"{ttl}   —   hole fraction per frame: " +
                 ", ".join(f"{h*100:.1f}%" for h in holes), fontsize=10.5, fontweight="bold")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"saved {args.out}")
    print("hole fraction:", [round(h * 100, 2) for h in holes])


if __name__ == "__main__":
    main()
