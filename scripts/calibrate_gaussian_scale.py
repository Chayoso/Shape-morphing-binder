"""Render a high-resolution target-only Gaussian scale calibration sheet.

This is deliberately not a simulation.  It samples the requested target exactly as the
pipeline does, keeps the same frozen material-surface estimator, and varies only the
dimensionless nearest-neighbour multiplier used for the Gaussian rest radius.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physmorph.pipeline.runner import _surface_weights  # noqa: E402
from physmorph.render.covariance import sigma0_from_nn  # noqa: E402
from physmorph.render.photoreal import render_3dgs  # noqa: E402
from physmorph.sampling import load_normalized  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tgt", default="assets/bunny.obj")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--surface_frac", type=float, default=0.5)
    ap.add_argument("--scales", default="0.8,1.0,1.2")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--out", default="output/gaussian_scale_20k.png")
    args = ap.parse_args()

    scales = [float(v) for v in args.scales.split(",")]
    if not scales or any(v <= 0 for v in scales):
        raise ValueError("--scales must contain positive comma-separated values")
    tgt = load_normalized(args.tgt, args.n, args.seed)
    weight = _surface_weights(tgt, 24, args.surface_frac, 0.05)
    mask = weight > 0.5
    surf = np.ascontiguousarray(tgt[mask], np.float32)
    eye = np.tile(np.eye(3, dtype=np.float32), (len(surf), 1, 1))

    # Material-space colouring exposes excessive overlap: large splats erase these bands.
    lo = surf.min(0); span = np.maximum(surf.max(0) - lo, 1e-6)
    uvw = (surf - lo) / span
    colors = np.stack([0.22 + 0.68 * uvw[:, 1],
                       0.28 + 0.58 * uvw[:, 2],
                       0.34 + 0.56 * uvw[:, 0]], axis=1).astype(np.float32)
    azimuths = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
    rows = []
    report = {"N": int(args.n), "surface_count": int(mask.sum()),
              "surface_frac": float(mask.mean()), "resolution": int(args.res),
              "scales": []}
    for scale in scales:
        sigma0 = sigma0_from_nn(surf, scale)
        views = [render_3dgs(surf, colors, F=eye, sigma0=sigma0, opacity=0.92,
                             azimuth=az, elevation=0.3, res=args.res,
                             center=tgt.mean(0)) for az in azimuths]
        rows.append(np.concatenate(views, axis=1))
        report["scales"].append({"scale": scale, "sigma0": sigma0,
                                 "three_sigma_radius": 3.0 * sigma0})
    sheet = np.concatenate(rows, axis=0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray((np.clip(sheet, 0, 1) * 255).astype(np.uint8)).save(out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
