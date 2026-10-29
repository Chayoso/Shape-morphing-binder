"""lead145.py DIR ARM_A ARM_B MESH [...] [--md OUT_MD] -- D145 B1 post-tag: the render lead over the trajectory from
read300_full.sh's every-frame readings (t3_NAME.log: the yardstick's exterior silhouette and shading against the
independent sample; e3_NAME.log: the display's 1 - IoU front), render arm A against twin B (CPU only).

Per mesh and measure, on a grid of simulated time (every 0.1 s from 0.1 s to the shortest run's last kept frame;
state / 240 s; each run's curve linearly interpolated): A's and B's min / median / max over seeds; the share of grid
times where the ranges are apart with A lower (A ahead) and with A higher (A behind); the first time from which A stays
ahead to the end of the grid; the ratio of the medians at t = 0.5 / 1 / 2 / 4 / 8 s and at the grid's end."""
import json
import re
import sys
from pathlib import Path

import numpy as np

args = sys.argv[1:]
md = None
if "--md" in args:
    i = args.index("--md")
    md = args[i + 1]
    del args[i:i + 2]
D, A, B, meshes = Path(args[0]), args[1], args[2], args[3:]
MEAS = [("yardstick silhouette", "t3", lambda r: r["exterior"]["sil"]),
        ("yardstick shading", "t3", lambda r: r["exterior"]["pbr"]),
        ("display front 1 - IoU", "e3", lambda r: 1 - r["to_target"]["front"]["iou"])]


def curves(kind, mesh, arm, get):
    out = []
    for f in sorted(D.glob(f"{kind}_{mesh}300k_{arm}[0-9]*.log")):
        if not re.search(rf"_{arm}\d+\.log$", f.name):
            continue
        pts = []
        for s in open(f):
            if not s.startswith("{"):
                continue
            try:
                r = json.loads(s)
                pts.append((int(r["state"]) / 240.0, float(get(r))))
            except (KeyError, ValueError, TypeError):
                continue
        pts.sort()
        if len(pts) > 1:
            out.append((np.array([p[0] for p in pts]), np.array([p[1] for p in pts])))
    return out


lines = []
for mesh in meshes:
    lines += [f"### Every kept frame, {mesh}: {A} against {B}", "",
              "| measure | runs A / B | grid (s) | A ahead, apart | A behind, apart | A ahead from (s) | "
              "A / B at 0.5 s | 1 s | 2 s | 4 s | 8 s | grid end |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, kind, get in MEAS:
        ca, cb = curves(kind, mesh, A, get), curves(kind, mesh, B, get)
        if not ca or not cb:
            continue
        t1 = min(c[0][-1] for c in ca + cb)
        grid = np.arange(0.1, t1 + 1e-9, 0.1)
        ya = np.array([np.interp(grid, c[0], c[1]) for c in ca])
        yb = np.array([np.interp(grid, c[0], c[1]) for c in cb])
        ahead = ya.max(0) < yb.min(0)
        behind = ya.min(0) > yb.max(0)
        start = None
        for k in range(len(grid)):
            if ahead[k:].all():
                start = grid[k]
                break
        ma, mb = np.median(ya, 0), np.median(yb, 0)
        at = []
        for T in (0.5, 1, 2, 4, 8):
            at.append("-" if T > grid[-1] else f"{np.interp(T, grid, ma) / np.interp(T, grid, mb):.2f}")
        lines.append(f"| {label} | {len(ca)} / {len(cb)} | 0.1-{grid[-1]:.1f} | {ahead.mean() * 100:.0f} % | "
                     f"{behind.mean() * 100:.0f} % | {'-' if start is None else f'{start:.1f}'} | " + " | ".join(at) +
                     f" | {ma[-1] / mb[-1]:.2f} |")
    lines.append("")
text = "\n".join(lines)
print(text)
if md:
    open(md, "w").write(text + "\n")
