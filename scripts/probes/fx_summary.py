"""Per-run telemetry of the render x u factorial (docs/surface_gradient.md §10) from the run json:
windows, accepted windows, delivered frames, minutes, the render channel's share of the accepted
control update g_share (mean over windows 1-20 / all), lambda, the cosine to the physics gradient,
D_vol at the first and the last window, and the end metrics. Deterministic per-window numbers,
untouched by run-to-run chaos.

Usage: fx_summary.py <out_dir> <target> ...     (reads fx_{11,11c,01,10,00,cut}_<target>.json)"""
import json
import sys

import numpy as np

out = sys.argv[1]
ARM = "render_full_dt_iso_nn"
CELLS = ["11", "11c", "01", "10", "00", "cut"]
for _a in list(sys.argv[2:]):
    if _a.startswith("--cells="):                   # candidate40.sh: a chosen list of cells
        CELLS = _a[len("--cells="):].split(",")
        sys.argv.remove(_a)


def mean_of(h, key, sl=slice(None)):
    v = [e.get(key) for e in h[sl] if isinstance(e, dict) and e.get(key) is not None]
    return float(np.mean(v)) if v else float("nan")


def first_of(h, key):
    for e in h:
        if isinstance(e, dict) and e.get(key) is not None:
            return float(e[key])
    return float("nan")


def last_of(h, key):
    for e in reversed(h):
        if isinstance(e, dict) and e.get(key) is not None:
            return float(e[key])
    return float("nan")


print(f"{'run':14s} {'win':>4s} {'acc':>4s} {'frames':>6s} {'min':>6s} {'gshare1-20':>10s} {'gshare':>7s} "
      f"{'lambda':>7s} {'g_cos':>7s} {'Dvol_w1':>8s} {'Dvol_end':>8s} {'dsil_end':>8s} {'chamfer':>8s} {'silIoU':>7s} {'detFmin':>7s}")
for T in sys.argv[2:]:
    for c in CELLS:
        name = f"fx_{c}_{T}"
        try:
            j = json.load(open(f"{out}/{name}.json"))
        except Exception as e:  # noqa: BLE001
            print(f"{name}: no json ({e})")
            continue
        r = j["arms"][ARM]
        h = r.get("history") or []
        met = r.get("metrics", {})
        acc = sum(1 for e in h if isinstance(e, dict) and e.get("accepted"))
        frames = int(r.get("deliver_n") or met.get("frames") or 0)
        minutes = float(r.get("seconds") or 0.0) / 60.0
        print(f"{name:14s} {len(h):4d} {acc:4d} {frames:6d} {minutes:6.1f} {mean_of(h, 'g_share', slice(0, 20)):10.3f} "
              f"{mean_of(h, 'g_share'):7.3f} {mean_of(h, 'lambda'):7.3f} {mean_of(h, 'g_cos'):7.3f} "
              f"{first_of(h, 'd_vol'):8.4f} {last_of(h, 'd_vol'):8.4f} {last_of(h, 'd_sil'):8.4f} "
              f"{float(met.get('chamfer', float('nan'))):8.4f} {float(met.get('sil_iou', float('nan'))):7.4f} "
              f"{float(met.get('detF_min', float('nan'))):7.3f}")
