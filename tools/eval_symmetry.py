#!/usr/bin/env python3
"""Evaluate simple bilateral symmetry on final checkpoint positions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def load_positions(run_dir: Path) -> np.ndarray:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"no checkpoints in {run_dir}")
    ckpts = sorted(ckpt_dir.glob("ckpt_ep*.npz"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint files in {ckpt_dir}")
    return np.load(ckpts[-1])["positions"].astype(np.float32)


def symmetry_score(x: np.ndarray, axis: int = 0, top_frac: float = 0.35):
    bbox_min = x.min(axis=0)
    bbox_max = x.max(axis=0)
    center = 0.5 * (bbox_min[axis] + bbox_max[axis])

    z_thresh = np.quantile(x[:, 2], 1.0 - top_frac)
    top_mask = x[:, 2] >= z_thresh
    x_top = x[top_mask]
    if x_top.shape[0] < 32:
        return None

    x_mirror = x_top.copy()
    x_mirror[:, axis] = 2.0 * center - x_mirror[:, axis]
    dd, _ = cKDTree(x_top).query(x_mirror, k=1)

    left = int((x_top[:, axis] < center).sum())
    right = int((x_top[:, axis] > center).sum())
    balance = abs(left - right) / max(left + right, 1)

    return {
        "top_count": int(x_top.shape[0]),
        "axis": int(axis),
        "center": float(center),
        "top_frac": float(top_frac),
        "mirror_nn_mean": float(dd.mean()),
        "mirror_nn_median": float(np.median(dd)),
        "mirror_nn_p90": float(np.quantile(dd, 0.9)),
        "left_count": left,
        "right_count": right,
        "left_right_balance": float(balance),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--axis", default="auto", help="x, y, z, or auto")
    ap.add_argument("--top-frac", type=float, default=0.35)
    args = ap.parse_args()

    axis_map = {"x": 0, "y": 1, "z": 2}
    print("run\taxis\tmirror_nn_mean\tmirror_nn_median\tmirror_nn_p90\tleft_right_balance\ttop_count")
    for run_str in args.runs:
        run_dir = Path(run_str)
        x = load_positions(run_dir)
        axes = [axis_map[args.axis]] if args.axis in axis_map else [0, 1]
        best = None
        for ax in axes:
            cur = symmetry_score(x, axis=ax, top_frac=args.top_frac)
            if cur is None:
                continue
            if best is None or cur["mirror_nn_mean"] < best["mirror_nn_mean"]:
                best = cur
        if best is None:
            print(f"{run_dir.name}\t-\t-\t-\t-\t-\t-")
            continue
        print(
            f"{run_dir.name}\t{best['axis']}\t{best['mirror_nn_mean']:.4f}\t"
            f"{best['mirror_nn_median']:.4f}\t{best['mirror_nn_p90']:.4f}\t"
            f"{best['left_right_balance']:.4f}\t{best['top_count']}"
        )


if __name__ == "__main__":
    main()
