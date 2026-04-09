#!/usr/bin/env python3
"""Summarize final metrics from run directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_KEYS = [
    "loss_total_mv",
    "worst_view_bce",
    "loss_physics",
    "alpha_mse",
    "corr_loss_total_mv",
    "corr_local_applied",
    "corr_local_bilateral",
    "corr_local_active_balance",
]


def load_last_row(run_dir: Path):
    losses_path = run_dir / "losses.json"
    if not losses_path.exists():
        return None
    data = json.loads(losses_path.read_text())
    if not data:
        return None
    return data[-1]


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="root output directory")
    ap.add_argument("--match", default="", help="substring filter on run name")
    ap.add_argument("--keys", nargs="*", default=DEFAULT_KEYS, help="metric keys to print")
    args = ap.parse_args()

    root = Path(args.root)
    runs = sorted([p for p in root.iterdir() if p.is_dir() and args.match in p.name])

    header = ["run", "episodes", "last_ep"] + args.keys
    print("\t".join(header))
    for run in runs:
        last = load_last_row(run)
        if last is None:
            continue
        episodes_path = run / "losses.json"
        episodes = len(json.loads(episodes_path.read_text()))
        row = [run.name, str(episodes), str(last.get("ep", "-"))]
        for key in args.keys:
            row.append(fmt(last.get(key)))
        print("\t".join(row))


if __name__ == "__main__":
    main()
