#!/usr/bin/env python3
"""Compare surface-aware runs episode-by-episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


KEYS = [
    "loss_total_mv",
    "loss_depth_mv",
    "loss_physics",
    "dFc_mean",
    "J_min",
    "J_max",
    "corr_loss_total_mv",
]


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "episode" in data:
        out = []
        for i, ep in enumerate(data["episode"]):
            rec = {"ep": ep}
            for k, v in data.items():
                if isinstance(v, list) and i < len(v):
                    rec[k] = v[i]
            out.append(rec)
        return out
    raise ValueError(f"Unsupported losses format: {path}")


def rec_ep(rec: dict) -> int:
    if "ep" in rec:
        return int(rec["ep"])
    if "episode" in rec:
        return int(rec["episode"])
    raise KeyError("Missing episode key")


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    args = ap.parse_args()

    base_path = Path(args.baseline) / "losses.json"
    cand_path = Path(args.candidate) / "losses.json"
    base = {rec_ep(r): r for r in load_records(base_path)}
    cand = {rec_ep(r): r for r in load_records(cand_path)}

    shared_eps = sorted(set(base) & set(cand))
    print("ep\tmetric\tbaseline\tcandidate\tdelta\tcandidate/baseline")
    for ep in shared_eps:
        b = base[ep]
        c = cand[ep]
        for key in KEYS:
            if key not in b and key not in c:
                continue
            bv = b.get(key)
            cv = c.get(key)
            delta = None
            ratio = None
            if isinstance(bv, (int, float)) and isinstance(cv, (int, float)):
                delta = cv - bv
                ratio = (cv / bv) if bv not in (0, 0.0) else None
            print(
                "\t".join(
                    [
                        str(ep),
                        key,
                        fmt(bv),
                        fmt(cv),
                        fmt(delta),
                        fmt(ratio),
                    ]
                )
            )

    last_ep = shared_eps[-1]
    b = base[last_ep]
    c = cand[last_ep]
    print("\nFINAL")
    for key in KEYS:
        if key not in b and key not in c:
            continue
        bv = b.get(key)
        cv = c.get(key)
        if isinstance(bv, (int, float)) and isinstance(cv, (int, float)) and bv not in (0, 0.0):
            improvement = 100.0 * (bv - cv) / abs(bv)
            print(f"{key}: {fmt(bv)} -> {fmt(cv)} ({improvement:+.2f}% better if lower)")
        else:
            print(f"{key}: {fmt(bv)} -> {fmt(cv)}")


if __name__ == "__main__":
    main()
