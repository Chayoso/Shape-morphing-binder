"""Extract one committed state of a run into a small self-contained (json, npz) pair that
paired_census.py / solid_census.py can read, so states from different machines can be
compared without moving gigabyte archives.

usage: python scripts/probes/extract_state.py PREFIX --out NEW_PREFIX (--level 30 | --best)
  --level L : first accepted commit with d_vol <= L      --best : the min-d_vol commit
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix"); ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="render_full_dt_iso_nn")
    ap.add_argument("--level", type=float, default=None)
    ap.add_argument("--best", action="store_true")
    a = ap.parse_args()
    d = json.load(open(a.prefix + ".json"))
    h = [q for q in d["arms"][a.arm]["history"] if q.get("d_vol") is not None
         and not q.get("null_commit") and q.get("frame_end")]
    if a.best or a.level is None:
        q = min(h, key=lambda r: r["d_vol"])
    else:
        q = next((r for r in h if r["d_vol"] <= a.level), h[-1])
    z = np.load(f"{a.prefix}_{a.arm}.npz")
    fr = z["frames"]
    state = fr[int(q["frame_end"]) - 1]
    rec = dict(q); rec["frame_end"] = 2
    json.dump({"arms": {a.arm: {"history": [rec]}}, "source": a.prefix}, open(a.out + ".json", "w"))
    np.savez(f"{a.out}_{a.arm}.npz", frames=np.stack([fr[0], state]), tgt=z["tgt"], deliver_n=2)
    print(f"extracted a{q['animation'] + 1} d_vol {q['d_vol']:.2f} frame {q['frame_end']} -> {a.out}")


if __name__ == "__main__":
    main()
