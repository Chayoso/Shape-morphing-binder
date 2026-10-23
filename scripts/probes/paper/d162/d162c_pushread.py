"""d162c_pushread.py OUT_JSONL PUSH_DIR [RUN ...] -- D162c (B), the CPU part: each saved push (d162c_push.py) split on the
state's outer layer as d162c_lib.decompose, with the outer layer's share of the whole push energy, before and after
arrival. One JSON line per state."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d162c_lib as C  # noqa: E402
from d162c_motion import RUNS, arrival_frame  # noqa: E402


def main(out, push_dir, names):
    for name in names:
        f = push_dir / f"push_{name}.npz"
        if not f.exists():
            continue
        t0 = time.time()
        group, prefix = RUNS[name]
        P = np.load(f)
        z = np.load(f"{prefix}_frames12.npz", allow_pickle=True)
        a = C.pitch_of(z["tgt"])
        arr = arrival_frame(prefix)
        for raw, k, v, L in zip(P["raws"], P["kept"], P["push"], P["L"]):
            x = np.asarray(z["frames"][int(k)], np.float64)
            idx, n = C.outer_layer(x, a)
            B, deg = C.band_operator(x[idx], a)
            v = np.asarray(v, np.float64)
            d = C.decompose(v[idx], n, B)
            row = dict(run=name, group=group, raw=int(raw), t=float(raw) / 240.0, phase="after" if raw >= arr else "before",
                       L_r=float(L), n_outer=int(len(idx)), outer_energy_share=float((v[idx] ** 2).sum() / (v ** 2).sum()),
                       **d)
            with open(out, "a") as fo:
                fo.write(json.dumps(row) + "\n")
        print(f"[d162c B read] {name}: {len(P['raws'])} states, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), sys.argv[3:] or list(RUNS))
