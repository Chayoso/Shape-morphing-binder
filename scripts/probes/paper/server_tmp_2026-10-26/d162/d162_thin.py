"""d162_thin.py RUN_DIR NAME [NAME ...] -- D162 disk: after the scene reads, each named run's kept-frame archive
(NAME_frames12.npz) is rewritten in place with every 4th kept frame and the last (about one state a window); every other
key (tgt, deliver_n, the albedos, the scene's parameters) is kept and `raws` names the frames kept. Prints the sizes."""
import os
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
for name in sys.argv[2:]:
    p = d / f"{name}_frames12.npz"
    before = p.stat().st_size
    z = dict(np.load(p, allow_pickle=True))
    raws = np.asarray(z["raws"])
    keep = sorted(set(range(0, len(raws), 4)) | {len(raws) - 1})
    z["frames"] = np.asarray(z["frames"])[keep]
    z["raws"] = raws[keep]
    tmp = d / f"{name}_frames12.thin.npz"
    np.savez(tmp, **z)
    os.replace(tmp, p)
    print(f"{name}: {len(raws)} -> {len(keep)} kept frames, {before / 1e6:.0f} -> {p.stat().st_size / 1e6:.0f} MB", flush=True)
