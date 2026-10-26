"""rule_diff.py MORPH_DIR MESH [...] -- FIGS_V2 (CPU): per rendered kept frame, how many pixels the display rule changed
(frames/ against frames_raw/, any channel off by > 24 of 255), and where."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

base = Path(sys.argv[1])
for m in sys.argv[2:]:
    for f in sorted((base / m / "frames").glob("*.jpg")):
        r = base / m / "frames_raw" / f.name
        if not r.exists():
            continue
        d = np.abs(np.asarray(Image.open(f)).astype(int) - np.asarray(Image.open(r)).astype(int)).max(-1)
        n = int((d > 24).sum())
        print(m, f.stem, n if n == 0 else f"{n} px, box x {np.where(d > 24)[1].min()}-{np.where(d > 24)[1].max()} "
              f"y {np.where(d > 24)[0].min()}-{np.where(d > 24)[0].max()}", flush=True)
