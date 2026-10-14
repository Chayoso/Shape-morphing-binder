"""D147 M-b: the floor drops P8 (the frozen grid floor) and P9 (the gated floor): each body's lowest particle above the
floor, in sample spacings, over time (read-only)."""
import json
import sys
from pathlib import Path

O = Path(sys.argv[1])
for t in ("P8_drop_old", "P9_drop_gate"):
    f = O / f"{t}.json"
    if not f.exists():
        print(t, "missing")
        continue
    d = json.loads(f.read_text())
    r, sp = d["records"], d["summary"]["spacing"]
    print(f"{t}: spacing {sp:.4f} wu, dx {d['summary']['dx']:.4f} wu")
    for k in range(0, len(r), max(1, len(r) // 24)):
        print(f"  t={r[k]['t']:.2f}  lowest above the floor (spacings) {[round(v / sp, 2) for v in r[k]['low']]}  "
              f"v_y body0 {r[k]['v0'][1]:+.3f}")
    print(f"  end: {[round(v / sp, 2) for v in r[-1]['low']]}")
