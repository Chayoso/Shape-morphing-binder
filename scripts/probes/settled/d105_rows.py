"""d105_rows.py — D105's momentum rows: per mesh, D98's render arm (L) and physics-only twin (P) at seed 97 and D105's (LV,
PV), each measure (lower is better) with the render arm against its twin (L − P over P) and D105's arm against D98's same
arm. Measures as d102_rows.py: the centre of mass's largest and final displacement (pitches), its largest and last-ten
velocity, net-over-gross linear and angular move (run, last 20 pairs), net rotation (deg), what still moves at the end,
the kinetic energy (last ten windows, and over the windows both arms of a pair have); the angular momentum at the window
starts (end, largest, mean) where the record has it (D105's runs; D98's seed-97 runs predate the record)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np                                             # noqa: E402
from d102_rows import G, measures                              # noqa: E402

for mesh in ("bunny", "dragon"):
    arms = {"P": ("d98", "P"), "L": ("d98", "L"), "PV": ("d105", "PV"), "LV": ("d105", "LV")}
    m, k = {}, {}
    for name, (d, a) in arms.items():
        m[name], k[name] = measures(f"{G}/{d}/m_{mesh}300k_{a}.log", f"{G}/{d}/{mesh}300k_{a}.json")
    for p, l in (("P", "L"), ("PV", "LV")):
        n = min(len(k[p]), len(k[l]))
        m[l]["kin_matched"], m[p]["kin_matched"] = np.mean(k[l][n - 10:n]), np.mean(k[p][n - 10:n])
    print(f"== {mesh} (render arm against its twin; D105 against D98's same arm)")
    rel = lambda a, b, key: 100 * (m[a][key] - m[b][key]) / max(abs(m[b][key]), 1e-30)   # noqa: E731
    for key in m["LV"]:
        cells = "  ".join(f"{a} {m[a][key]:.3g}" if key in m[a] else f"{a} -" for a in arms)
        pair = f"L/P {rel('L', 'P', key):+.0f} %" if key in m["L"] else "L/P -"
        print(f"   {key:12s} | {cells} | {pair}, LV/PV {rel('LV', 'PV', key):+.0f} % | "
              + (f"PV/P {rel('PV', 'P', key):+.0f} %, LV/L {rel('LV', 'L', key):+.0f} %" if key in m["L"] else ""))
