"""d107_summary.py ROWS_TXT EARLY_END LATE_START — D107 (2) in two numbers per field and band: the mean over the windows
before EARLY_END (the detail being written) and from LATE_START on (the plateau) of each field's "toward" (its correlation
with -offset in the band), and of the share of the band's offset each move closes per window; the band offset's rms at
the first window, at EARLY_END and at the last."""
import json, sys
import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
e_end, l_start = int(sys.argv[2]), int(sys.argv[3])
early = [r for r in rows if 1 <= r["window"] < e_end]
late = [r for r in rows if r["window"] >= l_start]
BANDS = ("2.7", "5.4", "10.8", "21.6")
print(f"windows: {len(rows)}; early 1..{e_end - 1} ({len(early)}), late {l_start}.. ({len(late)})")
for b in BANDS:
    at = {r["window"]: r[f"off_{b}"] for r in rows}
    near = min(at, key=lambda w: abs(w - e_end))
    print(f"== {b} pitches: offset rms first {rows[0][f'off_{b}']:.3f}, at window {near} {at[near]:.3f}, last {rows[-1][f'off_{b}']:.3f}")
    for f in ("render", "physics", "r_dFc", "p_dFc", "r_u", "p_u", "opt", "all", "free"):
        m = lambda rs, k: float(np.mean([r[k] for r in rs])) if rs else float("nan")   # noqa: E731
        line = f"   {f:8s} toward early {m(early, f'{f}_toward_{b}'):+.3f}, late {m(late, f'{f}_toward_{b}'):+.3f}"
        if f"{f}_closes_{b}" in rows[0]:
            line += f" | closes per window early {m(early, f'{f}_closes_{b}'):+.4f}, late {m(late, f'{f}_closes_{b}'):+.4f}"
        print(line)
