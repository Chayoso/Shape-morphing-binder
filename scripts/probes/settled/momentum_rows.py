"""momentum_rows.py LABEL=MOM_LOG,RUN_JSON [...] — one line of momentum per run: from the probe's log (kept frames) the
centre of mass's largest and final displacement, the net-over-gross linear and angular move (mean of the run, of the
last 20 pairs), the net rotation, what still moves at the end; from the run's record (every committed window) the
centre of mass's velocity (largest, mean of the last ten windows) and the kinetic energy (last window, mean of the
last ten)."""
import json, re, sys
print("run | centre of mass: largest / end (pitches) | its velocity: largest / last ten | net/gross linear: run / last 20 | net/gross angular: run / last 20 | "
      "net rotation (deg) | still moving at the end (pitches a pair) | kinetic energy: last / last ten")
for spec in sys.argv[1:]:
    label, rest = spec.split("=")
    mom, run = rest.split(",")
    s = [l for l in open(mom) if "kept frames" in l][-1]
    g = lambda pat: [float(v) for v in re.search(pat, s).groups()]
    far, end = g(r"largest displacement ([0-9.e+-]+) a, at the end ([0-9.e+-]+) a")
    lin = g(r"net / gross linear move: largest [0-9.e+-]+, mean ([0-9.e+-]+), last 20 ([0-9.e+-]+)")
    ang = g(r"net / gross angular move: largest [0-9.e+-]+, mean ([0-9.e+-]+), last 20 ([0-9.e+-]+)")
    rot, = g(r"net rotation over the run ([0-9.e+-]+) deg")
    still, = g(r"last 20: ([0-9.e+-]+) a\s*$")
    h = [r for r in json.load(open(run))["arms"]["render_full_dt_iso_nn"]["history"] if r.get("frame_end")]
    v = [sum(c * c for c in r["v_com"]) ** .5 for r in h]
    k = [r["kin"] for r in h]
    print(f"{label} | {far:.4f} / {end:.4f} | {max(v):.2e} / {sum(v[-10:]) / 10:.2e} | {lin[0]:.2e} / {lin[1]:.2e} | {ang[0]:.2e} / {ang[1]:.2e} | {rot:.4f} | {still:.4f} | {k[-1]:.2e} / {sum(k[-10:]) / 10:.2e}")
