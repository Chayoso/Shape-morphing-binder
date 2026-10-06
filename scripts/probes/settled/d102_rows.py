"""d102_rows.py — D102's verdict: per mesh and momentum measure, the render arm (L) and its physics-only twin (P) at the
three seeds (97: D98's pair, 101 and 103: D102's), the paired difference L − P at each seed, and the verdict: "behind"
when L is above P at every seed, "ahead" when below at every seed, "level" when the sign changes. Measures (lower is
better): the centre of mass's largest and final displacement (pitches), its largest and last-ten velocity, the
net-over-gross linear and angular move (run, last 20 pairs), the net rotation (deg), what still moves at the end, the
kinetic energy (last ten windows, and over the windows both runs have); from the record where it has it (D102's seeds) the
angular momentum at the window starts (the conserved quantity a rotation estimate approximates): end, largest, mean."""
import json, re
import numpy as np
G = "/data/relcfd/chayo/physmorph_v2/output/gpu"


def measures(mom, run):
    s = [l for l in open(mom) if "kept frames" in l][-1]
    g = lambda pat: [float(v) for v in re.search(pat, s).groups()]  # noqa: E731
    far, end = g(r"largest displacement ([0-9.e+-]+) a, at the end ([0-9.e+-]+) a")
    lin = g(r"net / gross linear move: largest [0-9.e+-]+, mean ([0-9.e+-]+), last 20 ([0-9.e+-]+)")
    ang = g(r"net / gross angular move: largest [0-9.e+-]+, mean ([0-9.e+-]+), last 20 ([0-9.e+-]+)")
    rot, = g(r"net rotation over the run ([0-9.e+-]+) deg")
    still, = g(r"last 20: ([0-9.e+-]+) a\s*$")
    h = [r for r in json.load(open(run))["arms"]["render_full_dt_iso_nn"]["history"] if r.get("frame_end") and not r.get("null_commit")]
    v = [sum(c * c for c in r["v_com"]) ** .5 for r in h]
    k = [r["kin"] for r in h]
    Ls = [float(np.linalg.norm(r["L_start"])) for r in h if r.get("L_start") is not None]
    extra = dict(Lmom_end=Ls[-1], Lmom_max=max(Ls), Lmom_mean=float(np.mean(Ls))) if Ls else {}
    return dict(**extra, com_far=far, com_end=end, vcom_max=max(v), vcom_last10=np.mean(v[-10:]), lin_run=lin[0], lin_last20=lin[1],
                ang_run=ang[0], ang_last20=ang[1], rotation=rot, still=still, kin_last10=np.mean(k[-10:])), k


if __name__ == "__main__":
    for mesh in ("bunny", "dragon"):
        runs = {97: {a: (f"{G}/d98/m_{mesh}300k_{a}.log", f"{G}/d98/{mesh}300k_{a}.json") for a in "LP"}}
        for seed in (101, 103):
            runs[seed] = {a: (f"{G}/d102/m_{mesh}300k_{a}s{seed}.log", f"{G}/d102/{mesh}300k_{a}s{seed}.json") for a in "LP"}
        table = {}
        for seed, arms in runs.items():
            try:
                mL, kL = measures(*arms["L"]); mP, kP = measures(*arms["P"])
            except (FileNotFoundError, AttributeError, IndexError):
                continue
            n = min(len(kL), len(kP))
            mL["kin_matched"], mP["kin_matched"] = np.mean(kL[n - 10:n]), np.mean(kP[n - 10:n])
            table[seed] = (mL, mP)
        print(f"== {mesh}: seeds {sorted(table)}")
        keys = []
        for s in sorted(table):
            keys += [kk for kk in table[s][0] if kk not in keys]
        for key in keys:
            seeds = [s for s in sorted(table) if key in table[s][0] and key in table[s][1]]
            diffs = [(table[s][0][key] - table[s][1][key]) / max(abs(table[s][1][key]), 1e-30) for s in seeds]
            verdict = "behind" if all(d > 0 for d in diffs) else "ahead" if all(d < 0 for d in diffs) else "level"
            cells = "  ".join(f"s{s}: L {table[s][0][key]:.3g} P {table[s][1][key]:.3g} ({100 * d:+.0f} %)" for s, d in zip(seeds, diffs))
            print(f"   {key:12s} {verdict:7s} | {cells}")
