"""d174_read.py MODE OUT_DIR -- D174's runtime A/B readings (CPU on hyde06), D172's readers unchanged (tmp/d172/d172_read.py,
on D170's and D166's): the candidate arms against Hu.
  scenes     coin / bunny relief / band 40k, seeds 97 / 99 (D166's scene reader), with Hu's rows from D172 job 185
  gal        the 6-mesh 40k gallery, seed 97 (D170's gallery reader against the tag's rule)
  scales     the 300k stylized dragon scales, seed 97 (D172's detail reader), Hu and the twin read again beside them
  aggregate  OUT_MD: the A/B table (each arm's numbers beside Hu's and whether every verdict is Hu's)
Arms: Hu2 / Hu4 = Hu (--lambda_ema 0 --settle_windows 30 --no-settle_relax --render_split u) with --render_every 2 / 4: the
render gradient's adjoint sweep at every 2nd / 4th iteration of a window, the window's last one reused between."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d172")
import d172_read as R2  # noqa: E402

R7, R6, G = R2.R7, R2.R6, R2.G
RUNS = G / "d174" / "runs"
ARMS = {"Hu2": "Hu with the render gradient every 2nd iteration", "Hu4": "Hu with the render gradient every 4th iteration"}
for a, t in ARMS.items():
    R6.NAMES[a] = t + ", settle 30 (relaxation off in the settle)"
    R7.ARM_NAMES[a] = t
    R2.NAMES[a] = t


def _verdict(D):
    p = D / "verdict.json"
    return json.load(open(p)) if p.exists() else None


def scenes(D):
    R7.wait_rc([str(RUNS / f"{sc}_{a}_{s}") for sc in ("coin", "bunny", "band") for a in ARMS for s in (97, 99)])
    cwd = os.getcwd()
    R6.scenes(str(D), str(RUNS), "D174: runtime A/B against Hu, 40k scenes", *[f"{a}:T" for a in ARMS])
    os.chdir(cwd)
    L = open(D / "summary.md").read().rstrip("\n").split("\n")
    refs = R2.ref_rows(G / "d172/auto/185_scenes/summary.md", ["Hu"])
    k = max(i for i, x in enumerate(L) if x.startswith("| "))
    L = L[:k + 1] + refs + L[k + 1:]
    L.append("Reference rows: Hu (D172 job 185), the same scenes, seeds and readers.")
    open(D / "summary.md", "w").write("\n".join(L) + "\n")


def gal(D):
    R7.RUNS = RUNS
    V, L = {}, ["# D174: runtime A/B against Hu, the 6-mesh 40k gallery, seed 97", ""]
    for a in ARMS:
        (D / a).mkdir(parents=True, exist_ok=True)
        R7.gal(D / a, a, (97,), f"D174: {a} ({ARMS[a]}) on the 6-mesh 40k gallery, seed 97")
        V[a] = json.load(open(D / a / "verdict.json"))["arms"][a]
        L += [x for x in open(D / a / "summary.md").read().splitlines()[2:] if x.strip()] + [""]
    verdict = "; ".join(f"{a} {v['status']} (holds {v['held']}/{v['n']}" + (f", misses {','.join(v['miss'])}" if v["miss"] else "")
                        + f"; rest {'ok' if v['rest'] else 'no'})" for a, v in V.items())
    L.append("VERDICT: " + verdict)
    open(D / "summary.md", "w").write("\n".join(L) + "\n")
    json.dump(dict(arms=V, verdict=verdict), open(D / "verdict.json", "w"), indent=1, default=float)


def scales(D):
    runs = {"T_97": G / "d172/runs/scl300k_T_97", "Hu_97": G / "d172/runs/scl300k_Hu_97"}
    runs.update({f"{a}_97": RUNS / f"scl300k_{a}_97" for a in ARMS})
    R7.DETAIL["scl300_174"] = dict(R7.DETAIL["scl300"], runs=runs)
    R7.JOB_DETAIL["scl300_174"] = ("scl300_174",)
    R2.detail("scl300_174", D)


def aggregate(out_md):
    A = G / "d174" / "auto"
    sc, ga, sl = _verdict(A / "ab_scenes"), _verdict(A / "ab_gal"), _verdict(A / "ab_scales")
    hu = {"relief": 0.45, "colour": 0.07, "kept": 0.91, "rest": True, "gal": "6/6", "scl": "PASS"}
    rows = ["| arm | relief min (coin, bunny) | colour min (band) | kept min | rest | gallery 40k s97 holds | 300k scales | every verdict Hu's? |",
            "|---|---|---|---|---|---|---|---|",
            f"| Hu (D172) | {hu['relief']} | {hu['colour']} | {hu['kept']} | ok | {hu['gal']} | PASS (share 0.94, kept 1.00) | - |"]
    for a in ARMS:
        s = (sc or {}).get("arms", {}).get(a, {})
        g = (ga or {}).get("arms", {}).get(a, {})
        lv = [v for k, v in (sl or {}).get("arms", {}).items() if k.endswith(f"{a}_97")]
        rel, col, kept = s.get("share_min"), s.get("colour_min"), s.get("kept_min")
        rest = bool(s.get("rest")) and s.get("vx_max") is not None and s["vx_max"] <= 2.0
        gh = f"{g.get('held')}/{g.get('n')}" if g else "-"
        st = (f"{lv[0].get('status')} (share {lv[0].get('share_end', float('nan')):.2f}, kept {lv[0].get('kept', float('nan')):.2f}, "
              f"{lv[0].get('windows')} windows)") if lv else "-"
        same = (rel is not None and rel >= 0.40 and col is not None and col < 0.80 and kept is not None and kept >= 0.80
                and rest and g and g.get("held") == g.get("n") == 6 and st.startswith("PASS"))
        f = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
        rows.append(f"| {a} | {f(rel)} | {f(col)} | {f(kept)} | {'ok' if rest else 'no'} | {gh} | {st} | {'yes' if same else 'no'} |")
    L = ["# D174 runtime A/B: the result-changing candidates against Hu (nothing adopted; the user's call)", "", *rows, "",
         "Hu's verdicts (D172): relief >= 0.40 (0.45), colour < 0.80 (0.07; Hu's band stays grey), kept >= 0.80, rest ok, "
         "gallery 6/6 within 0.003 / 0.002 / 0.02 of the tag's rule, 300k scales PASS. A candidate keeps Hu's verdicts when every "
         "one of these is the same.", ""]
    for name, d in (("scenes", sc), ("gallery", ga), ("300k scales", sl)):
        L.append(f"- {name}: " + (d.get("verdict", "-") if d else "not read yet"))
    open(out_md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    mode, out = sys.argv[1], Path(sys.argv[2])
    if mode == "aggregate":
        aggregate(out)
    else:
        out.mkdir(parents=True, exist_ok=True)
        {"scenes": scenes, "gal": gal, "scales": scales}[mode](out)
