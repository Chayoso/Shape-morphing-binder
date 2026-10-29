"""d174_read3.py MODE OUT_DIR -- D174's stop_tol A/B readings (CPU on hyde06) with D172's readers on the fast readers
(tmp/d172/d172_read_fast.py patches them; criteria and verdicts unchanged): Hu with the stop's own tolerance
--stop_tol 0.NN (arm D174_ARM = HsNN, repo_d174s; the step anneal and the latch keep tol 0.003) against Hu.
  scenes  coin / bunny relief / band 40k x 97 / 99 (D166's scene reader), Hu's rows from D172 job 185
  gal     the 6-mesh 40k gallery, seed 97 (D170's gallery reader against the tag's rule)
  scales  the 300k stylized dragon scales (D172's detail reader; Hu and the twin read again beside them)
  plq     the 300k SIGGRAPH plaque (D163's reader through D170's carve; Hu read again beside them)
  big     the 300k dragon and bunny: wall, attempts, sil_iou, chamfer, thin, rest, against repo_d174's Hu timing runs
  aggregate OUT_MD"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d172")
import d172_read_fast  # noqa: E402,F401  (points D172's readers at the fast ones)
import d172_read as R2  # noqa: E402

R7, R6, G = R2.R7, R2.R6, R2.G
RUNS = G / "d174" / "runs"
ARM = os.environ.get("D174_ARM", "Hs10")          # Hu --stop_tol 0.NN (the stop's own tolerance; anneal and latch at tol 0.003)
ARMS = {ARM: f"Hu with the stop's own tolerance --stop_tol 0.{ARM[2:]}"}
for a, t in ARMS.items():
    R6.NAMES[a] = t + ", settle 30 (relaxation off in the settle)"
    R7.ARM_NAMES[a] = t
    R2.NAMES[a] = t
    R7.CARVE[f"plq{a}"] = dict(R7.CARVE["plq"], name=f"SIGGRAPH plaque, {a}", full=RUNS / f"plq300k_{a}_97")
R7.CARVE["plqHu"]["name"] = "SIGGRAPH plaque, Hu"
R7.JOB_SCENES["plq174s"] = ("plqHu",) + tuple(f"plq{a}" for a in ARMS)
SET40 = (ARM,)
BIG = {"dragon": (ARM,), "bunny": (ARM,)}


def scenes(D):
    R7.wait_rc([str(RUNS / f"{sc}_{a}_{s}") for sc in ("coin", "bunny", "band") for a in SET40 for s in (97, 99)])
    cwd = os.getcwd()
    R6.scenes(str(D), str(RUNS), "D174: the stop A/B against Hu, 40k scenes", *[f"{a}:T" for a in SET40])
    os.chdir(cwd)
    L = open(D / "summary.md").read().rstrip("\n").split("\n")
    refs = R2.ref_rows(G / "d172/auto/185_scenes/summary.md", ["Hu"])
    k = max(i for i, x in enumerate(L) if x.startswith("| "))
    open(D / "summary.md", "w").write("\n".join(L[:k + 1] + refs + L[k + 1:]
                                                + ["Reference rows: Hu (D172 job 185), the same scenes, seeds and readers."]) + "\n")


def gal(D):
    R7.RUNS = RUNS
    V, L = {}, ["# D174: the stop A/B against Hu, the 6-mesh 40k gallery, seed 97", ""]
    for a in SET40:
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
    R7.DETAIL["scl300_ss"] = dict(R7.DETAIL["scl300"], runs=runs)
    R7.JOB_DETAIL["scl300_ss"] = ("scl300_ss",)
    R2.detail("scl300_ss", D)


def plq(D):
    R2.carve("plq174s", D)


def _wall(name):
    """Seconds of a run from output/gpu/d174/status.txt (its 'R NAME exit rc (S s)' line)."""
    for line in open(G / "d174" / "status.txt"):
        m = re.search(rf"R {re.escape(name)} exit (\d+) \((\d+) s\)", line)
        if m:
            return int(m.group(2))
    return None


def big(D):
    R7.wait_rc([str(RUNS / f"{m}300k_{a}_97") for m, arms in BIG.items() for a in arms])
    rows = ["| mesh | arm | wall | attempts (kept) | sil_iou | chamfer | thin | rest (G3, drift_rel) | end v_mean |",
            "|---|---|---|---|---|---|---|---|---|"]
    V = {}
    for m in ("dragon", "bunny"):
        for a, name in [("Hu", f"T_{m}300k_Hu_d174")] + [(a, f"{m}300k_{a}_97") for a in BIG[m]]:
            p = RUNS / f"{name}.json"
            if not p.exists():
                rows.append(f"| {m} | {a} | - | - | - | - | - | - | - |")
                continue
            arm = list(json.load(open(p))["arms"].values())[0]
            me, h, g = arm["metrics"], arm["history"], arm["gates"]
            att = [r for r in h if not r.get("settle")]
            kept = sum(1 for r in att if not r.get("null_commit") and not r.get("grad_converged"))
            vm = [r for r in h if r.get("v_mean") is not None][-1]["v_mean"]
            w = _wall(name)
            V[f"{m}_{a}"] = dict(wall=w, attempts=len(att), kept=kept, sil=me["sil_iou"], ch=me["chamfer"],
                                 thin=me.get("thin_uncovered"), rest=bool(g.get("G3_rest")), drift=g.get("drift_rel"), v_end=vm)
            rows.append(f"| {m} | {a} | {w / 60 if w else float('nan'):.1f} min | {len(att)} ({kept}) | {me['sil_iou']:.4f} | "
                        f"{me['chamfer']:.4f} | {me.get('thin_uncovered', float('nan')) * 100:.1f} % | "
                        f"{'ok' if g.get('G3_rest') else 'no'} ({g.get('drift_rel', float('nan')):.1e}) | {vm:.2e} |")
    L = ["# D174: the stop A/B, 300k dragon and bunny (solo, seed 97)", "", *rows, "",
         "Hu = repo_d174's timing runs (tol 0.003, the exact speed-ups); the arm = repo_d174s with --stop_tol (tol 0.003 kept); "
         "each run alone on a card; wall from the job's clock (prepare to save)."]
    open(D / "summary.md", "w").write("\n".join(L) + "\n")
    json.dump(dict(arms=V), open(D / "verdict.json", "w"), indent=1, default=float)


def aggregate(out_md):
    A = G / "d174" / "auto"
    L = [f"# D174: the stop's own tolerance ({ARM}: --stop_tol 0.{ARM[2:]}, the anneal and the latch at tol 0.003) against Hu", ""]
    for name in ("ss_big", "ss_scales", "ss_plq", "ss_gal", "ss_scenes"):
        p = A / name / "summary.md"
        L.append(f"## {name}")
        L += (open(p).read().splitlines()[2:] if p.exists() else ["not read yet"]) + [""]
    open(out_md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    mode, out = sys.argv[1], Path(sys.argv[2])
    if mode == "aggregate":
        aggregate(out)
    else:
        out.mkdir(parents=True, exist_ok=True)
        {"scenes": scenes, "gal": gal, "scales": scales, "plq": plq, "big": big}[mode](out)
