"""d172_read.py MODE OUT_DIR -- D172's auto-summaries (CPU on hyde06), on D170's readers (tmp/d170/d170_read.py) and
D166's scene reader (tmp/d166/d166_read.py), unchanged in their readings.
  scenes     B: Hu / Hf on the 40k coin / bunny relief / band, seeds 97 / 99 (D166's reader), with Rx's and R's rows
  gal        B: Hu / Hf on the 6-mesh 40k gallery, seed 97 (seed 99 in OUT_DIR/s99 when job 191 ran)
  plq        B: Hu / Hf on the 300k SIGGRAPH plaque (D163's reader)
  combined   C: the relief + paint scene (D161's relief reader and D160's colour reader on the same runs)
  scales40 | scales300   D: the stylized dragon scales (d172_detail.py: D161's reading inside the pattern)
  aggregate OUT_MD   output/gpu/D172_SUMMARY.md
Arms: Hu = dF_c + layer u + render loss (full), the held weight on u and the rule's per-window weight on dF_c; Hf = the reverse;
both with Rx's settle (--lambda_ema 0 --settle_windows 30 --no-settle_relax); the merit keeps the held weight."""
import glob
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d170")
import d170_read as R7  # noqa: E402
R6 = R7.R6
B, G = R7.B, R7.G
O = G / "d172"
RUNS = O / "runs"
A1, A2 = R7.A1, O / "assets"
PY, FULL, TWIN = R7.PY, R7.FULL, R7.TWIN
fmt, run_rec, sample_at, vx = R7.fmt, R7.run_rec, R7.sample_at, R7.vx
SHARE, COLOUR, KEEP, REST_X = 0.40, 0.80, 0.80, 2.0
NAMES = {"Hu": "held weight on u, the rule's on dF_c", "Hf": "held weight on dF_c, the rule's on u", "Rx": "held weight on both",
         "T": "twin"}
R7.ARM_NAMES.update(Hu="Hu (held on u, rule on dF_c)", Hf="Hf (held on dF_c, rule on u)", Rx="Rx (held on both)")
R6.NAMES.update(Hu="held on u, the rule on dF_c, settle 30 (relaxation off in the settle)",
                Hf="held on dF_c, the rule on u, settle 30 (relaxation off in the settle)")
VAR = (A2 / "variant.txt").read_text().strip() if (A2 / "variant.txt").exists() else ""


def _rebind(fn, reps):
    """D170's reader function with D172's paths (the source text replaced), its globals D170's module."""
    src = inspect.getsource(fn)
    for a, b in reps:
        src = src.replace(a, b)
    ns = {}
    exec(compile(src, f"<d172:{fn.__name__}>", "exec"), R7.__dict__, ns)
    return ns[fn.__name__]


detail = _rebind(R7.detail, [('"tmp/d170/d170_detail.py"', '"tmp/d172/d172_detail.py"'), ("D170 {job}", "D172 {job}"),
                             ('for lab in have}), open(spec, "w"))', 'for lab in have}, s_bg=sc.get("s_bg", 4.0)), open(spec, "w"))'),
                             ("both high-passed at 4 pitches", "both high-passed at half the scale spacing")])
carve = _rebind(R7.carve, [("D170 {job}: 300k carve scenes, Rx vs twin", "D172 {job}: 300k plaque, Hu / Hf vs twin")])
SCL = {"C": 5, "D": 6, "E": 6, "F": 8, "G": 8}
for n, k in ((40000, "scl40"), (300000, "scl300")):
    arms = ("T", "Rx", "Hu", "Hf") if n == 40000 else tuple(os.environ.get("D172_SCL300_ARMS", "T Hu").split())
    seeds = (97, 99) if n == 40000 else (97,)
    R7.DETAIL[k] = dict(name=f"stylized dragon scales {VAR}, {n // 1000}k", N=n, tgt=G / "d170/assets/d170smooth_dragon.obj",
                        smooth=G / "d170/assets/d170smooth_dragon.obj", detail=A2 / f"d172S{VAR}{n // 1000}_dragon.obj",
                        s_bg=SCL.get(VAR, 4) / 2.0,
                        runs={f"{a}_{s}": RUNS / f"scl{n // 1000}k_{a}_{s}" for a in arms for s in seeds})
R7.JOB_DETAIL.update(scales40=("scl40",), scales300=("scl300",))
for a in ("Hu", "Hf"):
    R7.CARVE[f"plq{a}"] = dict(R7.CARVE["plq"], name=f"SIGGRAPH plaque, {a}", full=RUNS / f"plq300k_{a}_97")
R7.JOB_SCENES["plq"] = ("plqHu", "plqHf")


def ref_rows(path, codes):
    try:
        return [x for x in open(path).read().splitlines() if any(x.startswith(f"| {c} |") for c in codes)]
    except OSError:
        return []


def scenes(D):
    R7.wait_rc([str(RUNS / f"{sc}_{a}_{s}") for sc in ("coin", "bunny", "band") for a in ("Hu", "Hf") for s in (97, 99)])
    cwd = os.getcwd()
    R6.scenes(str(D), str(RUNS), "D172 B: the render's authority by control block, 40k scenes (Hu / Hf)", "Hu:T", "Hf:T")
    os.chdir(cwd)
    L = open(D / "summary.md").read().rstrip("\n").split("\n")
    refs = ref_rows(G / "d166/auto/62_relaxoff/summary.md", ["Rx"]) + ref_rows(G / "d167/auto/104_rule/summary.md", ["R"])
    k = max(i for i, x in enumerate(L) if x.startswith("| "))
    L = L[:k + 1] + refs + L[k + 1:]
    L.append("Reference rows: Rx (D166 job 62, held on both) and R (D167 job 104, the rule on both, no settle), same scenes / "
             "seeds / readers.")
    open(D / "summary.md", "w").write("\n".join(L) + "\n")


def gal(D):
    R7.RUNS = RUNS
    for a in ("Hu", "Hf"):
        (D / a).mkdir(parents=True, exist_ok=True)
        R7.gal(D / a, a, (97,), f"D172 B: {a} ({NAMES[a]}) on the 6-mesh 40k gallery, seed 97")
        if all(Path(f"{RUNS}/gal_{m}_{a}_99.rc").exists() for m in R7.MESHES):
            (D / f"{a}_99").mkdir(parents=True, exist_ok=True)
            R7.gal(D / f"{a}_99", a, (97, 99), f"D172 B: {a} on the 6-mesh 40k gallery, seeds 97 / 99")
    V, L = {}, ["# D172 B: Hu / Hf on the 40k gallery", ""]
    for a in ("Hu", "Hf"):
        sub = D / f"{a}_99" if (D / f"{a}_99/verdict.json").exists() else D / a
        V[a] = json.load(open(sub / "verdict.json"))["arms"][a]
        L += [x for x in open(sub / "summary.md").read().splitlines()[2:] if x.strip()] + [""]
    verdict = "; ".join(f"{a} {v['status']} (holds {v['held']}/{v['n']}" + (f", misses {','.join(v['miss'])}" if v["miss"] else "")
                        + f"; rest {'ok' if v['rest'] else 'no'})" for a, v in V.items())
    L.append("VERDICT: " + verdict)
    open(D / "summary.md", "w").write("\n".join(L) + "\n")
    json.dump(dict(arms=V, verdict=verdict), open(D / "verdict.json", "w"), indent=1, default=float)


def plq(D):
    carve("plq", D)


def combined(D):
    labs = {f"{a}{s}": RUNS / f"cmb_{a}_{s}" for a in ("T", "Rx", "Hu", "Hf") for s in (97, 99)}
    R7.wait_rc([str(p) for p in labs.values()])
    recs = {k: run_rec(str(p)) for k, p in labs.items()}
    have = {k: p for k, p in labs.items() if recs[k] is not None and Path(f"{p}_frames12.npz").exists()}
    rel, col = D / "read_relief.json", D / "read_colour.jsonl"
    if not rel.exists():
        json.dump(dict(scene="cmb40k_d172", N=40000, smooth=str(A1 / "d161smooth_bunny.obj"), relief=str(A1 / "d161L40_bunny.obj"),
                       runs={k: str(p) for k, p in have.items()}), open(D / "scene_relief.json", "w"))
        R7.reader([PY, str(B / "tmp/d161/d161_read.py"), str(rel), str(D / "scene_relief.json")], D / "read_relief.log")
    if not (col.exists() and col.stat().st_size):
        L = [f"cmb_T_{s} band {labs[f'T{s}']}_frames12.npz mesh=bunny" for s in (97, 99) if f"T{s}" in have]
        L += [f"cmb_{a}_{s} band {labs[f'{a}{s}']}_frames12.npz mesh=bunny twin=cmb_T_{s} timeline"
              for a in ("Rx", "Hu", "Hf") for s in (97, 99) if f"{a}{s}" in have]
        (D / "jobs_colour.txt").write_text("\n".join(L) + "\n")
        R7.reader([PY, str(B / "tmp/d160/d160_read.py"), str(col), str(D / "jobs_colour.txt")], D / "read_colour.log")
    rd = json.load(open(rel))
    cl = {json.loads(x)["name"]: json.loads(x) for x in open(col) if x.strip()}
    cA = float(np.mean([v["relief"]["A"] for k, v in rd["refs"].items() if k.startswith("relief_s")]))
    rows = ["| arm | seed | relief stop | relief end | kept | colour r stop | colour r end | kept | v end / twin | sil_iou arm / twin | thin arm / twin |",
            "|" + "---|" * 11]
    V = {}
    for a in ("Rx", "Hu", "Hf"):
        v = dict(kind="combined", arm=a, rel_end=[], col_end=[], kept=[], vx=[], ok=True)
        for s in (97, 99):
            r, t = recs.get(f"{a}{s}"), recs.get(f"T{s}")
            if r is None or t is None or f"{a}{s}" not in rd["runs"] or f"cmb_{a}_{s}" not in cl:
                rows.append(f"| {a} | {s} | - | - | - | - | - | - | - | - | - |")
                v["ok"] = False
                continue
            ser = [q for q in rd["runs"][f"{a}{s}"]["series"] if q["relief"].get("A") is not None]
            raws, vals = [q["raw"] for q in ser], [q["relief"]["A"] / cA for q in ser]
            rs, _ = sample_at(raws, vals, r["stop_fe"] - 1)
            re_ = vals[-1]
            c, ct = cl[f"cmb_{a}_{s}"], cl[f"cmb_T_{s}"]
            mt, fl = ct["end"]["mis"], c["floor"]
            rr = lambda m: (mt - m) / (mt - fl)  # noqa: E731
            tl = np.asarray(c.get("timeline", []), np.float64)
            sm, _ = sample_at(tl[:, 0] * 240, tl[:, 1], r["stop_fe"] - 1) if tl.ndim == 2 else (None, False)
            cs, ce = (rr(sm) if sm is not None else None), rr(c["end"]["mis"])
            k1 = re_ / rs if rs and rs > 0.05 else None
            k2 = ce / cs if cs and cs > 0.05 else None
            ratio = vx(r, t)
            v["rel_end"].append(re_); v["col_end"].append(ce); v["kept"] += [k1, k2]; v["vx"].append(ratio)
            v["ok"] &= bool(re_ >= SHARE and ce >= COLOUR and None not in (k1, k2) and min(k1, k2) >= KEEP and ratio is not None and ratio <= REST_X)
            rows.append(f"| {a} | {s} | {fmt(rs)} | {fmt(re_)} | {fmt(k1)} | {fmt(cs)} | {fmt(ce)} | {fmt(k2)} | {fmt(ratio, '.2f')}x | "
                        f"{r['sil']:.4f} / {t['sil']:.4f} | {r['thin']:.3f} / {t['thin']:.3f} |")
        v["status"] = "PASS" if v["ok"] else "FAIL"
        V[a] = v
    verdict = "; ".join(f"{a} {v['status']} (relief min {fmt(min([x for x in v['rel_end'] if x is not None], default=None))}, colour min "
                        f"{fmt(min([x for x in v['col_end'] if x is not None], default=None))})" for a, v in V.items())
    L = ["# D172 C: the combined relief + paint scene (bunny 40k: the pictures D161's 'PM' relief painted with D160's band)", "",
         *rows, "", f"Arms: {FULL} as named, with Rx's settle; twin = {TWIN} with the same pictures and paint. Relief share = "
         "D161's reader (the relief amplitude / the relief mesh's own 40k samples); colour r = D160's reader ((twin mis - mis) / "
         "(twin mis - floor)). Criteria: relief >= 0.40 and colour r >= 0.80 at both seeds, kept >= 0.80, end v_mean <= 2x twin.",
         "", "VERDICT: " + verdict]
    open(D / "summary.md", "w").write("\n".join(L) + "\n")
    json.dump(dict(arms=V, verdict=verdict), open(D / "verdict.json", "w"), indent=1, default=float)


def scales(job, D):
    detail(job, D)


def aggregate(out_md):
    S = {Path(p).parent.name: json.load(open(p)) for p in sorted(glob.glob(str(O / "auto/*/verdict.json")))}
    get = lambda j: S.get(j, {}).get("arms", {})  # noqa: E731
    L = ["# D172: the render's authority split by control block (B), the combined relief + paint scene (C), stylized dragon "
         "scales (D)", ""]
    # B: the table the coordinator asked for
    sc, ga, pq = get("185_scenes"), get("186_gal"), get("187_plq")
    rx6 = json.load(open(G / "d166/auto/62_relaxoff/verdict.json"))["arms"].get("Rx", {})
    r7 = json.load(open(G / "d167/auto/104_rule/verdict.json"))["arms"].get("R", {})
    rxg = json.load(open(G / "d170/auto/174_gal/verdict.json"))["arms"].get("Rx", {})
    rg = json.load(open(G / "d167/auto/114_gal_rule/verdict.json"))["arms"].get("R", {})
    rxp = json.load(open(G / "d170/auto/170_carve_a/verdict.json"))["arms"].get("plq", {})
    rows = ["| arm | relief min (coin, bunny) | colour min (band) | kept min | rest max (x twin) | gallery 40k s97 holds | "
            "300k plaque share end / kept | 300k plaque shape (sil_iou, thin vs twin) |", "|" + "---|" * 8]

    def row(name, s, g, p):
        sh = (f"{fmt(p.get('share_end'))} / {fmt(p.get('kept'))}" if p.get("share_end") is not None else "-")
        shp = (f"{'ok' if p.get('shape_ok') else 'no'} ({p['sil'][0]:.4f} vs {p['sil'][1]:.4f}, {p['thin'][0]:.3f} vs {p['thin'][1]:.3f})"
               if p.get("sil") else "-")
        gh = f"{g.get('held', '-')}/{g.get('n', '-')}" + (f" (misses {','.join(g['miss'])})" if g.get("miss") else "") if g else "-"
        return (f"| {name} | {fmt(s.get('share_min'))} | {fmt(s.get('colour_min'))} | {fmt(s.get('kept_min'))} | "
                f"{fmt(s.get('vx_max'), '.1f')} | {gh} | {sh} | {shp} |")
    rows += [row("Hu (held on u, rule on dF_c)", sc.get("Hu", {}), ga.get("Hu", {}), pq.get("plqHu", {})),
             row("Hf (held on dF_c, rule on u)", sc.get("Hf", {}), ga.get("Hf", {}), pq.get("plqHf", {})),
             row("Rx (held on both; D166 / D170)", rx6, dict(rxg, n=rxg.get("n"), held=rxg.get("held")), rxp),
             row("R (the rule on both; D167, no settle)", r7, rg, {})]
    L += ["## B: which channel carries the content, which breaks arrival?", "", *rows, "",
          "Content = relief share (D161's reader) at the end, min over coin / bunny x seeds 97 / 99; colour r (D160's reader) on the "
          "band; kept = end / stop through the settle; rest = end v_mean / twin; gallery holds = sil_iou / chamfer / thin within "
          "0.003 / 0.002 / 0.02 of the tag's rule (R 97, d167 job 114) per mesh; the 300k plaque against its twin (d163). R's "
          "gallery row is against the tag's V (d167 job 114's own reading). Adoption: none here (the user's call).", ""]
    for j, t in (("185_scenes", "B scenes"), ("186_gal", "B gallery"), ("187_plq", "B plaque 300k"), ("188_combined", "C combined"),
                 ("189_scales40", "D scales 40k"), ("190_scales300", "D scales 300k")):
        v = S.get(j, {}).get("verdict")
        if v is None and (O / f"auto/{j}/summary.md").exists():
            v = next((x[9:] for x in open(O / f"auto/{j}/summary.md").read().splitlines() if x.startswith("VERDICT: ")), None)
        L.append(f"- {t}: {v or 'not read yet'}")
    L += ["", "## D: representability of the stylized scales (no run; a perfect N-sample of the relief read by d172_detail.py "
          "inside the pattern; the gate: corr >= 0.25 at 300k)", ""]
    for p in sorted(glob.glob(str(O / "size_probe/S*.json"))):
        try:
            d = json.load(open(p))
            c, r = d["ctx"]["97"], d["refs"]["detail_s197"]["relief"]
            L.append(f"- {Path(p).stem}: pitch {c['pitch']:.4f} wu, rms height {c['rms_H_pitch']:.2f} pitch, half-correlation "
                     f"{fmt(c['autocorr_half_pitch'])} pitch; a perfect sample A {r['A']:.2f}, corr {r['corr']:.2f}")
        except (OSError, KeyError, ValueError):
            pass
    L += [f"- chosen variant: '{VAR}'", "",
          f"Written {time.strftime('%Y-%m-%d %H:%M %Z')}. Code: repo_d172 = wip/d172 4a0b2537 (--render_split, the render_x paint fix).", ""]
    for p in sorted(glob.glob(str(O / "auto/*/summary.md"))):
        L += ["---", f"<!-- {p} -->", open(p).read().strip(), ""]
    tmp = Path(out_md + f".{os.getpid()}.tmp")
    tmp.write_text("\n".join(L) + "\n")
    os.replace(tmp, out_md)


if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    if mode == "aggregate":
        aggregate(out)
    else:
        D = Path(out)
        D.mkdir(parents=True, exist_ok=True)
        {"scenes": scenes, "gal": gal, "plq": plq, "combined": combined,
         "scales40": lambda D: scales("scales40", D), "scales300": lambda D: scales("scales300", D)}[mode](D)
