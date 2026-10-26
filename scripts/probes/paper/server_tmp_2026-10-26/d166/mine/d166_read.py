"""d166_read.py MODE ... -- D166 / D167 auto-summaries (CPU on hyde06). The relief / colour readings are D161's and D160's
readers, run unchanged (tmp/d161/d161_read.py, tmp/d160/d160_read.py); the gallery pieces are d162_gallery.pieces.
  wait   MODE2 OUT_DIR RUNDIR TITLE SPEC...  poll until every run the summary needs has its .rc (12 h at most)
  scenes OUT_DIR RUNDIR TITLE ARM:TWIN...    coin + bunny relief, colour band; 40k, seeds 97 / 99; RUNDIR/{scene}_{ARM}_{s}
  gallery OUT_DIR RUNDIR TITLE ARM:TWIN...   6 meshes 40k seed 97; RUNDIR/gal_{mesh}_{ARM}_97; refs: the tag's V / twin W
  big    OUT_DIR RUNDIR TITLE ARM:TWIN...    bunny + dragon 300k seed 97; RUNDIR/{mesh}300k_{ARM}_97; ref arm R (the tag)
  aggregate OUT_MD                           every summary + 'What to adopt'
TWIN: T = the tag twins (D161 runs40 coin / bunny, D160 runs40 band P; d155 / d154 W in the gallery), else a code in RUNDIR.
Each summary: OUT_DIR/summary.md (a table of <= 10 rows, a legend, one VERDICT line) and OUT_DIR/verdict.json."""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

B = Path("/data/relcfd/chayo/physmorph_v2")
G = B / "output/gpu"
A1 = G / "d161/assets"
PY = "/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python"
SEEDS = (97, 99)
SCN = {"coin": (A1 / "d161coin.obj", A1 / "d161L40coin.obj"), "bunny": (A1 / "d161smooth_bunny.obj", A1 / "d161L40_bunny.obj")}
TWIN0 = {"coin": "d161/runs40/coin40k_twin_{s}", "bunny": "d161/runs40/bunny40k_twin_{s}", "band": "d160/runs40/bunny40k_band_P{s}"}
MESHES = ("bunny", "dragon", "armadilo", "bob", "homer", "teapot")
BIG = ("bunny", "dragon")
AUTH_RELIEF, AUTH_COLOUR, REST_X, KEEP = 0.40, 0.80, 2.0, 0.80     # the criteria (stated in every summary)
FULL = "dF_c + layer u + render loss (full)"
TWIN = "dF_c + layer u (twin)"
NAMES = {  # arm code -> what it is (full arms unless the code starts with T)
    "Ul": "held weight + settle 30, u held at its last value", "Rx": "held weight + settle 30, no layer relaxation in the settle",
    "A10": "held + settle 30, assim 1.0", "A08": "held + settle 30, assim 0.8", "A025": "held + settle 30, assim 0.25",
    "UlA10": "held + settle 30, u held, assim 1.0", "RxA10": "held + settle 30, no relaxation in the settle, assim 1.0",
    "UlE3": "held + settle 30, u held, E x3", "RxE3": "held + settle 30, no relaxation in the settle, E x3",
    "R": "rule (tag default)", "RP": "rule + placement_at_cell", "H": "held (--lambda_ema 0)", "HP": "held + placement_at_cell",
    "F": "ref floor", "FP": "ref floor + placement_at_cell", "S": "ref shape", "SP": "ref shape + placement_at_cell",
    "X": "best weight definition x best D166 fix (chosen at run time, see the job's status.txt)",
    "T": "twin", "TP": "twin + placement_at_cell", "TA10": "twin, assim 1.0", "TA08": "twin, assim 0.8",
    "TA025": "twin, assim 0.25", "TE3": "twin, E x3"}


def name(code):
    return f"{TWIN}, {NAMES.get(code, code)}" if code.startswith("T") else f"{FULL}, {NAMES.get(code, code)}"


def fmt(x, f=".2f"):
    return "-" if x is None else format(x, f)


def cell(vals, f):
    return " / ".join("-" if v is None or v != v else format(v, f) for v in vals)


def run_rec(prefix):
    """The run's records: v_mean at the delivered end, the stop (last optimised window), its metrics; None if missing."""
    p = Path(f"{prefix}.json")
    if not p.exists():
        return None
    a = list(json.load(open(p))["arms"].values())[0]
    com = [h for h in a["history"] if h.get("frame_end") and not h.get("null_commit")]
    if not com:
        return None
    dn = a.get("deliver_n", com[-1]["frame_end"])
    end = next((h for h in com if h["frame_end"] == dn), com[-1])
    opt = [h for h in com if not h.get("settle")] or com
    m = a.get("metrics", {})
    return dict(v_end=end["v_mean"], stop_fe=opt[-1]["frame_end"], settled=len(opt) < len(com), windows=len(opt),
                sil=m.get("sil_iou"), ch=m.get("chamfer"), thin=m.get("thin_uncovered"), lam=opt[-1].get("lambda"),
                sec=a.get("seconds"))


def gl():
    sys.path.insert(0, str(B / "tmp/d162"))
    import d162_gallery as GL
    return GL


def twin_prefix(rundir, what, tw, s, mode):
    if mode == "scenes":
        return str(G / TWIN0[what].format(s=s)) if tw == "T" else f"{rundir}/{what}_{tw}_{s}"
    if mode == "gallery":
        if tw == "T":
            jp, _ = gl().paths(what, "W", s)
            return jp[:-5] if jp else f"{G}/d154/g40runs/{what}40k_W{s}"
        return f"{rundir}/gal_{what}_{tw}_{s}"
    return f"{rundir}/{what}300k_{tw}_{s}"


def parse(specs):
    return [tuple((sp.split(":") + ["T"])[:2]) for sp in specs]


def prefixes(mode, rundir, specs):
    out = []
    for arm, tw in parse(specs):
        if mode == "scenes":
            for sc in ("coin", "bunny", "band"):
                for s in SEEDS:
                    out += [f"{rundir}/{sc}_{arm}_{s}", twin_prefix(rundir, sc, tw, s, mode)]
        elif mode == "gallery":
            out += [f"{rundir}/gal_{m}_{arm}_97" for m in MESHES] + [twin_prefix(rundir, m, tw, 97, mode) for m in MESHES]
        else:
            out += [f"{rundir}/{m}300k_{a}_97" for m in BIG for a in (arm, tw, "R")]
    return sorted(set(out))


def wait(mode, out_dir, rundir, title, *specs):
    need = [p for p in prefixes(mode, rundir, specs) if p.startswith(str(rundir))]   # the refs outside RUNDIR exist
    t0, miss = time.time(), need
    while time.time() - t0 < 12 * 3600:
        miss = [p for p in need if not Path(f"{p}.rc").exists()]
        if not miss:
            return
        time.sleep(120)
    print("wait: timed out; missing " + " ".join(miss))


def scenes(out_dir, rundir, title, *specs):
    D = Path(out_dir)
    D.mkdir(parents=True, exist_ok=True)
    arms = parse(specs)
    procs = []
    for sc in ("coin", "bunny"):                    # D161's reader: one per scene, the job's full runs
        runs = {f"{a}{s}": f"{rundir}/{sc}_{a}_{s}" for a, _ in arms for s in SEEDS
                if Path(f"{rundir}/{sc}_{a}_{s}_frames12.npz").exists()}
        if not runs or (D / f"read_{sc}.json").exists():
            continue
        sm, rl = SCN[sc]
        json.dump(dict(scene=f"{sc}40k_d166", N=40000, smooth=str(sm), relief=str(rl), runs=runs), open(D / f"scene_{sc}.json", "w"))
        procs.append(subprocess.Popen([PY, str(B / "tmp/d161/d161_read.py"), str(D / f"read_{sc}.json"), str(D / f"scene_{sc}.json")],
                                      stdout=open(D / f"read_{sc}.log", "w"), stderr=subprocess.STDOUT))
    if not (D / "read_colour.jsonl").exists():      # D160's reader: the twins first, then the job's band runs
        L, seen = [], set()
        for a, tw in arms:
            for s in SEEDS:
                tp = twin_prefix(rundir, "band", tw, s, "scenes")
                tn = Path(tp).name
                if tn not in seen and Path(f"{tp}_frames12.npz").exists():
                    L.append(f"{tn} band {tp}_frames12.npz")
                    seen.add(tn)
        for a, tw in arms:
            for s in SEEDS:
                p = f"{rundir}/band_{a}_{s}"
                tn = Path(twin_prefix(rundir, "band", tw, s, "scenes")).name
                if Path(f"{p}_frames12.npz").exists() and f"band_{a}_{s}" != tn:
                    L.append(f"band_{a}_{s} band {p}_frames12.npz twin={tn} timeline")
        if L:
            open(D / "jobs_colour.txt", "w").write("\n".join(L) + "\n")
            procs.append(subprocess.Popen([PY, str(B / "tmp/d160/d160_read.py"), str(D / "read_colour.jsonl"), str(D / "jobs_colour.txt")],
                                          stdout=open(D / "read_colour.log", "w"), stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    rd = {sc: json.load(open(D / f"read_{sc}.json")) for sc in ("coin", "bunny") if (D / f"read_{sc}.json").exists()}
    col = ({json.loads(x)["name"]: json.loads(x) for x in open(D / "read_colour.jsonl") if x.strip()}
           if (D / "read_colour.jsonl").exists() else {})
    rows = ["| arm | scene | share stop 97/99 | share end | kept | v end / twin | sil_iou | chamfer | thin |",
            "|---|---|---|---|---|---|---|---|---|"]
    V = {}
    for a, tw in arms:
        ver = dict(share=[], colour=[], vx=[], kept=[], settled=False, sil=[])
        for sc in ("coin", "bunny", "band"):
            st, en, kp, vx, sil, ch, th = [], [], [], [], [], [], []
            for s in SEEDS:
                r, t = run_rec(f"{rundir}/{sc}_{a}_{s}"), run_rec(twin_prefix(rundir, sc, tw, s, "scenes"))
                a_stop = a_end = None
                if r is not None and sc in rd and f"{a}{s}" in rd[sc].get("runs", {}):
                    ceil = np.mean([v["relief"]["A"] for k, v in rd[sc]["refs"].items() if k.startswith("relief_s")])
                    ser = rd[sc]["runs"][f"{a}{s}"]["series"]
                    raws, A = np.array([q["raw"] for q in ser]), np.array([q["relief"]["A"] for q in ser]) / ceil
                    a_stop, a_end = float(np.interp(r["stop_fe"], raws, A)), float(A[-1])
                tn = Path(twin_prefix(rundir, "band", tw, s, "scenes")).name
                if r is not None and sc == "band" and f"band_{a}_{s}" in col and tn in col:
                    c = col[f"band_{a}_{s}"]
                    mt, fl = col[tn]["end"]["mis"], c["floor"]
                    rr = lambda m: (mt - m) / max(mt - fl, 1e-9)  # noqa: E731
                    tl = np.asarray(c.get("timeline") or [[0.0, c["end"]["mis"]]], np.float64)
                    a_stop = rr(float(np.interp(r["stop_fe"] / 240.0, tl[:, 0], tl[:, 1])))
                    a_end = rr(c["end"]["mis"])
                k = a_end / a_stop if (a_stop is not None and a_end is not None and a_stop > 0.05) else None
                st.append(a_stop)
                en.append(a_end)
                kp.append(k)
                vx.append(r["v_end"] / t["v_end"] if (r and t) else None)
                sil.append(r["sil"] if r else None)
                ch.append(r["ch"] if r else None)
                th.append(r["thin"] if r else None)
                ver["settled"] |= bool(r and r["settled"])
            ver["colour" if sc == "band" else "share"] += [x for x in en if x is not None]
            ver["kept"] += [x for x in kp if x is not None]
            ver["vx"] += [x for x in vx if x is not None]
            ver["sil"] += [x for x in sil if x is not None]
            rows.append(f"| {a} | {sc} | {cell(st, '.2f')} | {cell(en, '.2f')} | {cell(kp, '.2f')} | {cell(vx, '.1f')}x | "
                        f"{cell(sil, '.4f')} | {cell(ch, '.4f')} | {cell(th, '.3f')} |")
        sh, co = min(ver["share"], default=None), min(ver["colour"], default=None)
        ok_auth = sh is not None and co is not None and sh >= AUTH_RELIEF and co >= AUTH_COLOUR
        vmax = max(ver["vx"], default=None)
        ok_rest = vmax is not None and vmax <= REST_X
        kmin = min(ver["kept"], default=None) if ver["settled"] else None
        ok_keep = (kmin is not None and kmin >= KEEP) if ver["settled"] else None
        V[a] = dict(kind="scenes", authority=bool(ok_auth), rest=bool(ok_rest), kept=ok_keep, share_min=sh, colour_min=co,
                    vx_max=vmax, kept_min=kmin, sil_mean=float(np.mean(ver["sil"])) if ver["sil"] else None,
                    pass_=bool(ok_auth and ok_rest and (ok_keep is not False)))
    L = [f"# {title}", "", *rows, "",
         "Arms: " + "; ".join(f"{a} = {name(a)} (vs {name(tw) if tw != 'T' else TWIN})" for a, tw in arms) + ".",
         f"Share = relief amplitude / the target's own (D161 reader; coin, bunny), colour r (D160 reader; band); stop = the last "
         f"optimised window, end = delivered (after settling if settled); kept = end / stop. Criteria: authority = relief >= "
         f"{AUTH_RELIEF} and colour r >= {AUTH_COLOUR} on every run; rest = end v_mean <= {REST_X:g}x twin; kept >= {KEEP} "
         "(settled arms). Ref job 50 (held + settle, u zero): kept 0.29-0.43 relief, 1.00 colour.", ""]
    L.append("VERDICT: " + "; ".join(
        f"{a} {'PASS' if v['pass_'] else 'FAIL'} (authority {'ok' if v['authority'] else 'no'} min {fmt(v['share_min'])}/"
        f"{fmt(v['colour_min'])}, rest {'ok' if v['rest'] else 'no'} max {fmt(v['vx_max'], '.1f')}x"
        + (f", kept min {fmt(v['kept_min'])}" if v["kept"] is not None else "") + ")" for a, v in V.items()))
    open(D / "summary.md", "w").write("\n".join(L) + "\n")
    json.dump(dict(title=title, arms=V), open(D / "verdict.json", "w"), indent=1)


def gallery(out_dir, rundir, title, *specs):
    GL = gl()
    D = Path(out_dir)
    D.mkdir(parents=True, exist_ok=True)
    arms = parse(specs)

    def rd(prefix):
        r = run_rec(prefix)
        fr = glob.glob(f"{prefix}_frames12.npz")
        if r is not None and fr:
            z = np.load(fr[0], allow_pickle=True)
            r["p10"] = GL.pieces(np.asarray(z["frames"][-1], np.float64))[1]
        return r

    def f(r):
        return "-" if r is None else f"{r['sil']:.4f} / {r['ch']:.4f} / {r['thin']:.3f} / {r['v_end'] * 1e3:.1f} / {r.get('p10', '-')}"

    rows = ["| mesh | V (tag) | W (twin) | " + " | ".join(a for a, _ in arms) + " |", "|" + "---|" * (3 + len(arms))]
    V = {a: dict(kind="gallery", held=0, rest=True, n=0, miss=[]) for a, _ in arms}
    for m in MESHES:
        jv, _ = GL.paths(m, "V", 97)
        v = rd(jv[:-5]) if jv else None
        w = rd(twin_prefix(rundir, m, "T", 97, "gallery"))
        cells = []
        for a, tw in arms:
            r = rd(f"{rundir}/gal_{m}_{a}_97")
            t = w if tw == "T" else rd(twin_prefix(rundir, m, tw, 97, "gallery"))
            cells.append(r)
            if r is None or v is None:
                continue
            V[a]["n"] += 1
            ok = (r["sil"] >= v["sil"] - 0.003 and r["ch"] <= v["ch"] + 0.002 and r["thin"] <= v["thin"] + 0.02)
            V[a]["held"] += int(ok)
            if not ok:
                V[a]["miss"].append(m)
            if t is None or r["v_end"] > REST_X * t["v_end"]:
                V[a]["rest"] = False
        rows.append(f"| {m} | {f(v)} | {f(w)} | " + " | ".join(f(r) for r in cells) + " |")
    for a in V:
        V[a]["pass_"] = bool(V[a]["n"] == len(MESHES) and V[a]["held"] == V[a]["n"] and V[a]["rest"])
    L = [f"# {title}", "", *rows, "", "Cells: sil_iou / chamfer / thin / end v_mean x1e3 / pieces >= 10. Arms: "
         + "; ".join(f"{a} = {name(a)} (rest vs {name(tw) if tw != 'T' else TWIN})" for a, tw in arms)
         + f". Holds = sil_iou >= V - 0.003, chamfer <= V + 0.002, thin <= V + 0.02 on every mesh; rest = end v_mean <= "
         f"{REST_X:g}x twin.", "",
         "VERDICT: " + "; ".join(f"{a} {'PASS' if v['pass_'] else 'FAIL'} (holds {v['held']}/{v['n']}"
                                 + (f", misses {','.join(v['miss'])}" if v["miss"] else "") + f", rest {'ok' if v['rest'] else 'no'})"
                                 for a, v in V.items())]
    open(D / "summary.md", "w").write("\n".join(L) + "\n")
    json.dump(dict(title=title, arms=V), open(D / "verdict.json", "w"), indent=1)


def big(out_dir, rundir, title, *specs):
    D = Path(out_dir)
    D.mkdir(parents=True, exist_ok=True)
    arms = parse(specs)
    rows = ["| mesh | arm | sil_iou | chamfer | thin | v end / twin | windows | minutes | lambda last |", "|---|---|---|---|---|---|---|---|---|"]
    V = {a: dict(kind="big", held=0, rest=True, n=0) for a, _ in arms}
    for m in BIG:
        ref = run_rec(f"{rundir}/{m}300k_R_97")
        for a, tw in arms:
            r, t = run_rec(f"{rundir}/{m}300k_{a}_97"), run_rec(f"{rundir}/{m}300k_{tw}_97")
            if r is None:
                rows.append(f"| {m} | {a} | - | - | - | - | - | - | - |")
                V[a]["rest"] = False
                continue
            vx = r["v_end"] / t["v_end"] if t else None
            rows.append(f"| {m} | {a} | {fmt(r['sil'], '.4f')} | {fmt(r['ch'], '.4f')} | {fmt(r['thin'], '.3f')} | {fmt(vx, '.1f')}x | "
                        f"{r['windows']} | {(r['sec'] or 0) / 60:.0f} | {fmt(r['lam'], '.3g')} |")
            if vx is None or vx > REST_X:
                V[a]["rest"] = False
            if ref is not None and None not in (r["sil"], r["ch"], r["thin"]):
                V[a]["n"] += 1
                V[a]["held"] += int(r["sil"] >= ref["sil"] - 0.003 and r["ch"] <= ref["ch"] + 0.002 and r["thin"] <= ref["thin"] + 0.02)
    for a in V:
        V[a]["pass_"] = bool(V[a]["n"] == len(BIG) and V[a]["held"] == V[a]["n"] and V[a]["rest"])
    L = [f"# {title}", "", *rows, "", "Arms: " + "; ".join(f"{a} = {name(a)} (vs {name(tw)})" for a, tw in arms)
         + f". Holds = sil_iou >= R - 0.003, chamfer <= R + 0.002, thin <= R + 0.02 (R = the tag's rule at 300k, this run set); "
         f"rest = end v_mean <= {REST_X:g}x twin.", "",
         "VERDICT: " + "; ".join(f"{a} {'PASS' if v['pass_'] else 'FAIL'} (holds {v['held']}/{v['n']}, rest {'ok' if v['rest'] else 'no'})"
                                 for a, v in V.items())]
    open(D / "summary.md", "w").write("\n".join(L) + "\n")
    json.dump(dict(title=title, arms=V), open(D / "verdict.json", "w"), indent=1)


def aggregate(out_md):
    S = sorted(glob.glob(str(G / "d166/auto/*/summary.md")) + glob.glob(str(G / "d167/auto/*/summary.md")),
               key=lambda p: int(Path(p).parent.name.split("_")[0]))
    vs = {}
    for p in S:
        vj = Path(p).parent / "verdict.json"
        if vj.exists():
            for a, v in json.load(open(vj))["arms"].items():
                vs.setdefault(a, {})[v["kind"]] = v
    ef = B / "tmp/d166/expected.txt"
    exp = [x.strip() for x in open(ef) if x.strip()] if ef.exists() else []
    have = {Path(p).parent.name for p in S}
    L = ["# Render rest + weight definition (D166 / D167): master summary", "",
         f"Written {time.strftime('%Y-%m-%d %H:%M %Z')}; {len(have & set(exp))} of {len(exp)} job summaries present"
         + (f" (missing: {', '.join(e for e in exp if e not in have)})" if [e for e in exp if e not in have] else "") + ".", "",
         "## What to adopt", ""]
    p1 = {a: v["scenes"] for a, v in vs.items() if "scenes" in v and v["scenes"].get("kept") is not None}
    if p1:
        good = [a for a, v in p1.items() if v["pass_"]]
        best = max(p1, key=lambda a: (p1[a]["rest"], p1[a]["kept_min"] or 0))
        L.append("- D166 (rest under authority): " + (f"PASS: {'; '.join(a + ' = ' + name(a) for a in good)}." if good else
                 f"no arm passes (authority + rest <= {REST_X:g}x twin + kept >= {KEEP}); the best is {best} = {name(best)} "
                 f"(kept min {fmt(p1[best]['kept_min'])}, rest max {fmt(p1[best]['vx_max'], '.1f')}x)."))
        for a, v in p1.items():
            L.append(f"  - {a}: kept min {fmt(v['kept_min'])}, rest max {fmt(v['vx_max'], '.1f')}x, relief min {fmt(v['share_min'])}, "
                     f"colour min {fmt(v['colour_min'])}")
    p2 = {a: v for a, v in vs.items() if a in ("R", "RP", "H", "HP", "F", "FP", "S", "SP", "X")}
    if p2:
        ok = [a for a, v in p2.items() if v.get("scenes", {}).get("pass_") and v.get("gallery", {}).get("pass_", False)
              and v.get("big", {}).get("pass_", True)]
        L.append("- D167 (weight definition): " + (f"adopt {'; '.join(a + ' = ' + name(a) for a in ok)}: authority, rest and the "
                 "gallery hold (300k: " + ", ".join(f"{a} {'holds' if p2[a].get('big', {}).get('pass_') else 'not read'}" for a in ok) + ")."
                 if ok else "no definition meets authority + rest + gallery; per definition:"))
        for a, v in p2.items():
            sc, ga, bg = v.get("scenes", {}), v.get("gallery", {}), v.get("big", {})
            L.append(f"  - {a} ({NAMES.get(a, a)}): authority {sc.get('authority', '-')} (relief min {fmt(sc.get('share_min'))}, "
                     f"colour {fmt(sc.get('colour_min'))}), rest {sc.get('rest', '-')} ({fmt(sc.get('vx_max'), '.1f')}x); gallery "
                     f"{ga.get('held', '-')}/{ga.get('n', '-')} rest {ga.get('rest', '-')}; 300k {bg.get('held', '-')}/{bg.get('n', '-')} "
                     f"rest {bg.get('rest', '-')}")
    L += ["", f"Criteria: authority = relief share >= {AUTH_RELIEF} and colour r >= {AUTH_COLOUR} on every 40k run (held: coin "
          "0.55-0.59, bunny 0.66-0.83, colour 0.93); rest = end v_mean <= 2x twin; kept >= 0.8 after 30 settle windows; gallery "
          "holds = sil_iou / chamfer / thin within 0.003 / 0.002 / 0.02 of the tag on all 6 meshes; 300k against the tag rule R. "
          f"Arms: {FULL} vs {TWIN}.", ""]
    other = [p for pat in ("51_stiff", "53_*", "54_*", "55_*") for p in sorted(glob.glob(str(G / "d162/auto" / pat / "summary.md")))]
    for p in S + other:
        L += ["---", f"<!-- {p} -->", open(p).read().strip(), ""]
    tmp = Path(out_md + ".tmp")
    tmp.write_text("\n".join(L) + "\n")
    os.replace(tmp, out_md)
    print(f"{len(have & set(exp))} {len(exp)}")


if __name__ == "__main__":
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "wait":
        wait(*args)
    elif mode == "aggregate":
        aggregate(*args)
    else:
        {"scenes": scenes, "gallery": gallery, "big": big}[mode](*args)
