"""d170_read.py MODE OUT_DIR -- D170's auto-summaries (CPU on hyde06): Rx = dF_c + layer u + render loss (full) with the
weight held (--lambda_ema 0), then 30 zero-control windows with the outer-layer relaxation off (--settle_windows 30
--no-settle_relax), on repo_d166 (no code change). The readers are D161's / D163's / D160's, run unchanged, and D170's
detail reader (d170_detail.py: D161's reading with the detailed mesh's height in place of the letters' pattern).
  carve_a | carve_b   the 300k carve scenes (plaque, ripples | coin "PM", bunny flank "PM"): relief share at the stop / end
  wrinkle             the 300k mesh-relief pictures (D159's mesh_shell: bunny wrinkles, dragon scales): detail share
  dragon40 | dragon300  the dragon with render-only relief (3D target the smoothed dragon, pictures the detailed one)
  band300             the D160 colour band at 300k: colour r
  gal | big           the 40k gallery (seeds 97 / 99) | 300k bunny + dragon: shape against the tag's rule and the twin
  aggregate OUT_MD    output/gpu/D170_SUMMARY.md ("Adopt Rx?" first)
Each mode writes OUT_DIR/summary.md (<= 10 table rows, a legend, one VERDICT line) and OUT_DIR/verdict.json."""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d166")
import d166_read as R6  # noqa: E402  (run_rec, sample_at, finite, fmt; its criteria constants)

B = Path("/data/relcfd/chayo/physmorph_v2")
G = B / "output/gpu"
O = G / "d170"
RUNS = O / "runs"
A1 = G / "d161/assets"
A7 = O / "assets"
REPO = B / "repo_d166"
PY = "/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python"
FULL, TWIN = "dF_c + layer u + render loss (full)", "dF_c + layer u (twin)"
RXN = "weight held (--lambda_ema 0), then 30 zero-control windows with the outer-layer relaxation off (--settle_windows 30 --no-settle_relax)"
SHARE, COLOUR, REST_X, KEEP = 0.40, 0.80, 2.0, 0.80
SH_TOL = (0.003, 0.002, 0.02)
MESHES = ("bunny", "dragon", "armadilo", "bob", "homer", "teapot")
fmt, finite, run_rec, sample_at = R6.fmt, R6.finite, R6.run_rec, R6.sample_at


def _status_ok(path):
    """An exit-status file: 'CODE' or 'CODE SECONDS' (D163's chain writes the seconds beside the code)."""
    f = path.read_text().split()
    return len(f) in (1, 2) and all(s.isascii() and s.isdecimal() for s in f) and int(f[0]) == 0


R6.successful_status = _status_ok
SHAPE_GATE = (0.005, 0.05)      # carve / detail scenes: sil_iou >= twin - 0.005 and thin <= twin + 0.05 (the body arrived)

CARVE = {   # scene: reader, smooth (3D target), relief (render target), the Rx run, its twin
    "plq": dict(reader="d163", name="SIGGRAPH plaque", smooth=G / "d163/assets/d163plaque.obj",
                relief=G / "d163/assets/d163R300plaque.obj", full=RUNS / "plq300k_Rx_97", twin=G / "d163/runs300/plq300k_T_97"),
    "rip": dict(reader="d161", name="D162f ripples", smooth=A1 / "d161smooth_bunny.obj",
                relief=G / "d162f/assets/d162fR300ripple_bunny.obj", full=RUNS / "rip300k_Rx_97", twin=G / "d162f/runs/rip300k_T_97"),
    "coin": dict(reader="d161", name='coin "PM"', smooth=A1 / "d161coin.obj", relief=A7 / "d170L300coin.obj",
                 full=RUNS / "coin300k_Rx_97", twin=RUNS / "coin300k_T_97"),
    "rel": dict(reader="d161", name='bunny flank "PM"', smooth=A1 / "d161smooth_bunny.obj", relief=A7 / "d170L300_bunny.obj",
                full=RUNS / "rel300k_Rx_97", twin=G / "d162f/runs/rip300k_T_97"),
}
CARVE["plqNS"] = dict(CARVE["plq"], name="SIGGRAPH plaque, Rx without the reject stop", full=RUNS / "plq300k_RxNS_97")
JOB_SCENES = {"carve_a": ("plq", "rip"), "carve_b": ("coin", "rel"), "nostop_plq": ("plqNS",)}
DETAIL = {  # scene: N, tgt (frame), smooth canvas, detail mesh, runs {label: prefix}
    "wrkb": dict(name="bunny skin wrinkles (mesh_shell pictures)", N=300000, tgt=REPO / "assets/bunny.obj",
                 smooth=A1 / "d161smooth_bunny.obj", detail=REPO / "assets/bunny.obj",
                 runs={"Rx_97": RUNS / "wrkb300k_Rx_97", "T_97": G / "d167/runs/bunny300k_T_97", "R_97": G / "d167/runs/bunny300k_R_97"}),
    "wrkd": dict(name="dragon scales (mesh_shell pictures)", N=300000, tgt=REPO / "assets/dragon.obj",
                 smooth=A7 / "d170smooth_dragon.obj", detail=REPO / "assets/dragon.obj",
                 runs={"Rx_97": RUNS / "wrkd300k_Rx_97", "T_97": G / "d167/runs/dragon300k_T_97", "R_97": G / "d167/runs/dragon300k_R_97"}),
    "dsm40": dict(name="dragon, render-only relief, 40k", N=40000, tgt=A7 / "d170smooth_dragon.obj",
                  smooth=A7 / "d170smooth_dragon.obj", detail=REPO / "assets/dragon.obj",
                  runs={f"{a}_{s}": RUNS / f"dsm40k_{a}_{s}" for a in ("T", "Rx", "Rxm", "H0", "H0m", "R0m") for s in (97, 99)}),
    "dsm300": dict(name="dragon, render-only relief, 300k", N=300000, tgt=A7 / "d170smooth_dragon.obj",
                   smooth=A7 / "d170smooth_dragon.obj", detail=REPO / "assets/dragon.obj",
                   runs={f"{a}_97": RUNS / f"dsm300k_{a}_97" for a in os.environ.get("D170_DSM300_ARMS", "T Rxm").split()}),
}
ARM_NAMES = {"H0mr15": "held, relaxation off, mesh shell, exterior radius 1.5 pitches", "RxNS": "Rx without the reject stop (--reject_stop 0)", "Rx": "Rx (pictures from samples)", "Rxm": "Rx, pictures from the mesh shell", "H0": "held, relaxation off, samples",
             "H0m": "held, relaxation off, mesh shell", "R0m": "rule, relaxation off, mesh shell", "R0": "rule, relaxation off",
             "R": "tag (rule)"}
JOB_DETAIL = {"wrinkle": ("wrkb", "wrkd"), "dragon40": ("dsm40",), "dragon300": ("dsm300",)}
ARMS = f"Arms: Rx = {FULL}, {RXN}; twin = {TWIN} (lambda = 0)."


def wait_rc(prefixes, hours=12):
    t0 = time.time()
    while time.time() - t0 < hours * 3600:
        miss = [p for p in prefixes if not Path(f"{p}.rc").exists() and not Path(p).parent.joinpath(f"rc_{Path(p).name}").exists()]
        if not miss:
            return []
        time.sleep(120)
    return miss


def reader(cmd, log):
    env = dict(os.environ)
    rc = subprocess.call(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, cwd=str(REPO), env=env)
    if rc != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} exited {rc} (log {log})")


def vx(r, t):
    return r["v_end"] / t["v_end"] if r and t and t["v_end"] > 0 else None


def write(D, title, rows, legend, verdict, V, extra=()):
    L = [f"# {title}", "", *rows, "", legend, "", "VERDICT: " + verdict, *extra]
    open(D / "summary.md", "w").write("\n".join(L) + "\n")
    json.dump(dict(title=title, arms=V, verdict=verdict), open(D / "verdict.json", "w"), indent=1, default=float)


# ------------------------------------------------------------------------------------------------------------ carve
def carve(job, D):
    keys = JOB_SCENES[job]
    miss = wait_rc([str(CARVE[k][p]) for k in keys for p in ("full", "twin")])
    rows = ["| scene | share stop | share end | kept | twin share end | corr end | display share end | v end / twin | "
            "sil_iou full / twin | chamfer full / twin | thin full / twin | pieces>=10 |", "|" + "---|" * 12]
    V = {}
    for k in keys:
        sc = CARVE[k]
        r, t = run_rec(str(sc["full"])), run_rec(str(sc["twin"]))
        out = D / f"read_{k}.json"
        if r is not None and t is not None and not out.exists():
            spec = D / f"scene_{k}.json"
            if sc["reader"] == "d161":
                json.dump(dict(scene=f"{k}300k_d170", N=300000, smooth=str(sc["smooth"]), relief=str(sc["relief"]),
                               runs={"full97": str(sc["full"]), "twin97": str(sc["twin"])}), open(spec, "w"))
                reader([PY, str(B / "tmp/d161/d161_read.py"), str(out), str(spec)], D / f"read_{k}.log")
            else:
                json.dump(dict(scene=f"{k}300k_d170", N=300000, smooth=str(sc["smooth"]), relief=str(sc["relief"]),
                               runs={"Rx_97": str(sc["full"]), "T_97": str(sc["twin"])}, pairs=[["Rx_97", "T_97"]]), open(spec, "w"))
                reader([PY, str(B / "tmp/d163/d163_read.py"), str(out), str(spec)], D / f"read_{k}.log")
        v = dict(kind="carve", scene=sc["name"], status="INCOMPLETE")
        if r is None or t is None or not out.exists():
            rows.append(f"| {sc['name']} | - | - | - | - | - | - | - | - | - | - | - |")
            V[k] = v
            continue
        rd = json.load(open(out))
        fl, tw = ("full97", "twin97") if sc["reader"] == "d161" else ("Rx_97", "T_97")
        refs = [x["relief"]["A"] for kk, x in rd["refs"].items() if kk.startswith("relief_s") and kk.endswith("97")]
        cA = float(np.mean(refs))
        dref = rd["refs"].get("relief_s197", {}).get("display", {}).get("A")
        ser = rd["runs"][fl]["series"]
        raws = [q["raw"] for q in ser]
        vals = [q["relief"]["A"] / cA for q in ser]
        s_stop, _ = sample_at(raws, vals, r["stop_fe"] - 1)
        s_end = vals[-1] if raws[-1] == r["deliver_n"] - 1 else None
        kept = s_end / s_stop if s_stop and s_end is not None and s_stop > 0.05 else None
        tw_end = rd["runs"][tw]["series"][-1]["relief"]["A"] / cA
        en = rd["runs"][fl]["end"]
        corr = en["relief"]["corr"]
        dsh = en["display"]["A"] / dref if dref else None
        p10 = en["pieces"]["pieces_ge10"]
        ratio = vx(r, t)
        content = bool(s_end is not None and s_end >= SHARE and kept is not None and kept >= KEEP)
        rest = bool(ratio is not None and ratio <= REST_X)
        shape = bool(r["sil"] >= t["sil"] - SHAPE_GATE[0] and r["thin"] <= t["thin"] + SHAPE_GATE[1])
        ok = content and rest and shape
        v.update(status="PASS" if ok else "FAIL", content_ok=content, rest_ok=rest, shape_ok=shape, share_stop=s_stop, share_end=s_end, kept=kept, twin_share=tw_end, corr=corr,
                 display_share=dsh, vx=ratio, sil=(r["sil"], t["sil"]), ch=(r["ch"], t["ch"]), thin=(r["thin"], t["thin"]),
                 pieces10=p10, ceiling_A=cA, stop_raw=r["stop_fe"] - 1, end_raw=r["deliver_n"] - 1, windows=r["windows"],
                 minutes=(r["sec"] or 0) / 60, full=str(sc["full"]), twin=str(sc["twin"]))
        V[k] = v
        rows.append(f"| {sc['name']} | {fmt(s_stop)} | {fmt(s_end)} | {fmt(kept)} | {fmt(tw_end)} | {fmt(corr)} | {fmt(dsh)} | "
                    f"{fmt(ratio, '.2f')}x | {r['sil']:.4f} / {t['sil']:.4f} | {r['ch']:.4f} / {t['ch']:.4f} | "
                    f"{r['thin']:.3f} / {t['thin']:.3f} | {p10} |")
    legend = (f"{ARMS} 300k, seed 97. Share = the relief amplitude carved into the outer-layer particles over the target's own "
              "(its relief mesh's 300k samples read the same way: D161's reader; the plaque D163's), at the stop (the last "
              "optimised window) and at the end (after the 30-window settle); kept = end / stop; display share = the same on "
              "the display surface. Criteria: content = share end >= 0.40 and kept >= 0.80; rest = end v_mean <= 2x twin; "
              "shape = the body arrived (sil_iou >= twin - 0.005, thin <= twin + 0.05). Windows / minutes of the Rx run: "
              + ", ".join(f"{CARVE[k]['name']} {V[k].get('windows', '-')} / {fmt(V[k].get('minutes'), '.0f')}" for k in keys) + "."
              + (f" Missing: {' '.join(miss)}" if miss else ""))
    yn = lambda b: "ok" if b else "no"  # noqa: E731
    verdict = "; ".join(f"{CARVE[k]['name']} {V[k]['status']}" + (
        f" (content {yn(V[k]['content_ok'])}: share {fmt(V[k].get('share_end'))}, kept {fmt(V[k].get('kept'))}; rest {yn(V[k]['rest_ok'])} "
        f"{fmt(V[k].get('vx'), '.2f')}x; shape {yn(V[k]['shape_ok'])})" if V[k]["status"] != "INCOMPLETE" else "") for k in keys)
    write(D, f"D170 {job}: 300k carve scenes, Rx vs twin", rows, legend, verdict, V)


# ----------------------------------------------------------------------------------------------------------- detail
def detail(job, D):
    keys = JOB_DETAIL[job]
    rows = ["| scene | arm | seed | share stop | share end | kept | twin share end | corr end | v end / twin | "
            "sil_iou arm / twin | thin arm / twin |", "|" + "---|" * 11]
    V, extra = {}, []
    yn = lambda b: "ok" if b else "no"  # noqa: E731
    for k in keys:
        sc = DETAIL[k]
        miss = wait_rc([str(p) for p in sc["runs"].values()], hours=float(os.environ.get("D170_WAIT_H", "12")))
        recs = {lab: run_rec(str(p)) for lab, p in sc["runs"].items()}
        out = D / f"read_{k}.json"
        have = {lab: p for lab, p in sc["runs"].items() if recs[lab] is not None and Path(f"{p}_frames12.npz").exists()}
        if have and not out.exists():
            spec = D / f"scene_{k}.json"
            json.dump(dict(scene=k, N=sc["N"], tgt=str(sc["tgt"]), smooth=str(sc["smooth"]), detail=str(sc["detail"]),
                           runs={lab: str(p) for lab, p in have.items()},
                           stops={lab: recs[lab]["stop_fe"] - 1 for lab in have}), open(spec, "w"))
            reader([PY, str(B / "tmp/d170/d170_detail.py"), str(out), str(spec)], D / f"read_{k}.log")
        rd = json.load(open(out)) if out.exists() else None
        labs = list(sc["runs"])
        seeds = sorted({int(lab.rsplit("_", 1)[-1]) for lab in labs})
        arms = [a for a in dict.fromkeys(lab.rsplit("_", 1)[0] for lab in labs) if a not in ("T", "R")]
        for s in seeds:
            if rd is None or f"T_{s}" not in rd["runs"]:
                continue
            # floor-corrected: 0 = the smooth canvas's own sample (the reading's curvature bias), 1 = the detailed mesh's own
            # N-samples (what a perfect placement of N particles carries)
            cA = float(np.mean([rd["refs"][f"detail_s1{s}"]["relief"]["A"], rd["refs"][f"detail_s2{s}"]["relief"]["A"]]))
            fA = float(rd["refs"][f"smooth_s2{s}"]["relief"]["A"])

            def series(lab):
                ser = rd["runs"][lab]["series"]
                return [q["raw"] for q in ser], [((q["relief"]["A"] or 0.0) - fA) / (cA - fA) for q in ser], ser

            t = recs[f"T_{s}"]
            tw_end = series(f"T_{s}")[1][-1]
            tg_end = series(f"R_{s}")[1][-1] if f"R_{s}" in rd["runs"] else None
            for a in arms:
                lab = f"{a}_{s}"
                v = dict(kind="detail", scene=sc["name"], arm=a, seed=s, status="INCOMPLETE")
                r = recs.get(lab)
                if r is None or lab not in rd["runs"]:
                    rows.append(f"| {sc['name']} | {ARM_NAMES.get(a, a)} | {s} | - | - | - | {fmt(tw_end)} | - | - | - | - |")
                    V[f"{k}_{lab}"] = v
                    continue
                raws, vals, ser = series(lab)
                s_stop, _ = sample_at(raws, vals, r["stop_fe"] - 1)
                s_end = vals[-1] if raws[-1] == r["deliver_n"] - 1 else None
                kept = s_end / s_stop if s_stop and s_end is not None and s_stop > 0.05 else None
                corr = ser[-1]["relief"]["corr"]
                ratio = vx(r, t)
                above = s_end - tw_end if s_end is not None else None
                content = bool(s_end is not None and s_end >= SHARE and above is not None and above >= 0.10 and kept is not None and kept >= KEEP)
                rest = bool(ratio is not None and ratio <= REST_X)
                shape = bool(r["sil"] >= t["sil"] - SHAPE_GATE[0] and r["thin"] <= t["thin"] + SHAPE_GATE[1])
                v.update(status="PASS" if content and rest and shape else "FAIL", content_ok=content, rest_ok=rest, shape_ok=shape,
                         share_stop=s_stop, share_end=s_end, kept=kept, twin_share=tw_end, tag_share=tg_end, above_twin=above,
                         corr=corr, vx=ratio, sil=(r["sil"], t["sil"]), thin=(r["thin"], t["thin"]), ceiling_A=cA,
                         windows=r["windows"], full=str(sc["runs"][lab]), twin=str(sc["runs"][f"T_{s}"]))
                V[f"{k}_{lab}"] = v
                rows.append(f"| {sc['name']} | {ARM_NAMES.get(a, a)} | {s} | {fmt(s_stop)} | {fmt(s_end)} | {fmt(kept)} | {fmt(tw_end)}"
                            + (f" (tag {fmt(tg_end)})" if tg_end is not None else "") + f" | {fmt(corr)} | {fmt(ratio, '.2f')}x | "
                            f"{r['sil']:.4f} / {t['sil']:.4f} | {r['thin']:.3f} / {t['thin']:.3f} |")
        if rd is not None:
            for s, c in rd["ctx"].items():
                extra.append(f"Detail size ({sc['name']}, seed {s}, pitch {c['pitch']:.4f} wu): rms height {c['rms_H_pitch']:.2f} "
                             f"pitch ({c['rms_Hhp_pitch']:.2f} after the {c['s_bg_pitch']:g}-pitch high-pass), autocorrelation 1/2 at "
                             f"{fmt(c['autocorr_half_pitch'])} pitch, variance above one pitch {c['var_share_above_1pitch']:.2f}, "
                             f"read on {c['region_share']:.2f} of the canvas.")
            for s in seeds:
                ref = lambda kk: rd["refs"].get(f"{kk}{s}", {}).get("relief", {}).get("A")  # noqa: E731
                c1, c2 = ref("detail_s1"), ref("detail_s2")
                if None not in (c1, c2, ref("smooth_s2"), ref("tgt")):
                    cA, fA = np.mean([c1, c2]), ref("smooth_s2")
                    nz = lambda x: (x - fA) / (cA - fA)  # noqa: E731
                    extra.append(f"References (seed {s}, raw A): the detailed mesh's two N-samples {c1:.3f} / {c2:.3f} (corr "
                                 f"{rd['refs'][f'detail_s1{s}']['relief']['corr']:.2f}), the smooth canvas's own sample {fA:.3f} (the floor), the run's 3D "
                                 f"target sample {ref('tgt'):.3f} (share {nz(ref('tgt')):.2f}).")
    legend = (f"Arms: {FULL} with the named weight / pictures / relaxation, settle 30 with --no-settle_relax (Rx = held weight, "
              f"relaxation on until the stop); twin = {TWIN} (lambda = 0). Share = (A - floor) / (ceiling - floor): A = <h~, H~> / <H~, H~>, h the outer "
              "layer's offset from the smoothed canvas, H the detailed mesh's offset there (both high-passed at 4 pitches); the "
              "ceiling the detailed mesh's own samples at N, the floor the smooth canvas's own sample (d170_detail.py). Criteria: content = share end >= 0.40, >= twin + 0.10, "
              "kept >= 0.80; rest = end v_mean <= 2x twin; shape = sil_iou >= twin - 0.005, thin <= twin + 0.05.")
    verdict = "; ".join(f"{v.get('arm')} {v.get('seed')} {v['status']}" + (
        f" (share {fmt(v.get('share_end'))} vs twin {fmt(v.get('twin_share'))}, kept {fmt(v.get('kept'))}; rest {yn(v['rest_ok'])}; "
        f"shape {yn(v['shape_ok'])})" if v["status"] != "INCOMPLETE" else "") for kk, v in V.items())
    write(D, f"D170 {job}: detail relief, arms vs twin", rows[:12], legend, verdict, V, ["", *extra])
    return V


# ------------------------------------------------------------------------------------------------------------- band
def band300(D):
    full, twin = RUNS / "band300k_Rx_97", G / "d162f/runs/band300k_T_97"
    miss = wait_rc([str(full), str(twin)])
    r, t = run_rec(str(full)), run_rec(str(twin))
    out = D / "read_colour.jsonl"
    if r is not None and t is not None and not (out.exists() and out.stat().st_size):
        jf = D / "jobs_colour.txt"
        jf.write_text(f"band300k_T_97 band {twin}_frames12.npz mesh=bunny\n"
                      f"band300k_Rx_97 band {full}_frames12.npz mesh=bunny twin=band300k_T_97 timeline\n")
        reader([PY, str(B / "tmp/d160/d160_read.py"), str(out), str(jf)], D / "read_colour.log")
    V = dict(band=dict(kind="colour", status="INCOMPLETE"))
    rows = ["| scene | colour r stop | colour r end | kept | mis end Rx / twin | v end / twin | sil_iou Rx / twin | chamfer Rx / twin | "
            "thin Rx / twin |", "|" + "---|" * 9]
    if r is not None and t is not None and out.exists():
        col = {json.loads(x)["name"]: json.loads(x) for x in open(out) if x.strip()}
        c, ct = col["band300k_Rx_97"], col["band300k_T_97"]
        mt, fl, mis = ct["end"]["mis"], c["floor"], c["end"]["mis"]
        rr = lambda m: (mt - m) / (mt - fl)  # noqa: E731
        tl = np.asarray(c.get("timeline", []), np.float64)
        stop_mis, _ = sample_at(tl[:, 0] * 240, tl[:, 1], r["stop_fe"] - 1) if tl.ndim == 2 else (None, False)
        c_stop = rr(stop_mis) if stop_mis is not None else None
        c_end = rr(mis)
        kept = c_end / c_stop if c_stop and c_stop > 0.05 else None
        ratio = vx(r, t)
        ok = bool(c_end >= COLOUR and kept is not None and kept >= KEEP and ratio is not None and ratio <= REST_X)
        V["band"] = dict(kind="colour", status="PASS" if ok else "FAIL", colour_stop=c_stop, colour_end=c_end, kept=kept,
                         mis=(mis, mt), vx=ratio, sil=(r["sil"], t["sil"]), full=str(full), twin=str(twin))
        rows.append(f"| colour band (D160) | {fmt(c_stop)} | {fmt(c_end)} | {fmt(kept)} | {mis:.3f} / {mt:.3f} | {fmt(ratio, '.2f')}x | "
                    f"{r['sil']:.4f} / {t['sil']:.4f} | {r['ch']:.4f} / {t['ch']:.4f} | {r['thin']:.3f} / {t['thin']:.3f} |")
    legend = (f"{ARMS} 300k, seed 97. Colour r = (twin mis - mis) / (twin mis - floor), D160's reader (mis = the mis-coloured "
              "surface share); stop / end / kept as for the relief. Criteria: r end >= 0.80, kept >= 0.80, end v_mean <= 2x twin."
              + (f" Missing: {' '.join(miss)}" if miss else ""))
    v = V["band"]
    write(D, "D170 band300: the 300k colour band, Rx vs twin", rows, legend,
          f"band {v['status']}" + (f" (r {fmt(v.get('colour_end'))}, kept {fmt(v.get('kept'))}, rest {fmt(v.get('vx'), '.2f')}x)"
                                   if v["status"] != "INCOMPLETE" else ""), V)


# ---------------------------------------------------------------------------------------------------------- gallery
def gal(D, arm="Rx", seeds=(97, 99), title=None):
    GL = R6.gl()
    need = [str(RUNS / f"gal_{m}_{arm}_{s}") for m in MESHES for s in seeds]
    miss = wait_rc(need)

    def rd(prefix):
        r = run_rec(prefix)
        fr = glob.glob(f"{prefix}_frames12.npz") or glob.glob(f"{prefix}_end1.npz")
        if r is not None and fr:
            z = np.load(fr[0], allow_pickle=True)
            r["p10"] = GL.pieces(np.asarray(z["frames"][-1], np.float64))[1]
        return r

    def tagw(m, a, s):
        jp, _ = GL.paths(m, a, s)
        return rd(jp[:-5]) if jp else None

    rows = [f"| mesh | {arm} 97 | R 97 (d167) ; V 97 (tag) | {arm} 99 | V 99 (tag) | twin W 97 ; W 99 | d sil / chamfer / thin vs R (97; 99) |",
            "|" + "---|" * 7]
    f4 = lambda r: "-" if r is None else f"{r['sil']:.4f} / {r['ch']:.4f} / {r['thin']:.3f} / {r['v_end'] * 1e3:.1f} / {r.get('p10', '-')}"  # noqa: E731
    v = dict(kind="gallery", arm=arm, held=0, n=0, miss=[], rest=True, held_two=0, deltas={})
    for m in MESHES:
        xs = {s: rd(str(RUNS / f"gal_{m}_{arm}_{s}")) for s in seeds}
        r97, v97, v99 = rd(str(G / f"d167/runs/gal_{m}_R_97")), tagw(m, "V", 97), tagw(m, "V", 99)
        w97, w99 = tagw(m, "W", 97), tagw(m, "W", 99)
        dl, ok_all, ok_two = [], True, True
        for s, ref, alt, w in ((97, r97, v97, w97), (99, v99, None, w99)):
            if s not in seeds:
                dl.append("")
                continue
            x = xs[s]
            if x is None or ref is None:
                ok_all = ok_two = False
                dl.append("-")
                continue
            d = (x["sil"] - ref["sil"], x["ch"] - ref["ch"], x["thin"] - ref["thin"])
            ok_all &= d[0] >= -SH_TOL[0] and d[1] <= SH_TOL[1] and d[2] <= SH_TOL[2]
            refs2 = [q for q in (ref, alt) if q is not None]       # two runs of the rule at 97: d167's R and the tag's V
            ok_two &= (x["sil"] >= min(q["sil"] for q in refs2) - SH_TOL[0] and x["ch"] <= max(q["ch"] for q in refs2) + SH_TOL[1]
                       and x["thin"] <= max(q["thin"] for q in refs2) + SH_TOL[2])
            dl.append(f"{d[0]:+.4f} / {d[1]:+.4f} / {d[2]:+.3f}")
            if w is None or x["v_end"] > REST_X * w["v_end"]:
                v["rest"] = False
        v["n"] += 1
        v["held"] += int(ok_all)
        v["held_two"] += int(ok_two)
        v["deltas"][m] = dl
        if not ok_all:
            v["miss"].append(m)
        rows.append(f"| {m} | {f4(xs.get(97))} | {f4(r97)} ; {f4(v97)} | {f4(xs.get(99))} | {f4(v99)} | {f4(w97)} ; {f4(w99)} | "
                    f"{dl[0]} ; {dl[1]} |")
    v["pass_"] = bool(v["n"] == len(MESHES) and v["held"] == v["n"] and v["rest"])
    v["status"] = "PASS" if v["pass_"] else "FAIL"
    legend = (f"Arm {arm} = {FULL}, {ARM_NAMES.get(arm, RXN) if arm != 'Rx' else RXN}; twin = {TWIN}. 40k. Cells: sil_iou / chamfer / "
              "thin / end v_mean x1e3 / pieces >= 10. R 97 = the tag's rule on repo_d166 (d167 job 114), V = the tag's runs "
              "(d155 gate / d154), W = the twin. Holds = sil_iou >= R - 0.003, chamfer <= R + 0.002, thin <= R + 0.02 at each seed "
              "(R 99 = V 99); rest = end v_mean <= 2x twin (same seed). Two-baseline reading: within tolerance of the worse of "
              f"R 97 / V 97 at seed 97: {v['held_two']}/{v['n']}." + (f" Missing: {' '.join(miss)}" if miss else ""))
    write(D, title or f"D170 gallery: {arm} on the 6-mesh 40k gallery, seeds {' / '.join(map(str, seeds))}", rows, legend,
          f"{arm} {v['status']} (holds {v['held']}/{v['n']}" + (f", misses {','.join(v['miss'])}" if v["miss"] else "")
          + f"; two-baseline {v['held_two']}/{v['n']}; rest {'ok' if v['rest'] else 'no'})", {arm: v})


def r0h0(D):
    """The coordinator's diagnostic: D166's scene reader on R0 / H0 (twins D161 / D160's), then the R0 gallery (seed 97)."""
    S = D / "scenes"
    S.mkdir(parents=True, exist_ok=True)
    wait_rc([str(RUNS / f"{sc}_{a}_{s}") for sc in ("coin", "bunny", "band") for a in ("R0", "H0") for s in (97, 99)])
    R6.NAMES.update(R0="rule (tag default) + relaxation off throughout (--no-layer_relax), settle 30",
                    H0="held (--lambda_ema 0) + relaxation off throughout (--no-layer_relax), settle 30")
    cwd = os.getcwd()
    R6.scenes(str(S), str(RUNS), "D170: the relaxation off throughout, 40k scenes (R0 = rule, H0 = held)", "R0:T", "H0:T")
    os.chdir(cwd)
    G2 = D / "gallery"
    G2.mkdir(parents=True, exist_ok=True)
    gal(G2, "R0", (97,), "D170: R0 (rule + relaxation off throughout) on the 6-mesh 40k gallery, seed 97")
    sv = json.load(open(S / "verdict.json"))["arms"]
    gv = json.load(open(G2 / "verdict.json"))["arms"]["R0"]
    lines = [open(S / "summary.md").read().strip(), "", open(G2 / "summary.md").read().strip()]
    r0 = sv.get("R0", {})
    q1 = (f"R0 content >= 0.4 / 0.8? {'YES' if r0.get('authority') else 'NO'} (relief min {fmt(r0.get('share_min'))}, colour min "
          f"{fmt(r0.get('colour_min'))}; kept min {fmt(r0.get('kept_min'))}; rest max {fmt(r0.get('vx_max'), '.1f')}x)")
    q2 = (f"R0 gallery within tolerance of R? {'YES' if gv['held'] == gv['n'] else 'NO'} (holds {gv['held']}/{gv['n']}"
          + (f", misses {','.join(gv['miss'])}" if gv["miss"] else "") + f"; rest {'ok' if gv['rest'] else 'no'})")
    h0 = sv.get("H0", {})
    q3 = (f"H0: content min {fmt(h0.get('share_min'))} / {fmt(h0.get('colour_min'))}, kept min {fmt(h0.get('kept_min'))}, "
          f"rest max {fmt(h0.get('vx_max'), '.1f')}x")
    V = dict(R0=dict(r0, kind="r0h0_scenes"), H0=dict(h0, kind="r0h0_scenes"), R0_gal=gv)
    open(D / "summary.md", "w").write("\n".join(["# D170 job 178: the relaxation off throughout (R0 / H0), 40k", "", *lines, "",
                                                 f"VERDICT: {q1}; {q2}; {q3}"]) + "\n")
    json.dump(dict(arms=V, verdict=f"{q1}; {q2}; {q3}"), open(D / "verdict.json", "w"), indent=1, default=float)
def big(D):
    need = [str(RUNS / f"{m}300k_Rx_97") for m in ("bunny", "dragon")]
    miss = wait_rc(need)
    rows = ["| mesh | Rx sil / chamfer / thin | R (rule) | twin | d vs R | v end / twin | windows | minutes |", "|" + "---|" * 8]
    V = dict(Rx=dict(kind="big", held=0, n=0, rest=True))
    for m in ("bunny", "dragon"):
        x, ref, t = run_rec(str(RUNS / f"{m}300k_Rx_97")), run_rec(str(G / f"d167/runs/{m}300k_R_97")), run_rec(str(G / f"d167/runs/{m}300k_T_97"))
        g = lambda r: "-" if r is None else f"{r['sil']:.4f} / {r['ch']:.4f} / {r['thin']:.3f}"  # noqa: E731
        if x is None or ref is None:
            rows.append(f"| {m} | {g(x)} | {g(ref)} | {g(t)} | - | - | - | - |")
            V["Rx"]["rest"] = False
            continue
        d = (x["sil"] - ref["sil"], x["ch"] - ref["ch"], x["thin"] - ref["thin"])
        V["Rx"]["n"] += 1
        V["Rx"]["held"] += int(d[0] >= -SH_TOL[0] and d[1] <= SH_TOL[1] and d[2] <= SH_TOL[2])
        ratio = vx(x, t)
        if ratio is None or ratio > REST_X:
            V["Rx"]["rest"] = False
        rows.append(f"| {m} | {g(x)} | {g(ref)} | {g(t)} | {d[0]:+.4f} / {d[1]:+.4f} / {d[2]:+.3f} | {fmt(ratio, '.2f')}x | "
                    f"{x['windows']} | {(x['sec'] or 0) / 60:.0f} |")
    v = V["Rx"]
    v["pass_"] = bool(v["n"] == 2 and v["held"] == 2 and v["rest"])
    v["status"] = "PASS" if v["pass_"] else "FAIL"
    write(D, "D170 300k shape: Rx on bunny + dragon, seed 97", rows,
          f"{ARMS} Refs: d167 jobs 120 / 124 (repo_d166: the twin T, the rule R). Holds = sil_iou >= R - 0.003, chamfer <= R + 0.002, "
          "thin <= R + 0.02; rest = end v_mean <= 2x twin." + (f" Missing: {' '.join(miss)}" if miss else ""),
          f"Rx {v['status']} (holds {v['held']}/{v['n']}, rest {'ok' if v['rest'] else 'no'})", V)


# -------------------------------------------------------------------------------------------------------- aggregate
def aggregate(out_md):
    S = sorted(glob.glob(str(O / "auto/*/verdict.json")))
    vs = {Path(p).parent.name: json.load(open(p)) for p in S}
    get = lambda j: vs.get(j, {}).get("arms", {})  # noqa: E731
    yn = lambda b: "ok" if b else "no"  # noqa: E731
    L = ["# D170: Rx validation (held weight + settle 30 with the outer-layer relaxation off in the settle)", ""]
    # the top item (the user, 2026-10-28): the render carves the dragon's scales
    dr = {**get("173_dragon40"), **get("177_dragon300")}
    L += ["## Top item: does the render carve the dragon's scales? (3D target the smoothed dragon, pictures the detailed one)", ""]
    if dr:
        for k, v in dr.items():
            L.append(f"- {v.get('scene')}, {ARM_NAMES.get(v.get('arm'), v.get('arm'))}, seed {v.get('seed')}: {v['status']}" + (
                f" (share stop {fmt(v.get('share_stop'))} -> end {fmt(v.get('share_end'))} vs twin {fmt(v.get('twin_share'))}; rest "
                f"{fmt(v.get('vx'), '.2f')}x; shape {yn(v.get('shape_ok'))})" if v["status"] != "INCOMPLETE" else " (not read)"))
        for j in ("173_dragon40", "177_dragon300"):
            p = O / "auto" / j / "summary.md"
            if p.exists():
                L += [x for x in open(p).read().splitlines() if x.startswith("Detail size") or x.startswith("References")]
    else:
        L.append("- not read yet (jobs 168 / 173).")
    sz = []
    for n in (40000, 300000, 1000000, 1500000, 3000000):
        p = O / f"size_probe/size_dragon_{n}.json"
        if p.exists():
            d = json.load(open(p))
            c, r = d["ctx"]["97"], d["refs"]
            sz.append(f"{n // 1000}k: pitch {c['pitch']:.4f} wu, rms height {c['rms_H_pitch']:.2f} pitch, half-correlation "
                      f"{fmt(c['autocorr_half_pitch'])} pitch; a perfect N-sample carries A {r['detail_s197']['relief']['A']:.2f} at "
                      f"corr {r['detail_s197']['relief']['corr']:.2f} (floor {r['smooth_s297']['relief']['A']:.2f})")
    if sz:
        L += ["", "Scale size against the pitch (no run; output/gpu/d170/size_probe, the detailed dragon's own random N-samples read by "
              "d170_detail.py; the bunny's wrinkles at 300k: corr 0.27):", *[f"- {x}" for x in sz],
              "Reading: the scales are ~1 pitch wide and ~0.09 pitch high at 300k; the render's body surface is a radius-3-pitch "
              "field and its pixel 0.82 pitch, so neither the pictures of the body nor a random 300k layer resolve them; 40k "
              "shares above are inside the reading's own noise (the two detail samples differ by 0.10-0.43 in raw A)."]
    L += ["", "## Adopt Rx?", ""]
    carve_v = {**get("170_carve_a"), **get("171_carve_b")}
    det = get("172_wrinkle")
    gal_v, big_v, band_v = get("174_gal").get("Rx"), get("175_big").get("Rx"), get("176_band300").get("band")
    lines = []
    for k, v in carve_v.items():
        lines.append(f"- carve 300k, {v.get('scene', k)}: {v['status']}" + (
            f" (content {yn(v.get('content_ok'))}: share stop {fmt(v.get('share_stop'))} -> end {fmt(v.get('share_end'))}, kept "
            f"{fmt(v.get('kept'))}, twin {fmt(v.get('twin_share'))}; rest {yn(v.get('rest_ok'))} {fmt(v.get('vx'), '.2f')}x twin; shape "
            f"{yn(v.get('shape_ok'))}: sil_iou {v['sil'][0]:.4f} vs twin {v['sil'][1]:.4f}, thin {v['thin'][0]:.3f} vs {v['thin'][1]:.3f}; "
            f"{v.get('windows')} windows)" if v["status"] != "INCOMPLETE" else " (not read yet)"))
    for k, v in det.items():
        lines.append(f"- mesh-relief pictures, {v.get('scene', k)}: {v['status']}" + (
            f" (share end {fmt(v.get('share_end'))} vs twin {fmt(v.get('twin_share'))}, kept {fmt(v.get('kept'))}, "
            f"{fmt(v.get('vx'), '.2f')}x twin, shape {yn(v.get('shape_ok'))})" if v["status"] != "INCOMPLETE" else " (not read yet)"))
    if band_v:
        lines.append(f"- colour band 300k: {band_v['status']}" + (f" (r {fmt(band_v.get('colour_end'))}, kept {fmt(band_v.get('kept'))}, "
                     f"{fmt(band_v.get('vx'), '.2f')}x twin)" if band_v["status"] != "INCOMPLETE" else ""))
    if gal_v:
        lines.append(f"- gallery 40k (6 meshes x seeds 97/99): {gal_v['status']} (holds {gal_v['held']}/{gal_v['n']}"
                     + (f", misses {','.join(gal_v['miss'])}" if gal_v["miss"] else "") + f"; two-baseline {gal_v.get('held_two')}/{gal_v['n']}; "
                     f"rest {'ok' if gal_v['rest'] else 'no'})")
    if big_v:
        lines.append(f"- 300k shape (bunny, dragon): {big_v['status']} (holds {big_v['held']}/{big_v['n']}, rest {'ok' if big_v['rest'] else 'no'})")
    must = [v["status"] for v in carve_v.values()] + ([gal_v["status"]] if gal_v else ["INCOMPLETE"]) + ([big_v["status"]] if big_v else ["INCOMPLETE"])
    bad = [v.get("scene", k) for k, v in carve_v.items() if v["status"] == "FAIL"] \
        + (["gallery 40k"] if gal_v and gal_v["status"] == "FAIL" else []) + (["300k shape"] if big_v and big_v["status"] == "FAIL" else [])
    pending = 4 - sum(v["status"] != "INCOMPLETE" for v in carve_v.values()) + (0 if gal_v else 1) + (0 if big_v else 1)
    if bad:
        ans = (f"NO by the letter of the criteria; failing: {', '.join(bad)}" + (f" ({pending} item(s) still being read)" if pending else "")
               + ". Rx rests (end v_mean ~0) and keeps what the render wrote, but the held weight stops the runs before the body arrives.")
    elif pending:
        ans = f"NOT YET DECIDED: {pending} item(s) not read yet (see below)."
    else:
        ans = ("YES by the criteria: every 300k carve scene keeps >= 0.40 of the relief through the settle at rest with the body "
               "arrived, and the gallery and the 300k shapes hold.")
    L += [f"**{ans}**", "", *lines, "",
          "Criteria (set before reading; the shape gate added after the plaque's run log showed its early stop): carve content = "
          "share end >= 0.40 and kept >= 0.80, rest = end v_mean <= 2x twin, shape = sil_iou >= twin - 0.005 and thin <= twin + 0.05 "
          "(300k, seed 97); gallery sil_iou / chamfer / thin within 0.003 / 0.002 / 0.02 of the tag's rule on every mesh at both "
          "seeds, rest <= 2x twin; 300k shape the same against d167's R. Colour and the mesh-relief (wrinkle / scale) scenes are "
          f"reported, not gating. Arms: Rx = {FULL}, {RXN}; twin = {TWIN}.", ""]
    ns = [vs.get(j, {}).get("verdict") for j in ("179a_nostop_gal", "179b_nostop_plq")]
    L += ["## Diagnostic: is the shape loss the early stop? (Rx with --reject_stop 0, job 179)", "",
          f"- gallery 40k seed 97: {ns[0] or 'not read yet'}", f"- plaque 300k: {ns[1] or 'not read yet'}", ""]
    rv = vs.get("178_r0h0", {}).get("verdict")
    gate = O / "auto/168_gate/summary.md"
    L += ["## The relaxation off throughout (R0 / H0, the coordinator's diagnostic)", "",
          (f"- {rv}" if rv else "- not read yet (job 178)."),
          "- Switch: none existed in repo_d166 (window/setup.py:59 sets the relaxation's fraction 0 only for --baseline xu*; "
          "run/runner.py:313-316 only in the settle under --no-settle_relax; --no-layer_relief, pipeline_run.py:140-142, changes "
          "its reference to the neighbours' plane, not off). Added in repo_d170 (wip/d170 2664ca56): --layer_relax / "
          "--no-layer_relax (config.py:141, window/setup.py:59-60, pipeline_run.py:68-70), default on; V3 gate (job 168): "
          + (open(gate).read().split("VERDICT: ")[-1].strip() if gate.exists() else "pending") + ".", "",
          "## Combined carve + colour scene", "",
          "Not run. repo_d166 cannot draw a painted relief picture without a code change: physmorph/pipeline/target.py:156 makes "
          "the render's first picture sample render_x (--render_tgt), but target.py:215 takes its albedos from "
          "appearance.target_albedos(), whose first entry is painted on target_x (pipeline_run.py:330, make_appearance(args.appearance, "
          "src, tgt, ...)); found_disc_albedo (render/exterior.py:400-413) then indexes render_x's particles into target_x's paint "
          "(wrong colours, or out of range when n_r > N). Needed: sample 0's albedo = appearance.albedo_at(render_x) when render_x "
          "is given (as run/runner.py:120 / :136 already do for the floor).", "",
          f"Written {time.strftime('%Y-%m-%d %H:%M %Z')} from {len(S)} job summaries. Deliverables: output/gpu/d170/deliver/.", ""]
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
        if mode in JOB_SCENES:
            carve(mode, D)
        elif mode in JOB_DETAIL:
            detail(mode, D)
        else:
            {"band300": band300, "gal": gal, "big": big, "r0h0": r0h0,
             "nostop_gal": lambda D: gal(D, "RxNS", (97,), "D170 diagnostic: Rx without the reject stop (--reject_stop 0) on the 40k gallery, seed 97")}[mode](D)
