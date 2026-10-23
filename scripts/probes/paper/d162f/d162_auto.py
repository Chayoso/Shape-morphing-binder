"""d162_auto.py KIND OUT_MD ARGS... -- short summaries for output/gpu/d162/auto/<job>/summary.md (verdict lines, key numbers,
video paths), from the readers' own outputs. KIND: relief (D161 summary md files), ripple (D161 reader json, 300k), band
(D160 reader jsonl, 300k), settle (D162h: the stop against the settled end; prints VERDICT: A | B | mixed)."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")


def summ_json(md):
    t = open(md).read()
    return json.loads(t[t.index("```") + 3:t.rindex("```")].strip())


def end_v(prefix):
    arm = list(json.load(open(f"{prefix}.json"))["arms"].values())[0]
    com = [h for h in arm["history"] if h.get("frame_end") and not h.get("null_commit")]
    dn = arm["deliver_n"]
    return next((h for h in com if h["frame_end"] == dn), com[-1])["v_mean"], com, dn


def relief(out, title, *mds):
    L = [f"# {title}", ""]
    for md in mds:
        j = summ_json(md)
        L.append(f"## {Path(md).stem}")
        for p in j["P1"]:
            L.append(f"- P1 seed {p['seed']}: share of the ceiling {p['share_full']:.2f} (corr {p['corr_full']:.2f}); twin "
                     f"{p['share_twin']:.2f}")
        L.append("- P3 misses: " + (", ".join(f"{p['metric']} {p['seed']}" for p in j["P3"] if not p["pass_"]) or "none"))
        L.append("- P4 (end v_mean full / twin): " + "; ".join(f"{p['seed']}: {p['full']:.2e} / {p['twin']:.2e} "
                                                              f"({p['full'] / p['twin']:.1f}x, {'pass' if p['pass_'] else 'FAIL'})"
                                                              for p in j["P4"]))
        L.append("")
    open(out, "w").write("\n".join(L) + "\n")


def ripple(out, read_json, run_dir, *videos):
    d = json.load(open(read_json))
    ceil = np.mean([v["relief"]["A"] for k, v in d["refs"].items() if k.startswith("relief_s")])
    L = ["# D162f (b): 300k surface ripples from pictures (held vs twin dF_c + layer u)", ""]
    for lab in ("full97", "twin97"):
        e = d["runs"][lab]["series"][-1]["relief"]
        L.append(f"- {lab}: ridge share of the target's own {e['A'] / ceil:.2f} (corr {e['corr']:.2f})")
    vh, _, _ = end_v(f"{run_dir}/rip300k_H_97")
    vt, _, _ = end_v(f"{run_dir}/rip300k_T_97")
    L += [f"- rest cost: end v_mean held {vh:.2e} vs twin {vt:.2e} ({vh / vt:.1f}x)", "- videos: " + ", ".join(videos), ""]
    open(out, "w").write("\n".join(L) + "\n")


def band(out, jsonl, run_dir, video):
    rows = {json.loads(line)["name"]: json.loads(line) for line in open(jsonl)}
    a, t = rows["band300k_D_97"], rows["band300k_T_97"]
    r = (t["end"]["mis"] - a["end"]["mis"]) / max(t["end"]["mis"] - t["floor"], 1e-9)
    vh, _, _ = end_v(f"{run_dir}/band300k_D_97")
    vt, _, _ = end_v(f"{run_dir}/band300k_T_97")
    L = ["# D162f (c): 300k colour band (D160d vs twin dF_c + layer u)", "",
         f"- mis-coloured share: D160d {a['end']['mis']:.3f}, twin {t['end']['mis']:.3f}, floor {t['floor']:.3f}: r = {r:.2f}",
         f"- held-out colour PSNR: {a['img']['held']['colour']['psnr']:.2f} vs {t['img']['held']['colour']['psnr']:.2f}",
         f"- rest cost: end v_mean {vh:.2e} vs twin {vt:.2e} ({vh / vt:.1f}x)", f"- video: {video}", ""]
    open(out, "w").write("\n".join(L) + "\n")


def settle(out):
    """Per run: the stop (the last optimised window) against the settled end (30 zero-control windows): v_mean against the
    twin's end, and the relief share (D161's reader) or the mis-coloured share (D160's reader) at both. (A): every run
    settles to <= 2 x its twin's end speed with >= 0.8 of its relief (relief) or of its recovered colour (band) kept; (B):
    a relief run keeps < 0.8; otherwise mixed."""
    H, D = G / "d162h" / "runs", G / "d162h" / "after"
    L = ["# D162h: does the held arm rest once its controls are zero, and do the letters / colour stay?", ""]
    keep_all, rest_all, spring = [], [], False
    for sc in ("coin", "bunny"):
        rd = json.load(open(D / f"read_{sc}_Hs.json"))
        ceil = np.mean([v["relief"]["A"] for k, v in rd["refs"].items() if k.startswith("relief_s")])
        for s in (97, 99):
            p = f"{H}/{sc}40k_Hs_{s}"
            v_end, com, _ = end_v(p)
            stop = [h for h in com if not h.get("settle")][-1]
            vt, _, _ = end_v(f"{G}/d161/runs40/{sc}40k_twin_{s}")
            ser = rd["runs"][f"full{s}"]["series"]
            raws = np.array([q["raw"] for q in ser])
            A = np.array([q["relief"]["A"] for q in ser]) / ceil
            a_stop, a_end = float(np.interp(stop["frame_end"], raws, A)), float(A[-1])
            keep = a_end / a_stop if a_stop > 0.05 else float("nan")
            keep_all.append(keep)
            rest_all.append(v_end <= 2 * vt)
            spring |= bool(keep < 0.8)
            L.append(f"- {sc} {s}: v at the stop {stop['v_mean']:.2e}, settled {v_end:.2e}, twin {vt:.2e} ({v_end / vt:.1f}x); "
                     f"relief share at the stop {a_stop:.2f}, settled {a_end:.2f} (kept {keep:.2f})")
    rows = {json.loads(line)["name"]: json.loads(line) for line in open(D / "read_colour.jsonl")}
    for s in (97, 99):
        a, tw = rows.get(f"bunny40k_band_Hs_{s}"), rows.get(f"bunny40k_band_P{s}")
        if a is None or tw is None:
            continue
        p = f"{H}/bunny40k_band_Hs_{s}"
        v_end, com, _ = end_v(p)
        stop = [h for h in com if not h.get("settle")][-1]
        vt, _, _ = end_v(f"{G}/d160/runs40/bunny40k_band_P{s}")
        tl = np.asarray(a["timeline"], np.float64)
        mt, fl = tw["end"]["mis"], a["floor"]
        r = lambda m: (mt - m) / max(mt - fl, 1e-9)  # noqa: E731
        r_stop, r_end = r(float(np.interp(stop["frame_end"] / 240.0, tl[:, 0], tl[:, 1]))), r(a["end"]["mis"])
        keep = r_end / r_stop if r_stop > 0.05 else float("nan")
        keep_all.append(keep)
        rest_all.append(v_end <= 2 * vt)
        L.append(f"- band {s}: v at the stop {stop['v_mean']:.2e}, settled {v_end:.2e}, twin {vt:.2e} ({v_end / vt:.1f}x); "
                 f"colour r at the stop {r_stop:.2f}, settled {r_end:.2f} (kept {keep:.2f})")
    ok_keep = all(k >= 0.8 for k in keep_all if k == k)
    verdict = "A" if (all(rest_all) and ok_keep) else ("B" if spring else "mixed")
    L += ["", f"VERDICT: {verdict}  (A: rests with >= 0.8 kept everywhere -> premature stop, D162i; B: the relief springs "
          "back -> tension, D162g / baking; mixed: neither)", ""]
    open(out, "w").write("\n".join(L) + "\n")
    print(f"VERDICT: {verdict}")




def stiff(out):
    """Job 29: per scene and E (x3, x7; D161c is x1): relief share and corr at the stop and the settled end (D161's
    reader), the end v_mean against the twin's, Chamfer (the run's metric)."""
    S, D = G / "d162s" / "runs", G / "d162s" / "after"
    L = ["# Job 29: stiffer material under the held weight (E x3, x7; D161c is E x1: coin 0.47-0.53, bunny 0.54-0.58)", ""]
    for sc in ("coin", "bunny"):
        rd = json.load(open(D / f"read_{sc}.json"))
        ceil = np.mean([v["relief"]["A"] for k, v in rd["refs"].items() if k.startswith("relief_s")])
        vt, _, _ = end_v(f"{G}/d161/runs40/{sc}40k_twin_97")
        for k in (3, 7):
            lab, p = f"full{k}", f"{S}/{sc}40k_E{k}_97"
            if lab not in rd["runs"]:
                continue
            v_end, com, _ = end_v(p)
            stop = [h for h in com if not h.get("settle")][-1]
            ser = rd["runs"][lab]["series"]
            raws = np.array([q["raw"] for q in ser])
            A = np.array([q["relief"]["A"] for q in ser]) / ceil
            a_stop = float(np.interp(stop["frame_end"], raws, A))
            ch = list(json.load(open(f"{p}.json"))["arms"].values())[0]["metrics"]["chamfer"]
            L.append(f"- {sc} E x{k}: relief share at the stop {a_stop:.2f}, settled end {A[-1]:.2f} (corr {ser[-1]['relief']['corr']:.2f}); "
                     f"end v_mean {v_end:.2e} vs twin {vt:.2e} ({v_end / vt:.1f}x); chamfer {ch:.4f}")
    open(out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    {"relief": relief, "ripple": ripple, "band": band, "settle": settle, "stiff": stiff}[sys.argv[1]](*sys.argv[2:])
