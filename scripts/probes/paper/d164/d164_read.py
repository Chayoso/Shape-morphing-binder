"""d164_read.py OUT_MD [--n300] -- D164's readings (CPU, read-only; docs/experiments.md, D164): the D164 runs against
the tag's, per arm. 40k: output/gpu/d164/runs40/{mesh}40k_{L,P}{seed} against the tag's V / W (d155/gate for armadilo,
bunny, dragon, homer, teapot; d154/g40runs for the rest), seeds 97 / 99 (the tag's 101 as its own third run). 300k
(--n300): output/gpu/d164/runs300/{bunny,dragon}300k_{L,P}97 against B1's L / P 97 / 99 / 101 (d145t_b1).

(R1) null windows by reason, before / after, and the counterfactual inside the D164 runs (the tag's measure's verdict:
     `replay_rescued` on committed windows, `commit_replay_start` on null ones), the window index of each;
(R2) judged and committed windows, minutes, seconds per committed window;
(R3) sil_iou, thin_uncovered, chamfer (the run's metrics), v_mean / kin of the last committed window, pieces of >= 10
     particles at the delivered end (D162's rule: components at 2 x the median nearest distance); 40k: the exact-mesh
     readings of d154_mesh_read.py (D154_REF=own) from output/gpu/d164/read/g40_*.jsonl against d154/tables40's
     g40_table.json; 300k: t3end (the yardstick, exterior silhouette / shading), e3end (display front 1 - IoU), m_
     (momentum summary);
(R4) g_share, the last lambda;
G (pre-registered, D162's rule per arm against the same arm of the tag): a reading misses if the two-seed mean is worse
than the tag's worse seed of 97 / 99 by more than |tag97 - tag99|; rest if the larger end v_mean > 1.25 x the tag's
larger; pieces if more pieces >= 10 than the tag's larger; each miss is also read against the tag's own seed 101."""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
O = G / "d164"
MESHES = ("A armadilo beast bimba bob bunny C cheburashka cow dragon fandisk heart homer maxplanck nefertiti ogre spot "
          "teapot V").split()
GATE = ("armadilo", "bunny", "dragon", "homer", "teapot")
TAG_ARM = {"L": "V", "P": "W"}
RUN_READ = (("sil_iou", +1), ("thin_uncovered", -1), ("chamfer", -1))
MESH_READ = (("outline_sd", lambda r: r["exact_1024"]["outline_state_sd_a"]),
             ("iou1024", lambda r: r["exact_1024"]["one_minus_iou"]),
             ("normal_cl", lambda r: r["dist"]["normal_closest_deg"]),
             ("yard_sil", lambda r: r["yard_mesh"]["sil"]),
             ("yard_pbr", lambda r: r["yard_mesh"]["pbr"]))


def pieces(X):
    X = np.asarray(X, np.float64)
    tree = cKDTree(X)
    d1 = tree.query(X, 2, workers=8)[0][:, 1]
    pr = tree.query_pairs(2.0 * float(np.median(d1)), output_type="ndarray")
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(X), len(X)))
    _, lab = connected_components(g, directed=False)
    cnt = np.bincount(lab)
    rest = np.delete(cnt, np.argmax(cnt))
    return int(len(X) - cnt.max()), int((rest >= 10).sum())


def history_read(js_path):
    js = json.load(open(js_path))
    a = list(js["arms"].values())[0]
    H = a["history"]
    judged = [h for h in H if "animation" in h and not (h.get("grad_converged") and len(h) <= 2)]
    com = [h for h in H if h.get("frame_end") and not h.get("null_commit")]
    nulls = [h for h in H if h.get("null_commit") and not h.get("outer_rejected")]
    by = {}
    for h in nulls:
        by[h.get("null_reason")] = by.get(h.get("null_reason"), 0) + 1
    cr = [h["animation"] for h in nulls if h.get("null_reason") == "commit_replay"]
    rescued = [h["animation"] for h in H if h.get("replay_rescued")]
    cf_null = [h["animation"] for h in nulls if h.get("commit_replay_start")]
    met = a.get("metrics", {})
    sh = [h.get("g_share") for h in com if h.get("g_share") is not None]
    return dict(judged=len(judged), committed=len(com), outer_rejected=sum(1 for h in H if h.get("outer_rejected")),
                nulls=by, commit_replay=len(cr), commit_replay_at=cr, rescued=len(rescued), rescued_at=rescued,
                cf_tag_nulls=len(rescued) + len(cf_null), minutes=a.get("seconds", 0.0) / 60.0,
                s_per_commit=a.get("seconds", 0.0) / max(len(com), 1),
                sil_iou=met.get("sil_iou"), thin_uncovered=met.get("thin_uncovered"), chamfer=met.get("chamfer"),
                v_mean=com[-1].get("v_mean") if com else None, kin=com[-1].get("kin") if com else None,
                g_share=float(np.median(sh)) if sh else None, lam_last=com[-1].get("lambda") if com else None,
                deliver_n=a.get("deliver_n"))


def end_frame(stem):
    """(the delivered end frame, the run's target sample): the end-only file first (B1's 300k kept frames are ~1 GB)."""
    X = tg = None
    for suf in ("_end1.npz", "_frames12.npz"):
        p = Path(str(stem) + suf)
        if p.exists():
            z = np.load(p, allow_pickle=True)
            if X is None:
                X = np.asarray(z["frames"][-1])
            if tg is None and "tgt" in z.files:
                tg = np.asarray(z["tgt"])
            if tg is not None:
                break
    return X, tg


def stem40(mesh, arm, seed):
    if arm in ("V", "W"):
        for d in ([G / "d155/gate"] if mesh in GATE else []) + [G / "d154/g40runs"]:
            if (d / f"{mesh}40k_{arm}{seed}.json").exists():
                return d / f"{mesh}40k_{arm}{seed}"
        return None
    p = O / "runs40" / f"{mesh}40k_{arm}{seed}"
    return p if Path(str(p) + ".json").exists() else None


def mesh_rows():
    """D164's exact-mesh rows (output/gpu/d164/read/g40_*.jsonl) and the tag's (g40_table.json, per seed)."""
    mine = {}
    for f in glob.glob(str(O / "read/g40_*.jsonl")):
        for s in open(f):
            if s.startswith("{"):
                r = json.loads(s)
                mine[(r["mesh"], int(r["seed"]), r["label"])] = {k: fn(r) for k, fn in MESH_READ}
    tab = json.load(open(G / "d154/tables40/g40_table.json"))
    key = {"outline_sd": "outline_sd", "iou1024": "iou1024", "normal_cl": "normal_cl", "yard_sil": "yard_mesh_sil",
           "yard_pbr": "yard_mesh_pbr"}
    tag = {}
    for k, tk in key.items():
        for mesh, row in tab[tk]["meshes"].items():
            for ps in row.get("per_seed", []):
                for arm in ("V", "W", "own"):
                    tag.setdefault((mesh, int(ps["seed"]), arm), {})[k] = ps[arm]
    return mine, tag


def gallery(L):
    mine_mesh, tag_mesh = mesh_rows()
    R = {}
    for mesh in MESHES:
        for arm in ("L", "P", "V", "W"):
            for seed in ((97, 99) if arm in ("L", "P") else (97, 99, 101)):
                st = stem40(mesh, arm, seed)
                if st is None:
                    continue
                r = history_read(str(st) + ".json")
                X, tg = end_frame(st)
                if X is not None:
                    r["outside"], r["pieces10"] = pieces(X)
                r.update((mine_mesh if arm in ("L", "P") else tag_mesh).get((mesh, seed, arm), {}))
                if arm in ("L", "P"):
                    _, ttg = end_frame(stem40(mesh, TAG_ARM[arm], seed)) if stem40(mesh, TAG_ARM[arm], seed) else (None, None)
                    r["same_target"] = bool(tg is not None and ttg is not None and np.array_equal(tg, ttg))
                R[(mesh, arm, seed)] = r
    # (R1) / (R2): nulls and length, per arm
    L += ["## (R1) null windows and (R2) run length, 40k", "",
          "| arm | runs | judged | committed | commit_replay nulls | share | other nulls | rescued (tag would null) | minutes (median) | s / committed window (median) |",
          "|" + "---|" * 10]
    for arm, seeds in (("V", (97, 99, 101)), ("L", (97, 99)), ("W", (97, 99, 101)), ("P", (97, 99))):
        rows = [R[(m, arm, s)] for m in MESHES for s in seeds if (m, arm, s) in R]
        if not rows:
            continue
        j, c = sum(r["judged"] for r in rows), sum(r["committed"] for r in rows)
        cr = sum(r["commit_replay"] for r in rows)
        oth = {}
        for r in rows:
            for k, v in r["nulls"].items():
                if k != "commit_replay":
                    oth[k] = oth.get(k, 0) + v
        L.append(f"| {arm} | {len(rows)} | {j} | {c} | {cr} | {100 * cr / max(j, 1):.2f} % | {json.dumps(oth)} | "
                 f"{sum(r['cf_tag_nulls'] for r in rows)} ({100 * sum(r['cf_tag_nulls'] for r in rows) / max(j, 1):.2f} %) | "
                 f"{np.median([r['minutes'] for r in rows]):.2f} | {np.median([r['s_per_commit'] for r in rows]):.2f} |")
    # (R3) and G
    reads = [k for k, _ in RUN_READ] + [k for k, _ in MESH_READ]
    sign = dict(RUN_READ)
    sign.update({k: -1 for k, _ in MESH_READ})
    L += ["", "## (R3) per mesh, the D164 arm's two seeds against the tag's three (G: D162's rule)", "",
          "| mesh | arm | reading | tag 97 / 99 / 101 | D164 97 / 99 | miss | the tag's 101 alone would miss |", "|" + "---|" * 7]
    miss = {"L": {"read": [], "rest": [], "pieces": [], "cells": 0}, "P": {"read": [], "rest": [], "pieces": [], "cells": 0}}
    f4 = lambda v: "-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))  # noqa: E731
    for mesh in MESHES:
        for arm in ("L", "P"):
            ta = TAG_ARM[arm]
            t = [R.get((mesh, ta, s)) for s in (97, 99, 101)]
            d = [R.get((mesh, arm, s)) for s in (97, 99)]
            if t[0] is None or t[1] is None or None in d:
                L.append(f"| {mesh} | {arm} | (missing runs) | | | | |")
                continue
            for key in reads + ["v_mean", "pieces10", "committed", "minutes", "g_share"]:
                tv = [x.get(key) if x else None for x in t]
                dv = [x.get(key) for x in d]
                mark, m101 = "", ""
                if None not in (tv[0], tv[1], dv[0], dv[1]):
                    if key == "v_mean":
                        bad = max(dv) > 1.25 * max(tv[:2])
                        b101 = tv[2] is not None and tv[2] > 1.25 * max(tv[:2])
                        if bad:
                            miss[arm]["rest"].append(mesh)
                    elif key == "pieces10":
                        bad = max(dv) > max(tv[:2])
                        b101 = tv[2] is not None and tv[2] > max(tv[:2])
                        if bad:
                            miss[arm]["pieces"].append(mesh)
                    elif key in sign:
                        sg = sign[key]
                        worse = min(tv[:2]) if sg > 0 else max(tv[:2])
                        spread = abs(tv[0] - tv[1])
                        bad = sg * (worse - 0.5 * (dv[0] + dv[1])) > spread
                        b101 = tv[2] is not None and sg * (worse - tv[2]) > spread
                        miss[arm]["cells"] += 1
                        if bad:
                            miss[arm]["read"].append(f"{mesh}:{key}")
                    else:
                        bad = b101 = False
                    mark, m101 = ("MISS" if bad else ""), ("yes" if b101 else "")
                L.append(f"| {mesh} | {arm} | {key} | {' / '.join(f4(v) for v in tv)} | {' / '.join(f4(v) for v in dv)} | "
                         f"{mark} | {m101} |")
    L += [""]
    for arm in ("L", "P"):
        g = miss[arm]
        ok = len(g["read"]) <= g["cells"] // 10 and len(g["rest"]) <= 1 and not g["pieces"]
        L.append(f"- G, arm {arm} against the tag's {TAG_ARM[arm]}: reading misses {len(g['read'])} of {g['cells']} "
                 f"{g['read']}; rest misses {g['rest']}; pieces misses {g['pieces']}: G {'HOLDS' if ok else 'FAILS'}")
    same = [k for k, r in R.items() if k[1] in ("L", "P") and not r.get("same_target")]
    L.append(f"- target samples identical to the tag's run of the same mesh / seed: "
             f"{sum(1 for k, r in R.items() if k[1] in ('L', 'P') and r.get('same_target'))} of "
             f"{sum(1 for k in R if k[1] in ('L', 'P'))} (not: {same})")
    # inside the tag's three-run range
    inside = {"L": [0, 0], "P": [0, 0]}
    for mesh in MESHES:
        for arm in ("L", "P"):
            t = [R.get((mesh, TAG_ARM[arm], s)) for s in (97, 99, 101)]
            d = [R.get((mesh, arm, s)) for s in (97, 99)]
            if None in t or None in d:
                continue
            for key in reads:
                tv = [x.get(key) for x in t]
                dv = [x.get(key) for x in d]
                if None in tv or None in dv:
                    continue
                inside[arm][1] += 1
                inside[arm][0] += int(min(tv) <= 0.5 * (dv[0] + dv[1]) <= max(tv))
    for arm in ("L", "P"):
        L.append(f"- arm {arm}: two-seed mean inside the tag's 97 / 99 / 101 range on {inside[arm][0]} of {inside[arm][1]} "
                 "(mesh x reading) cells")
    return R


def n300(L):
    rows = {}
    for mesh in ("bunny", "dragon"):
        for arm in ("L", "P"):
            for src, seeds in (("d164/runs300", (97,)), ("d145t_b1", (97, 99, 101))):
                for s in seeds:
                    st = G / src / f"{mesh}300k_{arm}{s}"
                    if not Path(str(st) + ".json").exists():
                        continue
                    r = history_read(str(st) + ".json")
                    X, _ = end_frame(st)
                    if X is not None:
                        r["outside"], r["pieces10"] = pieces(X)
                    d = G / src
                    try:
                        t3 = [json.loads(s_) for s_ in open(d / f"t3end_{mesh}300k_{arm}{s}.log") if s_.startswith("{")][-1]
                        r.update(yard_sil=t3["exterior"]["sil"], yard_pbr=t3["exterior"]["pbr"])
                    except (OSError, IndexError, KeyError):
                        pass
                    try:
                        e3 = [json.loads(s_) for s_ in open(d / f"e3end_{mesh}300k_{arm}{s}.log") if s_.startswith("{")]
                        st_ = [x for x in e3 if x.get("state") not in ("target", "own") and "to_target" in x][-1]
                        r.update(display_front=1.0 - st_["to_target"]["front"]["iou"])
                    except (OSError, IndexError, KeyError):
                        pass
                    try:
                        mt = open(d / f"m_{mesh}300k_{arm}{s}.log").read()
                        m = re.search(r"net / gross angular move: largest ([0-9.e+-]+), mean ([0-9.e+-]+), last 20 ([0-9.e+-]+)", mt)
                        if m:
                            r.update(ang_largest=float(m.group(1)), ang_mean=float(m.group(2)), ang_last20=float(m.group(3)))
                    except OSError:
                        pass
                    rows[(mesh, arm, "D164" if src.startswith("d164") else "tag", s)] = r
    keys = ("sil_iou", "chamfer", "thin_uncovered", "yard_sil", "yard_pbr", "display_front", "v_mean", "kin", "pieces10",
            "ang_last20", "committed", "judged", "commit_replay", "cf_tag_nulls", "minutes", "g_share", "lam_last")
    L += ["", "## 300k: D164 seed 97 against B1 (the tag) 97 / 99 / 101", "",
          "| mesh | arm | reading | tag 97 / 99 / 101 | D164 97 | inside the tag's range |", "|" + "---|" * 6]
    f4 = lambda v: "-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))  # noqa: E731
    for mesh in ("bunny", "dragon"):
        for arm in ("L", "P"):
            t = [rows.get((mesh, arm, "tag", s), {}) for s in (97, 99, 101)]
            d = rows.get((mesh, arm, "D164", 97), {})
            for k in keys:
                tv = [x.get(k) for x in t]
                dv = d.get(k)
                ok = ""
                if dv is not None and None not in tv and isinstance(dv, (int, float)):
                    ok = "yes" if min(tv) <= dv <= max(tv) else ("above" if dv > max(tv) else "below")
                L.append(f"| {mesh} | {arm} | {k} | {' / '.join(f4(v) for v in tv)} | {f4(dv)} | {ok} |")
    return rows


def main():
    out = Path(sys.argv[1])
    L = ["# D164 readings", ""]
    R = gallery(L)
    rows = n300(L) if "--n300" in sys.argv else {}
    out.write_text("\n".join(L) + "\n")
    js = {"gallery": {f"{m}|{a}|{s}": r for (m, a, s), r in R.items()},
          "n300": {"|".join(map(str, k)): r for k, r in rows.items()}}
    out.with_suffix(".json").write_text(json.dumps(js, indent=1, default=str))
    print("\n".join(x for x in L if x.startswith("- ") or x.startswith("| L |") or x.startswith("| P |")
                    or x.startswith("| V |") or x.startswith("| W |")))


if __name__ == "__main__":
    main()
