"""d162_gallery.py OUT_MD OUT_JSON -- D162's gallery regression (CPU): the 19 plain 40k targets, arms F (render weight
referenced to its floor) and FC (+ placement terms at the loss cell), seeds 97 / 99, against the tag's render arm V and
twin W (D155's gate frames for armadilo / bunny / dragon / homer / teapot, D154 (3b)'s g40runs for the rest).

Per run: the run's own metrics (sil_iou, thin_uncovered, chamfer: against the 3D target sample), the end at rest (the
last committed window's v_mean and kin), pieces at the delivered end (particles outside the largest component at
radius 2 x the median nearest distance; pieces of >= 10 particles), windows, wall seconds, the last lambda.
Misses (pre-registered): for each reading, an arm misses on a mesh if its two-seed mean is worse than the tag's worse
seed (V97, V99) by more than the tag's two-run spread |V97 - V99|; rest misses if the arm's larger end v_mean exceeds
1.25 x the tag's larger; pieces miss if the arm has more pieces of >= 10 particles than the tag's larger count."""
import glob
import json
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
MESHES = ("bunny dragon armadilo homer teapot A beast bimba bob C cheburashka cow fandisk heart maxplanck nefertiti "
          "ogre spot V").split()
GATE = ("armadilo", "bunny", "dragon", "homer", "teapot")
READ = (("sil_iou", +1), ("thin_uncovered", -1), ("chamfer", -1))


def paths(mesh, arm, seed):
    if arm in ("V", "W"):
        cand = ([f"{G}/d155/gate/{mesh}40k_{arm}{seed}"] if mesh in GATE else []) + [f"{G}/d154/g40runs/{mesh}40k_{arm}{seed}"]
    else:
        cand = [f"{G}/d162/runs40/gal_{mesh}40k_{arm}_{seed}"]
    for c in cand:
        js = glob.glob(c + ".json")
        fr = glob.glob(c + "_frames12.npz") or glob.glob(c + "_end1.npz")
        if js and fr:
            return js[0], fr[0]
    return None, None


def pieces(X):
    d1 = cKDTree(X).query(X, 2, workers=8)[0][:, 1]
    pr = cKDTree(X).query_pairs(2.0 * float(np.median(d1)), output_type="ndarray")
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(X), len(X)))
    nc, lab = connected_components(g, directed=False)
    cnt = np.bincount(lab)
    rest = np.sort(np.delete(cnt, np.argmax(cnt)))[::-1]
    return int(len(X) - cnt.max()), int((rest >= 10).sum())


def read(mesh, arm, seed):
    jp, fp = paths(mesh, arm, seed)
    if jp is None:
        return None
    js = json.load(open(jp))
    a = list(js["arms"].values())[0]
    com = [h for h in a["history"] if h.get("frame_end") and not h.get("null_commit")]
    met = a.get("metrics", {})
    z = np.load(fp, allow_pickle=True)
    X = np.asarray(z["frames"][-1], np.float64)
    out_n, p10 = pieces(X)
    return dict(sil_iou=met.get("sil_iou"), thin_uncovered=met.get("thin_uncovered"), chamfer=met.get("chamfer"),
                v_mean=com[-1].get("v_mean"), kin=com[-1].get("kin"), outside=out_n, pieces10=p10, windows=len(com),
                seconds=a.get("seconds"), lam_last=com[-1].get("lambda"),
                render_excess_last=com[-1].get("render_excess"), render_floor=com[-1].get("render_floor"))


def main():
    out_md, out_js = sys.argv[1], sys.argv[2]
    R = {m: {f"{arm}{s}": read(m, arm, s) for arm in ("V", "W", "F", "FC") for s in (97, 99)} for m in MESHES}
    json.dump(R, open(out_js, "w"), indent=1)
    L = ["| mesh | reading | V97 / V99 | W97 / W99 | F97 / F99 | FC97 / FC99 | F miss | FC miss |", "|" + "---|" * 8]
    miss = {"F": {"read": 0, "rest": [], "pieces": []}, "FC": {"read": 0, "rest": [], "pieces": []}}
    cells = 0
    for m in MESHES:
        r = R[m]
        if r["V97"] is None or r["V99"] is None:
            L.append(f"| {m} | (no tag reference) |  |  |  |  |  |  |")
            continue
        for key, sign in READ + (("v_mean", -1), ("pieces10", -1), ("windows", 0)):
            vals = {k: (r[k] or {}).get(key) if r[k] else None for k in r}
            fmt = lambda a, b: f"{a if a is None else round(a, 4)} / {b if b is None else round(b, 4)}"  # noqa: E731
            mm = {}
            for arm in ("F", "FC"):
                a97, a99 = vals[f"{arm}97"], vals[f"{arm}99"]
                v97, v99 = vals["V97"], vals["V99"]
                if None in (a97, a99, v97, v99) or sign == 0:
                    mm[arm] = ""
                    continue
                if key == "v_mean":
                    bad = max(a97, a99) > 1.25 * max(v97, v99)
                    if bad:
                        miss[arm]["rest"].append(m)
                elif key == "pieces10":
                    bad = max(a97, a99) > max(v97, v99)
                    if bad:
                        miss[arm]["pieces"].append(m)
                else:
                    worse = min(v97, v99) if sign > 0 else max(v97, v99)
                    am = 0.5 * (a97 + a99)
                    bad = sign * (worse - am) > abs(v97 - v99)
                    if bad:
                        miss[arm]["read"] += 1
                mm[arm] = "MISS" if bad else ""
            if key in dict(READ):
                cells += 1
            L.append(f"| {m} | {key} | {fmt(vals['V97'], vals['V99'])} | {fmt(vals['W97'], vals['W99'])} | "
                     f"{fmt(vals['F97'], vals['F99'])} | {fmt(vals['FC97'], vals['FC99'])} | {mm.get('F', '')} | {mm.get('FC', '')} |")
    L += ["", f"cells (mesh x reading): {cells}"]
    for arm in ("F", "FC"):
        g = miss[arm]
        ok = g["read"] <= cells // 10 and len(g["rest"]) <= 1 and not g["pieces"]
        L.append(f"{arm}: reading misses {g['read']} of {cells}; rest misses {g['rest']}; pieces misses {g['pieces']}; "
                 f"G {'HOLDS' if ok else 'FAILS'}")
    open(out_md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-3:]))


if __name__ == "__main__":
    main()
