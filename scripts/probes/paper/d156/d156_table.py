"""d156_table.py READ_JSONL RUNS_DIR OUT_JSON -- D156: the shares, the pre-registered decision and the runs' records.

From d156_read.py's rows, per context (mesh, seed, frame), with H the hull's perfect sample, M the mesh's, Mp an
independent draw of the mesh:
  share_vol(X)  = (m_H - m_X) / (m_H - m_M)         m = the share of the particles in hull minus mesh
  share_surf(X) = (S_X - S_H) / (S_M - S_H)          S = the summed signed distance of the image-visible mesh surface
                                                     to X's exterior (also per depth bin)
  share_sil(X)  = (e_H - e_X) / (e_H - e_Mp)         e = 1 - IoU against M's exterior, the 46 held-out views
  share_img(X)  = (q_H - q_X) / (q_H - q_Mp)         q = the held-out studio views' MSE (shaded, normal map) and LPIPS
                                                     against M's display surface
Delta = share(L) - share(P) per mesh-seed; per mesh the mean over the seeds. The decision (the pre-registration):
PASS if Delta_surf (amended 13:59, before any D156 run: the share on the points at least half a pitch below the hull,
surf_deep; the all-points form alongside) >= 0.15 and Delta_vol >= 0.10 on >= 6 of 9 meshes with the same sign at both seeds on each and the
median Delta_surf >= 0.15; STRONG if Delta_surf >= 0.5 on >= 6 of 9; FAIL if Delta_surf < 0.05 on >= 6 of 9; else PARTIAL.
The runs' records (RUNS_DIR/MESH40k_ARM_SEED.json, _frames12.npz): minutes, committed windows, gates, the body's volume,
the render terms at the end, the centre-of-mass drift (pitches) and the net rotation (the polar part of the best affine
map from the first to the last kept frame, degrees)."""
import json
import sys
from collections import defaultdict

import numpy as np

rows = [json.loads(ln) for ln in open(sys.argv[1]) if ln.strip()]
runs_dir, out_path = sys.argv[2], sys.argv[3]
RUNF = sys.argv[4] if len(sys.argv) > 4 else "hull"           # the runs' frame: hull (D156) or sfs (D156d)
ctx = defaultdict(dict)
for r in rows:
    ctx[(r["mesh"], r["seed"], r["frame"])][r["label"]] = r


def share(ref0, ref1, x):
    d = ref0 - ref1
    return float((ref0 - x) / d) if abs(d) > 1e-12 else None


def shares(c, lab):
    X, H, M, Mp = c[lab], c["H"], c["M"], c["Mp"]
    out = dict(vol=share(H["vol"]["in_R"], M["vol"]["in_R"], X["vol"]["in_R"]),
               surf=share(-H["surf"]["S"], -M["surf"]["S"], -X["surf"]["S"]),
               surf_bins=[share(-h["S"], -m["S"], -x["S"]) for h, m, x in
                          zip(H["surf"]["bins"], M["surf"]["bins"], X["surf"]["bins"])],
               surf_deep=share(-sum(b["S"] for b in H["surf"]["bins"][1:]), -sum(b["S"] for b in M["surf"]["bins"][1:]),
                               -sum(b["S"] for b in X["surf"]["bins"][1:])),
               sil_held=share(H["sil"]["held"], Mp["sil"]["held"], X["sil"]["held"]),
               sil_train=share(H["sil"]["train"], Mp["sil"]["train"], X["sil"]["train"]))
    for kind in ("shaded", "normal"):
        for q in ("mse", "lpips"):
            out[f"img_{kind}_{q}"] = share(H["img"]["M"][kind][q], Mp["img"]["M"][kind][q], X["img"]["M"][kind][q])
    out["pic_sil"] = share(H["pic"]["sil"], Mp["pic"]["sil"], X["pic"]["sil"])
    out["pic_pbr"] = share(H["pic"]["pbr"], Mp["pic"]["pbr"], X["pic"]["pbr"])
    return out


def record(mesh, arm, seed):
    try:
        js = json.load(open(f"{runs_dir}/{mesh}40k_{arm}_{seed}.json"))
    except OSError:
        return None
    a = list(js["arms"].values())[0]
    h = a["history"]
    com = [r for r in h if r.get("frame_end") and not r.get("null_commit")]
    end = [r for r in h if "Jx_mean" in r]
    rec = dict(minutes=a["seconds"] / 60, windows=len(com), deliver_n=a["deliver_n"],
               G3_rest=a["gates"].get("G3_rest"), drift_rel=a["gates"].get("drift_rel"),
               jitter_rel=a["metrics"].get("jitter_rel"), Jx_mean=end[-1]["Jx_mean"] if end else None,
               d_sil_end=com[-1]["d_sil"] if com else None, d_pbr_end=com[-1]["d_pbr"] if com else None,
               lam_end=com[-1]["lambda"] if com else None)
    z = np.load(f"{runs_dir}/{mesh}40k_{arm}_{seed}_frames12.npz", allow_pickle=True)
    x0, x1 = np.asarray(z["frames"][0], np.float64), np.asarray(z["frames"][-1], np.float64)
    c0, c1 = x0.mean(0), x1.mean(0)
    A = np.linalg.lstsq(np.c_[x0 - c0, np.ones(len(x0))], x1 - c1, rcond=None)[0][:3].T
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    rec.update(com_drift_wu=float(np.linalg.norm(c1 - c0)),
               rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))))
    return rec


table, deltas = {}, defaultdict(dict)
for (mesh, seed, frame), c in sorted(ctx.items()):
    if not all(k in c for k in ("H", "M", "Mp")):
        continue
    a = c["H"]["ctx"]["pitch"]
    arms = ("L", "P") if frame in ("hull", "sfs") else ("V", "W")
    if frame == "sfs" and "S" in c:
        arms = arms + ("S",)                                          # D156d's solid itself: reading (a)
    t = {lab: shares(c, lab) for lab in arms if lab in c}
    t["Mp"] = shares(c, "Mp")                                   # the floor's own share (about 1 by construction)
    t["raw"] = {lab: dict(in_R=c[lab]["vol"]["in_R"], out_hull=c[lab]["vol"]["out_hull"],
                          compression=c[lab]["vol"]["compression"], surf_mean_a=c[lab]["surf"]["mean"],
                          sil_held=c[lab]["sil"]["held"], sil_train=c[lab]["sil"]["train"],
                          psnr_true=c[lab]["img"]["true"]["shaded"]["psnr"], lpips_true=c[lab]["img"]["true"]["shaded"]["lpips"],
                          pic_sil=c[lab]["pic"]["sil"], pic_pbr=c[lab]["pic"]["pbr"],
                          pieces=c[lab].get("pieces"))
                for lab in list(arms) + ["H", "M", "Mp"] if lab in c}
    t["ctx"] = c["H"]["ctx"]
    if frame == RUNF:
        for lab in ("L", "P"):
            rec = record(mesh, lab, seed)
            if rec is not None:
                rec["com_drift_pitch"] = rec.pop("com_drift_wu") / a
                t["raw"][lab]["record"] = rec
        if "L" in t and "P" in t:
            deltas[mesh][seed] = {k: (t["L"][k] - t["P"][k]) if isinstance(t["L"][k], float) and isinstance(t["P"][k], float)
                                  else None for k in ("vol", "surf", "surf_deep", "sil_held", "img_shaded_mse", "img_normal_mse",
                                                      "img_shaded_lpips", "pic_sil", "pic_pbr")}
    table[f"{mesh}_{seed}_{frame}"] = t

per_mesh = {}
for mesh, by_seed in deltas.items():
    ks = by_seed[next(iter(by_seed))].keys()
    per_mesh[mesh] = {k: dict(mean=float(np.mean([v[k] for v in by_seed.values() if v[k] is not None]))
                              if any(v[k] is not None for v in by_seed.values()) else None,
                              seeds={str(s): v[k] for s, v in by_seed.items()}) for k in ks}
n = len(per_mesh)
ds = {m: v["surf_deep"]["mean"] for m, v in per_mesh.items()}           # the amended primary (13:59 entry)
ds_all = {m: v["surf"]["mean"] for m, v in per_mesh.items()}          # the pre-registered all-points form, alongside
dv = {m: v["vol"]["mean"] for m, v in per_mesh.items()}
both_seeds = {m: all(np.sign(x) == np.sign(ds[m]) for x in per_mesh[m]["surf_deep"]["seeds"].values() if x is not None) and
              all(np.sign(x) == np.sign(dv[m]) for x in per_mesh[m]["vol"]["seeds"].values() if x is not None)
              for m in per_mesh}
k_pass = sum(1 for m in per_mesh if ds[m] >= 0.15 and dv[m] >= 0.10 and both_seeds[m])
k_strong = sum(1 for m in per_mesh if ds[m] >= 0.5)
k_fail = sum(1 for m in per_mesh if ds[m] < 0.05)
med = float(np.median(list(ds.values()))) if ds else None
need = int(np.ceil(6 * n / 9)) if n else 0
verdict = ("STRONG" if n and k_strong >= need else "PASS" if n and k_pass >= need and med >= 0.15 else
           "FAIL" if n and k_fail >= need else "PARTIAL")
k_pass_all = sum(1 for m in per_mesh if ds_all[m] >= 0.15 and dv[m] >= 0.10)
k_fail_all = sum(1 for m in per_mesh if ds_all[m] < 0.05)
summary = dict(all_points=dict(delta_surf=ds_all, k_pass=k_pass_all, k_fail=k_fail_all,
                               median=float(np.median(list(ds_all.values()))) if ds_all else None),
               meshes=n, need=need, k_pass=k_pass, k_strong=k_strong, k_fail=k_fail, median_delta_surf=med,
               verdict=verdict, delta_surf=ds, delta_vol=dv, same_sign_both_seeds=both_seeds)
json.dump(dict(summary=summary, per_mesh=per_mesh, table=table), open(out_path, "w"), indent=1)
print(json.dumps(summary, indent=1))
for m in sorted(per_mesh):
    v = per_mesh[m]
    print(f"{m:12s} d_vol {v['vol']['mean']:+.3f} {v['vol']['seeds']}  d_surf_deep {v['surf_deep']['mean']:+.3f} {v['surf_deep']['seeds']}  d_surf_all {v['surf']['mean']:+.3f}  "
          f"d_sil {v['sil_held']['mean'] if v['sil_held']['mean'] is not None else float('nan'):+.3f}")
