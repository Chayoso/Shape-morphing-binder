"""d155_b7.py MESH OUT_JSON [WORKERS] -- D155 (CPU, read-only): who moves the body in each phase of the run, from B7's
RG97 (the tag + hooks: at every window the committed end X_k and the lambda = 0 counterfactual Y_k from the same start
x0_k; output/gpu/d145t_b7/MESH300k_RG97_cf/cf_k.npz).

Per window k (pitches; pitch as D152):
  D   = X_k - x0_k      the window's actual displacement
  D0  = Y_k - x0_k      what the same window does with lambda = 0 (physics alone, from the same start)
  d   = X_k - Y_k       the render-caused part (D = D0 + d)
  on the END state's outer layer (D152's regions_end, Lagrangian), its rim / thin parts, and all particles:
  rms |D|, |D0|, |d|; the render's share of the actual displacement: sum d . D / sum |D|^2 (projection) and
  sum |d|^2 / sum |D|^2; the same on the normal component (n at x0_k, 24 nearest), the part the eye sees.
  the window's own terms with and without the render (the cf / render dicts): d_sil, d_pbr, d_vol, kin.
Only windows on the committed chain (x0_{k+1} = X_k bit for bit) inside the delivered run, as D152."""
import ast
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
mesh, out = sys.argv[1], sys.argv[2]
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
RUN = f"{G}/d145t_b7/{mesh}300k_RG97"
J = json.load(open(RUN + ".json"))
arm = next(iter(J["arms"].values()))
DN = int(arm["deliver_n"])
FE = {h["animation"]: h.get("frame_end") for h in arm["history"]}
NULL = {h["animation"]: bool(h.get("null_commit")) for h in arm["history"]}
tgt = np.asarray(np.load(f"{G}/d145t_b1/{mesh}300k_L97_end1.npz")["tgt"], np.float64)
PITCH = 0.708 * float(np.median(cKDTree(tgt).query(tgt[::50], 9)[0][:, 8]))
REG = {k: v.astype(bool) for k, v in np.load(f"{G}/d152/h345v2/{mesh}_regions_end.npz").items() if k != "conc"}
LAY = REG["layer"]


def stats(D, D0, d, n, msk):
    a, a0, r = D[msk], D0[msk], d[msk]
    DD = float((a * a).sum())
    o = dict(D=float(np.sqrt((a * a).sum(1).mean())), D0=float(np.sqrt((a0 * a0).sum(1).mean())),
             d=float(np.sqrt((r * r).sum(1).mean())), proj=float((r * a).sum() / max(DD, 1e-300)),
             e2=float((r * r).sum() / max(DD, 1e-300)), DD=DD, dD=float((r * a).sum()), dd=float((r * r).sum()))
    if n is not None:
        nn = n[msk]
        an, rn, a0n = (a * nn).sum(1), (r * nn).sum(1), (a0 * nn).sum(1)
        o.update(Dn=float(np.sqrt((an ** 2).mean())), D0n=float(np.sqrt((a0n ** 2).mean())),
                 dn=float(np.sqrt((rn ** 2).mean())), projn=float((rn * an).sum() / max(float((an ** 2).sum()), 1e-300)),
                 DDn=float((an ** 2).sum()), dDn=float((rn * an).sum()), ddn=float((rn ** 2).sum()))
    return o


def per_window(k):
    c = np.load(f"{RUN}_cf/cf_{k:04d}.npz", allow_pickle=True)
    if not len(c["x_render"]) or not len(c["x_phys"]):
        return k, None, None, None
    x0 = c["x0"].astype(np.float64)
    xr, xp = c["x_render"].astype(np.float64), c["x_phys"].astype(np.float64)
    D, D0, d = (xr - x0) / PITCH, (xp - x0) / PITCH, (xr - xp) / PITCH
    tree = cKDTree(x0)
    idx = np.flatnonzero(LAY)
    nb = tree.query(x0[idx], 25, workers=2)[1][:, 1:]
    n = np.zeros_like(x0)
    off = x0[idx] - x0[nb].mean(1)
    n[idx] = off / np.maximum(np.linalg.norm(off, axis=1, keepdims=True), 1e-12)
    row = {"k": k, "fe": FE.get(k), "null": NULL.get(k, False)}
    rr, rp = ast.literal_eval(str(c["render"])), ast.literal_eval(str(c["cf"]))
    for key in ("d_sil", "d_pbr", "d_vol", "kin", "loss_end"):
        row[f"{key}_r"], row[f"{key}_p"] = rr.get(key), rp.get(key)
    allm = np.ones(len(x0), bool)
    row["all"] = stats(D, D0, d, None, allm)
    for nm in ("layer", "rim", "thin"):
        row[nm] = stats(D, D0, d, n, REG[nm])
    return k, row, x0[:2000].copy(), xr[:2000].copy()


def main():
    ks = sorted(int(os.path.basename(f)[3:7]) for f in glob.glob(f"{RUN}_cf/cf_*.npz"))
    rows, X0s, XRs = {}, {}, {}
    with Pool(workers) as pool:
        for k, row, x0s, xrs in pool.imap_unordered(per_window, ks, chunksize=1):
            if row is None:
                continue
            rows[k], X0s[k], XRs[k] = row, x0s, xrs
    K = sorted(rows)
    for k in K:
        nxt = k + 1 in X0s and float(np.abs(X0s[k + 1] - XRs[k]).max()) == 0.0
        fe = rows[k]["fe"]
        rows[k]["chained"] = bool(nxt and fe is not None and fe <= DN + 1)
        rows[k]["t"] = fe / 240.0 if fe else None
        rows[k]["f"] = fe / DN if fe else None
    last = max((k for k in K if rows[k]["fe"] is not None and rows[k]["fe"] <= DN + 1 and not rows[k]["null"]),
               default=None)
    if last is not None:
        rows[last]["chained"] = True
    res = {"mesh": mesh, "pitch": PITCH, "deliver_n": DN, "windows": len(K),
           "chained": int(sum(r["chained"] for r in rows.values())), "layer": int(LAY.sum()),
           "rows": [rows[k] for k in K]}
    json.dump(res, open(out, "w"), default=float)
    print(f"{mesh}: pitch {PITCH:.5f}; {len(K)} windows, {res['chained']} chained; deliver_n {DN}", flush=True)


if __name__ == "__main__":
    main()
