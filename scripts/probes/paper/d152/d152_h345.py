"""d152_h345.py MESH OUT_DIR [WORKERS] -- D152 (H3 undo, H4 reach, H5 PCGrad) on B7's tag run RG97 (repo_tgf: the
gradient dump and the lambda = 0 counterfactual at EVERY window, output/gpu/d145t_b7/MESH300k_RG97_{dump,cf}); CPU
only, read-only. Measurement only.

Definitions (pitch = 0.708 x the target sample's median 8th-neighbour distance, D95; window k = animation index):
  X_k        the arm's committed end of window k (cf x_render; the next window's x0, checked)
  Y_k        the same window from the same start with lambda = 0 (cf x_phys)
  d_k        X_k - Y_k, the render-caused displacement of window k (per particle, pitches)
  n          the outward normal at the window's start x0 (x minus the mean of its 24 nearest, normalised)
  d_n        d_k . n (+ = the render put the particle further out than lambda = 0 would)
  regions at the END (Lagrangian, the arm's delivered state): layer (|x - mean of 32 nn| >= 0.5 spacing, as
             b7_proof's run part), rim (layer particles on the outline of any of the 18 loss views, one-pitch occupancy
             grid, an occupied cell with an empty 4-neighbour), thin (max-ball thickness < 2 MPM cells, as b7_proof),
             crevice (layer particles whose layer neighbours within 4 pitches lie on average above the tangent plane:
             mean of (x_j - x_i).n_i / |x_j - x_i| > +0.15, a concave patch), bulk (> 2 pitches below the layer)
  undo, next window, physics only:  u_k = - sum (Y_{k+1} - X_k) . d_k / sum |d_k|^2 over the layer (window k+1's
             lambda = 0 counterfactual starts from X_k): 1 = the next physics-only window takes d_k back entirely
  arm motion along d_k after j windows: c_k(j) = sum (X_{k+j} - X_k) . d_k / sum |d_k|^2 (layer)
  run sums:  S = sum_k d_k (late: k >= K0), G = sum_k |d_k|; F = X_end(RG97) - Z_end(P97, the twin, same seed and
             particles); R = X_end(RG97) - X_end(L97) (the arm's unhooked replicate: the run-to-run spread);
             beta = sum F.S / sum S.S (the share of the summed render push that survives into the final difference)
  scalar bookkeeping of the silhouette term (the run's own d_sil, the records): gain_k = d_sil(Y_k) - d_sil(X_k);
             drift_k = d_sil(Y_k) - d_sil(X_{k-1}) (a physics-only window from the render-improved start); the twin's
             own per-window change at the same simulated time for comparison
  reach (H4): gx_r = gx_sil + gx_pbr (the render's position covector at the window's evaluation end, unweighted);
             dx_r = xT_rend - xT_base (the end moved by a dFc step along the render's own control gradient, the norm of
             the window's accepted change); dx_ru = xT_rend_u - xT_base_u0 (the same through u); reach = cos(-gx, dx)
             over all particles; low-pass share = gx . W(gx) / gx . gx with W = the MPM grid's quadratic B-spline P2G
             then G2P (mass-normalised), the cell-scale part of the demand; the same for the physics
  cap (H4):  the share of particles whose step-mean |dFc| (Frobenius) >= 0.95 x dfc_clip, render pass and lambda = 0
  PCGrad (H5): |g_r - g_r,pc| / |g_r| from the dump (dFc psum fields, an upper-bound proxy) and cos^2 removed from the
             records' g_raw_cos (the one-sided projection removes cos^2 of the render gradient's squared norm)"""
import ast
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
mesh, out_dir = sys.argv[1], sys.argv[2]
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
os.makedirs(out_dir, exist_ok=True)
RUN = f"{G}/d145t_b7/{mesh}300k_RG97"
J = json.load(open(RUN + ".json"))
arm = next(iter(J["arms"].values()))
hist = arm["history"]
mpm = J["provenance"]["mpm"]
DX, GMIN, NX = float(mpm["dx"]), np.asarray(mpm["grid_min"], np.float64), int(mpm["nx"])
CLIP = float(arm["config"]["dfc_clip"])


def views18():
    """the render loss's 18 views, as render_loss.make_views(6, (0, 0.5, -0.5)) (each ring its own azimuth offset)"""
    out, elevs, n = [], (0.0, 0.5, -0.5), 6
    for i, phi in enumerate(elevs):
        off = (i / len(elevs)) * (2 * np.pi / n)
        for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
            out.append((float(t + off), float(phi)))
    return out


def regions_end(x, pitch, cell):
    tree = cKDTree(x)
    sp = float(np.median(tree.query(x[::15], 9)[0][:, 8]))
    nb = tree.query(x, 33)[1][:, 1:]
    cen = x[nb].mean(1)
    off = x - cen
    lay = np.linalg.norm(off, axis=1) >= 0.5 * sp
    nrm = off / np.maximum(np.linalg.norm(off, axis=1, keepdims=True), 1e-12)
    dep = cKDTree(x[lay]).query(x, 1)[0] / pitch
    rim = np.zeros(len(x), bool)
    for th, phi in views18():
        right = np.array([np.cos(th), 0.0, -np.sin(th)])
        up = np.array([-np.sin(phi) * np.sin(th), np.cos(phi), -np.sin(phi) * np.cos(th)])
        p = np.stack([x @ right, x @ up], 1)
        ij = np.floor((p - p.min(0)) / pitch).astype(int)
        occ = np.zeros(ij.max(0) + 3, bool)
        occ[ij[:, 0] + 1, ij[:, 1] + 1] = True
        edge = occ & ~ndimage.binary_erosion(occ)
        rim |= edge[ij[:, 0] + 1, ij[:, 1] + 1]
    rim &= lay
    rim3 = np.zeros(len(x), bool)                         # B7's rim (b7_proof.py): the three axis projections
    for hh, vv in ((0, 1), (2, 1), (0, 2)):
        ij = np.floor((x[:, [hh, vv]] - x[:, [hh, vv]].min(0)) / pitch).astype(int)
        occ = np.zeros(ij.max(0) + 3, bool)
        occ[ij[:, 0] + 1, ij[:, 1] + 1] = True
        edge = occ & ~ndimage.binary_erosion(occ)
        rim3 |= edge[ij[:, 0] + 1, ij[:, 1] + 1]
    rim3 &= lay
    lo = x.min(0) - 3 * pitch
    dims = np.ceil((x.max(0) + 3 * pitch - lo) / pitch).astype(int) + 1
    ijk = np.floor((x - lo) / pitch).astype(int)
    occ = np.zeros(dims, bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    occ = ndimage.binary_fill_holes(ndimage.binary_closing(occ, iterations=1))
    D = ndimage.distance_transform_edt(occ)
    thick = np.zeros(dims)
    for r in range(1, int(D.max()) + 1):
        core = D >= r
        if not core.any():
            break
        thick[(ndimage.distance_transform_edt(~core) <= r) & occ] = 2.0 * r * pitch
    h = thick[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    thin = (h > 0) & (h < 2.0 * cell)
    li = np.flatnonzero(lay)
    lt = cKDTree(x[li])
    conc = np.zeros(len(x))
    for a in range(0, len(li), 20000):
        idx = li[a:a + 20000]
        nbl = lt.query_ball_point(x[idx], 4.0 * pitch)
        for t, (i, js) in enumerate(zip(idx, nbl)):
            js = li[np.asarray(js, int)]
            js = js[js != i]
            if len(js) < 4:
                continue
            v = x[js] - x[i]
            conc[i] = float(np.mean((v @ nrm[i]) / np.maximum(np.linalg.norm(v, axis=1), 1e-12)))
    crevice = lay & (conc > 0.15)
    return dict(layer=lay, rim=rim, rim3=rim3, thin=thin, crevice=crevice, bulk=dep > 2.0), nrm, conc


def normals(x):
    tree = cKDTree(x)
    nb = tree.query(x, 25)[1][:, 1:]
    off = x - x[nb].mean(1)
    return off / np.maximum(np.linalg.norm(off, axis=1, keepdims=True), 1e-12)


def bspline(x):
    """quadratic B-spline weights of the MPM grid: (N, 27) node indices and weights"""
    q = (x - GMIN) / DX
    base = np.floor(q - 0.5).astype(np.int64)
    fx = q - base
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
    idx, wt = [], []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                node = base + np.array([a, b, c])
                node = np.clip(node, 0, NX - 1)
                idx.append((node[:, 0] * NX + node[:, 1]) * NX + node[:, 2])
                wt.append(w[a][:, 0] * w[b][:, 1] * w[c][:, 2])
    return np.stack(idx, 1), np.stack(wt, 1)


def lowpass(g, idx, wt):
    m = np.bincount(idx.ravel(), wt.ravel(), minlength=NX ** 3)
    out = np.zeros_like(g)
    for d in range(3):
        G = np.bincount(idx.ravel(), (wt * g[:, d:d + 1]).ravel(), minlength=NX ** 3) / np.maximum(m, 1e-30)
        out[:, d] = (wt * G[idx]).sum(1)
    return out


def cosv(a, b):
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def per_window(k):
    fc = f"{RUN}_cf/cf_{k:04d}.npz"
    fd = f"{RUN}_dump/win_{k:04d}.npz"
    if not (os.path.exists(fc) and os.path.exists(fd)):
        return k, None
    c = np.load(fc, allow_pickle=True)
    if not len(c["x_render"]) or not len(c["x_phys"]):
        return k, None
    x0 = c["x0"].astype(np.float64)
    xr, xp = c["x_render"].astype(np.float64), c["x_phys"].astype(np.float64)
    rr, rp = ast.literal_eval(str(c["render"])), ast.literal_eval(str(c["cf"]))
    z = np.load(fd)
    lay0 = z["layer_mask"] > 0.5
    d = (xr - xp) / PITCH
    n0 = normals(x0)
    dn = (d * n0).sum(1)
    row = dict(k=k, lam=float(z["lam_r"]), g_share=float(z["g_share"]),
               d_sil_r=rr.get("d_sil"), d_sil_p=rp.get("d_sil"), d_pbr_r=rr.get("d_pbr"), d_pbr_p=rp.get("d_pbr"),
               d_vol_r=rr.get("d_vol"), d_vol_p=rp.get("d_vol"), kin_r=rr.get("kin"), kin_p=rp.get("kin"),
               iters_r=rr.get("iters"), iters_p=rp.get("iters"), commit_r=rr.get("commit"), commit_p=rp.get("commit"))
    an = np.linalg.norm(d, axis=1)
    for nm, msk in REG.items():
        row[f"dmed_{nm}"] = float(np.median(an[msk]))
        row[f"drms_{nm}"] = float(np.sqrt((an[msk] ** 2).mean()))
        row[f"dn_rms_{nm}"] = float(np.sqrt((dn[msk] ** 2).mean()))
        row[f"dn_mean_{nm}"] = float(dn[msk].mean())
        row[f"d2share_{nm}"] = float((an[msk] ** 2).sum() / max((an ** 2).sum(), 1e-300))
    row["dmed_layer0"] = float(np.median(an[lay0]))
    dr, dp = IND_TREE.query(xr, 1)[0] / PITCH, IND_TREE.query(xp, 1)[0] / PITCH
    for nm in ("layer", "rim", "rim3", "thin", "crevice"):
        row[f"closer_{nm}"] = float(np.mean((dp - dr)[REG[nm]]))
    # H4 reach
    gxr = (z["gx_sil"] + z["gx_pbr"]).astype(np.float64)
    gxp = z["gx_phys"].astype(np.float64)
    dxr = (z["xT_rend"] - z["xT_base"]).astype(np.float64)
    dxp = (z["xT_phys"] - z["xT_base"]).astype(np.float64)
    dxru = (z["xT_rend_u"] - z["xT_base_u0"]).astype(np.float64)
    dxpu = (z["xT_phys_u"] - z["xT_base_u0"]).astype(np.float64)
    xT = z["xT0"].astype(np.float64)
    idx, wt = bspline(xT)
    row.update(reach_r_dfc=cosv(-gxr, dxr), reach_p_dfc=cosv(-gxp, dxp), reach_r_u=cosv(-gxr, dxru),
               reach_p_u=cosv(-gxp, dxpu))
    lr, lp = lowpass(gxr, idx, wt), lowpass(gxp, idx, wt)
    row.update(low_r=float((gxr * lr).sum() / max((gxr * gxr).sum(), 1e-300)),
               low_p=float((gxp * lp).sum() / max((gxp * gxp).sum(), 1e-300)),
               reach_r_dfc_low=cosv(-lr, dxr), reach_r_dfc_high=cosv(-(gxr - lr), dxr))
    g2 = (gxr ** 2).sum(1)
    for nm in ("layer", "rim", "rim3", "thin", "crevice", "bulk"):
        row[f"gx_r_share_{nm}"] = float(g2[REG[nm]].sum() / max(g2.sum(), 1e-300))
    row["dxr_layer_med"] = float(np.median(np.linalg.norm(dxr, axis=1)[lay0]) / PITCH)
    row["dxru_layer_med"] = float(np.median(np.linalg.norm(dxru, axis=1)[lay0]) / PITCH)
    # cap
    for nm, key in (("r", "dfc_render_mean"), ("p", "dfc_phys_mean")):
        f = np.linalg.norm(c[key].astype(np.float64), axis=1)
        row[f"cap_{nm}_all"] = float((f >= 0.95 * CLIP).mean())
        row[f"cap_{nm}_layer"] = float((f[lay0] >= 0.95 * CLIP).mean())
        row[f"dfc_{nm}_layer_med"] = float(np.median(f[lay0]))
    # H5 from the dump (dFc, per-particle psum norms: |psum - psum_pc| / |psum|, a proxy) and u
    gr, grpc = z["gl_rend_psum"].astype(np.float64), z["gl_rend_pc_psum"].astype(np.float64)
    ur, urpc = z["gu_rend"].astype(np.float64), z["gu_rend_pc"].astype(np.float64)
    row["pc_changed_dfc_proxy"] = float(np.linalg.norm(gr - grpc) / max(np.linalg.norm(gr), 1e-300))
    row["pc_changed_u"] = float(np.linalg.norm(ur - urpc) / max(np.linalg.norm(ur), 1e-300))
    return k, dict(row=row, d=d.astype(np.float32), x0=x0.astype(np.float32), xr=xr.astype(np.float32),
                   xp=xp.astype(np.float32), n0=n0.astype(np.float32))


# ---- setup: pitch, end states, regions ----
zl_end = np.load(f"{G}/d145t_b1/{mesh}300k_L97_end1.npz")
tgt = np.asarray(zl_end["tgt"], np.float64)
PITCH = 0.708 * float(np.median(cKDTree(tgt).query(tgt[::50], 9)[0][:, 8]))
CELL = DX
ind = np.asarray(np.load(f"{G}/d90/{mesh}_ind300k.npz")["tgt"], np.float64)
IND_TREE = cKDTree(ind)
zf = np.load(RUN + "_frames12.npz")
raws = np.asarray(zf["raws"]).astype(int)
fr = zf["frames"]
X_end = np.asarray(fr[-1], np.float64)
raw_end = int(raws[-1])
del fr, zf
Z_end = np.asarray(np.load(f"{G}/d145t_b1/{mesh}300k_P97_end1.npz")["frames"][-1], np.float64)
R_end = np.asarray(np.load(f"{G}/d145t_b1/{mesh}300k_L97_end1.npz")["frames"][-1], np.float64)
print(f"{mesh}: pitch {PITCH:.5f} wu, cell {CELL:.4f} wu ({CELL / PITCH:.2f} pitches), RG97 delivered raw {raw_end}",
      flush=True)
REG, N_END, CONC = regions_end(X_end, PITCH, CELL)
print("end regions:", {k: int(v.sum()) for k, v in REG.items()}, flush=True)
np.savez_compressed(os.path.join(out_dir, f"{mesh}_regions_end.npz"), **REG, conc=CONC.astype(np.float32))

ks = sorted(int(os.path.basename(f)[3:7]) for f in glob.glob(f"{RUN}_cf/cf_*.npz"))
rows, D, X0, XR, XP, N0 = {}, {}, {}, {}, {}, {}
with Pool(workers) as pool:
    for k, res in pool.imap_unordered(per_window, ks, chunksize=1):
        if res is None:
            continue
        rows[k] = res["row"]
        D[k], X0[k], XR[k], XP[k], N0[k] = res["d"], res["x0"], res["xr"], res["xp"], res["n0"]
        print(f"window {k}: |d| med layer {rows[k]['dmed_layer']:.4f} rim {rows[k]['dmed_rim']:.4f} thin "
              f"{rows[k]['dmed_thin']:.4f} crevice {rows[k]['dmed_crevice']:.4f} bulk {rows[k]['dmed_bulk']:.4f}; reach "
              f"dFc r/p {rows[k]['reach_r_dfc']:.3f}/{rows[k]['reach_p_dfc']:.3f} u r/p {rows[k]['reach_r_u']:.3f}/"
              f"{rows[k]['reach_p_u']:.3f}; low r/p {rows[k]['low_r']:.3f}/{rows[k]['low_p']:.3f}", flush=True)
K = sorted(rows)
fe = {h["animation"]: h.get("frame_end") for h in hist}
# the committed chain: window k's end is window k+1's start, and the window lies in the delivered run (best_truncate)
link = {k: (k + 1 in X0 and float(np.abs(X0[k + 1] - XR[k]).max()) == 0.0) for k in K}
chain = {k: (link[k] and fe.get(k) is not None and fe[k] <= raw_end + 1) for k in K}
last_k = max((k for k in K if fe.get(k) == raw_end + 1), default=None)
if last_k is not None:                                    # the delivered window itself: its end is the delivered state
    chain[last_k] = float(np.abs(XR[last_k] - X_end).max()) == 0.0 or chain[last_k]
print(f"chained windows {sum(chain.values())} of {len(K)}; delivered window {last_k}, "
      f"its end = the delivered state: {last_k is not None and float(np.abs(XR[last_k] - X_end).max()) == 0.0}",
      flush=True)
lay = REG["layer"]
for k in K:
    r = rows[k]
    r["chained"], r["linked"], r["frame_end"] = chain[k], link[k], fe.get(k)
    d = D[k][lay].astype(np.float64)
    dd = float((d * d).sum())
    if link[k]:
        m = (XP[k + 1][lay] - X0[k + 1][lay]).astype(np.float64) / PITCH
        a = (XR[k + 1][lay] - X0[k + 1][lay]).astype(np.float64) / PITCH
        r["undo_phys_next"] = -float((m * d).sum()) / max(dd, 1e-300)
        r["undo_arm_next"] = -float((a * d).sum()) / max(dd, 1e-300)
        big = np.linalg.norm(d, axis=1) >= np.median(np.linalg.norm(d, axis=1))
        r["undo_phys_next_top50"] = -float((m[big] * d[big]).sum()) / max(float((d[big] * d[big]).sum()), 1e-300)
    for j in (1, 2, 5, 10, 20):
        if k + j in XR and all(link.get(k + i, False) for i in range(j)):
            mv = (XR[k + j][lay] - XR[k][lay]).astype(np.float64) / PITCH       # X_{k+j} - X_k along the arm's path
            r[f"c_{j}"] = float((mv * d).sum()) / max(dd, 1e-300)
    # scalar bookkeeping: drift of a physics-only window from the render-improved start
    if link.get(k - 1) and rows[k - 1].get("d_sil_r") is not None and r.get("d_sil_p") is not None:
        r["sil_drift_phys"] = r["d_sil_p"] - rows[k - 1]["d_sil_r"]
    if r.get("d_sil_p") is not None and r.get("d_sil_r") is not None:
        r["sil_gain"] = r["d_sil_p"] - r["d_sil_r"]
# time of each window (simulated s) from the records
for k in K:
    rows[k]["t"] = (fe.get(k) or 0) / 240.0 if fe.get(k) else None
# ---- run sums against the final arm-vs-twin difference ----
F = (X_end - Z_end) / PITCH
Rr = (X_end - R_end) / PITCH
n_end = N_END


def surface_offset(x, nrm, other):
    """signed offset of x along its outward normal from the other end state's outer layer (mean over the other's 8
    nearest layer particles of (x - y) . n, pitches): the shape difference, blind to tangential sliding"""
    tr = cKDTree(other)
    sp = float(np.median(tr.query(other[::15], 9)[0][:, 8]))
    nb = tr.query(other, 33)[1][:, 1:]
    lay_o = np.linalg.norm(other - other[nb].mean(1), axis=1) >= 0.5 * sp
    yo = other[lay_o]
    idx = cKDTree(yo).query(x, 8)[1]
    return (((x[:, None, :] - yo[idx]) * nrm[:, None, :]).sum(2)).mean(1) / PITCH


LI = np.flatnonzero(REG["layer"])
H_twin = np.full(len(F), np.nan)
H_rep = np.full(len(F), np.nan)
H_ind_arm = np.full(len(F), np.nan)
H_twin[LI] = surface_offset(X_end[LI], n_end[LI], Z_end)
H_rep[LI] = surface_offset(X_end[LI], n_end[LI], R_end)
H_ind_arm[LI] = surface_offset(X_end[LI], n_end[LI], ind)
print(f"surface offsets on the layer (pitches): arm vs twin rms {np.sqrt(np.nanmean(H_twin ** 2)):.4f} mean "
      f"{np.nanmean(H_twin):+.4f}; arm vs its replicate rms {np.sqrt(np.nanmean(H_rep ** 2)):.4f}; arm vs the "
      f"independent sample rms {np.sqrt(np.nanmean(H_ind_arm ** 2)):.4f}", flush=True)
np.savez_compressed(os.path.join(out_dir, f"{mesh}_end_offsets.npz"), H_twin=H_twin.astype(np.float32),
                    H_rep=H_rep.astype(np.float32), H_ind_arm=H_ind_arm.astype(np.float32))


def sums(k0):
    S = np.zeros_like(F)
    Gs = np.zeros(len(F))
    Sn = np.zeros(len(F))
    Gn = np.zeros(len(F))
    used = 0
    for k in K:
        if k < k0 or not chain[k]:
            continue
        d = D[k].astype(np.float64)
        S += d
        Gs += np.linalg.norm(d, axis=1)
        dn = (d * N0[k].astype(np.float64)).sum(1)
        Sn += dn
        Gn += np.abs(dn)
        used += 1
    out = {"k0": k0, "windows": used}
    SN_SAVED[k0] = Sn.astype(np.float32)
    Fn, Rn = (F * n_end).sum(1), (Rr * n_end).sum(1)
    for nm, msk in REG.items():
        s, f, rr_ = S[msk], F[msk], Rr[msk]
        out[nm] = dict(
            G_mean=float(Gs[msk].mean()), S_rms=float(np.sqrt((s * s).sum(1).mean())),
            F_rms=float(np.sqrt((f * f).sum(1).mean())), R_rms=float(np.sqrt((rr_ * rr_).sum(1).mean())),
            beta=float((f * s).sum() / max((s * s).sum(), 1e-300)),
            beta_rep=float((rr_ * s).sum() / max((s * s).sum(), 1e-300)),
            corr_FS=cosv(f - f.mean(0), s - s.mean(0)),
            Gn_mean=float(Gn[msk].mean()), Sn_rms=float(np.sqrt((Sn[msk] ** 2).mean())),
            Fn_rms=float(np.sqrt((Fn[msk] ** 2).mean())), Rn_rms=float(np.sqrt((Rn[msk] ** 2).mean())),
            Fn_mean=float(Fn[msk].mean()), Sn_mean=float(Sn[msk].mean()),
            beta_n=float((Fn[msk] * Sn[msk]).sum() / max((Sn[msk] ** 2).sum(), 1e-300)),
            beta_n_rep=float((Rn[msk] * Sn[msk]).sum() / max((Sn[msk] ** 2).sum(), 1e-300)),
            corr_n=float(np.corrcoef(Fn[msk], Sn[msk])[0, 1]))
        lm = msk & np.isfinite(H_twin)
        if lm.sum() > 10:
            ht, hr, sn = H_twin[lm], H_rep[lm], Sn[lm]
            out[nm].update(H_twin_rms=float(np.sqrt((ht ** 2).mean())), H_rep_rms=float(np.sqrt((hr ** 2).mean())),
                           H_twin_mean=float(ht.mean()), beta_shape=float((ht * sn).sum() / max((sn ** 2).sum(), 1e-300)),
                           beta_shape_rep=float((hr * sn).sum() / max((sn ** 2).sum(), 1e-300)),
                           corr_shape=float(np.corrcoef(ht, sn)[0, 1]), corr_shape_rep=float(np.corrcoef(hr, sn)[0, 1]))
    return out


SN_SAVED = {}
t_of = {k: rows[k]["t"] for k in K}
k_late = [k for k in K if (t_of[k] or 0) >= 3.0]
summary = {"mesh": mesh, "pitch": PITCH, "cell": CELL, "windows": len(K), "chained": int(sum(chain.values())),
           "regions_end": {k: int(v.sum()) for k, v in REG.items()}, "sums_all": sums(0),
           "sums_k10": sums(10), "sums_late3s": sums(k_late[0] if k_late else 10**9),
           "k_late3s": k_late[0] if k_late else None}
# the twin's per-window change of d_sil at matched simulated time (B1's P97 records)
jp = json.load(open(f"{G}/d145t_b1/{mesh}300k_P97.json"))
hp = [h for h in next(iter(jp["arms"].values()))["history"] if h.get("d_sil") is not None and h.get("frame_end")]
tp = np.array([h["frame_end"] / 240.0 for h in hp])
sp_ = np.array([h["d_sil"] for h in hp])
summary["twin_sil_t"] = tp.tolist()
summary["twin_sil"] = sp_.tolist()
json.dump({"summary": summary, "rows": [rows[k] for k in K]}, open(os.path.join(out_dir, f"{mesh}_h345.json"), "w"),
          indent=1, default=float)
np.savez_compressed(os.path.join(out_dir, f"{mesh}_Sn.npz"), **{f"Sn_k{k0}": v for k0, v in SN_SAVED.items()})
print(json.dumps(summary, default=float)[:4000], flush=True)
