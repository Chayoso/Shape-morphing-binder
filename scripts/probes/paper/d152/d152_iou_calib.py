"""d152_iou_calib.py OUT_JSON MESH:SEED [...] -- D152 (H1): what the gallery's silhouette IoU can see. numpy copy of
physmorph.metrics.sil_iou (the tag's: a BINARY 3 x 3-pixel point splat at 128 px, 24 views = 8 azimuths x 3 elevations as
make_views, the extent 1.15 x the reference's largest radius), checked against endgen's values, then:
  res        the same IoU at 128 / 256 / 512 px (3 x 3 footprint), L, P and the floor (own sample): does the arm's share
             of the twin's excess over the floor grow when the pixel shrinks below the render's displacement?
  shift      the arm's end with its outer layer moved along the outward normal by delta pitches (uniform): 1 - IoU per
             pitch, the metric's sensitivity to a normal displacement of the surface
  body       the IoU of the largest connected set only (radius 2 x the median nearest-neighbour distance): how much of
             the error the particles off the body make
  pixels     at 128 px: the twin's error pixels (P xor ind), the arm's (L xor ind), the pixels where L and P differ
CPU only, read-only. Measurement only."""
import json
import sys

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"


def make_views(n_azim, elevs=(0.0, 0.5, -0.5)):
    views, n_r = [], len(elevs)
    for i, phi in enumerate(elevs):
        off = (i / n_r) * (2 * np.pi / n_azim)
        for t in np.linspace(0, 2 * np.pi, n_azim, endpoint=False):
            views.append((float(t + off), float(phi)))
    return views


VIEWS = make_views(8)


def splat(x, res, th, phi, extent):
    right = np.array([np.cos(th), 0.0, -np.sin(th)])
    up = np.array([-np.sin(phi) * np.sin(th), np.cos(phi), -np.sin(phi) * np.cos(th)])
    p = np.stack([x @ right, x @ up], 1)
    ij = np.floor((p + extent) / (2 * extent) * res).astype(np.int64)
    ij = ij[((ij >= 0) & (ij < res)).all(1)]
    img = np.zeros((res, res), bool)
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            img[np.clip(ij[:, 0] + ox, 0, res - 1), np.clip(ij[:, 1] + oy, 0, res - 1)] = True
    return img


def iou(x, ref, res=128, extent=None, masks=False):
    e = 1.15 * float(np.linalg.norm(ref, axis=1).max()) if extent is None else extent
    out, ms = [], []
    for th, phi in VIEWS:
        a, b = splat(x, res, th, phi, e), splat(ref, res, th, phi, e)
        u = int((a | b).sum())
        out.append(int((a & b).sum()) / u if u else 1.0)
        if masks:
            ms.append((a, b))
    return (float(np.mean(out)), ms) if masks else float(np.mean(out))


def layer_normals(x):
    tr = cKDTree(x)
    sp = float(np.median(tr.query(x[::15], 9)[0][:, 8]))
    nb = tr.query(x, 33)[1][:, 1:]
    off = x - x[nb].mean(1)
    lay = np.linalg.norm(off, axis=1) >= 0.5 * sp
    return lay, off / np.maximum(np.linalg.norm(off, axis=1, keepdims=True), 1e-12)


def main_body(x):
    tr = cKDTree(x)
    nn = float(np.median(tr.query(x[::20], 2)[0][:, 1]))
    A = tr.sparse_distance_matrix(tr, 2.0 * nn, output_type="coo_matrix")
    _, lab = connected_components(A, directed=False)
    big = np.bincount(lab).argmax()
    return lab == big


res_all = {}
for spec in sys.argv[2:]:
    mesh, seed = spec.split(":")
    zl = np.load(f"{G}/d145t_b1/{mesh}300k_L{seed}_end1.npz")
    zp = np.load(f"{G}/d145t_b1/{mesh}300k_P{seed}_end1.npz")
    own = np.asarray(zl["tgt"], np.float64)
    ind = np.asarray(np.load(f"{G}/d90/{mesh}_ind300k.npz")["tgt"], np.float64)
    L = np.asarray(zl["frames"][-1], np.float64)
    P = np.asarray(zp["frames"][-1], np.float64)
    tgt_sp = float(np.median(cKDTree(own).query(own[::50], 9)[0][:, 8]))
    pitch = 0.708 * tgt_sp
    ext = 1.15 * float(np.linalg.norm(ind, axis=1).max())
    r = {"mesh": mesh, "seed": int(seed), "pitch": pitch, "extent": ext}
    for res in (128, 256, 512):
        eL, eP, eF = 1 - iou(L, ind, res), 1 - iou(P, ind, res), 1 - iou(own, ind, res)
        r[f"res{res}"] = {"pixel_pitches": 2 * ext / res / pitch, "err_L": eL, "err_P": eP, "err_floor": eF,
                          "rel": (eP - eL) / eP, "gap_share": (eP - eL) / (eP - eF) if eP != eF else None}
        print(mesh, seed, res, r[f"res{res}"], flush=True)
    lay, nrm = layer_normals(L)
    base = 1 - iou(L, ind)
    r["shift"] = {}
    for dlt in (-0.3, -0.1, -0.03, 0.03, 0.1, 0.3):
        xs = L.copy()
        xs[lay] += dlt * pitch * nrm[lay]
        r["shift"][str(dlt)] = (1 - iou(xs, ind)) - base
    print(mesh, seed, "shift", r["shift"], flush=True)
    for nm, x in (("L", L), ("P", P), ("own", own)):
        m = main_body(x)
        r[f"body_{nm}"] = {"off_body": int((~m).sum()), "err_all": 1 - iou(x, ind), "err_body": 1 - iou(x[m], ind)}
    print(mesh, seed, "body", {k: r[k] for k in r if k.startswith("body_")}, flush=True)
    _, mL = iou(L, ind, masks=True)
    _, mP = iou(P, ind, masks=True)
    xl = sum(int((a ^ b).sum()) for a, b in mL)
    xp = sum(int((a ^ b).sum()) for a, b in mP)
    lp = fixed = broke = 0
    for (Lm, Im), (Pm, _) in zip(mL, mP):
        lp += int((Lm ^ Pm).sum())
        fixed += int(((Pm ^ Im) & ~(Lm ^ Im)).sum())       # the twin wrong, the arm right
        broke += int(((Lm ^ Im) & ~(Pm ^ Im)).sum())       # the arm wrong, the twin right
    union = sum(int((a | b).sum()) for a, b in mP)
    r["pixels"] = {"xor_L_ind": xl, "xor_P_ind": xp, "xor_L_P": lp, "P_wrong_L_right": fixed,
                   "L_wrong_P_right": broke, "union_P_ind": union}
    print(mesh, seed, "pixels", r["pixels"], flush=True)
    res_all[spec] = r
json.dump(res_all, open(sys.argv[1], "w"), indent=1)
