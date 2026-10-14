"""d148_wl_forensic.py -- D148c (walls) forensic on the CPU (2026-10-14, the coordinator's list).

  beads FRAMES12 TAG K:X0,Y0,X1,Y1[:full] [...]   WL's beads on the side render (azimuth 90)
  nodes FRAMES12 TAG W0 W1                          WL's own gap nodes per window (finite-difference velocities)
  dump  DUMP_DIR TAG W0 W1 FRAMES12_FOR_TGT         the same nodes on D135's state (FB's dump: v, the terms' gradients)

The run's geometry: the MPM grid 44^3 at dx (the 300k dragon), the walls' gap nodes and pieces (GapPlanes margin 1 on
the run's float32 loss cell, 560 nodes as the run), each particle's side at a node by the node's piece holding its
representative cell, else the node's plane (side_share's rule); the representative cells persistent over window starts
(gap_walls.rep_cells) from the window-start transport images: beads / nodes recompute them on the CPU at the kept frame
nearest each window start (grid_transport_displacement on the gate grid = the MPM-cell grid, blur one cell, as
objective.py); dump reads them from the dump (x + T_tgt - T_self: the same images, the dump's own solve). Node fields:
per side the cubic B-spline weighted mass, momentum (PIC: the particles' own velocities), the node volume (weights x the
run's volume per particle), split where both sides hold mass and the volume is below one cell (B's gate), approaching
where (v0 - v1) . n > 0 (n from side 0 to side 1)."""
import json
import os
import sys

import physmorph  # noqa: F401  (before torch)
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from physmorph.losses.grid_ot import grid_transport_displacement
from physmorph.losses.volumetric import target_mass_grid
from physmorph.mpm.gap_split import GapPlanes
from physmorph.mpm.gap_walls import BoxPieces, rep_cells

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
DX, NG, LRES = 0.2948594285057975, 44, 87
VP = 44.7107 / 300000                       # the run's gap_vp: the target's fill volume / N (dragon300k_W*.log)
GMIN = np.full(3, -DX * NG / 2.0)
if os.environ.get("D148_JSON"):             # another run's grid (the bunny): its provenance and arm config
    _a = json.load(open(os.environ["D148_JSON"]))
    _c = next(iter(_a["arms"].values()))["config"]
    DX, NG, LRES = float(_a["provenance"]["mpm"]["dx"]), int(_a["provenance"]["mpm"]["nx"]), int(_c["loss_res"])
    VP, GMIN = float(_c["gap_vp"]), np.asarray(_a["provenance"]["mpm"]["grid_min"], np.float64)
NEAREST = os.environ.get("D148_NEAREST", "0") == "1"   # D148d: the nearest piece outside the box
D0 = GMIN.astype(np.float32)
CELL = float((D0 + DX * np.full(3, NG, np.float32) - D0).max() / LRES)     # the run's loss cell (float32, target.py)
GDX = float((D0 + DX * np.full(3, NG, np.float32) - D0).max() / NG)        # the gate grid's cell (the MPM cell)
OFFS = np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 3)
T_ = os.path.dirname(os.path.abspath(__file__))


def bspline(r):
    r = np.abs(r)
    return np.where(r < 1, 2 / 3 - r ** 2 + r ** 3 / 2, np.where(r < 2, (2 - r) ** 3 / 6, 0.0))


class Geo:
    def __init__(self, tgt):
        self.gp = GapPlanes(tgt, DX, GMIN, (NG,) * 3, CELL, keep_boxes=True, margin=1)
        self.bp = BoxPieces(self.gp, nearest_pieces=NEAREST)
        self.pm, self.pn = self.gp.m.reshape(-1, 3), self.gp.n.reshape(-1, 3)
        self.nodes = np.nonzero(np.abs(self.pn).sum(1) > 0)[0]
        self.is_gap = np.zeros(NG ** 3, bool)
        self.is_gap[self.nodes] = True
        self.xg = GMIN + DX * np.stack(np.unravel_index(np.arange(NG ** 3), (NG,) * 3), 1)
        tt = torch.as_tensor(tgt, dtype=torch.float32)
        self.tgrid = target_mass_grid(tt, torch.ones(len(tt)), torch.as_tensor(D0), GDX, (NG,) * 3)
        bp = self.bp
        self.wl = dict(lo=bp.lo, cell=bp.cell, nc=bp.nc, occ=torch.as_tensor(bp.occ), nearest=torch.as_tensor(bp.nearest))
        print(f"[geo] {len(self.nodes)} walls gap nodes ({self.gp.n_three} with three or more pieces); loss cell {CELL:.9f}",
              flush=True)

    def images(self, x):
        xt = torch.as_tensor(x, dtype=torch.float32)
        disp = grid_transport_displacement(xt, torch.ones(len(xt)), self.tgrid, torch.as_tensor(D0), GDX, (NG,) * 3,
                                           eps=GDX ** 2, iters=1600, tol=0.01)
        return (xt + disp).numpy()

    def rep(self, q, prev):
        r = rep_cells(torch.as_tensor(q, dtype=torch.float32), torch.as_tensor(prev), self.wl["lo"], self.wl["cell"],
                      self.wl["nc"], self.wl["occ"], self.wl["nearest"])
        return r.numpy()

    def sides(self, g, q, rep):
        """sides (0/1) at node g of the particles with images q and representative cells rep; and by-pieces flags."""
        s = (((q - self.pm[g]) @ self.pn[g]) > 0).astype(np.int8)
        by = np.zeros(len(q), bool)
        off, ca, cb = self.bp.boxi[g]
        if off >= 0:
            ijk = np.stack(np.unravel_index(rep.astype(np.int64), tuple(self.bp.nc)), 1) - self.bp.box0[g]
            d = self.bp.boxd[g]
            if NEAREST:
                ijk = np.minimum(np.maximum(ijk, 0), d - 1)
            ins = ((ijk >= 0) & (ijk < d)).all(1)
            lab = np.full(len(q), -9, np.int64)
            li = ijk[ins]
            lab[ins] = self.bp.labels[off + (li[:, 0] * d[1] + li[:, 1]) * d[2] + li[:, 2]]
            s[lab == cb] = 1
            s[lab == ca] = 0
            by = (lab == ca) | (lab == cb)
        return s, by

    def piece_of(self, g, cells):
        """the label of node g's box piece holding each cell (-9 outside the box, 0 = empty)."""
        off = self.bp.boxi[g][0]
        ijk = np.stack(np.unravel_index(cells.astype(np.int64), tuple(self.bp.nc)), 1) - self.bp.box0[g]
        d = self.bp.boxd[g]
        if NEAREST:
            ijk = np.minimum(np.maximum(ijk, 0), d - 1)
        ins = ((ijk >= 0) & (ijk < d)).all(1)
        lab = np.full(len(cells), -9, np.int64)
        li = ijk[ins]
        lab[ins] = self.bp.labels[off + (li[:, 0] * d[1] + li[:, 1]) * d[2] + li[:, 2]]
        return lab

    def cell_of(self, x):
        nc = np.asarray(self.bp.nc)
        ci = np.minimum(np.maximum(np.floor((x - np.asarray(self.bp.lo)) / self.bp.cell).astype(np.int64), 0), nc - 1)
        return (ci[:, 0] * nc[1] + ci[:, 1]) * nc[2] + ci[:, 2]


def stencil(x):
    b = np.floor((x - GMIN) / DX).astype(int) - 1
    n = b[:, None, :] + OFFS[None]
    ok = ((n >= 0) & (n < NG)).all(2)
    lin = (n[..., 0] * NG + n[..., 1]) * NG + n[..., 2]
    return np.where(ok, lin, -1)


def weights(x, g):
    return bspline((x - GEO.xg[g]) / DX).prod(1)


def node_fields(g, x, v, q, rep, tree, grads=None):
    """node g's two fields from the particles in its support."""
    idx = np.asarray(tree.query_ball_point(GEO.xg[g], 2.0 * DX, p=np.inf), np.int64)
    if not len(idx):
        return None
    w = weights(x[idx], g)
    keep = w > 0
    idx, w = idx[keep], w[keep]
    s, by = GEO.sides(g, q[idx], rep[idx])
    out = dict(idx=idx, w=w, s=s, by=by, vol=float(w.sum() * VP))
    for k in (0, 1):
        sel = s == k
        out[f"m{k}"] = float(w[sel].sum())
        out[f"n{k}"] = int(sel.sum())
        out[f"v{k}"] = (w[sel, None] * v[idx[sel]]).sum(0) / max(out[f"m{k}"], 1e-30)
        if grads is not None:
            out[f"pull{k}"] = float((w[sel] * (-(grads[idx[sel]] @ GEO.pn[g]))).sum() / max(out[f"m{k}"], 1e-30))
    out["split"] = out["m0"] > 1e-12 and out["m1"] > 1e-12 and out["vol"] < DX ** 3
    out["vrel_n"] = float((out["v0"] - out["v1"]) @ GEO.pn[g])
    return out


def apart_of(x, a):
    pairs = cKDTree(x).query_pairs(1.5 * a, output_type="ndarray")
    gr = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(len(x), len(x)))
    _, cl = connected_components(gr, directed=False)
    return cl != np.bincount(cl).argmax()


def window_frames(raws, wmax):
    return [int(np.argmin(np.abs(raws - 40 * w))) for w in range(wmax + 1)]


def qrep_series(fr, raws, wmax, cache):
    """q and rep at the kept frame nearest every window start 0..wmax (cached)."""
    if os.path.exists(cache):
        z = np.load(cache)
        if int(z["wmax"]) >= wmax:
            return z["q"], z["rep"], z["ks"]
    ks = window_frames(raws, wmax)
    Q, R = [], []
    prev = np.full(fr.shape[1], -1, np.int64)
    for w, k in enumerate(ks):
        q = GEO.images(np.asarray(fr[k], np.float32))
        prev = GEO.rep(q, prev)
        Q.append(q.astype(np.float32))
        R.append(prev.copy())
        print(f"  [images] window {w} frame {k} (raw {raws[k]}, {raws[k] / 240:.3f} s)", flush=True)
    np.savez(cache, q=np.stack(Q), rep=np.stack(R), ks=np.asarray(ks), wmax=np.int64(wmax))
    return np.stack(Q), np.stack(R), np.asarray(ks)


def fd_velocity(fr, raws, k):
    k0, k1 = max(k - 1, 0), min(k + 1, len(raws) - 1)
    return (np.asarray(fr[k1], np.float64) - np.asarray(fr[k0], np.float64)) / ((raws[k1] - raws[k0]) / 240.0)


def select(fr, k, box, full):
    import math
    x0b, y0b, x1b, y1b = box
    tgt = np.asarray(Z["tgt"], np.float64)
    center = tgt.mean(0)
    radius = float(np.linalg.norm(tgt - center, axis=1).max())
    W, H, AZ, EL = 3840, 2160, math.radians(90.0), math.radians(18.0)
    cam = center + 3.6 * radius * np.array([math.cos(EL) * math.sin(AZ), math.sin(EL), math.cos(EL) * math.cos(AZ)])
    fwd = (center - cam) / np.linalg.norm(center - cam)
    right = np.cross([0.0, 1.0, 0.0], fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    tan_y = math.tan(math.radians(15.0))
    tan_x = tan_y * W / H
    d = np.asarray(fr[k], np.float64) - cam
    xc, yc, zc = d @ right, d @ up, d @ fwd
    px = ((xc / zc) / tan_x + 1) / 2 * W - 0.5
    py = (1 - (yc / zc) / tan_y) / 2 * H - 0.5
    wq, hq = (0, 0) if full else ((x1b - x0b) / 4.0, (y1b - y0b) / 4.0)
    return np.nonzero((px >= x0b + wq) & (px <= x1b - wq) & (py >= y0b + hq) & (py <= y1b - hq))[0]


def beads(path, tag, specs):
    fr, raws = Z["frames"], np.asarray(Z["raws"])
    tgt = np.asarray(Z["tgt"], np.float64)
    a = 0.708 * float(np.median(cKDTree(tgt).query(tgt[::10], 9)[0][:, 8]))
    parsed = []
    for sp in specs:
        p = sp.split(":")
        parsed.append((int(p[0]), [int(v) for v in p[1].split(",")], len(p) > 2 and p[2] == "full"))
    wmax = max(int(raws[k] // 40) for k, _, _ in parsed) + 1
    Q, R, ks = qrep_series(fr, raws, wmax, f"{T_}/wlf_{tag}_qrep.npz")
    xe = np.asarray(fr[-1], np.float64)
    for K, box, full in parsed:
        sel = select(fr, K, box, full)
        wk = int(raws[K] // 40)
        print(f"\n== {tag} bead at frame {K} ({raws[K] / 240:.2f} s, window {wk}), box {box}{' full' if full else ''}: "
              f"particles {sel.tolist()}", flush=True)
        if not len(sel):
            continue
        for w in range(max(0, wk - 6), wk + 2):
            if w >= len(ks):
                break
            k = int(ks[w])
            x = np.asarray(fr[k], np.float64)
            v = fd_velocity(fr, raws, k)
            q, rep = Q[w].astype(np.float64), R[w]
            tree = cKDTree(x)
            ap = apart_of(x, a)
            line = []
            for p in sel:
                st = stencil(x[p:p + 1])[0]
                gs = [int(g) for g in st if g >= 0 and GEO.is_gap[g]]
                rows = []
                for g in gs:
                    nf = node_fields(g, x, v, q, rep, tree)
                    i = int(np.nonzero(nf["idx"] == p)[0][0]) if (nf is not None and (nf["idx"] == p).any()) else -1
                    if i < 0:
                        continue
                    sp_, by = int(nf["s"][i]), bool(nf["by"][i])
                    same = int(((nf["s"] == sp_) & (nf["idx"] != p)).sum())
                    own, oth = nf[f"v{sp_}"], nf[f"v{1 - sp_}"]
                    lab = int(GEO.piece_of(g, np.asarray([rep[p]]))[0])
                    rows.append((g, sp_, by, same, nf[f"m{sp_}"] / max(nf["m0"] + nf["m1"], 1e-30), nf["split"],
                                 nf["vrel_n"], np.linalg.norm(v[p] - own), np.linalg.norm(v[p] - oth)
                                 if nf[f"m{1 - sp_}"] > 1e-12 else np.nan, lab, nf["vol"] / DX ** 3))
                if not rows:
                    line.append(f"p{p}: no walls node in its stencil (apart {bool(ap[p])})")
                    continue
                r = np.array([[rr[1], rr[2], rr[3], rr[4], rr[5], rr[6], rr[7], rr[8], rr[10]] for rr in rows], float)
                end_cell = GEO.cell_of(xe[p:p + 1])
                end_side = []
                for rr in rows:
                    lb = int(GEO.piece_of(rr[0], end_cell)[0])
                    off, ca, cb = GEO.bp.boxi[rr[0]]
                    end_side.append(1 if lb == cb else (0 if lb == ca else -1))
                es = np.asarray(end_side)
                # one wall across its nodes: the chosen piece (side 1 -> cb, side 0 -> ca) at each node as global
                # occupied cells; a pair of nodes contradicts when one's chosen wall shares a cell with the other's
                # other wall, agrees when the chosen walls share a cell
                walls = {}
                for rr in rows:
                    a0, lab, ca_, cb_ = GEO.gp.boxes[rr[0]]
                    def cells(c):
                        kk = np.stack(np.nonzero(lab == c), -1) + a0
                        return set(np.ravel_multi_index(kk.T, tuple(GEO.bp.nc)).tolist())
                    ch, ot = (cb_, ca_) if rr[1] == 1 else (ca_, cb_)
                    walls[rr[0]] = (cells(ch), cells(ot))
                gl = list(walls)
                agree = contra = flips = 0
                for i1 in range(len(gl)):
                    for i2 in range(i1 + 1, len(gl)):
                        s1, o1 = walls[gl[i1]]
                        s2, o2 = walls[gl[i2]]
                        contra += int(bool(s1 & o2) or bool(o1 & s2))
                        flips += int((bool(s1 & o2) and not s1 & s2) or (bool(o1 & s2) and not s1 & s2))
                        agree += int(bool(s1 & s2))
                endok = es[es >= 0] == r[es >= 0, 0]
                line.append(f"p{p}: node pairs whose chosen walls share cells {agree}, contradicting {contra} (flipped: the "
                            f"chosen wall met only as the other one {flips}); end "
                            f"position's wall = the chosen one at {int(endok.sum())} of {len(endok)} nodes that hold it")
                line.append(
                    f"p{p}: apart {int(ap[p])}; {len(rows)} walls nodes, side 1 at {r[:, 0].mean():.2f}, by pieces "
                    f"{r[:, 1].mean():.2f}; same-side others median {np.median(r[:, 2]):.0f} (alone at "
                    f"{(r[:, 2] == 0).mean():.2f}), own-side mass share {r[:, 3].mean():.2f}; split {r[:, 4].mean():.2f}, "
                    f"(v0-v1).n median {np.median(r[:, 5]):+.3f} (approaching {(r[:, 5] > 0).mean():.2f}); |v - own| "
                    f"{np.median(r[:, 6]):.3f}, |v - other| {np.nanmedian(r[:, 7]) if np.isfinite(r[:, 7]).any() else np.nan:.3f} "
                    f"wu/s, |v| {np.linalg.norm(v[p]):.3f}; node volume {np.median(r[:, 8]):.2f} cells; rep cell in piece "
                    f"{sorted(set(int(rr[9]) for rr in rows))}; end position's side {sorted(set(es.tolist()))} vs label "
                    f"{sorted(set(int(rr[1]) for rr in rows))}; image {np.linalg.norm(q[p] - x[p]) / DX:.2f} cells away")
            print(f"  window {w} ({raws[k] / 240:.2f} s):\n    " + "\n    ".join(line), flush=True)


def nodes_report(tag, wins, getter):
    """per window: the walls nodes' split / separating share and the velocity jump at the split nodes."""
    first = {}
    for w in wins:
        x, v, q, rep, grads = getter(w)
        tree = cKDTree(x)
        rows = []
        for g in GEO.nodes:
            nf = node_fields(int(g), x, v, q, rep, tree, grads)
            if nf is None or nf["m0"] + nf["m1"] < 1e-12:
                continue
            both = nf["m0"] > 1e-12 and nf["m1"] > 1e-12
            rows.append((int(g), both, nf["split"], nf["vrel_n"], np.linalg.norm(nf["v0"] - nf["v1"]) if both else 0.0,
                         nf["vol"] / DX ** 3, min(nf["m0"], nf["m1"]) / (nf["m0"] + nf["m1"]),
                         (nf.get("pull1", 0.0) - nf.get("pull0", 0.0)) if both else 0.0,
                         float(nf["w"][nf["by"]].sum() / max(nf["w"].sum(), 1e-30)), float(nf["by"].all()),
                         float(nf["w"][~nf["by"]].sum()), float(nf["w"].sum())))
        r = np.array([rr[1:] for rr in rows], float)
        both, split = r[:, 0] > 0, r[:, 1] > 0
        sep = split & (r[:, 2] <= 0)
        for rr in rows:
            if rr[2] and rr[3] <= 0 and rr[0] not in first:
                first[rr[0]] = w
        newly = [g for g, w0 in first.items() if w0 == w]
        nrow = r[np.isin([rr[0] for rr in rows], newly)] if newly else np.zeros((0, r.shape[1]))
        msg = (f"  window {w} ({w / 6:.2f} s): nodes with mass {len(rows)}, both sides {int(both.sum())}, split (vol < 1 "
               f"cell) {int(split.sum())}, separating {int(sep.sum())}; at the split nodes (v0-v1).n median "
               f"{np.median(r[split, 2]) if split.any() else np.nan:+.3f}, |v0-v1| median "
               f"{np.median(r[split, 3]) if split.any() else np.nan:.3f} wu/s, volume median "
               f"{np.median(r[split, 4]) if split.any() else np.nan:.2f} cells, minority side share "
               f"{np.median(r[split, 5]) if split.any() else np.nan:.2f}")
        if split.any():
            arr = split & (r[:, 8] > 0)
            msg += (f"; pieces decide {np.median(r[split, 7]):.2f} of a split node's mass (median), every particle at "
                    f"{int(arr.sum())} split nodes ({int((arr & (r[:, 2] <= 0)).sum())} separating)")
            msg += f"; the plane decides {r[split, 9].sum() / max(r[split, 10].sum(), 1e-30):.4f} of the split nodes' mass"
        if grads is not None and split.any():
            msg += (f"; the terms' pull apart (side 1's pull along n minus side 0's) median {np.median(r[split, 6]):+.3e},"
                    f" > 0 at {(r[split, 6] > 0).mean():.2f}")
        if len(nrow):
            msg += (f"\n      first separating here: {len(newly)} nodes, |v0-v1| median {np.median(nrow[:, 3]):.3f} wu/s, "
                    f"(v0-v1).n {np.median(nrow[:, 2]):+.3f}, volume {np.median(nrow[:, 4]):.2f} cells, minority share "
                    f"{np.median(nrow[:, 5]):.2f}")
        print(msg, flush=True)


if __name__ == "__main__":
    mode, src, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    if mode in ("beads", "nodes"):
        Z = np.load(src)
        GEO = Geo(np.asarray(Z["tgt"], np.float64))
        if mode == "beads":
            beads(src, tag, sys.argv[4:])
        else:
            w0, w1 = int(sys.argv[4]), int(sys.argv[5])
            fr, raws = Z["frames"], np.asarray(Z["raws"])
            Q, R, ks = qrep_series(fr, raws, w1, f"{T_}/wlf_{tag}_qrep.npz")

            def get(w):
                k = int(ks[w])
                return np.asarray(fr[k], np.float64), fd_velocity(fr, raws, k), Q[w].astype(np.float64), R[w], None
            print(f"== {tag}: the walls nodes per window (finite-difference velocities at the kept frame nearest each "
                  f"window start)", flush=True)
            nodes_report(tag, range(w0, w1 + 1), get)
    else:
        w0, w1 = int(sys.argv[4]), int(sys.argv[5])
        tgt = np.load(sys.argv[6])["tgt"].astype(np.float64)
        GEO = Geo(tgt)
        reps, prev = {}, np.full(len(tgt), -1, np.int64)
        for w in range(1, w1 + 1):
            z = np.load(f"{src}/d137_{w:04d}.npz")
            q = (z["x"] + z["T_tgt"] - z["T_self"]).astype(np.float64)
            prev = GEO.rep(q, prev)
            reps[w] = (q, prev.copy())

        def get(w):
            z = np.load(f"{src}/d137_{w:04d}.npz")
            G = sum(z["g_" + k].astype(np.float64) * float(z["s_" + k]) for k in ("ot", "surf", "near", "spray", "rend"))
            return z["x"].astype(np.float64), z["v"].astype(np.float64), reps[w][0], reps[w][1], G
        print(f"== {tag}: the walls nodes on D135's state at window ends (the dump's v; the terms' summed gradient at "
              f"the committed end)", flush=True)
        nodes_report(tag, range(w0, w1 + 1), get)
