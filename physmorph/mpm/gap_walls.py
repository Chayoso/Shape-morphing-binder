"""D148c (`--gap_split walls`): a particle's side at a gap node by the node's own pieces, persistent in the gap.

D148's sides were the side of each node's plane the particle's transport image fell on; neighbouring nodes' planes
differ (each its own closest pair's), so a particle near them was on side 1 at one node and on side 0 at the next and
both fields pulled on it (the beads' forensic: their particles on side 1 at 0.51-0.88 of their nodes). The fracture
methods give a particle the same physical side at every node (Nairn's one crack, Homel & Herbold's particle gradient,
CPIC's one tag per particle) and keep it (CPIC's persistence). Here a particle's side at node g is the piece of g's
kernel box (gap_split.GapPlanes' 26-connected occupied cells, joined through the box's one-cell ring so that a curved
wall clipped by the box stays one piece) that holds its representative cell: the occupied cell holding its window-start
transport image, or, while that image lies in an empty cell (the gap: the transport's saddle), the cell it had before
(initially the nearest occupied cell). Neighbouring nodes' boxes overlap and their pieces agree on the cells they
share, so a particle has one physical side at every node that holds its cell. Where the cell lies outside the node's
box or in a third piece, the node's plane test decides (gap_kernels.gap_weight); the walls of one region are not
joined into global labels: a hole's rim or a V's two flanks are one wall globally and two sides locally.
D148d (`--gap_split nearest`): where the representative cell lies outside the node's box, the side is the node's piece
nearest to it (the piece of the box's occupied cell nearest to the box cell closest to it) in place of the plane: a
wall is continuous, so the wall that holds the kept cell continues into the neighbouring box, and the wall across the
gap lies at least the gap farther. The plane stays only where that piece is a third one (D148c's forensic: the
particles in the gap kept a cell behind them, outside the boxes of their far stencil nodes, where the plane decided).
"""
from __future__ import annotations

import numpy as np


class BoxPieces:
    """From a GapPlanes built with keep_boxes (and margin 1): per MPM node box0 (ngrid, 3) int32 (the box's first
    occupancy cell), boxd (ngrid, 3) int32 (its size), boxi (ngrid, 3) int32 (offset into labels, side 0's piece, side
    1's piece; offset -1 off the gap nodes), labels (flat int32, every gap node's box labels in turn), and the occupancy
    for the representative cells: occ (cells,) bool, nearest (cells,) int64 (the nearest occupied cell), nc (3,).
    nearest_pieces (D148d): labels holds, for every box cell, the piece of the box's occupied cell nearest to it (the
    cell's own piece where it is occupied), for the kernels' read at the box cell closest to a representative cell
    outside the box (side_share, clampk = 1)."""

    def __init__(self, gp, nearest_pieces: bool = False):
        from scipy import ndimage
        ng = int(np.prod(gp.dims))
        self.box0 = np.zeros((ng, 3), np.int32)
        self.boxd = np.zeros((ng, 3), np.int32)
        self.boxi = np.full((ng, 3), -1, np.int32)
        chunks, off = [], 0
        for g, (a0, lab, ca, cb) in gp.boxes.items():
            self.box0[g], self.boxd[g], self.boxi[g] = a0, lab.shape, (off, ca, cb)
            if nearest_pieces and (lab == 0).any() and (lab > 0).any():
                ij = ndimage.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
                lab = lab[ij[0], ij[1], ij[2]]
            chunks.append(lab.astype(np.int32).reshape(-1))
            off += lab.size
        self.labels = np.concatenate(chunks) if chunks else np.zeros(1, np.int32)
        self.occ = gp.occ.reshape(-1)
        idx = ndimage.distance_transform_edt(~gp.occ, return_distances=False, return_indices=True)
        self.nearest = np.ravel_multi_index(tuple(idx.reshape(3, -1)), gp.nc).astype(np.int64)
        self.nc, self.lo, self.cell = np.asarray(gp.nc, np.int64), gp.lo, gp.cell
        self.n_nodes = len(gp.boxes)


def rep_cells(q, prev, lo, cell, nc, occ, nearest):
    """Each particle's representative cell (torch): the occupied cell holding its image q; while q lies in an empty
    cell, its previous one (prev >= 0), else the nearest occupied cell to q's."""
    import torch
    lo_t = torch.as_tensor(np.asarray(lo, np.float32), device=q.device)
    nct = torch.as_tensor(np.asarray(nc), device=q.device)
    ci = torch.minimum(torch.clamp(torch.floor((q - lo_t) / cell).long(), min=0), nct - 1)
    lin = (ci[:, 0] * nct[1] + ci[:, 1]) * nct[2] + ci[:, 2]
    keep = torch.where(prev >= 0, prev, nearest[lin])
    return torch.where(occ[lin], lin, keep)


def side_by_pieces(bp, rep, g, clamp: bool = False):
    """numpy reference of the kernels' rule (tests, readers): 0 / 1 by the node's pieces, -1 where they do not decide;
    clamp (D148d, bp built with nearest_pieces): a cell outside the box read at the box cell closest to it."""
    off, ca, cb = bp.boxi[g]
    if off < 0 or rep < 0:
        return -1
    ijk = np.array(np.unravel_index(int(rep), tuple(bp.nc)))
    loc = ijk - bp.box0[g]
    d = bp.boxd[g]
    if clamp:
        loc = np.minimum(np.maximum(loc, 0), d - 1)
    if (loc < 0).any() or (loc >= d).any():
        return -1
    lb = bp.labels[off + (loc[0] * d[1] + loc[1]) * d[2] + loc[2]]
    return 1 if lb == cb else (0 if lb == ca else -1)


OFF7 = np.stack(np.meshgrid(*[np.arange(-3, 4)] * 3, indexing="ij"), -1).reshape(-1, 3)   # the neighbour offsets


class CarryTables:
    """D148e (`--gap_split carry`): the walls carried between overlapping boxes. gidx (ngrid,) int32: each gap node's
    index k (-1 elsewhere); pairtab (k * 343 + offset) int32: the pair id of node g with the gap node g + o, o in
    [-3, 3]^3 (-1 none); pcode (pid * lmax + L) int32: for that pair, the side at g that g + o's piece L continues as,
    by the occupied cells of the two boxes' overlap: 0 (only g's side 0 piece), 1 (only its side 1 piece), 2 (both,
    neither, or no cell of L in the overlap: uncertain)."""

    def __init__(self, gp):
        dims = tuple(int(d) for d in gp.dims)
        nodes = sorted(gp.boxes)
        self.gidx = np.full(int(np.prod(dims)), -1, np.int32)
        for k, g in enumerate(nodes):
            self.gidx[g] = k
        self.lmax = max([int(b[1].max()) for b in gp.boxes.values()] + [0]) + 1
        self.pairtab = np.full(max(len(nodes), 1) * 343, -1, np.int32)
        codes = []
        ijk = np.stack(np.unravel_index(np.asarray(nodes, np.int64), dims), 1) if nodes else np.zeros((0, 3), int)
        for k, g in enumerate(nodes):
            a0, lab, ca, cb = gp.boxes[g]
            a1 = a0 + np.asarray(lab.shape)
            for t, o in enumerate(OFF7):
                j = ijk[k] + o
                if (j < 0).any() or (j >= np.asarray(dims)).any():
                    continue
                g2 = int(np.ravel_multi_index(tuple(j), dims))
                if self.gidx[g2] < 0:
                    continue
                b0, lab2, _, _ = gp.boxes[g2]
                lo, hi = np.maximum(a0, b0), np.minimum(a1, b0 + np.asarray(lab2.shape))
                code = np.full(self.lmax, 2, np.int32)
                if (hi > lo).all():
                    sub = lab[lo[0] - a0[0]:hi[0] - a0[0], lo[1] - a0[1]:hi[1] - a0[1], lo[2] - a0[2]:hi[2] - a0[2]]
                    sub2 = lab2[lo[0] - b0[0]:hi[0] - b0[0], lo[1] - b0[1]:hi[1] - b0[1], lo[2] - b0[2]:hi[2] - b0[2]]
                    for L in np.unique(sub2[sub2 > 0]):
                        s = set(np.unique(sub[sub2 == L]).tolist())
                        has_a, has_b = ca in s, cb in s
                        code[int(L)] = 0 if has_a and not has_b else (1 if has_b and not has_a else 2)
                self.pairtab[k * 343 + t] = len(codes)
                codes.append(code)
        self.pcode = np.concatenate(codes) if codes else np.full(self.lmax, 2, np.int32)
        self.n_pairs = len(codes)


def carry_sides(bp, ct, g, rep, base, dims):
    """numpy reference of the kernels' D148e rule at node g for particles with representative cells rep (n,) and base
    nodes base (n, 3): 0 / 1 by g's pieces where g's box holds the cell (-1 for a third piece); else the side the piece
    holding it at the particle's stencil nodes continues as at g (pcode), -1 where uncertain or the stencil nodes
    disagree (the node is then the single field)."""
    dims = np.asarray(dims)
    off, ca, cb = bp.boxi[g]
    n = len(rep)
    out = np.full(n, -1, np.int8)
    if off < 0:
        return out
    r = np.stack(np.unravel_index(rep.astype(np.int64), tuple(bp.nc)), 1)

    def label_in(node, sel):
        o2, _, _ = bp.boxi[node]
        loc = r[sel] - bp.box0[node]
        d = bp.boxd[node]
        ins = ((loc >= 0) & (loc < d)).all(1)
        lab = np.full(len(loc), -1, np.int64)
        li = loc[ins]
        lab[ins] = bp.labels[o2 + (li[:, 0] * d[1] + li[:, 1]) * d[2] + li[:, 2]]
        return ins, lab
    allp = np.ones(n, bool)
    ins, lab = label_in(g, allp)
    out[ins & (lab == cb)] = 1
    out[ins & (lab == ca)] = 0
    rest = np.nonzero(~ins)[0]
    if not len(rest):
        return out
    gi = np.asarray(np.unravel_index(g, tuple(dims)))
    res = np.full(len(rest), -1, np.int64)
    unc = np.zeros(len(rest), bool)
    kg = int(ct.gidx[g])
    for o in np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 3):
        j = base[rest] + o
        ok = ((j >= 0) & (j < dims)).all(1)
        g2 = np.where(ok, np.ravel_multi_index(tuple(np.clip(j, 0, dims - 1).T), tuple(dims)), 0)
        for node in np.unique(g2[ok]):
            sel = ok & (g2 == node)
            if bp.boxi[node][0] < 0:
                continue
            ins2, lab2 = label_in(node, rest[sel])
            if not ins2.any():
                continue
            oo = np.asarray(np.unravel_index(node, tuple(dims))) - gi
            pid = ct.pairtab[kg * 343 + int(((oo[0] + 3) * 7 + (oo[1] + 3)) * 7 + (oo[2] + 3))] if kg >= 0 else -1
            idx = np.nonzero(sel)[0][ins2]
            lb = lab2[ins2]
            code = ct.pcode[max(pid, 0) * ct.lmax + np.clip(lb, 0, ct.lmax - 1)]
            c = np.where((pid >= 0) & (lb >= 0) & (lb < ct.lmax), code, 2)
            unc[idx] |= c == 2
            first = (c != 2) & (res[idx] < 0)
            res[idx[first]] = c[first]
            unc[idx] |= (c != 2) & (res[idx] >= 0) & (res[idx] != c)
    out[rest] = np.where(unc | (res < 0), -1, res).astype(np.int8)
    return out
