"""The target's gaps at the kernel's scale and the plane between their two sides (D148, `--gap_split`).

A grid node's velocity is shared by every particle within the cubic kernel's reach of it (the box |x - x_g|_inf < 2 dx).
At a node outside the target where the target inside that box falls apart into two or more pieces, the two sides of a
gap of the target are within one node's reach, and the single field cannot carry the jump their separation needs (D137).
Such a node is a gap node:
its two sides are the two pieces with the largest kernel-weighted target mass, and the plane between them is the
perpendicular bisector of their closest pair of target points (the gap's medial plane where it is narrowest in the box).

No constant is chosen here: the box is the kernel's reach, the pieces are the 26-connected components of the target's
occupied cells at the loss cell (the resolution at which the objective sees the target), the weights are the kernel's.
"""
from __future__ import annotations

import numpy as np


def bspline1(r):
    """The 1-D cubic B-spline of r (in cells)."""
    r = np.abs(r)
    return np.where(r < 1.0, 0.5 * r ** 3 - r ** 2 + 2.0 / 3.0, np.where(r < 2.0, (2.0 - r) ** 3 / 6.0, 0.0))


class GapPlanes:
    """Per MPM node: m (the plane's point) and n (its unit normal from side 0 to side 1; zero = not a gap node).

    y: (M, 3) the target sample; dx, gmin (3,), dims (nx, ny, nz): the MPM grid (nodes at gmin + i dx); cell: the
    occupancy cell (the loss cell). keep_boxes keeps each gap node's component labels (for readers)."""

    def __init__(self, y, dx: float, gmin, dims, cell: float, keep_boxes: bool = False):
        from scipy import ndimage
        from scipy.spatial import cKDTree
        y = np.asarray(y, np.float64)
        gmin = np.asarray(gmin, np.float64).reshape(3)
        dims = tuple(int(d) for d in dims)
        self.dx, self.gmin, self.dims, self.cell = float(dx), gmin, dims, float(cell)
        lo = gmin - 2.0 * dx
        nc = np.ceil((np.asarray(dims) * dx + 4.0 * dx) / cell).astype(np.int64) + 1
        ci = np.floor((y - lo) / cell).astype(np.int64)
        ok = ((ci >= 0) & (ci < nc)).all(1)
        ci, y = ci[ok], y[ok]
        lin = np.ravel_multi_index(ci.T, nc)
        order = np.argsort(lin, kind="stable")
        lin_s, ys = lin[order], y[order]
        count = np.bincount(lin, minlength=int(np.prod(nc))).reshape(nc)
        occ = count > 0
        starts = np.searchsorted(lin_s, np.arange(int(np.prod(nc))), side="left")
        self.lo, self.nc, self.occ = lo, nc, occ
        # the cells whose centres lie inside a node's box: centre lo + (k + 1/2) cell in (x_g - 2 dx, x_g + 2 dx)
        sat = np.zeros(nc + 1, np.int64)
        sat[1:, 1:, 1:] = occ.cumsum(0).cumsum(1).cumsum(2)
        ngrid = int(np.prod(dims))
        self.m = np.zeros((ngrid, 3), np.float32)
        self.n = np.zeros((ngrid, 3), np.float32)
        self.ncomp = np.zeros(ngrid, np.int8)
        self.boxes = {} if keep_boxes else None
        st = np.ones((3, 3, 3), bool)
        ii = np.stack(np.meshgrid(*[np.arange(d) for d in dims], indexing="ij"), -1).reshape(-1, 3)
        xg = gmin + ii * dx
        k0 = np.ceil((xg - 2.0 * dx - lo) / cell - 0.5 + 1e-9).astype(np.int64)
        k1 = np.floor((xg + 2.0 * dx - lo) / cell - 0.5 - 1e-9).astype(np.int64) + 1      # exclusive
        k0, k1 = np.clip(k0, 0, nc), np.clip(k1, 0, nc)
        tot = (sat[k1[:, 0], k1[:, 1], k1[:, 2]] - sat[k0[:, 0], k1[:, 1], k1[:, 2]] - sat[k1[:, 0], k0[:, 1], k1[:, 2]]
               - sat[k1[:, 0], k1[:, 1], k0[:, 2]] + sat[k0[:, 0], k0[:, 1], k1[:, 2]] + sat[k0[:, 0], k1[:, 1], k0[:, 2]]
               + sat[k1[:, 0], k0[:, 1], k0[:, 2]] - sat[k0[:, 0], k0[:, 1], k0[:, 2]])
        n_multi = n_three = n_inside = 0
        own = np.floor((xg - lo) / cell).astype(np.int64)            # the occupancy cell the node lies in
        own_in = ((own >= 0) & (own < nc)).all(1)
        for g in np.nonzero(tot > 1)[0]:
            a0, a1 = k0[g], k1[g]
            if own_in[g] and occ[own[g, 0], own[g, 1], own[g, 2]]:
                n_inside += 1                                        # inside the target: not a gap of it
                continue
            sub = occ[a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]]
            lab, nl = ndimage.label(sub, structure=st)
            if nl < 2:
                continue
            kk = np.stack(np.nonzero(sub), -1)
            cc = lo + (kk + a0 + 0.5) * cell
            r = (cc - xg[g]) / dx
            w = bspline1(r[:, 0]) * bspline1(r[:, 1]) * bspline1(r[:, 2])
            cnt = count[a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]][sub]
            lb = lab[sub]
            W = np.bincount(lb, weights=w * cnt, minlength=nl + 1)[1:]
            rank = np.argsort(-W)
            if W[rank[1]] <= 0.0:
                continue
            n_multi += 1
            n_three += int(nl > 2 and W[rank[2]] > 0.0)
            ca, cb = rank[0] + 1, rank[1] + 1

            def pts(c):
                cells = kk[lb == c] + a0
                li = np.ravel_multi_index(cells.T, nc)
                return np.concatenate([ys[starts[q]:starts[q] + count.flat[q]] for q in li])
            # the closest pair: the cells of each side near the other first, then their points
            ta = cc[lb == ca]
            tb = cc[lb == cb]
            dab, _ = cKDTree(tb).query(ta)
            dba, _ = cKDTree(ta).query(tb)
            reach = float(min(dab.min(), dba.min())) + 2.0 * np.sqrt(3.0) * cell
            pa_cells = (kk[lb == ca] + a0)[dab <= reach]
            pb_cells = (kk[lb == cb] + a0)[dba <= reach]
            pa = np.concatenate([ys[starts[q]:starts[q] + count.flat[q]]
                                 for q in np.ravel_multi_index(pa_cells.T, nc)])
            pb = np.concatenate([ys[starts[q]:starts[q] + count.flat[q]]
                                 for q in np.ravel_multi_index(pb_cells.T, nc)])
            d, j = cKDTree(pb).query(pa)
            i = int(np.argmin(d))
            a, b = pa[i], pb[j[i]]
            nv = b - a
            ln = float(np.linalg.norm(nv))
            if ln <= 0.0:
                continue
            self.m[g] = 0.5 * (a + b)
            self.n[g] = nv / ln
            self.ncomp[g] = min(nl, 127)
            if keep_boxes:
                self.boxes[int(g)] = (a0.copy(), lab.astype(np.int16), int(ca), int(cb))
        self.n_gap, self.n_three, self.n_inside = n_multi, n_three, n_inside
        self.mask = np.linalg.norm(self.n, axis=1) > 0

    def side(self, q, g):
        """The side (0 / 1) of points q (K, 3) at nodes g (K,) by the plane test; -1 where g is not a gap node."""
        s = (np.einsum("ij,ij->i", q - self.m[g], self.n[g]) > 0.0).astype(np.int8)
        return np.where(self.mask[g], s, -1)

    def side_comp(self, q, g):
        """The side by the components: 0 / 1 when q's occupancy cell lies in node g's box and in its side 0 / 1 piece;
        -1 otherwise (outside the box, an empty cell, or a third piece)."""
        out = np.full(len(g), -1, np.int8)
        ci = np.floor((q - self.lo) / self.cell).astype(np.int64)
        for t, gg in enumerate(g):
            bx = self.boxes.get(int(gg))
            if bx is None:
                continue
            a0, lab, ca, cb = bx
            r = ci[t] - a0
            if (r < 0).any() or (r >= lab.shape).any():
                continue
            c = lab[r[0], r[1], r[2]]
            out[t] = 0 if c == ca else (1 if c == cb else -1)
        return out
