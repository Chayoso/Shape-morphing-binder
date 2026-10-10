"""The exterior: the surface of a particle set as the zero set of a field, and discs on it (D59).

The field is negative inside the body and positive outside. The discs sit one to each cell of a lattice fixed in
space that the zero set crosses (the cell's centre projected onto the zero set, kept where it stays in its cell), so
the same surface gives the same discs. Nothing here has mass or acts on the simulation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as nnf

_SPAN = 8192                                                   # lattice coordinates per axis in one key


def key(c):
    return (c[..., 0] * _SPAN + c[..., 1]) * _SPAN + c[..., 2]


def unkey(k):
    return torch.stack((k // (_SPAN * _SPAN), (k // _SPAN) % _SPAN, k % _SPAN), -1)


def cube(lo, hi, device):
    r = torch.arange(lo, hi, device=device)
    return torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)


def connected_sets(pts, r):
    """The display's rule (scripts/probes/settled/surface_layer_probe.py, the e3 records): the connected sets of the
    discs, each linked to those of its 8 nearest within 2.2 r (r: the lattice pitch): their number, and a mask of the
    discs outside the largest set."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from .knn_gpu import knn_self_torch
    d, nb = knn_self_torch(pts, 9)
    link = d[:, 1:] < 2.2 * r
    i = torch.arange(len(pts), device=pts.device)[:, None].expand_as(link)[link].cpu().numpy()
    j = nb[:, 1:][link].cpu().numpy()
    n, label = connected_components(coo_matrix((np.ones(len(i), bool), (i, j)), shape=(len(pts),) * 2), directed=False)
    return n, torch.as_tensor(label != np.bincount(label).argmax(), device=pts.device)


class Bins:
    """The particles in cells of a given side: the particles within that distance of a point are in the 27 cells
    around it."""

    def __init__(self, x, side):
        self.side, self.lo = side, x.min(0).values - side
        cell = key(((x - self.lo) / side).long())
        order = torch.argsort(cell)
        self.cells, counts = torch.unique_consecutive(cell[order], return_counts=True)
        self.width = int(counts.max())
        row = torch.arange(len(self.cells), device=x.device).repeat_interleave(counts)
        self.table = torch.full((len(self.cells), self.width), -1, dtype=torch.long, device=x.device)
        self.table[row, torch.arange(len(x), device=x.device) - (counts.cumsum(0) - counts)[row]] = order

    def around(self, q):
        """The indices of the particles in the 27 cells around each point, -1 where a slot is empty."""
        cell = key(((q - self.lo) / self.side).floor().long()[:, None, :] + cube(-1, 2, q.device)[None])
        at = torch.searchsorted(self.cells, cell).clamp(max=len(self.cells) - 1)
        return torch.where((self.cells[at] == cell)[..., None], self.table[at], -1).reshape(len(q), -1)


class ZhuBridson:
    """f(q) = |q - xbar(q)| - offset (Zhu and Bridson 2005): xbar the mean of the particles within `radius` of q
    weighted by (1 - (d / radius)^2)^3. A thin sheet of particles keeps a thickness of two offsets and a particle
    alone is a sphere. D59: radius 3 pitches, offset 0.8 pitches (pitch: the volume sample's, (V / N)^(1/3));
    D61: Solenthaler's factor on the offset was tried and not kept."""

    def __init__(self, x, pitch, radius=3., offset=.8, device_field=False):
        """device_field (D138): on CUDA the field is the one-kernel form of render/exterior_wp.py (the same field to
        float rounding, 10x faster); the window's disc search uses it, every other caller the tensor form below."""
        self.x, self.pitch, self.radius, self.offset = x, pitch, radius * pitch, offset * pitch
        self.device_field = bool(device_field) and x.is_cuda
        self._bins = self._dbins = None

    @property
    def bins(self):
        if self._bins is None:
            self._bins = Bins(self.x, self.radius)
        return self._bins

    def __call__(self, q, grad=True):
        """f(q), its gradient in q (None without `grad`: the lattice's corners need the sign alone, and on a 300k
        state the gradient was half of the field's 1.2 s at the 3.2M nodes, D96), and the summed weight (zero where
        no particle is within the radius)."""
        if self.device_field:
            from .exterior_wp import DeviceBins, zb_field
            if self._dbins is None:
                self._dbins = DeviceBins(self.x.detach().float(), self.radius)
            return zb_field(self.x, self._dbins, self.radius, self.offset, q, grad)
        x, R = self.x, self.radius
        f, g, s = [], [], []
        for qc in q.split(max(4096, int(2e6 / self.bins.width))):
            idx = self.bins.around(qc)
            p = x[idx.clamp_min(0)]
            qc = qc.detach().requires_grad_(grad)
            with torch.set_grad_enabled(grad):
                w = (1. - (qc[:, None, :] - p).square().sum(-1) / (R * R)).clamp_min(0.) ** 3 * (idx >= 0)
                sw = w.sum(1)
                xbar = (w[..., None] * p).sum(1) / sw.clamp_min(1e-12)[:, None]
                fc = (qc - xbar).norm(dim=1) - self.offset
                if grad:
                    g.append(torch.autograd.grad(fc.sum(), qc)[0])
            f.append(fc.detach()); s.append(sw.detach())
        f, s = torch.cat(f), torch.cat(s)
        return torch.where(s > 1e-6, f, f.new_full((), float("inf"))), torch.cat(g) if grad else None, s

    def project(self, q, steps=6, stay=None):
        """Newton steps onto the zero set, a step no longer than half a pitch: the points that end on it (and, with
        `stay`, no farther than that from where they started along any axis) and the gradient there."""
        start = q
        for _ in range(steps):
            f, g, _ = self(q)
            step = torch.nan_to_num(f, posinf=0.)[:, None] * g / g.square().sum(1, keepdim=True).clamp_min(1e-12)
            n = step.norm(dim=1, keepdim=True)
            q = q - step * (.5 * self.pitch / n.clamp_min(1e-12)).clamp(max=1.)
        f, g, _ = self(q)
        ok = (f.abs() < .02 * self.pitch) & torch.isfinite(q).all(1)
        if stay is not None:
            ok &= (q - start).abs().max(1).values <= stay
        return q[ok], g[ok]

    def neighbours(self, q, skin, most=256):
        """For each point the particles within the radius plus `skin` (the nearest `most` of them), -1 padded."""
        reach = self.radius + skin
        bins, out = Bins(self.x, reach), []
        for qc in q.split(max(1024, int(4e5 / bins.width))):
            idx = bins.around(qc)
            d = torch.where(idx >= 0, (qc[:, None, :] - self.x[idx.clamp_min(0)]).norm(dim=-1), qc.new_full((), float("inf")))
            d, at = d.topk(min(most, d.shape[1]), dim=1, largest=False)
            out.append(torch.where(d < reach, idx.gather(1, at), -1))
        idx = torch.cat(out)
        return idx[:, :int((idx >= 0).sum(1).max())]


class Tracked:
    """Discs of a field's zero set that follow the particles within one window: found at one state (points p0, unit
    normals n0, the field's slope there, each disc's particles), read at another. A disc moves along n0 by the
    field's value at p0 over that slope (one Newton step, the surface's displacement to first order) and takes the
    field's gradient at p0 as its normal; both are functions of the particles, so a loss on the discs reaches them."""

    def __init__(self, field, lattice, h, skin):
        pts, g, _, _ = lattice.discs(field, h, refine=False)
        self.p0, self.n0, self.slope = pts, nnf.normalize(g, dim=1), g.norm(dim=1).clamp_min(1e-6)
        self.idx = field.neighbours(pts, skin)
        self.radius, self.offset, self.h = field.radius, field.offset, h

    def keep(self, mask):
        """Restrict the tracked set to the discs where `mask` holds; the kept discs read exactly as before."""
        self.p0, self.n0, self.slope, self.idx = self.p0[mask], self.n0[mask], self.slope[mask], self.idx[mask]
        return self

    def body_only(self, h) -> int:
        """D127: keep the discs of the largest connected set alone (connected_sets at the lattice pitch, the display's
        rule), so discs apart from the body carry no term; returns the number dropped."""
        _, apart = connected_sets(self.p0, h)
        self.keep(~apart)
        return int(apart.sum())

    def read(self, x):
        """(points, unit normals, displacement along n0) of the discs at the particles x."""
        R, p0 = self.radius, self.p0
        valid = self.idx >= 0
        p = x.index_select(0, self.idx.clamp_min(0).reshape(-1)).view(*self.idx.shape, 3)   # its backward adds, without a sort
        u = p0[:, None, :] - p
        t = (1. - u.square().sum(-1) / (R * R)).clamp_min(0.) * valid
        w = t ** 3
        sw = w.sum(1).clamp_min(1e-12)
        xbar = (w[..., None] * p).sum(1) / sw[:, None]
        d = p0 - xbar
        dist = d.norm(dim=1).clamp_min(1e-12)
        dw = (-6. / (R * R)) * (t * t)[..., None] * u                             # the weights' gradients in q
        J = (torch.einsum("nsa,nsb->nab", p, dw) - xbar[:, :, None] * dw.sum(1)[:, None, :]) / sw[:, None, None]
        dhat = d / dist[:, None]
        normal = nnf.normalize(dhat - torch.einsum("nab,na->nb", J, dhat), dim=1)     # the field's gradient in q
        move = -(dist - self.offset) / self.slope
        return p0 + move[:, None] * self.n0, normal, move


class Lattice:
    """A lattice fixed in space (its corner `origin`); particles farther than `reach` from `center` along an axis
    are left out."""

    def __init__(self, center, reach):
        self.center, self.reach, self.origin = center, reach, center - 1.07 * reach

    def near_nodes(self, x, pitch, ring):
        """The integer coordinates of the nodes of that pitch around the cells that hold particles."""
        held = ((x - self.center).abs() < self.reach).all(1)
        cells = unkey(torch.unique(key(((x[held] - self.origin) / pitch).long())))
        return unkey(torch.unique(key(cells[:, None, :] + cube(-ring, ring + 2, x.device)[None])))

    def at(self, nodes, pitch):
        return self.origin + pitch * nodes.float()

    def zero_set_nodes(self, field, h):
        """The nodes of pitch h, within two cells of two pitches of a particle's cell, that lie within h of the zero
        set by the field's own slope."""
        nodes = self.near_nodes(field.x, 2 * h, 2)
        nodes = self.at((2 * nodes[:, None, :] + cube(0, 2, nodes.device)[None]).reshape(-1, 3), h)
        f, g, _ = field(nodes)
        return nodes[f.abs() < h * g.norm(dim=1)]

    @staticmethod
    def crossed(cells, nodes, f):
        """Of the cells (given by their lowest corners), those with corners on both sides of the zero set; a corner
        that is not among the nodes (sorted keys, with their f) is outside."""
        corner = key(cells[:, None, :] + cube(0, 2, cells.device)[None])
        at = torch.searchsorted(nodes, corner).clamp(max=len(nodes) - 1)
        v = torch.where(nodes[at] == corner, f[at], f.new_full((), float("inf")))
        return cells[(v.min(1).values < 0.) & (v.max(1).values > 0.)]

    def crossed_cells(self, field, h, refine=True):
        """The cells of pitch h that the zero set crosses, and the number of nodes the field was read at.
        refine: they are looked for inside the cells of two pitches that the zero set crosses and the ring around
        those (a bump may enter a cell through a face without reaching a corner); without it they are the cells of
        pitch h with corners on both sides. Either way a closed set that holds no node of the first lattice is not
        found (D59: 0.2-2 % of the zero set, pockets under the surface for the most part)."""
        first = 2 * h if refine else h
        coarse = self.near_nodes(field.x, first, 2)
        big = self.crossed(coarse, key(coarse), field(self.at(coarse, first), grad=False)[0])
        if not refine:
            return big, len(coarse)
        big = unkey(torch.unique(key(big[:, None, :] + cube(-1, 2, big.device)[None])))
        fine = torch.unique(key(2 * big[:, None, :] + cube(0, 3, big.device)[None]))
        f = field(self.at(unkey(fine), h), grad=False)[0]
        return self.crossed((2 * big[:, None, :] + cube(0, 2, big.device)[None]).reshape(-1, 3), fine, f), len(coarse) + len(fine)

    def discs(self, field, h, refine=True):
        """One disc for each crossed cell of pitch h: (points, the field's gradient there, cells crossed, nodes read)."""
        cells, nodes = self.crossed_cells(field, h, refine)
        pts, g = field.project(self.at(cells.float() + .5, h), steps=4, stay=.5 * h)
        return pts, g, len(cells), nodes
