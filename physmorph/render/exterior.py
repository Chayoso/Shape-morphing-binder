"""The exterior: the surface of a particle set as the zero set of a field, and discs on it (D59).

The field is negative inside the body and positive outside. The discs sit one to each cell of a lattice fixed in
space that the zero set crosses (the cell's centre projected onto the zero set, kept where it stays in its cell), so
the same surface gives the same discs. Nothing here has mass or acts on the simulation. Two fields: Zhu and Bridson's
(D59, the default) and the anisotropic density field (D140, `--exterior_field aniso`).
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


LINK, FLOOR, SUPPORT, ALONE = 1.5, 1. / 12., 3., .8          # D140's constants, in pitches (FLOOR in pitches^2)
LINKS_K = 32                                                   # the nearest searched for the links (truncation recorded)


def link_lists(x, pitch):
    """Each particle's links (D140): the particles within LINK pitches of it, itself included (column 0), -1 padded;
    and the share of particles whose LINKS_K nearest all lie within LINK (their lists may be cut short)."""
    from .knn_gpu import knn_self_torch
    k = min(LINKS_K, len(x))
    d, nb = knn_self_torch(x.detach(), k)
    d, nb = d.reshape(len(x), k), nb.reshape(len(x), k).long()          # one column when k = 1
    near = d < LINK * pitch
    return torch.where(near, nb, -1), float(near[:, -1].float().mean()) if k == LINKS_K else 0.


def covariances(x, rows, links, pitch):
    """C_j of the particles `rows` (D140) from their link lists: the links' covariance about their weighted mean,
    weights (1 - (d / LINK a)^2)^3, plus FLOOR a^2 I (the variance of one pitch's uniform cell). Differentiable in x;
    a link stretched past LINK pitches weighs zero."""
    valid = links >= 0
    xj = x.index_select(0, rows)
    p = x.index_select(0, links.clamp_min(0).reshape(-1)).view(*links.shape, 3)
    w = (1. - (p - xj[:, None, :]).square().sum(-1) / (LINK * pitch) ** 2).clamp_min(0.) ** 3 * valid
    sw = w.sum(1).clamp_min(1e-12)
    m = (w[..., None] * p).sum(1) / sw[:, None]
    d = p - m[:, None, :]
    C = torch.einsum("nk,nka,nkb->nab", w, d, d) / sw[:, None, None]
    return C + (FLOOR * pitch * pitch) * torch.eye(3, dtype=x.dtype, device=x.device)


def largest_eigenvalue(C):
    """The largest eigenvalue of each symmetric 3 x 3 matrix, in closed form (Smith 1961; cuSOLVER's batched solver
    refuses a 300k batch)."""
    q = C.diagonal(dim1=1, dim2=2).sum(1) / 3.
    p1 = C[:, 0, 1] ** 2 + C[:, 0, 2] ** 2 + C[:, 1, 2] ** 2
    p2 = (C.diagonal(dim1=1, dim2=2) - q[:, None]).square().sum(1) + 2. * p1
    p = (p2 / 6.).sqrt()
    B = (C - q[:, None, None] * torch.eye(3, dtype=C.dtype, device=C.device)) / p.clamp_min(1e-30)[:, None, None]
    det = (B[:, 0, 0] * (B[:, 1, 1] * B[:, 2, 2] - B[:, 1, 2] * B[:, 2, 1])
           - B[:, 0, 1] * (B[:, 1, 0] * B[:, 2, 2] - B[:, 1, 2] * B[:, 2, 0])
           + B[:, 0, 2] * (B[:, 1, 0] * B[:, 2, 1] - B[:, 1, 1] * B[:, 2, 0]))
    phi = torch.acos((det / 2.).clamp(-1., 1.)) / 3.
    return torch.where(p > 1e-12 * q.abs().clamp_min(1e-30), q + 2. * p * torch.cos(phi), q)


def _level():
    """(level, slope) of Phi^(1/3) at an isolated particle's surface, ALONE pitches out (unit pitch): there Phi^(1/3) is
    its kernel's 1 - s^2 / 9 = 1 - r^2 / (9 FLOOR), so the level is that at r = ALONE and the slope 2 r / (9 FLOOR)."""
    return 1. - ALONE ** 2 / (SUPPORT ** 2 * FLOOR), 2. * ALONE / (SUPPORT ** 2 * FLOOR)


class Anisotropic(ZhuBridson):
    """D140 (Yu and Turk 2013, its constants derived): Phi(q) = sum_j k(s_j), s_j^2 = (q - x_j)^T C_j^-1 (q - x_j),
    k(s) = (1 - s^2 / 9)^3 on s < 3 (three of the particle's own standard deviations: 0.87 pitches for an isolated
    particle, about 1.5 inside the body); C_j = the covariance of its links (covariances()); the surface at the level an
    isolated particle takes at 0.8 pitches, written f = (c^(1/3) - Phi^(1/3)) pitch / slope: Phi^(1/3) is one kernel's
    1 - s^2 / 9 near an isolated particle, so f is a distance there to second order, the projection's tolerance and step
    keep their meaning and one Newton step reads a moved surface (Tracked) as for Zhu and Bridson's (the form (c - Phi),
    Phi a cube near its level, read a 0.05-pitch translation 0.06 pitches off); no smoothing of the centres. A flattened sheet's kernels are flat, so no kernel reaches across an empty
    gap between two near surfaces: the field reads density, not a centroid that falls between them.
    On CUDA the field is the one-kernel form of render/exterior_wp.py; elsewhere the tensor form below."""

    grows = True                                               # its zero set may lie beyond two lattice cells

    def __init__(self, x, pitch):
        self.x, self.pitch = x, pitch
        self.offset = ALONE * pitch
        c, slope = _level()
        self.level, self.scale = c, pitch / slope
        with torch.no_grad():
            self.links, self.truncated = link_lists(x, pitch)
            rows = torch.arange(len(x), device=x.device)
            C = covariances(x.detach(), rows, self.links, pitch)
            self.cinv = torch.linalg.inv(C)
            self.reach = torch.nan_to_num(SUPPORT * largest_eigenvalue(C.double()).clamp_min(0.).sqrt(), nan=0.).to(x.dtype)
        self.radius = float(self.reach.max())
        self.device_field = x.is_cuda
        self._bins = self._dbins = None

    def __call__(self, q, grad=True):
        if self.device_field:
            from .exterior_wp import DeviceBins, aniso_field
            if self._dbins is None:
                self._dbins = DeviceBins(self.x.detach().float(), self.radius)
                self._cis = self.cinv.float()[self._dbins.order].contiguous()
            return aniso_field(self._dbins, self._cis, self.level, self.scale, q, grad)
        f, g, s = [], [], []
        for qc in q.split(max(1024, int(4e5 / self.bins.width))):
            idx = self.bins.around(qc)
            r = qc[:, None, :] - self.x[idx.clamp_min(0)]
            Cr = torch.einsum("nwab,nwb->nwa", self.cinv[idx.clamp_min(0)], r)
            t = (1. - (r * Cr).sum(-1) / SUPPORT ** 2).clamp_min(0.) * (idx >= 0)
            phi = (t ** 3).sum(1)
            cr = phi.clamp_min(1e-30) ** (1. / 3.)
            f.append(torch.where(phi > 0., (self.level - cr) * self.scale, phi.new_full((), float("inf"))))
            s.append(phi)
            if grad:                                       # -grad(Phi^(1/3)) scale, grad(Phi) = -2/3 sum t^2 C^-1 r
                g.append((2. / 9.) * self.scale / (cr * cr)[:, None] * ((t * t)[..., None] * Cr).sum(1))
        return torch.cat(f), torch.cat(g) if grad else None, torch.cat(s)

    def neighbours(self, q, skin, most=256):
        """For each point the particles whose kernel reaches within `skin` of it (the nearest `most`), -1 padded."""
        reach = self.radius + skin
        bins, out = Bins(self.x, reach), []
        for qc in q.split(max(1024, int(4e5 / bins.width))):
            idx = bins.around(qc)
            d = (qc[:, None, :] - self.x[idx.clamp_min(0)]).norm(dim=-1)
            d = torch.where((idx >= 0) & (d < self.reach[idx.clamp_min(0)] + skin), d, qc.new_full((), float("inf")))
            d, at = d.topk(min(most, d.shape[1]), dim=1, largest=False)
            out.append(torch.where(torch.isfinite(d), idx.gather(1, at), -1))
        idx = torch.cat(out)
        return idx[:, :int((idx >= 0).sum(1).max())]


def make_field(kind, x, pitch, radius=3., offset=.8, device_field=False):
    """The exterior's field by its name (D140's switch): "zb" Zhu and Bridson's, "aniso" the anisotropic one."""
    if kind == "aniso":
        return Anisotropic(x, pitch)
    if kind != "zb":
        raise ValueError(f"unknown exterior field {kind!r}")
    return ZhuBridson(x, pitch, radius=radius, offset=offset, device_field=device_field)


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


class TrackedAniso(Tracked):
    """Tracked discs of the anisotropic field (D140): found as Tracked's; read by recomputing, from the particles, the
    covariances of the particles each disc reads (their link lists fixed at the search, as the discs' particle lists
    are) and the field and its gradient at the disc, so a loss on the discs reaches the particles through the centres
    and the covariances alike."""

    def __init__(self, field, lattice, h, skin):
        super().__init__(field, lattice, h, skin)
        self.rows, self.loc = torch.unique(self.idx.clamp_min(0), return_inverse=True)
        self.links = field.links.index_select(0, self.rows)
        self.pitch, self.level, self.scale = field.pitch, field.level, field.scale

    def read(self, x):
        """(points, unit normals, displacement along n0) of the discs at the particles x."""
        Ci = torch.linalg.inv(covariances(x, self.rows, self.links, self.pitch))
        valid = self.idx >= 0
        p = x.index_select(0, self.idx.clamp_min(0).reshape(-1)).view(*self.idx.shape, 3)
        r = self.p0[:, None, :] - p
        Cr = torch.einsum("nwab,nwb->nwa", Ci.index_select(0, self.loc.reshape(-1)).view(*self.idx.shape, 3, 3), r)
        t = (1. - (r * Cr).sum(-1) / SUPPORT ** 2).clamp_min(0.) * valid
        phi = (t ** 3).sum(1)
        normal = nnf.normalize(((t * t)[..., None] * Cr).sum(1), dim=1)          # -grad(Phi), the field's gradient
        move = -(self.level - phi.clamp_min(1e-30) ** (1. / 3.)) * self.scale / self.slope
        return self.p0 + move[:, None] * self.n0, normal, move


def tracked(field, lattice, h, skin):
    """Tracked discs of either field."""
    return (TrackedAniso if isinstance(field, Anisotropic) else Tracked)(field, lattice, h, skin)


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

    def grow(self, field, keys, f, pitch, chunk=1 << 19):
        """D140 (a field whose zero set may lie beyond the ring of near_nodes, Anisotropic): the nodes (sorted keys)
        and their values extended by the 26 neighbours of every inside node until no inside node lies on the edge
        of the set, so that every cell with an inside corner has all its corners read."""
        edge = keys[f < 0.]
        ring = cube(-1, 2, keys.device)
        while len(edge):
            new = []
            for e in edge.split(chunk):
                nb = key(unkey(e)[:, None, :] + ring[None]).reshape(-1)
                at = torch.searchsorted(keys, nb).clamp(max=len(keys) - 1)
                new.append(nb[keys[at] != nb])
            new = torch.unique(torch.cat(new))
            if not len(new):
                break
            fn = field(self.at(unkey(new), pitch), grad=False)[0]
            keys, order = torch.sort(torch.cat([keys, new]))
            f = torch.cat([f, fn])[order]
            edge = new[fn < 0.]
        return keys, f

    def crossed_cells(self, field, h, refine=True):
        """The cells of pitch h that the zero set crosses, and the number of nodes the field was read at.
        refine: they are looked for inside the cells of two pitches that the zero set crosses and the ring around
        those (a bump may enter a cell through a face without reaching a corner); without it they are the cells of
        pitch h with corners on both sides. Either way a closed set that holds no node of the first lattice is not
        found (D59: 0.2-2 % of the zero set, pockets under the surface for the most part)."""
        first = 2 * h if refine else h
        coarse = self.near_nodes(field.x, first, 2)
        if getattr(field, "grows", False):
            keys, f = self.grow(field, key(coarse), field(self.at(coarse, first), grad=False)[0], first)
            coarse = unkey(keys)
            big = self.crossed(coarse, keys, f)
        else:
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
