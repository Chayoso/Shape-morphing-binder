"""The exterior: the surface of a particle set as the zero set of a field, and discs on it (D59).

The field is negative inside the body and positive outside. The discs sit one to each cell of a lattice fixed in
space that the zero set crosses (the cell's centre projected onto the zero set, kept where it stays in its cell), so
the same surface gives the same discs. Nothing here has mass or acts on the simulation.
"""
from __future__ import annotations

import math

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


def largest_eigenvalue(J):
    """The largest real part among the eigenvalues of real 3x3 matrices (the characteristic cubic in closed form)."""
    a = -(J[:, 0, 0] + J[:, 1, 1] + J[:, 2, 2])
    b = (J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0] + J[:, 0, 0] * J[:, 2, 2] - J[:, 0, 2] * J[:, 2, 0]
         + J[:, 1, 1] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 1])
    c = -torch.linalg.det(J)
    Q = (a * a - 3. * b) / 9.
    T = (2. * a ** 3 - 9. * a * b + 27. * c) / 54.
    root = Q.clamp_min(1e-20).sqrt()
    theta = torch.acos((T / root ** 3).clamp(-1. + 1e-6, 1. - 1e-6))
    three = -2. * root * torch.cos((theta + 2. * math.pi) / 3.) - a / 3.           # three real roots: the largest
    A = -torch.sign(T) * (T.abs() + (T * T - Q ** 3).clamp_min(0.).sqrt()).clamp_min(1e-20) ** (1. / 3.)
    AB = A + torch.where(A.abs() > 1e-12, Q / torch.where(A.abs() > 1e-12, A, torch.ones_like(A)), torch.zeros_like(A))
    one = torch.maximum(AB - a / 3., -.5 * AB - a / 3.)                           # one real root, a complex pair
    return torch.where(T * T < Q ** 3, three, one)


class ZhuBridson:
    """f(q) = |q - xbar(q)| - offset (Zhu and Bridson 2005): xbar the mean of the particles within `radius` of q
    weighted by (1 - (d / radius)^2)^3. A thin sheet of particles keeps a thickness of two offsets and a particle
    alone is a sphere. D59: radius 3 pitches, offset 0.8 pitches (pitch: the volume sample's, (V / N)^(1/3)).

    gaps: the offset is multiplied by Solenthaler, Schlaefli and Pajarola's factor (2007, Eq. 23-26): where xbar
    moves faster than q, between near but separate bodies and in concavities, the field would put surface that
    belongs to none; the factor is 1 while the largest eigenvalue of d xbar / d q is below 0.4 and falls to 0 at 2
    (gamma^3 - 3 gamma^2 + 3 gamma, gamma = (2 - eigenvalue) / 1.6)."""

    def __init__(self, x, pitch, radius=3., offset=.8, gaps=False):
        self.x, self.pitch, self.radius, self.offset, self.gaps = x, pitch, radius * pitch, offset * pitch, gaps
        self.bins = Bins(x, self.radius)

    def __call__(self, q):
        """f(q), its gradient in q, and the summed weight (zero where no particle is within the radius)."""
        x, R = self.x, self.radius
        f, g, s = [], [], []
        for qc in q.split(max(2048, int((6e5 if self.gaps else 2e6) / self.bins.width))):
            idx = self.bins.around(qc)
            p = x[idx.clamp_min(0)]
            qc = qc.detach().requires_grad_(True)
            with torch.enable_grad():
                u = qc[:, None, :] - p
                t = (1. - u.square().sum(-1) / (R * R)).clamp_min(0.) * (idx >= 0)
                w = t ** 3
                sw = w.sum(1)
                xbar = (w[..., None] * p).sum(1) / sw.clamp_min(1e-12)[:, None]
                offset = self.offset
                if self.gaps:
                    dw = (-6. / (R * R)) * (t * t)[..., None] * u                 # the weights' gradients in q
                    J = (torch.einsum("nsa,nsb->nab", p, dw) - xbar[:, :, None] * dw.sum(1)[:, None, :]) / sw.clamp_min(1e-12)[:, None, None]
                    gamma = ((2. - largest_eigenvalue(J)) / 1.6).clamp(0., 1.)
                    offset = offset * (1. - (1. - gamma) ** 3)
                fc = (qc - xbar).norm(dim=1) - offset
                gc, = torch.autograd.grad(fc.sum(), qc)
            f.append(fc.detach()); g.append(torch.nan_to_num(gc)); s.append(sw.detach())
        f, g, s = torch.cat(f), torch.cat(g), torch.cat(s)
        return torch.where(s > 1e-6, f, f.new_full((), float("inf"))), g, s

    def project(self, q, steps=6, stay=None):
        """Newton steps onto the zero set, a step no longer than half a pitch: the points that end on it (and, with
        `stay`, no farther than that from where they started along any axis) and the unit gradient there."""
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
        return q[ok], nnf.normalize(g[ok], dim=1)


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

    def crossed_cells(self, field, h):
        """The cells of pitch h that the zero set crosses, and the number of nodes the field was read at. They are
        looked for inside the cells of two pitches that the zero set crosses and the ring around those (a bump may
        enter a cell through a face without reaching a corner); a closed set that holds no node of two pitches is
        not found (D59: 0.2-2 % of the zero set, pockets under the surface for the most part)."""
        coarse = self.near_nodes(field.x, 2 * h, 2)
        big = self.crossed(coarse, key(coarse), field(self.at(coarse, 2 * h))[0])
        big = unkey(torch.unique(key(big[:, None, :] + cube(-1, 2, big.device)[None])))
        fine = torch.unique(key(2 * big[:, None, :] + cube(0, 3, big.device)[None]))
        f = field(self.at(unkey(fine), h))[0]
        return self.crossed((2 * big[:, None, :] + cube(0, 2, big.device)[None]).reshape(-1, 3), fine, f), len(coarse) + len(fine)

    def discs(self, field, h):
        """One disc for each crossed cell of pitch h: (points, unit normals, cells crossed, nodes read)."""
        cells, nodes = self.crossed_cells(field, h)
        pts, normals = field.project(self.at(cells.float() + .5, h), steps=4, stay=.5 * h)
        return pts, normals, len(cells), nodes
