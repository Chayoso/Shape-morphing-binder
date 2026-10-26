"""d162c_lib.py -- D162c's decomposition (docs, D162c): the outer particles of a state, their normals, and a per-particle
vector field on them split into the stroke band (what a neighbourhood within half a stroke width moves in common) and the
sub-pitch rest, each into normal and tangential. CPU."""
import numpy as np
import scipy.sparse as sps
import torch
from scipy.spatial import cKDTree

from physmorph.pipeline.shape_read import zero_set
from physmorph.render.exterior import ZhuBridson

STROKE = 3.8          # pitches: D161's letters' stroke width
OFFSET = 0.8          # pitches: the field's offset (render/exterior.py)


def pitch_of(tgt):
    """0.708 x the target sample's median 8th-neighbour distance (D157's field pitch, as d162_windows.py reads it)."""
    t = np.asarray(tgt, np.float64)
    sub = t[np.random.default_rng(0).choice(len(t), min(20000, len(t)), replace=False)]
    return 0.708 * float(np.median(cKDTree(sub).query(sub, 9)[0][:, 8])) * (len(sub) / len(t)) ** (1 / 3)


def outer_layer(x, a):
    """(indices, unit normals) of the outermost layer of x: signed distance to the zero set of its Zhu-Bridson field at
    pitch a of at least -(0.8 a + half a spacing), the normal its field's gradient there."""
    x = np.asarray(x, np.float64)
    zs = zero_set(x, a)
    sd = zs.signed(x)
    idx = np.flatnonzero(sd >= -(OFFSET * a + 0.5 * a / 0.708))
    field = ZhuBridson(torch.as_tensor(x, dtype=torch.float32), a)
    _, g, _ = field(torch.as_tensor(x[idx], dtype=torch.float32))
    g = g.detach().numpy().astype(np.float64)
    n = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-30)
    return idx, n


def band_operator(xo, a):
    """The row-normalised averaging over the outer particles within half a stroke width (itself included)."""
    tree = cKDTree(xo)
    pairs = tree.query_pairs(0.5 * STROKE * a, output_type="ndarray")
    m = len(xo)
    i = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(m)])
    j = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(m)])
    W = sps.csr_matrix((np.ones(len(i)), (i, j)), shape=(m, m))
    deg = np.asarray(W.sum(1)).ravel()
    return sps.diags(1.0 / deg) @ W, float(deg.mean())


def decompose(v, n, B):
    """Shares of the four energies (band-normal, band-tangential, sub-normal, sub-tangential) of v on the outer particles,
    their sum and the rms of v."""
    v = np.asarray(v, np.float64)
    vb = B @ v
    vs = v - vb

    def split(u):
        un = (u * n).sum(1, keepdims=True) * n
        return float((un ** 2).sum()), float(((u - un) ** 2).sum())

    bn, bt = split(vb)
    sn, st = split(vs)
    tot = bn + bt + sn + st
    return dict(band_normal=bn / tot, band_tan=bt / tot, sub_normal=sn / tot, sub_tan=st / tot,
                normal=(bn + sn) / tot, band=(bn + bt) / tot, rms=float(np.sqrt((v ** 2).sum(1).mean())))
