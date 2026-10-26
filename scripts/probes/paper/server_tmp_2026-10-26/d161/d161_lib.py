"""d161_lib.py -- D161's shared pieces (CPU, numpy / scipy; no pipeline state).

- target_frame: the run's target frame (prepare.prepare's load_normalized of the target at seed + 1, matched to the
  source sample's volume; d156_demand.py's route), so that a mesh, a pattern and a reading live where the run's
  particles live.
- Surface: points of the smooth target mesh's surface in that frame (area-uniform), outward normals smoothed over
  their k nearest (sign-aligned), the nearest foot of any point and its depth below the surface.
- Patterns P on the surface with their surface gradient: Zonal (a sphere: cos(m theta) about an axis, one exact
  wavelength 2 pi R / m), Stripes (cos(k y.e) inside a smooth window), Letters (a raster of text projected onto the
  surface: orthographic along a direction for the bunny's flank, equirectangular about an axis for a sphere).
- displacement: the incompressible relief field of a single-wavelength pattern, exp(-k d) [P n + grad_s P / k]
  (the gradient of the harmonic exp(-k d) cos(k s) / k: divergence-free for a flat surface), so that a relief of
  amplitude delta moves material without compressing it, as a body carrying the relief at uniform density would.
- read_relief: the relief a point set carries against a pattern: its offset from the smooth surface regressed on an
  affine background plus the pattern over the pattern's region; the amplitude (length) and the correlation of the
  background-free offset with the pattern.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

NT = 8


def target_frame(src: str, tgt: str, n: int, seed: int):
    """(frame, target sample): load_normalized of the target as prepare.prepare puts it (match_density leaves the
    target and its frame untouched; only the source is rescaled)."""
    from physmorph.sampling.mesh import load_normalized
    _, v_src = load_normalized(src, n, seed, return_volume=True, sample="stratified")
    frame = {}
    x_t = load_normalized(tgt, n, seed + 1, match_volume=v_src, sample="stratified", frame=frame)
    return frame, x_t


def mesh_in_frame(path: str, frame: dict):
    """(V, F) of the (oriented) mesh at `path` in the frame: (vertices - offset) x scale, as sample_in_frame."""
    from physmorph.sampling.mesh import _oriented
    m = _oriented(path)
    return (np.asarray(m.vertices, np.float64) - frame["offset"]) * frame["scale"], np.asarray(m.faces, np.int64)


class Surface:
    """n area-uniform points of the smooth mesh's surface in the frame with outward normals: the faces' normals,
    turned all at once if as a whole they point into the body (target.target_relief's rule), averaged over the k
    nearest points with each neighbour's sign aligned to the point's own (a smooth normal that a sliver face does not
    flip)."""

    def __init__(self, path: str, frame: dict, n: int, inside_pts: np.ndarray, k: int = 16, seed: int = 0):
        from physmorph.sampling.mesh import surface_in_frame
        P, Nf = surface_in_frame(path, frame, n, seed)
        P, Nf = P.astype(np.float64), Nf.astype(np.float64)
        inside = inside_pts[cKDTree(inside_pts).query(P, 8, workers=NT)[1]].mean(1)
        if np.sign(((P - inside) * Nf).sum(1)).mean() < 0:
            Nf = -Nf
        self.tree = cKDTree(P)
        nb = self.tree.query(P, k, workers=NT)[1]
        s = np.sign((Nf[nb] * Nf[:, None, :]).sum(2))
        Ns = (s[..., None] * Nf[nb]).sum(1)
        self.P = P
        self.N = Ns / np.maximum(np.linalg.norm(Ns, axis=1, keepdims=True), 1e-12)
        self.n = len(P)

    def foot(self, x: np.ndarray):
        """(index of the nearest surface point, depth below the surface along its normal: > 0 inside)."""
        _, i = self.tree.query(np.asarray(x, np.float64), 1, workers=NT)
        return i, ((self.P[i] - x) * self.N[i]).sum(1)


def _tangential(g: np.ndarray, n: np.ndarray) -> np.ndarray:
    return g - (g * n).sum(1, keepdims=True) * n


class Zonal:
    """P = cos(m theta), theta the polar angle about `axis` through `center`; on a sphere of radius R one wavelength
    w = 2 pi R / m along the meridians (m an integer: smooth at both poles). k = m / R."""

    def __init__(self, center, axis, R: float, m: int):
        self.c, self.a = np.asarray(center, np.float64), np.asarray(axis, np.float64) / np.linalg.norm(axis)
        self.R, self.m, self.k = float(R), int(m), float(m) / float(R)
        self.wavelength = 2 * np.pi * self.R / self.m

    def at(self, S: Surface):
        r = S.P - self.c
        rh = r / np.linalg.norm(r, axis=1, keepdims=True)
        ct = np.clip(rh @ self.a, -1, 1)
        th = np.arccos(ct)
        st = np.sqrt(np.maximum(1 - ct * ct, 1e-12))
        e_th = (ct[:, None] * rh - self.a[None]) / st[:, None]              # toward increasing theta
        P = np.cos(self.m * th)
        G = -(self.m / self.R) * np.sin(self.m * th)[:, None] * e_th          # the surface gradient of P
        return P, _tangential(G, S.N), np.ones(S.n)


class Stripes:
    """P = W(y) cos(k y.e): planar stripes of wavelength 2 pi / k along e inside a smooth window W = cos^2 of the
    distance from `center` over `radius` (zero beyond it). The surface wavelength is 2 pi / (k |e_t|) where the
    surface turns away from e."""

    def __init__(self, e, k: float, center, radius: float):
        self.e = np.asarray(e, np.float64) / np.linalg.norm(e)
        self.k, self.c, self.rho = float(k), np.asarray(center, np.float64), float(radius)
        self.wavelength = 2 * np.pi / self.k

    def at(self, S: Surface):
        r = S.P - self.c
        dist = np.linalg.norm(r, axis=1)
        inside = dist < self.rho
        W = np.where(inside, np.cos(0.5 * np.pi * dist / self.rho) ** 2, 0.0)
        dW = np.where(inside, -np.pi / self.rho * np.cos(0.5 * np.pi * dist / self.rho)
                      * np.sin(0.5 * np.pi * dist / self.rho), 0.0)[:, None] * r / np.maximum(dist, 1e-12)[:, None]
        ph = self.k * (S.P @ self.e)
        P = W * np.cos(ph)
        G = dW * np.cos(ph)[:, None] - (W * self.k * np.sin(ph))[:, None] * self.e[None]
        return P, _tangential(G, S.N), W


def displacement(x: np.ndarray, S: Surface, P: np.ndarray, G: np.ndarray, k: float, foot=None):
    """(N,3) the incompressible relief field of unit surface amplitude at the points x: exp(-k d) [P n + G / k] at
    each point's nearest foot (d its depth, clamped at 0). foot: (index, depth) from S.foot(x), if known."""
    i, d = S.foot(x) if foot is None else foot
    dec = np.exp(-k * np.maximum(d, 0.0))[:, None]
    return dec * (P[i][:, None] * S.N[i] + G[i] / k)


def read_relief(Q: np.ndarray, S: Surface, P: np.ndarray, W: np.ndarray | None = None, reach: float = np.inf,
                wmin: float = 0.5):
    """The relief the points Q carry against the pattern P (values on S's points): each point's outward offset h from
    its foot (-depth), over the points whose foot lies where W >= wmin and within `reach` of the surface, regressed as
    h = c0 + c . y + A P(y) (an affine background); returns dict(A (length), corr (Pearson of h minus the background
    against P minus its mean), n, rms_h (after the background), rms_P)."""
    i, d = S.foot(Q)
    h = -d
    sel = np.abs(h) < reach
    if W is not None:
        sel &= W[i] >= wmin
    if sel.sum() < 20:
        return dict(A=float("nan"), corr=float("nan"), n=int(sel.sum()), rms_h=float("nan"), rms_P=float("nan"))
    y, hp, pp = S.P[i[sel]], h[sel], P[i[sel]]
    X = np.concatenate([np.ones((len(y), 1)), y - y.mean(0), pp[:, None]], 1)
    coef, *_ = np.linalg.lstsq(X, hp, rcond=None)
    Xb = X[:, :4]
    cb, *_ = np.linalg.lstsq(np.concatenate([Xb, pp[:, None]], 1), hp, rcond=None)
    bg = Xb @ cb[:4]
    r = hp - bg
    pc = pp - pp.mean()
    corr = float((r - r.mean()) @ pc / max(np.linalg.norm(r - r.mean()) * np.linalg.norm(pc), 1e-30))
    return dict(A=float(coef[-1]), corr=corr, n=int(sel.sum()), rms_h=float(np.sqrt(((r - r.mean()) ** 2).mean())),
                rms_P=float(np.sqrt((pc ** 2).mean())))
