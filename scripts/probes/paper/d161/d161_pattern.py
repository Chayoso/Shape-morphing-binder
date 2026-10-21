"""d161_pattern.py -- D161: the relief pattern P in [0, 1] as a function of a point of the frame (and its outward normal),
shared by the mesh builder (d161_mesh.py: P at the relief mesh's vertices) and the reader (d161_read.py: P at the
smooth surface's points, where the run's particles stand). A raster `tex` (texel `texel`, origin (u0, v0)) read
bilinearly at the point's map coordinates (u, v):
  sphere  u = R phi, v = R psi: longitude phi about the vertical axis through `center` (from +z toward -x, so the text reads from outside, minus lon0)
          and latitude psi, R the sphere's mean radius (the text keeps its size along the equator);
  flank   u = (p - c) . fwd, v = (p - c) . up, orthographic along `lat` onto the +lat side, kept where the point faces
          that side (n . lat ramps from 0.25 to 0.5) and lies within one pitch of the front-most surface along -lat
          (a z-buffer over the map from the points the pattern is evaluated on, or given)."""
from __future__ import annotations

import numpy as np


def bilinear(tex, x, y):
    nx, ny = tex.shape
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    out = np.zeros(len(x))
    for dx, wx in ((0, 1 - fx), (1, fx)):
        for dy, wy in ((0, 1 - fy), (1, fy)):
            i, j = x0 + dx, y0 + dy
            ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
            out[ok] += (wx * wy)[ok] * tex[i[ok], j[ok]]
    return out


class Pattern:
    def __init__(self, kind, tex, texel, u0, v0, consts):
        self.kind, self.tex, self.texel, self.u0, self.v0, self.c = kind, tex, float(texel), float(u0), float(v0), consts

    @classmethod
    def build(cls, spec, frame, x_t, a, tex, texel, u0, v0):
        kind = spec["map"]
        cen = x_t.mean(0).astype(np.float64)
        if kind == "sphere":
            consts = dict(center=cen, R=float(np.linalg.norm(x_t - cen, axis=1).max()), lon0=np.radians(spec.get("lon0", 0.0)))
        elif kind == "coin":                 # both flat faces (+z, -z), each read from its own side (the text not mirrored)
            off = np.asarray(spec.get("offset_uv", [0.0, 0.0]), np.float64) * a
            ex, ey, ez = np.eye(3)
            consts = dict(lat=ez, fwd=-ex, up=ey, c=np.array([cen[0], cen[1], x_t[:, 2].max()]) - off[0] * ex + off[1] * ey,
                          lat2=-ez, fwd2=ex, c2=np.array([cen[0], cen[1], x_t[:, 2].min()]) + off[0] * ex + off[1] * ey,
                          pitch=a)
        else:
            ext = x_t.max(0) - x_t.min(0)
            lat = np.eye(3)[0] if ext[0] < ext[2] else np.eye(3)[2]
            fwd = np.eye(3)[2] if ext[0] < ext[2] else np.eye(3)[0]
            up = np.cross(lat, fwd)
            if up[1] < 0:
                fwd, up = -fwd, -up
            band = np.abs(x_t[:, 1] - cen[1]) < 0.5
            cf = x_t[band][np.argmax((x_t[band] - cen) @ lat)].astype(np.float64)
            off = np.asarray(spec.get("offset_uv", [0.0, 0.0]), np.float64) * a
            consts = dict(lat=lat, fwd=fwd, up=up, c=cf + off[0] * fwd + off[1] * up, pitch=a)
        return cls(kind, tex, texel, u0, v0, consts)

    def uv(self, p, back=False):
        c = self.c
        if self.kind == "coin":
            q = p - (c["c2"] if back else c["c"])
            return q @ (c["fwd2"] if back else c["fwd"]), q @ c["up"]
        if self.kind == "sphere":
            r = p - c["center"]
            phi = np.arctan2(-r[:, 0], r[:, 2]) - c["lon0"]   # increasing to the camera's right seen from +z
            phi = (phi + np.pi) % (2 * np.pi) - np.pi
            psi = np.arcsin(np.clip(r[:, 1] / np.maximum(np.linalg.norm(r, axis=1), 1e-12), -1, 1))
            return c["R"] * phi, c["R"] * psi
        q = p - c["c"]
        return q @ c["fwd"], q @ c["up"]

    def at(self, p, n, faces=None, zpts=None):
        """P at points p (outward normals n). Flank, coin: the z-buffer from zpts (default: p itself)."""
        if self.kind == "coin":
            return np.maximum(self._ortho(p, n, zpts, False), self._ortho(p, n, zpts, True))
        if self.kind == "flank":
            return self._ortho(p, n, zpts, False)
        u, v = self.uv(p)
        return bilinear(self.tex, (u - self.u0) / self.texel, (v - self.v0) / self.texel)

    def _ortho(self, p, n, zpts, back):
        u, v = self.uv(p, back)
        P = bilinear(self.tex, (u - self.u0) / self.texel, (v - self.v0) / self.texel)
        if True:
            c = self.c
            lat = c["lat2"] if back else c["lat"]
            facing = np.clip(((n @ lat) - 0.25) / 0.25, 0, 1)
            zp = p if zpts is None else zpts
            zu, zv = self.uv(zp, back)
            cell = 0.5 * c["pitch"]
            iu, iv = np.floor(zu / cell).astype(np.int64), np.floor(zv / cell).astype(np.int64)
            key = (iu - iu.min()) * (iv.max() - iv.min() + 2) + (iv - iv.min())
            depth = zp @ lat
            zb = {}
            order = np.argsort(key)
            ks, ds = key[order], depth[order]
            uniq, start = np.unique(ks, return_index=True)
            mx = np.maximum.reduceat(ds, start)
            zb = dict(zip(uniq.tolist(), mx.tolist()))
            pu, pv = np.floor(u / cell).astype(np.int64), np.floor(v / cell).astype(np.int64)
            pk = (pu - iu.min()) * (iv.max() - iv.min() + 2) + (pv - iv.min())
            front = np.array([zb.get(int(k), -np.inf) for k in pk])
            vis = (p @ lat) >= front - c["pitch"]
            P = P * facing * vis
        return P

    def save(self, path):
        np.savez(path, kind=self.kind, tex=self.tex, texel=self.texel, u0=self.u0, v0=self.v0,
                 **{f"c_{k}": np.asarray(v) for k, v in self.c.items()})

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        consts = {k[2:]: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files if k.startswith("c_")}
        return cls(str(z["kind"]), z["tex"], float(z["texel"]), float(z["u0"]), float(z["v0"]), consts)
