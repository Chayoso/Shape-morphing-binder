"""d163_pattern.py -- D163: the plaque's lettering P in [0, 1] as a function of a point of the frame (and its outward
normal), shared by the mesh builder (d163_mesh.py) and the reader (d163_read.py). D161's Pattern (d161_pattern.py) with
one more map, "plaque": two flat faces (+z, -z), each with its OWN raster (the front word on +z, the back word on -z),
each read orthographically from outside in the true (right-handed) view:
  front  seen from +z (camera at +z, up +y): the reader's right is +x, so u = (p - c) . (+x), v = (p - c) . y;
  back   seen from -z (camera at -z, up +y): the reader's right is -x, so u = (p - c2) . (-x), v = (p - c2) . y.
(D161's coin map put fwd = -x on +z: its letters read correctly in a renderer whose right is up x forward, i.e. a mirror;
here the letters are right in the true geometry, and every mirrored renderer is flipped back at the image.)
Each face keeps the points that face it (n . lat ramps 0.25 -> 0.5) and lie within one pitch of the front-most surface
along -lat (D161's z-buffer). P is the larger of the two faces' values. Other maps: d161_pattern.Pattern's."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "d161"))
from d161_pattern import Pattern as P161, bilinear  # noqa: E402,F401


class Pattern(P161):
    def __init__(self, kind, tex, texel, u0, v0, consts, tex2=None, u02=0.0, v02=0.0):
        super().__init__(kind, tex, texel, u0, v0, consts)
        self.tex2, self.u02, self.v02 = tex2, float(u02), float(v02)

    @classmethod
    def build_plaque(cls, x_t, a, tex, u0, v0, tex2, u02, v02, texel, offset_uv=(0.0, 0.0)):
        cen = x_t.mean(0).astype(np.float64)
        off = np.asarray(offset_uv, np.float64) * a
        ex, ey, ez = np.eye(3)
        consts = dict(lat=ez, fwd=ex, up=ey, c=np.array([cen[0], cen[1], x_t[:, 2].max()]) - off[0] * ex + off[1] * ey,
                      lat2=-ez, fwd2=-ex, c2=np.array([cen[0], cen[1], x_t[:, 2].min()]) + off[0] * ex + off[1] * ey,
                      pitch=a)
        return cls("plaque", tex, texel, u0, v0, consts, tex2=tex2, u02=u02, v02=v02)

    def uv(self, p, back=False):
        if self.kind == "plaque":
            c = self.c
            q = p - (c["c2"] if back else c["c"])
            return q @ (c["fwd2"] if back else c["fwd"]), q @ c["up"]
        return super().uv(p, back)

    def face(self, p, n, zpts=None, back=False):
        """P of one face of the plaque (front: back=False)."""
        tex, u0, v0 = (self.tex2, self.u02, self.v02) if back else (self.tex, self.u0, self.v0)
        u, v = self.uv(p, back)
        P = bilinear(tex, (u - u0) / self.texel, (v - v0) / self.texel)
        c = self.c
        lat = c["lat2"] if back else c["lat"]
        facing = np.clip(((n @ lat) - 0.25) / 0.25, 0, 1)
        zp = p if zpts is None else zpts
        zu, zv = self.uv(zp, back)
        cell = 0.5 * c["pitch"]
        iu, iv = np.floor(zu / cell).astype(np.int64), np.floor(zv / cell).astype(np.int64)
        span = iv.max() - iv.min() + 2
        key = (iu - iu.min()) * span + (iv - iv.min())
        depth = zp @ lat
        order = np.argsort(key)
        ks, ds = key[order], depth[order]
        uniq, start = np.unique(ks, return_index=True)
        mx = np.maximum.reduceat(ds, start)
        pu, pv = np.floor(u / cell).astype(np.int64), np.floor(v / cell).astype(np.int64)
        pk = (pu - iu.min()) * span + (pv - iv.min())
        pos = np.searchsorted(uniq, pk)
        ok = (pos < len(uniq)) & (pk >= 0)
        front = np.full(len(p), -np.inf)
        front[ok] = np.where(uniq[np.minimum(pos[ok], len(uniq) - 1)] == pk[ok], mx[np.minimum(pos[ok], len(uniq) - 1)], -np.inf)
        vis = (p @ lat) >= front - c["pitch"]
        return P * facing * vis

    def at(self, p, n, faces=None, zpts=None):
        if self.kind == "plaque":
            return np.maximum(self.face(p, n, zpts, False), self.face(p, n, zpts, True))
        return super().at(p, n, faces, zpts)

    def save(self, path):
        extra = {} if self.tex2 is None else dict(tex2=self.tex2, u02=self.u02, v02=self.v02)
        np.savez(path, kind=self.kind, tex=self.tex, texel=self.texel, u0=self.u0, v0=self.v0, **extra,
                 **{f"c_{k}": np.asarray(v) for k, v in self.c.items()})

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        consts = {k[2:]: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files if k.startswith("c_")}
        if "tex2" in z.files:
            return cls(str(z["kind"]), z["tex"], float(z["texel"]), float(z["u0"]), float(z["v0"]), consts,
                       tex2=z["tex2"], u02=float(z["u02"]), v02=float(z["v02"]))
        return cls(str(z["kind"]), z["tex"], float(z["texel"]), float(z["u0"]), float(z["v0"]), consts)
