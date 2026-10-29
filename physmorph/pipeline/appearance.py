"""Appearance (D160): a fixed colour per particle from the source, colour pictures of the target.

Every term of the physics objective (the Sinkhorn transport, the surface proximity, the end drift, the released motion,
the cleanup) reads only the set of particle positions: exchanging two particles changes none of them. Which material
ends where, and so the colour layout of the end state, is a null space of the physics objective. A scene gives every
source particle a fixed albedo, carried by its index (advected with the material, never a control), and every point of
the target samples the albedo of the target's paint; the render's shading term then compares colour pictures
(render_loss.shaded_discs with albedo, target.build_target), and only the render can choose where each colour goes.

The scenes paint with two albedos, WHITE and RED, by fields of position fixed once from the samples' own quantiles (so
the volumes painted are exact shares of each sample); the same field paints every target draw. Source: isosphere; the
bunny scenes read the ears' base and the head's direction of the bunny's frame (BUNNY_EAR_BASE,
BUNNY_HEAD), measured once on its target samples.
  white  one channel of ones on both sides: the term without colour, bit for bit (the reduction test)
  half   source red above its median height; target red above its median height (the transport's own layout, the
         control: the lambda = 0 twin is nearly right)
  band   source red above its median height; target red above its 65 % height quantile (the top 35 %): a short move
  ears   source red above its 95 % height quantile (the top 5 %); target red above the ears' base (bunny,
         BUNNY_EAR_BASE): a medium move
  turn   source red above its median height; target red on the head's side of the target's median plane across the
         head's horizontal direction (bunny, BUNNY_HEAD): a long move
  stripes  (D173) three vertical stripes across the deliverables' front view, the middle one red: s = x . e_h, e_h the front
         camera's horizontal image axis (render/studio.StudioRaster at azimuth FRONT_AZ: (cos, 0, -sin)), cut at the
         target's 1/3 and 2/3 quantiles of s; source red its top third (above its 2/3 height quantile), the same budget.
         The transport carries the source's top to the target's top, which the stripes cut across: the twin's layout
         meets the paint at chance (Cohen's kappa 0.00-0.01 on the bunny twins at 40k / 300k, 0.02-0.03 on the outer
         layer; band 0.68 / 0.78, half 0.88; tmp/d173/d173_layout_check.py), so only the render can lay the stripes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WHITE = (1.0, 1.0, 1.0)
RED = (0.9, 0.25, 0.2)
SCENES = ("white", "half", "band", "ears", "turn", "stripes")   # and "texture" (D173, make_appearance with a TextureMap)
FRONT_AZ = 35.0                       # D173 (stripes): the deliverables' front camera azimuth, degrees (d160_studio / d170_studio)


@dataclass
class Appearance:
    scene: str
    src: np.ndarray                       # (N, C) the body particles' albedos, by index
    tgt: np.ndarray                       # (N, C) the target sample's
    draws: list = field(default_factory=list)   # [(N, C)] the further target samples' (the render's picture draws)
    params: dict = field(default_factory=dict)  # the scene's numbers (thresholds, directions), for the record
    note: str = ""

    def target_albedos(self) -> list:
        return [self.tgt] + list(self.draws)

    tex: object = None                          # D173 (texture): the TextureMap the target is painted by

    def albedo_at(self, x: np.ndarray) -> np.ndarray:
        """D162: the target's paint at any points (an independent target sample for the render's floor)."""
        if self.scene == "white":
            return np.ones((len(x), 1), np.float32)
        if self.scene == "texture":
            return self.tex.at(x)
        return paint(target_red(self.params, x), self.params)


class TextureMap:
    """D173: an OBJ's UV-mapped image (map_Kd) as an albedo field of the target's frame. The OBJ's vertices are in the raw
    coordinates of the target mesh (--tgt): they are put in the frame by the target's own orientation and the frame's
    offset and scale, as mesh_in_frame puts the target. A point's albedo is the image's colour (sRGB to linear) at the
    nearest of n area-uniform points of the textured surface (their UVs barycentric over the face's corners, read
    bilinearly). Volume points take the colour of the surface nearest to them."""

    def __init__(self, obj: str, img: str, frame: dict, orient_path: str, n: int = 2_000_000, seed: int = 0):
        from PIL import Image
        from scipy.spatial import cKDTree
        from ..sampling.orientation import orient_name, rotation
        v, vt, fv, ft = [], [], [], []
        for line in open(obj):
            t = line.split()
            if not t:
                continue
            if t[0] == "v":
                v.append([float(c) for c in t[1:4]])
            elif t[0] == "vt":
                vt.append([float(c) for c in t[1:3]])
            elif t[0] == "f":
                idx = [c.split("/") for c in t[1:]]
                for k in range(1, len(idx) - 1):            # fan-triangulated
                    tri = (idx[0], idx[k], idx[k + 1])
                    fv.append([int(c[0]) - 1 for c in tri])
                    ft.append([int(c[1]) - 1 for c in tri])
        V = np.asarray(v, np.float64) @ rotation(orient_name(orient_path)).T
        V = (V - frame["offset"]) * frame["scale"]
        VT, FV, FT = np.asarray(vt, np.float64), np.asarray(fv), np.asarray(ft)
        tri = V[FV]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        rng = np.random.default_rng(seed)
        f = rng.choice(len(FV), n, p=area / area.sum())
        r1, r2 = rng.random((2, n))
        sq = np.sqrt(r1)
        b = np.stack([1 - sq, sq * (1 - r2), sq * r2], 1)
        self.P = (b[:, :, None] * tri[f]).sum(1)
        uv = (b[:, :, None] * VT[FT[f]]).sum(1)
        im = np.asarray(Image.open(img).convert("RGB"), np.float64) / 255.0
        H, W = im.shape[:2]
        x, y = uv[:, 0] % 1.0 * (W - 1), (1.0 - uv[:, 1] % 1.0) * (H - 1)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
        fx, fy = (x - x0)[:, None], (y - y0)[:, None]
        c = (im[y0, x0] * (1 - fx) * (1 - fy) + im[y0, x1] * fx * (1 - fy) + im[y1, x0] * (1 - fx) * fy + im[y1, x1] * fx * fy)
        self.C = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)   # sRGB to linear
        self.tree = cKDTree(self.P)
        self.obj, self.img = obj, img
        self.center = self.P.mean(0)
        u = self.P - self.center
        self.dirs = cKDTree(u / np.linalg.norm(u, axis=1, keepdims=True))

    def at(self, x: np.ndarray) -> np.ndarray:
        return self.C[self.tree.query(np.asarray(x, np.float64), 1, workers=-1)[1]]

    def radial(self, x: np.ndarray) -> np.ndarray:
        """The colour of the textured surface in each point's direction from its own sample's centroid (the source's
        paint: the texture projected radially, a fixed map the identity-blind transport does not follow)."""
        u = np.asarray(x, np.float64) - np.asarray(x, np.float64).mean(0)
        return self.C[self.dirs.query(u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12), 1, workers=-1)[1]]


# The bunny's ears' base height in its normalised frame (the mesh's frame, the same at every N and seed): the 40k
# target samples of seeds 98 / 100 / 102 split into two pieces (the ears) above 3.22 / 3.32 / 3.49 by a one-pitch slab
# scan (tmp/d160/d160_ears.py); 3.2 keeps the ears down to their base (0.9 % of the volume, 4 % of the surface).
BUNNY_EAR_BASE = 3.2
# The bunny's head direction (horizontal): from the sample's centroid to the centroid of the slab a tenth of its height
# deep below the ears' base, (-0.71, 0.70) / (-0.76, 0.66) / (-0.81, 0.59) on those samples.
BUNNY_HEAD = (-0.76, 0.0, 0.65)


def scene_params(scene: str, src: np.ndarray, tgt: np.ndarray) -> dict:
    """The scene's numbers, from the source and the target samples."""
    if scene not in SCENES:
        raise ValueError(f"unknown appearance scene {scene!r} (one of {SCENES})")
    ys, yt = src[:, 1].astype(np.float64), tgt[:, 1].astype(np.float64)
    p = {"scene": scene, "white": WHITE, "red": RED}
    if scene == "white":
        return p
    p["src_y"] = float(np.quantile(ys, {"ears": 0.95, "stripes": 2.0 / 3.0}.get(scene, 0.5)))
    if scene == "half":
        p["tgt_y"] = float(np.quantile(yt, 0.5))
    elif scene == "band":
        p["tgt_y"] = float(np.quantile(yt, 0.65))
    elif scene == "ears":
        p["tgt_y"] = BUNNY_EAR_BASE
    elif scene == "turn":
        h = np.asarray(BUNNY_HEAD) / np.linalg.norm(BUNNY_HEAD)
        c = tgt.mean(0).astype(np.float64)
        p["head_dir"], p["centre"] = h.tolist(), c.tolist()
        p["tgt_h"] = float(np.median((tgt - c) @ h))
    elif scene == "stripes":
        az = np.radians(FRONT_AZ)
        e_h = np.array([np.cos(az), 0.0, -np.sin(az)])
        s = tgt.astype(np.float64) @ e_h
        p["axis"], p["tgt_s"] = e_h.tolist(), [float(np.quantile(s, 1.0 / 3.0)), float(np.quantile(s, 2.0 / 3.0))]
    return p


def source_red(p: dict, x: np.ndarray) -> np.ndarray:
    """The source paint: True where red."""
    return np.asarray(x)[:, 1] >= p["src_y"]


def target_red(p: dict, x: np.ndarray) -> np.ndarray:
    """The target paint at any points (the samples, a state's discs): True where red."""
    x = np.asarray(x, np.float64)
    if p["scene"] == "turn":
        return (x - np.asarray(p["centre"])) @ np.asarray(p["head_dir"]) >= p["tgt_h"]
    if p["scene"] == "stripes":
        s = x @ np.asarray(p["axis"])
        return (s >= p["tgt_s"][0]) & (s < p["tgt_s"][1])
    return x[:, 1] >= p["tgt_y"]


def paint(red: np.ndarray, p: dict) -> np.ndarray:
    return np.where(np.asarray(red)[:, None], np.asarray(p["red"], np.float32), np.asarray(p["white"], np.float32))


def make_appearance(scene: str, src: np.ndarray, tgt: np.ndarray, draws=None, texture: TextureMap | None = None) -> Appearance:
    """The scene's albedos for a run: the source sample's (the body's, by index), the target sample's and every
    further target draw's. D173 (scene "texture", `texture` a TextureMap): the target and its draws take the image's
    colour at their points; the source takes it projected radially (TextureMap.radial), so each source particle carries
    one of the texture's colours, laid out by a map the physics objective never reads."""
    draws = list(draws or [])
    if scene == "texture":
        p = {"scene": "texture", "obj": getattr(texture, "obj", ""), "img": getattr(texture, "img", "")}
        return Appearance(scene, texture.radial(src), texture.at(tgt), [texture.at(d) for d in draws], p,
                          "the texture's colours: the source projected radially, the target looked up", tex=texture)
    p = scene_params(scene, src, tgt)
    if scene == "white":
        one = lambda x: np.ones((len(x), 1), np.float32)  # noqa: E731
        return Appearance(scene, one(src), one(tgt), [one(d) for d in draws], p, "one channel of ones (the reduction)")
    a_src = paint(source_red(p, src), p)
    a_tgt = paint(target_red(p, tgt), p)
    a_draws = [paint(target_red(p, d), p) for d in draws]
    note = (f"source red share {float(source_red(p, src).mean()):.4f}, target red share "
            f"{float(target_red(p, tgt).mean()):.4f}; " + ", ".join(f"{k} {v}" for k, v in p.items()
                                                                      if k not in ("scene", "white", "red")))
    return Appearance(scene, a_src, a_tgt, a_draws, p, note)
