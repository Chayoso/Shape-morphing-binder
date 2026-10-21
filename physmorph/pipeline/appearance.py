"""Appearance (D160): a fixed colour per particle from the source, colour pictures of the target.

Every term of the physics objective (the Sinkhorn transport, the surface proximity, the end drift, the released motion,
the cleanup) reads only the set of particle positions: exchanging two particles changes none of them. Which material
ends where, and so the colour layout of the end state, is a null space of the physics objective. A scene gives every
source particle a fixed albedo, carried by its index (advected with the material, never a control), and every point of
the target samples the albedo of the target's paint; the render's shading term then compares colour pictures
(render_loss.shaded_discs with albedo, target.build_target), and only the render can choose where each colour goes.

The scenes paint with two albedos, WHITE and RED, by fields of position fixed once from the samples' own quantiles (so
the volumes painted are exact shares of each sample); the same field paints every target draw. Source: isosphere; the
bunny-specific scenes read the bunny's ears and head from its target sample.
  white  one channel of ones on both sides: the term without colour, bit for bit (the reduction test)
  half   source red above its median height; target red above its median height (the transport's own layout, the
         control: the lambda = 0 twin is nearly right)
  band   source red above its median height; target red above its 65 % height quantile (the top 35 %): a short move
  ears   source red above its 95 % height quantile (the top 5 %); target red above the ears' base (bunny): a medium move
  turn   source red above its median height; target red on the head's side of the median plane across the head's
         horizontal direction (bunny): a long move
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WHITE = (1.0, 1.0, 1.0)
RED = (0.9, 0.25, 0.2)
SCENES = ("white", "half", "band", "ears", "turn")


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


def pitch_of(x: np.ndarray) -> float:
    """The volume sample's pitch, 0.708 x the median 8th-neighbour distance (target.build_target's)."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(x).query(x, 9)
    return 0.708 * float(np.median(d[:, 8]))


def _pieces(pts: np.ndarray, link: float) -> np.ndarray:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    if len(pts) < 2:
        return np.zeros(len(pts), int)
    pairs = cKDTree(pts).query_pairs(link, output_type="ndarray")
    return connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(pts),) * 2),
                                directed=False)[1]


def ear_base(tgt: np.ndarray, pitch: float) -> float:
    """The bunny's ears' base: scanning a one-pitch slab of the target sample down from two pitches below the top in
    half-pitch steps, the top of the first slab that is one piece (pieces linked within 1.5 pitches, a piece counted
    when it holds a tenth of the slab) after slabs of two pieces (the ears)."""
    y = tgt[:, 1]
    two = False
    for h in np.arange(y.max() - 2 * pitch, y.min(), -0.5 * pitch):
        sl = tgt[(y >= h) & (y < h + pitch)]
        if not len(sl):
            continue
        big = int((np.bincount(_pieces(sl, 1.5 * pitch)) >= 0.1 * len(sl)).sum())
        two = two or big >= 2
        if two and big == 1:
            return float(h + pitch)
    raise ValueError("ear_base: no slab of two pieces above a slab of one (not a bunny?)")


def head_direction(tgt: np.ndarray, y_ear: float) -> np.ndarray:
    """The bunny's head direction: from the target sample's centroid to the centroid of its slab a tenth of its height
    deep below the ears' base, horizontal, unit."""
    c = tgt.mean(0)
    H = float(tgt[:, 1].max() - tgt[:, 1].min())
    sl = tgt[(tgt[:, 1] < y_ear) & (tgt[:, 1] >= y_ear - 0.1 * H)]
    d = (sl.mean(0) - c).astype(np.float64)
    d[1] = 0.0
    return d / np.linalg.norm(d)


def scene_params(scene: str, src: np.ndarray, tgt: np.ndarray) -> dict:
    """The scene's numbers, from the source and the target samples."""
    if scene not in SCENES:
        raise ValueError(f"unknown appearance scene {scene!r} (one of {SCENES})")
    ys, yt = src[:, 1].astype(np.float64), tgt[:, 1].astype(np.float64)
    p = {"scene": scene, "white": WHITE, "red": RED}
    if scene == "white":
        return p
    p["src_y"] = float(np.quantile(ys, 0.95 if scene == "ears" else 0.5))
    if scene == "half":
        p["tgt_y"] = float(np.quantile(yt, 0.5))
    elif scene == "band":
        p["tgt_y"] = float(np.quantile(yt, 0.65))
    elif scene in ("ears", "turn"):
        a = pitch_of(tgt)
        p["pitch"], p["ear_base"] = a, ear_base(tgt, a)
        if scene == "ears":
            p["tgt_y"] = p["ear_base"]
        else:
            h = head_direction(tgt, p["ear_base"])
            c = tgt.mean(0).astype(np.float64)
            p["head_dir"], p["centre"] = h.tolist(), c.tolist()
            p["tgt_h"] = float(np.median((tgt - c) @ h))
    return p


def source_red(p: dict, x: np.ndarray) -> np.ndarray:
    """The source paint: True where red."""
    return np.asarray(x)[:, 1] >= p["src_y"]


def target_red(p: dict, x: np.ndarray) -> np.ndarray:
    """The target paint at any points (the samples, a state's discs): True where red."""
    x = np.asarray(x, np.float64)
    if p["scene"] == "turn":
        return (x - np.asarray(p["centre"])) @ np.asarray(p["head_dir"]) >= p["tgt_h"]
    return x[:, 1] >= p["tgt_y"]


def paint(red: np.ndarray, p: dict) -> np.ndarray:
    return np.where(np.asarray(red)[:, None], np.asarray(p["red"], np.float32), np.asarray(p["white"], np.float32))


def make_appearance(scene: str, src: np.ndarray, tgt: np.ndarray, draws=None) -> Appearance:
    """The scene's albedos for a run: the source sample's (the body's, by index), the target sample's and every
    further target draw's."""
    draws = list(draws or [])
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
