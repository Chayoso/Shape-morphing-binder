"""Per-asset up-axis (assets/orientation.json): the OBJ collection mixes z-up and y-up meshes,
and the renderers put +y up. The rotation is applied ONCE, at load time (load_normalized), and
the archive records it as `orient` so that renderers can rotate older, un-oriented archives
at render time instead (the morph itself is rotation-equivariant: no gravity, no floor).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

_TABLE = None
_ROT = {
    "id": np.eye(3),
    "x-90": np.array([[1, 0, 0], [0, math.cos(-math.pi / 2), -math.sin(-math.pi / 2)], [0, math.sin(-math.pi / 2), math.cos(-math.pi / 2)]]),
    "x+90": np.array([[1, 0, 0], [0, math.cos(math.pi / 2), -math.sin(math.pi / 2)], [0, math.sin(math.pi / 2), math.cos(math.pi / 2)]]),
    "z-90": np.array([[math.cos(-math.pi / 2), -math.sin(-math.pi / 2), 0], [math.sin(-math.pi / 2), math.cos(-math.pi / 2), 0], [0, 0, 1]]),
    "z+90": np.array([[math.cos(math.pi / 2), -math.sin(math.pi / 2), 0], [math.sin(math.pi / 2), math.cos(math.pi / 2), 0], [0, 0, 1]]),
}


def table() -> dict:
    global _TABLE
    if _TABLE is None:
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "orientation.json")
        _TABLE = {}
        if os.path.exists(p):
            _TABLE = {k: v for k, v in json.load(open(p, encoding="utf-8")).items() if not k.startswith("_")}
    return _TABLE


def asset_name(path_or_name: str) -> str:
    """'assets/bunny.obj' -> 'bunny'; 'h150v7_bunny_render_full_dt_iso_nn.npz' -> 'bunny'."""
    b = os.path.basename(str(path_or_name))
    b = b.split(".")[0]
    for suf in ("_render_full_dt_iso_nn", "_render_full", "_physics_only"):
        if b.endswith(suf):
            b = b[: -len(suf)]
    if "_" in b and not os.path.exists(os.path.join("assets", b + ".obj")):
        b = b.split("_")[-1]                                  # <prefix>_<target>
    return b


def orient_name(path_or_name: str) -> str:
    return table().get(asset_name(path_or_name), "id")


def rotation(name: str) -> np.ndarray:
    return np.asarray(_ROT[name], np.float64)


def rotation_for(path_or_name: str) -> np.ndarray:
    return rotation(orient_name(path_or_name))


def apply(x: np.ndarray, name: str) -> np.ndarray:
    """Rotate points (…,3) by the named rotation (no-op for 'id')."""
    if name == "id":
        return x
    R = rotation(name).astype(x.dtype)
    return x @ R.T


def orient_archive(z, npz_path: str):
    """(frames, tgt, src, name) of a pipeline archive in y-up. Archives written after the
    loader applied the rotation carry `orient`; older ones are rotated here by the table."""
    frames = np.asarray(z["frames"]); tgt = np.asarray(z["tgt"]); src = np.asarray(z["src"]) if "src" in z.files else None
    if "orient" in z.files:
        return frames, tgt, src, str(z["orient"])
    name = orient_name(npz_path)
    if name == "id":
        return frames, tgt, src, "id"
    return apply(frames, name), apply(tgt, name), (apply(src, name) if src is not None else None), name
