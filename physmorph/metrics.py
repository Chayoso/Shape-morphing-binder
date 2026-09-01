"""Metrics for the v2 gates (docs/pipeline_v2.md §5). RAW simulation state only —
the renderer is never consumed (AGENTS.md rule 3), and — after the adversarial round —
NO metric shares an operator with any loss:

  * sil_iou / hole_frac use a BINARY 3x3-footprint point splat (numpy, the quicklook.py
    construction), not the soft CIC alpha the loss optimises — compaction/density games
    that raise the soft alpha do not move these numbers;
  * every projected quantity uses ONE FIXED extent derived from the TARGET, shared across
    arms and frames — the per-call autoscale let a single ejecta particle shrink the body
    and close holes (verified adversarially: one stray flipped hole_frac 7.7%→4.4%);
  * jitter excludes runner-held (duplicated) frames — measuring the padding, not the
    physics, made gate G3 unfailable.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .pipeline.render_loss import make_views


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance."""
    da = cKDTree(b).query(a, k=1, workers=-1)[0]
    db = cKDTree(a).query(b, k=1, workers=-1)[0]
    return float(da.mean() + db.mean())


def target_extent(tgt: np.ndarray, pad: float = 1.15) -> float:
    """The ONE shared projection extent: pad * max particle radius of the TARGET.
    |x·u| <= ||x||_2 for unit u, so every view of the target fits at any (theta, phi)."""
    return float(np.linalg.norm(np.ascontiguousarray(tgt, np.float32), axis=1).max()) * pad


def _splat_body(x, res, theta, phi, extent):
    """Binary body mask: orthographic 3x3-footprint point splat at a FIXED extent.
    Basis matches losses.silhouette._project (right, up as functions of theta/phi)."""
    right = np.array([np.cos(theta), 0.0, -np.sin(theta)], np.float32)
    up = np.array([-np.sin(phi) * np.sin(theta), np.cos(phi),
                   -np.sin(phi) * np.cos(theta)], np.float32)
    p = np.stack([x @ right, x @ up], 1)
    rel = (p + extent) / (2 * extent) * res
    ij = np.floor(rel).astype(np.int64)
    ok = (ij >= 0).all(1) & (ij < res).all(1)
    ij = ij[ok]
    cov = np.zeros((res, res), np.float32)
    flat = cov.reshape(-1)
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            i2 = np.clip(ij[:, 0] + ox, 0, res - 1)
            j2 = np.clip(ij[:, 1] + oy, 0, res - 1)
            np.add.at(flat, i2 * res + j2, 1.0)
    return cov > 0


def sil_iou(x, tgt, extent=None, n_azim=8, elevs=(0.0, 0.5, -0.5), res=128) -> float:
    """Mean multi-view IoU of BINARY splat bodies. Independent of the soft-alpha loss
    operator (no CIC weights, no 1-exp saturation, no threshold-on-density)."""
    x = np.ascontiguousarray(x, np.float32)
    tgt = np.ascontiguousarray(tgt, np.float32)
    e = target_extent(tgt) if extent is None else extent
    ious = []
    for th, phi in make_views(n_azim, elevs):
        a = _splat_body(x, res, th, phi, e)
        b = _splat_body(tgt, res, th, phi, e)
        u = (a | b).sum()
        ious.append((a & b).sum() / u if u else 1.0)
    return float(np.mean(ious))


def hole_frac(x, extent, res=160, views=((0.6, 0.18), (2.2, 0.18))) -> float:
    """Mean over views of (filled-silhouette minus body) / filled — background visible
    inside the body. extent is REQUIRED (pass target_extent(tgt)) so the number is
    commensurable across frames, arms and with its own threshold."""
    x = np.ascontiguousarray(x, np.float32)
    out = []
    for az, el in views:
        body = _splat_body(x, res, az, el, extent)
        filled = ndimage.binary_fill_holes(body)
        n = filled.sum()
        out.append(float((filled & ~body).sum() / n) if n else 0.0)
    return float(np.mean(out))


def outside_frac(x, extent) -> float:
    """Fraction of particles beyond the (target-derived) extent box — ejecta telemetry
    for exactly the far field where the render loss has no pixels."""
    x = np.ascontiguousarray(x, np.float32)
    return float((np.abs(x) > extent).any(1).mean())


def jitter(frames, tail=10, n_held=0) -> dict:
    """Tail rest-stability over SIMULATED frames only (held/duplicated tail excluded):
    mean per-particle displacement per frame, absolute and relative to the final bbox
    diagonal (gate G3)."""
    end = len(frames) - int(n_held)
    if end < 2:
        return {"jitter_abs": 0.0, "jitter_rel": 0.0, "jitter_max_abs": 0.0}
    tail = min(tail, end - 1)
    ds = [float(np.linalg.norm(frames[i + 1] - frames[i], axis=1).mean())
          for i in range(end - 1 - tail, end - 1)]
    xf = frames[end - 1]
    diag = float(np.linalg.norm(xf.max(0) - xf.min(0))) + 1e-9
    return {"jitter_abs": float(np.mean(ds)), "jitter_rel": float(np.mean(ds) / diag),
            "jitter_max_abs": float(np.max(ds))}


def summarize(frames, tgt, F_frames=None, n_held=0, tail=10) -> dict:
    """All gate metrics for one arm. frames: list of (N,3); tgt: (M,3)."""
    tgt = np.ascontiguousarray(tgt, np.float32)
    xf = np.ascontiguousarray(frames[-1], np.float32)
    e = target_extent(tgt)
    out = {"chamfer": chamfer(xf, tgt), "sil_iou": sil_iou(xf, tgt, extent=e),
           "hole_frac": hole_frac(xf, e), "hole_frac_tgt": hole_frac(tgt, e),
           "outside_frac": outside_frac(xf, e), "extent": e,
           "bbox_diag": float(np.linalg.norm(xf.max(0) - xf.min(0))),
           "frames": len(frames), "n_held": int(n_held)}
    out.update(jitter(frames, tail, n_held))
    if F_frames is not None and len(F_frames):
        out["detF_min"] = min(float(np.linalg.det(F).min()) for F in F_frames)
    return out
