"""Gate metrics of a run, computed on the device from the archived frames. RAW simulation
state only, and no metric shares an operator with a loss:

  * sil_iou / hole_frac use a BINARY 3x3-footprint point splat, not the soft CIC alpha the
    loss optimises, so density games that raise the soft alpha do not move them;
  * every projected quantity uses ONE FIXED extent derived from the TARGET (a per-call
    autoscale lets a single ejected particle shrink the body and close holes);
  * jitter excludes held (duplicated) frames.
"""
from __future__ import annotations

import numpy as np
import torch

from . import gpu
from .pipeline.render_loss import make_views


def _t(x) -> torch.Tensor:
    return gpu.tensor(x)


def chamfer(a, b) -> float:
    """Symmetric mean nearest-neighbour distance."""
    a, b = _t(a), _t(b)
    da = gpu.KNN(b).query(a, 1)[0]
    db = gpu.KNN(a).query(b, 1)[0]
    return float(da.mean() + db.mean())


def target_extent(tgt, pad: float = 1.15) -> float:
    """The ONE shared projection extent: pad x the target's max particle radius
    (|x.u| <= |x| for a unit u, so every view of the target fits)."""
    return float(_t(tgt).norm(dim=1).max()) * pad


def _splat_body(x: torch.Tensor, res: int, theta: float, phi: float, extent: float) -> torch.Tensor:
    """Binary body mask: orthographic 3x3-footprint point splat at a FIXED extent (the basis
    of losses.silhouette._project)."""
    right = x.new_tensor([np.cos(theta), 0.0, -np.sin(theta)])
    up = x.new_tensor([-np.sin(phi) * np.sin(theta), np.cos(phi), -np.sin(phi) * np.cos(theta)])
    p = torch.stack([x @ right, x @ up], 1)
    ij = torch.floor((p + extent) / (2 * extent) * res).long()
    ij = ij[((ij >= 0) & (ij < res)).all(1)]
    flat = torch.zeros(res * res, device=x.device, dtype=torch.float64)
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            i2 = (ij[:, 0] + ox).clamp(0, res - 1)
            j2 = (ij[:, 1] + oy).clamp(0, res - 1)
            flat += torch.bincount(i2 * res + j2, minlength=res * res).double()
    return flat.reshape(res, res) > 0


def sil_iou(x, tgt, extent=None, n_azim=8, elevs=(0.0, 0.5, -0.5), res=128) -> float:
    """Mean multi-view IoU of BINARY splat bodies."""
    x, tgt = _t(x), _t(tgt)
    e = target_extent(tgt) if extent is None else extent
    ious = []
    for th, phi in make_views(n_azim, elevs):
        a = _splat_body(x, res, th, phi, e)
        b = _splat_body(tgt, res, th, phi, e)
        u = int((a | b).sum())
        ious.append(int((a & b).sum()) / u if u else 1.0)
    return float(np.mean(ious))


def hole_frac(x, extent, res=160, views=((0.6, 0.18), (2.2, 0.18))) -> float:
    """Mean over views of (filled silhouette minus body) / filled: background visible
    inside the body, at the target-derived extent."""
    x = _t(x)
    out = []
    for az, el in views:
        body = _splat_body(x, res, az, el, extent)
        filled = gpu.fill_holes(body)
        n = int(filled.sum())
        out.append(float(int((filled & ~body).sum()) / n) if n else 0.0)
    return float(np.mean(out))


def outside_frac(x, extent) -> float:
    """Fraction of particles beyond the target-derived extent box."""
    return float((_t(x).abs() > extent).any(1).float().mean())


def stray_frac(x, k: int = 8, factor: float = 3.0) -> float:
    """Fraction of ISOLATED particles (kNN distance > factor x median)."""
    from .render.knn_gpu import knn_self_torch
    d, _ = knn_self_torch(_t(x), k + 1)
    dk = d[:, -1]
    return float((dk > factor * dk.median()).float().mean())


def ejection_trajectory(frames, extent, samples: int = 120) -> dict:
    """Max outside_frac and stray_frac over `samples` evenly spaced frames."""
    idx = sorted(set(np.linspace(0, len(frames) - 1, samples).astype(int)))
    outs = [outside_frac(frames[i], extent) for i in idx]
    strays = [stray_frac(frames[i]) for i in idx]
    return {"outside_max": float(max(outs)), "stray_max": float(max(strays)), "stray_final": strays[-1]}


def jitter(frames, tail=10, n_held=0) -> dict:
    """Tail rest-stability over simulated frames: mean per-particle displacement per frame,
    absolute and relative to the final bbox diagonal."""
    end = len(frames) - int(n_held)
    if end < 2:
        return {"jitter_abs": 0.0, "jitter_rel": 0.0, "jitter_max_abs": 0.0}
    tail = min(tail, end - 1)
    ds = [float((_t(frames[i + 1]) - _t(frames[i])).norm(dim=1).mean())
          for i in range(end - 1 - tail, end - 1)]
    xf = _t(frames[end - 1])
    diag = float((xf.max(0).values - xf.min(0).values).norm()) + 1e-9
    return {"jitter_abs": float(np.mean(ds)), "jitter_rel": float(np.mean(ds) / diag),
            "jitter_max_abs": float(np.max(ds))}


def tgt_nn_metrics(x, tgt, k_med: float = 2.0) -> dict:
    """Distance to the nearest TARGET particle in units of the target's median spacing:
    the fraction beyond k_med spacings, beyond 4.5, and the tail."""
    x, tgt = _t(x), _t(tgt)
    tree = gpu.KNN(tgt)
    nn_t = gpu.median(tree.query(tgt, 2)[0][:, 1])
    d = tree.query(x, 1)[0][:, 0]
    out = d > k_med * nn_t
    return {"tgt_nn_spacing": nn_t, "out_nn_frac": float(out.double().mean()),
            "out_nn_far_frac": float((d > 4.5 * nn_t).double().mean()),
            "out_nn_mean": float(d[out].mean()) if bool(out.any()) else 0.0,
            "out_nn_p95": float(torch.quantile(d, 0.95)), "out_nn_max": float(d.max())}


def out_dt_frac(x, tgt, res: int = 160, cells: float = 2.0) -> float:
    """DIAGNOSTIC: fraction of particles farther than `cells` fine cells from the dilated
    target support (dead radius ~3.5 cells; tgt_nn_metrics is the honest endpoint metric)."""
    from .losses.volumetric import target_dt_grid, target_mass_grid
    x, tgt = _t(x), _t(tgt)
    extent = float(tgt.abs().max()) * 1.25
    dx = 3.0 * extent / res
    gmin = torch.tensor([-1.5 * extent] * 3, device=tgt.device)
    dt3 = target_dt_grid(target_mass_grid(tgt, torch.ones(len(tgt), device=tgt.device), gmin, dx,
                                          (res,) * 3), dx, (res,) * 3, clamp=2 * extent).reshape(res, res, res)
    idx = ((x - gmin) / dx).long().clamp(0, res - 1)
    return float((dt3[idx[:, 0], idx[:, 1], idx[:, 2]] > cells * dx).double().mean())


def summarize(frames, tgt, n_held=0, tail=10, detF_min=None) -> dict:
    """All gate metrics of one run. frames: (N,3) arrays; tgt: (M,3). detF_min: the
    minimum det F over the delivered windows (measured on the device during the run)."""
    xf, tgt_t = _t(frames[-1]), _t(tgt)
    e = target_extent(tgt_t)
    out = {"chamfer": chamfer(xf, tgt_t), "sil_iou": sil_iou(xf, tgt_t, extent=e),
           "hole_frac": hole_frac(xf, e), "hole_frac_tgt": hole_frac(tgt_t, e),
           "outside_frac": outside_frac(xf, e), "extent": e, "out_dt_frac": out_dt_frac(xf, tgt_t),
           **tgt_nn_metrics(xf, tgt_t),
           "bbox_diag": float((xf.max(0).values - xf.min(0).values).norm()),
           "frames": len(frames), "n_held": int(n_held)}
    out.update(jitter(frames, tail, n_held))
    out.update(ejection_trajectory(frames, e))
    if detF_min is not None:
        out["detF_min"] = float(detF_min)
    return out
