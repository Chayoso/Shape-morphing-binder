"""The live viewer's packets: the binary /state protocol and the telemetry header.

Binary /state protocol: ``<u32 hlen><json hdr>`` followed by little-endian
float32 arrays.  v2 is ``x[N,3]``, ``cov6[N,6]`` then its optional diagnostics.
v3 inserts the objective-visible render representation immediately after those
parent arrays: ``render_x[R,3]``, ``render_cov6[R,6]``, ``render_opacity[R]``.
Physics/grid/gradient arrays remain parent-sized.  The padded JSON header and
payload are both 4-byte aligned for JS typed arrays.
"""
from __future__ import annotations

import json
import struct

import numpy as np

_TRI = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])      # upper-triangle index of a 3x3

_HDR_KEYS = ("loss", "d_vol", "d_render", "d_dt", "d_fill", "lambda", "kin",
             "g_raw_cos", "g_cos", "g_share", "g_phys_norm", "g_rend_norm",
             "render_work", "render_work_x", "render_work_F",
             "phys_work", "phys_work_x", "phys_work_F", "phys_work_v",
             "step_norm", "predicted_decrease",
             "v_absmax", "v_mean", "move", "grad_norm", "dfc_absmax",
             "accepted", "rejected", "outer_merit", "outer_gain", "reversal_cos",
             "outer_accepted", "outer_gate_latched", "gauss_condition_p95", "gauss_condition_max",
             "gauss_radius_over_spacing_p95", "gauss_radius_over_spacing_max",
             "floater_frac", "support_faded", "gradient_snapshot",
             "gradient_snapshot_commit")


def _json_value(value):
    """Convert scalar telemetry to strict JSON; non-finite values become ``null``."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _telemetry_header(seq: int, rec: dict) -> dict:
    hdr = {
        "seq": int(seq), "phase": rec.get("phase", "commit"),
        "sweep": rec.get("sweep"), "commit": int(rec.get("animation", -1)) + 1,
        "run": rec.get("run", 0), "E": rec.get("loss"),
        "lam": rec.get("lambda"), "gnorm": rec.get("grad_norm"),
        "dfc": rec.get("dfc_absmax"), "acc": rec.get("accepted"),
        "rej": rec.get("rejected"),
        "Jmin": rec.get("Jmin_traj", rec.get("Jmin")),
    }
    for key in _HDR_KEYS:
        hdr[key] = rec.get(key)
    return {key: _json_value(value) for key, value in hdr.items()}


def _array(name: str, value, shape: tuple[int | None, ...]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != len(shape) or any(want is not None and got != want
                                     for got, want in zip(arr.shape, shape)):
        expected = "x".join("*" if n is None else str(n) for n in shape)
        raise ValueError(f"{name} must have shape ({expected}), got {arr.shape}")
    return np.ascontiguousarray(arr, dtype="<f4")


def pack_state(seq: int, rec: dict, x: np.ndarray, cov: np.ndarray,
               nodes: np.ndarray | None = None, nodeq: np.ndarray | None = None,
               dt_pp: np.ndarray | None = None, particleq: np.ndarray | None = None,
               grad_phys: np.ndarray | None = None,
               grad_render: np.ndarray | None = None,
               render_weight: np.ndarray | None = None,
               render_x: np.ndarray | None = None,
               render_cov: np.ndarray | None = None,
               render_opacity: np.ndarray | None = None) -> bytes:
    x = _array("x", x, (None, 3))
    n = len(x)
    cov = _array("cov", cov, (n, 3, 3))
    if (nodes is None) != (nodeq is None):
        raise ValueError("nodes and nodeq must either both be present or both be absent")
    if nodes is not None:
        nodes = _array("nodes", nodes, (None, 3))
        nodeq = _array("nodeq", nodeq, (len(nodes), None))
    if dt_pp is not None:
        dt_pp = _array("dt_pp", dt_pp, (n,))
    if particleq is not None:
        particleq = _array("particleq", particleq, (n, None))
    if grad_phys is not None:
        grad_phys = _array("grad_phys", grad_phys, (n, 3))
    if grad_render is not None:
        grad_render = _array("grad_render", grad_render, (n, 3))
    if render_weight is not None:
        render_weight = _array("render_weight", render_weight, (n,))
    render_values = (render_x, render_cov, render_opacity)
    if any(v is not None for v in render_values) and not all(v is not None
                                                              for v in render_values):
        raise ValueError("render_x, render_cov, and render_opacity must be provided together")
    if render_x is not None:
        render_x = _array("render_x", render_x, (None, 3))
        render_n = len(render_x)
        render_cov = _array("render_cov", render_cov, (render_n, 3, 3))
        render_opacity = _array("render_opacity", render_opacity, (render_n,))
    else:
        render_n = 0

    arrays = [x, cov[:, _TRI[0], _TRI[1]]]
    if render_x is not None:
        arrays += [render_x, render_cov[:, _TRI[0], _TRI[1]], render_opacity]
    arrays += ([] if nodes is None else [nodes, nodeq])
    arrays += [a for a in (dt_pp, particleq, grad_phys, grad_render, render_weight)
               if a is not None]
    hdr = _telemetry_header(seq, rec)
    hdr.update({
        "protocol": 3 if render_x is not None else 2, "n": n,
        "r": render_n, "render_primitives": render_x is not None,
        "a": 0 if nodes is None else int(len(nodes)),
        "q": 0 if nodeq is None else int(nodeq.shape[1]),
        "pq": 0 if particleq is None else int(particleq.shape[1]),
        "dt_pp": dt_pp is not None,
        "grad_phys": grad_phys is not None, "grad_render": grad_render is not None,
        "render_weight": render_weight is not None,
        "payload_floats": int(sum(a.size for a in arrays)),
        "nonfinite": {
            name: int(np.size(a) - np.isfinite(a).sum())
            for name, a in (("x", x), ("cov", cov), ("nodes", nodes),
                            ("nodeq", nodeq), ("dt", dt_pp),
                            ("particleq", particleq), ("grad_phys", grad_phys),
                            ("grad_render", grad_render),
                            ("render_weight", render_weight),
                            ("render_x", render_x), ("render_cov", render_cov),
                            ("render_opacity", render_opacity))
            if a is not None
        },
    })
    hj = json.dumps(hdr, allow_nan=False).encode("utf-8")
    hj += b" " * (-len(hj) % 4)
    return struct.pack("<I", len(hj)) + hj + b"".join(a.tobytes() for a in arrays)


def grid_fields(x, v, gmin, ldx, res, F=None):
    """CIC diagnostics: mean speed, mass, momentum, kinetic, J and strain."""
    x = _array("x", x, (None, 3))
    v = _array("v", v, (len(x), 3))
    if F is None:
        F = np.tile(np.eye(3, dtype=np.float32), (len(x), 1, 1))
    F = _array("F", F, (len(x), 3, 3))
    gmin = _array("gmin", gmin, (3,))
    if int(res) <= 0 or not np.isfinite(ldx) or float(ldx) <= 0:
        raise ValueError("res and ldx must be positive")
    valid = (np.isfinite(x).all(1) & np.isfinite(v).all(1)
             & np.isfinite(F).all(axis=(1, 2)))
    x, v, F = x[valid], v[valid], F[valid]
    rel = (x - gmin) / ldx
    base = np.floor(rel).astype(np.int64)
    frac = rel - base
    m = np.zeros(res ** 3, np.float64)
    mv = np.zeros(res ** 3, np.float64)
    mom = np.zeros((res ** 3, 3), np.float64)
    ke = np.zeros(res ** 3, np.float64)
    j_acc = np.zeros(res ** 3, np.float64)
    s_acc = np.zeros(res ** 3, np.float64)
    Jp = np.linalg.det(F)
    strainp = np.linalg.norm(np.swapaxes(F, 1, 2) @ F - np.eye(3), axis=(1, 2))
    sp = np.linalg.norm(v, axis=1)
    for ox in (0, 1):
        wx = frac[:, 0] if ox else 1 - frac[:, 0]
        for oy in (0, 1):
            wy = frac[:, 1] if oy else 1 - frac[:, 1]
            for oz in (0, 1):
                wz = frac[:, 2] if oz else 1 - frac[:, 2]
                w = wx * wy * wz
                ii = np.clip(base[:, 0] + ox, 0, res - 1)
                jj = np.clip(base[:, 1] + oy, 0, res - 1)
                kk = np.clip(base[:, 2] + oz, 0, res - 1)
                idx = (ii * res + jj) * res + kk
                n_cells = res ** 3                      # bincount == add.at, 50x faster
                m += np.bincount(idx, weights=w, minlength=n_cells)
                mv += np.bincount(idx, weights=w * sp, minlength=n_cells)
                for c in range(3):
                    mom[:, c] += np.bincount(idx, weights=w * v[:, c], minlength=n_cells)
                ke += np.bincount(idx, weights=w * 0.5 * sp * sp, minlength=n_cells)
                j_acc += np.bincount(idx, weights=w * Jp, minlength=n_cells)
                s_acc += np.bincount(idx, weights=w * strainp, minlength=n_cells)
    # Every positive-weight CIC node is real diagnostic data.  The old 0.5 cutoff
    # dropped all eight nodes of a particle centred in a cell (weight 0.125 each).
    act = np.nonzero(m > 1e-12)[0]
    i, j, k = act // (res * res), (act // res) % res, act % res
    nodes = np.stack([i, j, k], 1).astype(np.float32) * ldx + gmin
    nodeq = np.stack([mv[act] / m[act], m[act],
                      np.linalg.norm(mom[act], axis=1), ke[act] / m[act],
                      j_acc[act] / m[act], s_acc[act] / m[act]], 1).astype(np.float32)
    return nodes, nodeq


def particle_fields(v, F, dt_pp):
    """Raw-state particle quantities: speed, J, condition(F), target distance."""
    v = _array("v", v, (None, 3))
    F = _array("F", F, (len(v), 3, 3))
    dt_pp = _array("dt_pp", dt_pp, (len(v),))
    out = np.full((len(v), 4), np.nan, np.float32)
    good_v = np.isfinite(v).all(1)
    out[good_v, 0] = np.linalg.norm(v[good_v], axis=1)
    good_F = np.isfinite(F).all(axis=(1, 2))
    if good_F.any():
        Fg = F[good_F]
        sv = np.linalg.svd(Fg, compute_uv=False)
        out[good_F, 1] = np.linalg.det(Fg)
        out[good_F, 2] = sv[:, 0] / np.maximum(sv[:, -1], 1e-8)
    out[:, 3] = dt_pp
    return out

