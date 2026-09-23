"""Standard 3DGS binary .ply writer (no external deps) + frame->ply helper.

Field order/semantics match graphdeco GaussianModel.save_ply so any web splat
viewer loads it: scale stored as log, opacity as logit, rot as WXYZ, color as SH DC.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..render.covariance import cov_from_F, decompose_cov, sigma0_from_nn

C0 = 0.28209479177387814

_FIELDS = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
           + ["scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def write_splat_ply(path, xyz, scales, quats_wxyz,
                    color=(0.27, 0.51, 0.71), opacity=0.95):
    """Write a 3DGS .ply. scales linear (stored as log), quats WXYZ."""
    xyz = np.ascontiguousarray(xyz, np.float32)
    N = len(xyz)
    color = np.asarray(color, np.float32)
    if color.ndim == 1:
        color = np.tile(color, (N, 1))
    f_dc = ((color - 0.5) / C0).astype(np.float32)
    normals = np.zeros((N, 3), np.float32)
    log_scale = np.log(np.clip(scales, 1e-8, None)).astype(np.float32)
    op = np.full((N, 1), _logit(opacity), np.float32) if np.isscalar(opacity) \
        else _logit(np.asarray(opacity, np.float32)).reshape(N, 1)
    rot = np.ascontiguousarray(quats_wxyz, np.float32)
    data = np.concatenate([xyz, normals, f_dc, op, log_scale, rot], axis=1).astype("<f4")

    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {N}\n"
    header += "".join(f"property float {f}\n" for f in _FIELDS)
    header += "end_header\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(data.tobytes())


def frame_to_ply(path, x, F=None, sigma0=None, sigma0_scale=0.7,
                 color=(0.27, 0.51, 0.71), opacity=0.95):
    """Export one frame (positions x [N,3], optional F [N,3,3]) to a 3DGS .ply."""
    x = np.ascontiguousarray(x, np.float32)
    N = len(x)
    if sigma0 is None:
        sigma0 = sigma0_from_nn(x, sigma0_scale)
    if F is None:
        F = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
    cov = cov_from_F(F, sigma0)
    scales, quats = decompose_cov(cov)
    write_splat_ply(path, x, scales, quats, color=color, opacity=opacity)
    return sigma0
