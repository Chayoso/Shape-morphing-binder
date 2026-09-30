"""Deformation-gradient repair at window ends (on the device).

Only numerical pathologies are repaired, and both are counted: non-finite rows are reset
to the identity, and reflections (det F < 0) are turned into the nearest proper
deformation by flipping the last left singular vector (what corotated_R does in
constitutive.py). Singular values are never clamped: a silent projection would rewrite the
state every window without a counter.
"""
from __future__ import annotations

import torch


def repair_F(F: torch.Tensor):
    """(F_repaired (N,3,3), n_nonfinite_reset, n_reflection_flips). The input is returned
    unchanged (the same tensor) when nothing needs a repair."""
    F = F.reshape(-1, 3, 3)
    bad = ~torch.isfinite(F).all(dim=(1, 2))
    n_bad = int(bad.sum())
    Fw = F.clone() if n_bad else F
    if n_bad:
        Fw[bad] = torch.eye(3, device=F.device, dtype=F.dtype)
    flip = torch.linalg.det(Fw) < 0
    n_flip = int(flip.sum())
    if n_bad == 0 and n_flip == 0:
        return F, 0, 0
    out = Fw.clone()
    if n_flip:
        U, S, Vh = torch.linalg.svd(Fw[flip])
        U = U.clone()
        U[:, :, -1] *= -1.0
        out[flip] = U @ torch.diag_embed(S) @ Vh
    return out, n_bad, n_flip
