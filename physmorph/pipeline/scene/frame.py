"""Each body's own frame in the multi-body scene (D147): the Buckens (shape-matching) frame and the rigid velocity.

The frame of a body is its centroid c and the polar factor R of A = sum_p (x_p - c)(X_p - Xbar)^T, X the body's
source sample at launch in its target's orientation (Mueller et al. 2005; Canavin's Buckens frame, a function of the
configuration alone). Its objective is the frozen single-body objective read on y = R^T (x - c), so it does not change
under a rigid motion of the body, and its position gradient carries no net force or moment. R is differentiated
exactly: for A = R S, dR = R [w]x with w = (tr S I - S)^-1 vee(R^T dA - dA^T R) (McAdams et al. 2011, eq. 12), finite
for the sphere's A = s I where the SVD's own backward divides by sigma_i^2 - sigma_j^2.

The velocity terms read each particle's velocity less the body's least-squares rigid velocity v_c + w x r, w = I^-1 L
(Lu, Jin & Fedkiw 2016; the deformation's kinetic energy, Koenig): a body that flies, spins or tumbles at rest pays
nothing.
"""
from __future__ import annotations

import torch


def _vee(M: torch.Tensor) -> torch.Tensor:
    """(M32, M13, M21) of a 3x3 (the axial vector of a skew matrix)."""
    return torch.stack((M[2, 1], M[0, 2], M[1, 0]))


def _hat(w: torch.Tensor) -> torch.Tensor:
    z = torch.zeros((), dtype=w.dtype, device=w.device)
    return torch.stack((torch.stack((z, -w[2], w[1])), torch.stack((w[2], z, -w[0])),
                        torch.stack((-w[1], w[0], z))))


class PolarRotation(torch.autograd.Function):
    """R = polar(A) (det R = +1) with the closed-form backward A_bar = R [k]x, k = (tr S I - S)^-1 vee(R^T G - G^T R)."""

    @staticmethod
    def forward(ctx, A: torch.Tensor):
        Ad = A.double()
        U, s, Vh = torch.linalg.svd(Ad)
        d = torch.sign(torch.linalg.det(U @ Vh))
        D = torch.diag(torch.stack((torch.ones_like(d), torch.ones_like(d), d)))
        R = U @ D @ Vh
        S = R.T @ Ad
        ctx.save_for_backward(R, 0.5 * (S + S.T))
        return R.to(A.dtype)

    @staticmethod
    def backward(ctx, G: torch.Tensor):
        R, S = ctx.saved_tensors
        Gd = G.double()
        H = torch.trace(S) * torch.eye(3, dtype=S.dtype, device=S.device) - S
        k = torch.linalg.solve(H, _vee(R.T @ Gd - Gd.T @ R))
        return (R @ _hat(k)).to(G.dtype)


def polar_rotation(A: torch.Tensor) -> torch.Tensor:
    return PolarRotation.apply(A)


def body_frame(x: torch.Tensor, Xq: torch.Tensor):
    """(y, c, R): the body's positions in its own frame, y = (x - c) R (rows), its centroid and rotation. Xq: the
    reference X - Xbar (fixed, no gradient). Differentiable in x (R through the polar backward)."""
    c = x.mean(0)
    p = x - c
    A = p.T @ Xq
    R = polar_rotation(A)
    return p @ R, c, R


def rigid_velocity(x: torch.Tensor, v: torch.Tensor):
    """(v_c, w): the body's least-squares rigid velocity, v_c the mean velocity and w = I^-1 sum r x v about the
    centroid (equal masses)."""
    c = x.mean(0)
    r = x - c
    vc = v.mean(0)
    L = torch.linalg.cross(r, v - vc, dim=1).sum(0)
    rr = (r * r).sum(1)
    eye = torch.eye(3, dtype=x.dtype, device=x.device)
    Iner = rr.sum() * eye - r.T @ r
    w = torch.linalg.solve(Iner.double(), L.double()).to(x.dtype)
    return vc, w


def deformation_velocity(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """v - (v_c + w x r): what remains of each particle's velocity once the body's rigid motion is taken out."""
    vc, w = rigid_velocity(x, v)
    r = x - x.mean(0)
    return v - vc - torch.linalg.cross(w.expand_as(r), r, dim=1)
