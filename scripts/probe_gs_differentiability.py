"""Differentiability probe for a grid-level Gauss-Seidel (VBD-style) solve.

Question (pre-build check): if each commit becomes a quasi-static GRID solve
  u* = argmin_u E(u; th) ,  E = 0.5 e^th u^T K u - f^T u + w_R * R(x0 + W u)
(K = elasticity proxy on grid nodes, W = CIC particle-from-grid interpolation,
R = asymmetric soft-silhouette render energy), can we still get dL/dth for an
outer objective L(u*) — i.e. does the material/system-ID channel survive?

Two candidate gradients, both checked against central finite differences:
  UNROLLED  backprop through the actual colored-GS sweep map (tape through solver)
  IFT       adjoint at the fixed point: solve H lam = dL/du*, dL/dth = -lam^T d(gradE)/dth
            (H = full Hessian at u*; the machinery DiffPD uses for Projective Dynamics)

Toy scale (CPU, float64, seconds): grid 5^3 nodes x 3 dof, 300 particles, 2-color GS
with per-node Jacobi-block preconditioning. This is a numerical feasibility check,
not a performance test.
"""
from __future__ import annotations

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

G = 5                      # grid nodes per axis
NG = G ** 3
NP = 300
RES = 24                   # silhouette image res
K_SIL = 1.5
W_R = 3.0                  # render energy weight


def build_problem():
    # grid node coords in [0,4]^3, spacing 1
    idx = torch.stack(torch.meshgrid(*[torch.arange(G)] * 3, indexing="ij"), -1).reshape(-1, 3)
    # graph Laplacian over 6-neighbourhood (elasticity proxy), tiled to 3 dof
    A = torch.zeros(NG, NG)
    for a in range(NG):
        for b in range(a + 1, NG):
            if (idx[a] - idx[b]).abs().sum() == 1:
                A[a, b] = A[b, a] = -1.0
    A += torch.diag(-A.sum(1) + 0.1)                     # SPD (small mass term)
    K = torch.kron(A, torch.eye(3))                      # (3NG, 3NG)
    # particles + CIC weights from the 8 surrounding nodes
    xp = torch.rand(NP, 3) * (G - 1)
    Wm = torch.zeros(NP, NG)
    base = xp.floor().long().clamp(max=G - 2)
    frac = xp - base.double()
    for ox in (0, 1):
        for oy in (0, 1):
            for oz in (0, 1):
                w = ((frac[:, 0] if ox else 1 - frac[:, 0])
                     * (frac[:, 1] if oy else 1 - frac[:, 1])
                     * (frac[:, 2] if oz else 1 - frac[:, 2]))
                n = (base[:, 0] + ox) * G * G + (base[:, 1] + oy) * G + (base[:, 2] + oz)
                Wm[torch.arange(NP), n] += w
    f = torch.randn(3 * NG) * 0.05                       # small external load
    # target silhouette: particles shifted/stretched
    x_t = (xp - xp.mean(0)) * torch.tensor([1.15, 0.9, 1.0]) + xp.mean(0) + 0.15
    return K, Wm, xp, f, x_t


def soft_sil(x, extent=6.0):
    """Differentiable CIC coverage image (RES,RES), project on xy."""
    rel = (x[:, :2] + 1.0) / extent * RES
    b = rel.floor()
    fr = rel - b
    img = torch.zeros(RES * RES, dtype=x.dtype)
    for ox in (0, 1):
        wx = fr[:, 0] if ox else 1 - fr[:, 0]
        for oy in (0, 1):
            wy = fr[:, 1] if oy else 1 - fr[:, 1]
            ii = (b[:, 0] + ox).clamp(0, RES - 1)
            jj = (b[:, 1] + oy).clamp(0, RES - 1)
            img = img.index_add(0, (ii * RES + jj).long(), wx * wy)
    return 1.0 - torch.exp(-K_SIL * img.reshape(RES, RES))


def R_energy(x, a_t):
    a = soft_sil(x)
    deficit = torch.clamp(a_t - a, min=0)
    excess = torch.clamp(a - a_t, min=0)
    return (2.0 * deficit.pow(2) + excess.pow(2)).mean()


def E_total(u, th, K, Wm, xp, f, a_t):
    x = xp + (Wm @ u.reshape(NG, 3))
    return 0.5 * torch.exp(th) * u @ (K @ u) - f @ u + W_R * R_energy(x, a_t)


def gs_solve(th, K, Wm, xp, f, a_t, sweeps, step=0.9, track_grad=False):
    """2-color damped block-GS with per-node Jacobi 3x3 preconditioning.
    Differentiable end-to-end when track_grad (the UNROLLED path)."""
    idx = torch.stack(torch.meshgrid(*[torch.arange(G)] * 3, indexing="ij"), -1).reshape(-1, 3)
    colors = [(idx.sum(1) % 2 == c) for c in (0, 1)]
    masks = [c.repeat_interleave(3).double() for c in colors]
    Kdiag = torch.stack([K[3 * g:3 * g + 3, 3 * g:3 * g + 3] for g in range(NG)])  # (NG,3,3)
    u = torch.zeros(3 * NG, requires_grad=track_grad)
    for _ in range(sweeps):
        for m in masks:
            if track_grad:
                g = torch.autograd.grad(E_total(u, th, K, Wm, xp, f, a_t), u,
                                        create_graph=True)[0]
            else:
                ur = u.detach().requires_grad_(True)
                g = torch.autograd.grad(E_total(ur, th, K, Wm, xp, f, a_t), ur)[0]
            P = torch.exp(th) * Kdiag + 0.1 * torch.eye(3)         # node-block precond
            d = torch.linalg.solve(P, g.reshape(NG, 3).unsqueeze(-1)).squeeze(-1).reshape(-1)
            u = u - step * m * d
    return u


def outer_L(u, x_t, Wm, xp):
    x = xp + (Wm @ u.reshape(NG, 3))
    return (x - x_t).pow(2).sum()


def main():
    K, Wm, xp, f, x_t = build_problem()
    a_t = soft_sil(x_t).detach()
    th0 = torch.tensor(0.3)

    # reference: central finite difference through a DEEP solve (well converged)
    eps = 1e-5
    Ls = []
    for s in (+eps, -eps):
        u = gs_solve(th0 + s, K, Wm, xp, f, a_t, sweeps=400)
        Ls.append(outer_L(u.detach(), x_t, Wm, xp).item())
    fd = (Ls[0] - Ls[1]) / (2 * eps)
    print(f"[probe] FD reference dL/dth = {fd:.6f}")

    for sweeps in (10, 40, 160):
        # UNROLLED: tape through the GS map itself
        th = th0.clone().requires_grad_(True)
        u = gs_solve(th, K, Wm, xp, f, a_t, sweeps=sweeps, track_grad=True)
        L = outer_L(u, x_t, Wm, xp)
        (g_unroll,) = torch.autograd.grad(L, th)
        # IFT at the SAME point: H lam = dL/du ;  dL/dth = -lam^T d(gradE)/dth
        ud = u.detach().requires_grad_(True)
        thd = th.detach().requires_grad_(True)
        gE = torch.autograd.grad(E_total(ud, thd, K, Wm, xp, f, a_t), ud, create_graph=True)[0]
        H = torch.autograd.functional.hessian(
            lambda v: E_total(v, thd.detach(), K, Wm, xp, f, a_t), ud.detach())
        dLdu = torch.autograd.grad(outer_L(ud, x_t, Wm, xp), ud, retain_graph=True)[0]
        lam = torch.linalg.solve(H, dLdu)
        (dgdth,) = torch.autograd.grad(gE @ lam.detach(), thd)
        g_ift = -dgdth
        uu = u.detach().clone().requires_grad_(True)
        gn = float(torch.autograd.grad(
            E_total(uu, th.detach(), K, Wm, xp, f, a_t), uu)[0].norm())
        e_u = abs(float(g_unroll) - fd) / max(abs(fd), 1e-12)
        e_i = abs(float(g_ift) - fd) / max(abs(fd), 1e-12)
        print(f"[probe] sweeps={sweeps:4d}  |gradE|={gn:.2e}  "
              f"unrolled={float(g_unroll):.6f} (rel err {e_u:.1%})  "
              f"IFT={float(g_ift):.6f} (rel err {e_i:.1%})")


if __name__ == "__main__":
    main()
