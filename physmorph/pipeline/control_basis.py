"""Coarse control basis: dFc(t, p) = sum_k w_k(x0_p) * B(t) C[knot, node_k].

Why (docs/render_controls_physics.md §4): a per-particle, per-step control field has
9·N·T degrees of freedom; an image loss touches ~N_surface·pixels of them one at a
time with sub-pixel signal (the measured 1e3–1e4 norm gap to the mass term is loss
SCALE, but the per-entry render signal is also noise-level: the 12k finite-difference
check FAILED on single control entries while directional checks passed). The
render-guided physics literature (PhysDreamer, NeuMA, OmniPhysGS, PAC-NeRF) never
optimises a per-particle field — the image gradient is ACCUMULATED into a low-
dimensional physical field (a voxel/neural material field, a global velocity). This
module is that field for the control stress: trilinear nodes on a coarse grid over
the window's start positions, piecewise-linear in time over K knots. One node update
moves every particle in its support — the Gauss–Seidel “propagate to the neighbours”
that VBD-style sweeps would provide, realised by the basis instead of a solver
(§6). The basis is frozen per window (weights from x0), so the map C -> dFc is
LINEAR and exactly differentiable in torch before the Warp bridge.

Per-particle mode (grid=0) is the identity map — the legacy path is unchanged.
"""
from __future__ import annotations

import numpy as np
import torch


class ControlBasis:
    def __init__(self, x0: np.ndarray, T: int, grid: int, tknots: int,
                 device: str = "cuda", pad_frac: float = 0.05):
        x0 = np.ascontiguousarray(x0, np.float32)
        if x0.ndim != 2 or x0.shape[1] != 3:
            raise ValueError("x0 must have shape (N,3)")
        if not np.isfinite(x0).all():
            raise ValueError("x0 must be finite")
        self.N, self.T = int(x0.shape[0]), int(T)
        self.grid = int(grid)
        self.tknots = int(tknots) if tknots and tknots > 0 else self.T
        if self.tknots > self.T:
            raise ValueError(f"tknots ({self.tknots}) may not exceed T ({self.T})")
        if self.grid < 0:
            raise ValueError("grid must be >= 0 (0 = per-particle control)")
        self.device = device
        if self.grid > 0:
            if self.grid < 2:
                raise ValueError("a control grid needs >= 2 nodes per axis")
            lo, hi = x0.min(0), x0.max(0)
            side = float((hi - lo).max())
            # uniform cubic cells over the bbox expanded by pad_frac per side, so every
            # G >= 2 is a valid grid (REFUTE F2: a one-CELL margin made G=2,3 degenerate —
            # h ~ 4e6, every particle on one node — and the projection test passed
            # vacuously on that grid)
            h = side * (1.0 + 2.0 * pad_frac) / (self.grid - 1)
            centre = 0.5 * (lo + hi)
            origin = centre - 0.5 * (self.grid - 1) * h
            rel = (x0 - origin) / h
            base = np.floor(rel).astype(np.int64)
            base = np.clip(base, 0, self.grid - 2)
            frac = np.clip(rel - base, 0.0, 1.0).astype(np.float32)
            G = self.grid
            idx = np.zeros((self.N, 8), np.int64)
            w = np.zeros((self.N, 8), np.float32)
            c = 0
            for ox in (0, 1):
                wx = frac[:, 0] if ox else 1.0 - frac[:, 0]
                for oy in (0, 1):
                    wy = frac[:, 1] if oy else 1.0 - frac[:, 1]
                    for oz in (0, 1):
                        wz = frac[:, 2] if oz else 1.0 - frac[:, 2]
                        idx[:, c] = ((base[:, 0] + ox) * G + (base[:, 1] + oy)) * G + (base[:, 2] + oz)
                        w[:, c] = wx * wy * wz
                        c += 1
            self.h, self.origin = float(h), origin.astype(np.float32)
            self.idx = torch.as_tensor(idx, device=device)
            self.w = torch.as_tensor(w, device=device)
            self.n_nodes = G ** 3
            # per-node support mass (sum of weights): nodes nobody touches have no
            # gradient and are reported, never optimised into (they stay at zero)
            self.support = torch.zeros(self.n_nodes, device=device).index_add_(
                0, self.idx.reshape(-1), self.w.reshape(-1))
        else:
            self.n_nodes = self.N
            self.h, self.origin, self.idx, self.w, self.support = 0.0, None, None, None, None
        # piecewise-linear time interpolation matrix (T, K)
        K = self.tknots
        Bt = np.zeros((self.T, K), np.float32)
        if K == 1:
            Bt[:, 0] = 1.0
        else:
            u = np.arange(self.T, dtype=np.float64) / max(self.T - 1, 1) * (K - 1)
            k0 = np.minimum(np.floor(u).astype(np.int64), K - 2)
            a = (u - k0).astype(np.float32)
            Bt[np.arange(self.T), k0] = 1.0 - a
            Bt[np.arange(self.T), k0 + 1] = a
        self.Bt = torch.as_tensor(Bt, device=device)

    # ---- shapes -------------------------------------------------------------
    @property
    def per_particle(self) -> bool:
        return self.grid == 0 and self.tknots == self.T

    def leaf_shape(self) -> tuple:
        return (self.tknots, self.n_nodes, 3, 3)

    def zeros(self) -> torch.Tensor:
        return torch.zeros(self.leaf_shape(), device=self.device, requires_grad=True)

    # ---- the linear map -----------------------------------------------------
    def expand(self, C: torch.Tensor) -> torch.Tensor:
        """(K, n_nodes, 3, 3) coefficients -> (T, N, 3, 3) per-particle, per-step dFc."""
        if tuple(C.shape) != self.leaf_shape():
            raise ValueError(f"coefficients must have shape {self.leaf_shape()}, got {tuple(C.shape)}")
        if self.per_particle:
            return C
        K = self.tknots
        if self.grid > 0:
            Cn = C.reshape(K, self.n_nodes, 9)
            gathered = Cn[:, self.idx]                          # (K, N, 8, 9)
            Cp = (gathered * self.w[None, :, :, None]).sum(2)   # (K, N, 9)
        else:
            Cp = C.reshape(K, self.N, 9)
        out = torch.einsum("tk,knj->tnj", self.Bt, Cp)          # (T, N, 9)
        return out.reshape(self.T, self.N, 3, 3)

    def _apply_W(self, Cn: torch.Tensor) -> torch.Tensor:
        """(K, n_nodes, 9) node values -> (K, N, 9) particle values (spatial part)."""
        return (Cn[:, self.idx] * self.w[None, :, :, None]).sum(2)

    def _apply_Wt(self, P: torch.Tensor) -> torch.Tensor:
        """(K, N, 9) particle values -> (K, n_nodes, 9) = W^T P."""
        out = torch.zeros(P.shape[0], self.n_nodes, 9, device=P.device, dtype=P.dtype)
        flat_idx = self.idx.reshape(-1)
        for k in range(P.shape[0]):
            out[k].index_add_(0, flat_idx, (P[k][:, None, :] * self.w[:, :, None]).reshape(-1, 9))
        return out

    def project(self, dfc: torch.Tensor, cg_iters: int = 30) -> torch.Tensor:
        """LEAST-SQUARES restriction of a (T,N,3,3) field onto the basis:
        argmin_C ||W C - d||^2 solved by Jacobi-preconditioned conjugate gradients on the
        normal equations (W^T W) C = W^T d (matrix-free: gather / index_add). Exact for a
        field in the range of the basis (REFUTE F2: the previous lumped D^-1 W^T
        restriction is a node-space SMOOTHER, W D^-1 W^T != I, and contracted a basis
        field by 38-61 % per application). Used to convert a warm start whose node grid
        moved with the window start positions."""
        dfc = dfc.reshape(self.T, self.N, 9)
        Bp = torch.linalg.pinv(self.Bt)                          # time: small dense pinv
        Ck = torch.einsum("kt,tnj->knj", Bp, dfc)                # (K, N, 9)
        if self.grid == 0:
            return Ck.reshape(self.leaf_shape())
        b = self._apply_Wt(Ck)
        diag = torch.zeros(self.n_nodes, device=dfc.device, dtype=dfc.dtype).index_add_(
            0, self.idx.reshape(-1), (self.w * self.w).reshape(-1)).clamp_min(1e-12)
        live = (self.support > 0).to(dfc.dtype)[None, :, None]
        Minv = (1.0 / diag)[None, :, None] * live

        def A(C):
            return self._apply_Wt(self._apply_W(C)) * live
        C = torch.zeros_like(b)
        r = b * live
        z = Minv * r
        p = z.clone()
        rz = (r * z).sum()
        for _ in range(cg_iters):
            if float(rz) <= 1e-30:
                break
            Ap = A(p)
            alpha = rz / (p * Ap).sum().clamp_min(1e-30)
            C = C + alpha * p
            r = r - alpha * Ap
            z = Minv * r
            rz_new = (r * z).sum()
            p = z + (rz_new / rz.clamp_min(1e-30)) * p
            rz = rz_new
        return C.reshape(self.leaf_shape())

    def describe(self) -> dict:
        return {"control_grid": self.grid, "control_tknots": self.tknots,
                "n_nodes": int(self.n_nodes),
                "n_dof": int(self.tknots * self.n_nodes * 9),
                "n_dof_per_particle": int(self.T * self.N * 9),
                "cell": self.h,
                "empty_nodes": (int((self.support <= 1e-12).sum())
                                if self.support is not None else 0)}
