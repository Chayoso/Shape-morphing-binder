# -*- coding: utf-8 -*-
"""
runtime_surface_hybrid_faiss.py
================================

HYBRID FAISS + FULLY DIFFERENTIABLE runtime surface upsampler.

This module keeps **all function names and signatures** stable while applying
a set of patches for smoothness, differentiability and speed. Every change is
documented inline in English comments.

Key additions & fixes (non‑breaking):
1) HybridFAISSKNN
   - FAISS forward + differentiable reweighting backward
   - Cache invalidation when data moves (no stale indexes)
   - AMP stability: logits kept in FP32
   - Optional **soft‑radius** candidate pool (removes k-th neighbor discontinuity)

2) Surface quality
   - Soft normal orientation via tanh (no hard flips → smoother grads)
   - Optional **surface_smoother_diff** (tangent Laplacian + MLS)
   - Periodic MLS projection during equalization to kill interior drift

3) Sampling
   - Gumbel-Softmax with isolated generator (deterministic & differentiable)
   - Pure matrix mixing (Y@x, Y@normals, Y@spacing): no gather in the hot path

4) Robustness
   - Numerical clamps on exp/log to avoid NaN/Inf in AMP/FP16
   - Self-neighbor masking in density equalization

Author: CHAYO (Hybrid FAISS)
Date: 2025-10-16
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
from contextlib import nullcontext

import torch
import torch.nn.functional as F

# Optional FAISS for speed
try:
    import faiss
    import faiss.contrib.torch_utils  # noqa: F401
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False
    warnings.warn("FAISS not available - fallback to pure torch (slower)")


# ============================================================================
# Configuration (kept compatible)
# ============================================================================
def default_cfg() -> Dict:
    """Default configuration with hybrid FAISS + differentiability (compatible)."""
    return {
        "differentiable": True,
        "use_hybrid_faiss": True,
        "use_surface_detection": True,
        "k_surface": 36,
        "thr_percentile": 8.0,
        "ema_beta": 0.95,
        "hysteresis": 0.03,
        "soft_tau": 0.08,
        "M": 50_000,
        "surf_jitter_alpha": 0.6,
        "thickness": 0.00,
        "density_gamma": 2.5,
        "use_F_kernel": True,
        "k_F": 32,
        "h_mul": 1.5,
        "sigma0": 0.08,
        "ed": {
            "enabled": True,
            "num_nodes": 180,
            "node_knn": 8,
            "point_knn_nodes": 8,
            "lambda_lap": 1.0e-2,
        },
        "post_equalize": {
            "enabled": True,
            "iters": 8,
            "k": 32,
            "step": 0.45,
            "annealing": 0.9,         # step annealing rate: step *= annealing^iter
            "radius_mul": 1.2,
            "use_mls_projection": True,
            "mls_iters": 2,
            "mls_step": 1.0,
        },
        # Speed/quality knobs
        "use_faiss_ivf": True,
        "use_amp": True,
        "sampling_tau": 0.2,
        "surface_power": 4.0,
        "knn_tau": 0.15,
        "ivf_nlist": 100,
        "ivf_nprobe": 10,
        "use_soft_radius": False,
        "soft_radius_candidates": 128,
        "smoother": {
            "enabled": False,
            "iters": 3,
            "k": 24,
            "step": 0.12,
            "lambda_normal": 0.15,   # fraction of normal component kept (0=only tangent move)
            "mls_every": 2           # do MLS projection every N iterations
        },
    }


# ============================================================================
# Utilities
# ============================================================================
def _ensure_torch(x, device='cuda', dtype=torch.float32):
    """Convert array-like to torch tensor on the given device/dtype."""
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.from_numpy(np.asarray(x)).to(device=device, dtype=dtype)

def _normalize(v, eps=1e-9):
    """L2-normalize last dimension, safe for zeros."""
    return v / (torch.norm(v, dim=-1, keepdim=True) + eps)


# ============================================================================
# HYBRID: FAISS search + differentiable reweighting (patched)
# ============================================================================
class HybridFAISSKNN:
    """
    Forward: FAISS gives k nearest neighbor *indices* quickly (no grads).
    Backward: we recompute distances on the gathered neighbors and build
              a softmax weight field (differentiable w.r.t. query & data).

    Patches:
    - Index cache invalidation when the data storage changes or periodically.
    - FP32 logits in AMP to avoid underflow/overflow in softmax.
    - Optional soft-radius: compute weights over a larger candidate pool Kc
      then pick top-k by *weight* (no hard boundary at the k-th distance).
    """
    def __init__(self, use_faiss: bool = True, use_ivf: bool = True,
                 tau: float = 0.15, nlist: int = 100, nprobe: int = 10,
                 use_soft_radius: bool = False, soft_radius_candidates: int = 128):
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.use_ivf = use_ivf
        self.tau = float(tau)
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self.use_soft_radius = bool(use_soft_radius)
        self.soft_radius_candidates = int(soft_radius_candidates)
        self._index_cache = {}
        self._epoch = 0  # manual invalidation counter

    def clear_cache(self):
        """Drop all FAISS indices (free memory)."""
        self._index_cache.clear()

    def invalidate_cache(self):
        """Increase epoch so next call rebuilds the index."""
        self._epoch += 1
        self._index_cache.clear()

    # -- public API (signature preserved) ----------------------------------
    def __call__(self, query: torch.Tensor, data: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = query.device
        k = int(min(k, data.shape[0]))

        if not (self.use_faiss and FAISS_AVAILABLE):
            return self._torch_soft_knn(query, data, k)

        if self.use_soft_radius:
            return self._hybrid_faiss_soft_radius(query, data, k)
        else:
            return self._hybrid_faiss_knn(query, data, k)

    # -- implementations ----------------------------------------------------
    def _build_index(self, data: torch.Tensor, D: int, nlist: int, nprobe: int, cache_key):
        """Create or fetch a FAISS index for this data snapshot."""
        if cache_key in self._index_cache:
            return self._index_cache[cache_key]

        # Clear stale entries and build a fresh index
        self._index_cache.clear()
        data_np = data.detach().cpu().float().numpy()

        if self.use_ivf:
            if data.is_cuda:
                res = faiss.StandardGpuResources()
                quantizer = faiss.GpuIndexFlatL2(res, D)
                index = faiss.GpuIndexIVFFlat(quantizer, D, nlist, faiss.METRIC_L2)
            else:
                quantizer = faiss.IndexFlatL2(D)
                index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_L2)

            if not index.is_trained:
                train_size = min(data_np.shape[0], 100_000)
                train_sel = np.random.choice(data_np.shape[0], train_size, replace=False) \
                            if data_np.shape[0] > train_size else np.arange(data_np.shape[0])
                index.train(data_np[train_sel])
            index.add(data_np)
            index.nprobe = nprobe
        else:
            if data.is_cuda:
                res = faiss.StandardGpuResources()
                index = faiss.GpuIndexFlatL2(res, D)
            else:
                index = faiss.IndexFlatL2(D)
            index.add(data_np)

        self._index_cache[cache_key] = index
        return index

    def _hybrid_faiss_knn(self, query: torch.Tensor, data: torch.Tensor, k: int):
        """Standard hybrid mode: exact k by FAISS, differentiable weights."""
        N, D = query.shape
        M = data.shape[0]
        nlist = min(self.nlist, max(4, M // 100))
        nprobe = min(self.nprobe, nlist)
        # Include data storage pointer and epoch in the cache key
        data_ptr = int(data.untyped_storage().data_ptr())
        cache_key = (M, D, nlist, data_ptr, self._epoch)

        index = self._build_index(data, D, nlist, nprobe, cache_key)

        # FAISS search (no grads)
        q_np = query.detach().cpu().float().numpy()
        d_np, i_np = index.search(q_np, k)
        distances = torch.from_numpy(d_np).to(query.device)
        indices = torch.from_numpy(i_np).to(query.device, dtype=torch.long)

        # Differentiable reweighting on the gathered neighbors
        if query.requires_grad or data.requires_grad:
            neigh = data[indices]                          # [N,k,D]
            qx = query.unsqueeze(1).float()                # [N,1,D] (FP32 logits)
            dist = torch.norm(qx - neigh.float(), dim=2)   # [N,k]
            logits = -dist / self.tau
            weights = F.softmax(logits, dim=1).to(query.dtype)
        else:
            weights = torch.exp(-distances / self.tau)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        return indices, weights

    def _hybrid_faiss_soft_radius(self, query: torch.Tensor, data: torch.Tensor, k: int):
        """
        Soft‑radius mode:
        1) Fetch a *larger* candidate pool of size Kc with FAISS.
        2) Build weights over ALL candidates (smooth attention field).
        3) Return the top‑k by weight (renormalized). Shapes stay [N,k].
        """
        N, D = query.shape
        M = data.shape[0]
        Kc = min(self.soft_radius_candidates, M)
        nlist = min(self.nlist, max(4, M // 100))
        nprobe = min(self.nprobe, nlist)
        data_ptr = int(data.untyped_storage().data_ptr())
        cache_key = (M, D, nlist, data_ptr, self._epoch)

        index = self._build_index(data, D, nlist, nprobe, cache_key)
        q_np = query.detach().cpu().float().numpy()
        d_np, i_np = index.search(q_np, Kc)
        distances = torch.from_numpy(d_np).to(query.device)
        idx_all = torch.from_numpy(i_np).to(query.device, dtype=torch.long)

        if query.requires_grad or data.requires_grad:
            neigh = data[idx_all]                          # [N,Kc,D]
            qx = query.unsqueeze(1).float()
            dist = torch.norm(qx - neigh.float(), dim=2)   # [N,Kc]
            logits = -dist / self.tau
            w_all = F.softmax(logits, dim=1)               # [N,Kc]
            topw, topj = torch.topk(w_all, k=min(k, Kc), dim=1)
            batch = torch.arange(N, device=query.device).unsqueeze(1).expand(-1, topj.shape[1])
            indices = idx_all[batch, topj]                 # [N,k]
            weights = topw / (topw.sum(dim=1, keepdim=True) + 1e-9)
            weights = weights.to(query.dtype)
        else:
            w_all = torch.exp(-distances / self.tau)
            w_all = w_all / (w_all.sum(dim=1, keepdim=True) + 1e-9)
            topw, topj = torch.topk(w_all, k=min(k, Kc), dim=1)
            batch = torch.arange(N, device=query.device).unsqueeze(1).expand(-1, topj.shape[1])
            indices = idx_all[batch, topj]
            weights = topw / (topw.sum(dim=1, keepdim=True) + 1e-9)
        return indices, weights

    def _torch_soft_knn(self, query: torch.Tensor, data: torch.Tensor, k: int):
        """Pure torch fallback: differentiable attention over distances."""
        D = torch.cdist(query, data, p=2)               # [N,M]
        logits = -D / self.tau
        attn = F.softmax(logits, dim=1)                 # [N,M]
        topw, topi = torch.topk(attn, k=k, dim=1)       # [N,k]
        weights = topw / (topw.sum(dim=1, keepdim=True) + 1e-9)
        return topi, weights


# ============================================================================
# Batched PCA (kept signature, softer normal orientation)
# ============================================================================
def batched_pca_surface_optimized(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted PCA per point. Returns (normals, surface_variance, local_spacing)."""
    neighbors = x[indices]                                       # [N,k,3]
    centroid = torch.einsum('nk,nkd->nd', weights, neighbors)    # [N,3]
    centered = neighbors - centroid.unsqueeze(1)                 # [N,k,3]

    sqrt_w = torch.sqrt(weights).unsqueeze(-1)                   # [N,k,1]
    weighted = centered * sqrt_w                                 # [N,k,3]

    cov = torch.einsum('nki,nkj->nij', weighted, weighted)       # [N,3,3]
    cov = cov / (weights.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-9)

    evals, evecs = torch.linalg.eigh(cov)                        # ascending
    evals = torch.clamp(evals, min=1e-12)
    surfvar = evals[:, 0] / (evals.sum(dim=1) + 1e-12)           # smallest / trace
    n_raw = evecs[:, :, 0]                                       # normal from min-eigenvec

    # Softer orientation: tanh(s * dot) instead of hard sign flip
    global_c = x.mean(dim=0)
    to_out = x - centroid
    mask = torch.norm(to_out, dim=1) < 1e-6
    to_out[mask] = x[mask] - global_c
    dot = torch.einsum('nd,nd->n', n_raw, to_out)
    sign = torch.tanh(10.0 * dot).unsqueeze(-1)                  # [-1,+1] smooth
    normals = _normalize(n_raw * sign)

    # Local spacing for adaptive jitter
    dists = torch.norm(neighbors - x.unsqueeze(1), dim=-1)       # [N,k]
    spacing = torch.einsum('nk,nk->n', dists, weights) / (weights.sum(dim=1) + 1e-9)
    return normals, surfvar, spacing


# ============================================================================
# Differentiable soft quantile (kept signature)
# ============================================================================
def soft_quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    """
    Piecewise-differentiable quantile approximation using linear interpolation
    between the two nearest sorted values.
    """
    N = x.shape[0]
    xs, _ = torch.sort(x)
    idx = torch.tensor(q * (N - 1), dtype=torch.float32, device=x.device)
    idx_low = torch.clamp(torch.floor(idx).long(), 0, N-1)
    idx_high = torch.clamp(idx_low + 1, 0, N-1)
    w = (idx - idx_low.float())
    return (1 - w) * xs[idx_low] + w * xs[idx_high]


# ============================================================================
# Surface detection (kept signature; uses soft quantile)
# ============================================================================
def compute_surface_mask_diff(
    x: torch.Tensor,
    knn: HybridFAISSKNN,
    k_surface: int,
    thr_percentile: float,
    ema_prev: Optional[float],
    ema_beta: float,
    hysteresis: float,
    soft_tau: float,
    surface_power: float,
    state_out: Dict
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """
    Classify "surface-like" seeds and produce a differentiable sampling
    distribution (surf_prob). Lower PCA variance ⇒ more surface-like.
    """
    idx, w = knn(x, x, k_surface)
    normals, surfvar, spacing = batched_pca_surface_optimized(x, idx, w)

    thr_now = soft_quantile(surfvar, thr_percentile / 100.0)
    ema_thr = float(thr_now) if ema_prev is None else float(ema_beta * ema_prev + (1 - ema_beta) * thr_now)
    band = hysteresis * max(ema_thr, 1e-6)
    thr_low, thr_high = ema_thr - band, ema_thr + band

    # Soft decision and power sharpening
    score = torch.sigmoid(-(surfvar - thr_high) / max(soft_tau, 1e-6))
    surf_prob = (score ** surface_power)
    surf_prob = surf_prob / (surf_prob.sum() + 1e-12)

    # Minimum floor avoids dead seeds during annealing
    min_p = surf_prob.max() * 1e-4
    surf_prob = torch.clamp(surf_prob, min=min_p)
    surf_prob = surf_prob / surf_prob.sum()

    state_out.update({"ema_thr": float(ema_thr), "thr_low": float(thr_low), "thr_high": float(thr_high)})
    return surf_prob, normals, spacing, float(thr_low), float(thr_high)


# ============================================================================
# Gumbel-Softmax one-hot with isolated RNG (kept signature)
# ============================================================================
def gumbel_softmax_onehot(
    probs: torch.Tensor,
    M: int,
    tau: float = 0.2,
    hard: bool = True,
    seed: int = 0,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Draw M samples from a categorical defined by `probs` using Gumbel-Softmax.
    The returned matrix Y has shape [M,N] and behaves like a one-hot selection
    in the forward pass, while keeping gradients in the backward pass.
    """
    N = probs.shape[0]
    device = probs.device
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(seed)

    u = torch.rand(M, N, generator=generator, device=device)
    g = -torch.log(-torch.log(u + 1e-9) + 1e-9)
    logits = (probs.clamp_min(1e-12).log().unsqueeze(0) + g) / max(tau, 1e-6)
    y_soft = F.softmax(logits, dim=1)              # [M,N]

    if hard:
        idx = y_soft.argmax(dim=1)
        y_hard = F.one_hot(idx, num_classes=N).float()
        return y_hard - y_soft.detach() + y_soft   # straight‑through
    return y_soft


# ============================================================================
# Surface point sampling (kept signature)
# ============================================================================
def sample_surface_points_diff(
    x: torch.Tensor,
    normals: torch.Tensor,
    spacing: torch.Tensor,
    probs: torch.Tensor,
    M: int,
    alpha: float,
    thickness: float,
    density_gamma: float,
    seed: int = 0,
    tau: float = 0.2,
    generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fully differentiable sampling of M points on the surface.
    Sampling uses a soft one‑hot matrix Y so that `mu = Y@x` mixes anchors
    continuously (no hard gather in the hot path).
    """
    device, dtype = x.device, x.dtype
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(seed)

    # Importance sampling: prefer thin/under‑sampled areas
    w_import = (probs ** 1.5) * (1.0 / (spacing ** density_gamma + 1e-9))
    w_import = w_import / (w_import.sum() + 1e-12)

    Y = gumbel_softmax_onehot(w_import, M=M, tau=tau, hard=True, seed=seed, generator=generator)  # [M,N]

    mu_anchors = Y @ x                 # [M,3]
    n = _normalize(Y @ normals)        # [M,3]
    h = (Y @ spacing.unsqueeze(1)).squeeze(1)  # [M]

    # Build orthonormal tangent basis
    a = torch.tensor([1., 0., 0.], device=device, dtype=dtype).expand(M, 3).clone()
    col = torch.abs(torch.einsum('md,md->m', n, a)) > 0.9
    a[col] = torch.tensor([0., 1., 0.], device=device, dtype=dtype)
    t1 = _normalize(a - torch.einsum('md,md->m', a, n).unsqueeze(-1) * n)
    t2 = _normalize(torch.cross(n, t1, dim=1))

    # Stochastic tangent jitter (isolated RNG → deterministic under the same seed)
    U = torch.randn(M, 1, generator=generator, device=device, dtype=dtype).clamp(-3, 3)
    V = torch.randn(M, 1, generator=generator, device=device, dtype=dtype).clamp(-3, 3)
    Z = (torch.rand(M, 1, generator=generator, device=device, dtype=dtype) * 2.0 - 1.0)

    theta = torch.rand(M, 1, generator=generator, device=device, dtype=dtype) * 2 * np.pi
    c, s = torch.cos(theta), torch.sin(theta)
    U_rot = U * c - V * s
    V_rot = U * s + V * c

    alpha_noise = 0.4 + torch.rand(M, 1, generator=generator, device=device, dtype=dtype) * 1.2
    h_scale = torch.clamp(h / (h.mean() + 1e-9), 0.3, 2.5).unsqueeze(-1)
    alpha_adapt = float(alpha) * alpha_noise * h_scale

    tangent_offset = alpha_adapt * h.unsqueeze(-1) * (U_rot * t1 + V_rot * t2)
    normal_offset = (float(thickness) * Z) * n
    micro = 0.2 * float(alpha) * h.unsqueeze(-1) * torch.randn(M, 3, generator=generator, device=device, dtype=dtype)

    mu = mu_anchors + tangent_offset + normal_offset + micro
    return mu, n, mu_anchors


# ============================================================================
# MLS / oriented-point projection (kept signature)
# ============================================================================
def project_to_mls_surface_diff(
    P: torch.Tensor,
    X: torch.Tensor,
    N: torch.Tensor,
    iters: int = 2,
    k: int = 32,
    step: float = 1.0,
    knn: Optional[HybridFAISSKNN] = None,
    knn_tau: float = 0.15
) -> torch.Tensor:
    """Project points onto an oriented‑point MLS surface s(P)=0 (differentiable)."""
    for _ in range(int(iters)):
        if knn is not None:
            idx, w = knn(P, X, k)
            Q = X[idx]                    # [M,k,3]
            n = N[idx]                    # [M,k,3]
            V = P.unsqueeze(1) - Q        # [M,k,3]
            s = (w * (V * n).sum(-1)).sum(1, keepdim=True)   # [M,1]
            nbar = _normalize((w.unsqueeze(-1) * n).sum(1))  # [M,3]
        else:
            D = torch.cdist(P, X)                                 # [M,N]
            logits = -D / float(knn_tau)
            attn = F.softmax(logits, dim=1)
            topw, topi = torch.topk(attn, k=min(k, X.shape[0]), dim=1)
            w = topw / (topw.sum(dim=1, keepdim=True) + 1e-9)
            Q = X[topi]; n = N[topi]
            V = P.unsqueeze(1) - Q
            s = (w * (V * n).sum(-1)).sum(1, keepdim=True)
            nbar = _normalize((w.unsqueeze(-1) * n).sum(1))
        P = P - float(step) * s * nbar
    return P


# ============================================================================
# Density equalization (kept signature; adds robustness)
# ============================================================================
def density_equalize_diff(
    pts: torch.Tensor,
    nrms: torch.Tensor,
    anchors: torch.Tensor,
    x_low: torch.Tensor,
    normals_low: torch.Tensor,
    thickness: float,
    knn: HybridFAISSKNN,
    cfg: Dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adjust sampling density while staying close to the MLS surface.
    Self-neighbors are masked to avoid singular attraction.
    """
    if not cfg.get("enabled", True):
        return pts, nrms

    iters = int(cfg.get("iters", 8))
    k = int(cfg.get("k", 32))
    step0 = float(cfg.get("step", 0.45))
    annealing = float(cfg.get("annealing", 0.9))
    rmul = float(cfg.get("radius_mul", 1.2))
    use_mls = bool(cfg.get("use_mls_projection", True))
    mls_iters = int(cfg.get("mls_iters", 2))
    mls_step = float(cfg.get("mls_step", 1.0))
    knn_tau = float(cfg.get("knn_tau", 0.15))

    P, N = pts, nrms
    
    for it in range(iters):
        if it > 0 and it % 3 == 0:
            knn.invalidate_cache()  # points moved substantially

        idx, w = knn(P, P, k)
        Q = P[idx]                            # [M,k,3]
        diff = Q - P[:, None, :]              # [M,k,3]
        dist = torch.norm(diff.float(), dim=-1)

        h = rmul * ((dist[:, 1:].mean(dim=1) + 1e-6)[:, None])  # [M,1]
        kernel = -((dist / h) ** 2).clamp(min=-80.0)            # avoid exp overflow
        W = torch.exp(kernel) * w

        # cancel self-weight
        self_mask = (idx == torch.arange(P.shape[0], device=P.device).unsqueeze(1))
        W = W.masked_fill(self_mask, 0.0)
        W = W / (W.sum(dim=1, keepdim=True) + 1e-9)

        rho = W.sum(dim=1, keepdim=True)
        rho_star = rho.mean()
        s = torch.tanh((rho - rho_star) / (rho_star + 1e-6))

        disp = torch.einsum('nk,nkd->nd', W, diff) / (rho + 1e-6)
        step = step0 * (annealing ** it)  # step annealing
        P = P - step * s * disp

        if use_mls:
            knn.invalidate_cache()
            P = project_to_mls_surface_diff(P, x_low, normals_low,
                                            iters=mls_iters, k=k, step=mls_step,
                                            knn=knn, knn_tau=knn_tau)

    return P, N


# ============================================================================
# Optional: tangent Laplacian smoother
# ============================================================================
def surface_smoother_diff(
    P: torch.Tensor,
    X: torch.Tensor,
    N: torch.Tensor,
    knn: HybridFAISSKNN,
    iters: int = 3,
    k: int = 24,
    step: float = 0.12,
    lambda_normal: float = 0.15,
    mls_every: int = 2
) -> torch.Tensor:
    """
    Gentle surface smoothing:
    - Compute Laplacian displacement against k neighbors.
    - Project displacement onto the tangent plane (reduces "voxel" bumps).
    - Keep a small fraction of the normal component (lambda_normal).
    - Re-project to MLS every mls_every steps.
    """
    # Interpolate normals from X to P
    k_norm = min(8, X.shape[0])
    idx_N, w_N = knn(P, X, k_norm)
    N_neighbors = N[idx_N]  # [M, k_norm, 3]
    n = _normalize((w_N.unsqueeze(-1) * N_neighbors).sum(1))  # [M, 3]
    
    for t in range(int(iters)):
        idx, w = knn(P, P, min(k, P.shape[0]-1))
        Q = P[idx]
        diff = Q - P[:, None, :]
        dist = torch.norm(diff, dim=-1)
        h = (dist[:, 1:].mean(dim=1) + 1e-6).unsqueeze(-1)
        ww = torch.exp(- (dist / h) ** 2) * w
        ww[:, 0] = 0.0
        ww = ww / (ww.sum(dim=1, keepdim=True) + 1e-9)

        lap = torch.einsum('nk,nkd->nd', ww, diff)  # [M,3]

        # tangent/normal split
        lap_n = (lap * n).sum(-1, keepdim=True) * n
        lap_t = lap - lap_n
        update = step * (lap_t + lambda_normal * lap_n)
        P = P + update

        if mls_every > 0 and (t + 1) % mls_every == 0:
            knn.invalidate_cache()
            P = project_to_mls_surface_diff(P, X, N, iters=1, k=k, step=1.0, knn=knn)
    return P


# ============================================================================
# F smoothing (kept signature)
# ============================================================================
def smooth_F_diff_optimized(x: torch.Tensor, F: torch.Tensor, cfg: Dict) -> torch.Tensor:
    """Smooth per-point frames/tensors with a sparse node graph (unchanged API)."""
    if not cfg.get("enabled", True):
        return F

    K = min(int(cfg.get("num_nodes", 180)), x.shape[0])
    node_knn = int(cfg.get("node_knn", 8))
    point_knn = int(cfg.get("point_knn_nodes", 8))
    lam = float(cfg.get("lambda_lap", 1.0e-2))

    device, dtype = F.device, F.dtype
    N = x.shape[0]

    torch.manual_seed(42)
    sel = torch.randperm(N, device=device)[:K]
    Xn = x[sel]

    D_nodes = torch.cdist(Xn, Xn, p=2)
    k_node = min(node_knn, K - 1)
    _, j_nodes = torch.topk(D_nodes, k=k_node + 1, dim=1, largest=False)
    j_nodes = j_nodes[:, 1:]

    L = torch.zeros(K, K, device=device, dtype=dtype)
    row = torch.arange(K, device=device).unsqueeze(1).expand(-1, k_node).flatten()
    col = j_nodes.flatten()
    L.index_put_((row, col), torch.tensor(-1.0, device=device), accumulate=True)
    L[torch.arange(K, device=device), torch.arange(K, device=device)] = (j_nodes >= 0).sum(dim=1).float()

    D_p2n = torch.cdist(x, Xn, p=2)
    k_point = min(point_knn, K)
    d, j = torch.topk(D_p2n, k=k_point, dim=1, largest=False)

    h = d.mean(dim=1, keepdim=True) + 1e-9
    w_sparse = torch.exp(-(d / h) ** 2)
    w_sparse = w_sparse / (w_sparse.sum(dim=1, keepdim=True) + 1e-9)

    W = torch.zeros(N, K, device=device, dtype=dtype)
    r = torch.arange(N, device=device).unsqueeze(1).expand(-1, k_point).flatten()
    c = j.flatten()
    v = w_sparse.flatten()
    W.index_put_((r, c), v)

    WtW = torch.einsum('nk,nm->km', W, W)
    A = WtW + lam * L
    F_flat = F.reshape(N, 9)
    rhs = torch.einsum('nk,nr->kr', W, F_flat)
    Y = torch.linalg.solve(A, rhs)
    F_smooth_flat = torch.einsum('nk,kr->nr', W, Y)
    return F_smooth_flat.reshape(N, 3, 3)


# ============================================================================
# Main entry (signature preserved)
# ============================================================================
def synthesize_runtime_surface(
    x_low: torch.Tensor,
    F_low: torch.Tensor,
    cfg: Dict,
    ema_state: Optional[Dict] = None,
    seed: int = 1234,
    differentiable: bool = True,
    return_torch: bool = True
) -> Dict:
    """
    Full runtime surface synthesis with hybrid FAISS and differentiable pipeline.
    This function preserves the original signature and return structure.
    """
    if ema_state is None:
        ema_state = {}

    device = x_low.device if torch.is_tensor(x_low) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_low = _ensure_torch(x_low, device=device)
    F_low = _ensure_torch(F_low, device=device).reshape(-1, 3, 3)

    # Extract config
    use_hybrid = bool(cfg.get("use_hybrid_faiss", True))
    k_surface = int(cfg.get("k_surface", 36))
    thr_pct = float(cfg.get("thr_percentile", 8.0))
    ema_beta = float(cfg.get("ema_beta", 0.95))
    hys = float(cfg.get("hysteresis", 0.03))
    tau = float(cfg.get("soft_tau", 0.08))
    M = int(cfg.get("M", 50_000))
    alpha = float(cfg.get("surf_jitter_alpha", 0.6))
    thickness = float(cfg.get("thickness", 0.00))
    density_g = float(cfg.get("density_gamma", 2.5))
    sampling_tau = float(cfg.get("sampling_tau", 0.2))
    surface_power = float(cfg.get("surface_power", 4.0))
    knn_tau = float(cfg.get("knn_tau", 0.15))
    use_ivf = bool(cfg.get("use_faiss_ivf", True))
    use_amp = bool(cfg.get("use_amp", True) and device.type == 'cuda')

    # Hybrid FAISS kNN
    knn = HybridFAISSKNN(
        use_faiss=use_hybrid and FAISS_AVAILABLE,
        use_ivf=use_ivf,
        tau=knn_tau,
        nlist=int(cfg.get("ivf_nlist", 100)),
        nprobe=int(cfg.get("ivf_nprobe", 10)),
        use_soft_radius=bool(cfg.get("use_soft_radius", False)),
        soft_radius_candidates=int(cfg.get("soft_radius_candidates", 128))
    )

    # Isolated RNG for reproducibility
    generator = torch.Generator(device=device).manual_seed(int(seed))
    amp_ctx = torch.cuda.amp.autocast() if use_amp else nullcontext()

    with amp_ctx:
        # -- surface detection ------------------------------------------------
        if cfg.get("use_surface_detection", True):
            surf_prob, normals, spacing, thr_low, thr_high = compute_surface_mask_diff(
                x_low, knn, k_surface, thr_pct,
                ema_state.get("ema_thr"), ema_beta, hys, tau, surface_power, ema_state
            )
        else:
            N = x_low.shape[0]
            surf_prob = torch.full((N,), 1.0 / N, device=device)
            idx, w = knn(x_low, x_low, k_surface)
            normals, _, spacing = batched_pca_surface_optimized(x_low, idx, w)
            thr_low = thr_high = 0.0
            ema_state.update({"ema_thr": 0.0, "thr_low": 0.0, "thr_high": 0.0})

        # -- sampling ---------------------------------------------------------
        pts, nrm, anchors = sample_surface_points_diff(
            x_low, normals, spacing, surf_prob,
            M, alpha, thickness, density_g, seed,
            tau=sampling_tau, generator=generator
        )

        # -- density equalization (+MLS) -------------------------------------
        eq_cfg = dict(cfg.get("post_equalize", {}))
        eq_cfg['knn_tau'] = knn_tau
        if eq_cfg.get("enabled", True):
            pts, nrm = density_equalize_diff(pts, nrm, anchors, x_low, normals, thickness, knn, eq_cfg)

        # -- optional tangent smoother (very gentle) -------------------------
        smooth_cfg = cfg.get("smoother", {})
        if smooth_cfg.get("enabled", False):
            pts = surface_smoother_diff(
                pts, x_low, normals, knn,
                iters=int(smooth_cfg.get("iters", 3)),
                k=int(smooth_cfg.get("k", 24)),
                step=float(smooth_cfg.get("step", 0.12)),
                lambda_normal=float(smooth_cfg.get("lambda_normal", 0.15)),
                mls_every=int(smooth_cfg.get("mls_every", 2)),
            )

        # -- F smoothing & local covariance ----------------------------------
        F_smooth = F_low
        if cfg.get("use_F_kernel", True):
            ed = cfg.get("ed", {})
            if ed.get("enabled", True):
                F_smooth = smooth_F_diff_optimized(x_low, F_low, ed)

        k_F = int(cfg.get("k_F", 32))
        sigma0 = float(cfg.get("sigma0", 0.08))
        idxF, wF = knn(pts, x_low, k_F)
        F_neighbors = F_smooth[idxF]                         # [M,k,3,3]
        F_loc = torch.einsum('mk,mkrc->mrc', wF, F_neighbors)
        cov = (sigma0 ** 2) * torch.matmul(F_loc, F_loc.transpose(-2, -1))

    if use_amp:
        pts, nrm, F_loc, cov = pts.float(), nrm.float(), F_loc.float(), cov.float()

    knn.clear_cache()

    debug = {
        "thr_low": float(thr_low),
        "thr_high": float(thr_high),
        "ema_thr": float(ema_state.get("ema_thr", 0.0)),
        "mean_prob": float(surf_prob.mean().item()),
        "use_hybrid_faiss": use_hybrid and FAISS_AVAILABLE,
        "use_faiss_ivf": use_ivf,
        "use_soft_radius": bool(cfg.get("use_soft_radius", False)),
        "sampling_tau": sampling_tau,
        "surface_power": surface_power,
        "knn_tau": knn_tau,
    }

    if return_torch:
        return {
            "points": pts,
            "normals": nrm,
            "F_smooth": F_loc,
            "cov": cov,
            "debug": debug,
            "state": ema_state,
            "anchors": anchors,
            "surf_prob": surf_prob,
        }
    else:
        return {
            "points": pts.detach().cpu().numpy(),
            "normals": nrm.detach().cpu().numpy(),
            "F_smooth": F_loc.detach().cpu().numpy(),
            "cov": cov.detach().cpu().numpy(),
            "debug": debug,
            "state": ema_state,
        }

# # ============================================================================
# # PNG Export Functions
# # ============================================================================
def _as_np(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)

def save_comparison_png(path, current_before=None, current_after=None, radial_after=None, 
                        target_before=None, dpi=160, ptsize=0.5):
    """Save 3-panel comparison."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    X  = _as_np(current_before) if current_before is not None else None
    PU = _as_np(current_after) if current_after is not None else None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if PU is None or PU.size == 0:
        fig = plt.figure(figsize=(3,3))
        fig.text(0.5,0.5,"No points", ha="center")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    fig = plt.figure(figsize=(18,5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    if X is not None and X.size > 0:
        ax1.scatter(X[:,0], X[:,1], X[:,2], s=ptsize, alpha=0.6)
    ax1.set_title("Current (before upsampling)")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

    ax2.scatter(PU[:,0], PU[:,1], PU[:,2], s=ptsize, alpha=0.6)
    ax2.set_title("Current (after upsampling)")

    c = PU.mean(0)
    r = np.linalg.norm(PU - c[None,:], axis=1)
    sc = ax3.scatter(PU[:,0], PU[:,1], PU[:,2], c=r, s=ptsize, alpha=0.6, cmap="viridis")
    ax3.set_title("Radial color (after upsampling)")
    fig.colorbar(sc, ax=ax3, shrink=0.6)

    ALL = PU if PU.size>0 else (X if X is not None and X.size>0 else PU)
    mins, maxs = ALL.min(0), ALL.max(0)
    rng = (maxs - mins).max() * 0.5
    mid = (maxs + mins) * 0.5
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(mid[0]-rng, mid[0]+rng)
        ax.set_ylim(mid[1]-rng, mid[1]+rng)
        ax.set_zlim(mid[2]-rng, mid[2]+rng)
        ax.set_box_aspect([1,1,1])
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def save_axis_hist_png(path, pts, dpi=160):
    """Save axis histograms."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    PU = _as_np(pts)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if PU is None or PU.size == 0:
        fig = plt.figure(figsize=(3,3))
        fig.text(0.5,0.5,"No points", ha="center")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(1,3, figsize=(16,4))
    for j,ax in enumerate(axes):
        ax.hist(PU[:,j], bins=48, alpha=0.85)
        ax.set_title(["X-Axis Distribution","Y-Axis Distribution","Z-Axis Distribution"][j])
        ax.set_xlabel(f"{'XYZ'[j]} Coordinate")
        ax.set_ylabel("Count")
    fig.suptitle("Runtime Surface (Axis Distribution)")
    fig.tight_layout(rect=[0,0,1,0.92])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
 
def save_ply_xyz(path: Path, xyz: np.ndarray):
    """Save PLY file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        np.savetxt(f, xyz, fmt="%.6f")

def save_gaussians_npz(path: Path, xyz: np.ndarray, cov: np.ndarray, rgb=None, opacity=None):
    """Save Gaussian splatting data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb is None:
        rgb = np.ones_like(xyz, dtype=np.float32)
    if opacity is None:
        opacity = np.ones((len(xyz),1), dtype=np.float32)
    np.savez_compressed(
        path, 
        xyz=xyz.astype(np.float32),
        cov=cov.astype(np.float32),
        rgb=rgb.astype(np.float32),
        opacity=opacity.astype(np.float32)
    )