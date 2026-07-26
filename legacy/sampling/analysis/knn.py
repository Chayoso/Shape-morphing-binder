"""
Hybrid FAISS KNN with differentiable reweighting.

This module implements a KNN search that combines:
- Fast approximate nearest neighbor search using FAISS (forward pass)
- Differentiable softmax attention weights (backward pass)

Key ideas:
- Straight-through-ish pattern: indices are hard (non-diff), weights are soft (diff)
- Soft-radius over-fetch smooths neighbor-set transitions at cell/Voronoi boundaries
- Tie-break jitter + robust normalization + row-wise max-sub stabilize softmax

Author: CHAYO pipeline patch (2025-10)
"""

from typing import Tuple
import warnings
import numpy as np
import torch
import torch.nn.functional as F

from ..utils.config import EPS_SAFE

try:
    import faiss
    import faiss.contrib.torch_utils  # noqa: F401
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False
    warnings.warn("FAISS not available - fallback to pure torch (slower)")


class HybridFAISSKNN:
    """
    Hybrid FAISS KNN search with differentiable weights.

    Forward pass:  FAISS → fast (approximate/exact) k-NN indices (non-differentiable)
    Backward pass: Recompute distances on selected neighbors → soft weights (differentiable)
    """

    def __init__(
        self,
        use_faiss: bool = True,
        use_ivf: bool = True,
        tau: float = 0.18,                 # reduce winner-take-all vs 0.1~0.15
        nlist: int = 100,
        nprobe: int = 10,
        use_soft_radius: bool = True,      # enable soft-radius by default
        soft_radius_candidates: int = 128, # >= 4 * k recommended
        fallback_chunk_size: int = 5000,
        tie_break_eps: float = 1e-4,       # tiny jitter to break equidistant ties
        uniform_eps: float = 1e-3          # uniform mix to avoid collapse in ultrasparse regions
    ):
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.use_ivf = use_ivf
        self.tau = float(tau)
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self.use_soft_radius = bool(use_soft_radius)
        self.soft_radius_candidates = int(soft_radius_candidates)
        self.fallback_chunk_size = int(fallback_chunk_size)
        self.tie_break_eps = float(tie_break_eps)
        self.uniform_eps = float(uniform_eps)

        self._index_cache = {}
        self._epoch = 0
        self._gpu_resources = None

    # -------------------- public utils --------------------

    def set_tau(self, tau: float):
        """Dynamically adjust temperature (e.g., schedule)."""
        self.tau = float(tau)

    def clear_cache(self):
        """Free cached FAISS indices/resources."""
        self._index_cache.clear()
        self._gpu_resources = None

    def invalidate_cache(self):
        """Force rebuild on next call (call after in-place data updates)."""
        self._epoch += 1
        self._index_cache.clear()

    # -------------------- main entry --------------------

    def __call__(
        self,
        query: torch.Tensor,
        data: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find k nearest neighbors with differentiable weights.

        Returns:
            indices: (N, k) long
            weights: (N, k) float, sums to 1 along dim=1
        """
        k = int(min(k, data.shape[0]))

        if not (self.use_faiss and FAISS_AVAILABLE):
            return self._torch_soft_knn(query, data, k)

        if self.use_soft_radius:
            return self._hybrid_faiss_soft_radius(query, data, k)
        else:
            return self._hybrid_faiss_knn(query, data, k)

    # -------------------- index builders --------------------

    def _build_index(self, data: torch.Tensor, D: int, nlist: int, nprobe: int, cache_key):
        if cache_key in self._index_cache:
            return self._index_cache[cache_key]

        self._index_cache.clear()
        data_np = data.detach().cpu().float().numpy()
        N = data_np.shape[0]

        nlist_adjusted = max(1, min(nlist, N // 40))  # ~40 pts per cell

        if self.use_ivf:
            index = self._build_ivf_index(data_np, D, N, nlist_adjusted, nprobe, data.is_cuda)
        else:
            index = self._build_flat_index(data_np, D, data.is_cuda)

        self._index_cache[cache_key] = index
        return index

    def _build_ivf_index(self, data_np: np.ndarray, D: int, N: int, nlist: int, nprobe: int, is_cuda: bool):
        if N < nlist * 39:
            return self._build_flat_index(data_np, D, is_cuda)
        if is_cuda:
            return self._build_gpu_ivf_index(data_np, D, N, nlist, nprobe)
        else:
            return self._build_cpu_ivf_index(data_np, D, N, nlist, nprobe)

    def _build_gpu_ivf_index(self, data_np: np.ndarray, D: int, N: int, nlist: int, nprobe: int):
        if self._gpu_resources is None:
            self._gpu_resources = faiss.StandardGpuResources()
        cpu_quantizer = faiss.IndexFlatL2(D)
        cpu_index = faiss.IndexIVFFlat(cpu_quantizer, D, nlist, faiss.METRIC_L2)
        if not cpu_index.is_trained:
            train_size = min(N, 100_000)
            sel = np.random.choice(N, train_size, replace=False) if N > train_size else np.arange(N)
            cpu_index.train(data_np[sel])
        cpu_index.add(data_np)
        index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index)
        index.nprobe = min(nprobe, nlist)
        return index

    def _build_cpu_ivf_index(self, data_np: np.ndarray, D: int, N: int, nlist: int, nprobe: int):
        quantizer = faiss.IndexFlatL2(D)
        index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_L2)
        if not index.is_trained:
            train_size = min(N, 100_000)
            sel = np.random.choice(N, train_size, replace=False) if N > train_size else np.arange(N)
            index.train(data_np[sel])
        index.add(data_np)
        index.nprobe = min(nprobe, nlist)
        return index

    def _build_flat_index(self, data_np: np.ndarray, D: int, is_cuda: bool):
        if is_cuda:
            if self._gpu_resources is None:
                self._gpu_resources = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(self._gpu_resources, D)
        else:
            index = faiss.IndexFlatL2(D)
        index.add(data_np)
        return index

    # -------------------- differentiable weights --------------------

    def _compute_differentiable_weights(
        self, query: torch.Tensor, data: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Differentiable softmax attention with tie-break and robust normalization.
        """
        neigh = data[indices]                         # (N, k, D)
        qx = query.unsqueeze(1).to(torch.float32)     # (N, 1, D)
        diff = (qx - neigh.to(torch.float32))         # (N, k, D)

        # squared distance (smooth gradient near 0)
        dist2 = (diff * diff).sum(dim=2)              # (N, k)

        # tie-break jitter (no-grad)
        if self.tie_break_eps > 0.0:
            with torch.no_grad():
                noise = (self.tie_break_eps * torch.randn_like(dist2)).detach()  # constant noise
                dist2 = dist2 + noise  # keep graph on dist2

        # robust row-wise normalization (median/MAD)
        with torch.no_grad():
            med = dist2.median(dim=1, keepdim=True).values
            mad = (dist2 - med).abs().median(dim=1, keepdim=True).values
            scale = torch.clamp(mad, min=1e-6)

        dnorm = (dist2 - med) / scale

        # temperature + row-wise max-sub
        tau = max(float(self.tau), 1e-4)
        logits = -dnorm / tau
        logits = logits - logits.max(dim=1, keepdim=True).values

        w = torch.softmax(logits, dim=1).to(query.dtype)

        # uniform-ε mix to avoid collapse in ultrasparse neighborhoods
        if self.uniform_eps > 0.0:
            k = w.shape[1]
            w = (1.0 - self.uniform_eps) * w + (self.uniform_eps / k)

        # exact renormalization
        w = w / (w.sum(dim=1, keepdim=True) + EPS_SAFE)
        return w

    def _compute_simple_weights(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Inference-only path w/ stability (softmax form for consistency).
        """
        logits = -distances / max(float(self.tau), 1e-4)
        logits = logits - logits.max(dim=1, keepdim=True).values
        weights = torch.softmax(logits, dim=1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + EPS_SAFE)
        return weights

    # -------------------- FAISS hybrid modes --------------------

    def _hybrid_faiss_knn(self, query: torch.Tensor, data: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard hybrid mode: FAISS for indices, differentiable weights.
        """
        N, D = query.shape
        M = data.shape[0]

        # auto-tune nlist
        nlist = min(self.nlist, max(4, M // 100))
        nprobe = min(self.nprobe, nlist)

        # cache key
        data_ptr = int(data.untyped_storage().data_ptr())
        cache_key = (M, D, nlist, data_ptr, self._epoch)

        index = self._build_index(data, D, nlist, nprobe, cache_key)

        # FAISS search
        q_np = query.detach().cpu().float().numpy()
        d_np, i_np = index.search(q_np, k)
        distances = torch.from_numpy(d_np).to(query.device)
        indices = torch.from_numpy(i_np).to(query.device, dtype=torch.long)

        # weights
        if query.requires_grad or data.requires_grad:
            weights = self._compute_differentiable_weights(query, data, indices)
        else:
            weights = self._compute_simple_weights(distances)

        return indices, weights

    def _hybrid_faiss_soft_radius(self, query: torch.Tensor, data: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Soft-radius mode: fetch larger pool (Kc), score all, select top-k by weight (ST-ish).
        """
        N, D = query.shape
        M = data.shape[0]
        Kc = int(min(self.soft_radius_candidates, M))

        nlist = min(self.nlist, max(4, M // 100))
        nprobe = min(self.nprobe, nlist)
        data_ptr = int(data.untyped_storage().data_ptr())
        cache_key = (M, D, nlist, data_ptr, self._epoch)

        index = self._build_index(data, D, nlist, nprobe, cache_key)

        # candidates
        q_np = query.detach().cpu().float().numpy()
        d_np, i_np = index.search(q_np, Kc)
        idx_all = torch.from_numpy(i_np).to(query.device, dtype=torch.long)

        if query.requires_grad or data.requires_grad:
            neigh = data[idx_all]                       # (N, Kc, D)
            qx = query.unsqueeze(1).float()
            diff = qx - neigh.float()
            dist2 = (diff * diff).sum(dim=2)            # (N, Kc)

            if self.tie_break_eps > 0.0:
                with torch.no_grad():
                    noise = (self.tie_break_eps * torch.randn_like(dist2)).detach()  # constant noise
                    dist2 = dist2 + noise  # keep graph on dist2

            # robust normalize (optional but helpful in ultra-sparse)
            with torch.no_grad():
                med = dist2.median(dim=1, keepdim=True).values
                mad = (dist2 - med).abs().median(dim=1, keepdim=True).values
                scale = torch.clamp(mad, min=1e-6)
            dnorm = (dist2 - med) / scale

            logits = -dnorm / max(float(self.tau), 1e-4)
            logits = logits - logits.max(dim=1, keepdim=True).values
            w_all = F.softmax(logits, dim=1)

            if self.uniform_eps > 0.0:
                w_all = (1.0 - self.uniform_eps) * w_all + (self.uniform_eps / Kc)
        else:
            distances = torch.from_numpy(d_np).to(query.device)
            logits = -distances / max(float(self.tau), 1e-4)
            logits = logits - logits.max(dim=1, keepdim=True).values
            w_all = torch.softmax(logits, dim=1)

        # straight-through hard top-k indices (forward only)
        with torch.no_grad():
            _, topj_hard = torch.topk(w_all, k=min(k, Kc), dim=1)

        batch = torch.arange(N, device=query.device).unsqueeze(1).expand(-1, topj_hard.shape[1])
        topw = w_all[batch, topj_hard]          # keeps gradient path
        indices = idx_all[batch, topj_hard]

        weights = topw / (topw.sum(dim=1, keepdim=True) + EPS_SAFE)
        if query.requires_grad or data.requires_grad:
            weights = weights.to(query.dtype)
        return indices, weights

    # -------------------- pure torch fallback --------------------

    def _torch_soft_knn_chunked(
        self,
        query: torch.Tensor,
        data: torch.Tensor,
        k: int,
        chunk_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = query.shape[0]
        M = data.shape[0]
        device = query.device

        all_indices = []
        all_weights = []

        for i in range(0, N, chunk_size):
            chunk_end = min(i + chunk_size, N)
            chunk = query[i:chunk_end]  # (b, D)

            Dmat = torch.cdist(chunk, data, p=2)  # (b, M)

            if self.tie_break_eps > 0.0:
                with torch.no_grad():
                    Dmat = Dmat + self.tie_break_eps * torch.randn_like(Dmat)

            logits = -Dmat / max(float(self.tau), 1e-4)
            logits = logits - logits.max(dim=1, keepdim=True).values
            attn = F.softmax(logits, dim=1)

            with torch.no_grad():
                _, topi_hard = torch.topk(attn, k=k, dim=1)

            bsz = chunk_end - i
            batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(-1, k)
            topw = attn[batch_idx, topi_hard]
            weights = topw / (topw.sum(dim=1, keepdim=True) + EPS_SAFE)

            all_indices.append(topi_hard)
            all_weights.append(weights)

            del Dmat, logits, attn
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.cat(all_indices), torch.cat(all_weights)

    def _torch_soft_knn(self, query: torch.Tensor, data: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        N, M = query.shape[0], data.shape[0]
        full_memory = N * M * 4  # bytes (float32)
        memory_threshold = int(1e9)  # ~1GB

        if full_memory > memory_threshold:
            chunk_size = max(1, int(memory_threshold / (M * 4)))
            chunk_size = min(chunk_size, self.fallback_chunk_size)
            return self._torch_soft_knn_chunked(query, data, k, chunk_size)

        Dmat = torch.cdist(query, data, p=2)  # (N, M)

        if self.tie_break_eps > 0.0:
            with torch.no_grad():
                Dmat = Dmat + self.tie_break_eps * torch.randn_like(Dmat)

        logits = -Dmat / max(float(self.tau), 1e-4)
        logits = logits - logits.max(dim=1, keepdim=True).values
        attn = F.softmax(logits, dim=1)

        with torch.no_grad():
            _, topi_hard = torch.topk(attn, k=k, dim=1)

        batch_idx = torch.arange(N, device=query.device).unsqueeze(1).expand(-1, k)
        topw = attn[batch_idx, topi_hard]
        weights = topw / (topw.sum(dim=1, keepdim=True) + EPS_SAFE)

        return topi_hard, weights
