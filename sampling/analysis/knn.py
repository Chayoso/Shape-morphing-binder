"""Hybrid FAISS KNN with differentiable reweighting."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple
import warnings

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
    Forward: FAISS gives k nearest neighbor indices quickly (no grads).
    Backward: recompute distances on gathered neighbors and build
              a softmax weight field (differentiable w.r.t. query & data).
    """
    
    def __init__(
        self, 
        use_faiss: bool = True, 
        use_ivf: bool = True,
        tau: float = 0.15, 
        nlist: int = 100, 
        nprobe: int = 10,
        use_soft_radius: bool = False, 
        soft_radius_candidates: int = 128
    ):
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.use_ivf = use_ivf
        self.tau = float(tau)
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self.use_soft_radius = bool(use_soft_radius)
        self.soft_radius_candidates = int(soft_radius_candidates)
        self._index_cache = {}
        self._epoch = 0
    
    def clear_cache(self):
        """Drop all FAISS indices (free memory)."""
        self._index_cache.clear()
    
    def invalidate_cache(self):
        """Increase epoch so next call rebuilds the index."""
        self._epoch += 1
        self._index_cache.clear()
    
    def __call__(
        self, 
        query: torch.Tensor, 
        data: torch.Tensor, 
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find k nearest neighbors with differentiable weights."""
        device = query.device
        k = int(min(k, data.shape[0]))
        
        if not (self.use_faiss and FAISS_AVAILABLE):
            return self._torch_soft_knn(query, data, k)
        
        if self.use_soft_radius:
            return self._hybrid_faiss_soft_radius(query, data, k)
        else:
            return self._hybrid_faiss_knn(query, data, k)
    
    def _build_index(self, data: torch.Tensor, D: int, nlist: int, nprobe: int, cache_key):
        """Create or fetch a FAISS index for this data snapshot."""
        if cache_key in self._index_cache:
            return self._index_cache[cache_key]
        
        self._index_cache.clear()
        data_np = data.detach().cpu().float().numpy()
        N = data_np.shape[0]
        
        nlist_adjusted = max(1, min(nlist, N // 40))
        
        if self.use_ivf:
            index = self._build_ivf_index(data_np, D, N, nlist_adjusted, nprobe, data.is_cuda)
        else:
            index = self._build_flat_index(data_np, D, data.is_cuda)
        
        self._index_cache[cache_key] = index
        return index
    
    def _build_ivf_index(self, data_np: np.ndarray, D: int, N: int, nlist: int, nprobe: int, is_cuda: bool):
        """Build IVF index for faster search."""
        if N < nlist * 39:
            return self._build_flat_index(data_np, D, is_cuda)
        
        if is_cuda:
            return self._build_gpu_ivf_index(data_np, D, N, nlist, nprobe)
        else:
            return self._build_cpu_ivf_index(data_np, D, N, nlist, nprobe)
    
    def _build_gpu_ivf_index(self, data_np: np.ndarray, D: int, N: int, nlist: int, nprobe: int):
        """Build GPU IVF index."""
        res = faiss.StandardGpuResources()
        cpu_quantizer = faiss.IndexFlatL2(D)
        cpu_index = faiss.IndexIVFFlat(cpu_quantizer, D, nlist, faiss.METRIC_L2)
        
        if not cpu_index.is_trained:
            train_size = min(N, 100_000)
            train_sel = np.random.choice(N, train_size, replace=False) if N > train_size else np.arange(N)
            cpu_index.train(data_np[train_sel])
        
        cpu_index.add(data_np)
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        index.nprobe = min(nprobe, nlist)
        return index
    
    def _build_cpu_ivf_index(self, data_np: np.ndarray, D: int, N: int, nlist: int, nprobe: int):
        """Build CPU IVF index."""
        quantizer = faiss.IndexFlatL2(D)
        index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_L2)
        
        if not index.is_trained:
            train_size = min(N, 100_000)
            train_sel = np.random.choice(N, train_size, replace=False) if N > train_size else np.arange(N)
            index.train(data_np[train_sel])
        
        index.add(data_np)
        index.nprobe = min(nprobe, nlist)
        return index
    
    def _build_flat_index(self, data_np: np.ndarray, D: int, is_cuda: bool):
        """Build flat index."""
        if is_cuda:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, D)
        else:
            index = faiss.IndexFlatL2(D)
        index.add(data_np)
        return index
    
    def _compute_differentiable_weights(self, query: torch.Tensor, data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Compute differentiable weights from gathered neighbors."""
        neigh = data[indices]
        qx = query.unsqueeze(1).float()
        dist = torch.norm(qx - neigh.float(), dim=2)
        logits = -dist / self.tau
        weights = F.softmax(logits, dim=1).to(query.dtype)
        return weights
    
    def _compute_simple_weights(self, distances: torch.Tensor) -> torch.Tensor:
        """Compute simple exponential weights (no grad)."""
        weights = torch.exp(-distances / self.tau)
        weights = weights / (weights.sum(dim=1, keepdim=True) + EPS_SAFE)
        return weights
    
    def _hybrid_faiss_knn(self, query: torch.Tensor, data: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard hybrid mode: exact k by FAISS, differentiable weights."""
        N, D = query.shape
        M = data.shape[0]
        
        nlist = min(self.nlist, max(4, M // 100))
        nprobe = min(self.nprobe, nlist)
        data_ptr = int(data.untyped_storage().data_ptr())
        cache_key = (M, D, nlist, data_ptr, self._epoch)
        
        index = self._build_index(data, D, nlist, nprobe, cache_key)
        
        q_np = query.detach().cpu().float().numpy()
        d_np, i_np = index.search(q_np, k)
        distances = torch.from_numpy(d_np).to(query.device)
        indices = torch.from_numpy(i_np).to(query.device, dtype=torch.long)
        
        if query.requires_grad or data.requires_grad:
            weights = self._compute_differentiable_weights(query, data, indices)
        else:
            weights = self._compute_simple_weights(distances)
        
        return indices, weights
    
    def _hybrid_faiss_soft_radius(self, query: torch.Tensor, data: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Soft-radius mode: fetch larger candidate pool, then select top-k by weight."""
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
            neigh = data[idx_all]
            qx = query.unsqueeze(1).float()
            dist = torch.norm(qx - neigh.float(), dim=2)
            logits = -dist / self.tau
            w_all = F.softmax(logits, dim=1)
        else:
            w_all = torch.exp(-distances / self.tau)
            w_all = w_all / (w_all.sum(dim=1, keepdim=True) + EPS_SAFE)
        
        topw, topj = torch.topk(w_all, k=min(k, Kc), dim=1)
        batch = torch.arange(N, device=query.device).unsqueeze(1).expand(-1, topj.shape[1])
        indices = idx_all[batch, topj]
        weights = topw / (topw.sum(dim=1, keepdim=True) + EPS_SAFE)
        
        if query.requires_grad or data.requires_grad:
            weights = weights.to(query.dtype)
        
        return indices, weights
    
    def _torch_soft_knn(self, query: torch.Tensor, data: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pure torch fallback: differentiable attention over distances."""
        D = torch.cdist(query, data, p=2)
        logits = -D / self.tau
        attn = F.softmax(logits, dim=1)
        topw, topi = torch.topk(attn, k=k, dim=1)
        weights = topw / (topw.sum(dim=1, keepdim=True) + EPS_SAFE)
        return topi, weights