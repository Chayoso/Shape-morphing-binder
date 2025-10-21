"""
Hybrid FAISS KNN with differentiable reweighting.

This module implements a KNN search that combines:
- Fast approximate nearest neighbor search using FAISS (forward pass)
- Differentiable softmax attention weights (backward pass)

The key innovation is the "straight-through estimator" pattern:
- Forward: Use FAISS for O(N log M) fast index selection
- Backward: Recompute distances on selected neighbors for gradients

This allows gradient flow through query positions and data points
while maintaining FAISS's speed advantage.
"""

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
    Hybrid FAISS KNN search with differentiable weights.
    
    Architecture:
        Forward pass:  FAISS → fast approximate k-NN indices (non-differentiable)
        Backward pass: Recompute distances on selected neighbors → softmax weights (differentiable)
    
    This design achieves:
    - Speed: FAISS gives O(N log M) or O(N √M) complexity (vs O(NM) for brute force)
    - Differentiability: Weights computed via temperature-scaled softmax maintain gradients
    - Flexibility: Fallback to pure PyTorch when FAISS unavailable
    
    Modes:
        1. Standard KNN: Fetch exactly k neighbors, compute soft weights
        2. Soft Radius: Fetch larger pool (e.g. 128), select top-k by weight (more flexible)
        3. Pure Torch: Brute-force with automatic memory chunking (fallback)
    
    Caching:
        FAISS indices are cached per-data-pointer to avoid rebuilding.
        Use invalidate_cache() when data changes in-place.
    
    Memory Management:
        Pure torch mode auto-detects memory pressure and chunks queries
        to avoid OOM on large datasets (e.g., N=100k, M=500k).
    
    Example:
        >>> knn = HybridFAISSKNN(use_faiss=True, tau=0.15, nlist=100)
        >>> query = torch.randn(10000, 3, requires_grad=True)
        >>> data = torch.randn(50000, 3, requires_grad=True)
        >>> indices, weights = knn(query, data, k=32)  # (10000, 32), (10000, 32)
        >>> # indices: hard neighbor indices (from FAISS)
        >>> # weights: soft attention weights (differentiable w.r.t. query & data)
        >>> loss = (weights * some_feature[indices]).sum()
        >>> loss.backward()  # Gradients flow through weights to query/data
    """
    
    def __init__(
        self, 
        use_faiss: bool = True, 
        use_ivf: bool = True,
        tau: float = 0.15, 
        nlist: int = 100, 
        nprobe: int = 10,
        use_soft_radius: bool = False, 
        soft_radius_candidates: int = 128,
        fallback_chunk_size: int = 5000
    ):
        """
        Initialize hybrid FAISS KNN search.
        
        Args:
            use_faiss: If True and FAISS available, use FAISS; else pure torch fallback
            use_ivf: If True, use IVF (Inverted File) index for faster search on large datasets
                     IVF trades accuracy for speed: O(N√M) vs O(N log M) for flat
            tau: Temperature for softmax attention weights
                 - Lower tau (e.g., 0.05) → sharper, more peaked weights
                 - Higher tau (e.g., 0.5) → smoother, more uniform weights
                 - Default 0.15 is a good balance for most tasks
            nlist: Number of Voronoi cells for IVF index (only used if use_ivf=True)
                   - More cells → faster search but more memory
                   - Rule of thumb: nlist ≈ √M for M data points
                   - Auto-adjusted to min(nlist, M//40) to avoid degenerate cases
            nprobe: Number of cells to visit during search (only used if use_ivf=True)
                    - More probes → more accurate but slower
                    - Default 10 is good tradeoff (visits 10% of 100 cells)
            use_soft_radius: If True, use soft radius mode (fetch more, select top-k by weight)
                            - Allows gradient-based neighbor selection
                            - Slightly more expensive but more flexible
            soft_radius_candidates: Pool size for soft radius mode (e.g., 128)
                                   - Fetch this many candidates, select top-k by weight
                                   - Only used if use_soft_radius=True
            fallback_chunk_size: Chunk size for pure torch fallback when memory limited
                                - Queries processed in chunks to avoid OOM
                                - Auto-tuned based on available memory in practice
        
        Attributes:
            _index_cache: Dict mapping (M, D, nlist, data_ptr, epoch) → FAISS index
            _epoch: Counter incremented on invalidate_cache() to force rebuild
        """
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.use_ivf = use_ivf
        self.tau = float(tau)
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self.use_soft_radius = bool(use_soft_radius)
        self.soft_radius_candidates = int(soft_radius_candidates)
        self.fallback_chunk_size = int(fallback_chunk_size)
        self._index_cache = {}
        self._epoch = 0
        self._gpu_resources = None
    
    def clear_cache(self):
        """
        Clear all cached FAISS indices to free GPU/CPU memory.
        
        Use this when you're done with a dataset and want to reclaim memory.
        Next call will rebuild indices from scratch.
        
        Example:
            >>> knn = HybridFAISSKNN()
            >>> knn(query1, data1, k=32)  # Builds index for data1
            >>> knn.clear_cache()  # Free memory
            >>> knn(query2, data2, k=32)  # Builds new index for data2
        """
        self._index_cache.clear()
        
        if self._gpu_resources is not None:
            self._gpu_resources = None
    
    def invalidate_cache(self):
        """
        Invalidate cache to force index rebuild on next call.
        
        Use this when data changes in-place (e.g., during training when
        anchor positions are updated via gradient descent).
        
        Increments internal epoch counter so cache keys won't match.
        
        Example:
            >>> data = torch.randn(10000, 3, requires_grad=True)
            >>> knn(query, data, k=32)  # Builds index
            >>> # ... training step updates data via optimizer ...
            >>> knn.invalidate_cache()  # Force rebuild next call
            >>> knn(query, data, k=32)  # Uses updated data
        """
        self._epoch += 1
        self._index_cache.clear()
    
    def __call__(
        self, 
        query: torch.Tensor, 
        data: torch.Tensor, 
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find k nearest neighbors with differentiable weights.
        
        Main entry point. Automatically selects best strategy:
        1. FAISS hybrid mode (if use_faiss=True and FAISS available)
        2. Pure torch fallback (if FAISS unavailable or use_faiss=False)
        
        Differentiability:
            - If query.requires_grad or data.requires_grad: computes soft weights
            - Otherwise: uses simple exponential weights (no autograd overhead)
        
        Args:
            query: (N, D) query points - can have requires_grad=True
            data: (M, D) database points - can have requires_grad=True
            k: Number of nearest neighbors to return
               Auto-clamped to min(k, M) to avoid errors
        
        Returns:
            indices: (N, k) LongTensor of neighbor indices into data
                     - Hard selection (discrete, no gradient)
                     - But surrounded by differentiable weights
            weights: (N, k) FloatTensor of attention weights
                     - Sum to 1.0 along dim=1 (normalized)
                     - Differentiable w.r.t. query and data positions
                     - If no gradients needed: simple exp(-d/tau) weights
                     - If gradients needed: softmax(-d/tau) for full differentiability
        
        Example:
            >>> query = torch.randn(1000, 3, requires_grad=True)
            >>> data = torch.randn(5000, 3, requires_grad=True)
            >>> indices, weights = knn(query, data, k=32)
            >>> 
            >>> # Use in interpolation
            >>> neighbors = data[indices]  # (1000, 32, 3)
            >>> interpolated = (weights.unsqueeze(-1) * neighbors).sum(dim=1)  # (1000, 3)
            >>> 
            >>> # Gradients flow through weights
            >>> loss = interpolated.sum()
            >>> loss.backward()
            >>> print(query.grad.shape)  # (1000, 3) ✓
            >>> print(data.grad.shape)   # (5000, 3) ✓
        """
        device = query.device
        k = int(min(k, data.shape[0]))
        
        if not (self.use_faiss and FAISS_AVAILABLE):
            return self._torch_soft_knn(query, data, k)
        
        if self.use_soft_radius:
            return self._hybrid_faiss_soft_radius(query, data, k)
        else:
            return self._hybrid_faiss_knn(query, data, k)
    
    def _build_index(self, data: torch.Tensor, D: int, nlist: int, nprobe: int, cache_key):
        """
        Build or retrieve cached FAISS index for given data.
        
        Caching strategy:
            - Key: (M, D, nlist, data_ptr, epoch)
            - If found: return cached index (O(1))
            - If not found: clear cache, build new index, cache it
        
        Index selection logic:
            - If use_ivf=True and M > nlist*39: IVF index (fast for M > 10k)
            - Otherwise: Flat index (exact but slower)
            - GPU acceleration if data.is_cuda=True
        
        Args:
            data: (M, D) database points (on GPU or CPU)
            D: Dimensionality
            nlist: Number of IVF cells (auto-adjusted)
            nprobe: Number of cells to probe
            cache_key: Tuple for cache lookup
        
        Returns:
            index: FAISS index (CPU or GPU)
                   - Trained and populated with data
                   - Ready for .search() calls
        
        Note:
            Data is converted to float32 numpy for FAISS compatibility.
            FAISS operates on CPU numpy arrays even for GPU indices.
        """
        if cache_key in self._index_cache:
            return self._index_cache[cache_key]
        
        self._index_cache.clear()
        data_np = data.detach().cpu().float().numpy()
        N = data_np.shape[0]
        
        # Auto-adjust nlist to avoid degenerate cases
        # Rule: each cell should have ~40 points on average
        nlist_adjusted = max(1, min(nlist, N // 40))
        
        if self.use_ivf:
            index = self._build_ivf_index(data_np, D, N, nlist_adjusted, nprobe, data.is_cuda)
        else:
            index = self._build_flat_index(data_np, D, data.is_cuda)
        
        self._index_cache[cache_key] = index
        return index
    
    def _build_ivf_index(self, data_np: np.ndarray, D: int, N: int, nlist: int, nprobe: int, is_cuda: bool):
        """
        Build IVF (Inverted File) index for faster approximate search.
        
        IVF index structure:
            - Partition space into nlist Voronoi cells using k-means
            - Each cell stores points nearest to its centroid
            - Search: probe nprobe cells, search within those cells
            - Complexity: O(N√M) vs O(N log M) for flat
        
        Training requirement:
            - IVF indices must be trained on sample of data (k-means clustering)
            - Uses min(N, 100k) points for training to limit cost
            - Training is one-time cost, amortized over many searches
        
        Fallback:
            - If N < nlist*39, too few points per cell → use flat index
        
        Args:
            data_np: (N, D) numpy float32 array
            D: Dimensionality
            N: Number of points
            nlist: Number of Voronoi cells
            nprobe: Number of cells to probe during search
            is_cuda: If True, build GPU index; else CPU
        
        Returns:
            index: Trained IVF index (IndexIVFFlat)
                   - GPU index if is_cuda=True
                   - CPU index otherwise
        """
        if N < nlist * 39:
            # Too few points, fall back to flat
            return self._build_flat_index(data_np, D, is_cuda)
        
        if is_cuda:
            return self._build_gpu_ivf_index(data_np, D, N, nlist, nprobe)
        else:
            return self._build_cpu_ivf_index(data_np, D, N, nlist, nprobe)
    
    def _build_gpu_ivf_index(self, data_np: np.ndarray, D: int, N: int, nlist: int, nprobe: int):
        """
        Build GPU-accelerated IVF index.
        
        Process:
            1. Create CPU quantizer (for k-means clustering)
            2. Create CPU IVF index
            3. Train on sample of data (k-means to find centroids)
            4. Add all data points to index
            5. Transfer to GPU using faiss.index_cpu_to_gpu
        
        GPU advantages:
            - 10-100× faster search than CPU for large datasets
            - Batch queries processed in parallel
            - Essential for real-time applications
        
        Args:
            data_np: (N, D) numpy float32 array
            D: Dimensionality
            N: Number of points
            nlist: Number of IVF cells
            nprobe: Cells to probe during search
        
        Returns:
            index: GPU IVF index with nprobe set
        
        Memory:
            - Requires GPU memory for: centroids + point indices + partial distance tables
            - Typical: ~4-8 bytes per point + O(nlist*D) for centroids
        """
        if self._gpu_resources is None:
            self._gpu_resources = faiss.StandardGpuResources()
            
        cpu_quantizer = faiss.IndexFlatL2(D)
        cpu_index = faiss.IndexIVFFlat(cpu_quantizer, D, nlist, faiss.METRIC_L2)
        
        if not cpu_index.is_trained:
            train_size = min(N, 100_000)
            train_sel = np.random.choice(N, train_size, replace=False) if N > train_size else np.arange(N)
            cpu_index.train(data_np[train_sel])
        
        cpu_index.add(data_np)
        index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index)
        index.nprobe = min(nprobe, nlist)
        return index
    
    def _build_cpu_ivf_index(self, data_np: np.ndarray, D: int, N: int, nlist: int, nprobe: int):
        """
        Build CPU IVF index.
        
        Same as GPU version but stays on CPU. Useful when:
        - GPU not available
        - Dataset small enough that CPU is sufficient
        - GPU memory limited
        
        Process identical to _build_gpu_ivf_index but skips GPU transfer.
        
        Args:
            data_np: (N, D) numpy float32 array
            D: Dimensionality
            N: Number of points
            nlist: Number of IVF cells
            nprobe: Cells to probe
        
        Returns:
            index: CPU IVF index with nprobe set
        """
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
        """
        Build flat (brute-force) index for exact k-NN search.
        
        Flat index:
            - No approximation, always returns exact k nearest neighbors
            - Complexity: O(N·M·D) for N queries, M data points, D dimensions
            - Fast for small datasets (M < 10k)
            - Slow for large datasets (M > 100k)
        
        GPU acceleration:
            - GPU flat index 10-50× faster than CPU for M > 10k
            - Essential for real-time on large datasets
        
        Args:
            data_np: (M, D) numpy float32 array
            D: Dimensionality
            is_cuda: If True, create GPU index; else CPU
        
        Returns:
            index: IndexFlatL2 (GPU or CPU)
                   - No training needed
                   - Ready immediately after .add()
        """
        if is_cuda:
            if self._gpu_resources is None:
                self._gpu_resources = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(self._gpu_resources, D)
        else:
            index = faiss.IndexFlatL2(D)
        index.add(data_np)
        return index
    
    def _compute_differentiable_weights(self, query: torch.Tensor, data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable softmax attention weights on gathered neighbors.
        
        This is the key to differentiability in hybrid mode:
            - Indices from FAISS are discrete (no gradient)
            - But we recompute distances on selected neighbors
            - Softmax over distances creates smooth, differentiable weights
            - Gradients flow back to query and data positions
        
        Formula:
            w_i = exp(-||query - neighbor_i|| / tau) / Σ exp(-||query - neighbor_j|| / tau)
        
        Gradient properties:
            - ∂w/∂query: moves query toward high-weight neighbors
            - ∂w/∂data: pulls neighbors toward query proportional to weight
        
        Args:
            query: (N, D) query points with requires_grad=True
            data: (M, D) data points with requires_grad=True
            indices: (N, k) neighbor indices (from FAISS, discrete)
        
        Returns:
            weights: (N, k) softmax attention weights
                     - Sum to 1.0 along dim=1
                     - Differentiable w.r.t. query and data
                     - Clamped to query.dtype for mixed precision
        
        Complexity:
            - O(N·k·D) for distance computation
            - O(N·k) for softmax
            - Much cheaper than full O(N·M·D) brute force
        """
        neigh = data[indices]  # (N, k, D) - maintains gradient
        qx = query.unsqueeze(1).float()  # (N, 1, D)
        dist = torch.norm(qx - neigh.float(), dim=2)  # (N, k) - differentiable
        logits = -dist / self.tau
        weights = F.softmax(logits, dim=1).to(query.dtype)  # (N, k)
        return weights
    
    def _compute_simple_weights(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Compute simple exponential weights without gradient tracking.
        
        Used when neither query nor data requires gradients.
        Avoids autograd overhead for inference-only scenarios.
        
        Formula:
            w_i = exp(-d_i / tau) / Σ exp(-d_j / tau)
        
        Args:
            distances: (N, k) squared L2 distances from FAISS
                       - Already computed, no need to recompute
        
        Returns:
            weights: (N, k) normalized exponential weights
                     - Sum to 1.0 along dim=1
                     - No gradient tracking (pure inference)
        """
        weights = torch.exp(-distances / self.tau)
        weights = weights / (weights.sum(dim=1, keepdim=True) + EPS_SAFE)
        return weights
    
    def _hybrid_faiss_knn(self, query: torch.Tensor, data: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard hybrid mode: FAISS for indices, differentiable weights.
        
        Algorithm:
            1. Build/retrieve FAISS index for data (cached)
            2. Search for exactly k nearest neighbors using FAISS
            3. If gradients needed: recompute distances and softmax weights
            4. Else: use simple exponential weights (faster)
        
        This is the most common mode:
            - Fast: FAISS gives O(N log M) or O(N √M) search
            - Exact k neighbors: no over-fetching
            - Differentiable: weights flow gradients
        
        Caching:
            - Index cached by (M, D, nlist, data_ptr, epoch)
            - Rebuilds only when data changes or invalidate_cache() called
        
        Args:
            query: (N, D) query points
            data: (M, D) database points
            k: Number of neighbors
        
        Returns:
            indices: (N, k) neighbor indices (hard selection from FAISS)
            weights: (N, k) softmax weights (differentiable)
        
        Complexity:
            Forward:  O(N log M) or O(N √M) for FAISS search + O(N·k·D) for weights
            Backward: O(N·k·D) for gradient computation through softmax
        """
        N, D = query.shape
        M = data.shape[0]
        
        # Auto-tune nlist based on dataset size
        nlist = min(self.nlist, max(4, M // 100))
        nprobe = min(self.nprobe, nlist)
        
        # Cache key includes data pointer to detect changes
        data_ptr = int(data.untyped_storage().data_ptr())
        cache_key = (M, D, nlist, data_ptr, self._epoch)
        
        # Build or retrieve index
        index = self._build_index(data, D, nlist, nprobe, cache_key)
        
        # FAISS search (on CPU numpy)
        q_np = query.detach().cpu().float().numpy()
        d_np, i_np = index.search(q_np, k)
        distances = torch.from_numpy(d_np).to(query.device)
        indices = torch.from_numpy(i_np).to(query.device, dtype=torch.long)
        
        # Compute weights (differentiable or simple)
        if query.requires_grad or data.requires_grad:
            weights = self._compute_differentiable_weights(query, data, indices)
        else:
            weights = self._compute_simple_weights(distances)
        
        return indices, weights
    
    def _hybrid_faiss_soft_radius(self, query: torch.Tensor, data: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Soft-radius mode: fetch larger pool, select top-k by soft weight.
        
        Algorithm:
            1. FAISS searches for Kc >> k candidates (e.g., 128)
            2. Compute softmax weights over all Kc candidates
            3. Hard-select top-k indices by weight (straight-through)
            4. Renormalize weights for selected k neighbors
        
        Advantages over standard mode:
            - More flexible neighbor selection (gradient-influenced)
            - Can adapt neighbor set based on learned features
            - Better for tasks where neighbor quality matters more than distance
        
        Disadvantages:
            - Slightly slower (more FAISS results to process)
            - Uses more memory (Kc > k candidates)
        
        Straight-through estimator:
            - Forward: Hard selection of top-k by weight (discrete)
            - Backward: Gradients flow through soft weights (continuous)
            - Allows learning which neighbors to select
        
        Args:
            query: (N, D) query points
            data: (M, D) database points
            k: Final number of neighbors to return
        
        Returns:
            indices: (N, k) selected neighbor indices
                     - Hard selection but surrounded by soft weights
            weights: (N, k) renormalized softmax weights
                     - Differentiable w.r.t. query and data
        
        Complexity:
            - FAISS search: O(N log M) for Kc candidates
            - Weight computation: O(N·Kc·D)
            - Selection: O(N·Kc log k) for top-k
        """
        N, D = query.shape
        M = data.shape[0]
        Kc = min(self.soft_radius_candidates, M)
        
        # Build index (same as standard mode)
        nlist = min(self.nlist, max(4, M // 100))
        nprobe = min(self.nprobe, nlist)
        data_ptr = int(data.untyped_storage().data_ptr())
        cache_key = (M, D, nlist, data_ptr, self._epoch)
        
        index = self._build_index(data, D, nlist, nprobe, cache_key)
        
        # Fetch Kc candidates
        q_np = query.detach().cpu().float().numpy()
        d_np, i_np = index.search(q_np, Kc)
        distances = torch.from_numpy(d_np).to(query.device)
        idx_all = torch.from_numpy(i_np).to(query.device, dtype=torch.long)
        
        # Compute weights over all Kc candidates
        if query.requires_grad or data.requires_grad:
            neigh = data[idx_all]  # (N, Kc, D)
            qx = query.unsqueeze(1).float()
            dist = torch.norm(qx - neigh.float(), dim=2)  # (N, Kc)
            logits = -dist / self.tau
            w_all = F.softmax(logits, dim=1)  # Fully differentiable
        else:
            w_all = torch.exp(-distances / self.tau)
            w_all = w_all / (w_all.sum(dim=1, keepdim=True) + EPS_SAFE)
        
        # Hard selection of top-k by weight (straight-through)
        with torch.no_grad():
            _, topj_hard = torch.topk(w_all, k=min(k, Kc), dim=1)
        
        # Gather soft weights and indices for selected k
        batch = torch.arange(N, device=query.device).unsqueeze(1).expand(-1, topj_hard.shape[1])
        topw = w_all[batch, topj_hard]  # Maintains gradient connection
        indices = idx_all[batch, topj_hard]
        
        # Renormalize weights to sum to 1
        weights = topw / (topw.sum(dim=1, keepdim=True) + EPS_SAFE)
        
        if query.requires_grad or data.requires_grad:
            weights = weights.to(query.dtype)
        
        return indices, weights
    
    def _torch_soft_knn_chunked(
        self, 
        query: torch.Tensor, 
        data: torch.Tensor, 
        k: int,
        chunk_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Chunked pure torch KNN for memory-efficient fallback.
        
        Problem:
            - Full distance matrix is N×M which can exceed GPU memory
            - Example: N=100k, M=500k, D=3 → 100k×500k×4 bytes = 200GB!
        
        Solution:
            - Process queries in chunks of size chunk_size
            - Each chunk computes chunk_size×M distance matrix
            - Memory: chunk_size×M instead of N×M
            - Example: chunk=5000 → 5k×500k×4 = 10GB (feasible)
        
        Algorithm:
            1. Split queries into chunks
            2. For each chunk:
               a. Compute distances to all M data points
               b. Softmax to get attention weights
               c. Hard-select top-k by weight (straight-through)
               d. Renormalize selected weights
               e. Free memory immediately
            3. Concatenate results
        
        Trade-offs:
            - Memory: O(chunk_size × M) vs O(N × M)
            - Speed: Slightly slower due to loop overhead
            - Differentiability: Fully maintained
        
        Args:
            query: (N, D) query points
            data: (M, D) database points
            k: Number of neighbors
            chunk_size: Size of query chunks (auto-tuned if called from _torch_soft_knn)
        
        Returns:
            indices: (N, k) neighbor indices
            weights: (N, k) softmax weights
        
        Memory savings:
            - Full: N×M×4 bytes
            - Chunked: chunk_size×M×4 bytes per iteration
            - Peak: max(chunk_size×M, k×N) typically
        """
        N = query.shape[0]
        M = data.shape[0]
        device = query.device
        
        all_indices = []
        all_weights = []
        
        for i in range(0, N, chunk_size):
            chunk_end = min(i + chunk_size, N)
            chunk = query[i:chunk_end]  # (chunk_size, D)
            
            # Compute distances for this chunk only
            D = torch.cdist(chunk, data, p=2)  # (chunk_size, M)
            logits = -D / self.tau
            attn = F.softmax(logits, dim=1)  # (chunk_size, M)
            
            # Hard index selection (no grad) but soft weights (with grad)
            with torch.no_grad():
                _, topi_hard = torch.topk(attn, k=k, dim=1)
            
            # Gather soft weights for selected indices
            chunk_n = chunk_end - i
            batch_idx = torch.arange(chunk_n, device=device).unsqueeze(1).expand(-1, k)
            topw = attn[batch_idx, topi_hard]
            
            # Renormalize
            weights = topw / (topw.sum(dim=1, keepdim=True) + EPS_SAFE)
            
            all_indices.append(topi_hard)
            all_weights.append(weights)
            
            # Aggressive memory cleanup
            del D, logits, attn
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return torch.cat(all_indices), torch.cat(all_weights)
    
    def _torch_soft_knn(self, query: torch.Tensor, data: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pure PyTorch fallback with automatic memory management.
        
        Used when:
            - FAISS not available
            - use_faiss=False
            - Small datasets where FAISS overhead not worth it
        
        Algorithm:
            1. Estimate memory footprint of full N×M distance matrix
            2. If > 1GB threshold: use chunked version
            3. Else: use fast single-pass version
        
        Fast version (small data):
            - Compute full N×M distance matrix
            - Softmax over all M points per query
            - Hard-select top-k by weight (straight-through)
            - Renormalize selected weights
        
        Chunked version (large data):
            - See _torch_soft_knn_chunked docstring
        
        Differentiability:
            - Full autograd support through softmax attention
            - Straight-through estimator for index selection
            - Gradients flow to query and data positions
        
        Args:
            query: (N, D) query points
            data: (M, D) database points
            k: Number of neighbors
        
        Returns:
            indices: (N, k) neighbor indices
            weights: (N, k) normalized attention weights
        
        Complexity:
            - Full version: O(N·M·D) forward, O(N·M·D) backward
            - Chunked version: Same but split across iterations
        
        Memory:
            - Full: O(N·M) for distance matrix
            - Chunked: O(chunk_size·M) per iteration
            - Auto-selects based on 1GB threshold
        """
        N, M = query.shape[0], data.shape[0]
        
        # Estimate memory usage and decide strategy
        full_memory = N * M * 4  # bytes (float32)
        memory_threshold = 1e9  # 1GB
        
        if full_memory > memory_threshold:
            # Large dataset: use chunked version to avoid OOM
            chunk_size = max(1, int(memory_threshold / (M * 4)))
            chunk_size = min(chunk_size, self.fallback_chunk_size)
            return self._torch_soft_knn_chunked(query, data, k, chunk_size)
        
        # Small dataset: fast single-pass version
        D = torch.cdist(query, data, p=2)  # (N, M)
        logits = -D / self.tau
        attn = F.softmax(logits, dim=1)  # (N, M) - fully differentiable
        
        # Hard index selection (no grad) but soft weights (with grad)
        with torch.no_grad():
            _, topi_hard = torch.topk(attn, k=k, dim=1)
        
        # Gather soft weights for selected indices
        batch_idx = torch.arange(N, device=query.device).unsqueeze(1).expand(-1, k)
        topw = attn[batch_idx, topi_hard]
        
        # Renormalize weights
        weights = topw / (topw.sum(dim=1, keepdim=True) + EPS_SAFE)
        
        # Straight-through: hard indices, soft gradients
        topi = topi_hard
        
        return topi, weights