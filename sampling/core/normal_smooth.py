"""
Laplacian-based normal smoothing with differentiable operations.

This module implements iterative normal smoothing using spatial neighborhoods
and adaptive bandwidth estimation. All operations are fully differentiable
for end-to-end learning.

Key features:
- Spatial neighbor-based smoothing (not just topology)
- Adaptive bandwidth via soft median (differentiable)
- Gaussian kernel weighting by distance
- EMA-style blending for stability
"""

import torch
from typing import Dict
from ..utils.config import EPS_SAFE
from ..utils.utils import normalize


# def soft_median(x: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
#     """
#     Differentiable soft median approximation using smooth ranking.
    
#     Classical median is non-differentiable (discrete selection).
#     This soft version uses continuous weighting around the median position.
    
#     Algorithm:
#         1. For each element, compute soft rank via sigmoid comparisons
#         2. Weight elements by distance to median rank (K/2)
#         3. Return weighted average of elements near median
    
#     Mathematical formulation:
#         For each element xᵢ:
#             soft_rank(xᵢ) = Σⱼ sigmoid((xⱼ - xᵢ)/τ)
        
#         Weight by distance to median:
#             wᵢ = exp(-|soft_rank(xᵢ) - K/2| / τ)
        
#         Soft median:
#             m̃ = Σᵢ (wᵢ / Σⱼwⱼ) · xᵢ
    
#     Why soft median for bandwidth:
#         - More robust than mean (less affected by outliers)
#         - More stable than min/max (no extreme values)
#         - Differentiable (unlike hard median)
#         - Preserves gradient flow for learning
    
#     Differentiability:
#         - sigmoid for comparisons: smooth and differentiable ✓
#         - exp for weighting: differentiable ✓
#         - No discrete selections (argmin/argmax) ✓
#         - Gradients flow through all elements
    
#     Temperature parameter τ:
#         - Lower τ (e.g., 0.01): sharper, closer to hard median
#         - Higher τ (e.g., 0.1): smoother, closer to mean
#         - Default 0.05: good balance for most cases
    
#     Args:
#         x: (N, K) values to compute median over
#            - N: number of independent groups (e.g., points)
#            - K: number of values per group (e.g., neighbor distances)
#         tau: Temperature for soft comparisons
#              - Controls smoothness of approximation
#              - Should be small relative to value range
    
#     Returns:
#         median: (N, 1) soft median values
#                 - One median per row
#                 - Differentiable w.r.t. input x
#                 - Approximately equals hard median for small tau
    
#     Complexity:
#         - Time: O(N·K²) for pairwise comparisons
#         - Memory: O(N·K²) for comparison matrix
#         - Can be slow for large K (e.g., K > 100)
    
#     Example:
#         >>> x = torch.randn(1000, 32, requires_grad=True)
#         >>> soft_med = soft_median(x, tau=0.05)
#         >>> soft_med.shape
#         torch.Size([1000, 1])
#         >>> 
#         >>> # Check differentiability
#         >>> loss = soft_med.sum()
#         >>> loss.backward()
#         >>> assert x.grad is not None  # Gradients flow ✓
#         >>> 
#         >>> # Compare with hard median
#         >>> hard_med = torch.median(x, dim=1, keepdim=True)[0]
#         >>> torch.allclose(soft_med, hard_med, atol=0.1)  # Close but not exact
#         True
    
#     Notes:
#         - Processes each row independently (no batched operations yet)
#         - Could be optimized with parallel computation for large N
#         - Trade-off: differentiability vs computational cost
#     """
#     N, K = x.shape
    
#     # Process each row independently
#     result = []
#     for i in range(N):
#         xi = x[i]  # (K,) - values for this row
        
#         # Step 1: Compute pairwise comparisons
#         xi_expanded = xi.unsqueeze(0)  # (1, K)
#         xi_diff = xi_expanded.T - xi_expanded  # (K, K) - all pairwise differences
        
#         # Step 2: Soft ranking via sigmoid
#         # P[i,j] ≈ 1 if xi[i] > xi[j], ≈ 0 if xi[i] < xi[j]
#         P = torch.sigmoid(xi_diff / tau)  # (K, K)
#         soft_rank = P.sum(dim=1)  # (K,) - how many elements are smaller
        
#         # Step 3: Weight elements near median rank
#         median_rank = K / 2.0  # Median position
#         dist_to_median = torch.abs(soft_rank - median_rank)  # (K,)
#         weights = torch.exp(-dist_to_median / tau)  # (K,) - Gaussian-like weights
#         weights = weights / (weights.sum() + EPS_SAFE)  # Normalize
        
#         # Step 4: Weighted average of elements near median
#         median_val = (weights * xi).sum()  # Scalar
#         result.append(median_val)
    
#     return torch.stack(result).unsqueeze(1)  # (N, 1)


def soft_median(x: torch.Tensor, sigma_idx: float = 1.0) -> torch.Tensor:
    """
    Vectorized soft median via index-space Gaussian weighting. O(N·K).
    Args:
        x: (N, K) values per row (each row smoothed independently)
        sigma_idx: standard deviation in rank space (controls smoothness);
                   ~0.8–1.2 works well for K≈24
    Returns:
        (N, 1) soft median per row
    """
    N, K = x.shape
    device, dtype = x.device, x.dtype

    # (1) Sort values within each row in ascending order.
    #     d_sorted[n, :] holds the sorted values of row n.
    d_sorted, _ = torch.sort(x, dim=1)  # (N, K)

    # (2) Build Gaussian weights in rank space centered at the median rank r = (K-1)/2.
    #     This avoids pairwise comparisons and yields a smooth, differentiable
    #     approximation to the median by emphasizing middle-ranked elements.
    j = torch.arange(K, device=device, dtype=dtype).view(1, K)  # (1, K), rank indices 0..K-1
    r = (K - 1) / 2.0                                           # scalar, median rank in index space
    #     sigma_idx controls how wide the kernel is over ranks (larger ⇒ smoother).
    w = torch.exp(-0.5 * ((j - r) / max(sigma_idx, 1e-6)) ** 2)  # (1, K)
    w = w / (w.sum(dim=1, keepdim=True) + EPS_SAFE)              # normalize weights along rank axis

    # (3) Soft median = weighted average of the sorted values using the rank-Gaussian weights.
    med = (d_sorted * w).sum(dim=1, keepdim=True)  # (N, 1)
    return med



def compute_spatial_weights(
    positions: torch.Tensor,
    neighbor_positions: torch.Tensor,
    bandwidth: torch.Tensor
) -> torch.Tensor:
    """
    Compute spatial Gaussian weights based on distance.
    
    Gaussian kernel weighting is standard in spatial smoothing:
        w(x, xᵢ) = exp(-||x - xᵢ||² / h²)
    
    where:
        - x: query position
        - xᵢ: neighbor position
        - h: bandwidth (controls influence radius)
    
    Properties:
        - Nearby points get high weight (smooth influence)
        - Far points get low weight (local operation)
        - h controls locality (small h = local, large h = global)
    
    Args:
        positions: (N, 3) query positions
        neighbor_positions: (N, k, 3) neighbor positions
        bandwidth: (N, 1) adaptive bandwidth per point
    
    Returns:
        weights: (N, k) spatial weights
                 - Higher for closer neighbors
                 - Differentiable w.r.t. all inputs
    """
    # Compute spatial distances
    diff = neighbor_positions - positions.unsqueeze(1)  # (N, k, 3)
    dist = torch.norm(diff, dim=-1)  # (N, k)
    
    # Gaussian kernel with adaptive bandwidth
    weights = torch.exp(-(dist / bandwidth) ** 2)  # (N, k)
    
    return weights


def mask_self_neighbor(weights: torch.Tensor) -> torch.Tensor:
    """
    Mask out self-neighbor (first neighbor) for smoothing.
    
    In KNN with k neighbors, the first neighbor is typically the point itself
    (distance = 0). For smoothing, we want to use only true neighbors, not self.
    
    Implementation:
        - Creates mask with 0 for first column, 1 for rest
        - Multiplies weights element-wise (out-of-place for autograd)
    
    Differentiability:
        - Uses element-wise multiplication (not in-place)
        - Gradients flow through mask operation ✓
        - No discrete branching
    
    Args:
        weights: (N, k) weights where first column is self
    
    Returns:
        masked_weights: (N, k) weights with self zeroed out
                        - weights[:, 0] = 0
                        - weights[:, 1:] unchanged
    """
    mask = torch.ones_like(weights)
    mask[:, 0] = 0.0  # Zero out first neighbor (self)
    return weights * mask


def smooth_normals(
    normals: torch.Tensor,
    positions: torch.Tensor,
    knn,
    cfg: Dict
) -> torch.Tensor:
    """
    Iterative normal smoothing using spatial Laplacian with adaptive bandwidth.
    
    Classical Laplacian smoothing updates each normal to be closer to its
    neighbors' average. This implementation adds:
        1. Spatial weighting (closer neighbors have more influence)
        2. Adaptive bandwidth (dense regions → smaller radius)
        3. EMA-style blending (stability over iterations)
    
    Algorithm per iteration:
        1. Find k spatial nearest neighbors for each point
        2. Estimate adaptive bandwidth h via soft median of distances
        3. Compute spatial weights: w(d) = exp(-(d/h)²)
        4. Combine with KNN attention weights
        5. Smooth: n'ᵢ = Σⱼ wⱼnⱼ (weighted average of neighbor normals)
        6. Blend: nᵢ ← λn'ᵢ + (1-λ)nᵢ (EMA update)
        7. Normalize: nᵢ ← nᵢ / ||nᵢ||
    
    Mathematical formulation:
        At iteration t:
            n̄ᵢ⁽ᵗ⁾ = Σⱼ wᵢⱼ · nⱼ⁽ᵗ⁻¹⁾
            nᵢ⁽ᵗ⁾ = normalize(λn̄ᵢ⁽ᵗ⁾ + (1-λ)nᵢ⁽ᵗ⁻¹⁾)
        
        where:
            wᵢⱼ = exp(-dᵢⱼ²/hᵢ²) · aᵢⱼ (spatial × attention)
            hᵢ = soft_median({dᵢⱼ}ⱼ₌₂..ₖ) (adaptive bandwidth)
    
    Why spatial smoothing:
        - Topology-agnostic (works without mesh connectivity)
        - Handles noise and outliers robustly
        - Preserves sharp features (small h in high-curvature regions)
        - Naturally adapts to point density
    
    Adaptive bandwidth benefits:
        - Dense regions: smaller h → more local smoothing
        - Sparse regions: larger h → wider smoothing
        - Automatically scales with point cloud resolution
        - Prevents over-smoothing in detailed areas
    
    EMA blending (λ parameter):
        - λ = 1.0: full Laplacian (aggressive smoothing)
        - λ = 0.5: equal blend (balanced)
        - λ = 0.8: typical value (smooth but stable)
        - Higher λ: faster convergence but less stable
    
    Differentiability:
        All operations are differentiable:
        - KNN search: soft attention weights ✓
        - Soft median: continuous approximation ✓
        - Gaussian weights: exp function ✓
        - Weighted averaging: linear combination ✓
        - Normalization: differentiable (except at origin) ✓
        
        Enables:
        - Learning optimal smoothing parameters (iters, λ, k)
        - Joint optimization with point positions
        - End-to-end surface reconstruction
    
    Args:
        normals: (N, 3) input surface normals
                 - Should be unit vectors (will be normalized if not)
                 - Can have requires_grad=True for learning
        positions: (N, 3) spatial positions for neighbor finding
                   - Used to define spatial neighborhoods
                   - Should match normals (same point cloud)
        knn: KNN function (e.g., HybridFAISSKNN)
             - Returns (indices, weights) for k nearest neighbors
             - Must support differentiable weights
        cfg: Configuration dictionary with keys:
             - 'iters' (int): Number of smoothing iterations (default: 2)
             - 'k' (int): Number of spatial neighbors (default: 16)
             - 'lambda_smooth' (float): EMA blend factor (default: 0.8)
    
    Returns:
        smoothed_normals: (N, 3) smoothed unit normals
                          - Same shape as input
                          - More spatially coherent (less noisy)
                          - Differentiable w.r.t. input normals and positions
    
    Complexity:
        Per iteration:
        - KNN search: O(N log N) with FAISS or O(N²) fallback
        - Soft median: O(N·k²) for bandwidth estimation
        - Smoothing: O(N·k·3) for weighted averaging
        
        Total: O(iters · N · (log N + k²))
    
    Example:
        >>> # Setup
        >>> positions = torch.randn(10000, 3)
        >>> normals = torch.randn(10000, 3)
        >>> normals = normalize(normals)  # Ensure unit normals
        >>> 
        >>> knn = HybridFAISSKNN(tau=0.15)
        >>> cfg = {
        ...     'iters': 2,
        ...     'k': 16,
        ...     'lambda_smooth': 0.8
        ... }
        >>> 
        >>> # Smooth normals
        >>> smooth_n = smooth_normals(normals, positions, knn, cfg)
        >>> 
        >>> # Check outputs
        >>> smooth_n.shape  # (10000, 3)
        >>> torch.allclose(torch.norm(smooth_n, dim=1), torch.ones(10000))
        True  # Still unit vectors ✓
        >>> 
        >>> # Smoother than input (lower variance)
        >>> normals.std() > smooth_n.std()
        True
    
    Typical parameter ranges:
        - iters: 1-3 (more iterations = smoother but slower)
        - k: 8-32 (more neighbors = smoother but less local)
        - lambda_smooth: 0.5-0.9 (higher = more aggressive smoothing)
    
    Notes:
        - First neighbor (self) is masked out in smoothing
        - Bandwidth estimated from k-1 neighbors (excluding self)
        - Normals normalized after each iteration for stability
        - Can be iterated further if needed (just call again)
    """
    # Parse configuration
    iters = int(cfg.get("iters", 2))
    k = int(cfg.get("k", 16))
    lambda_smooth = float(cfg.get("lambda_smooth", 0.8))
    sigma_idx = float(cfg.get("sigma_idx", 1.0))
    
    # Initialize smoothed normals (will be updated in-place conceptually)
    normals_smooth = normals.clone()
    
    # Iterative smoothing
    for t in range(iters):
        # Step 1: Find spatial neighbors
        # Uses current positions (fixed) to define neighborhoods
        idx, w = knn(positions, positions, k)  # (N, k), (N, k)
        
        # Step 2: Gather neighbor normals
        neighbor_normals = normals_smooth[idx]  # (N, k, 3)
        
        # Step 3: Compute spatial distances for adaptive weighting
        neighbor_positions = positions[idx]  # (N, k, 3)
        diff = neighbor_positions - positions.unsqueeze(1)  # (N, k, 3)
        dist = torch.norm(diff, dim=-1)  # (N, k)
        
        # Step 4: Estimate adaptive bandwidth via soft median
        # Use neighbors 1:k (exclude self at index 0)
        h = soft_median(dist[:, 1:], sigma_idx=sigma_idx) + EPS_SAFE  # (N, 1)
        
        # Step 5: Compute spatial weights (Gaussian kernel)
        spatial_weights = compute_spatial_weights(
            positions, neighbor_positions, h
        )  # (N, k)
        
        # Step 6: Mask out self-neighbor
        spatial_weights = mask_self_neighbor(spatial_weights)  # (N, k)
        
        # Step 7: Combine with KNN attention weights
        # Multiply: spatial proximity × learned attention
        combined_weights = spatial_weights * w  # (N, k)
        combined_weights = combined_weights / (
            combined_weights.sum(dim=1, keepdim=True) + EPS_SAFE
        )  # Renormalize
        
        # Step 8: Weighted average of neighbor normals
        avg_normal = (combined_weights.unsqueeze(-1) * neighbor_normals).sum(dim=1)  # (N, 3)
        avg_normal = normalize(avg_normal, eps=EPS_SAFE)  # Unit vector
        
        # Step 9: EMA-style blending for stability
        # Blend between new average and previous normal
        normals_smooth = lambda_smooth * avg_normal + (1 - lambda_smooth) * normals_smooth
        normals_smooth = normalize(normals_smooth, eps=EPS_SAFE)  # Maintain unit length
    
    return normals_smooth


__all__ = [
    'smooth_normals',
    'soft_median',
    'compute_spatial_weights',
    'mask_self_neighbor',
]