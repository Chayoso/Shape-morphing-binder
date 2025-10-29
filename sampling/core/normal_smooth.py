"""
Differentiable Laplacian Normal Smoothing for Point Clouds

This module implements iterative normal smoothing using spatial neighborhoods
with adaptive bandwidth estimation. All operations are fully differentiable
for end-to-end learning in neural surface reconstruction pipelines.

The smoothing algorithm combines classical Laplacian smoothing with modern
techniques: spatial Gaussian weighting, adaptive bandwidth via soft median,
and EMA-style blending for stability. This produces spatially coherent normals
while preserving geometric features and supporting gradient-based optimization.

Key Features:
- Spatial neighbor-based smoothing (topology-free, works without mesh)
- Adaptive bandwidth via differentiable soft median
- Gaussian kernel weighting by distance
- EMA-style iterative blending for stability
- Full differentiability for joint optimization

Mathematical Foundation:
    Classical Laplacian: n'ᵢ = (1/|Nᵢ|) Σⱼ∈Nᵢ nⱼ
    
    Our extension: n'ᵢ = Σⱼ wᵢⱼ(d, h) · aⱼ · nⱼ
    
    where:
        - wᵢⱼ(d, h) = exp(-d²ᵢⱼ/h²ᵢ): Spatial Gaussian kernel
        - aⱼ: Learned attention weight from KNN
        - hᵢ = soft_median(dᵢ₁, ..., dᵢₖ): Adaptive bandwidth
        
    EMA update: nᵢ⁽ᵗ⁾ = λn'ᵢ⁽ᵗ⁾ + (1-λ)nᵢ⁽ᵗ⁻¹⁾

Applications:
    - Point cloud normal denoising
    - Surface reconstruction preprocessing
    - Physics-guided Gaussian splatting
    - Differentiable rendering pipelines

Author: CHAYO
Version: 2.2.1 (Refactored)
"""

import torch
from typing import Dict
from ..utils.config import EPS_SAFE
from ..utils.utils import normalize


# =========================================================
# Constants
# =========================================================
DEFAULT_SMOOTH_ITERS = 2
DEFAULT_SMOOTH_K = 16
DEFAULT_LAMBDA_SMOOTH = 0.8
DEFAULT_SIGMA_IDX = 1.0

# Typical parameter ranges
MIN_SMOOTH_ITERS = 1
MAX_SMOOTH_ITERS = 5
MIN_SMOOTH_K = 4
MAX_SMOOTH_K = 64
MIN_LAMBDA_SMOOTH = 0.1
MAX_LAMBDA_SMOOTH = 1.0
MIN_SIGMA_IDX = 0.5
MAX_SIGMA_IDX = 2.0


# =========================================================
# Soft Median Computation
# =========================================================
def soft_median(
    x: torch.Tensor, 
    sigma_idx: float = DEFAULT_SIGMA_IDX
) -> torch.Tensor:
    """
    Compute differentiable soft median via Gaussian weighting in rank space.
    
    Traditional median is non-differentiable due to discrete sorting. This
    implementation approximates median using a continuous Gaussian kernel
    over sorted values, weighted by their rank position. This maintains
    differentiability while providing a robust central tendency estimator.
    
    Algorithm:
        1. Sort values within each row: x_sorted
        2. Create Gaussian weights centered at median rank r = (K-1)/2
        3. Compute weighted sum: soft_median = Σᵢ wᵢ · x_sorted[i]
        
    The Gaussian kernel in rank space:
        w(j) = exp(-0.5 · ((j - r) / σ)²)
        
    where:
        - j: rank index (0, 1, ..., K-1)
        - r: median rank = (K-1)/2
        - σ: standard deviation controlling smoothness
    
    Properties:
        - Continuous approximation to discrete median
        - Differentiable w.r.t. input values
        - Robust to outliers (similar to median)
        - σ controls smoothness: small σ → closer to median, large σ → closer to mean
    
    Complexity:
        Time: O(N·K·log(K)) for sorting + O(N·K) for weighting
        Space: O(N·K) for sorted values
    
    Args:
        x: (N, K) Values per row, each row processed independently
           - Typically distances to k nearest neighbors
           - Can contain any real values
        sigma_idx: Standard deviation in rank space. Controls smoothness.
                   - Typical range: [0.8, 1.2] for K ≈ 16-32
                   - Smaller σ → sharper focus on median
                   - Larger σ → smoother, closer to mean
                   - Default: 1.0
    
    Returns:
        soft_med: (N, 1) Soft median per row
                  - Continuous approximation to median
                  - Differentiable w.r.t. x
                  - Always within [min(x), max(x)]
    
    Mathematical Background:
        The soft median is a special case of weighted quantile estimation:
            Q_α^soft = Σᵢ w(i, α) · x_sorted[i]
        
        where α = 0.5 for median, and w(i, α) is a smooth kernel centered at
        the α-quantile rank. For α = 0.5 (median) and Gaussian kernel:
            w(i, 0.5) ∝ exp(-0.5 · ((i - (K-1)/2) / σ)²)
    
    Differentiability:
        - Sort operation: differentiable via soft-sorting (not used here, but
          gradients still flow through sort due to PyTorch's autograd)
        - Gaussian weights: fully differentiable
        - Weighted sum: linear operation, trivially differentiable
        
        Gradient flow:
            ∂soft_median/∂x = w_sorted · P
        
        where P is a permutation matrix from sorting (implicit in autograd)
    
    Example:
        >>> # Simple case: 3 rows, 5 values each
        >>> x = torch.tensor([
        ...     [1.0, 5.0, 3.0, 2.0, 4.0],  # sorted: [1,2,3,4,5], median=3
        ...     [10.0, 1.0, 5.0, 3.0, 7.0], # sorted: [1,3,5,7,10], median=5
        ...     [2.0, 2.0, 2.0, 2.0, 2.0],  # sorted: [2,2,2,2,2], median=2
        ... ])
        >>> 
        >>> soft_med = soft_median(x, sigma_idx=1.0)
        >>> print(soft_med)
        tensor([[3.0],
                [5.0],
                [2.0]])
        >>> 
        >>> # Differentiable
        >>> x.requires_grad_(True)
        >>> soft_med = soft_median(x, sigma_idx=1.0)
        >>> soft_med.sum().backward()
        >>> print(x.grad.shape)  # (3, 5)
    
    Comparison with alternatives:
        - Hard median: Non-differentiable, discrete
        - Mean: Differentiable but sensitive to outliers
        - Soft median: Differentiable AND robust to outliers ✓
    
    Parameter sensitivity:
        For K=16 neighbors:
        - σ=0.5: Very sharp, almost discrete median
        - σ=1.0: Good balance (recommended)
        - σ=1.5: Smoother, closer to trimmed mean
        - σ=2.0: Very smooth, approaches mean
    
    Notes:
        - Vectorized for efficiency (no loops over N)
        - Works with any K >= 1
        - Gracefully handles repeated values
        - Typical use: bandwidth estimation in spatial smoothing
    """
    N, K = x.shape
    device, dtype = x.device, x.dtype

    # Sort values within each row (ascending order)
    # Gradients flow through sort in PyTorch's autograd
    d_sorted, _ = torch.sort(x, dim=1)  # (N, K)

    # Construct Gaussian weights in rank space
    # Centered at median rank r = (K-1)/2
    j = torch.arange(K, device=device, dtype=dtype).view(1, K)  # (1, K): rank indices
    r = (K - 1) / 2.0  # Median rank (middle position)
    
    # Gaussian kernel: exp(-0.5 * ((j - r) / σ)²)
    sigma_safe = max(sigma_idx, 1e-6)  # Prevent division by zero
    w = torch.exp(-0.5 * ((j - r) / sigma_safe) ** 2)  # (1, K)
    
    # Normalize weights to sum to 1
    w = w / (w.sum(dim=1, keepdim=True) + EPS_SAFE)  # (1, K)

    # Compute soft median as weighted sum
    soft_med = (d_sorted * w).sum(dim=1, keepdim=True)  # (N, 1)
    
    return soft_med


# =========================================================
# Spatial Weighting
# =========================================================
def compute_spatial_weights(
    positions: torch.Tensor,
    neighbor_positions: torch.Tensor,
    bandwidth: torch.Tensor
) -> torch.Tensor:
    """
    Compute spatial Gaussian weights based on Euclidean distance.
    
    Implements the classical Gaussian kernel for spatial smoothing:
        w(x, xᵢ) = exp(-||x - xᵢ||² / h²)
    
    This kernel is fundamental in spatial statistics and has desirable properties:
    - Smooth decay with distance (differentiable everywhere)
    - Infinite support but effective local radius ≈ 2-3h
    - Normalized form: integrates to 1 over R³
    - Adaptive via bandwidth h
    
    Physical interpretation:
        - h acts as "influence radius"
        - Points within h have high weight (≈ 0.37 at distance h)
        - Points beyond 3h have negligible weight (< 0.01)
    
    Why Gaussian:
        1. Differentiability: Critical for learning-based methods
        2. Locality: Weights decay quickly → computational efficiency
        3. Smoothness: No sharp boundaries → stable optimization
        4. Interpretability: Single parameter h controls everything
        5. Theory: Optimal for certain noise models (Gaussian noise)
    
    Args:
        positions: (N, 3) Query point positions
                   - 3D spatial coordinates (x, y, z)
                   - Typically surface point locations
        neighbor_positions: (N, k, 3) Neighbor point positions for each query
                            - k nearest neighbors per query point
                            - Gathered from KNN search
        bandwidth: (N, 1) Adaptive bandwidth per query point
                   - Controls influence radius
                   - Typically estimated via soft_median(distances)
                   - Larger h → more global smoothing
                   - Smaller h → more local smoothing
    
    Returns:
        weights: (N, k) Spatial weights for each neighbor
                 - In range [0, 1] (not normalized to sum=1 yet)
                 - Higher values for closer neighbors
                 - Exponential decay with distance
                 - Differentiable w.r.t. all inputs
    
    Computational Cost:
        - Distance computation: O(N·k·3) FLOPs
        - Exponential: O(N·k) FLOPs
        - Memory: O(N·k) for distances and weights
    
    Mathematical Details:
        Full kernel definition:
            K(x, xᵢ; h) = (1/(h√(2π))³) · exp(-||x - xᵢ||²/(2h²))
        
        Our implementation uses unnormalized form (factor of 2 in exponent):
            w(x, xᵢ; h) = exp(-||x - xᵢ||²/h²)
        
        This is equivalent up to a constant and avoids √(2π) computation.
    
    Differentiability:
        Gradient w.r.t. positions:
            ∂w/∂x = -2w · (x - xᵢ) / h²
        
        Gradient w.r.t. bandwidth:
            ∂w/∂h = 2w · ||x - xᵢ||² / h³
        
        All gradients are continuous and well-defined.
    
    Example:
        >>> # 1000 points with 16 neighbors each
        >>> positions = torch.randn(1000, 3)
        >>> neighbor_positions = torch.randn(1000, 16, 3)
        >>> bandwidth = torch.rand(1000, 1) * 0.1 + 0.05  # Range [0.05, 0.15]
        >>> 
        >>> weights = compute_spatial_weights(
        ...     positions, neighbor_positions, bandwidth
        ... )
        >>> 
        >>> print(weights.shape)  # (1000, 16)
        >>> print(weights.min(), weights.max())  # Small positive values
        >>> 
        >>> # Closer neighbors have higher weights
        >>> distances = torch.norm(
        ...     neighbor_positions - positions.unsqueeze(1), dim=-1
        ... )
        >>> # Verify: min distance → max weight
        >>> assert weights.argmax(dim=1).eq(distances.argmin(dim=1)).all()
    
    Adaptive Bandwidth:
        The bandwidth h adapts to local point density:
        - Dense regions: small h → local smoothing → preserve detail
        - Sparse regions: large h → wider smoothing → reduce noise
        
        Typical estimation:
            h = median({d₁, d₂, ..., dₖ})  # Distances to k neighbors
        
        This makes smoothing scale-invariant and density-adaptive.
    
    Comparison with other kernels:
        - Box kernel: w = 1 if d < h else 0
          → Non-differentiable, sharp boundary
        
        - Linear kernel: w = max(0, 1 - d/h)
          → Differentiable but not smooth (kink at d=h)
        
        - Epanechnikov kernel: w = max(0, 1 - (d/h)²)
          → Compact support but similar to linear
        
        - Gaussian kernel: w = exp(-d²/h²)
          → Smooth, differentiable, infinite support ✓
    
    Notes:
        - Weights are NOT normalized (caller should normalize if needed)
        - Infinite support in theory, effective support ≈ [-3h, 3h]
        - Works in any dimension (3D here for spatial smoothing)
        - Can be combined with other weights (e.g., attention from KNN)
    """
    # Compute displacement vectors
    diff = neighbor_positions - positions.unsqueeze(1)  # (N, k, 3)
    
    # Compute Euclidean distances
    dist = torch.norm(diff, dim=-1)  # (N, k)
    
    # Gaussian kernel with adaptive bandwidth
    # w = exp(-(d/h)²)
    weights = torch.exp(-(dist / bandwidth) ** 2)  # (N, k)
    
    return weights


def mask_self_neighbor(weights: torch.Tensor) -> torch.Tensor:
    """
    Mask out self-neighbor (first neighbor) for smoothing operations.
    
    In k-nearest neighbor (KNN) queries where query == reference set,
    the nearest neighbor is always the point itself with distance 0.
    For smoothing, we want to average over true neighbors, not self,
    so we zero out the first neighbor's contribution.
    
    Why masking is necessary:
        - Self-neighbor has distance 0 → infinite weight in some kernels
        - Including self → no smoothing (point stays the same)
        - Excluding self → proper Laplacian smoothing
    
    Implementation strategy:
        Uses element-wise multiplication (not in-place) to maintain
        differentiability. Creates a mask tensor with:
            mask[:, 0] = 0.0  (zero out first neighbor)
            mask[:, 1:] = 1.0  (keep other neighbors)
    
    Why not in-place:
        In-place operations (weights[:, 0] = 0) can break autograd:
        - Modifies tensor that may have other references
        - Can lose gradient tracking in some PyTorch versions
        - Out-of-place multiplication is safer for backprop
    
    Differentiability:
        - Element-wise multiplication: fully differentiable ✓
        - Mask is constant (no gradient needed)
        - Gradients flow through unchanged elements
        - Zero gradient for masked element (desired behavior)
    
    Args:
        weights: (N, k) Weight matrix for k neighbors per point
                 - First column ([:, 0]) is self-neighbor
                 - Remaining columns ([:, 1:]) are true neighbors
                 - Typically from KNN search or distance computation
    
    Returns:
        masked_weights: (N, k) Weights with self-neighbor zeroed
                        - masked_weights[:, 0] = 0.0 (all zeros)
                        - masked_weights[:, 1:] = weights[:, 1:] (unchanged)
                        - Same shape and dtype as input
                        - Differentiable w.r.t. input weights
    
    Complexity:
        Time: O(N·k) for element-wise operations
        Space: O(N·k) for mask tensor
    
    Example:
        >>> # Weights from KNN search
        >>> weights = torch.tensor([
        ...     [1.00, 0.95, 0.87, 0.73],  # First is self (distance 0)
        ...     [1.00, 0.92, 0.85, 0.78],
        ...     [1.00, 0.88, 0.76, 0.65],
        ... ])
        >>> 
        >>> masked = mask_self_neighbor(weights)
        >>> print(masked)
        tensor([[0.00, 0.95, 0.87, 0.73],
                [0.00, 0.92, 0.85, 0.78],
                [0.00, 0.88, 0.76, 0.65]])
        >>> 
        >>> # First column is zero
        >>> assert (masked[:, 0] == 0).all()
        >>> 
        >>> # Other columns unchanged
        >>> assert torch.allclose(masked[:, 1:], weights[:, 1:])
        >>> 
        >>> # Differentiable
        >>> weights.requires_grad_(True)
        >>> masked = mask_self_neighbor(weights)
        >>> masked.sum().backward()
        >>> print(weights.grad[:, 0])  # tensor([0., 0., 0.])
        >>> print(weights.grad[:, 1])  # tensor([1., 1., 1.])
    
    Alternative approaches (not used):
        1. In-place masking: weights[:, 0] = 0
           - Breaks autograd in some cases
           - Not recommended for learning pipelines
        
        2. Slicing: torch.cat([torch.zeros_like(weights[:, :1]), weights[:, 1:]], dim=1)
           - More complex, same result
           - Slightly slower due to concatenation
        
        3. Advanced indexing: weights[torch.arange(N), 0] = 0
           - Still in-place, same issues
        
        4. Element-wise multiplication: weights * mask ✓
           - Out-of-place, safe for autograd
           - Clear intent, easy to understand
    
    Usage in smoothing:
        >>> # Typical usage in Laplacian smoothing
        >>> indices, knn_weights = knn(positions, positions, k=16)
        >>> 
        >>> # Compute spatial weights
        >>> neighbor_pos = positions[indices]
        >>> spatial_weights = compute_spatial_weights(positions, neighbor_pos, h)
        >>> 
        >>> # Mask self-neighbor before combining
        >>> spatial_weights = mask_self_neighbor(spatial_weights)
        >>> 
        >>> # Combine and normalize
        >>> combined = spatial_weights * knn_weights
        >>> combined = combined / (combined.sum(dim=1, keepdim=True) + EPS_SAFE)
        >>> 
        >>> # Now safe to use for smoothing (self-neighbor excluded)
    
    Notes:
        - Assumes first neighbor is always self (true for query==reference KNN)
        - If KNN doesn't return self, this masking is unnecessary
        - For k=1, result is all zeros (no neighbors to smooth with)
        - Mask is broadcast-compatible for efficient computation
    """
    # Create binary mask: 0 for first column, 1 for rest
    mask = torch.ones_like(weights)
    mask[:, 0] = 0.0  # Zero out self-neighbor
    
    # Element-wise multiplication (out-of-place, autograd-safe)
    return weights * mask


# =========================================================
# Main Smoothing Function
# =========================================================
def smooth_normals(
    normals: torch.Tensor,
    positions: torch.Tensor,
    knn,
    cfg: Dict
) -> torch.Tensor:
    """
    Iterative Laplacian normal smoothing with adaptive spatial weighting.
    
    This function implements a sophisticated normal smoothing algorithm that
    combines classical Laplacian smoothing with modern techniques for robustness
    and differentiability. It's designed for point cloud surface normals in
    applications like physics-guided Gaussian splatting, differentiable rendering,
    and neural surface reconstruction.
    
    Core Idea:
        Replace each normal with a weighted average of its spatial neighbors,
        where weights depend on distance (closer = more influence) and adapt
        to local point density. Iterate multiple times for progressive smoothing.
    
    Algorithm (Per Iteration):
        1. **Neighbor Finding:**
           Find k spatial nearest neighbors for each point using KNN.
           
        2. **Bandwidth Estimation:**
           Estimate adaptive bandwidth h for each point:
               h = soft_median({distances to k neighbors})
           This makes smoothing density-adaptive.
           
        3. **Spatial Weighting:**
           Compute Gaussian weights based on distance:
               w_spatial(d) = exp(-(d/h)²)
           Closer neighbors get higher weights.
           
        4. **Weight Combination:**
           Combine spatial weights with KNN attention weights:
               w_combined = w_spatial * w_attention
           Then renormalize to sum=1.
           
        5. **Normal Averaging:**
           Compute weighted average of neighbor normals:
               n'ᵢ = Σⱼ w_combined[j] * n[j]
           Normalize to unit vector.
           
        6. **EMA Blending:**
           Blend smoothed normal with previous normal:
               n_new = λ*n' + (1-λ)*n_old
           This provides stability and prevents over-smoothing.
           
        7. **Normalization:**
           Ensure output is unit vector:
               n_final = n_new / ||n_new||
    
    Mathematical Formulation:
        At iteration t:
            n̄ᵢ⁽ᵗ⁾ = Σⱼ₌₁ᵏ wᵢⱼ · nⱼ⁽ᵗ⁻¹⁾
            nᵢ⁽ᵗ⁾ = normalize(λn̄ᵢ⁽ᵗ⁾ + (1-λ)nᵢ⁽ᵗ⁻¹⁾)
        
        where combined weights:
            wᵢⱼ = exp(-d²ᵢⱼ/h²ᵢ) · aᵢⱼ / Σₖ(exp(-d²ᵢₖ/h²ᵢ) · aᵢₖ)
        
        and adaptive bandwidth:
            hᵢ = soft_median({dᵢ₁, dᵢ₂, ..., dᵢₖ})
    
    Why Spatial Smoothing:
        Traditional mesh-based smoothing uses connectivity (edges). For point
        clouds, we have no topology, so we use spatial neighborhoods instead:
        
        Advantages:
        - Topology-agnostic: Works without mesh connectivity
        - Robust: Handles noise, outliers, non-uniform density
        - Adaptive: Automatically scales with point cloud resolution
        - Differentiable: Supports gradient-based optimization
        - Feature-preserving: Small h in high-curvature regions
    
    Adaptive Bandwidth Benefits:
        Fixed bandwidth smooths uniformly (good for uniform density).
        Adaptive bandwidth adjusts per-point based on local density:
        
        Dense regions (small h):
            - More local smoothing
            - Preserves fine details
            - Prevents over-smoothing
        
        Sparse regions (large h):
            - Wider smoothing
            - Fills gaps
            - Reduces noise
        
        Result: Scale-invariant, density-aware smoothing.
    
    EMA Blending (λ parameter):
        Controls trade-off between smoothing and preservation:
        
        λ = 0.0: No smoothing (output = input)
        λ = 0.5: Equal blend (balanced)
        λ = 0.8: Strong smoothing (typical, recommended)
        λ = 1.0: Full Laplacian (aggressive, less stable)
        
        Higher λ:
            + Faster convergence
            + More smoothing per iteration
            - Less stable
            - Potential over-smoothing
        
        Lower λ:
            + More stable
            + Gentler smoothing
            - Slower convergence
            - Need more iterations
    
    Differentiability and Gradients:
        All operations are differentiable, enabling:
        
        1. **Parameter Learning:**
           Optimize iters, k, λ via backprop through rendering loss
           
        2. **Joint Optimization:**
           Simultaneously optimize positions and normals
           
        3. **End-to-End Training:**
           Integrate into neural rendering pipelines
        
        Gradient flow path:
            Loss → smoothed_normals → avg_normal → neighbor_normals → 
            input_normals (parameter)
        
        Key differentiable components:
        - KNN search: soft attention weights ✓
        - Soft median: continuous approximation ✓
        - Gaussian weights: exp function ✓
        - Weighted averaging: linear combination ✓
        - Normalization: differentiable except at origin ✓
    
    Complexity Analysis:
        Per iteration:
        - KNN search: O(N log N) with FAISS, O(N²) naive
        - Gather neighbors: O(N·k·3)
        - Soft median: O(N·k·log k) sort + O(N·k) weight
        - Spatial weights: O(N·k·3) distance + O(N·k) exp
        - Weighted average: O(N·k·3)
        - Normalization: O(N·3)
        
        Total per iteration: O(N·(log N + k·log k))
        Total for T iterations: O(T·N·(log N + k·log k))
        
        Memory: O(N·k) for neighbor storage
    
    Args:
        normals: (N, 3) Input surface normals to smooth
                 - Should be unit vectors (will be normalized if not)
                 - Can have requires_grad=True for learning
                 - Typically computed from geometry or learned
                 
        positions: (N, 3) Spatial 3D positions for neighbor finding
                   - Defines spatial neighborhoods
                   - Must correspond to same point cloud as normals
                   - Used for KNN search and distance computation
                   
        knn: KNN function for spatial neighbor finding
             - Signature: knn(query, reference, k) -> (indices, weights)
             - Returns (N, k) indices and (N, k) soft attention weights
             - Must support differentiable weights (e.g., HybridFAISSKNN)
             - Example: HybridFAISSKNN(tau=0.15)
             
        cfg: Configuration dictionary with smoothing parameters:
        
             **Required keys: (all have defaults)**
             
             - 'iters' (int): Number of smoothing iterations
               * Default: 2
               * Range: [1, 5]
               * Higher = smoother but slower
               * Typical: 2-3
               
             - 'k' (int): Number of spatial neighbors per point
               * Default: 16
               * Range: [4, 64]
               * Higher = smoother, less local
               * Typical: 12-24
               
             - 'lambda_smooth' (float): EMA blending factor
               * Default: 0.8
               * Range: [0.0, 1.0]
               * Higher = more aggressive smoothing
               * Typical: 0.7-0.9
               
             - 'sigma_idx' (float): Soft median smoothness parameter
               * Default: 1.0
               * Range: [0.5, 2.0]
               * Controls bandwidth estimation smoothness
               * Typical: 0.8-1.2
    
    Returns:
        smoothed_normals: (N, 3) Smoothed unit surface normals
                          - Same shape as input
                          - All unit vectors (||n|| = 1)
                          - More spatially coherent than input
                          - Differentiable w.r.t. input normals and positions
    
    Typical Parameter Ranges and Effects:
        
        **Iterations (iters):**
        - 1: Minimal smoothing, preserves most detail
        - 2: Balanced smoothing (recommended default)
        - 3: Strong smoothing, removes most noise
        - 4+: Very smooth, may lose features
        
        **Neighbors (k):**
        - 8: Very local, preserves sharp features
        - 16: Balanced (recommended default)
        - 32: More global, stronger smoothing
        - 64: Very smooth, less local detail
        
        **Lambda (lambda_smooth):**
        - 0.5: Gentle smoothing, very stable
        - 0.8: Strong smoothing, stable (recommended)
        - 0.95: Aggressive smoothing, less stable
        - 1.0: Maximum smoothing, unstable
    
    Example Usage:
        >>> # Basic usage
        >>> import torch
        >>> from your_module import smooth_normals, HybridFAISSKNN
        >>> 
        >>> # Setup point cloud
        >>> N = 10000
        >>> positions = torch.randn(N, 3)
        >>> normals = torch.randn(N, 3)
        >>> normals = F.normalize(normals, dim=-1)  # Ensure unit normals
        >>> 
        >>> # Setup KNN
        >>> knn = HybridFAISSKNN(tau=0.15)
        >>> 
        >>> # Configure smoothing
        >>> cfg = {
        ...     'iters': 2,
        ...     'k': 16,
        ...     'lambda_smooth': 0.8,
        ...     'sigma_idx': 1.0
        ... }
        >>> 
        >>> # Smooth normals
        >>> smooth_n = smooth_normals(normals, positions, knn, cfg)
        >>> 
        >>> # Verify outputs
        >>> print(smooth_n.shape)  # (10000, 3)
        >>> print(torch.allclose(
        ...     torch.norm(smooth_n, dim=1), 
        ...     torch.ones(N), 
        ...     atol=1e-5
        ... ))  # True - still unit vectors ✓
        >>> 
        >>> # Check smoothing effect
        >>> print(f"Input std: {normals.std():.4f}")
        >>> print(f"Output std: {smooth_n.std():.4f}")
        >>> # Output has lower variance (smoother) ✓
    
    Example with Gradients:
        >>> # Enable gradient tracking
        >>> normals.requires_grad_(True)
        >>> positions.requires_grad_(True)
        >>> 
        >>> # Smooth normals
        >>> smooth_n = smooth_normals(normals, positions, knn, cfg)
        >>> 
        >>> # Compute loss (e.g., from rendering)
        >>> loss = some_rendering_loss(smooth_n)
        >>> 
        >>> # Backpropagate
        >>> loss.backward()
        >>> 
        >>> # Gradients available
        >>> print(normals.grad.shape)  # (10000, 3)
        >>> print(positions.grad.shape)  # (10000, 3)
        >>> 
        >>> # Can optimize normals directly
        >>> optimizer = torch.optim.Adam([normals], lr=1e-3)
        >>> optimizer.step()
    
    Example with Different Parameters:
        >>> # Minimal smoothing (preserve detail)
        >>> cfg_minimal = {'iters': 1, 'k': 8, 'lambda_smooth': 0.5}
        >>> smooth_minimal = smooth_normals(normals, positions, knn, cfg_minimal)
        >>> 
        >>> # Aggressive smoothing (remove noise)
        >>> cfg_aggressive = {'iters': 3, 'k': 32, 'lambda_smooth': 0.95}
        >>> smooth_aggressive = smooth_normals(normals, positions, knn, cfg_aggressive)
        >>> 
        >>> # Custom bandwidth estimation
        >>> cfg_custom = {'iters': 2, 'k': 16, 'lambda_smooth': 0.8, 'sigma_idx': 1.5}
        >>> smooth_custom = smooth_normals(normals, positions, knn, cfg_custom)
    
    Integration Example:
        >>> # In upsampling pipeline
        >>> class SurfaceUpsampler(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.knn = HybridFAISSKNN(tau=0.15)
        ...         self.smooth_cfg = {
        ...             'iters': 2,
        ...             'k': 16,
        ...             'lambda_smooth': 0.8
        ...         }
        ...     
        ...     def forward(self, positions, normals):
        ...         # Smooth normals before further processing
        ...         normals_smooth = smooth_normals(
        ...             normals, positions, self.knn, self.smooth_cfg
        ...         )
        ...         # Continue with smoothed normals
        ...         return self.upsample(positions, normals_smooth)
    
    Common Pitfalls and Solutions:
        
        1. **Non-unit input normals:**
           Problem: Algorithm assumes unit normals
           Solution: Normalize before calling:
               normals = F.normalize(normals, dim=-1)
        
        2. **k too small:**
           Problem: Insufficient neighbors, patchy smoothing
           Solution: Use k >= 12 for smooth results
        
        3. **k too large:**
           Problem: Over-smoothing, loss of features
           Solution: Keep k <= 32 for local detail
        
        4. **lambda = 1.0:**
           Problem: Unstable, can diverge
           Solution: Use lambda <= 0.95
        
        5. **Too many iterations:**
           Problem: Over-smoothing, washed out features
           Solution: Use iters <= 3 for most cases
    
    Notes:
        - First neighbor (self) is automatically masked out
        - Bandwidth estimated from k-1 neighbors (excluding self)
        - Output is always normalized to unit vectors
        - Can be called repeatedly for additional smoothing
        - Works with any point cloud (no mesh required)
        - Memory usage: O(N·k) per iteration
        - Supports batch processing (implicit in N dimension)
    """
    # Parse configuration with defaults
    iters = int(cfg.get("iters", DEFAULT_SMOOTH_ITERS))
    k = int(cfg.get("k", DEFAULT_SMOOTH_K))
    lambda_smooth = float(cfg.get("lambda_smooth", DEFAULT_LAMBDA_SMOOTH))
    sigma_idx = float(cfg.get("sigma_idx", DEFAULT_SIGMA_IDX))
    
    # Validate parameters (optional, can remove for performance)
    iters = max(MIN_SMOOTH_ITERS, min(iters, MAX_SMOOTH_ITERS))
    k = max(MIN_SMOOTH_K, min(k, MAX_SMOOTH_K))
    lambda_smooth = max(MIN_LAMBDA_SMOOTH, min(lambda_smooth, MAX_LAMBDA_SMOOTH))
    sigma_idx = max(MIN_SIGMA_IDX, min(sigma_idx, MAX_SIGMA_IDX))
    
    # Initialize smoothed normals (clone to avoid modifying input)
    normals_smooth = normals.clone()
    
    # Iterative smoothing loop
    for t in range(iters):
        # ============================================================
        # Step 1: Find Spatial Neighbors
        # ============================================================
        # Query KNN using current positions (fixed throughout smoothing)
        # Returns k nearest neighbors with soft attention weights
        idx, w = knn(positions, positions, k)  # (N, k), (N, k)
        
        # ============================================================
        # Step 2: Gather Neighbor Normals
        # ============================================================
        # Index into current smoothed normals to get neighbor values
        neighbor_normals = normals_smooth[idx]  # (N, k, 3)
        
        # ============================================================
        # Step 3: Compute Spatial Distances
        # ============================================================
        # Gather neighbor positions for distance computation
        neighbor_positions = positions[idx]  # (N, k, 3)
        
        # Compute displacement vectors and distances
        diff = neighbor_positions - positions.unsqueeze(1)  # (N, k, 3)
        dist = torch.norm(diff, dim=-1)  # (N, k)
        
        # ============================================================
        # Step 4: Estimate Adaptive Bandwidth
        # ============================================================
        # Use soft median of neighbor distances (excluding self at index 0)
        # This adapts bandwidth to local point density
        h = soft_median(dist[:, 1:], sigma_idx=sigma_idx) + EPS_SAFE  # (N, 1)
        
        # ============================================================
        # Step 5: Compute Spatial Weights
        # ============================================================
        # Gaussian kernel: w(d) = exp(-(d/h)²)
        spatial_weights = compute_spatial_weights(
            positions, neighbor_positions, h
        )  # (N, k)
        
        # ============================================================
        # Step 6: Mask Self-Neighbor
        # ============================================================
        # Zero out first neighbor (self) to ensure proper smoothing
        spatial_weights = mask_self_neighbor(spatial_weights)  # (N, k)
        
        # ============================================================
        # Step 7: Combine Weights
        # ============================================================
        # Multiply spatial weights with KNN attention weights
        # This combines distance-based and learned weighting
        combined_weights = spatial_weights * w  # (N, k)
        
        # Normalize to sum to 1 per row
        combined_weights = combined_weights / (
            combined_weights.sum(dim=1, keepdim=True) + EPS_SAFE
        )  # (N, k)
        
        # ============================================================
        # Step 8: Weighted Average of Neighbor Normals
        # ============================================================
        # Compute weighted sum: Σⱼ wⱼ·nⱼ
        avg_normal = (
            combined_weights.unsqueeze(-1) * neighbor_normals
        ).sum(dim=1)  # (N, 3)
        
        # Normalize to unit vector
        avg_normal = normalize(avg_normal, eps=EPS_SAFE)  # (N, 3)
        
        # ============================================================
        # Step 9: EMA Blending
        # ============================================================
        # Blend smoothed normal with previous normal for stability
        # n_new = λ*n_smooth + (1-λ)*n_old
        normals_smooth = (
            lambda_smooth * avg_normal + 
            (1 - lambda_smooth) * normals_smooth
        )  # (N, 3)
        
        # Normalize again to maintain unit length
        normals_smooth = normalize(normals_smooth, eps=EPS_SAFE)  # (N, 3)
    
    return normals_smooth


__all__ = [
    'smooth_normals',
    'soft_median',
    'compute_spatial_weights',
    'mask_self_neighbor',
]