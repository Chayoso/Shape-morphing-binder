"""
PCA-based surface analysis with differentiable operations.

This module performs local surface analysis using weighted PCA to extract:
- Surface normals (principal directions of variation)
- Surface quality metrics (planarity measure)
- Local point spacing (for adaptive operations)

All operations are fully differentiable for end-to-end learning.
"""

import torch
from typing import Tuple
from ..utils.config import EPS_SAFE, EPS_PCA, TANH_SCALE
from ..utils.utils import normalize


def compute_weighted_centroid(neighbors: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Compute weighted centroid of neighbor points.
    
    Mathematical formulation:
        c = Σᵢ wᵢ · xᵢ / Σᵢ wᵢ
    
    where wᵢ are normalized weights (sum to 1), so denominator is implicit.
    
    This is the weighted mean position of the local neighborhood,
    representing the "center of mass" for surface fitting.
    
    Differentiability:
        - Gradients flow through weights and neighbor positions
        - ∂c/∂wᵢ: shifts centroid toward xᵢ
        - ∂c/∂xᵢ: pulls centroid proportional to wᵢ
    
    Args:
        neighbors: (N, k, 3) neighbor point positions
                   - N: number of query points
                   - k: number of neighbors per point
                   - 3: spatial dimensions (x, y, z)
        weights: (N, k) normalized attention weights
                 - Should sum to ~1.0 along dim=1
                 - From KNN with softmax (differentiable)
    
    Returns:
        centroid: (N, 3) weighted center positions
                  - One centroid per query point
                  - Used as local coordinate system origin
    
    Complexity:
        - Time: O(N·k·3) for einsum
        - Memory: O(N·3) output
    
    Example:
        >>> neighbors = torch.randn(1000, 32, 3)  # 1000 points, 32 neighbors each
        >>> weights = torch.softmax(torch.randn(1000, 32), dim=1)
        >>> centroid = compute_weighted_centroid(neighbors, weights)
        >>> centroid.shape
        torch.Size([1000, 3])
    """
    return torch.einsum('nk,nkd->nd', weights, neighbors)


def compute_weighted_covariance(centered: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Compute weighted covariance matrix for PCA.
    
    Mathematical formulation:
        Σ = (1 / Σwᵢ) · Σᵢ wᵢ · (xᵢ - c)(xᵢ - c)ᵀ
    
    where:
        - xᵢ - c: centered neighbor positions
        - wᵢ: importance weights
        - Σ: 3×3 covariance matrix
    
    Physical interpretation:
        - Covariance captures the "spread" of points in each direction
        - Eigenvectors: principal directions of variation
        - Eigenvalues: amount of variance in each direction
    
    For surface analysis:
        - Largest eigenvector → tangent direction (max variation)
        - Smallest eigenvector → normal direction (min variation)
        - Middle eigenvector → orthogonal tangent
    
    Differentiability:
        - Fully differentiable w.r.t. centered positions and weights
        - Gradient stable due to sqrt(w) formulation
        - Avoids numerical issues with very small weights
    
    Args:
        centered: (N, k, 3) centered neighbor positions (neighbors - centroid)
                  - Already mean-subtracted
                  - Ready for covariance computation
        weights: (N, k) normalized attention weights
                 - Used to weight each neighbor's contribution
    
    Returns:
        cov: (N, 3, 3) covariance matrices
             - Symmetric positive semi-definite
             - One matrix per query point
             - Ready for eigendecomposition
    
    Implementation notes:
        - Uses sqrt(w) to weight points before outer product
        - Equivalent to: Σ (√wᵢ·xᵢ)(√wᵢ·xᵢ)ᵀ
        - More numerically stable than direct weighting
    
    Complexity:
        - Time: O(N·k·9) for einsum (3×3 outer products)
        - Memory: O(N·9) for output matrices
    """
    # Weight by sqrt for numerical stability
    sqrt_w = torch.sqrt(weights).unsqueeze(-1)  # (N, k, 1)
    weighted = centered * sqrt_w  # (N, k, 3)
    
    # Weighted outer product: Σ (√w·x)(√w·x)ᵀ = Σ w·x·xᵀ
    cov = torch.einsum('nki,nkj->nij', weighted, weighted)  # (N, 3, 3)
    
    # Normalize by sum of weights
    cov = cov / (weights.sum(dim=1, keepdim=True).unsqueeze(-1) + EPS_SAFE)
    
    return cov


def extract_normal_from_pca(
    evecs: torch.Tensor, 
    x: torch.Tensor, 
    centroid: torch.Tensor
) -> torch.Tensor:
    """
    Extract and orient surface normal from PCA eigenvectors.
    
    Algorithm:
        1. Take smallest eigenvector (minimal variance direction) as normal candidate
        2. Orient normal to point "outward" from local surface
        3. Use soft orientation based on point-to-centroid direction
    
    Orientation strategy:
        - Compare normal with vector from centroid to point
        - If they align (dot > 0): normal points outward ✓
        - If they oppose (dot < 0): flip normal to point outward
        - Use smooth tanh for differentiable sign determination
    
    Why orientation matters:
        - Unoriented normals cause rendering artifacts (wrong shading)
        - Consistent orientation needed for surface reconstruction
        - "Outward" convention matches computer graphics standards
    
    Differentiability:
        - Eigenvectors from torch.linalg.eigh: differentiable ✓
        - tanh-based sign: smooth and differentiable ✓
        - No discrete branching, only soft weighting
    
    Edge cases:
        - Point coincides with centroid: use global centroid as fallback
        - Ensures stable orientation even for degenerate cases
    
    Args:
        evecs: (N, 3, 3) eigenvector matrices from PCA
               - evecs[:, :, 0]: smallest eigenvalue (normal direction)
               - evecs[:, :, 1]: middle eigenvalue (tangent 1)
               - evecs[:, :, 2]: largest eigenvalue (tangent 2)
               - Columns are unit vectors (from eigh)
        x: (N, 3) original query point positions
           - Used to determine "outward" direction
        centroid: (N, 3) local weighted centroids
                  - Local coordinate origin for each point
    
    Returns:
        normals: (N, 3) oriented surface normals
                 - Unit vectors (normalized)
                 - Point "outward" from local surface
                 - Differentiable w.r.t. x, evecs, centroid
    
    Complexity:
        - Time: O(N·3) for vector operations
        - Memory: O(N·3) for output normals
    
    Example:
        >>> evecs = torch.randn(1000, 3, 3)
        >>> x = torch.randn(1000, 3)
        >>> centroid = torch.randn(1000, 3)
        >>> normals = extract_normal_from_pca(evecs, x, centroid)
        >>> torch.allclose(torch.norm(normals, dim=1), torch.ones(1000))
        True  # Normals are unit vectors
    """
    # Smallest eigenvector = normal candidate (minimal variance)
    n_raw = evecs[:, :, 0]  # (N, 3)
    
    # Compute "outward" reference direction
    global_c = x.mean(dim=0)  # (3,) - global centroid as fallback
    to_out = x - centroid  # (N, 3) - vector from local centroid to point
    
    # Handle degenerate case: point coincides with centroid
    mask = torch.norm(to_out, dim=1) < 1e-6  # (N,) boolean mask
    to_out[mask] = x[mask] - global_c  # Use global direction instead
    
    # Compute orientation via smooth dot product
    dot = torch.einsum('nd,nd->n', n_raw, to_out)  # (N,) - alignment score
    
    # Smooth sign function via tanh (differentiable)
    # TANH_SCALE controls sharpness: larger → closer to hard sign
    sign = torch.tanh(TANH_SCALE * dot).unsqueeze(-1)  # (N, 1) ∈ [-1, 1]
    
    # Orient normal: positive sign if aligned, negative if opposed
    normals = normalize(n_raw * sign)  # (N, 3) - unit vectors
    
    return normals


def compute_local_spacing(
    neighbors: torch.Tensor, 
    x: torch.Tensor, 
    weights: torch.Tensor
) -> torch.Tensor:
    """
    Compute local point spacing for adaptive operations.
    
    Purpose:
        Measure the characteristic distance between points in each
        local neighborhood. Used for:
        - Adaptive jitter during upsampling
        - Scale-aware feature computation
        - Density-adaptive smoothing
    
    Mathematical formulation:
        s = Σᵢ wᵢ · ||xᵢ - x|| / Σᵢ wᵢ
    
    where:
        - ||xᵢ - x||: distance from query to neighbor i
        - wᵢ: attention weight (importance)
        - s: weighted average distance
    
    Interpretation:
        - High spacing: sparse region (points far apart)
        - Low spacing: dense region (points close together)
        - Captures local point density
    
    Applications:
        1. Adaptive jitter: jitter ∝ spacing (preserve density)
        2. Adaptive Gaussian scale: σ ∝ spacing (match resolution)
        3. Feature radius: search radius ∝ spacing (scale-aware)
    
    Differentiability:
        - Fully differentiable w.r.t. neighbor positions and query position
        - Gradients flow through distance computation and weighting
        - Stable even for very small distances (no division by zero)
    
    Args:
        neighbors: (N, k, 3) neighbor point positions
                   - Local neighborhood for each query
        x: (N, 3) query point positions
           - Center of each local neighborhood
        weights: (N, k) normalized attention weights
                 - Higher weight → more influence on spacing
                 - Should sum to ~1.0 along dim=1
    
    Returns:
        spacing: (N,) local point spacing values
                 - One scalar per query point
                 - Units: same as input coordinates
                 - Typical range: 0.01 to 0.5 (normalized coordinates)
    
    Complexity:
        - Time: O(N·k·3) for distance computation
        - Memory: O(N·k) for intermediate distances
    
    Example:
        >>> neighbors = torch.randn(1000, 32, 3)
        >>> x = torch.randn(1000, 3)
        >>> weights = torch.softmax(torch.randn(1000, 32), dim=1)
        >>> spacing = compute_local_spacing(neighbors, x, weights)
        >>> spacing.shape
        torch.Size([1000])
        >>> # Denser neighborhoods have smaller spacing
        >>> # Sparser neighborhoods have larger spacing
    """
    # Compute distances from query to all neighbors
    dists = torch.norm(neighbors - x.unsqueeze(1), dim=-1)  # (N, k)
    
    # Weighted average distance
    spacing = torch.einsum('nk,nk->n', dists, weights)  # (N,)
    spacing = spacing / (weights.sum(dim=1) + EPS_SAFE)  # Normalize
    
    return spacing


def batched_pca_surface_optimized(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform batched weighted PCA surface analysis for all points.
    
    This is the main function that orchestrates local surface analysis.
    For each query point:
        1. Gather k nearest neighbors
        2. Compute weighted centroid (local coordinate origin)
        3. Build weighted covariance matrix
        4. Eigendecomposition to find principal directions
        5. Extract and orient surface normal (smallest eigenvector)
        6. Compute surface quality (planarity measure)
        7. Compute local point spacing
    
    Geometric interpretation:
        - Performs local plane fitting via PCA
        - Normal: direction perpendicular to best-fit plane
        - Surface variance: how well points fit a plane (0 = perfect plane)
        - Spacing: characteristic distance between points
    
    Why PCA for surface analysis:
        - Robust to noise (least-squares fit)
        - Provides orthonormal frame (normal + 2 tangents)
        - Fast: O(k) per point with batched operations
        - Differentiable: end-to-end learning possible
    
    Surface quality metric:
        surface_variance = λ₀ / (λ₀ + λ₁ + λ₂)
        
        where λ₀ ≤ λ₁ ≤ λ₂ are eigenvalues (sorted ascending)
        
        - surface_variance ≈ 0: planar surface (sharp feature)
        - surface_variance ≈ 1/3: isotropic noise (poor surface)
        - Lower is better (more planar)
    
    Differentiability:
        All operations are differentiable:
        - Neighbor gathering via indexing: ✓
        - Weighted statistics: ✓
        - Eigendecomposition (torch.linalg.eigh): ✓
        - Normal orientation via tanh: ✓
        
        Enables:
        - Learning optimal neighbor weights
        - Optimizing point positions
        - End-to-end surface-aware networks
    
    Args:
        x: (N, 3) query point positions
           - Full point cloud
           - Each point will get surface analysis
        indices: (N, k) neighbor indices
                 - From KNN search
                 - indices[i, :] are k neighbors of point i
        weights: (N, k) normalized attention weights
                 - From differentiable KNN (softmax weights)
                 - Higher weight = more important neighbor
                 - Should sum to ~1.0 along dim=1
    
    Returns:
        normals: (N, 3) oriented surface normals
                 - Unit vectors pointing "outward"
                 - Perpendicular to local best-fit plane
                 - Differentiable w.r.t. x, indices, weights
        
        surface_variance: (N,) surface quality metric
                         - ∈ [0, 1] where lower is better
                         - 0: perfect planar surface
                         - ~0.33: no clear surface structure
                         - Useful for confidence weighting
        
        local_spacing: (N,) local point spacing
                      - Weighted average distance to neighbors
                      - Units: same as input coordinates
                      - Used for adaptive operations
    
    Complexity:
        - Time: O(N·k·3) for covariance + O(N·27) for eigendecomp
        - Memory: O(N·k·3) for neighbors + O(N·9) for covariance
        - Batched: all N points processed in parallel
    
    Example:
        >>> # Setup
        >>> x = torch.randn(10000, 3, requires_grad=True)
        >>> knn = HybridFAISSKNN(tau=0.15)
        >>> indices, weights = knn(x, x, k=32)
        >>> 
        >>> # Surface analysis
        >>> normals, surf_var, spacing = batched_pca_surface_optimized(x, indices, weights)
        >>> 
        >>> # Check outputs
        >>> normals.shape          # (10000, 3)
        >>> surf_var.shape         # (10000,)
        >>> spacing.shape          # (10000,)
        >>> 
        >>> # Normals are unit vectors
        >>> torch.allclose(torch.norm(normals, dim=1), torch.ones(10000))
        True
        >>> 
        >>> # Surface variance in [0, 1]
        >>> assert (surf_var >= 0).all() and (surf_var <= 1).all()
        >>> 
        >>> # Differentiability check
        >>> loss = normals.sum()
        >>> loss.backward()
        >>> assert x.grad is not None  # Gradients flow to input points ✓
    
    Notes:
        - EPS_PCA: small constant added to eigenvalues for numerical stability
        - EPS_SAFE: prevents division by zero in normalizations
        - TANH_SCALE: controls sharpness of normal orientation
        - All hyperparameters imported from ..utils.config
    """
    # Step 1: Gather neighbors
    neighbors = x[indices]  # (N, k, 3) - maintains gradient connection
    
    # Step 2: Compute weighted centroid (local origin)
    centroid = compute_weighted_centroid(neighbors, weights)  # (N, 3)
    
    # Step 3: Center neighbors (translate to local coordinates)
    centered = neighbors - centroid.unsqueeze(1)  # (N, k, 3)
    
    # Step 4: Compute weighted covariance matrix
    cov = compute_weighted_covariance(centered, weights)  # (N, 3, 3)
    
    # Step 5: Eigendecomposition (PCA)
    # evals sorted ascending: λ₀ ≤ λ₁ ≤ λ₂
    # evecs[:, :, i] corresponds to evals[:, i]
    evals, evecs = torch.linalg.eigh(cov)  # (N, 3), (N, 3, 3)
    evals = torch.clamp(evals, min=EPS_PCA)  # Numerical stability
    
    # Step 6: Surface quality metric (planarity)
    # How much variance is in normal direction vs total
    surfvar = evals[:, 0] / (evals.sum(dim=1) + EPS_PCA)  # (N,)
    
    # Step 7: Extract and orient normal
    normals = extract_normal_from_pca(evecs, x, centroid)  # (N, 3)
    
    # Step 8: Compute local spacing
    spacing = compute_local_spacing(neighbors, x, weights)  # (N,)
    
    return normals, surfvar, spacing


__all__ = [
    'compute_weighted_centroid',
    'compute_weighted_covariance',
    'extract_normal_from_pca',
    'compute_local_spacing',
    'batched_pca_surface_optimized',
]