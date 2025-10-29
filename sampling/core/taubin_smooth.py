"""
Differentiable Taubin Smoothing for Point Cloud Geometry

This module implements Taubin smoothing, a sophisticated two-pass Laplacian
smoothing technique that prevents volume shrinkage while maintaining full
differentiability for gradient-based optimization.

Classical Laplacian smoothing smooths geometry but causes shrinkage (volume loss)
over iterations. Taubin smoothing solves this by alternating between smoothing
and inflation passes, creating a low-pass filter that preserves volume.

Algorithm Overview:
    1. Forward pass: Laplacian smoothing with positive λ (smooth)
    2. Backward pass: Laplacian inflation with negative λ (unshrink)
    
    This composition acts as a low-pass filter on surface frequencies while
    maintaining mean curvature and volume.

Mathematical Foundation:
    For a point p with neighbors N(p):
    
    Laplacian: Δp = centroid(N(p)) - p
    
    Taubin two-pass:
        Pass 1: p' = p + λ₁·Δp     (λ₁ > 0, smoothing)
        Pass 2: p'' = p' + λ₂·Δp'  (λ₂ < 0, inflation)
    
    Optimal parameters for stability:
        λ₂ = -λ₁ / (1 - λ₁·κ)
    where κ is the scale factor (typically 0.1).

Key Features:
    - Volume-preserving smoothing (no shrinkage)
    - Differentiable two-pass Laplacian
    - Optional tangent-only projection (preserve surface topology)
    - Fixed connectivity per iteration (stability)
    - Memory-efficient implementation

Applications:
    - Point cloud denoising without shrinkage
    - Surface smoothing in reconstruction
    - Geometry regularization in neural rendering
    - Physics-guided Gaussian splatting

Advantages over Standard Laplacian:
    - No volume loss over iterations
    - Better feature preservation
    - More stable convergence
    - Can use more iterations safely

Author: CHAYO
Version: 2.2.1 (Refactored)

References:
    Taubin, G. (1995). "A Signal Processing Approach to Fair Surface Design."
    SIGGRAPH 1995.
"""

import torch
from typing import Dict, Tuple
from ..utils.config import EPS_SAFE


# =========================================================
# Constants
# =========================================================
DEFAULT_TAUBIN_ITERS = 3
DEFAULT_TAUBIN_K = 24
DEFAULT_LAMBDA_SMOOTH = 0.6
DEFAULT_LAMBDA_INFLATE = -0.53
DEFAULT_TANGENT_ONLY = True

# Parameter ranges
MIN_TAUBIN_ITERS = 1
MAX_TAUBIN_ITERS = 10
MIN_TAUBIN_K = 4
MAX_TAUBIN_K = 64
MIN_LAMBDA_SMOOTH = 0.1
MAX_LAMBDA_SMOOTH = 0.9
MIN_LAMBDA_INFLATE = -0.9
MAX_LAMBDA_INFLATE = -0.1

# Typical parameter recommendations
RECOMMENDED_LAMBDA_PAIRS = [
    (0.5, -0.45),   # Gentle smoothing
    (0.6, -0.53),   # Balanced (default)
    (0.7, -0.63),   # Strong smoothing
]


# =========================================================
# Laplacian Computation
# =========================================================
def compute_laplacian(
    P: torch.Tensor,
    knn,
    k: int
) -> torch.Tensor:
    """
    Compute Laplacian displacement for each point.
    
    The Laplacian operator on a point cloud is the vector from each point
    to the weighted centroid of its neighbors. It measures local deviation
    from the neighborhood average and is fundamental to mesh smoothing.
    
    Mathematical Definition:
        For point p with k neighbors {n₁, n₂, ..., nₖ} and weights {w₁, ..., wₖ}:
        
        Centroid: c = Σᵢ wᵢ·nᵢ / Σᵢ wᵢ  (weighted average)
        Laplacian: Δp = c - p             (displacement to centroid)
    
    Interpretation:
        - Δp = 0: Point is at the centroid (locally flat)
        - |Δp| large: Point deviates from neighbors (high curvature)
        - Direction of Δp: Points toward "smoothed" position
    
    Properties:
        - Fully differentiable: gradients flow through KNN and averaging
        - Scale-dependent: |Δp| depends on point cloud scale
        - Weighted: KNN weights allow soft/learned neighborhoods
    
    Args:
        P: (N, 3) Point positions in 3D space
           - Input geometry to compute Laplacian on
           - Can have requires_grad=True for optimization
           
        knn: KNN function for neighbor finding
             - Signature: knn(query, reference, k) -> (indices, weights)
             - Returns (N, k) neighbor indices and soft weights
             - Example: HybridFAISSKNN(tau=0.15)
             
        k: Number of nearest neighbors to consider
           - Typical range: [8, 32]
           - Larger k: smoother Laplacian, more global
           - Smaller k: more local, preserves features
    
    Returns:
        lap: (N, 3) Laplacian displacement vectors
             - Points from current position toward smoothed position
             - Magnitude indicates local roughness
             - Direction indicates smoothing direction
             - Differentiable w.r.t. P
    
    Complexity:
        Time: O(N·log(N)) for KNN + O(N·k·3) for weighted sum
        Space: O(N·k) for neighbor storage
    
    Differentiability:
        Gradient w.r.t. positions:
            ∂Δp/∂P = ∂centroid/∂P - I
        
        Where ∂centroid/∂P depends on KNN weights (assumed constant)
        and neighbor positions.
    
    Example:
        >>> # Compute Laplacian for 1000 points
        >>> positions = torch.randn(1000, 3, requires_grad=True)
        >>> knn = HybridFAISSKNN(tau=0.15)
        >>> 
        >>> lap = compute_laplacian(positions, knn, k=16)
        >>> 
        >>> print(lap.shape)  # (1000, 3)
        >>> print(lap.mean(dim=0))  # Close to zero (mean centering)
        >>> 
        >>> # Laplacian magnitude indicates roughness
        >>> roughness = torch.norm(lap, dim=1)
        >>> print(f"Mean roughness: {roughness.mean():.4f}")
        >>> 
        >>> # Gradients flow
        >>> loss = lap.norm()
        >>> loss.backward()
        >>> print(positions.grad.shape)  # (1000, 3)
    
    Usage in Smoothing:
        The Laplacian is used to update positions:
            P_new = P + λ·Δp
        
        Where λ controls smoothing strength:
        - λ > 0: Move toward centroid (smoothing)
        - λ < 0: Move away from centroid (inflation)
        - λ = 0: No change
    
    Relationship to Mesh Laplacian:
        For meshes, the Laplacian is typically:
            Δp = (1/|N|)·Σᵢ(nᵢ - p) = centroid - p
        
        Our weighted version generalizes this:
            Δp = Σᵢ wᵢ·nᵢ - p
        
        With uniform weights wᵢ=1/k, they're equivalent.
    
    Notes:
        - Assumes KNN returns meaningful soft weights
        - Works with any point cloud (no connectivity needed)
        - Can be computed on subsets for efficiency
        - Differentiable but assumes fixed KNN graph
    """
    # Find k nearest neighbors with soft weights
    idx, w = knn(P, P, k)  # idx: (N, k), w: (N, k)
    
    # Gather neighbor positions
    Q = P[idx]  # (N, k, 3)
    
    # Compute weighted centroid
    # centroid = Σⱼ wⱼ·Pⱼ (weights already normalized by KNN)
    centroid = (w.unsqueeze(-1) * Q).sum(dim=1)  # (N, 3)
    
    # Laplacian displacement: vector from point to centroid
    lap = centroid - P  # (N, 3)
    
    return lap


def split_tangent_normal(
    lap: torch.Tensor,
    normals: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose Laplacian into tangent and normal components.
    
    For surface point clouds, the Laplacian displacement has two components:
    1. Normal component: Movement perpendicular to surface (changes thickness)
    2. Tangent component: Movement parallel to surface (preserves thickness)
    
    This decomposition allows selective smoothing:
    - Tangent-only: Smooth while preserving surface position
    - Normal-only: Adjust thickness without smoothing
    - Both: Full Laplacian smoothing
    
    Mathematical Decomposition:
        Given Laplacian Δp and surface normal n (unit vector):
        
        Normal component:
            Δp_n = (Δp · n)·n  (projection onto normal)
        
        Tangent component:
            Δp_t = Δp - Δp_n   (rejection from normal)
        
        Verification:
            Δp = Δp_t + Δp_n   (orthogonal decomposition)
            Δp_t · n = 0       (tangent is perpendicular)
    
    Geometric Interpretation:
        - Δp_n: How much point moves in/out of surface
        - Δp_t: How much point moves along surface
        - |Δp_n| / |Δp|: Fraction of "thickness" change
        - |Δp_t| / |Δp|: Fraction of "sliding" along surface
    
    Why Tangent-Only Smoothing:
        In point clouds representing thin surfaces:
        - Normal displacement can punch through surfaces
        - Normal displacement can change topology
        - Tangent smoothing preserves surface structure
        - Tangent smoothing is safer for thin geometry
    
    Args:
        lap: (N, 3) Laplacian displacement vectors
             - From compute_laplacian()
             - Can be any 3D vector field
             
        normals: (N, 3) Surface normal vectors
                 - Should be unit vectors (||n|| = 1)
                 - Define local surface orientation
                 - Typically from PCA or gradient estimation
    
    Returns:
        lap_t: (N, 3) Tangent component of Laplacian
               - Parallel to surface (lap_t · n = 0)
               - Preserves surface thickness
               - Differentiable w.r.t. lap and normals
               
        lap_n: (N, 3) Normal component of Laplacian
               - Perpendicular to surface (lap_n ∥ n)
               - Changes surface thickness
               - Differentiable w.r.t. lap and normals
    
    Properties:
        - Orthogonality: lap_t · lap_n = 0
        - Completeness: lap = lap_t + lap_n
        - Magnitude: |lap|² = |lap_t|² + |lap_n|²
    
    Complexity:
        Time: O(N·3) for dot product and vector operations
        Space: O(N·3) for output vectors
    
    Differentiability:
        Gradients w.r.t. Laplacian:
            ∂lap_n/∂lap = n ⊗ n  (outer product, projection matrix)
            ∂lap_t/∂lap = I - n ⊗ n  (rejection matrix)
        
        Gradients w.r.t. normals:
            ∂lap_n/∂n = (lap · n)·I + lap ⊗ n
            ∂lap_t/∂n = -∂lap_n/∂n
    
    Example:
        >>> # Decompose Laplacian
        >>> lap = torch.randn(1000, 3)
        >>> normals = F.normalize(torch.randn(1000, 3), dim=-1)
        >>> 
        >>> lap_t, lap_n = split_tangent_normal(lap, normals)
        >>> 
        >>> # Verify orthogonality
        >>> tangent_normal_dot = (lap_t * normals).sum(dim=1)
        >>> print(tangent_normal_dot.abs().max())  # ~0 (perpendicular)
        >>> 
        >>> # Verify completeness
        >>> reconstructed = lap_t + lap_n
        >>> print(torch.allclose(reconstructed, lap))  # True
        >>> 
        >>> # Normal component is parallel to normal
        >>> normal_direction = F.normalize(lap_n, dim=-1)
        >>> alignment = (normal_direction * normals).sum(dim=1).abs()
        >>> print(alignment.mean())  # ~1.0 (parallel)
        >>> 
        >>> # Magnitude analysis
        >>> total_mag = torch.norm(lap, dim=1)
        >>> tangent_mag = torch.norm(lap_t, dim=1)
        >>> normal_mag = torch.norm(lap_n, dim=1)
        >>> print(f"Tangent ratio: {(tangent_mag/total_mag).mean():.2f}")
        >>> print(f"Normal ratio: {(normal_mag/total_mag).mean():.2f}")
    
    Usage in Taubin Smoothing:
        >>> # Tangent-only smoothing (preserve thickness)
        >>> lap = compute_laplacian(P, knn, k=16)
        >>> lap_t, lap_n = split_tangent_normal(lap, normals)
        >>> P_smooth = P + lambda_smooth * lap_t  # Only tangent
        >>> 
        >>> # Full smoothing (allow thickness change)
        >>> P_smooth_full = P + lambda_smooth * lap  # Both components
    
    Comparison:
        Tangent-only vs Full Laplacian:
        
        Tangent-only:
        + Preserves surface structure
        + Safer for thin geometry
        + No topology changes
        - Less smoothing power
        - May leave normal noise
        
        Full Laplacian:
        + Stronger smoothing
        + Removes all types of noise
        - Can change thickness
        - May punch through surfaces
    
    Notes:
        - Assumes normals are unit vectors (not enforced)
        - If normals have length ≠ 1, normal component scales accordingly
        - Differentiable through all operations
        - Can be applied to any vector field, not just Laplacians
    """
    # Compute normal component: projection onto normal
    # lap_n = (lap · n)·n
    projection_scalar = (lap * normals).sum(dim=-1, keepdim=True)  # (N, 1)
    lap_n = projection_scalar * normals  # (N, 3)
    
    # Compute tangent component: rejection from normal
    # lap_t = lap - lap_n
    lap_t = lap - lap_n  # (N, 3)
    
    return lap_t, lap_n


# =========================================================
# Main Taubin Smoothing Function
# =========================================================
def taubin_smooth(
    P: torch.Tensor,
    normals: torch.Tensor,
    knn,
    cfg: Dict
) -> torch.Tensor:
    """
    Apply differentiable Taubin smoothing to point cloud geometry.
    
    Taubin smoothing is a sophisticated geometric smoothing technique that
    prevents volume shrinkage (a major problem with standard Laplacian smoothing)
    by alternating between smoothing and inflation passes. This creates a
    low-pass filter on surface frequencies while preserving mean curvature.
    
    Core Innovation:
        Standard Laplacian smoothing:
            P ← P + λ·Δp  (repeated iterations cause shrinkage)
        
        Taubin smoothing:
            P ← P + λ₁·Δp      (smoothing pass, λ₁ > 0)
            P ← P + λ₂·Δp'     (inflation pass, λ₂ < 0)
        
        The negative second pass counteracts volume loss while maintaining
        smoothing of high-frequency noise.
    
    Algorithm (Per Iteration):
        1. **Fixed Graph:** Compute KNN once, reuse for both passes
           - Prevents oscillation and instability
           - More efficient than recomputing
           
        2. **Smoothing Pass (λ₁ > 0):**
           - Compute Laplacian: Δp = centroid(neighbors) - p
           - Optional: Project to tangent plane (Δp_t)
           - Update: p' = p + λ₁·Δp_t
           
        3. **Inflation Pass (λ₂ < 0):**
           - Compute Laplacian on smoothed positions: Δp' = centroid' - p'
           - Optional: Project to tangent plane (Δp'_t)
           - Update: p'' = p' + λ₂·Δp'_t
           
        4. **Iterate:** Repeat steps 2-3 for specified iterations
    
    Mathematical Analysis:
        For optimal volume preservation:
            λ₂ ≈ -λ₁ / (1 - λ₁·κ)
        
        where κ is the scale factor (typically 0.1).
        
        Example: λ₁ = 0.6, κ = 0.1
            λ₂ = -0.6 / (1 - 0.06) = -0.638 ≈ -0.63
        
        Our defaults: λ₁ = 0.6, λ₂ = -0.53 (slightly less aggressive)
    
    Why Taubin Works:
        1. **Frequency Analysis:**
           - Smoothing pass: attenuates high frequencies
           - Inflation pass: amplifies all frequencies slightly
           - Net effect: low-pass filter with gain ≈ 1 at DC
        
        2. **Volume Preservation:**
           - Positive λ shrinks volume (moves toward centroid)
           - Negative λ expands volume (moves away from centroid)
           - Balanced λ₁, λ₂ maintains volume
        
        3. **Curvature:**
           - Both passes smooth curvature
           - Net effect: reduced high-frequency curvature variation
    
    Advantages over Standard Laplacian:
        ✓ No volume shrinkage (major improvement)
        ✓ More iterations possible (no accumulating bias)
        ✓ Better feature preservation
        ✓ More stable convergence
        ✓ Can smooth aggressively without collapse
    
    Tangent-Only Option:
        When tangent_only=True, smoothing only affects positions parallel
        to the surface (not perpendicular). This:
        
        Benefits:
        + Preserves surface topology
        + No thickness changes
        + Safer for thin geometry
        + No self-intersections
        
        Trade-offs:
        - Slightly less smoothing power
        - Normal-direction noise remains
    
    Differentiability:
        All operations are differentiable:
        - KNN search: fixed graph (gradients through weights)
        - Weighted averaging: linear operation ✓
        - Tangent projection: differentiable projection ✓
        - Position updates: addition and scaling ✓
        
        Enables:
        - Learning optimal λ parameters
        - Joint optimization with rendering
        - End-to-end surface reconstruction
    
    Args:
        P: (N, 3) Point positions to smooth
           - Input geometry (point cloud)
           - Can have requires_grad=True for optimization
           - Will be iteratively smoothed
           
        normals: (N, 3) Surface normals for tangent projection
                 - Should be unit vectors (||n|| = 1)
                 - Used only if tangent_only=True
                 - Can be from PCA, gradient, or learned
                 
        knn: KNN function for neighbor finding
             - Signature: knn(query, reference, k) -> (indices, weights)
             - Returns soft weights for differentiability
             - Example: HybridFAISSKNN(tau=0.15)
             
        cfg: Configuration dictionary with keys:
        
             **Optional parameters (all have defaults):**
             
             - 'iters' (int): Number of Taubin iterations
               * Default: 3
               * Range: [1, 10]
               * More iterations = smoother but slower
               * Typical: 2-4
               
             - 'k' (int): Number of neighbors for Laplacian
               * Default: 24
               * Range: [4, 64]
               * More neighbors = smoother, more global
               * Typical: 16-32
               
             - 'lambda_smooth' (float): Smoothing pass strength
               * Default: 0.6
               * Range: [0.1, 0.9]
               * Positive value for smoothing
               * Larger = more aggressive smoothing
               * Typical: 0.5-0.7
               
             - 'lambda_inflate' (float): Inflation pass strength
               * Default: -0.53
               * Range: [-0.9, -0.1]
               * Negative value for inflation (unshrinking)
               * Should be ≈ -λ₁/(1-λ₁·κ)
               * Typical: -0.5 to -0.65
               
             - 'tangent_only' (bool): Use tangent-only projection
               * Default: True
               * True: Smooth only in tangent direction (preserve thickness)
               * False: Full Laplacian smoothing (allow thickness change)
    
    Returns:
        P_smooth: (N, 3) Smoothed point positions
                  - Same shape as input
                  - Volume-preserved (no shrinkage)
                  - Smoother than input (less noise)
                  - Differentiable w.r.t. input P
    
    Complexity:
        Per iteration:
        - KNN search: O(N·log(N)) once
        - Two Laplacian computations: 2 × O(N·k·3)
        - Position updates: O(N·3)
        
        Total for T iterations: O(T·N·(log(N) + k))
        Memory: O(N·k) for neighbor storage
    
    Typical Parameter Combinations:
        
        **Gentle smoothing (preserve details):**
        ```python
        cfg = {
            'iters': 2,
            'k': 16,
            'lambda_smooth': 0.5,
            'lambda_inflate': -0.45,
            'tangent_only': True
        }
        ```
        
        **Balanced smoothing (recommended default):**
        ```python
        cfg = {
            'iters': 3,
            'k': 24,
            'lambda_smooth': 0.6,
            'lambda_inflate': -0.53,
            'tangent_only': True
        }
        ```
        
        **Aggressive smoothing (denoise heavy):**
        ```python
        cfg = {
            'iters': 4,
            'k': 32,
            'lambda_smooth': 0.7,
            'lambda_inflate': -0.63,
            'tangent_only': False
        }
        ```
    
    Example Usage:
        >>> # Basic usage
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from module import taubin_smooth, HybridFAISSKNN
        >>> 
        >>> # Setup noisy point cloud
        >>> N = 5000
        >>> positions = torch.randn(N, 3) * 0.5  # Sphere-ish
        >>> positions = F.normalize(positions, dim=-1)  # On sphere
        >>> positions = positions + torch.randn(N, 3) * 0.02  # Add noise
        >>> 
        >>> # Compute normals (for tangent projection)
        >>> normals = F.normalize(positions, dim=-1)  # Radial normals
        >>> 
        >>> # Setup KNN
        >>> knn = HybridFAISSKNN(tau=0.15)
        >>> 
        >>> # Configure smoothing
        >>> cfg = {
        ...     'iters': 3,
        ...     'k': 24,
        ...     'lambda_smooth': 0.6,
        ...     'lambda_inflate': -0.53,
        ...     'tangent_only': True
        ... }
        >>> 
        >>> # Smooth
        >>> smooth_pos = taubin_smooth(positions, normals, knn, cfg)
        >>> 
        >>> # Check results
        >>> print(f"Input variance: {positions.var():.4f}")
        >>> print(f"Output variance: {smooth_pos.var():.4f}")
        >>> # Both similar (volume preserved) ✓
        >>> 
        >>> # Smoothness improved
        >>> lap_before = compute_laplacian(positions, knn, k=24)
        >>> lap_after = compute_laplacian(smooth_pos, knn, k=24)
        >>> print(f"Roughness before: {lap_before.norm(dim=1).mean():.4f}")
        >>> print(f"Roughness after: {lap_after.norm(dim=1).mean():.4f}")
        >>> # After < Before (smoother) ✓
    
    Example with Gradients:
        >>> # Enable gradient tracking
        >>> positions.requires_grad_(True)
        >>> 
        >>> # Smooth with backprop
        >>> smooth_pos = taubin_smooth(positions, normals, knn, cfg)
        >>> 
        >>> # Compute loss (e.g., from rendering)
        >>> loss = some_rendering_loss(smooth_pos)
        >>> 
        >>> # Backpropagate
        >>> loss.backward()
        >>> 
        >>> # Optimize input positions
        >>> optimizer = torch.optim.Adam([positions], lr=1e-3)
        >>> optimizer.step()
        >>> 
        >>> print(positions.grad.shape)  # (N, 3)
    
    Example Comparing Tangent-Only:
        >>> # Tangent-only (preserve thickness)
        >>> cfg_tangent = {'iters': 3, 'k': 24, 'tangent_only': True}
        >>> smooth_tangent = taubin_smooth(positions, normals, knn, cfg_tangent)
        >>> 
        >>> # Full smoothing (allow thickness change)
        >>> cfg_full = {'iters': 3, 'k': 24, 'tangent_only': False}
        >>> smooth_full = taubin_smooth(positions, normals, knn, cfg_full)
        >>> 
        >>> # Tangent-only preserves distance to origin
        >>> dist_original = torch.norm(positions, dim=1)
        >>> dist_tangent = torch.norm(smooth_tangent, dim=1)
        >>> dist_full = torch.norm(smooth_full, dim=1)
        >>> 
        >>> print(f"Original mean radius: {dist_original.mean():.4f}")
        >>> print(f"Tangent-only mean radius: {dist_tangent.mean():.4f}")
        >>> print(f"Full smoothing mean radius: {dist_full.mean():.4f}")
        >>> # Tangent-only ≈ Original (thickness preserved)
        >>> # Full may differ (thickness changed)
    
    Integration Example:
        >>> class SurfaceSmoother(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.knn = HybridFAISSKNN(tau=0.15)
        ...         self.cfg = {
        ...             'iters': 3,
        ...             'k': 24,
        ...             'lambda_smooth': 0.6,
        ...             'lambda_inflate': -0.53,
        ...             'tangent_only': True
        ...         }
        ...     
        ...     def forward(self, positions, normals):
        ...         # Apply Taubin smoothing as regularization
        ...         smooth_pos = taubin_smooth(
        ...             positions, normals, self.knn, self.cfg
        ...         )
        ...         return smooth_pos
    
    Common Pitfalls and Solutions:
        
        1. **Shrinkage still occurs:**
           Problem: λ₂ not negative enough
           Solution: Use λ₂ ≈ -λ₁/(1-λ₁·κ) with κ=0.1
        
        2. **Oscillation/instability:**
           Problem: |λ₂| too large
           Solution: Use |λ₂| < |λ₁|, typically λ₂ ≈ -0.9·λ₁
        
        3. **Insufficient smoothing:**
           Problem: λ₁ too small or too few iterations
           Solution: Increase λ₁ to 0.6-0.7 or add iterations
        
        4. **Over-smoothing features:**
           Problem: Too many iterations or k too large
           Solution: Use iters ≤ 4, k ≤ 32
        
        5. **Self-intersections:**
           Problem: Using tangent_only=False on thin geometry
           Solution: Set tangent_only=True
    
    Comparison with Other Smoothing:
        
        **Standard Laplacian:**
        - Simpler (one pass)
        - Causes shrinkage ✗
        - Limited iterations
        
        **Taubin (this):**
        - Two passes per iteration
        - No shrinkage ✓
        - Unlimited iterations
        
        **Bilateral:**
        - Edge-preserving
        - More expensive
        - Different use case
    
    Notes:
        - KNN graph computed once per iteration (fixed connectivity)
        - Memory cleanup between passes (del statements)
        - Assumes normals are unit vectors for tangent projection
        - Can be applied iteratively (call multiple times)
        - Works best on relatively uniform point clouds
        - More effective than standard Laplacian for denoising
    
    Performance Tips:
        - Use smaller k for faster computation
        - Reduce iterations if speed critical
        - Consider caching KNN graph if smoothing multiple times
        - Use AMP (automatic mixed precision) for large point clouds
    
    References:
        Gabriel Taubin (1995). "A Signal Processing Approach to Fair Surface Design."
        SIGGRAPH '95. DOI: 10.1145/218380.218473
        
        The original paper introduces the signal processing perspective on
        surface smoothing and derives optimal parameters for volume preservation.
    """
    # ============================================================
    # Parse Configuration Parameters
    # ============================================================
    iters = int(cfg.get("iters", DEFAULT_TAUBIN_ITERS))
    k = int(cfg.get("k", DEFAULT_TAUBIN_K))
    lambda_smooth = float(cfg.get("lambda_smooth", DEFAULT_LAMBDA_SMOOTH))
    lambda_inflate = float(cfg.get("lambda_inflate", DEFAULT_LAMBDA_INFLATE))
    tangent_only = bool(cfg.get("tangent_only", DEFAULT_TANGENT_ONLY))

    # ============================================================
    # Build Fixed KNN Graph
    # ============================================================
    # Classical Taubin uses fixed connectivity to prevent oscillation.
    # We compute the KNN graph ONCE at the beginning and reuse it for
    # all iterations and both passes. This is more stable and efficient
    # than recomputing the graph after each position update.
    idx, w = knn(P, P, k)  # idx: (N, k), w: (N, k) - FIXED!

    # ============================================================
    # Iterative Two-Pass Smoothing
    # ============================================================
    for t in range(iters):
        # ========================================================
        # PASS 1: Smoothing (Positive λ)
        # ========================================================
        # Move points toward their neighborhood centroid to smooth.
        # This reduces high-frequency noise but causes shrinkage.
        
        # Gather neighbor positions using fixed graph
        Q = P[idx]  # (N, k, 3)
        
        # Compute weighted centroid
        centroid = (w.unsqueeze(-1) * Q).sum(dim=1)  # (N, 3)
        
        # Laplacian: displacement from point to centroid
        lap_smooth = centroid - P  # (N, 3)

        # Optional: Project to tangent plane
        if tangent_only:
            # Decompose into tangent and normal components
            # Keep only tangent: lap_t = lap - (lap·n)·n
            lap_n = (lap_smooth * normals).sum(dim=-1, keepdim=True) * normals
            lap_t = lap_smooth - lap_n
            
            # Update positions using tangent component only
            P_temp = P + lambda_smooth * lap_t
        else:
            # Update positions using full Laplacian
            P_temp = P + lambda_smooth * lap_smooth

        # ========================================================
        # PASS 2: Inflation (Negative λ)
        # ========================================================
        # Move points away from their neighborhood centroid to counteract
        # the shrinkage from Pass 1. The negative λ "inflates" the geometry.
        # 
        # IMPORTANT: Reuse the SAME fixed graph (idx, w) for stability.
        
        # Gather neighbor positions from smoothed positions
        Q2 = P_temp[idx]  # (N, k, 3)
        
        # Compute new weighted centroid
        centroid2 = (w.unsqueeze(-1) * Q2).sum(dim=1)  # (N, 3)
        
        # Laplacian on smoothed positions
        lap_inflate = centroid2 - P_temp  # (N, 3)

        # Optional: Project to tangent plane
        if tangent_only:
            # Decompose inflation Laplacian
            lap2_n = (lap_inflate * normals).sum(dim=-1, keepdim=True) * normals
            lap2_t = lap_inflate - lap2_n
            
            # Update with negative λ (inflation)
            P = P_temp + lambda_inflate * lap2_t
        else:
            # Update with negative λ (full inflation)
            P = P_temp + lambda_inflate * lap_inflate

        # ========================================================
        # Memory Cleanup
        # ========================================================
        # Explicitly delete large temporary tensors to help GPU memory reuse
        del Q, centroid, lap_smooth
        del Q2, centroid2, lap_inflate
        
        if tangent_only:
            del lap_n, lap_t, lap2_n, lap2_t

    return P


__all__ = [
    "taubin_smooth",
    "compute_laplacian",
    "split_tangent_normal",
]