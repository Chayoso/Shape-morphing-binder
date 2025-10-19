"""
Sampling with Gumbel-Softmax and tangent space jittering.

This module implements differentiable point upsampling for surface reconstruction:
- Gumbel-Softmax for differentiable categorical sampling
- Tangent space jittering for surface-aligned perturbations
- Adaptive jitter scaling based on local point density
- Straight-through estimator for discrete sampling with gradient flow

All operations maintain differentiability for end-to-end learning.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from ..utils.config import CLAMP_GUMBEL, CLAMP_RANDN, CLAMP_SPACING, EPS_SAFE
from ..utils.utils import normalize


# ============================================================================
# Gumbel-Softmax Sampling
# ============================================================================

def generate_gumbel_noise(
    batch_M: int,
    N: int,
    generator: torch.Generator,
    device: torch.device
) -> torch.Tensor:
    """
    Generate Gumbel(0, 1) noise for reparameterization trick.
    
    The Gumbel distribution is used in the Gumbel-Max trick for
    sampling from categorical distributions. The Gumbel(0,1) CDF is:
        F(g) = exp(-exp(-g))
    
    Standard sampling:
        U ~ Uniform(0, 1)
        G = -log(-log(U))  # Gumbel(0, 1)
    
    Why Gumbel:
        - Enables differentiable sampling via Gumbel-Softmax
        - Reparameterization trick: randomness in U (no gradient), 
          but log transforms are differentiable
        - Used to convert argmax (discrete) into softmax (continuous)
    
    Clamping:
        - Prevents log(0) and log(log(0)) numerical issues
        - CLAMP_GUMBEL typically [1e-10, 1-1e-10]
    
    Args:
        batch_M: Number of samples to generate
        N: Number of categories (dimension of each sample)
        generator: PyTorch random generator for reproducibility
        device: Device for tensor allocation
    
    Returns:
        g: (batch_M, N) Gumbel(0, 1) noise
           - Independent Gumbel samples for each position
           - Used to perturb log-probabilities in Gumbel-Softmax
    
    Example:
        >>> gen = torch.Generator(device='cuda').manual_seed(42)
        >>> g = generate_gumbel_noise(100, 1000, gen, 'cuda')
        >>> g.shape
        torch.Size([100, 1000])
        >>> # Gumbel mean ≈ 0.577 (Euler's constant), std ≈ 1.28
        >>> g.mean(), g.std()
        (tensor(0.572), tensor(1.26))
    """
    u = torch.rand(batch_M, N, generator=generator, device=device)
    u = torch.clamp(u, *CLAMP_GUMBEL)  # Numerical stability
    g = -torch.log(-torch.log(u))  # Gumbel(0, 1)
    return g


def gumbel_softmax_sample(
    probs: torch.Tensor,
    M: int,
    tau: float = 0.2,
    generator: Optional[torch.Generator] = None,
    batch_size: int = 5000
) -> torch.Tensor:
    """
    Sample M indices from categorical distribution using Gumbel-Softmax.
    
    Classical problem:
        - Need to sample M points from N candidates with probabilities probs
        - torch.multinomial() is discrete (no gradient!)
        - Need differentiable sampling for end-to-end learning
    
    Solution: Gumbel-Softmax (Jang et al. 2017, Maddison et al. 2017)
        
        1. Gumbel-Max trick (discrete):
           i* = argmax_i (log(pᵢ) + Gᵢ)  where Gᵢ ~ Gumbel(0,1)
        
        2. Gumbel-Softmax (continuous relaxation):
           yᵢ = exp((log(pᵢ) + Gᵢ)/τ) / Σⱼ exp((log(pⱼ) + Gⱼ)/τ)
        
        3. Straight-through estimator:
           Forward:  y_hard = one_hot(argmax(y_soft))  (discrete)
           Backward: use y_soft gradients (continuous)
    
    Temperature τ controls sharpness:
        - τ → 0: approaches one-hot (discrete, like argmax)
        - τ → ∞: approaches uniform (continuous, like softmax)
        - τ = 0.2: typical value (sharp but smooth gradients)
    
    Why batched:
        - Sampling M >> N can exceed memory (M × N matrix)
        - Process in batches of batch_size to limit memory
        - Example: M=100k, N=10k → 100k×10k×4 = 4GB per batch
    
    Differentiability:
        - Straight-through estimator: discrete forward, continuous backward
        - Gradients flow through y_soft (softmax) to probs
        - Learns which categories are important via ∂loss/∂probs
    
    Args:
        probs: (N,) categorical distribution (importance weights)
               - Should be non-negative
               - Will be normalized internally (via log-softmax)
               - Can have requires_grad=True for learning
        M: Number of samples to draw
           - Upsampling factor: M > N typically
           - Example: N=10k anchors → M=100k upsampled points
        tau: Temperature for Gumbel-Softmax
             - Lower: more discrete (sharper selection)
             - Higher: more uniform (smoother gradients)
             - Default 0.2 works well for most cases
        generator: Random generator for reproducibility
                   - If None, creates new generator with seed 0
        batch_size: Maximum samples per batch (memory control)
                    - Larger: faster but more memory
                    - Smaller: slower but memory-safe
    
    Returns:
        Y: (M, N) matrix representing sampled indices
           - Each row is approximately one-hot
           - Y[i, j] ≈ 1 if sample i selected category j, ≈ 0 otherwise
           - Differentiable w.r.t. probs via straight-through estimator
           - Use as: sampled_x = Y @ x  (differentiable interpolation)
    
    Complexity:
        - Time: O(M·N) for softmax computations
        - Memory: O(batch_size·N) per iteration
        - Total memory: O(M·N) for output (can be large!)
    
    Example:
        >>> # Importance weights (e.g., from surface variance)
        >>> probs = torch.rand(10000, requires_grad=True)
        >>> probs = probs / probs.sum()  # Normalize
        >>> 
        >>> # Sample 100k points
        >>> Y = gumbel_softmax_sample(probs, M=100000, tau=0.2)
        >>> Y.shape
        torch.Size([100000, 10000])
        >>> 
        >>> # Use for interpolation
        >>> x = torch.randn(10000, 3)
        >>> sampled_points = Y @ x  # (100000, 3)
        >>> 
        >>> # Check differentiability
        >>> loss = sampled_points.sum()
        >>> loss.backward()
        >>> assert probs.grad is not None  # Gradients flow ✓
    
    Notes:
        - Output Y is sparse-ish (mostly zeros, few ones)
        - But represented as dense matrix (trade-off for differentiability)
        - Consider using sparse tensors for very large M, N
    """
    N = probs.shape[0]
    device = probs.device
    
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(0)
    
    Y_list = []
    num_batches = (M + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, M)
        batch_M = end_idx - start_idx
        
        # Step 1: Generate Gumbel noise
        g = generate_gumbel_noise(batch_M, N, generator, device)  # (batch_M, N)
        
        # Step 2: Gumbel-Softmax
        safe_probs = torch.clamp(probs, min=1e-10)  # Numerical stability
        logits = (safe_probs.log().unsqueeze(0) + g) / max(tau, 1e-6)  # (batch_M, N)
        y_soft = F.softmax(logits, dim=1)  # (batch_M, N) - continuous
        
        # Step 3: Straight-through estimator
        idx = y_soft.argmax(dim=1)  # (batch_M,) - hard selection
        y_hard = F.one_hot(idx, num_classes=N).float()  # (batch_M, N) - one-hot
        
        # Trick: forward uses y_hard, backward uses y_soft
        y_batch = y_hard - y_soft.detach() + y_soft  # (batch_M, N)
        
        Y_list.append(y_batch)
        
        # Free memory
        del g, logits, y_soft, idx, y_hard
    
    Y = torch.cat(Y_list, dim=0)  # (M, N)
    return Y


# ============================================================================
# Tangent Frame & Jittering
# ============================================================================

def build_tangent_frame(
    normals: torch.Tensor,
    M: int,
    device: torch.device,
    dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build orthonormal tangent frame {t1, t2} perpendicular to normals.
    
    Problem:
        Given surface normal n, we need two orthogonal tangent vectors
        t1, t2 such that {t1, t2, n} form an orthonormal basis.
    
    Solution: Gram-Schmidt orthogonalization
        1. Start with arbitrary vector a (e.g., [1,0,0])
        2. If a ≈ n (nearly parallel), use different vector (e.g., [0,1,0])
        3. Orthogonalize: t1 = normalize(a - (a·n)n)  (project a onto tangent plane)
        4. Complete frame: t2 = normalize(n × t1)  (cross product)
    
    Why tangent frame:
        - Allows surface-aligned perturbations (jittering)
        - Preserves surface structure during upsampling
        - Natural parameterization for 2D offsets on surface
    
    Differentiability:
        - All operations differentiable (normalize, cross product)
        - Gradients flow through normals to influence frame orientation
        - Used in differentiable rendering and surface optimization
    
    Stability:
        - Handles degenerate case when n ≈ [1,0,0]
        - Threshold 0.9 prevents nearly parallel vectors
        - Ensures numerical stability of orthogonalization
    
    Args:
        normals: (M, 3) surface normal vectors
                 - Should be unit vectors (will work if not, but slower)
                 - Can have requires_grad=True
        M: Number of points
        device: Device for tensor allocation
        dtype: Data type (typically float32)
    
    Returns:
        t1: (M, 3) first tangent vector
            - Perpendicular to normals
            - Unit length
            - Roughly aligned with x-axis (unless n ≈ x-axis)
        t2: (M, 3) second tangent vector
            - Perpendicular to both normals and t1
            - Unit length
            - Completes right-handed coordinate system
    
    Properties:
        - t1 · n = 0  (perpendicular)
        - t2 · n = 0  (perpendicular)
        - t1 · t2 = 0  (mutually perpendicular)
        - ||t1|| = ||t2|| = ||n|| = 1  (unit vectors)
        - t2 = n × t1  (right-handed)
    
    Complexity:
        - Time: O(M·3) for vector operations
        - Memory: O(M·3) for each output
    
    Example:
        >>> normals = torch.randn(1000, 3)
        >>> normals = normalize(normals)  # Unit normals
        >>> t1, t2 = build_tangent_frame(normals, 1000, 'cuda', torch.float32)
        >>> 
        >>> # Check orthogonality
        >>> torch.allclose(
        ...     torch.einsum('md,md->m', t1, normals),
        ...     torch.zeros(1000),
        ...     atol=1e-5
        ... )
        True
        >>> 
        >>> # Check unit length
        >>> torch.allclose(torch.norm(t1, dim=1), torch.ones(1000))
        True
    """
    # Start with x-axis as reference
    a = torch.tensor([1., 0., 0.], device=device, dtype=dtype).expand(M, 3).clone()
    
    # Detect near-parallel cases (normal ≈ x-axis)
    dot_ax = torch.abs(torch.einsum('md,md->m', normals, a))  # (M,)
    parallel_mask = dot_ax > 0.9  # Threshold for "too parallel"
    
    # For parallel cases, use y-axis instead
    a[parallel_mask] = torch.tensor([0., 1., 0.], device=device, dtype=dtype)
    
    # Gram-Schmidt: orthogonalize a w.r.t. normal
    # t1 = a - (a·n)n  (project a onto tangent plane)
    proj = torch.einsum('md,md->m', a, normals).unsqueeze(-1)  # (M, 1)
    t1 = normalize(a - proj * normals)  # (M, 3)
    
    # Complete orthonormal frame via cross product
    # t2 = n × t1  (perpendicular to both)
    t2 = normalize(torch.cross(normals, t1, dim=1))  # (M, 3)
    
    return t1, t2


def generate_tangent_jitter(
    M: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random tangent space displacements with rotation.
    
    Purpose:
        Create random 2D offsets in tangent space, then randomly rotate
        to avoid axis-aligned artifacts during upsampling.
    
    Algorithm:
        1. Sample (U, V) from standard normal (Gaussian jitter)
        2. Sample random rotation angle θ ∈ [0, 2π]
        3. Apply rotation: (U', V') = Rot(θ) @ (U, V)
    
    Why rotation:
        - Prevents grid-aligned patterns in upsampled points
        - Isotropic distribution in tangent plane
        - More natural surface coverage
    
    Why Gaussian:
        - Most points near anchor (concentrated)
        - Rare points far from anchor (smooth falloff)
        - Standard in point cloud jittering
    
    Differentiability:
        - Random noise (U, V, θ) has no gradient (reparameterization trick)
        - But operations on noise are differentiable
        - Used in stochastic gradient methods
    
    Args:
        M: Number of points to generate jitter for
        generator: Random generator for reproducibility
        device: Device for tensor allocation
        dtype: Data type
    
    Returns:
        U_rot: (M, 1) rotated U-component of tangent offset
               - Gaussian distributed, rotated
               - Clamped to CLAMP_RANDN (typically [-3, 3]) for stability
        V_rot: (M, 1) rotated V-component of tangent offset
               - Gaussian distributed, rotated
               - Clamped to CLAMP_RANDN for stability
    
    Usage:
        tangent_offset = U_rot * t1 + V_rot * t2
        where t1, t2 are tangent basis vectors
    
    Complexity:
        - Time: O(M) for random generation and rotation
        - Memory: O(M) for outputs
    
    Example:
        >>> gen = torch.Generator(device='cuda').manual_seed(42)
        >>> U_rot, V_rot = generate_tangent_jitter(1000, gen, 'cuda', torch.float32)
        >>> U_rot.shape, V_rot.shape
        (torch.Size([1000, 1]), torch.Size([1000, 1]))
        >>> 
        >>> # Check distribution (approximately Gaussian after rotation)
        >>> U_rot.mean(), U_rot.std()
        (tensor(0.02), tensor(0.97))  # Mean ≈ 0, std ≈ 1
    """
    # Sample Gaussian noise
    U = torch.randn(M, 1, generator=generator, device=device, dtype=dtype)
    V = torch.randn(M, 1, generator=generator, device=device, dtype=dtype)
    
    # Clamp for numerical stability (avoid extreme outliers)
    U = U.clamp(*CLAMP_RANDN)  # Typically [-3, 3]
    V = V.clamp(*CLAMP_RANDN)
    
    # Random rotation angle per point
    theta = torch.rand(M, 1, generator=generator, device=device, dtype=dtype) * 2 * np.pi
    c, s = torch.cos(theta), torch.sin(theta)  # (M, 1)
    
    # 2D rotation matrix: [c -s; s c]
    U_rot = U * c - V * s  # (M, 1)
    V_rot = U * s + V * c  # (M, 1)
    
    return U_rot, V_rot


def compute_adaptive_jitter_scale(
    spacing: torch.Tensor,
    alpha: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Compute adaptive jitter scale based on local point density.
    
    Motivation:
        - Dense regions: points close together → small jitter (preserve detail)
        - Sparse regions: points far apart → large jitter (fill gaps)
        - Uniform jitter would over-smooth dense areas and under-fill sparse areas
    
    Formula:
        scale = α · noise · (spacing / mean_spacing)
    
    where:
        - α: base jitter magnitude (global parameter)
        - noise: random variation ∈ [0.4, 1.6] (prevents uniformity)
        - spacing / mean_spacing: local density adaptation
    
    Adaptive scaling:
        - spacing > mean: sparse region → scale > α (larger jitter)
        - spacing < mean: dense region → scale < α (smaller jitter)
        - Clamped to CLAMP_SPACING (typically [0.3, 2.0]) for stability
    
    Why random noise:
        - Adds stochasticity for better coverage
        - Prevents axis-aligned or grid-like patterns
        - Range [0.4, 1.6]: moderate variation (not too uniform, not too chaotic)
    
    Differentiability:
        - Spacing can have gradients (from point positions)
        - Noise is non-differentiable (reparameterization trick)
        - Scale operation is differentiable w.r.t. spacing
    
    Args:
        spacing: (M,) local point spacing from KNN analysis
                 - Larger values: sparse neighborhoods
                 - Smaller values: dense neighborhoods
                 - Can have requires_grad=True
        alpha: Base jitter magnitude (global scale)
               - Typical range: 0.2-0.5
               - Controls overall jitter strength
        generator: Random generator
        device: Device for tensor allocation
        dtype: Data type
    
    Returns:
        alpha_adapt: (M, 1) adaptive jitter scales
                     - One scale per point
                     - Automatically adjusts to local density
                     - Differentiable w.r.t. spacing
    
    Complexity:
        - Time: O(M) for computations
        - Memory: O(M) for output
    
    Example:
        >>> spacing = torch.tensor([0.1, 0.5, 1.0])  # varying densities
        >>> gen = torch.Generator().manual_seed(42)
        >>> alpha_adapt = compute_adaptive_jitter_scale(
        ...     spacing, alpha=0.35, generator=gen,
        ...     device='cpu', dtype=torch.float32
        ... )
        >>> alpha_adapt.squeeze()
        tensor([0.12, 0.42, 0.95])  # Scales with spacing ✓
    """
    M = spacing.shape[0]
    
    # Random scale variation per point
    # Range [0.4, 1.6]: moderate randomness
    alpha_noise = 0.4 + torch.rand(M, 1, generator=generator, device=device, dtype=dtype) * 1.2
    
    # Adaptive scaling based on local density
    # spacing / mean_spacing: relative density (1.0 = average)
    h_scale = spacing / (spacing.mean() + EPS_SAFE)  # (M,)
    h_scale = torch.clamp(h_scale, *CLAMP_SPACING).unsqueeze(-1)  # (M, 1), clamped
    
    # Combined scale: base × noise × density
    alpha_adapt = float(alpha) * alpha_noise * h_scale  # (M, 1)
    
    return alpha_adapt


# ============================================================================
# Main Sampling Function
# ============================================================================

def sample_points(
    x: torch.Tensor,
    normals: torch.Tensor,
    spacing: torch.Tensor,
    probs: torch.Tensor,
    cfg: Dict,
    generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample M upsampled points from N anchors with tangent space jittering.
    
    This is the main upsampling function that combines:
        1. Importance sampling (Gumbel-Softmax)
        2. Tangent space jittering (surface-aligned)
        3. Adaptive scaling (density-aware)
        4. Multi-scale perturbations (tangent + normal + micro)
    
    Algorithm:
        1. Sample M indices from N anchors using Gumbel-Softmax (differentiable)
        2. Interpolate anchor properties: positions, normals, spacing
        3. Build local tangent frame {t1, t2, n} for each sampled point
        4. Generate random tangent offsets with rotation
        5. Compute adaptive jitter scale based on local spacing
        6. Apply multi-scale perturbations:
           a. Tangent offset: α·h·(U·t1 + V·t2)  (surface-aligned)
           b. Normal offset: thickness·Z·n  (surface thickness)
           c. Micro jitter: 0.2·α·h·ε  (high-frequency detail)
        7. Combine: x_new = anchor + tangent + normal + micro
    
    Why this design:
        - Importance sampling: focuses upsampling on uncertain/complex regions
        - Tangent jitter: preserves surface structure (stays on surface)
        - Adaptive scale: respects local density (dense → small, sparse → large)
        - Multi-scale: captures both surface variation and fine detail
        - Differentiable: end-to-end learning of sampling distribution
    
    Upsampling factor:
        M / N typically 5-20×
        Example: N=10k anchors → M=100k upsampled points
    
    Differentiability:
        All operations differentiable:
        - Gumbel-Softmax: straight-through estimator ✓
        - Interpolation: matrix multiplication (Y @ x) ✓
        - Tangent frame: normalize, cross product ✓
        - Jitter application: linear combination ✓
        
        Gradients flow to:
        - probs: learn importance weights
        - x: optimize anchor positions
        - normals: refine surface orientation
        - spacing: adjust local scale
    
    Args:
        x: (N, 3) anchor point positions
           - Typically from downsampled or sparse point cloud
           - Can have requires_grad=True for position optimization
        normals: (N, 3) surface normals at anchors
                 - Should be unit vectors
                 - Can have requires_grad=True
        spacing: (N,) local point spacing at anchors
                 - From KNN analysis (average distance to neighbors)
                 - Used for adaptive jitter scaling
        probs: (N,) importance weights for sampling
               - Higher prob → more likely to sample near this anchor
               - Should be non-negative (will be normalized)
               - Can have requires_grad=True to learn importance
        cfg: Configuration dictionary with keys:
             - 'M' (int): Number of upsampled points (default: 50000)
             - 'tau' (float): Gumbel-Softmax temperature (default: 0.2)
             - 'alpha' (float): Base jitter magnitude (default: 0.35)
             - 'thickness' (float): Normal offset scale (default: 0.0)
                * 0.0: no thickness (2D surface)
                * 0.1: thin shell (volumetric)
        generator: Random generator for reproducibility
                   - If None, creates new with seed 0
    
    Returns:
        points: (M, 3) upsampled point positions
                - Jittered around sampled anchors
                - Surface-aligned perturbations
                - Differentiable w.r.t. inputs
        
        normals_out: (M, 3) interpolated normals
                     - Corresponding to sampled points
                     - Unit vectors
                     - Differentiable w.r.t. input normals
        
        anchors: (M, 3) anchor positions (before jitter)
                 - The "base" positions that were jittered
                 - Useful for visualization and debugging
                 - Differentiable w.r.t. input x
    
    Complexity:
        - Time: O(M·N) for Gumbel-Softmax + O(M) for jittering
        - Memory: O(M·N) for sampling matrix Y (can be large!)
        - Dominant cost: Gumbel-Softmax (batched to limit memory)
    
    Example:
        >>> # Setup anchors
        >>> N = 10000
        >>> x = torch.randn(N, 3, requires_grad=True)
        >>> normals = normalize(torch.randn(N, 3, requires_grad=True))
        >>> spacing = torch.rand(N) * 0.1 + 0.05  # [0.05, 0.15]
        >>> 
        >>> # Importance weights (e.g., from surface variance)
        >>> probs = torch.rand(N, requires_grad=True)
        >>> probs = probs / probs.sum()
        >>> 
        >>> # Configuration
        >>> cfg = {
        ...     'M': 100000,     # 10× upsampling
        ...     'tau': 0.2,      # Sharp sampling
        ...     'alpha': 0.35,   # Moderate jitter
        ...     'thickness': 0.0 # Pure surface
        ... }
        >>> 
        >>> # Upsample
        >>> points, normals_out, anchors = sample_points(
        ...     x, normals, spacing, probs, cfg
        ... )
        >>> 
        >>> points.shape       # (100000, 3) ✓
        >>> normals_out.shape  # (100000, 3) ✓
        >>> 
        >>> # Check differentiability
        >>> loss = points.sum()
        >>> loss.backward()
        >>> assert x.grad is not None        # Position gradients ✓
        >>> assert probs.grad is not None    # Importance gradients ✓
        >>> assert normals.grad is not None  # Normal gradients ✓
    
    Notes:
        - Micro jitter (0.2 factor) adds high-frequency detail
        - Thickness parameter for volumetric reconstruction
        - All random operations use generator for reproducibility
        - Memory scales with M·N (use smaller batches if OOM)
    """
    device, dtype = x.device, x.dtype
    
    # Parse configuration
    M = int(cfg.get("M", 50000))
    tau = float(cfg.get("tau", 0.2))
    alpha = float(cfg.get("alpha", 0.35))
    thickness = float(cfg.get("thickness", 0.0))
    
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(0)
    
    # Step 1: Sample indices using Gumbel-Softmax (differentiable)
    Y = gumbel_softmax_sample(probs, M=M, tau=tau, generator=generator)  # (M, N)
    
    # Step 2: Interpolate anchor properties
    anchors = Y @ x  # (M, 3) - weighted combination of anchor positions
    n = normalize(Y @ normals)  # (M, 3) - interpolated normals (unit)
    h = (Y @ spacing.unsqueeze(1)).squeeze(1)  # (M,) - interpolated spacing
    
    # Step 3: Build tangent frame for surface-aligned jittering
    t1, t2 = build_tangent_frame(n, M, device, dtype)  # (M, 3), (M, 3)
    
    # Step 4: Generate random tangent displacements (with rotation)
    U_rot, V_rot = generate_tangent_jitter(M, generator, device, dtype)  # (M, 1), (M, 1)
    
    # Step 5: Random normal displacement (for thickness)
    Z = (torch.rand(M, 1, generator=generator, device=device, dtype=dtype) * 2.0 - 1.0)  # [-1, 1]
    
    # Step 6: Compute adaptive jitter scale
    alpha_adapt = compute_adaptive_jitter_scale(h, alpha, generator, device, dtype)  # (M, 1)
    
    # Step 7: Apply multi-scale perturbations
    
    # Tangent offset (surface-aligned)
    tangent_offset = alpha_adapt * h.unsqueeze(-1) * (U_rot * t1 + V_rot * t2)  # (M, 3)
    
    # Normal offset (thickness / volumetric)
    normal_offset = (thickness * Z) * n  # (M, 3)
    
    # Micro jitter (high-frequency detail)
    micro_eps = torch.randn(M, 3, generator=generator, device=device, dtype=dtype)
    micro_jitter = 0.2 * alpha * h.unsqueeze(-1) * micro_eps  # (M, 3)
    
    # Step 8: Combine all offsets
    points = anchors + tangent_offset + normal_offset + micro_jitter  # (M, 3)
    
    return points, n, anchors


__all__ = [
    'sample_points',
    'gumbel_softmax_sample',
    'build_tangent_frame',
    'generate_tangent_jitter',
    'compute_adaptive_jitter_scale',
    'generate_gumbel_noise',
]