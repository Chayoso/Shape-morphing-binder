"""
Differentiable Importance Sampling for Surface Point Generation

This module implements memory-efficient, fully differentiable importance sampling
for upsampling sparse anchor points to dense surface points in physics-guided
Gaussian splatting. It combines Gumbel-Softmax sampling with repulsion fields,
adaptive jitter, and sparse-aware density weighting.

Key Features:
- Full gradient flow for end-to-end training
- Memory-efficient streaming Top-K Gumbel-Softmax
- Repulsion field with gradient support and chunked computation
- Optional coverage seed sampling for uniform initialization
- Low-variance quota allocation with Gumbel residuals
- Sparse-aware importance via inverse-density and spacing bias

Architecture:
- Phase A: Optional coverage seed sampling (detached for stability)
- Phase B: Quota allocation + Gumbel residuals (differentiable)

Author: CHAYO
Version: 2.2.1 (Refactored)
"""

import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Tuple, Optional, Dict

from ..utils.config import CLAMP_GUMBEL, CLAMP_RANDN, CLAMP_SPACING, EPS_SAFE
from ..utils.utils import normalize


# =========================================================
# Constants
# =========================================================
DEFAULT_TAU = 0.2
DEFAULT_ALPHA = 0.45
DEFAULT_THICKNESS_GAMMA = 0.35
DEFAULT_MICRO_JITTER_SCALE = 0.05
DEFAULT_PLANE_SNAP_BETA = 0.5
DEFAULT_INSIDE_BARRIER_LAMBDA = 1.0
DEFAULT_COVERAGE_FRACTION_MAX = 0.15
DEFAULT_GS_BATCH = 4096
DEFAULT_GS_ANCHOR_CHUNK = 16384
DEFAULT_PROB_FLOOR = 1e-6
DEFAULT_UNIFORM_MIX = 0.10
DEFAULT_DENSITY_FLOOR_TAU = 1.0
DEFAULT_DENSITY_FLOOR_GAMMA = 2.0

# Spatial hashing constants for voxel stratification
HASH_PRIME_1 = 73856093
HASH_PRIME_2 = 19349663
HASH_PRIME_3 = 83492791


# =========================================================
# Streaming Top-K Gumbel-Softmax
# =========================================================
def _topk_softmax_streaming_over_N(
    logp_row: torch.Tensor,
    b: int,
    N: int,
    tau: float,
    generator: torch.Generator,
    device: torch.device,
    eligible_mask_f: Optional[torch.Tensor] = None,
    topk_pool: int = 32,
    anchor_chunk: int = 65536,
    use_amp: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-efficient streaming Top-K Gumbel-Softmax sampling.
    
    Processes large anchor sets in chunks to avoid OOM, maintaining only
    the top-K candidates in memory. Uses Gumbel-Softmax for differentiable
    sampling with straight-through estimator.
    
    Args:
        logp_row: (1, N) Log probabilities (single row for broadcasting)
        b: Batch size (number of samples to draw)
        N: Total number of anchors
        tau: Temperature for Gumbel-Softmax. Lower → more discrete.
        generator: Random number generator for reproducibility
        device: Computation device
        eligible_mask_f: Optional (N,) boolean mask for valid anchors
        topk_pool: Number of top candidates to maintain. Default 32.
        anchor_chunk: Chunk size for processing. Default 65536.
        use_amp: Use automatic mixed precision for memory efficiency
        
    Returns:
        y_soft: (b, topk_pool) Soft probabilities over top-K candidates
        topk_inds: (b, topk_pool) Global indices of top-K candidates
        
    Algorithm:
        1. For small N <= anchor_chunk: Direct Top-K
        2. For large N: Stream through chunks, maintain running Top-K
        3. Apply Gumbel noise: logits = (log_p + Gumbel) / tau
        4. Mask ineligible anchors with -inf
        5. Extract Top-K and compute softmax
        
    Notes:
        - Fully differentiable via Gumbel-Softmax trick
        - Memory usage: O(b * topk_pool) regardless of N
        - Streaming ensures scalability to millions of anchors
        
    Example:
        >>> logp = torch.log(torch.rand(1, 100000))
        >>> y_soft, indices = _topk_softmax_streaming_over_N(
        ...     logp, b=256, N=100000, tau=0.2, 
        ...     generator=gen, device='cuda'
        ... )
        >>> print(y_soft.shape, indices.shape)  # (256, 32), (256, 32)
    """
    assert logp_row.shape == (1, N), f"Expected shape (1, {N}), got {logp_row.shape}"
    topk_pool = min(topk_pool, N)

    # Fast path for small N
    if N <= anchor_chunk:
        g = generate_gumbel_noise(b, N, generator, device)
        logits = (logp_row + g) / max(tau, 1e-6)

        if eligible_mask_f is not None:
            mask_expand = eligible_mask_f.unsqueeze(0).expand(b, -1)
            logits = torch.where(
                mask_expand > 0.5, 
                logits, 
                torch.tensor(-1e9, device=device, dtype=logits.dtype)
            )

        topk_logits, topk_inds = torch.topk(logits, topk_pool, dim=1)
        y_soft = F.softmax(topk_logits, dim=1)
        return y_soft, topk_inds
    
    # Streaming path for large N
    topk_logits_running = torch.full(
        (b, topk_pool), -1e9, 
        device=device, dtype=logp_row.dtype
    )
    topk_inds_running = torch.zeros(
        (b, topk_pool), 
        device=device, dtype=torch.long
    )

    autocast_ctx = (
        torch.cuda.amp.autocast 
        if (use_amp and device.type == 'cuda') 
        else nullcontext
    )
    
    with autocast_ctx():
        num_chunks = (N + anchor_chunk - 1) // anchor_chunk
        
        for chunk_idx in range(num_chunks):
            start_n = chunk_idx * anchor_chunk
            end_n = min((chunk_idx + 1) * anchor_chunk, N)
            chunk_size = end_n - start_n

            # Process chunk
            g_chunk = generate_gumbel_noise(b, chunk_size, generator, device)
            logp_chunk = logp_row[:, start_n:end_n]
            logits_chunk = (logp_chunk + g_chunk) / max(tau, 1e-6)

            # Apply eligibility mask
            if eligible_mask_f is not None:
                mask_chunk = eligible_mask_f[start_n:end_n].unsqueeze(0).expand(b, -1)
                logits_chunk = torch.where(
                    mask_chunk > 0.5, 
                    logits_chunk,
                    torch.tensor(-1e9, device=device, dtype=logits_chunk.dtype)
                )

            # Extract top-K from chunk
            chunk_topk = min(topk_pool, chunk_size)
            topk_logits_chunk, topk_inds_local = torch.topk(
                logits_chunk, chunk_topk, dim=1
            )
            topk_inds_chunk = topk_inds_local + start_n

            # Merge with running top-K
            combined_logits = torch.cat([topk_logits_running, topk_logits_chunk], dim=1)
            combined_inds = torch.cat([topk_inds_running, topk_inds_chunk], dim=1)

            topk_logits_running, topk_positions = torch.topk(
                combined_logits, topk_pool, dim=1
            )
            topk_inds_running = torch.gather(combined_inds, 1, topk_positions)

            # Free chunk memory
            del g_chunk, logp_chunk, logits_chunk, topk_logits_chunk, topk_inds_local

    # Final softmax over top-K
    topk_logits_running = topk_logits_running.to(logp_row.dtype)
    y_soft = F.softmax(topk_logits_running, dim=1)
    return y_soft, topk_inds_running


def _voxel_stratified_indices(
    x: torch.Tensor,
    probs: torch.Tensor,
    num_samples: int,
    grid_size: float
) -> torch.Tensor:
    """
    Stratified sampling using spatial voxel grid.
    
    Divides 3D space into voxels and samples from each voxel proportionally
    to its total probability mass. Ensures spatial coverage by preventing
    over-sampling from a single region.
    
    Args:
        x: (N, 3) Point positions
        probs: (N,) Sampling probabilities [0, 1], must sum to ~1
        num_samples: Number of samples to draw
        grid_size: Voxel size for spatial partitioning
        
    Returns:
        indices: (num_samples,) Sampled point indices
        
    Algorithm:
        1. Compute voxel indices using spatial hashing
        2. Aggregate probability per voxel
        3. Sample voxels proportionally to their probability
        4. Within each sampled voxel, sample a point proportionally
        
    Notes:
        - Uses spatial hashing for efficient voxel assignment
        - Fallback to multinomial sampling if grid fails
        - May return fewer samples if voxels are empty
        
    Example:
        >>> points = torch.randn(10000, 3)
        >>> probs = torch.softmax(torch.randn(10000), dim=0)
        >>> indices = _voxel_stratified_indices(points, probs, 1000, 0.1)
        >>> sampled_points = points[indices]
    """
    N = x.shape[0]
    device = x.device
    
    if N == 0 or num_samples == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    # Normalize to grid coordinates
    x_min = x.min(dim=0).values
    x_normalized = (x - x_min) / grid_size
    voxel_inds = x_normalized.floor().long()

    # Spatial hashing: compute unique voxel ID
    voxel_hash = (
        voxel_inds[:, 0] * HASH_PRIME_1 +
        voxel_inds[:, 1] * HASH_PRIME_2 +
        voxel_inds[:, 2] * HASH_PRIME_3
    )

    unique_voxels, inverse_inds = torch.unique(voxel_hash, return_inverse=True)
    num_voxels = unique_voxels.shape[0]

    # Fallback if no voxels
    if num_voxels == 0:
        return torch.multinomial(probs, num_samples, replacement=False)

    # Aggregate probability per voxel
    voxel_probs = torch.zeros(num_voxels, device=device, dtype=probs.dtype)
    voxel_probs.scatter_add_(0, inverse_inds, probs)
    voxel_probs = voxel_probs / (voxel_probs.sum() + 1e-8)

    # Sample voxels
    num_samples_per_voxel = torch.multinomial(
        voxel_probs, num_samples, replacement=True
    )

    # Sample points within each voxel
    sampled_indices = []
    for voxel_idx in num_samples_per_voxel:
        mask = (inverse_inds == voxel_idx)
        points_in_voxel = torch.where(mask)[0]
        
        if points_in_voxel.shape[0] == 0:
            # Empty voxel: sample randomly
            sampled_indices.append(torch.randint(0, N, (1,), device=device))
        else:
            # Sample proportionally within voxel
            probs_in_voxel = probs[points_in_voxel]
            probs_in_voxel = probs_in_voxel / (probs_in_voxel.sum() + 1e-8)
            local_idx = torch.multinomial(probs_in_voxel, 1)
            sampled_indices.append(points_in_voxel[local_idx])

    result = torch.cat(sampled_indices, dim=0)
    
    # Adjust to exact num_samples
    if result.shape[0] > num_samples:
        result = result[:num_samples]
    elif result.shape[0] < num_samples:
        extra = torch.multinomial(probs, num_samples - result.shape[0], replacement=True)
        result = torch.cat([result, extra], dim=0)

    return result


# =========================================================
# Gumbel Noise Generation
# =========================================================
def generate_gumbel_noise(
    batch_M: int,
    N: int,
    generator: torch.Generator,
    device: torch.device
) -> torch.Tensor:
    """
    Generate i.i.d. Gumbel(0, 1) noise.
    
    Gumbel distribution is used in Gumbel-Softmax trick for differentiable
    sampling. Generated via inverse CDF: G = -log(-log(U)) where U ~ Uniform(0,1).
    
    Args:
        batch_M: Batch dimension (number of independent samples)
        N: Number of categories (anchor count)
        generator: PyTorch random generator for reproducibility
        device: Computation device
        
    Returns:
        g: (batch_M, N) Gumbel(0, 1) noise
        
    Notes:
        - Clamped uniform samples prevent log(0) = -inf
        - CLAMP_GUMBEL defined in config (typically [1e-10, 1-1e-10])
        
    Mathematical Background:
        The Gumbel distribution has CDF: F(x) = exp(-exp(-x))
        Inverse CDF: F^(-1)(u) = -log(-log(u))
        
    Example:
        >>> gen = torch.Generator(device='cuda').manual_seed(42)
        >>> g = generate_gumbel_noise(256, 10000, gen, 'cuda')
        >>> print(g.mean(), g.std())  # Should be ~0.577, ~1.28 (Gumbel moments)
    """
    u = torch.rand(batch_M, N, generator=generator, device=device)
    u = torch.clamp(u, *CLAMP_GUMBEL)
    g = -torch.log(-torch.log(u))
    return g


# =========================================================
# Repulsion Field Computation
# =========================================================
def _choose_repulsion_refs(
    write_ptr: int, 
    max_refs: int, 
    stride: int
) -> Tuple[int, slice]:
    """
    Select subset of already-sampled anchors as repulsion references.
    
    Picks from the tail (most recent) with striding to decorrelate batches
    and reduce memory usage. This prevents clustering by repelling new
    samples from previously selected anchors.
    
    Args:
        write_ptr: Current write position (number of points sampled so far)
        max_refs: Maximum number of reference anchors
        stride: Stride for subsampling (e.g., stride=4 → every 4th anchor)
        
    Returns:
        num: Number of reference anchors selected
        ref_slice: Slice object for indexing reference anchors
        
    Strategy:
        - Start from most recent anchors (tail of buffer)
        - Subsample with stride to reduce memory
        - Limit to max_refs to prevent OOM
        
    Example:
        >>> # 1000 points sampled, want ~500 refs with stride=2
        >>> num, slice_obj = _choose_repulsion_refs(1000, 500, 2)
        >>> print(num, slice_obj)  # 500, slice(0, 1000, 2)
        >>> refs = anchors_out[slice_obj]  # Get references
    """
    if write_ptr <= 0:
        return 0, slice(0, 0)
    
    num = min(max_refs, (write_ptr + stride - 1) // stride)
    start = max(0, write_ptr - num * stride)
    return num, slice(start, write_ptr, stride)


def _build_repulsion_field_chunked(
    x: torch.Tensor,
    anchors_out: torch.Tensor,
    write_ptr: int,
    spacing: torch.Tensor,
    *,
    radius_scale: float = 1.5,
    max_refs: int = 20000,
    stride: int = 4,
    chunk_N: int = 32768,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
    enable_grad: bool = True
) -> torch.Tensor:
    """
    Compute differentiable repulsion field using kernel density estimation.
    
    Builds a repulsion field that penalizes sampling near already-selected
    anchors, promoting spatial diversity. Uses Gaussian kernel with adaptive
    bandwidth and memory-efficient chunking.
    
    Args:
        x: (N, 3) Candidate anchor positions
        anchors_out: (M, 3) Buffer of already-sampled anchors
        write_ptr: Number of valid entries in anchors_out
        spacing: (N,) Local spacing at each candidate (used for bandwidth)
        radius_scale: Multiplier for kernel bandwidth. Default 1.5.
        max_refs: Maximum reference anchors for repulsion. Default 20000.
        stride: Subsampling stride for references. Default 4.
        chunk_N: Chunk size for distance computation. Default 32768.
        device: Computation device (defaults to x.device)
        use_amp: Use automatic mixed precision
        enable_grad: Enable gradient computation (True for differentiability)
        
    Returns:
        repulse: (N,) Repulsion field strength [0, inf), normalized to mean=1
        
    Algorithm:
        1. Select reference anchors from already-sampled points
        2. Compute adaptive bandwidth: sigma = radius_scale * median(spacing)
        3. For each chunk of candidates:
           - Compute squared distances to all references
           - Apply Gaussian kernel: k(d) = exp(-d^2 / (2*sigma^2))
           - Average kernel responses
        4. Normalize repulsion field to mean=1
        
    Notes:
        - Fully differentiable when enable_grad=True
        - Memory usage: O(chunk_N * num_refs)
        - Returns zeros if no references available (write_ptr=0)
        - Higher repulsion → lower sampling probability
        
    Mathematical Background:
        Kernel Density Estimation: rho(x) = (1/R) * sum_i K((x - x_i) / h)
        Gaussian kernel: K(d) = exp(-d^2 / 2)
        Bandwidth: h = radius_scale * median(spacing)
        
    Example:
        >>> x = torch.randn(50000, 3, requires_grad=True)
        >>> anchors = torch.randn(10000, 3)
        >>> spacing = torch.rand(50000)
        >>> repulsion = _build_repulsion_field_chunked(
        ...     x, anchors, 10000, spacing,
        ...     enable_grad=True
        ... )
        >>> # Gradients flow back to x
        >>> repulsion.mean().backward()
        >>> print(x.grad.shape)  # (50000, 3)
    """
    if write_ptr <= 0:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    device = device or x.device

    # Select reference anchors
    _, ref_slice = _choose_repulsion_refs(write_ptr, max_refs=max_refs, stride=stride)
    xa = anchors_out[ref_slice]  # (R, 3)
    
    if xa.numel() == 0:
        return torch.zeros(x.shape[0], device=device, dtype=x.dtype)

    # Adaptive bandwidth: use median spacing as scale
    spacing_sorted, _ = torch.sort(spacing)
    median_idx = spacing.shape[0] // 2
    sigma_base = (
        spacing_sorted[median_idx] 
        if spacing.shape[0] > 0 
        else spacing.mean()
    )
    sigma2 = (radius_scale * sigma_base) ** 2 + EPS_SAFE

    N = x.shape[0]
    repulse = torch.empty(N, device=device, dtype=x.dtype)

    # Gradient context
    grad_context = nullcontext() if enable_grad else torch.no_grad()
    autocast_ctx = (
        torch.cuda.amp.autocast 
        if (use_amp and x.is_cuda) 
        else nullcontext
    )
    
    with grad_context:
        with autocast_ctx():
            # Chunked distance computation
            for start in range(0, N, chunk_N):
                end = min(start + chunk_N, N)
                
                # Squared Euclidean distance
                d2 = torch.cdist(x[start:end], xa, p=2) ** 2
                
                # Gaussian kernel
                k = torch.exp(-d2 / (2.0 * sigma2))
                
                # Average over references
                repulse[start:end] = k.mean(dim=1).to(x.dtype)
                
                del d2, k

    # Normalize to mean=1
    repulse = repulse / (repulse.mean() + EPS_SAFE)
    return repulse


# =========================================================
# Density and Spacing Bias
# =========================================================
def _apply_density_or_spacing_bias(
    p: torch.Tensor, 
    spacing: torch.Tensor, 
    cfg: Dict
) -> torch.Tensor:
    """
    Apply gentle differentiable bias using anchor density or spacing.
    
    Adjusts sampling probabilities to favor sparse regions (high spacing,
    low density) while maintaining a floor to prevent complete exclusion.
    This promotes uniform coverage across the surface.
    
    Args:
        p: (N,) Current sampling probabilities (must sum to 1)
        spacing: (N,) Local spacing estimate at each anchor
        cfg: Configuration dict with keys:
            - use_anchor_density: Use density-based bias (default: False)
            - anchor_density_values: (N,) density if use_anchor_density=True
            - anchor_density_beta: Bias strength [0, 1] (default: 0.3)
            - bias_floor: Minimum uniform mixing [0, 1] (default: 0.3)
            - spacing_bias_gamma: Spacing exponent (default: 0.5)
            
    Returns:
        p: (N,) Adjusted probabilities (normalized to sum=1)
        
    Bias Modes:
        1. Density-based (if use_anchor_density=True):
           - bias = (1/density)^beta
           - Higher density → lower sampling probability
           
        2. Spacing-based (fallback):
           - bias = spacing^gamma, clamped to [0.5, 2.5]
           - Higher spacing → higher sampling probability
           
    Floor Blending:
        final_bias = (1-floor) * bias + floor * uniform
        - floor=0.0: Full bias
        - floor=0.3: 30% uniform, 70% biased
        - floor=1.0: No bias (uniform)
        
    Notes:
        - Fully differentiable
        - Returns unchanged if p is not a tensor
        - Normalizes probabilities after bias application
        
    Example:
        >>> p = torch.softmax(torch.randn(1000), dim=0)
        >>> spacing = torch.rand(1000)
        >>> cfg = {'spacing_bias_gamma': 0.5}
        >>> p_biased = _apply_density_or_spacing_bias(p, spacing, cfg)
    """
    if not isinstance(p, torch.Tensor):
        return p

    p = p / (p.sum() + 1e-8)

    # Try density-based bias first
    use_rho = bool(cfg.get("use_anchor_density", False))
    rho = cfg.get("anchor_density_values", None)

    if use_rho and isinstance(rho, torch.Tensor) and rho.numel() == p.numel():
        rho = rho.to(device=p.device, dtype=p.dtype)
        beta = float(cfg.get("anchor_density_beta", 0.3))
        beta = max(0.0, min(beta, 1.0))

        bias_floor = float(cfg.get("bias_floor", 0.3))
        bias_floor = max(0.0, min(bias_floor, 1.0))

        # Inverse density bias: (1/rho)^beta
        bias_weights = (rho + 1e-6).pow(-beta)
        bias_weights = bias_weights / (bias_weights.mean() + 1e-8)

        # Blend with uniform
        uniform_weights = torch.ones_like(bias_weights)
        bias_weights = (
            (1.0 - bias_floor) * bias_weights + 
            bias_floor * uniform_weights
        )

        p = p * bias_weights
        p = p / (p.sum() + 1e-8)
        return p

    # Fallback: spacing-based bias
    if isinstance(spacing, torch.Tensor) and spacing.numel() == p.numel():
        s_norm = spacing.to(device=p.device, dtype=p.dtype) / (spacing.mean() + 1e-8)
        gamma = float(cfg.get("spacing_bias_gamma", 0.5))
        w = torch.clamp(s_norm.pow(gamma), 0.5, 2.5)
        p = p * w
        p = p / (p.sum() + 1e-8)

    return p


# =========================================================
# Dense Gumbel-Softmax Sampler
# =========================================================
def gumbel_softmax_sample(
    probs: torch.Tensor,
    M: int,
    tau: float = DEFAULT_TAU,
    generator: Optional[torch.Generator] = None,
    batch_size: int = 5000
) -> torch.Tensor:
    """
    Dense Gumbel-Softmax sampler that materializes full assignment matrix.
    
    Samples M points from N anchors using Gumbel-Softmax trick with
    straight-through estimator. Returns one-hot-like matrix with gradients
    flowing through soft relaxation.
    
    Args:
        probs: (N,) Sampling probabilities for each anchor
        M: Number of samples to draw
        tau: Temperature for Gumbel-Softmax. Default 0.2.
             Lower → more discrete, Higher → more uniform
        generator: Random number generator for reproducibility
        batch_size: Batch size for processing. Default 5000.
        
    Returns:
        Y: (M, N) Assignment matrix. Each row is a soft one-hot vector.
           Uses straight-through estimator: forward=one-hot, backward=softmax
           
    Algorithm:
        1. Add Gumbel noise to log probabilities
        2. Divide by temperature: logits = (log_p + G) / tau
        3. Compute softmax: y_soft = softmax(logits)
        4. Get hard argmax: y_hard = one_hot(argmax(y_soft))
        5. Straight-through: y = y_hard - detach(y_soft) + y_soft
        
    Notes:
        - Fully differentiable via straight-through estimator
        - Batched processing for memory efficiency
        - Returns stochastic assignments with gradient flow
        
    Memory Usage:
        O(M * N) in final output, but processes in batches of batch_size
        
    Example:
        >>> probs = torch.softmax(torch.randn(1000), dim=0)
        >>> Y = gumbel_softmax_sample(probs, M=5000, tau=0.2)
        >>> print(Y.shape)  # (5000, 1000)
        >>> print(Y.sum(dim=1).mean())  # ~1.0 (each row sums to 1)
        >>> 
        >>> # Gradient flows through Y
        >>> loss = Y.mean()
        >>> loss.backward()
    """
    N = probs.shape[0]
    device = probs.device
    
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(0)

    Y_list = []
    num_batches = (M + batch_size - 1) // batch_size
    safe_probs = torch.clamp(probs, min=1e-10)
    logp = safe_probs.log().unsqueeze(0)  # (1, N)

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, M)
        b = end - start

        # Gumbel-Softmax
        g = generate_gumbel_noise(b, N, generator, device)
        logits = (logp + g) / max(tau, 1e-6)
        y_soft = F.softmax(logits, dim=1)
        
        # Hard argmax
        idx = y_soft.argmax(dim=1)
        y_hard = F.one_hot(idx, num_classes=N).float()
        
        # Straight-through estimator
        y_batch = y_hard - y_soft.detach() + y_soft
        Y_list.append(y_batch)

        del g, logits, y_soft, idx, y_hard

    return torch.cat(Y_list, dim=0)


# =========================================================
# Soft Coverage Sampling
# =========================================================
def _soft_coverage_sample(
    p: torch.Tensor,
    num_samples: int,
    tau: float = 0.1,
    generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable coverage sampling via Gumbel-Softmax.
    
    Samples initial "coverage seeds" to ensure basic spatial coverage
    before importance sampling. Uses Gumbel-Softmax for gradient flow.
    
    Args:
        p: (N,) Sampling probabilities
        num_samples: Number of coverage seeds
        tau: Temperature for Gumbel-Softmax. Lower → more discrete. Default 0.1.
        generator: Random number generator
        
    Returns:
        soft_weights: (num_samples, N) Soft assignment weights
        hard_indices: (num_samples,) Hard sampled indices (argmax)
        
    Notes:
        - Lower temperature (tau) than main sampling for diversity
        - Typically used with detached probabilities for stability
        - Hard indices used for actual point generation
        
    Example:
        >>> p = torch.softmax(torch.randn(1000), dim=0)
        >>> soft, hard = _soft_coverage_sample(p, 100, tau=0.1)
        >>> print(soft.shape, hard.shape)  # (100, 1000), (100,)
    """
    N = p.shape[0]
    device = p.device

    if generator is None:
        generator = torch.Generator(device=device)

    g = generate_gumbel_noise(num_samples, N, generator, device)
    logp = p.log().unsqueeze(0)
    logits = (logp + g) / max(tau, 1e-6)
    soft_weights = F.softmax(logits, dim=1)
    hard_indices = soft_weights.argmax(dim=1)
    
    return soft_weights, hard_indices


# =========================================================
# Tangent Frame and Jitter Generation
# =========================================================
def build_tangent_frame(
    normals: torch.Tensor,
    M: int,
    device: torch.device,
    dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build orthonormal tangent frame for surface point jittering.
    
    Constructs two orthogonal tangent vectors {t1, t2} perpendicular to
    the surface normal at each point. Used for in-plane jittering.
    
    Args:
        normals: (M, 3) Surface normals (assumed normalized)
        M: Number of points
        device: Computation device
        dtype: Data type
        
    Returns:
        t1: (M, 3) First tangent vector
        t2: (M, 3) Second tangent vector
        
    Algorithm:
        1. Choose reference vector: [1,0,0] if |n_x| < 0.9, else [0,1,0]
        2. First tangent: t1 = normalize(ref × n)
        3. Second tangent: t2 = normalize(n × t1)
        4. Result: {t1, t2, n} form orthonormal basis
        
    Notes:
        - Reference vector chosen to avoid degeneracy (parallel vectors)
        - Both tangents normalized and mutually orthogonal
        - Differentiable through normalization
        
    Example:
        >>> normals = F.normalize(torch.randn(1000, 3), dim=-1)
        >>> t1, t2 = build_tangent_frame(normals, 1000, 'cuda', torch.float32)
        >>> # Verify orthogonality
        >>> print((t1 * t2).sum(dim=1).abs().max())  # ~0.0
        >>> print((t1 * normals).sum(dim=1).abs().max())  # ~0.0
    """
    ref = torch.zeros(M, 3, device=device, dtype=dtype)
    abs_nx = normals[:, 0].abs()
    mask = abs_nx < 0.9
    ref[mask, 0] = 1.0
    ref[~mask, 1] = 1.0

    t1 = F.normalize(torch.cross(ref, normals, dim=-1), dim=-1)
    t2 = F.normalize(torch.cross(normals, t1, dim=-1), dim=-1)
    
    return t1, t2


def generate_tangent_jitter(
    M: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random tangent-plane displacements from uniform disk.
    
    Produces random 2D offsets (U, V) uniformly distributed within a unit
    disk. Used for jittering points within the tangent plane.
    
    Args:
        M: Number of samples
        generator: Random number generator
        device: Computation device
        dtype: Data type
        
    Returns:
        U: (M, 1) First tangent component
        V: (M, 1) Second tangent component
        
    Algorithm:
        Inverse CDF method for uniform disk:
        1. Sample radius: r = sqrt(uniform(0,1))
        2. Sample angle: theta = 2π * uniform(0,1)
        3. Convert to Cartesian: U = r*cos(θ), V = r*sin(θ)
        
    Notes:
        - Uniform distribution within disk (not just on boundary)
        - Sqrt transform ensures uniform area density
        - Differentiable through all operations
        
    Example:
        >>> U, V = generate_tangent_jitter(1000, gen, 'cuda', torch.float32)
        >>> r = torch.sqrt(U**2 + V**2)
        >>> print(r.max())  # ~1.0 (within unit disk)
    """
    r = torch.sqrt(torch.rand(M, 1, generator=generator, device=device, dtype=dtype))
    theta = 2.0 * np.pi * torch.rand(M, 1, generator=generator, device=device, dtype=dtype)
    U = r * torch.cos(theta)
    V = r * torch.sin(theta)
    return U, V


def compute_adaptive_jitter_scale(
    spacing: torch.Tensor,
    base_alpha: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Compute adaptive jitter scale with random perturbation.
    
    Adds small random variation to the jitter scale to break regularity
    and improve surface quality. Prevents artifacts from perfectly uniform
    sampling patterns.
    
    Args:
        spacing: (M,) Local spacing at each point
        base_alpha: Base jitter multiplier (typically 0.3-0.6)
        generator: Random number generator
        device: Computation device
        dtype: Data type
        
    Returns:
        alpha: (M, 1) Adaptive jitter scale for each point
        
    Formula:
        alpha = base_alpha * (1 + 0.1 * noise)
        where noise ~ N(0, 1), clamped to [0.5, 1.5]
        
    Notes:
        - 10% random variation around base_alpha
        - Clamping prevents extreme values
        - Differentiable through perturbation
        
    Example:
        >>> spacing = torch.rand(1000)
        >>> alpha = compute_adaptive_jitter_scale(
        ...     spacing, base_alpha=0.45, generator=gen, 
        ...     device='cuda', dtype=torch.float32
        ... )
        >>> print(alpha.min(), alpha.max())  # ~0.5, ~1.5
    """
    M = spacing.shape[0]
    noise = torch.randn(M, 1, generator=generator, device=device, dtype=dtype)
    alpha = base_alpha * (1.0 + 0.1 * noise)
    alpha = torch.clamp(alpha, 0.5, 1.5)
    return alpha


# =========================================================
# Main Sampling Function
# =========================================================
def sample_points(
    x: torch.Tensor,
    normals: torch.Tensor,
    spacing: torch.Tensor,
    probs: torch.Tensor,
    cfg: Dict,
    generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Memory-efficient importance sampling with sparse-aware bias.
    
    Main entry point for upsampling sparse anchors to dense surface points.
    Combines importance sampling, repulsion fields, adaptive jittering, and
    differentiable point generation with full gradient flow for training.
    
    Args:
        x: (N, 3) Anchor positions
        normals: (N, 3) Surface normals at anchors (assumed normalized)
        spacing: (N,) Local spacing estimate (inter-anchor distance)
        probs: (N,) Base importance scores [0, 1] (need not sum to 1)
        cfg: Configuration dictionary with numerous parameters (see below)
        generator: Random number generator for reproducibility
        
    Returns:
        points_out: (M, 3) Generated surface points
        normals_out: (M, 3) Normals at generated points
        anchors_out: (M, 3) Parent anchor for each generated point
        anchor_selection_count: (N,) Number of times each anchor was selected
        
    Configuration Parameters (cfg dict):
        **Required:**
        - M: int - Number of output points
        
        **Sampling:**
        - tau: float - Gumbel temperature (default: 0.2)
        - alpha: float - Tangent jitter scale (default: 0.45)
        - thickness: float - Normal displacement (0 for adaptive, default: 0.0)
        - thickness_gamma: float - Adaptive thickness scale (default: 0.35)
        - thickness_one_sided: bool - One-sided thickness (default: True)
        - micro_jitter_scale: float - Micro-scale noise (default: 0.05)
        - tangent_micro_only: bool - Micro jitter in tangent only (default: True)
        
        **Geometric Constraints:**
        - plane_snap: bool - Snap to tangent plane (default: True)
        - plane_snap_beta: float - Snap strength (default: 0.5)
        - inside_barrier_lambda: float - Prevent inward displacement (default: 1.0)
        
        **Coverage (Phase A):**
        - ensure_anchor_coverage: bool - Use coverage seeds (default: False)
        - coverage_fraction_max: float - Max coverage fraction (default: 0.15)
        
        **Batching:**
        - gs_batch: int - Batch size for point generation (default: 4096)
        - gs_anchor_chunk: int - Chunk size for anchor processing (default: 16384)
        - topk_pool: int - Top-K pool size (default: auto)
        - use_amp_for_logits: bool - Use AMP for logits (default: False)
        - reuse_tmp_buffers: bool - Reuse temp buffers (default: False)
        
        **Probability Adjustment:**
        - prob_floor: float - Minimum probability (default: 1e-6)
        - prob_floor_mode: str - 'density' or 'uniform' (default: 'density')
        - density_floor_tau: float - Floor sigmoid scale (default: 1.0)
        - density_floor_gamma: float - Floor power (default: 2.0)
        - uniform_mix: float - Uniform mixing fraction (default: 0.10)
        
        **Sparse-Aware Bias:**
        - sparse_bias: bool - Enable sparse bias (default: True)
        - sparse_beta: float - Density bias strength (default: 1.0)
        - sparse_spacing_gamma: float - Spacing bias power (default: 0.5)
        - sparse_uniform_mix: float - Secondary uniform mix (default: 0.05)
        
        **Density-Based Bias (from Stage 2):**
        - use_anchor_density: bool - Use anchor density (default: False)
        - anchor_density_values: Tensor - Pre-computed density (N,)
        - anchor_density_beta: float - Density bias strength (default: 0.3)
        - bias_floor: float - Minimum uniform weight (default: 0.3)
        - spacing_bias_gamma: float - Spacing fallback power (default: 0.5)
        
    Algorithm Pipeline:
        **Initialization:**
        1. Parse configuration
        2. Initialize RNG and buffers
        3. Build base probability with floor
        
        **Phase A: Coverage Seeds (Optional)**
        - Sample ~15% of points using detached probabilities
        - Ensures basic coverage before importance sampling
        - Points generated with adaptive jitter
        
        **Phase B: Quota Allocation + Gumbel Residuals**
        1. Compute target counts: n_target = p * remaining
        2. Floor allocation: n_floor = floor(n_target)
        3. Residual allocation: Sample R points via Gumbel-Top-K
           - Residual: R = remaining - sum(n_floor)
           - Score: log(fractional_part) + Gumbel
           - Select top-R by score
        4. Generate points in batches with straight-through gradients
        
        **Point Generation (Both Phases):**
        1. Select anchor and extract (position, normal, spacing)
        2. Build tangent frame {t1, t2, n}
        3. Generate jitter:
           - Tangent: alpha * spacing * (U*t1 + V*t2)
           - Normal: thickness * Z * n
           - Micro: micro_scale * random_3d
        4. Apply geometric constraints:
           - Plane snap: project to tangent plane
           - Inside barrier: prevent inward penetration
        5. Store point, normal, anchor
        
    Memory Management:
        - Streaming Top-K for large anchor sets
        - Chunked repulsion field computation
        - Batched point generation
        - Explicit memory cleanup (del statements)
        - Optional temporary buffer reuse
        
    Differentiability:
        - Phase A: Uses detached probabilities (no gradients)
        - Phase B: Full gradient flow via straight-through estimator
        - Repulsion field: Optional gradient computation
        - All geometric operations: Differentiable
        
    Notes:
        - Exports probability statistics to debug/anchor_probabilities.json
        - Handles edge cases (N=0, M=0, etc.)
        - Supports GPU with mixed precision
        - Reproducible with fixed generator seed
        
    Example:
        >>> # Setup
        >>> anchors = torch.randn(3000, 3, requires_grad=True)
        >>> normals = F.normalize(torch.randn(3000, 3), dim=-1)
        >>> spacing = torch.rand(3000)
        >>> probs = torch.softmax(torch.randn(3000), dim=0)
        >>> 
        >>> cfg = {
        ...     'M': 100000,
        ...     'alpha': 0.45,
        ...     'ensure_anchor_coverage': True,
        ...     'sparse_bias': True,
        ...     'debug': {'verbose': True}
        ... }
        >>> 
        >>> points, norms, parents, counts = sample_points(
        ...     anchors, normals, spacing, probs, cfg
        ... )
        >>> 
        >>> print(points.shape)  # (100000, 3)
        >>> print(counts.sum())  # 100000 (total selections)
        >>> 
        >>> # Backward pass
        >>> loss = points.mean()
        >>> loss.backward()
        >>> print(anchors.grad.shape)  # (3000, 3)
    """
    N = x.shape[0]
    device = x.device
    dtype = x.dtype

    if generator is None:
        generator = torch.Generator(device=device).manual_seed(0)

    # ============================================================
    # Parse Configuration
    # ============================================================
    M = int(cfg['M'])
    tau = float(cfg.get("tau", DEFAULT_TAU))
    alpha = float(cfg.get("alpha", DEFAULT_ALPHA))
    thickness = float(cfg.get("thickness", 0.0))
    thickness_gamma = float(cfg.get("thickness_gamma", DEFAULT_THICKNESS_GAMMA))
    thickness_one_sided = bool(cfg.get("thickness_one_sided", True))
    micro_scale = float(cfg.get("micro_jitter_scale", DEFAULT_MICRO_JITTER_SCALE))
    tangent_micro_only = bool(cfg.get("tangent_micro_only", True))
    plane_snap = bool(cfg.get("plane_snap", True))
    plane_snap_beta = float(cfg.get("plane_snap_beta", DEFAULT_PLANE_SNAP_BETA))
    inside_barrier_lambda = float(cfg.get("inside_barrier_lambda", DEFAULT_INSIDE_BARRIER_LAMBDA))

    ensure_cover = bool(cfg.get("ensure_anchor_coverage", False))
    coverage_fraction_max = float(cfg.get("coverage_fraction_max", DEFAULT_COVERAGE_FRACTION_MAX))

    gs_batch = int(cfg.get("gs_batch", DEFAULT_GS_BATCH))
    gs_anchor_chunk = int(cfg.get("gs_anchor_chunk", DEFAULT_GS_ANCHOR_CHUNK))
    topk_pool = int(cfg.get("topk_pool", max(64, min(4096, N // 200 if N >= 200 else 64))))
    use_amp_for_logits = bool(cfg.get("use_amp_for_logits", False))
    reuse_tmp_buffers = bool(cfg.get("reuse_tmp_buffers", False))

    prob_floor = float(cfg.get("prob_floor", DEFAULT_PROB_FLOOR))
    prob_floor_mode = str(cfg.get("prob_floor_mode", "density"))
    density_floor_tau = float(cfg.get("density_floor_tau", DEFAULT_DENSITY_FLOOR_TAU))
    density_floor_gamma = float(cfg.get("density_floor_gamma", DEFAULT_DENSITY_FLOOR_GAMMA))
    uniform_mix = float(cfg.get("uniform_mix", DEFAULT_UNIFORM_MIX))

    # ============================================================
    # Build Base Probability Distribution
    # ============================================================
    # Adaptive probability floor based on density
    if prob_floor_mode == "density":
        h = spacing / (spacing.mean() + EPS_SAFE)
        floor_scale = torch.sigmoid((h - 1.0) / max(density_floor_tau, 1e-6))
        per_floor = prob_floor * (floor_scale.pow(density_floor_gamma))
    else:
        per_floor = torch.full_like(probs, prob_floor)

    safe_probs = torch.maximum(probs, per_floor)
    p = safe_probs / (safe_probs.sum() + EPS_SAFE)

    # Mix with uniform for robustness
    if uniform_mix > 0.0:
        uniform_vec = torch.full_like(p, 1.0 / N)
        p = (1.0 - uniform_mix) * p + uniform_mix * uniform_vec
        p = p / (p.sum() + EPS_SAFE)

    # Apply Stage-2 density/spacing bias
    p = _apply_density_or_spacing_bias(p, spacing, cfg)

    # ============================================================
    # Sparse-Aware Inverse Density + Spacing Bias
    # ============================================================
    if bool(cfg.get("sparse_bias", True)):
        rho = cfg.get("anchor_density_values", None)
        beta = float(cfg.get("sparse_beta", 1.0))
        gamma = float(cfg.get("sparse_spacing_gamma", 0.5))
        uni_mix2 = float(cfg.get("sparse_uniform_mix", 0.05))

        # Inverse density weight
        if isinstance(rho, torch.Tensor) and rho.numel() == p.numel():
            w_rho = (rho.to(p.device, p.dtype) + 1e-6).pow(-beta)
            w_rho = w_rho / (w_rho.mean() + 1e-8)
        else:
            w_rho = torch.ones_like(p)

        # Spacing weight
        s_norm = spacing.to(p.device, p.dtype) / (spacing.mean() + 1e-8)
        w_h = torch.clamp(s_norm.pow(gamma), 0.5, 2.5)

        # Apply combined bias
        p = p * w_rho * w_h
        p = p / (p.sum() + EPS_SAFE)

        # Secondary uniform mix
        if uni_mix2 > 0.0:
            p = (1.0 - uni_mix2) * p + uni_mix2 * (torch.full_like(p, 1.0 / N))
            p = p / (p.sum() + EPS_SAFE)

    # ============================================================
    # Debug: Export Probability Statistics
    # ============================================================
    import json, os
    os.makedirs("debug", exist_ok=True)
    prob_data = {
        "anchor_count": N,
        "probabilities": {
            "min": float(p.min().item()),
            "max": float(p.max().item()),
            "mean": float(p.mean().item()),
            "std": float(p.std().item()),
        }
    }
    with open("debug/anchor_probabilities.json", "w") as f:
        json.dump(prob_data, f, indent=2)
    
    print(f"\n[DEBUG] Anchor selection probabilities exported")
    print(f"  Min: {prob_data['probabilities']['min']:.6e}")
    print(f"  Max: {prob_data['probabilities']['max']:.6e}")
    print(f"  Ratio: {prob_data['probabilities']['max']/max(prob_data['probabilities']['min'],1e-12):.2f}x\n")

    # ============================================================
    # Initialize Output Buffers
    # ============================================================
    points_out = torch.empty(M, 3, device=device, dtype=dtype)
    normals_out = torch.empty(M, 3, device=device, dtype=dtype)
    anchors_out = torch.empty(M, 3, device=device, dtype=dtype)
    anchor_selection_count = torch.zeros(N, device=device, dtype=torch.long)

    # Optional temporary buffers for reuse
    tmp_delta = tmp_proj = tmp_delta_ortho = None
    if reuse_tmp_buffers:
        tmp_delta = torch.empty(gs_batch, 3, device=device, dtype=dtype)
        tmp_proj = torch.empty(gs_batch, 1, device=device, dtype=dtype)
        tmp_delta_ortho = torch.empty(gs_batch, 3, device=device, dtype=dtype)

    # ============================================================
    # Phase A: Coverage Seed Sampling (Optional)
    # ============================================================
    write_ptr = 0
    if ensure_cover:
        total_cover = int(min(M, max(1, round(coverage_fraction_max * M))))
        total_cover = min(total_cover, N)
        
        if total_cover > 0:
            # Sample using detached probabilities for stability
            base_idx = torch.multinomial(
                p.detach(), 
                num_samples=total_cover, 
                replacement=False
            )
        else:
            base_idx = torch.empty(0, dtype=torch.long, device=device)

        base_count = int(base_idx.numel())
        if base_count > 0:
            # Track selections
            anchor_selection_count.scatter_add_(
                0, base_idx, 
                torch.ones_like(base_idx, dtype=torch.long)
            )

            # Extract anchor data
            xb = x[base_idx]
            nb = normalize(normals[base_idx])
            hb = spacing[base_idx]

            # Generate jittered points
            t1b, t2b = build_tangent_frame(nb, base_count, device, dtype)
            Ub, Vb = generate_tangent_jitter(base_count, generator, device, dtype)
            Zb = torch.rand(base_count, 1, generator=generator, device=device, dtype=dtype)
            if not thickness_one_sided:
                Zb = Zb * 2.0 - 1.0
            
            alpha_b = compute_adaptive_jitter_scale(hb, alpha, generator, device, dtype)

            # Tangent displacement
            tangent_b = alpha_b * hb.unsqueeze(-1) * (Ub * t1b + Vb * t2b)
            
            # Normal displacement
            if thickness == 0.0:
                normal_b = (thickness_gamma * hb.unsqueeze(-1) * Zb) * nb
            else:
                normal_b = (thickness * Zb) * nb

            # Micro-scale jitter
            if tangent_micro_only:
                Um, Vm = generate_tangent_jitter(base_count, generator, device, dtype)
                micro_b = micro_scale * alpha * hb.unsqueeze(-1) * (Um * t1b + Vm * t2b)
            else:
                micro_b = micro_scale * alpha * hb.unsqueeze(-1) * torch.randn(
                    base_count, 3, generator=generator, device=device, dtype=dtype
                )

            pb = xb + tangent_b + normal_b + micro_b

            # Apply geometric constraints
            if plane_snap:
                delta = pb - xb
                proj = (delta * nb).sum(dim=1, keepdim=True) * nb
                delta_ortho = delta - proj
                pb = xb + delta_ortho + plane_snap_beta * proj + normal_b

            if inside_barrier_lambda > 0.0:
                delta = pb - xb
                delta_n = (delta * nb).sum(dim=1, keepdim=True)
                pb = pb + torch.relu(-delta_n) * (inside_barrier_lambda * nb)

            # Store results
            points_out[write_ptr:write_ptr+base_count] = pb
            anchors_out[write_ptr:write_ptr+base_count] = xb
            normals_out[write_ptr:write_ptr+base_count] = nb
            write_ptr += base_count

            # Cleanup
            del xb, nb, hb, t1b, t2b, Ub, Vb, Zb, alpha_b
            del tangent_b, normal_b, micro_b, pb

    # ============================================================
    # Phase B: Quota Allocation + Gumbel Residuals
    # ============================================================
    remain = M - write_ptr
    if remain > 0:
        # Compute target counts
        target = p * remain
        n_floor = torch.floor(target).to(torch.long)
        R = int(remain - int(n_floor.sum().item()))

        # Allocate residuals via Gumbel-Top-K
        if R > 0:
            frac = (target - n_floor.float()).clamp_min(0)
            g = -torch.log(-torch.log(torch.rand_like(frac)))
            score = torch.log(frac + 1e-12) + g
            add_idx = torch.topk(score, k=R, dim=0).indices
            n = n_floor
            n[add_idx] += 1
        else:
            n = n_floor

        # Create shuffled index list
        index_list = torch.repeat_interleave(torch.arange(N, device=device), n)
        perm = torch.randperm(index_list.numel(), device=device, generator=generator)
        index_list = index_list[perm]

        # Process in batches
        num_batches = (remain + gs_batch - 1) // gs_batch
        for bidx in range(num_batches):
            start = write_ptr + bidx * gs_batch
            end = min(write_ptr + (bidx + 1) * gs_batch, M)
            b = end - start
            if b <= 0:
                break

            # Get hard indices for this batch
            hard_idx = index_list[(bidx * gs_batch):(bidx * gs_batch + b)]

            # Extract anchor data
            anchors_hard = x[hard_idx]
            n_hard = normals[hard_idx]
            h_hard = spacing[hard_idx]

            # Track selections
            anchor_selection_count.scatter_add_(
                0, hard_idx, 
                torch.ones_like(hard_idx, dtype=torch.long)
            )

            # Straight-through estimator for gradients
            anchors_soft = anchors_hard.detach() + torch.zeros_like(anchors_hard)
            n_soft = n_hard.detach()
            h_soft = h_hard.detach()

            anchors_st = anchors_hard - anchors_soft.detach() + anchors_soft
            n_st = normalize(n_hard - n_soft.detach() + n_soft)
            h_st = h_hard - h_soft.detach() + h_soft

            # Generate jittered points (same as Phase A)
            t1, t2 = build_tangent_frame(n_st, b, device, dtype)
            U, V = generate_tangent_jitter(b, generator, device, dtype)
            Z = torch.rand(b, 1, generator=generator, device=device, dtype=dtype)
            if not thickness_one_sided:
                Z = Z * 2.0 - 1.0

            alpha_b = compute_adaptive_jitter_scale(h_st, alpha, generator, device, dtype)
            tangent = alpha_b * h_st.unsqueeze(-1) * (U * t1 + V * t2)
            
            if thickness == 0.0:
                normal = (thickness_gamma * h_st.unsqueeze(-1) * Z) * n_st
            else:
                normal = (thickness * Z) * n_st

            if tangent_micro_only:
                Um, Vm = generate_tangent_jitter(b, generator, device, dtype)
                micro = micro_scale * alpha * h_st.unsqueeze(-1) * (Um * t1 + Vm * t2)
            else:
                micro = micro_scale * alpha * h_st.unsqueeze(-1) * torch.randn(
                    b, 3, generator=generator, device=device, dtype=dtype
                )

            pb = anchors_st + tangent + normal + micro

            # Apply geometric constraints
            if plane_snap:
                delta = pb - anchors_st
                proj = (delta * n_st).sum(dim=1, keepdim=True) * n_st
                delta_ortho = delta - proj
                pb = anchors_st + delta_ortho + plane_snap_beta * proj + normal

            if inside_barrier_lambda > 0.0:
                delta = pb - anchors_st
                delta_n = (delta * n_st).sum(dim=1, keepdim=True)
                pb = pb + torch.relu(-delta_n) * (inside_barrier_lambda * n_st)

            # Store results
            points_out[start:end] = pb
            anchors_out[start:end] = anchors_st
            normals_out[start:end] = n_st

            # Cleanup batch
            del hard_idx, anchors_hard, n_hard, h_hard
            del anchors_soft, n_soft, h_soft
            del anchors_st, n_st, h_st
            del t1, t2, U, V, Z, alpha_b, tangent, normal, micro, pb

        write_ptr += remain

    return points_out, normals_out, anchors_out, anchor_selection_count


__all__ = [
    'sample_points',
    'gumbel_softmax_sample',
    'build_tangent_frame',
    'generate_tangent_jitter',
    'compute_adaptive_jitter_scale',
    'generate_gumbel_noise',
    '_build_repulsion_field_chunked',
    '_soft_coverage_sample',
    '_topk_softmax_streaming_over_N',
    '_voxel_stratified_indices',
]