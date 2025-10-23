
"""
Sampling with Gumbel-Softmax and tangent space jittering (memory-safe, coverage-safe).

Key changes vs. the dense-Y version:
- No dense Y ∈ ℝ^{M×N}. We stream small batches, compute soft weights y_soft on-the-fly,
  immediately project to anchors/normals/spacing and discard y_soft (low memory).
- Straight-through (ST) mixing per batch to keep gradients wrt probs:
      out = hard - soft.detach() + soft
- Optional "ensure_anchor_coverage": include every anchor at least once when M ≥ N
  so all particles can render at least one Gaussian.

PATCH APPLIED (2025-10):
- ✅ FIX 1: Remove hard zeros → prob_floor + uniform_mix
- ✅ FIX 2: Coverage seed includes ALL anchors
- ✅ FIX 3: Soft plane-snap with beta parameter
- ✅ FIX 4: Top-k candidate pool for anti-mode-collapse
- ✅ FIX 5: Adaptive thickness scaling with spacing
- 🔥 NEW FIX 6: Surface-constrained uniform mix
- 🔥 NEW FIX 7: Density-based floor (spacing-weighted)
- 🔥 NEW FIX 8: One-sided thickness with inside barrier
- 🔥 NEW FIX 9: Surface-only coverage seed
- 🔥 NEW FIX 10: Surface mask on top-k pool
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
    Generate Gumbel(0, 1) noise for reparameterization.
    """
    u = torch.rand(batch_M, N, generator=generator, device=device)
    u = torch.clamp(u, *CLAMP_GUMBEL)
    g = -torch.log(-torch.log(u))
    return g


def gumbel_softmax_sample(
    probs: torch.Tensor,
    M: int,
    tau: float = 0.2,
    generator: Optional[torch.Generator] = None,
    batch_size: int = 5000
) -> torch.Tensor:
    """
    Return dense selection matrix Y ∈ ℝ^{M×N} (approx. one-hot rows).

    NOTE (memory): For large N and M this dense output is prohibitive.
    Prefer calling `sample_points` which now uses streaming ST and never
    materializes Y. This function remains for API compatibility.
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
        start, end = i * batch_size, min((i + 1) * batch_size, M)
        b = end - start
        g = generate_gumbel_noise(b, N, generator, device)
        logits = (logp + g) / max(tau, 1e-6)
        y_soft = F.softmax(logits, dim=1)
        idx = y_soft.argmax(dim=1)
        y_hard = F.one_hot(idx, num_classes=N).float()
        y_batch = y_hard - y_soft.detach() + y_soft
        Y_list.append(y_batch)
        del g, logits, y_soft, idx, y_hard

    return torch.cat(Y_list, dim=0)


# ============================================================================
# Tangent frame & jitter (unchanged signatures)
# ============================================================================

def build_tangent_frame(
    normals: torch.Tensor,
    M: int,
    device: torch.device,
    dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build orthonormal tangent frame {t1, t2} perpendicular to normals.
    """
    a = torch.tensor([1., 0., 0.], device=device, dtype=dtype).expand(M, 3).clone()
    dot_ax = torch.abs(torch.einsum('md,md->m', normals, a))
    parallel_mask = dot_ax > 0.9
    a[parallel_mask] = torch.tensor([0., 1., 0.], device=device, dtype=dtype)

    proj = torch.einsum('md,md->m', a, normals).unsqueeze(-1)
    t1 = normalize(a - proj * normals)
    t2 = normalize(torch.cross(normals, t1, dim=1))
    return t1, t2


def generate_tangent_jitter(
    M: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random 2D offsets (U,V) in tangent space with per-point rotation.
    """
    U = torch.randn(M, 1, generator=generator, device=device, dtype=dtype).clamp(*CLAMP_RANDN)
    V = torch.randn(M, 1, generator=generator, device=device, dtype=dtype).clamp(*CLAMP_RANDN)

    theta = torch.rand(M, 1, generator=generator, device=device, dtype=dtype) * 2 * np.pi
    c, s = torch.cos(theta), torch.sin(theta)
    U_rot = U * c - V * s
    V_rot = U * s + V * c
    return U_rot, V_rot


def compute_adaptive_jitter_scale(
    spacing: torch.Tensor,
    alpha: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Compute adaptive jitter scale based on local spacing.
    scale = α · noise · clamp(spacing / mean_spacing, CLAMP_SPACING)
    """
    M = spacing.shape[0]
    alpha_noise = 0.4 + torch.rand(M, 1, generator=generator, device=device, dtype=dtype) * 1.2
    h_scale = spacing / (spacing.mean() + EPS_SAFE)
    h_scale = torch.clamp(h_scale, *CLAMP_SPACING).unsqueeze(-1)
    return float(alpha) * alpha_noise * h_scale


# ============================================================================
# Main sampling (FULLY PATCHED VERSION - 2025-10)
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
    Memory-safe, coverage-safe upsampling with COMPREHENSIVE HOLE-FIX patches:
      - NO hard zeros in sampling distribution (prob_floor + uniform_mix)
      - 🔥 NEW: Surface-constrained uniform mix (only to top-q anchors)
      - 🔥 NEW: Density-based floor (spacing-weighted per-anchor)
      - 🔥 NEW: Surface-only coverage seed
      - 🔥 NEW: Surface mask on top-k candidate pool
      - 🔥 NEW: One-sided thickness with inside barrier
      - Soft plane-snap with beta parameter
      - Adaptive thickness scaling
    """
    device, dtype = x.device, x.dtype
    N = x.shape[0]

    # --- config ---
    M = int(cfg.get("M", 50000))
    tau = float(cfg.get("tau", 0.3))
    alpha = float(cfg.get("alpha", 0.35))
    thickness = float(cfg.get("thickness", 0.0))
    gs_batch = int(cfg.get("gs_batch", 2048))
    ensure_cover = bool(cfg.get("ensure_anchor_coverage", True))
    micro_scale = float(cfg.get("micro_jitter_scale", 0.2))
    tangent_micro_only = bool(cfg.get("tangent_micro_only", True))

    # 🔥 PATCH FIX 1: Remove hard zeros
    prob_floor = float(cfg.get("prob_floor", 1e-8))
    uniform_mix = float(cfg.get("uniform_mix", 0.02))

    # 🔥 PATCH FIX 3: Soft plane-snap
    plane_snap = bool(cfg.get("plane_snap", True))
    plane_snap_beta = float(cfg.get("plane_snap_beta", 0.5))

    # 🔥 PATCH FIX 4: Top-k candidate pool
    topk_pool = int(cfg.get("topk_pool", 8))

    # 🔥 PATCH FIX 5: Adaptive thickness
    thickness_gamma = float(cfg.get("thickness_gamma", 0.15))

    # 🔥 NEW FIX 6-10: Surface-constrained sampling
    surface_support_q = float(cfg.get("surface_support_q", 0.80))
    min_surface_anchors = int(cfg.get("min_surface_anchors", max(256, N // 20)))
    uniform_mix_surface_only = bool(cfg.get("uniform_mix_surface_only", True))
    coverage_only_surface = bool(cfg.get("coverage_only_surface", True))
    mask_topk_with_surface = bool(cfg.get("mask_topk_with_surface", True))
    
    # 🔥 NEW FIX 7: Density-based floor
    prob_floor_mode = str(cfg.get("prob_floor_mode", "density"))
    density_floor_tau = float(cfg.get("density_floor_tau", 1.0))
    density_floor_gamma = float(cfg.get("density_floor_gamma", 2.0))
    dead_anchor_eps = float(cfg.get("dead_anchor_eps", 1e-12))
    
    # 🔥 NEW FIX 8: One-sided thickness with inside barrier
    thickness_one_sided = bool(cfg.get("thickness_one_sided", True))
    inside_barrier_lambda = float(cfg.get("inside_barrier_lambda", 1.0))

    if generator is None:
        generator = torch.Generator(device=device).manual_seed(0)

    # ========================================================================
    # 🔥 NEW: Build surface eligible mask (self-guided from probability)
    # ========================================================================
    with torch.no_grad():
        q_thr = torch.quantile(probs.detach(), torch.tensor(surface_support_q, device=device))
        eligible_mask = (probs.detach() >= q_thr)  # (N,)
        
        # Ensure minimum number of surface anchors
        if eligible_mask.sum() < min_surface_anchors:
            topk_idx = torch.topk(probs.detach(), k=min_surface_anchors, largest=True).indices
            tmp = torch.zeros_like(eligible_mask)
            tmp[topk_idx] = True
            eligible_mask = tmp
    
    eligible_mask_f = eligible_mask.float()

    # ========================================================================
    # 🔥 NEW FIX 7: Density-based floor (spacing-weighted)
    # ========================================================================
    if prob_floor_mode == "density":
        # h_i = spacing_i / mean(spacing)
        h = spacing / (spacing.mean() + EPS_SAFE)
        # floor_scale = sigmoid((h - 1) / tau) ^ gamma
        floor_scale = torch.sigmoid((h - 1.0) / max(density_floor_tau, 1e-6))
        per_floor = prob_floor * (floor_scale.pow(density_floor_gamma))
    else:
        per_floor = torch.full_like(probs, prob_floor)

    # Apply floor only to surface anchors, dead_anchor_eps to inside
    safe_floor = per_floor * eligible_mask_f + dead_anchor_eps * (1.0 - eligible_mask_f)
    safe_probs = torch.maximum(probs, safe_floor)
    p = safe_probs / (safe_probs.sum() + EPS_SAFE)

    # ========================================================================
    # 🔥 NEW FIX 6: Surface-constrained uniform mix
    # ========================================================================
    if uniform_mix > 0.0:
        if uniform_mix_surface_only and eligible_mask.any():
            # Distribute uniform mass only to surface anchors
            uniform_vec = eligible_mask_f / (eligible_mask_f.sum() + EPS_SAFE)
        else:
            # Original: uniform to all anchors
            uniform_vec = torch.full_like(p, 1.0 / N)
        p = (1.0 - uniform_mix) * p + uniform_mix * uniform_vec

    logp = p.log().unsqueeze(0)  # (1, N)

    # --- outputs ---
    points_out = torch.empty(M, 3, device=device, dtype=dtype)
    normals_out = torch.empty(M, 3, device=device, dtype=dtype)
    anchors_out = torch.empty(M, 3, device=device, dtype=dtype)

    # ========================================================================
    # 🔥 NEW FIX 9: Surface-only coverage seed
    # ========================================================================
    write_ptr = 0
    if ensure_cover:
        if coverage_only_surface and eligible_mask.any():
            # Coverage seed from surface anchors only
            cand = torch.nonzero(eligible_mask, as_tuple=False).squeeze(1)
            if M >= cand.numel():
                base_idx = cand
            else:
                vals = p[cand]
                sel = torch.topk(vals, k=M, largest=True).indices
                base_idx = cand[sel]
        else:
            # Original: all anchors
            if M >= N:
                base_idx = torch.arange(N, device=device, dtype=torch.long)
            else:
                base_idx = torch.topk(p, k=M, largest=True).indices

        base_count = int(base_idx.numel())
        if base_count > 0:
            xb = x[base_idx]
            nb = normalize(normals[base_idx])
            hb = spacing[base_idx]

            # tangent frame & jitter
            t1b, t2b = build_tangent_frame(nb, base_count, device, dtype)
            Ub, Vb = generate_tangent_jitter(base_count, generator, device, dtype)
            
            # 🔥 NEW FIX 8: One-sided thickness Z ∈ [0,1]
            if thickness_one_sided:
                Zb = torch.rand(base_count, 1, generator=generator, device=device, dtype=dtype)
            else:
                Zb = (torch.rand(base_count, 1, generator=generator, device=device, dtype=dtype) * 2.0 - 1.0)
            
            alpha_b = compute_adaptive_jitter_scale(hb, alpha, generator, device, dtype)

            tangent_b = alpha_b * hb.unsqueeze(-1) * (Ub * t1b + Vb * t2b)
            
            # 🔥 PATCH FIX 5: thickness_scales_with_spacing
            if thickness == 0.0:
                normal_b = (thickness_gamma * hb.unsqueeze(-1) * Zb) * nb
            else:
                normal_b = (thickness * Zb) * nb

            # micro jitter → tangent only
            if tangent_micro_only:
                Um, Vm = generate_tangent_jitter(base_count, generator, device, dtype)
                micro_b = micro_scale * alpha * hb.unsqueeze(-1) * (Um * t1b + Vm * t2b)
            else:
                micro_b = micro_scale * alpha * hb.unsqueeze(-1) * \
                          torch.randn(base_count, 3, generator=generator, device=device, dtype=dtype)

            pb = xb + tangent_b + normal_b + micro_b

            # 🔥 PATCH FIX 3: soft_plane_snap
            if plane_snap:
                delta = pb - xb
                proj = (delta * nb).sum(dim=1, keepdim=True) * nb
                delta_ortho = delta - proj
                pb = xb + delta_ortho + plane_snap_beta * proj + normal_b

            # 🔥 NEW FIX 8: Inside barrier (push back points with negative normal displacement)
            if inside_barrier_lambda > 0.0:
                delta = pb - xb
                delta_n = (delta * nb).sum(dim=1, keepdim=True)  # (base_count, 1)
                # If delta_n < 0 (moving inward), push back outward
                pb = pb + torch.relu(-delta_n) * (inside_barrier_lambda * nb)

            points_out[write_ptr:write_ptr+base_count]  = pb
            anchors_out[write_ptr:write_ptr+base_count] = xb
            normals_out[write_ptr:write_ptr+base_count] = nb
            write_ptr += base_count

            del xb, nb, hb, t1b, t2b, Ub, Vb, Zb, alpha_b, tangent_b, normal_b, micro_b, pb

    # ========================================================================
    # Phase B: Streaming Gumbel-ST with surface-masked top-k
    # ========================================================================
    remain = M - write_ptr
    if remain > 0:
        num_batches = (remain + gs_batch - 1) // gs_batch
        for bidx in range(num_batches):
            start = write_ptr + bidx * gs_batch
            end   = min(write_ptr + (bidx + 1) * gs_batch, M)
            b = end - start
            if b <= 0: break

            # Gumbel-softmax
            g = generate_gumbel_noise(b, N, generator, device)
            logits = (logp + g) / max(tau, 1e-6)
            y_soft_full = F.softmax(logits, dim=1)

            # 🔥 NEW FIX 10: Apply surface mask to top-k pool
            if mask_topk_with_surface and eligible_mask.any():
                # Mask out non-surface anchors before top-k
                y_soft_full = y_soft_full * eligible_mask_f.unsqueeze(0)
                y_soft_full = y_soft_full / (y_soft_full.sum(dim=1, keepdim=True) + EPS_SAFE)

            # 🔥 PATCH FIX 4: topk_candidate_pool
            if topk_pool > 0 and topk_pool < N:
                vals, inds = torch.topk(y_soft_full, k=topk_pool, dim=1)
                y_soft = torch.zeros_like(y_soft_full)
                y_soft.scatter_(1, inds, vals)
                y_soft = y_soft / (y_soft.sum(dim=1, keepdim=True) + EPS_SAFE)
            else:
                y_soft = y_soft_full

            idx = y_soft.argmax(dim=1)

            anchors_soft = y_soft @ x
            anchors_hard = x[idx]
            anchors_st   = anchors_hard - anchors_soft.detach() + anchors_soft

            n_soft = normalize(y_soft @ normals)
            n_hard = normals[idx]
            n_st   = normalize(n_hard - n_soft.detach() + n_soft)

            h_soft = (y_soft @ spacing.unsqueeze(1)).squeeze(1)
            h_hard = spacing[idx]
            h_st   = h_hard - h_soft.detach() + h_soft

            # tangent & micro jitter
            t1, t2 = build_tangent_frame(n_st, b, device, dtype)
            U, V = generate_tangent_jitter(b, generator, device, dtype)
            
            # 🔥 NEW FIX 8: One-sided thickness
            if thickness_one_sided:
                Z = torch.rand(b, 1, generator=generator, device=device, dtype=dtype)
            else:
                Z = (torch.rand(b, 1, generator=generator, device=device, dtype=dtype) * 2.0 - 1.0)
            
            alpha_b = compute_adaptive_jitter_scale(h_st, alpha, generator, device, dtype)

            tangent = alpha_b * h_st.unsqueeze(-1) * (U * t1 + V * t2)
            
            # 🔥 PATCH FIX 5: thickness_scales_with_spacing
            if thickness == 0.0:
                normal = (thickness_gamma * h_st.unsqueeze(-1) * Z) * n_st
            else:
                normal = (thickness * Z) * n_st

            if tangent_micro_only:
                Um, Vm = generate_tangent_jitter(b, generator, device, dtype)
                micro  = micro_scale * alpha * h_st.unsqueeze(-1) * (Um * t1 + Vm * t2)
            else:
                micro  = micro_scale * alpha * h_st.unsqueeze(-1) * \
                         torch.randn(b, 3, generator=generator, device=device, dtype=dtype)

            pb = anchors_st + tangent + normal + micro

            # 🔥 PATCH FIX 3: soft_plane_snap
            if plane_snap:
                delta = pb - anchors_st
                proj = (delta * n_st).sum(dim=1, keepdim=True) * n_st
                delta_ortho = delta - proj
                pb = anchors_st + delta_ortho + plane_snap_beta * proj + normal

            # 🔥 NEW FIX 8: Inside barrier
            if inside_barrier_lambda > 0.0:
                delta = pb - anchors_st
                delta_n = (delta * n_st).sum(dim=1, keepdim=True)  # (b, 1)
                pb = pb + torch.relu(-delta_n) * (inside_barrier_lambda * n_st)

            points_out[start:end]  = pb
            anchors_out[start:end] = anchors_st
            normals_out[start:end] = n_st

            # cleanup
            del g, logits, y_soft_full, y_soft, idx
            del anchors_soft, anchors_hard, anchors_st, n_soft, n_hard, n_st
            del h_soft, h_hard, h_st
            del t1, t2, U, V, Z, alpha_b, tangent, normal
            del micro, pb, delta, delta_n

    return points_out, normals_out, anchors_out

__all__ = [
    'sample_points',
    'gumbel_softmax_sample',
    'build_tangent_frame',
    'generate_tangent_jitter',
    'compute_adaptive_jitter_scale',
    'generate_gumbel_noise',
]