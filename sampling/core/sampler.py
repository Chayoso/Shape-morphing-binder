"""
Sampling with Gumbel-Softmax and tangent space jittering (memory-safe, coverage-safe).

Key changes vs. the dense-Y version:
- No dense Y ∈ ℝ^{M×N}. We stream small batches, compute soft weights y_soft on-the-fly,
  immediately project to anchors/normals/spacing and discard y_soft (low memory).
- Straight-through (ST) mixing per batch to keep gradients wrt probs:
      out = hard - soft.detach() + soft
- Optional "ensure_anchor_coverage": include every anchor at least once when M ≥ N
  so all particles can render at least one Gaussian.

Function signatures are unchanged.
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
# Main sampling (unchanged signature, new implementation)
# ============================================================================

# def sample_points(
#     x: torch.Tensor,
#     normals: torch.Tensor,
#     spacing: torch.Tensor,
#     probs: torch.Tensor,
#     cfg: Dict,
#     generator: Optional[torch.Generator] = None
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Memory-safe, coverage-safe point upsampling with tangent jitter.

#     Core idea:
#       - Stream small batches for Gumbel-Softmax; DO NOT build dense Y.
#       - Straight-through mixing keeps gradients wrt `probs`:
#             out = hard - soft.detach() + soft
#       - Optional 'ensure_anchor_coverage': include each anchor at least once
#         when M ≥ N, so every particle shows up in rendering.

#     Args:
#         x: (N, 3) anchor positions
#         normals: (N, 3) anchor normals (unit)
#         spacing: (N,) local spacing per anchor
#         probs: (N,) importance weights (non-negative; will be normalized)
#         cfg: dict with keys:
#              - 'M' (int): number of upsampled points
#              - 'tau' (float): Gumbel temperature
#              - 'alpha' (float): tangent jitter magnitude
#              - 'thickness' (float): normal offset magnitude
#              - 'gs_batch' (int, optional): streaming batch size for Gumbel (default 2048)
#              - 'ensure_anchor_coverage' (bool, optional): include each anchor once if M≥N (default True)
#              - 'micro_jitter_scale' (float, optional): extra HF jitter scale (default 0.2)
#         generator: torch.Generator for reproducibility

#     Returns:
#         points: (M, 3) upsampled points
#         normals_out: (M, 3) normals at those points (ST-interpolated)
#         anchors: (M, 3) underlying anchor positions (ST-interpolated)
#     """
#     device, dtype = x.device, x.dtype
#     N = x.shape[0]

#     # Parse config
#     M = int(cfg.get("M", 50000))
#     tau = float(cfg.get("tau", 0.2))
#     alpha = float(cfg.get("alpha", 0.35))
#     thickness = float(cfg.get("thickness", 0.0))
#     gs_batch = int(cfg.get("gs_batch", 2048))
#     ensure_cover = bool(cfg.get("ensure_anchor_coverage", True))
#     micro_scale = float(cfg.get("micro_jitter_scale", 0.2))

#     if generator is None:
#         generator = torch.Generator(device=device).manual_seed(0)

#     # Normalize probabilities to a valid categorical distribution
#     safe_probs = torch.clamp(probs, min=1e-12)
#     norm_probs = safe_probs / (safe_probs.sum() + EPS_SAFE)  # (N,)
#     logp = norm_probs.log().unsqueeze(0)  # (1, N) for broadcasting

#     # ----------------------------------------------------------------------
#     # Phase A: coverage seed (optional) - ensure every anchor appears once
#     # ----------------------------------------------------------------------
#     base_idx = []
#     if ensure_cover and M >= N:
#         # Include each anchor exactly once
#         base_idx = torch.arange(N, device=device, dtype=torch.long)
#     elif ensure_cover and M < N:
#         # If M < N, take top-M by probability to maximize coverage of likely anchors
#         base_idx = torch.topk(norm_probs, k=M, largest=True).indices
#     base_count = len(base_idx)

#     # Prepare output buffers
#     points_out = torch.empty(M, 3, device=device, dtype=dtype)
#     normals_out = torch.empty(M, 3, device=device, dtype=dtype)
#     anchors_out = torch.empty(M, 3, device=device, dtype=dtype)

#     # Fill coverage block (no grad wrt probs; still grad wrt x/normals/spacing)
#     write_ptr = 0
#     if base_count > 0:
#         xb = x[base_idx]
#         nb = normalize(normals[base_idx])
#         hb = spacing[base_idx]

#         # Tangent frame & jitter
#         t1b, t2b = build_tangent_frame(nb, base_count, device, dtype)
#         Ub, Vb = generate_tangent_jitter(base_count, generator, device, dtype)
#         Zb = (torch.rand(base_count, 1, generator=generator, device=device, dtype=dtype) * 2.0 - 1.0)
#         alpha_b = compute_adaptive_jitter_scale(hb, alpha, generator, device, dtype)

#         tangent_b = alpha_b * hb.unsqueeze(-1) * (Ub * t1b + Vb * t2b)
#         normal_b  = (thickness * Zb) * nb
#         micro_b   = micro_scale * alpha * hb.unsqueeze(-1) * torch.randn(base_count, 3, generator=generator, device=device, dtype=dtype)

#         points_out[write_ptr:write_ptr+base_count]  = xb + tangent_b + normal_b + micro_b
#         anchors_out[write_ptr:write_ptr+base_count] = xb
#         normals_out[write_ptr:write_ptr+base_count] = nb
#         write_ptr += base_count

#         # free temps
#         del xb, nb, hb, t1b, t2b, Ub, Vb, Zb, alpha_b, tangent_b, normal_b, micro_b

#     # ----------------------------------------------------------------------
#     # Phase B: streaming Gumbel-ST for the remaining samples
#     # ----------------------------------------------------------------------
#     remain = M - write_ptr
#     if remain > 0:
#         num_batches = (remain + gs_batch - 1) // gs_batch
#         for bidx in range(num_batches):
#             start = write_ptr + bidx * gs_batch
#             end   = min(write_ptr + (bidx + 1) * gs_batch, M)
#             b = end - start
#             if b <= 0:
#                 continue

#             # Gumbel-Softmax (streamed)
#             g = generate_gumbel_noise(b, N, generator, device)          # (b, N)
#             logits = (logp + g) / max(tau, 1e-6)                        # (b, N)
#             y_soft = F.softmax(logits, dim=1)                           # (b, N)
#             idx = y_soft.argmax(dim=1)                                  # (b,)

#             # ST interpolation for anchors / normals / spacing
#             anchors_soft = y_soft @ x                                   # (b, 3)
#             anchors_hard = x[idx]                                       # (b, 3)
#             anchors_st   = anchors_hard - anchors_soft.detach() + anchors_soft

#             n_soft = normalize(y_soft @ normals)                        # (b, 3)
#             n_hard = normals[idx]                                       # (b, 3)
#             n_st   = normalize(n_hard - n_soft.detach() + n_soft)

#             h_soft = (y_soft @ spacing.unsqueeze(1)).squeeze(1)         # (b,)
#             h_hard = spacing[idx]                                       # (b,)
#             h_st   = h_hard - h_soft.detach() + h_soft                  # (b,)

#             # Tangent frame & jitter for this batch
#             t1, t2 = build_tangent_frame(n_st, b, device, dtype)
#             U, V = generate_tangent_jitter(b, generator, device, dtype)
#             Z = (torch.rand(b, 1, generator=generator, device=device, dtype=dtype) * 2.0 - 1.0)
#             alpha_b = compute_adaptive_jitter_scale(h_st, alpha, generator, device, dtype)

#             tangent = alpha_b * h_st.unsqueeze(-1) * (U * t1 + V * t2)
#             normal  = (thickness * Z) * n_st
#             micro   = micro_scale * alpha * h_st.unsqueeze(-1) * torch.randn(b, 3, generator=generator, device=device, dtype=dtype)

#             points_out[start:end]  = anchors_st + tangent + normal + micro
#             anchors_out[start:end] = anchors_st
#             normals_out[start:end] = n_st

#             # aggressively free temporaries
#             del g, logits, y_soft, idx
#             del anchors_soft, anchors_hard, anchors_st
#             del n_soft, n_hard, n_st
#             del h_soft, h_hard, h_st
#             del t1, t2, U, V, Z, alpha_b, tangent, normal, micro

#     return points_out, normals_out, anchors_out

def sample_points(
    x: torch.Tensor,
    normals: torch.Tensor,
    spacing: torch.Tensor,
    probs: torch.Tensor,
    cfg: Dict,
    generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Memory-safe, coverage-safe upsampling with inside-suppression:
      - tangent-only micro jitter
      - optional plane snapping (kill unintended normal component)
      - coverage only for anchors with prob >= min_anchor_prob
    """
    device, dtype = x.device, x.dtype
    N = x.shape[0]

    # --- config ---
    M = int(cfg.get("M", 50000))
    tau = float(cfg.get("tau", 0.2))
    alpha = float(cfg.get("alpha", 0.35))
    thickness = float(cfg.get("thickness", 0.0))
    gs_batch = int(cfg.get("gs_batch", 2048))
    ensure_cover = bool(cfg.get("ensure_anchor_coverage", True))
    micro_scale = float(cfg.get("micro_jitter_scale", 0.2))

    # inside-suppression knobs
    min_anchor_prob = float(cfg.get("min_anchor_prob", 1e-4))
    tangent_micro_only = bool(cfg.get("tangent_micro_only", True))
    plane_snap = bool(cfg.get("plane_snap", True))

    if generator is None:
        generator = torch.Generator(device=device).manual_seed(0)

    # --- mask out near-zero anchors (likely interior/noisy) ---
    probs_mask = probs.clone()
    probs_mask[probs_mask < min_anchor_prob] = 0.0
    probs_sum = probs_mask.sum()
    # if all zero (extreme case), fall back to uniform
    if probs_sum <= 0:
        probs_mask = torch.ones_like(probs_mask)
        probs_sum = probs_mask.sum()
    norm_probs = probs_mask / (probs_sum + EPS_SAFE)
    logp = norm_probs.clamp_min(1e-12).log().unsqueeze(0)  # (1,N)

    # --- outputs ---
    points_out = torch.empty(M, 3, device=device, dtype=dtype)
    normals_out = torch.empty(M, 3, device=device, dtype=dtype)
    anchors_out = torch.empty(M, 3, device=device, dtype=dtype)

    # =============== Phase A: Coverage seed (filtered by min_anchor_prob) ===============
    write_ptr = 0
    if ensure_cover:
        valid_idx = torch.nonzero(probs_mask > 0, as_tuple=False).squeeze(-1)
        n_valid = int(valid_idx.numel())
        if M >= n_valid and n_valid > 0:
            base_idx = valid_idx
        elif n_valid > 0:
            # pick top-M within valid anchors
            base_idx = valid_idx[torch.topk(norm_probs[valid_idx], k=M, largest=True).indices]
        else:
            base_idx = torch.empty(0, dtype=torch.long, device=device)

        base_count = int(base_idx.numel())
        if base_count > 0:
            xb = x[base_idx]
            nb = normalize(normals[base_idx])
            hb = spacing[base_idx]

            # tangent frame & jitter
            t1b, t2b = build_tangent_frame(nb, base_count, device, dtype)
            Ub, Vb = generate_tangent_jitter(base_count, generator, device, dtype)
            Zb = (torch.rand(base_count, 1, generator=generator, device=device, dtype=dtype) * 2.0 - 1.0)
            alpha_b = compute_adaptive_jitter_scale(hb, alpha, generator, device, dtype)

            tangent_b = alpha_b * hb.unsqueeze(-1) * (Ub * t1b + Vb * t2b)
            normal_b  = (thickness * Zb) * nb

            # micro jitter → tangent only
            if tangent_micro_only:
                Um, Vm = generate_tangent_jitter(base_count, generator, device, dtype)
                micro_b = micro_scale * alpha * hb.unsqueeze(-1) * (Um * t1b + Vm * t2b)
            else:
                micro_b = micro_scale * alpha * hb.unsqueeze(-1) * \
                          torch.randn(base_count, 3, generator=generator, device=device, dtype=dtype)

            pb = xb + tangent_b + normal_b + micro_b

            # plane snap: allow only thickness component, remove normal component
            if plane_snap:
                delta = pb - xb
                proj = (delta * nb).sum(dim=1, keepdim=True) * nb
                delta_ortho = delta - proj        # orthogonal to normal direction
                pb = xb + delta_ortho + normal_b  # keep thick shell if any

            points_out[write_ptr:write_ptr+base_count]  = pb
            anchors_out[write_ptr:write_ptr+base_count] = xb
            normals_out[write_ptr:write_ptr+base_count] = nb
            write_ptr += base_count

            del xb, nb, hb, t1b, t2b, Ub, Vb, Zb, alpha_b, tangent_b, normal_b, micro_b, pb

    # ================== Phase B: Streaming Gumbel-ST for the rest ==================
    remain = M - write_ptr
    if remain > 0:
        num_batches = (remain + gs_batch - 1) // gs_batch
        for bidx in range(num_batches):
            start = write_ptr + bidx * gs_batch
            end   = min(write_ptr + (bidx + 1) * gs_batch, M)
            b = end - start
            if b <= 0: break

            # Gumbel-softmax over filtered distribution
            g = generate_gumbel_noise(b, N, generator, device)
            logits = (logp + g) / max(tau, 1e-6)
            y_soft = F.softmax(logits, dim=1)
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

            # tangent & micro jitter (tangent-only 옵션)
            t1, t2 = build_tangent_frame(n_st, b, device, dtype)
            U, V = generate_tangent_jitter(b, generator, device, dtype)
            Z = (torch.rand(b, 1, generator=generator, device=device, dtype=dtype) * 2.0 - 1.0)
            alpha_b = compute_adaptive_jitter_scale(h_st, alpha, generator, device, dtype)

            tangent = alpha_b * h_st.unsqueeze(-1) * (U * t1 + V * t2)
            normal  = (thickness * Z) * n_st

            if tangent_micro_only:
                Um, Vm = generate_tangent_jitter(b, generator, device, dtype)
                micro  = micro_scale * alpha * h_st.unsqueeze(-1) * (Um * t1 + Vm * t2)
            else:
                micro  = micro_scale * alpha * h_st.unsqueeze(-1) * \
                         torch.randn(b, 3, generator=generator, device=device, dtype=dtype)

            pb = anchors_st + tangent + normal + micro

            if plane_snap:
                delta = pb - anchors_st
                proj = (delta * n_st).sum(dim=1, keepdim=True) * n_st
                delta_ortho = delta - proj
                pb = anchors_st + delta_ortho + normal  # NEW tensor, not in-place

            points_out[start:end]  = pb
            anchors_out[start:end] = anchors_st
            normals_out[start:end] = n_st

            # cleanup
            del g, logits, y_soft, idx
            del anchors_soft, anchors_hard, anchors_st, n_soft, n_hard, n_st
            del h_soft, h_hard, h_st
            del t1, t2, U, V, Z, alpha_b, tangent, normal
            del micro, pb
            if plane_snap:
                del delta

    return points_out, normals_out, anchors_out



__all__ = [
    'sample_points',
    'gumbel_softmax_sample',
    'build_tangent_frame',
    'generate_tangent_jitter',
    'compute_adaptive_jitter_scale',
    'generate_gumbel_noise',
]
