"""
Fast Local-Soft ST Sampler (Safe + Optimized)

Key improvements:
1. NaN/Inf prevention + fp32 normalization
2. Unique center ratio enforcement → unique anchors↑
3. Soft-NMS suppression (optional) → diversity↑, over-concentration↓
4. Pre-compute KNN graph + tangent basis (once)
5. bmm vectorization + AMP

Memory: O(B·S)

Author: CHAYO
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


# =========================================================================
# Safe probability normalization (NaN/Inf removal)
# =========================================================================
def normalize_prob(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Safe probability normalization: NaN/Inf removal + sum=1
    """
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    x = x.clamp(min=0.0)
    s = x.sum()
    if (not torch.isfinite(s)) or (s <= eps):
        return torch.full_like(x, 1.0 / x.numel())
    return x / s


# =========================================================================
# π debug breakdown (5-line key metrics)
# =========================================================================
@torch.no_grad()
def pi_breakdown(p: torch.Tensor, p_surf_raw: torch.Tensor, 
                 p_target: float = 0.93, gate_k: float = 60.0, 
                 tag: str = "") -> float:
    """
    π stage-by-stage diagnostics: sum, min/max, surf_mass, effective support size
    
    Returns:
        surf_mass (float)
    """
    with torch.amp.autocast("cuda", enabled=False):
        tau = torch.quantile(p_surf_raw.float(), 1.0 - p_target)
        gate = torch.sigmoid(gate_k * (p_surf_raw.float() - tau))
        surf = float((p * gate).sum())  # Surface mass
        
        print(f"[{tag}] sum={float(p.sum()):.6f}, min={float(p.min()):.3e}, "
              f"max={float(p.max()):.3e}, surf_mass={surf:.3f}, finite={bool(torch.isfinite(p).all())}")
        
        # Effective support size (entropy), top mass
        val, _ = torch.sort(p, descending=True)
        top1 = float(val[0])
        H = - (p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum()
        eff = float(torch.exp(H))  # effective support size
        print(f"   top1={top1:.3e}, eff_size≈{eff:.0f} / N={p.numel()}")
        
        return surf


# =========================================================================
# 0) Tangent basis / KNN cache
# =========================================================================
@torch.no_grad()
def precompute_tangent_bases(normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute tangent bases: t1, t2 perpendicular to normals."""
    absn = normals.abs()
    min_axis = absn.argmin(dim=-1)
    ref = torch.zeros_like(normals)
    ref[torch.arange(normals.size(0), device=normals.device), min_axis] = 1.0
    t1 = F.normalize(torch.cross(normals, ref, dim=-1), dim=-1)
    t2 = F.normalize(torch.cross(normals, t1, dim=-1), dim=-1)
    return t1, t2


@torch.no_grad()
def precompute_knn_indices(anchors: torch.Tensor, S: int = 12, exclude_self: bool = True) -> torch.Tensor:
    """
    Precompute KNN graph: (N, S)
    
    Args:
        exclude_self: If True, return S+1 neighbors and exclude self (index 0)
    """
    k = S + 1 if exclude_self else S
    try:
        import faiss  # type: ignore
        xb = anchors.detach().float().cpu().numpy()
        index = faiss.IndexFlatL2(3)
        index.add(xb)
        _, I = index.search(xb, k)
        idx = torch.from_numpy(I).to(anchors.device)
    except Exception:
        d = torch.cdist(anchors, anchors)
        idx = torch.topk(-d, k=k, dim=1).indices
    
    # Exclude self (remove first column)
    if exclude_self:
        idx = idx[:, 1:]  # (N, S) - exclude self
    
    return idx


# =========================================================================
# 1) π generation pipeline (4 stages)
# =========================================================================
def build_pi_base(
    p_surf_raw: torch.Tensor,
    w_den: torch.Tensor,
    *,
    alpha: float = 2.1,
    beta: float = 0.7
) -> torch.Tensor:
    """STAGE1: Base π distribution (p_surf^α × w_den^β)"""
    with torch.amp.autocast("cuda", enabled=False):
        pi_base = (p_surf_raw.clamp(min=1e-8).float().pow(alpha)) * (w_den.clamp(min=1e-6).float().pow(beta))
        pi = normalize_prob(pi_base)
    return pi


@torch.no_grad()
def build_pi_complete(
    p_surf_raw: torch.Tensor,
    w_den: torch.Tensor,
    anchors: torch.Tensor,
    nn_idx_all: torch.Tensor,
    cfg: dict,
    debug: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Complete π generation (STAGE1-4).
    
    Returns:
        pi_cache: (N,) Final distribution
        log_pi_cache: (N,) Log of pi_cache
    """
    alpha = float(cfg.get("alpha", 2.1))
    beta = float(cfg.get("beta", 0.7))
    p_target = float(cfg.get("p_target", 0.85))
    gate_k = float(cfg.get("gate_k", 90.0))
    gamma_flat = float(cfg.get("gamma_flat", 0.70))
    keep_ratio = float(cfg.get("keep_ratio", 0.65))
    beta_soft = float(cfg.get("beta_soft", 0.95))
    eps_mix = float(cfg.get("eps_mix", 0.002))
    cap_mult = float(cfg.get("cap_mult", 18.0))
    use_soft_nms = bool(cfg.get("use_soft_nms", False))
    suppress_lambda = float(cfg.get("suppress_lambda", 0.15))
    
    with torch.amp.autocast("cuda", enabled=False):
        # STAGE1: Base π
        pi = build_pi_base(p_surf_raw, w_den, alpha=alpha, beta=beta)
        if debug:
            pi_breakdown(pi, p_surf_raw, p_target, gate_k, tag="STAGE1_base")
        
        # STAGE2: Flatten
        if gamma_flat != 1.0:
            pi = normalize_prob(pi.pow(gamma_flat))
            if debug:
                pi_breakdown(pi, p_surf_raw, p_target, gate_k, tag="STAGE2_flatten")
        
        # STAGE3: Soft-NMS (optional, default OFF)
        if use_soft_nms:
            pi = soft_nms_pi(pi, anchors, nn_idx_all, suppress_lambda=suppress_lambda)
            if debug:
                pi_breakdown(pi, p_surf_raw, p_target, gate_k, tag="STAGE3_after_nms")
        else:
            if debug:
                print(f"[STAGE3] NMS OFF (Soft-cut instead)")
        
        # STAGE4: Soft-cut rescaling + ε-mix + Global Cap
        # 4-1) Compute gate
        tau = torch.quantile(p_surf_raw.float(), 1.0 - p_target)
        gate = torch.sigmoid(gate_k * (p_surf_raw.float() - tau))
        
        # 4-2) Soft-cut + rescale
        g_thr = torch.quantile(gate, 1.0 - keep_ratio)
        mask_soft = ((gate - g_thr).clamp_min(0) / (1 - g_thr + 1e-8))
        mask_soft = mask_soft.pow(beta_soft)
        
        # 4-3) Two-bucket reweighting (precise surf_mass control)
        surf_mass_target = float(cfg.get("surf_mass_target", 0.88))  # Target: 0.85~0.90
        
        # Surface/rest distributions (independent normalization)
        pi_surf = normalize_prob((pi * mask_soft).clamp_min(0.0))
        pi_rest = normalize_prob((pi * (1.0 - mask_soft)).clamp_min(1e-12))
        
        # Mix by target ratio
        pi = normalize_prob(surf_mass_target * pi_surf + (1.0 - surf_mass_target) * pi_rest)
        
        # ε-mix (exploration mass)
        pi = normalize_prob((1 - eps_mix) * pi + eps_mix / pi.numel())
        
        # 4-4) Global cap
        p_cap = cap_mult / pi.numel()
        pi = normalize_prob(torch.minimum(pi, torch.full_like(pi, p_cap)))
        
        # Debug output
        if debug:
            surf_actual = float((pi * mask_soft).sum())
            surf_gate = pi_breakdown(pi, p_surf_raw, p_target, gate_k, tag="STAGE4_final")
            print(f"  Soft-cut: keep_ratio={keep_ratio:.2f}, beta_soft={beta_soft:.2f}")
            print(f"  Peak upper: top1={float(pi.max()):.3e} vs cap={p_cap:.3e}")
            print(f"  surf_mass: actual(mask_soft)={surf_actual:.3f}, gate={surf_gate:.3f}")
            print(f"  ✅ Target: surf_mass_target={surf_mass_target:.3f} (0.85~0.90)")
            print()
        
        # Simplex constraint verification
        if not (torch.isfinite(pi).all() and (pi >= 0).all() and abs(float(pi.sum()) - 1.0) < 1e-6):
            print(f"[WARN] Simplex constraint violation → automatic recovery")
            pi = normalize_prob(pi)
        
        # Store mask_soft for consistent surf_mass calculation
        cfg["_mask_soft"] = mask_soft.detach()
        
        # Final cache
        pi_cache = pi
        log_pi_cache = torch.log(pi_cache.clamp_min(1e-12))
    
    return pi_cache, log_pi_cache


# =========================================================================
# 2) Soft-NMS suppression (optional)
# =========================================================================
@torch.no_grad()
def soft_nms_pi(
    pi: torch.Tensor,
    anchors: torch.Tensor,
    nn_idx_all: torch.Tensor,
    suppress_lambda: float = 0.6,
    r_scale: float = 0.9,
    meanNN: Optional[float] = None
) -> torch.Tensor:
    """
    Soft-NMS: Gradually suppress nearby masses → Diversity↑
    
    Args:
        pi: (N,) Distribution
        anchors: (N, 3)
        nn_idx_all: (N, S) KNN indices
        suppress_lambda: Suppression strength (0~1)
        r_scale: Radius scale
        meanNN: Mean NN distance (computed if None)
        
    Returns:
        pi_suppressed: (N,)
    """
    if meanNN is None:
        idx = torch.randint(0, anchors.size(0), (min(4096, anchors.size(0)),), device=anchors.device)
        d = torch.cdist(anchors[idx], anchors[idx])
        d.fill_diagonal_(1e9)
        meanNN = float(d.min(dim=1).values.median().item())
    
    r = r_scale * meanNN
    
    # Gradually suppress nearby masses
    neigh = anchors[nn_idx_all]  # (N, S, 3)
    center = anchors[:, None, :]
    d2 = (neigh - center).pow(2).sum(-1)  # (N, S)
    w = torch.exp(-d2 / (r * r))
    supp = (w * pi[nn_idx_all]).sum(dim=1)  # (N,)
    
    # 🔥 Negative prevention + safe normalization (suppress proportional to own mass)
    # Old: pi - λ*supp (absolute suppression) → Problem: chain suppression leaves few
    # New: pi * (1 - λ*supp/pi) (relative suppression) → Prevent convergence to 0
    decay = torch.exp(-suppress_lambda * supp / (pi + 1e-12))
    pi_supp = pi * decay
    pi = normalize_prob(pi_supp.clamp(min=1e-12))
    
    return pi


# =========================================================================
# 3) Voxel hash-based diversity guarantee (vectorized batch-taboo replacement)
# =========================================================================
@torch.no_grad()
def sample_centers_with_voxel_quota(
    pi: torch.Tensor,
    anchors: torch.Tensor,
    batch_size: int,
    meanNN: float,
    voxel_scale: float = 0.9,
    bucket_quota: Optional[Dict[int, int]] = None,
    cat: Optional[torch.distributions.Categorical] = None
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """
    Voxel hash for spatial bucket-quota diversity (fully vectorized)
    
    Args:
        pi: (N,) Sampling distribution
        anchors: (N, 3) Anchor positions
        batch_size: Number of centers to sample
        meanNN: Mean nearest neighbor distance
        voxel_scale: Voxel size = voxel_scale * meanNN
        bucket_quota: Pre-computed quota dict (if None, compute now)
        cat: Pre-built Categorical sampler
        
    Returns:
        centers: (B,) Center indices
        bucket_quota: Updated quota dict (cache for next call)
    """
    device = pi.device
    N = pi.numel()
    voxel_size = voxel_scale * meanNN
    
    # Voxel hash key calculation (spatial hash)
    vox = (anchors / voxel_size).floor().long()  # (N, 3)
    key = (vox[:, 0] * 73856093 ^ vox[:, 1] * 19349663 ^ vox[:, 2] * 83492791) & 0x7fffffff  # (N,)
    
    # Bucket-wise quota calculation (once, then cache)
    if bucket_quota is None:
        idx_sort = torch.argsort(key)
        key_sorted = key[idx_sort]
        pi_sorted = pi[idx_sort]
        
        # Find bucket boundaries
        unique_keys = key_sorted.unique()
        bucket_mass = {}
        
        for k in unique_keys.tolist():
            mask = (key_sorted == k)
            bucket_mass[k] = float(pi_sorted[mask].sum())
        
        tot = sum(bucket_mass.values())
        bucket_quota = {k: max(1, int(round(bucket_mass[k] / tot * batch_size)))
                       for k in bucket_mass}
    
    # Initial sampling (with replacement)
    if cat is None:
        cat = torch.distributions.Categorical(probs=pi)
    centers = cat.sample((batch_size,))  # (B,)
    
    # Bucket-wise excess check and re-sample
    key_b = key[centers]  # (B,)
    unique_keys, counts_per_key = key_b.unique(return_counts=True)
    
    over_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
    
    for k, c in zip(unique_keys.tolist(), counts_per_key.tolist()):
        quota_k = bucket_quota.get(k, 1)
        if c > quota_k:
            # Find position in corresponding bucket
            pos = (key_b == k).nonzero(as_tuple=True)[0]
            # Mask only excess
            over = pos[quota_k:]
            over_mask[over] = True
    
    # Re-sample excess (exclude only over-quota buckets, not all selected)
    if over_mask.any():
        over_idx = torch.nonzero(over_mask, as_tuple=False).squeeze(1)  # (n_over,)
        over_keys = key_b[over_idx]  # (n_over,) Keys of over-quota slots
        
        # Exclude only over-quota buckets (vectorized)
        pi_masked = pi.clone()
        pi_masked[torch.isin(key, over_keys)] = 0.0  # Zero out over-quota buckets
        
        s = pi_masked.sum()
        if s > 1e-12:
            pi_masked = pi_masked / s
            centers[over_idx] = torch.multinomial(pi_masked, num_samples=over_idx.numel(), replacement=True)
        # else: keep original (fallback)
    
    return centers, bucket_quota


# =========================================================================
# 4) Fast Local-Soft (optimized: vectorized + cached)
# =========================================================================
def sample_points_fast(
    anchors: torch.Tensor,
    normals: torch.Tensor,
    p_surf_raw: torch.Tensor,
    w_den: torch.Tensor,
    pi: torch.Tensor,
    log_pi: torch.Tensor,  # Cached log(pi)
    *,
    M: int,
    nn_idx_all: torch.Tensor,
    t1_all: torch.Tensor,
    t2_all: torch.Tensor,
    S: int = 16,
    tau_local: float = 0.50,
    batch_size: int = 32768,
    jitter_coef: float = 0.22,
    use_amp: bool = True,
    use_st: bool = False,  # Use ST (hard-soft mixing)
    # Light mode parameters
    u_local: float = 0.20,  # Uniform mixing (0.15~0.30)
    w_cap: Optional[float] = None,  # Local weight cap (0.10~0.30)
    # Dynamic bias suppression
    bias_target: float = 3.0,  # Target bias
    use_dynamic_cap: bool = True,  # Dynamic cap mask
    center_tau: float = 3.5,  # Center flattening
    # Voxel diversity
    use_voxel_diversity: bool = True,
    voxel_scale: float = 0.9,
    # Cache (injected from outside)
    meanNN_cache: Optional[float] = None,
    bucket_quota_cache: Optional[Dict[int, int]] = None,
    cat_cache: Optional[torch.distributions.Categorical] = None,
    rng: Optional[torch.Generator] = None
) -> Dict[str, torch.Tensor]:
    """
    Fast Local-Soft sampler (optimized: vectorized + cached).
    
    Args:
        log_pi: Pre-computed log(pi) cache
        use_st: Use Straight-Through (hard-soft mixing). False → faster
        use_voxel_diversity: Use voxel-hashing for spatial diversity
        voxel_scale: Voxel size = voxel_scale * meanNN
        meanNN_cache: Cached mean NN distance
        bucket_quota_cache: Cached voxel bucket quotas
        cat_cache: Cached Categorical sampler
    """
    device, dtype, N = anchors.device, anchors.dtype, anchors.size(0)
    if rng is None:
        rng = torch.Generator(device=device).manual_seed(2025)
    
    # Mean NN cache (computed once per episode)
    if meanNN_cache is None:
        with torch.no_grad():
            idx_samp = torch.randint(0, N, (min(4096, N),), device=device)
            d = torch.cdist(anchors[idx_samp], anchors[idx_samp])
            d.fill_diagonal_(1e9)
            meanNN = d.min(dim=1).values.median().item()
    else:
        meanNN = meanNN_cache
    
    j0 = jitter_coef * float(meanNN)
    
    # Output buffers
    mu = torch.empty(M, 3, device=device, dtype=dtype)
    nrm = torch.empty(M, 3, device=device, dtype=dtype)
    pup = torch.empty(M, device=device, dtype=dtype)
    parent_idx_out = torch.empty(M, device=device, dtype=torch.long)  # Hard parent indices
    counts = torch.zeros(N, device=device, dtype=dtype)
    
    # Sampler for global centers with global flattening
    cat_global = torch.distributions.Categorical(logits=log_pi / max(1e-6, center_tau))
    
    # Voxel quota cache (once per episode)
    bucket_quota = bucket_quota_cache
    
    num_batches, write = (M + batch_size - 1) // batch_size, 0
    
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if (use_amp and device.type == "cuda")
        else torch.amp.autocast("cuda", enabled=False)
    )
    
    with amp_ctx:
        for b_idx in range(num_batches):
            B = min(batch_size, M - write)
            
            # --- Dynamic per-anchor cap (uses parent counts) ---
            with torch.no_grad():
                if use_dynamic_cap and write > 0:
                    mean_sofar = write / max(1, N)
                    cap = torch.ceil(torch.tensor(bias_target * max(1.0, float(mean_sofar)), device=pi.device))
                    alive_mask = (counts < cap)  # [N] bool
                    masked_logits = (log_pi / max(1e-6, center_tau)).clone()
                    masked_logits = torch.where(alive_mask, masked_logits, masked_logits.new_full(masked_logits.shape, -1e9))
                    cat_centers = torch.distributions.Categorical(logits=masked_logits)
                else:
                    cat_centers = cat_global
            
            # --- 4.1 Sample parent centers (enforce diversity per voxel) ---
            if use_voxel_diversity:
                centers, bucket_quota = sample_centers_with_voxel_quota(
                    pi, anchors, B, meanNN, voxel_scale, bucket_quota, cat_centers
                )
            else:
                centers = cat_centers.sample((B,))
            
            parent_idx_out[write:write+B] = centers
            
            # --- 4.2 Local S-neighbor gather + Soft mixing ---
            nn_idx = nn_idx_all[centers]  # (B, S)
            A = anchors[nn_idx]
            Nr = normals[nn_idx]
            Ps = p_surf_raw[nn_idx]
            
            # Local logits
            logpi_local = log_pi[nn_idx]  # (B, S)
            
            # Local softmax with temperature
            tau_adjusted = max(tau_local, 1.15)
            W_soft = F.softmax(logpi_local / tau_adjusted, dim=1)
            
            # Uniform mixing
            W_soft = (1.0 - u_local) * W_soft + (u_local / S)
            
            # Local weight cap
            if w_cap is not None:
                W_soft = torch.minimum(W_soft, W_soft.new_full(W_soft.shape, w_cap))
            
            # Normalize
            W = W_soft / (W_soft.sum(dim=1, keepdim=True) + 1e-12)
            
            # --- Local barycentric interpolation (μ_b, n_b) ---
            mu_b = torch.bmm(W.unsqueeze(1), A).squeeze(1)  # [B,3]
            n_b = F.normalize(torch.bmm(W.unsqueeze(1), Nr).squeeze(1), dim=-1)
            wloc = (W * w_den[nn_idx]).sum(dim=1)  # [B]
            p_upb = (W * Ps).sum(dim=1)  # [B]
            
            # --- Jitter in tangent plane, stronger in sparser zones ---
            t1_b = t1_all[centers]
            t2_b = t2_all[centers]
            u = torch.randn(B, 1, generator=rng, device=device, dtype=dtype)
            v = torch.randn(B, 1, generator=rng, device=device, dtype=dtype)
            j = j0 * (0.7 * p_upb + 0.3 * wloc)
            mu_b = mu_b + j.unsqueeze(-1) * (u * t1_b + v * t2_b)
            
            # --- Write batch outputs ---
            mu[write:write+B] = mu_b
            nrm[write:write+B] = n_b
            pup[write:write+B] = p_upb
            
            # --- HARD parent-anchor counts (monotone selection preserved) ---
            counts.index_add_(0, centers, torch.ones_like(centers, dtype=dtype))
            
            write += B
    
    return {
        "mu": mu,
        "n": nrm,
        "p_up": pup,
        "parent_idx": parent_idx_out,  # Hard parent indices per point
        "counts": counts,  # HARD counts per anchor (integer in float)
        "meanNN": meanNN,
        "bucket_quota": bucket_quota,
    }


# =========================================================================
# Wrapper: sample_points (pipeline compatibility)
# =========================================================================
def sample_points(
    x: torch.Tensor,
    normals: torch.Tensor,
    spacing: torch.Tensor,
    probs: Optional[torch.Tensor],
    cfg: dict,
    generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast local-soft sampler wrapper with parent-anchor tracking.
    
    Args:
        x: (N, 3) Anchor positions
        normals: (N, 3) Surface normals
        spacing: (N,) Local spacing
        probs: (N,) Initial π (optional, recomputed if None)
        cfg: Config dict
        generator: RNG
        
    Returns:
        points, normals_out, anchors, anchor_selection_count
    """
    device = x.device
    N = x.shape[0]
    
    # Extract config
    M = int(cfg.get("M", 50000))
    S = int(cfg.get("local_S", 16))
    tau_local = float(cfg.get("tau_local", 1.40))
    local_batch = int(cfg.get("local_batch", 32768))
    jitter_coef = float(cfg.get("micro_jitter_j0", 0.22))
    use_amp = bool(cfg.get("use_amp", True))
    use_st = bool(cfg.get("use_st", False))
    use_voxel_diversity = bool(cfg.get("use_voxel_diversity", True))
    voxel_scale = float(cfg.get("voxel_scale", 0.50))
    debug_pi = bool(cfg.get("debug", {}).get("pi_breakdown", False))
    
    p_surf_raw = cfg.get("p_surf_raw")
    w_den = cfg.get("w_density")
    
    # Precompute KNN & tangents
    nn_idx_all = precompute_knn_indices(x, S=S, exclude_self=True)
    t1_all, t2_all = precompute_tangent_bases(normals)
    
    # Build or use π
    if probs is not None:
        pi_cache = normalize_prob(probs.detach().float()).to(x.dtype)
        log_pi_cache = torch.log(pi_cache.clamp_min(1e-12))
        if debug_pi:
            pi_breakdown(pi_cache, p_surf_raw, cfg.get("p_target", 0.85), cfg.get("gate_k", 120.0), tag="used_pi (external)")
    else:
        if p_surf_raw is None or w_den is None:
            raise ValueError("cfg must contain 'p_surf_raw' and 'w_density' when probs is None")
        pi_cache, log_pi_cache = build_pi_complete(
            p_surf_raw, w_den, x, nn_idx_all, cfg, debug=debug_pi
        )
    
    # Fast sampler with HARD parent counts
    center_tau = float(cfg.get("center_tau", 6.0))
    result = sample_points_fast(
        anchors=x,
        normals=normals,
        p_surf_raw=p_surf_raw,
        w_den=w_den,
        pi=pi_cache,
        log_pi=log_pi_cache,
        M=M,
        nn_idx_all=nn_idx_all,
        t1_all=t1_all,
        t2_all=t2_all,
        S=S,
        tau_local=tau_local,
        batch_size=local_batch,
        jitter_coef=jitter_coef,
        use_amp=use_amp,
        use_st=use_st,
        u_local=float(cfg.get("u_local", 0.68)),
        w_cap=float(cfg.get("w_cap", 0.10)) if cfg.get("w_cap") is not None else None,
        bias_target=float(cfg.get("bias_target", 2.5)),
        use_dynamic_cap=bool(cfg.get("use_dynamic_cap", True)),
        center_tau=center_tau,
        use_voxel_diversity=use_voxel_diversity,
        voxel_scale=voxel_scale,
        meanNN_cache=cfg.get("_cache", {}).get("meanNN", None),
        bucket_quota_cache=cfg.get("_cache", {}).get("bucket_quota", None),
        cat_cache=None,  # Will be created inside
        rng=generator,
    )
    
    # Parent positions for diagnostics / stage3 export
    parent_idx = result["parent_idx"]
    anchors_selected = x.index_select(0, parent_idx)
    
    # Diagnostics with correlations
    with torch.no_grad():
        sc = result["counts"]  # HARD parent counts
        
        # Correlation: spacing vs selection_count
        def corr(a, b):
            a = (a - a.mean()) / (a.std() + 1e-6)
            b = (b - b.mean()) / (b.std() + 1e-6)
            return float((a*b).mean().item())
        
        corr_spacing_count = corr(spacing, sc)
        
        # Basic stats
        mean_c = float(sc.mean())
        max_c = float(sc.max())
        
        if debug_pi:
            print(f"[Selection Stats] mean={mean_c:.2f}, max={max_c:.2f}, bias={max_c/(mean_c+1e-6):.2f}")
            print(f"[Correlation] corr(spacing, sel_count) = {corr_spacing_count:+.3f}")
    
    # Pass back caches for next call
    if "_cache" not in cfg:
        cfg["_cache"] = {}
    cfg["_cache"]["meanNN"] = result["meanNN"]
    cfg["_cache"]["bucket_quota"] = result["bucket_quota"]
    cfg["p_up"] = result["p_up"]
    
    # Store diagnostics with correlation
    if "diagnostics" not in cfg:
        cfg["diagnostics"] = {}
    cfg["diagnostics"]["corr_spacing_count"] = corr_spacing_count
    
    return result["mu"], result["n"], anchors_selected, result["counts"]


__all__ = [
    'sample_points',
    'sample_points_fast',
    'precompute_knn_indices',
    'precompute_tangent_bases',
    'build_pi_base',
    'build_pi_complete',
    'soft_nms_pi',
    'normalize_prob',
    'pi_breakdown',
    'sample_centers_with_voxel_quota',
]
