"""
Fast Local-Soft ST Sampler (Safe + Optimized, +uniform_pi, +auto_normalize)

Key improvements:
1. NaN/Inf prevention + fp32 normalization
2. Unique center ratio enforcement → unique anchors↑
3. Soft-NMS suppression (optional) → diversity↑, over-concentration↓
4. Pre-compute KNN graph + tangent basis (once)
5. bmm vectorization + AMP
6. NEW: uniform_pi option → 균일 분포로 대량 리샘플
7. CRITICAL: Auto-normalize p_surf_raw → jitter explosion 방지

Memory: O(B·S)

Author: CHAYO (patched v2.2)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

# ⚡ Fast approximate quantile
from sampling.utils.quantile import approx_quantile

# =========================================================================
# Safe probability normalization (NaN/Inf removal)
# =========================================================================
def normalize_prob(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
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
    with torch.amp.autocast("cuda", enabled=False):
        tau  = approx_quantile(p_surf_raw.float(), 1.0 - p_target, n_bins=(1<<14))
        gate = torch.sigmoid(gate_k * (p_surf_raw.float() - tau))
        surf = float((p * gate).sum())
        print(f"[{tag}] sum={float(p.sum()):.6f}, min={float(p.min()):.3e}, "
              f"max={float(p.max()):.3e}, surf_mass={surf:.3f}, finite={bool(torch.isfinite(p).all())}")
        val, _ = torch.sort(p, descending=True)
        top1 = float(val[0])
        H = - (p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum()
        eff = float(torch.exp(H))
        print(f"   top1={top1:.3e}, eff_size≈{eff:.0f} / N={p.numel()}")
        return surf

# =========================================================================
# 0) Tangent basis / KNN cache
# =========================================================================
@torch.no_grad()
def precompute_tangent_bases(normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    absn = normals.abs()
    min_axis = absn.argmin(dim=-1)
    ref = torch.zeros_like(normals)
    ref[torch.arange(normals.size(0), device=normals.device), min_axis] = 1.0
    t1 = F.normalize(torch.cross(normals, ref, dim=-1), dim=-1)
    t2 = F.normalize(torch.cross(normals, t1, dim=-1), dim=-1)
    return t1, t2

@torch.no_grad()
def precompute_knn_indices(anchors: torch.Tensor, S: int = 12, exclude_self: bool = True) -> torch.Tensor:
    k = S + 1 if exclude_self else S
    try:
        import faiss  # type: ignore
        xb = anchors.detach().float().cpu().numpy()
        index = faiss.IndexFlatL2(3); index.add(xb)
        _, I = index.search(xb, k)
        idx = torch.from_numpy(I).to(anchors.device)
    except Exception:
        d = torch.cdist(anchors, anchors)
        idx = torch.topk(-d, k=k, dim=1).indices
    if exclude_self:
        idx = idx[:, 1:]
    return idx

# =========================================================================
# 1) π generation pipeline (uniform_pi 지원)
# =========================================================================
def build_pi_base(p_surf_raw: torch.Tensor, w_den: torch.Tensor, *, alpha: float = 2.1, beta: float = 0.7) -> torch.Tensor:
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
    # ----- NEW: 균일 모드 -----
    if bool(cfg.get("uniform_pi", True)):
        N = anchors.shape[0]
        pi = torch.full((N,), 1.0 / max(N, 1), device=anchors.device, dtype=anchors.dtype)
        return pi, torch.log(pi.clamp_min(1e-12))

    alpha = float(cfg.get("alpha", 2.1))
    beta  = float(cfg.get("beta", 0.7))
    p_target   = float(cfg.get("p_target", 0.85))
    gate_k     = float(cfg.get("gate_k", 90.0))
    gamma_flat = float(cfg.get("gamma_flat", 0.70))
    keep_ratio = float(cfg.get("keep_ratio", 0.65))
    beta_soft  = float(cfg.get("beta_soft", 0.95))
    eps_mix    = float(cfg.get("eps_mix", 0.002))
    cap_mult   = float(cfg.get("cap_mult", 18.0))
    use_soft_nms     = bool(cfg.get("use_soft_nms", False))
    suppress_lambda  = float(cfg.get("suppress_lambda", 0.15))

    with torch.amp.autocast("cuda", enabled=False):
        pi = build_pi_base(p_surf_raw, w_den, alpha=alpha, beta=beta)
        if debug: pi_breakdown(pi, p_surf_raw, p_target, gate_k, tag="STAGE1_base")

        if gamma_flat != 1.0:
            pi = normalize_prob(pi.pow(gamma_flat))
            if debug: pi_breakdown(pi, p_surf_raw, p_target, gate_k, tag="STAGE2_flatten")

        if use_soft_nms:
            pi = soft_nms_pi(pi, anchors, nn_idx_all, suppress_lambda=suppress_lambda)
            if debug: pi_breakdown(pi, p_surf_raw, p_target, gate_k, tag="STAGE3_after_nms")

        tau = approx_quantile(p_surf_raw.float(), 1.0 - p_target, n_bins=(1<<14))
        gate = torch.sigmoid(gate_k * (p_surf_raw.float() - tau))
        g_thr = approx_quantile(gate, 1.0 - keep_ratio, n_bins=(1<<14))
        mask_soft = ((gate - g_thr).clamp_min(0) / (1 - g_thr + 1e-8)).pow(beta_soft)

        surf_mass_target = float(cfg.get("surf_mass_target", 0.88))
        pi_surf = normalize_prob((pi * mask_soft).clamp_min(0.0))
        pi_rest = normalize_prob((pi * (1.0 - mask_soft)).clamp_min(1e-12))
        pi = normalize_prob(surf_mass_target * pi_surf + (1.0 - surf_mass_target) * pi_rest)

        pi = normalize_prob((1 - eps_mix) * pi + eps_mix / pi.numel())
        p_cap = cap_mult / pi.numel()
        pi = normalize_prob(torch.minimum(pi, torch.full_like(pi, p_cap)))

        if debug:
            surf_actual = float((pi * mask_soft).sum())
            pi_breakdown(pi, p_surf_raw, p_target, gate_k, tag="STAGE4_final")
            print(f"  surf_mass(actual)={surf_actual:.3f}")

        if not (torch.isfinite(pi).all() and (pi >= 0).all() and abs(float(pi.sum()) - 1.0) < 1e-6):
            pi = normalize_prob(pi)

        log_pi = torch.log(pi.clamp_min(1e-12))
    return pi, log_pi

# =========================================================================
# 2) Soft-NMS suppression (optional)
# =========================================================================
@torch.no_grad()
def soft_nms_pi(pi: torch.Tensor, anchors: torch.Tensor, nn_idx_all: torch.Tensor,
                suppress_lambda: float = 0.6, r_scale: float = 0.9, meanNN: Optional[float] = None) -> torch.Tensor:
    if meanNN is None:
        idx = torch.randint(0, anchors.size(0), (min(4096, anchors.size(0)),), device=anchors.device)
        d = torch.cdist(anchors[idx], anchors[idx]); d.fill_diagonal_(1e9)
        meanNN = float(d.min(dim=1).values.median().item())
    r = r_scale * meanNN
    neigh = anchors[nn_idx_all]; center = anchors[:, None, :]
    d2 = (neigh - center).pow(2).sum(-1)
    w  = torch.exp(-d2 / (r * r))
    supp = (w * pi[nn_idx_all]).sum(dim=1)
    decay = torch.exp(-suppress_lambda * supp / (pi + 1e-12))
    return normalize_prob((pi * decay).clamp(min=1e-12))

# =========================================================================
# 3) Voxel hash-based diversity guarantee
# =========================================================================
@torch.no_grad()
def sample_centers_with_voxel_quota(
    pi: torch.Tensor, anchors: torch.Tensor, batch_size: int, meanNN: float,
    voxel_scale: float = 0.9, bucket_quota: Optional[Dict[int, int]] = None, cat: Optional[torch.distributions.Categorical] = None
) -> Tuple[torch.Tensor, Dict[int, int]]:
    device = pi.device; N = pi.numel()
    voxel_size = voxel_scale * meanNN
    vox = (anchors / voxel_size).floor().long()
    key = (vox[:, 0] * 73856093 ^ vox[:, 1] * 19349663 ^ vox[:, 2] * 83492791) & 0x7fffffff

    if bucket_quota is None:
        unique_keys, inverse = key.unique(return_inverse=True)
        bucket_mass = torch.zeros(len(unique_keys), device=device, dtype=pi.dtype)
        bucket_mass.scatter_add_(0, inverse, pi)
        total = bucket_mass.sum()
        bucket_quota = {k.item(): int(max(1, round(float((bucket_mass[i] / (total + 1e-12)).item()) * batch_size)))
                        for i, k in enumerate(unique_keys)}

    centers = torch.multinomial(pi, num_samples=batch_size, replacement=True)

    key_b = key[centers]
    uniq_b, cnt_b = key_b.unique(return_counts=True)
    over_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
    for k, c in zip(uniq_b.tolist(), cnt_b.tolist()):
        q = bucket_quota.get(k, 1)
        if c > q:
            pos = (key_b == k).nonzero(as_tuple=True)[0]
            over_mask[pos[q:]] = True

    if over_mask.any():
        over_idx  = over_mask.nonzero(as_tuple=False).squeeze(1)
        over_keys = key_b[over_idx]
        pi_masked = pi.clone()
        pi_masked[torch.isin(key, over_keys)] = 0.0
        s = pi_masked.sum()
        if s > 1e-12:
            pi_masked = pi_masked / s
            centers[over_idx] = torch.multinomial(pi_masked, num_samples=over_idx.numel(), replacement=True)
    return centers, bucket_quota

# =========================================================================
# 4) Fast Local-Soft (optimized: vectorized + cached)
# =========================================================================
def sample_points_fast(
    anchors: torch.Tensor, normals: torch.Tensor, p_surf_raw: torch.Tensor, w_den: torch.Tensor,
    pi: torch.Tensor, log_pi: torch.Tensor, *, M: int, nn_idx_all: torch.Tensor, t1_all: torch.Tensor, t2_all: torch.Tensor,
    S: int = 16, tau_local: float = 0.50, batch_size: int = 32768, jitter_coef: float = 0.22,
    use_amp: bool = True, use_st: bool = False, u_local: float = 0.20, w_cap: Optional[float] = None,
    bias_target: float = 3.0, use_dynamic_cap: bool = True, center_tau: float = 3.5,
    use_voxel_diversity: bool = True, voxel_scale: float = 0.9, meanNN_cache: Optional[float] = None,
    bucket_quota_cache: Optional[Dict[int, int]] = None, cat_cache: Optional[torch.distributions.Categorical] = None,
    rng: Optional[torch.Generator] = None
) -> Dict[str, torch.Tensor]:

    device, dtype, N = anchors.device, anchors.dtype, anchors.size(0)
    if rng is None:
        rng = torch.Generator(device=device).manual_seed(2025)

    if meanNN_cache is None:
        with torch.no_grad():
            idx_samp = torch.randint(0, N, (min(4096, N),), device=device)
            d = torch.cdist(anchors[idx_samp], anchors[idx_samp]); d.fill_diagonal_(1e9)
            meanNN = d.min(dim=1).values.median().item()
    else:
        meanNN = meanNN_cache
    j0 = jitter_coef * float(meanNN)

    mu  = torch.empty(M, 3, device=device, dtype=dtype)
    nrm = torch.empty(M, 3, device=device, dtype=dtype)
    pup = torch.empty(M, device=device, dtype=dtype)
    parent_idx_out = torch.empty(M, device=device, dtype=torch.long)
    counts = torch.zeros(N, device=device, dtype=dtype)

    bucket_quota = bucket_quota_cache
    num_batches, write = (M + batch_size - 1) // batch_size, 0

    amp_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
               if (use_amp and device.type == "cuda")
               else torch.amp.autocast("cuda", enabled=False))

    with amp_ctx:
        for _ in range(num_batches):
            B = min(batch_size, M - write)

            with torch.no_grad():
                if use_dynamic_cap and write > 0:
                    mean_sofar = write / max(1, N)
                    cap = torch.ceil(torch.tensor(bias_target * max(1.0, float(mean_sofar)), device=pi.device))
                    alive = (counts < cap)
                    masked_probs = torch.where(alive, pi, torch.zeros_like(pi))
                    masked_probs = masked_probs / (masked_probs.sum() + 1e-12)
                else:
                    masked_probs = pi

            if use_voxel_diversity:
                centers, bucket_quota = sample_centers_with_voxel_quota(
                    masked_probs, anchors, B, meanNN, voxel_scale, bucket_quota, cat=None
                )
            else:
                centers = torch.multinomial(masked_probs, num_samples=B, replacement=True, generator=rng)

            parent_idx_out[write:write+B] = centers

            nn_idx = nn_idx_all[centers]
            A  = anchors[nn_idx]
            Nr = normals[nn_idx]
            Ps = p_surf_raw[nn_idx]
            logpi_local = log_pi[nn_idx]
            tau_adj = max(tau_local, 1.15)
            W_soft = F.softmax(logpi_local / tau_adj, dim=1)
            W_soft = (1.0 - u_local) * W_soft + (u_local / S)
            if w_cap is not None:
                W_soft = torch.minimum(W_soft, W_soft.new_full(W_soft.shape, w_cap))
            W = W_soft / (W_soft.sum(dim=1, keepdim=True) + 1e-12)

            mu_b = torch.bmm(W.unsqueeze(1), A).squeeze(1)
            n_b  = F.normalize(torch.bmm(W.unsqueeze(1), Nr).squeeze(1), dim=-1)
            wloc = (W * w_den[nn_idx]).sum(dim=1)
            p_upb = (W * Ps).sum(dim=1)

            t1_b = t1_all[centers]; t2_b = t2_all[centers]
            u = torch.randn(B, 1, generator=rng, device=device, dtype=dtype)
            v = torch.randn(B, 1, generator=rng, device=device, dtype=dtype)
            
            # CRITICAL: Clamp p_upb to [0,1] to prevent jitter explosion
            # Even if p_surf_raw is normalized, weighted average can occasionally exceed 1.0
            p_upb = p_upb.clamp(0, 1)
            j = j0 * (0.7 * p_upb + 0.3 * wloc)
            mu_b = mu_b + j.unsqueeze(-1) * (u * t1_b + v * t2_b)

            mu[write:write+B]  = mu_b
            nrm[write:write+B] = n_b
            pup[write:write+B] = p_upb
            counts.index_add_(0, centers, torch.ones_like(centers, dtype=dtype))

            write += B

    return {"mu": mu, "n": nrm, "p_up": pup, "parent_idx": parent_idx_out,
            "counts": counts, "meanNN": meanNN, "bucket_quota": bucket_quota}

# =========================================================================
# Wrapper: sample_points (pipeline compatibility)
# =========================================================================
def sample_points(
    x: torch.Tensor, normals: torch.Tensor, spacing: torch.Tensor,
    probs: Optional[torch.Tensor], cfg: dict, generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    device = x.device; N = x.shape[0]
    M = int(cfg.get("M", 50000))
    S = int(cfg.get("local_S", 16))
    tau_local   = float(cfg.get("tau_local", 1.40))
    local_batch = int(cfg.get("local_batch", 32768))
    jitter_coef = float(cfg.get("micro_jitter_j0", 0.22))
    use_amp     = bool(cfg.get("use_amp", True))
    use_st      = bool(cfg.get("use_st", False))
    use_voxel_diversity = bool(cfg.get("use_voxel_diversity", True))
    voxel_scale = float(cfg.get("voxel_scale", 0.50))
    debug_pi    = bool(cfg.get("debug", {}).get("pi_breakdown", False))
    uniform_pi  = bool(cfg.get("uniform_pi", False))

    p_surf_raw = cfg.get("p_surf_raw")
    w_den      = cfg.get("w_density")

    # 안전장치: 없으면 균일 스칼라로 채움
    if p_surf_raw is None:
        p_surf_raw = torch.ones(N, device=device, dtype=x.dtype)
    if w_den is None:
        w_den = torch.ones(N, device=device, dtype=x.dtype)
    
    # CRITICAL: Auto-normalize p_surf_raw to [0,1] to prevent jitter explosion
    # If p_surf_raw is 1/|SDF| (range [300, 10000]), jitter becomes huge!
    with torch.no_grad():
        p_min, p_max = p_surf_raw.min(), p_surf_raw.max()
        if p_max > 10.0:  # Likely unnormalized (1/|SDF|)
            if debug_pi:
                print(f"\n{'='*60}")
                print(f"[AUTO-NORMALIZE] p_surf_raw detected!")
                print(f"{'='*60}")
                print(f"  Original range: [{float(p_min):.1f}, {float(p_max):.1f}]")
                print(f"  Likely 1/|SDF| values → normalizing to [0,1]...")
            p_surf_raw = (p_surf_raw - p_min) / (p_max - p_min + 1e-8)
            if debug_pi:
                print(f"  Normalized range: [{float(p_surf_raw.min()):.4f}, {float(p_surf_raw.max()):.4f}]")
                print(f"  ✓ Jitter explosion prevented!")
                print(f"{'='*60}\n")
        elif p_max > 2.0:  # Suspicious range
            if debug_pi:
                print(f"[WARNING] p_surf_raw max={float(p_max):.2f} > 2.0, clamping to [0,1]...")
            p_surf_raw = p_surf_raw.clamp(0, 1)

    nn_idx_all = precompute_knn_indices(x, S=S, exclude_self=True)
    t1_all, t2_all = precompute_tangent_bases(normals)

    # π 구성
    if probs is not None:
        pi_cache   = normalize_prob(probs.detach().float()).to(x.dtype)
        log_pi_cache = torch.log(pi_cache.clamp_min(1e-12))
        if debug_pi:
            pi_breakdown(pi_cache, p_surf_raw, cfg.get("p_target", 0.85), cfg.get("gate_k", 120.0), tag="used_pi (external)")
    else:
        # uniform_pi 스위치가 켜지면 build_pi_complete 내부에서 균일 π 생성
        pi_cache, log_pi_cache = build_pi_complete(
            p_surf_raw, w_den, x, nn_idx_all, cfg if uniform_pi else cfg, debug=debug_pi
        )

    result = sample_points_fast(
        anchors=x, normals=normals, p_surf_raw=p_surf_raw, w_den=w_den,
        pi=pi_cache, log_pi=log_pi_cache, M=M,
        nn_idx_all=nn_idx_all, t1_all=t1_all, t2_all=t2_all, S=S, tau_local=tau_local,
        batch_size=local_batch, jitter_coef=jitter_coef, use_amp=use_amp, use_st=use_st,
        u_local=float(cfg.get("u_local", 0.68)),
        w_cap=float(cfg.get("w_cap", 0.10)) if cfg.get("w_cap") is not None else None,
        bias_target=float(cfg.get("bias_target", 2.5)),
        use_dynamic_cap=bool(cfg.get("use_dynamic_cap", True)),
        center_tau=float(cfg.get("center_tau", 6.0)),
        use_voxel_diversity=use_voxel_diversity, voxel_scale=voxel_scale,
        meanNN_cache=cfg.get("_cache", {}).get("meanNN", None),
        bucket_quota_cache=cfg.get("_cache", {}).get("bucket_quota", None),
        rng=generator,
    )

    parent_idx = result["parent_idx"]; anchors_selected = x.index_select(0, parent_idx)

    with torch.no_grad():
        sc = result["counts"]
        def corr(a, b):
            a = (a - a.mean()) / (a.std() + 1e-6)
            b = (b - b.mean()) / (b.std() + 1e-6)
            return float((a*b).mean().item())
        corr_spacing_count = corr(spacing, sc)
        if bool(cfg.get("debug", {}).get("pi_breakdown", False)):
            print(f"[Selection Stats] mean={float(sc.mean()):.2f}, max={float(sc.max()):.2f}, "
                  f"bias={float(sc.max()/(sc.mean()+1e-6)):.2f}")
            print(f"[Correlation] corr(spacing, sel_count) = {corr_spacing_count:+.3f}")

    if "_cache" not in cfg: cfg["_cache"] = {}
    cfg["_cache"]["meanNN"]     = result["meanNN"]
    cfg["_cache"]["bucket_quota"]= result["bucket_quota"]
    cfg["p_up"] = result["p_up"]
    if "diagnostics" not in cfg: cfg["diagnostics"] = {}
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