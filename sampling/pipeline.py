"""
Modified Differentiable Point Cloud Upsampling Pipeline.

CHANGES FROM ORIGINAL:
- STAGE 2: Replaced "Volume Filtering" with "Anchor Redistribution"
- New approach identifies sparse regions and generates additional anchors
- Improves surface coverage by evening out anchor spacing

This module implements a complete 6-stage pipeline for high-quality point cloud
upsampling with deformation-aware covariance estimation. All stages are fully
differentiable for end-to-end learning.

═══════════════════════════════════════════════════════════════════════════════
PIPELINE OVERVIEW (MODIFIED)
═══════════════════════════════════════════════════════════════════════════════

Input:  N sparse anchors with deformation gradients {xᵢ, Fᵢ}
Output: M dense points with anisotropic covariances {pⱼ, Σⱼ}  (M >> N)

         ┌────────────────────────────────────────────────────┐
         │  INPUT: Sparse Point Cloud + Deformation Field     │
         │  • x_low: (N, 3) anchor positions                  │
         │  • F_low: (N, 3, 3) deformation gradients          │
         └─────────────────┬──────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────────────┐
         │  STAGE 1: Surface Detection (PCA Analysis)          │
         │  ─────────────────────────────────────────          │
         │  • Weighted PCA on local neighborhoods              │
         │  • Extract: normals, surface variance, spacing      │
         │  • Compute: surf_prob = f(planarity, EMA)           │
         │                                                     │
         │  Output: {normals, surf_prob, spacing}              │
         └─────────────────┬───────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────────────┐
         │  STAGE 2: Anchor Redistribution ★ NEW ★            │
         │  ────────────────────────────────────               │
         │  • Identify sparse regions via density analysis    │
         │  • Generate new anchors in sparse areas             │
         │  • Project onto local surface patches               │
         │  • Merge with original anchors                      │
         │                                                     │
         │  Output: {anchors_new, probs_new, normals_new}      │
         └─────────────────┬───────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────────────┐
         │  STAGE 3: Importance Sampling (Gumbel-Softmax)      │
         │  ──────────────────────────────────────────         │
         │  • Sample M indices: Y ~ GumbelSoftmax(probs, τ)    │
         │  • Interpolate: anchors = Y @ x_low                 │
         │  • Build tangent frame {t₁, t₂, n}                  │
         │  • Jitter: p = anchor + α·h·(U·t₁ + V·t₂) + ε       │
         │                                                     │
         │  Output: {points (M,3), normals_up, anchors}        │
         └─────────────────┬───────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────────────┐
         │  STAGE 4: Taubin Smoothing (Shrinkage-Free)         │
         │  ────────────────────────────────────────           │
         │  • Laplacian pass:  p' = p + λ·L·p                  │
         │  • Inflation pass:  p" = p' + μ·L·p'                │ 
         │  • Constraint: tangent motion only (preserve n)     │
         │                                                     │
         │  Output: {smoothed_points}                          │
         └─────────────────┬───────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────────────┐
         │  STAGE 5: Normal Smoothing (Spatial Laplacian)      │
         │  ───────────────────────────────────────────        │
         │  • Adaptive bandwidth: h = soft_median(distances)   │
         │  • Spatial weights: w = exp(-d²/h²)                 │ 
         │  • Smooth: n' = normalize(Σ wᵢ·nᵢ)                  │
         │  • EMA blend: n ← λ·n' + (1-λ)·n                    │
         │                                                     │
         │  Output: {smoothed_normals}                         │
         └─────────────────┬───────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────────────────┐
         │  STAGE 6: Covariance Construction (F-field)         │
         │  ────────────────────────────────────────           │
         │  • Smooth F-field: Graph Laplacian on anchors       │
         │  • Interpolate F to upsampled points via KNN        │
         │  • Polar decomposition: F = R·S                     │
         │  • Build covariance: Σ = R·S·Σ₀·S·Rᵀ                │
         │                                                     │
         │  Output: {cov (M,3,3), F_interp}                    │
         └─────────────────┬───────────────────────────────────┘
                           │
                           ▼
         ┌────────────────────────────────────────────────────┐
         │  OUTPUT: Dense Point Cloud with Covariances        │
         │  • points: (M, 3) upsampled positions              │
         │  • normals: (M, 3) smoothed normals                │
         │  • cov: (M, 3, 3) anisotropic covariances          │
         │  • F_interp: (M, 3, 3) interpolated F-field        │
         └────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
KEY CHANGES IN STAGE 2
═══════════════════════════════════════════════════════════════════════════════

OLD: Volume Filtering
- Soft geometric consistency check
- Filters out volume points using normal consensus
- Reduces probability weights

NEW: Anchor Redistribution
- Identifies sparse surface regions using density analysis
- Generates new candidate anchors in sparse areas
- Projects candidates onto local surface patches
- Increases anchor count to improve coverage

Benefits:
+ More uniform surface coverage
+ Better handling of sparse regions
+ Proactive gap filling (vs reactive filtering)
+ Maintains differentiability

"""

import torch
from typing import Dict, Optional
from pathlib import Path

from .utils.config import default_cfg, validate_cfg
from .utils.utils import ensure_torch, as_numpy
from .analysis.knn import HybridFAISSKNN, FAISS_AVAILABLE
from .core.surface_detect import detect_surface
# REMOVED: from .core.volume_filter import apply_volume_filter
from .core.anchor_redistribution import redistribute_anchors  # ★ NEW ★
from .core.sampler import sample_points
from .core.taubin_smooth import taubin_smooth
from .core.normal_smooth import smooth_normals
from .geometry.covariance import build_covariance
from .io.export import save_comparison_png, save_anchor_visualization


def upsample(
    x_low: torch.Tensor,
    F_low: torch.Tensor,
    cfg: Optional[Dict] = None,
    state: Optional[Dict] = None,
    seed: int = 1234,
    return_torch: bool = True
) -> Dict:
    """
    Main differentiable point cloud upsampling pipeline.
    
    ** MODIFIED VERSION **
    STAGE 2 now uses Anchor Redistribution instead of Volume Filtering.
    
    Transforms sparse point cloud with deformation gradients into dense
    point cloud with anisotropic covariances for high-quality rendering.
    
    Pipeline stages:
        1. Surface Detection: PCA-based planarity analysis
        2. Anchor Redistribution: Densify sparse regions (★ NEW ★)
        3. Importance Sampling: Gumbel-Softmax with tangent jitter
        4. Taubin Smoothing: Shrinkage-free Laplacian smoothing
        5. Normal Smoothing: Spatial Laplacian with adaptive bandwidth
        6. Covariance Construction: Deformation-aware via F-field
    
    All operations are fully differentiable, enabling:
        - Learning importance sampling distribution
        - Joint optimization with neural networks
        - End-to-end surface reconstruction
    
    Args:
        x_low: (N, 3) coarse anchor point positions
        F_low: (N, 3, 3) deformation gradient tensors at anchors
        cfg: Configuration dictionary (optional)
        state: State dictionary for EMA and caching (optional)
        seed: Random seed for reproducibility
        return_torch: Return format (default: True)
    
    Returns:
        result: Dictionary containing:
            - points: (M, 3) upsampled point positions
            - normals: (M, 3) surface normals
            - cov: (M, 3, 3) anisotropic covariance matrices
            - F_interp: (M, 3, 3) interpolated deformation gradients
            - anchors: (M, 3) anchor positions before jittering
            - debug: Dictionary with diagnostic information
            - state: Updated state dictionary
    """
    
    # ========================================================================
    # Setup and Initialization
    # ========================================================================
    
    # Initialize configuration
    if cfg is None:
        cfg = default_cfg()
    else:
        cfg = validate_cfg(cfg)
    
    # Initialize state
    if state is None:
        state = {}
    
    N = x_low.shape[0] if torch.is_tensor(x_low) else len(x_low)
    
    # Determine device
    device = x_low.device if torch.is_tensor(x_low) else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Ensure torch tensors on correct device
    x_low = ensure_torch(x_low, device=device)
    F_low = ensure_torch(F_low, device=device).reshape(-1, 3, 3)
    
    # Setup KNN module
    knn_cfg = cfg.get("knn", {})
    knn = HybridFAISSKNN(
        use_faiss=knn_cfg.get("use_faiss", True) and FAISS_AVAILABLE,
        use_ivf=knn_cfg.get("use_ivf", True),
        tau=knn_cfg.get("tau", 0.15),
        nlist=knn_cfg.get("nlist", 100),
        nprobe=knn_cfg.get("nprobe", 10),
    )
    
    # Random generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Verbose mode
    verbose = cfg.get("debug", {}).get("verbose", False)
    
    # ========================================================================
    # STAGE 1: Surface Detection (PCA-based Planarity)
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 1/6: Surface Detection (PCA-based Planarity)")
        print("="*80)
    
    surf_cfg = cfg.get("surface_detection", {})
    if surf_cfg.get("enabled", True):
        surf_prob, normals, spacing, state = detect_surface(
            x_low, knn, surf_cfg, state
        )
    else:
        # Uniform probability fallback
        N = x_low.shape[0]
        surf_prob = torch.full((N,), 1.0 / N, device=device)
        
        # Still need normals for downstream stages
        from .analysis.pca import batched_pca_surface_optimized
        k = surf_cfg.get("k", 48)
        idx, w = knn(x_low, x_low, k)
        normals, _, spacing = batched_pca_surface_optimized(x_low, idx, w)
    
    if verbose:
        print(f"✓ Computed surface probabilities for {len(x_low)} points")
        print(f"  Mean prob: {surf_prob.mean():.6f}")
        print(f"  Max prob:  {surf_prob.max():.6f}")
        print(f"  Min prob:  {surf_prob.min():.6f}")
    
    # ========================================================================
    # STAGE 2: Anchor Redistribution ★ NEW ★
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 2/6: Anchor Redistribution (Sparse Region Densification)")
        print("="*80)
        print("  [NEW] Replacing Volume Filtering with Anchor Redistribution")
    
    # Get anchor redistribution config
    anchor_cfg = cfg.get("anchor_redistribution", {})
    
    if anchor_cfg.get("enabled", True):
        # Store original count
        N_original = len(x_low)
        
        # Redistribute anchors to densify sparse regions
        x_low_new, normals_new, surf_prob_new = redistribute_anchors(
            points=x_low,
            normals=normals,
            surf_prob=surf_prob,
            spacing=spacing,
            knn=knn,
            cfg=anchor_cfg,
        )
        
        # Update variables with redistributed anchors
        N_new = len(x_low_new)
        N_added = N_new - N_original
        
        if verbose:
            print(f"✓ Anchor redistribution complete")
            print(f"  Original anchors:    {N_original:,}")
            print(f"  New candidates:      {N_added:,}")
            print(f"  Total anchors:       {N_new:,}")
            if N_added > 0:
                print(f"  Densification:       {100*N_added/N_original:.1f}% increase")
                print(f"  New prob stats:")
                print(f"    • mean: {surf_prob_new.mean():.6f}")
                print(f"    • max:  {surf_prob_new.max():.6f}")
                print(f"    • min:  {surf_prob_new.min():.6f}")
        
        # Expand F_low to match new anchor count by duplicating nearest neighbor F
        if N_added > 0:
            # For new candidates, interpolate F from original anchors
            _, indices = knn.query(x_low_new[N_original:], k=1)  # Find nearest original
            F_new_candidates = F_low[indices.squeeze()]  # (N_added, 3, 3)
            F_low = torch.cat([F_low, F_new_candidates], dim=0)
        
        # Update main variables
        x_low = x_low_new
        normals = normals_new
        filtered_prob = surf_prob_new
        
        # Update spacing for new points (recompute for all)
        if N_added > 0:
            from .analysis.pca import batched_pca_surface_optimized
            k_spacing = surf_cfg.get("k", 48)
            idx, w = knn(x_low, x_low, k_spacing)
            _, _, spacing = batched_pca_surface_optimized(x_low, idx, w)
        
        # For compatibility, create a volume_weight tensor (all 1.0 since we're not filtering)
        volume_weight = torch.ones_like(filtered_prob)
        
    else:
        # Redistribution disabled - use original anchors
        filtered_prob = surf_prob
        volume_weight = torch.ones_like(surf_prob)
        
        if verbose:
            print("⊘ Anchor redistribution disabled (using original anchors)")
    
    # Proactive cleanup
    del surf_prob
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ========================================================================
    # STAGE 3: Importance Sampling (Gumbel-Softmax + Tangent Jitter)
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 3/6: Importance Sampling (Gumbel-Softmax + Tangent Jitter)")
        print("="*80)

    samp_cfg = cfg.get("sampling", {})
    N = len(x_low)  # number of anchors (may be increased from redistribution)
    
    # Extract key parameters for logging
    M = int(samp_cfg.get("M", 50000))
    tau = float(samp_cfg.get("tau", 0.2))
    alpha = float(samp_cfg.get("alpha", 0.35))
    thickness = float(samp_cfg.get("thickness", 0.0))
    gs_batch = int(samp_cfg.get("gs_batch", 2048))
    ensure_cover = bool(samp_cfg.get("ensure_anchor_coverage", True))
    micro_jitter_scale = float(samp_cfg.get("micro_jitter_scale", 0.2))
    tangent_micro_only = bool(samp_cfg.get("tangent_micro_only", True))
    
    # Hole-fix patches
    prob_floor = float(samp_cfg.get("prob_floor", 1e-8))
    uniform_mix = float(samp_cfg.get("uniform_mix", 0.02))
    plane_snap = bool(samp_cfg.get("plane_snap", True))
    plane_snap_beta = float(samp_cfg.get("plane_snap_beta", 0.5))
    topk_pool = int(samp_cfg.get("topk_pool", 8))
    thickness_gamma = float(samp_cfg.get("thickness_gamma", 0.15))
    
    # Surface-constrained sampling
    surface_support_q = float(samp_cfg.get("surface_support_q", 0.80))
    prob_floor_mode = str(samp_cfg.get("prob_floor_mode", "density"))
    uniform_mix_surface_only = bool(samp_cfg.get("uniform_mix_surface_only", True))
    coverage_only_surface = bool(samp_cfg.get("coverage_only_surface", True))
    mask_topk_with_surface = bool(samp_cfg.get("mask_topk_with_surface", True))
    
    # Density-based floor
    density_floor_tau = float(samp_cfg.get("density_floor_tau", 1.0))
    density_floor_gamma = float(samp_cfg.get("density_floor_gamma", 2.0))
    
    # One-sided thickness
    thickness_one_sided = bool(samp_cfg.get("thickness_one_sided", True))
    inside_barrier_lambda = float(samp_cfg.get("inside_barrier_lambda", 1.0))

    # Rough peak per batch for logits/softmax (float32)
    est_mb_per_batch = (gs_batch * max(N, 1) * 4) / (1024**2)

    if verbose:
        print(f"- Anchors (N): {N:,}")
        print(f"- Target samples (M): {M:,}  (upsampling {M/max(N,1):.1f}×)")
        print(f"\n  [Core Sampling]")
        print(f"  • Gumbel tau: {tau:.3f} | alpha: {alpha:.3f} | thickness: {thickness:.3f}")
        print(f"  • gs_batch: {gs_batch} (≈ {est_mb_per_batch:.1f} MB/batch)")
        print(f"  • micro_jitter: {micro_jitter_scale:.3f} (tangent_only={tangent_micro_only})")
        print(f"\n  [Hole-Fix Patches]")
        print(f"  • prob_floor: {prob_floor:.1e} | uniform_mix: {uniform_mix:.3f}")
        print(f"  • plane_snap: {plane_snap} (beta={plane_snap_beta:.2f})")
        print(f"  • topk_pool: {topk_pool} | thickness_gamma: {thickness_gamma:.2f}")
        print(f"\n  [Surface Constraints]")
        print(f"  • surface_support_q: {surface_support_q:.2f}")
        print(f"  • prob_floor_mode: {prob_floor_mode}")
        print(f"  • uniform_mix_surface_only: {uniform_mix_surface_only}")
        print(f"  • coverage_only_surface: {coverage_only_surface}")
        print(f"  • mask_topk_with_surface: {mask_topk_with_surface}")
        print(f"\n  [Density Floor]")
        print(f"  • density_floor_tau: {density_floor_tau:.2f}")
        print(f"  • density_floor_gamma: {density_floor_gamma:.2f}")
        print(f"\n  [One-Sided Thickness]")
        print(f"  • thickness_one_sided: {thickness_one_sided}")
        print(f"  • inside_barrier_lambda: {inside_barrier_lambda:.2f}")
        print(f"\n  [Coverage]")
        print(f"  • ensure_anchor_coverage: {ensure_cover}")
        
        # Warnings
        if micro_jitter_scale > 0.25 and not tangent_micro_only:
            print(f"\n  [WARN] High micro_jitter_scale with isotropic micro may increase interior leakage")
        if thickness > 0.0 and not thickness_one_sided:
            print(f"\n  [WARN] Symmetric thickness can create interior points")

    # Call sampler (memory-safe ST version keeps the same signature)
    points, normals_up, anchors = sample_points(
        x_low, normals, spacing, filtered_prob, samp_cfg, generator
    )

    # 🎨 Export anchor visualization (debug mode)
    debug_cfg = cfg.get("debug", {})
    if debug_cfg.get("export_anchors", False):
        export_dir = Path(debug_cfg.get("export_dir", "debug/"))
        export_dir.mkdir(parents=True, exist_ok=True)
        anchor_path = export_dir / "anchor_sampling.png"
        try:
            save_anchor_visualization(
                anchor_path,
                x_low=x_low,
                anchors=anchors,
                surf_prob=filtered_prob,  # Updated variable name
                volume_weight=volume_weight,
                dpi=debug_cfg.get("png_dpi", 160),
                ptsize=debug_cfg.get("png_ptsize", 0.5)
            )
        except Exception as e:
            if verbose:
                print(f"  [WARN] Failed to save anchor visualization: {e}")

    # Proactive cleanup (helps control transient peaks between stages)
    del filtered_prob, spacing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Result summary
    if verbose:
        up_factor = len(points) / max(N, 1)
        print(f"\n✓ Sampled {len(points):,} points from {N:,} anchors ({up_factor:.1f}×)")
        if len(points) != M:
            print(f"  [INFO] Expected {M:,} but got {len(points):,} points")

    
    # ========================================================================
    # STAGE 4: Taubin Smoothing (Shrinkage-Free Laplacian)
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 4/6: Taubin Smoothing (Shrinkage-Free Laplacian)")
        print("="*80)
    
    taubin_cfg = cfg.get("taubin", {})
    if taubin_cfg.get("enabled", True):
        points = taubin_smooth(points, normals_up, knn, taubin_cfg)
        
        if verbose:
            n_iters = taubin_cfg.get('iters', 3)
            lam = taubin_cfg.get('lambda_smooth', 0.33)
            mu = taubin_cfg.get('lambda_inflate', -0.53)
            print(f"✓ Applied {n_iters} iterations of Taubin smoothing")
            print(f"  λ (smooth): {lam:+.3f}")
            print(f"  μ (inflate): {mu:+.3f}")
    else:
        if verbose:
            print("⊘ Taubin smoothing disabled")
    
    # ========================================================================
    # STAGE 5: Normal Smoothing (Spatial Laplacian)
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 5/6: Normal Smoothing (Spatial Laplacian)")
        print("="*80)
    
    norm_cfg = cfg.get("normal_smooth", {})
    if norm_cfg.get("enabled", True):
        normals_up = smooth_normals(normals_up, points, knn, norm_cfg)
        
        if verbose:
            n_iters = norm_cfg.get('iters', 2)
            lam = norm_cfg.get('lambda_smooth', 0.8)
            k = norm_cfg.get('k', 16)
            print(f"✓ Applied {n_iters} iterations of normal smoothing")
            print(f"  λ (blend): {lam:.3f}")
            print(f"  k (neighbors): {k}")
    else:
        if verbose:
            print("⊘ Normal smoothing disabled")
    
    # ========================================================================
    # STAGE 6: Covariance Construction (F-field Interpolation)
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 6/6: Covariance Construction (F-field Interpolation)")
        print("="*80)
    
    cov_cfg = cfg.get("covariance", {})
    cov, F_interp, _ = build_covariance(points, x_low, F_low, knn, cov_cfg)
    
    if verbose:
        use_polar = cov_cfg.get("use_polar_decomposition", True)
        sigma0 = cov_cfg.get("sigma0", 0.08)
        k_F = cov_cfg.get("k_F", 32)
        print(f"✓ Built {len(cov):,} covariance matrices")
        print(f"  Method: {'Polar decomposition' if use_polar else 'Direct (FF^T)'}")
        print(f"  σ₀ (base scale): {sigma0:.4f}")
        print(f"  k_F (neighbors): {k_F}")
    
    # ========================================================================
    # Prepare Output
    # ========================================================================
    debug_info = {
        "N_input": len(x_low),
        "M_output": len(points),
        "upsampling_factor": len(points) / len(x_low),
        "surface_detection": surf_cfg.get("enabled", True),
        "anchor_redistribution": anchor_cfg.get("enabled", True),  # Updated
        "taubin_smoothing": taubin_cfg.get("enabled", True),
        "normal_smoothing": norm_cfg.get("enabled", True),
        "mean_volume_weight": float(volume_weight.mean().detach().item()),
        "device": str(device),
        "seed": seed,
    }
    
    if verbose:
        print("="*80)
        print("Pipeline Complete!")
        print(f"  Input:  {debug_info['N_input']:,} points")
        print(f"  Output: {debug_info['M_output']:,} points")
        print(f"  Factor: {debug_info['upsampling_factor']:.1f}×")
        print("="*80 + "\n")
        
    # ========================================================================
    # Cleanup
    # ========================================================================
    del normals, volume_weight
    perf_cfg = cfg.get("performance", {})
    if perf_cfg.get("clear_cache", True):
        knn.clear_cache()
        
        if verbose:
            print("\n" + "="*80)
            print("Cleanup: FAISS cache cleared")
    
  
    # Convert to numpy if requested
    if return_torch:
        return {
            "points": points,
            "normals": normals_up,
            "cov": cov,
            "F_interp": F_interp,
            "anchors": anchors,
            "debug": debug_info,
            "state": state,
        }
    else:
        return {
            "points": as_numpy(points),
            "normals": as_numpy(normals_up),
            "cov": as_numpy(cov),
            "F_interp": as_numpy(F_interp),
            "anchors": as_numpy(anchors),
            "debug": debug_info,
            "state": state,
        }

__all__ = [
    "upsample",
]