# sampling/core/stage4_diagnostics.py
"""
Stage 4 Projection & Smoothing Diagnostics

성공 기준 (권장 임계값):
  - Re-projection residual: |φ(x_proj)| mean ≤ 0.05·Δx, p95 ≤ 0.08·Δx
  - Projection drift: ‖x_proj - x_in‖ mean ≤ 0.15·Δx, p95 ≤ 0.4·Δx
  - Normal unit: ‖n‖ mean ≈ 1.00±0.02, min ≥ 0.95
  - Normal change: ∠(n_before, n_after) mean ≤ 5°, p95 ≤ 12°
  - Voxel duplication: mean ≤ 1.6, p95 ≤ 3, max ≤ 6
  - Coverage uniformity: EffSize ≥ 0.6·N_anchors
  - Boundary clamp ratio: ≤ 1%

Author: CHAYO
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional

from .levelset_ops import phi5, world_to_grid, grad_normals


@torch.no_grad()
def diag_stage4(
    levelset,                         # LevelSetGrid (phi, bbox_min/max)
    points_ref: torch.Tensor,         # Stage 4 결과 포인트 (M,3)
    normals_ref: torch.Tensor,        # Stage 4 노멀 (M,3)
    points_in: torch.Tensor = None,   # Stage 3 샘플 포인트 (선택)
    normals_in: torch.Tensor = None,  # Stage 3 노멀 (선택, 없으면 자동 계산)
    voxel_quota_scale: float = 0.9,   # 복셀 크기 스케일 (중복 체크용)
) -> Dict:
    """
    Stage 4 품질 진단
    
    Returns:
        dict with diagnostics:
          - voxel_size: Δx (grid voxel size)
          - phi_abs: |φ| residual stats
          - drift: projection drift stats (if points_in provided)
          - normal_norm: ‖n‖ stats
          - normal_vs_sdf_deg: angle between normals_ref and ∇φ
          - normal_in_to_out_deg: angle change (normals computed from points_in if not provided)
          - voxel_dup: voxel duplication stats
          - boundary_clamp_ratio: ratio of points clamped to bbox
    """
    device = points_ref.device
    dtype  = points_ref.dtype
    R      = levelset.phi.shape[0]
    
    # 🔥 normals_in이 제공되지 않으면 points_in에서 정확하게 계산 (fallback)
    # 일반적으로는 pipeline에서 Stage 3 직후 ∇φ로 재계산된 normals_up을 전달함
    if points_in is not None:
        # Recompute normals at points_in position (ignore provided normals_in)
        # This ensures we compare ∇φ(points_in) vs ∇φ(points_ref)
        normals_in = grad_normals(levelset, points_in, require_grad_points=False, detach_phi=True)
    
    # voxel size (Δx)
    vox = ((levelset.bbox_max - levelset.bbox_min) / (R - 1)).mean()
    
    # 1) |φ| residual @ projected points
    sdf5 = phi5(levelset.phi)
    g5   = world_to_grid(points_ref, levelset.bbox_min, levelset.bbox_max).view(1,-1,1,1,3)
    phi_vals = F.grid_sample(sdf5, g5, mode='bilinear', padding_mode='border', align_corners=True).view(-1)
    abs_phi  = phi_vals.abs()
    
    # 2) Projection drift
    drift = None
    if points_in is not None:
        drift = (points_ref - points_in).norm(dim=-1)
    
    # 3) Normal diagnostics
    n_mag = normals_ref.norm(dim=-1)
    
    # SDF로 재계산한 노멀과 각도 비교
    n_sdf = grad_normals(levelset, points_ref, require_grad_points=False, detach_phi=True)
    cos_sim = (F.normalize(normals_ref, dim=-1) * n_sdf).sum(dim=-1).clamp(-1,1)
    angle_deg_ref_vs_sdf = torch.rad2deg(torch.acos(cos_sim))
    
    # 4) Normal change vs input (선택)
    angle_deg_in_out = None
    if normals_in is not None:
        cos_sim_in = (F.normalize(normals_in, dim=-1) * F.normalize(normals_ref, dim=-1)).sum(dim=-1).clamp(-1,1)
        angle_deg_in_out = torch.rad2deg(torch.acos(cos_sim_in))
    
    # 5) Voxel duplication / coverage
    vsize = float(voxel_quota_scale * vox.item())
    vox_idx = (points_ref / max(vsize, 1e-6)).floor().long()
    
    # 32-bit 해시
    key = (vox_idx[:,0]*73856093 ^ vox_idx[:,1]*19349663 ^ vox_idx[:,2]*83492791) & 0x7fffffff
    _, counts = key.unique(return_counts=True)
    
    # 6) Boundary clamp ratio
    eps_bbox = vox * 0.1  # 10% of voxel size tolerance
    near_min = (points_ref - levelset.bbox_min.unsqueeze(0)).abs() < eps_bbox
    near_max = (levelset.bbox_max.unsqueeze(0) - points_ref).abs() < eps_bbox
    boundary_mask = (near_min | near_max).any(dim=1)
    boundary_ratio = float(boundary_mask.float().mean())
    
    # Helper function for statistics
    def stats(t: torch.Tensor):
        if t is None:
            return None
        return dict(
            mean=float(t.mean()),
            p50=float(t.median()),
            p95=float(t.quantile(0.95)),
            p99=float(t.quantile(0.99)),
            max=float(t.max()),
            min=float(t.min())
        )
    
    out = dict(
        voxel_size=float(vox),
        phi_abs=stats(abs_phi),
        drift=stats(drift) if drift is not None else None,
        normal_norm=stats(n_mag),
        normal_vs_sdf_deg=stats(angle_deg_ref_vs_sdf),
        normal_in_to_out_deg=stats(angle_deg_in_out) if angle_deg_in_out is not None else None,
        voxel_dup=dict(
            mean=float(counts.float().mean()),
            p95=float(counts.float().quantile(0.95)),
            max=int(counts.max())
        ),
        boundary_clamp_ratio=boundary_ratio,
    )
    
    return out


def print_stage4_diagnostics(diag: Dict, verbose: bool = True):
    """
    Stage 4 진단 결과를 예쁘게 출력
    """
    if not verbose:
        return
    
    vox = diag['voxel_size']
    
    print("\n" + "="*80)
    print("STAGE 4 DIAGNOSTICS")
    print("="*80)
    
    # 1) Voxel size
    print(f"\nGrid Info:")
    print(f"  Voxel size (Δx): {vox:.6f}")
    
    # 2) φ residual
    phi = diag['phi_abs']
    print(f"\n|φ| Residual (surface projection quality):")
    print(f"  Mean:  {phi['mean']:.6f}  ({phi['mean']/vox:.3f}·Δx)")
    print(f"  P95:   {phi['p95']:.6f}  ({phi['p95']/vox:.3f}·Δx)")
    print(f"  Max:   {phi['max']:.6f}  ({phi['max']/vox:.3f}·Δx)")
    
    # Success check
    if phi['p95'] <= 0.08 * vox:
        print(f"  ✅ PASS: p95 ≤ 0.08·Δx")
    else:
        print(f"  ❌ WARN: p95 > 0.08·Δx (increase proj_iters or lr)")
    
    # 3) Projection drift
    if diag['drift'] is not None:
        drift = diag['drift']
        print(f"\nProjection Drift (‖x_proj - x_in‖):")
        print(f"  Mean:  {drift['mean']:.6f}  ({drift['mean']/vox:.3f}·Δx)")
        print(f"  P95:   {drift['p95']:.6f}  ({drift['p95']/vox:.3f}·Δx)")
        print(f"  Max:   {drift['max']:.6f}  ({drift['max']/vox:.3f}·Δx)")
        
        if drift['p95'] <= 0.4 * vox:
            print(f"  ✅ PASS: p95 ≤ 0.4·Δx")
        else:
            print(f"  ❌ WARN: p95 > 0.4·Δx (reduce jitter or increase center_tau)")
    
    # 4) Normal magnitude
    nmag = diag['normal_norm']
    print(f"\nNormal Magnitude (‖n‖, should be ≈1.0):")
    print(f"  Mean:  {nmag['mean']:.6f}")
    print(f"  Min:   {nmag['min']:.6f}")
    print(f"  Max:   {nmag['max']:.6f}")
    
    if abs(nmag['mean'] - 1.0) <= 0.02 and nmag['min'] >= 0.95:
        print(f"  ✅ PASS: mean ≈ 1.0±0.02, min ≥ 0.95")
    else:
        print(f"  ❌ WARN: normals not well normalized (increase SDF resolution or check gradients)")
    
    # 5) Normal angle vs SDF
    nang_sdf = diag['normal_vs_sdf_deg']
    print(f"\nNormal vs ∇φ Angle:")
    print(f"  Mean:  {nang_sdf['mean']:.2f}°")
    print(f"  P95:   {nang_sdf['p95']:.2f}°")
    print(f"  Max:   {nang_sdf['max']:.2f}°")
    
    if nang_sdf['p95'] <= 5.0:
        print(f"  ✅ PASS: p95 ≤ 5°")
    elif nang_sdf['p95'] <= 12.0:
        print(f"  ⚠️  OK: p95 ≤ 12° (acceptable)")
    else:
        print(f"  ❌ WARN: p95 > 12° (normals inconsistent with SDF)")
    
    # 6) Normal change (if available)
    if diag['normal_in_to_out_deg'] is not None:
        nang_chg = diag['normal_in_to_out_deg']
        print(f"\nNormal Change (before → after):")
        print(f"  Mean:  {nang_chg['mean']:.2f}°")
        print(f"  P95:   {nang_chg['p95']:.2f}°")
        print(f"  Max:   {nang_chg['max']:.2f}°")
        
        if nang_chg['p95'] <= 12.0:
            print(f"  ✅ PASS: p95 ≤ 12°")
        else:
            print(f"  ❌ WARN: p95 > 12° (reduce λ_smooth or add reprojection)")
    
    # 7) Voxel duplication
    vdup = diag['voxel_dup']
    print(f"\nVoxel Duplication (points per voxel):")
    print(f"  Mean:  {vdup['mean']:.2f}")
    print(f"  P95:   {vdup['p95']:.2f}")
    print(f"  Max:   {vdup['max']}")
    
    if vdup['mean'] <= 1.6 and vdup['p95'] <= 3 and vdup['max'] <= 6:
        print(f"  ✅ PASS: good coverage uniformity")
    else:
        print(f"  ❌ WARN: clustering detected (increase voxel_scale or reduce anchor σ)")
    
    # 8) Boundary clamp
    bclamp = diag['boundary_clamp_ratio']
    print(f"\nBoundary Clamp Ratio: {bclamp*100:.2f}%")
    
    if bclamp <= 0.01:
        print(f"  ✅ PASS: ≤ 1%")
    else:
        print(f"  ⚠️  WARN: > 1% (increase bbox padding if not boundary mesh)")
    
    print("="*80 + "\n")


__all__ = ["diag_stage4", "print_stage4_diagnostics"]

