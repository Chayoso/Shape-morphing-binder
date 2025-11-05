# sampling/core/levelset_ops.py

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

EPS = 1e-8

# -------------------------
# 0) 공용: grid 변환 도우미
# -------------------------

def world_to_grid(points: torch.Tensor, bmin: torch.Tensor, bmax: torch.Tensor) -> torch.Tensor:
    return (points - bmin) / (bmax - bmin + EPS) * 2.0 - 1.0

def grid_grad_to_world(grad_grid: torch.Tensor, bmin: torch.Tensor, bmax: torch.Tensor) -> torch.Tensor:
    return grad_grid * (2.0 / (bmax - bmin + EPS)).to(grad_grid.dtype)

def phi5(phi_xyz: torch.Tensor) -> torch.Tensor:
    # (x,y,z) -> (D,H,W) for grid_sample
    return phi_xyz.permute(2, 1, 0).unsqueeze(0).unsqueeze(0).float()

# ------------------------------------------------------
# 1) φ=0 투영 (미분 제어 가능)
# ------------------------------------------------------

def project_zero(
    levelset, 
    points: torch.Tensor, 
    iters: int = 3, 
    lr: float = 0.8,
    require_grad_points: bool = False,
    detach_phi: bool = True,
    clamp_step_mult: float = 1.0,
    tol: float = 0.0,
    alpha_st: float = 0.05
) -> torch.Tensor:
    """
    φ=0으로 투영 (미분 제어 가능)
    
    Args:
        levelset: LevelSetGrid 인스턴스
        points: (N,3) world 좌표
        iters: 투영 반복 횟수
        lr: 학습률
        require_grad_points: True면 좌표에 대한 미분 보존 (morphing용)
        detach_phi: True면 φ에 대한 미분 차단 (안정성↑)
        clamp_step_mult: step 크기 제한 (Δx 단위, 0.6~1.0)
        tol: 조기 종료 threshold (0이면 비활성)
        alpha_st: ST-trick gradient mixing ratio (0.03~0.05)
    """
    # 🔥 Δx 계산 (step clamp 기준)
    dx = float(((levelset.bbox_max - levelset.bbox_min) / (levelset.phi.shape[0] - 1)).max().item())
    # φ detach 여부
    phi_vol = levelset.phi.detach() if detach_phi else levelset.phi
    sdf5 = phi5(phi_vol)
    
    if not require_grad_points:
        # Target mode: Simple projection without gradient
        pts = points.clone()
        max_step = clamp_step_mult * dx  # 🔥 Δx 기준
        
        with torch.no_grad():
            for _ in range(iters):
                g5 = world_to_grid(pts, levelset.bbox_min, levelset.bbox_max).view(1, -1, 1, 1, 3).float()
                g5 = g5.requires_grad_(True)
                
                with torch.enable_grad():
                    vals = F.grid_sample(sdf5, g5, mode='bilinear', padding_mode='border', align_corners=True).view(-1)
                    gg = torch.autograd.grad(vals.sum(), g5)[0].view(-1, 3)
                
                gw = grid_grad_to_world(gg, levelset.bbox_min, levelset.bbox_max)
                gn = gw.norm(dim=-1, keepdim=True).clamp_min(1e-4 * dx)  # 🔥 평평한 곳 난사 방지
                
                step = lr * (vals.unsqueeze(-1) / gn) * (gw / gn)
                
                # 🔥 (1) |φ|-기반 trust region: 너무 멀리 나가면 축소
                phi_abs = vals.abs().unsqueeze(-1)  # (N,1)
                shrink = (phi_abs / (0.5 * dx)).clamp_min(1.0)  # |φ|>0.5Δx면 축소 시작
                step = step / shrink
                
                # 🔥 (2) Δx-기준 하드 클램프
                step = step.clamp(-max_step, max_step)
                
                pts = torch.clamp(pts - step, min=levelset.bbox_min, max=levelset.bbox_max)
                
                if tol > 0 and vals.abs().max() < tol:
                    break
        
        return pts.detach()
    
    else:
        # 🔥 Morphing mode: Projection with gradient tracking + ST-trick
        pts = points.detach().clone()
        max_step = clamp_step_mult * dx  # 🔥 Δx 기준
        
        with torch.no_grad():
            for _ in range(iters):
                g5 = world_to_grid(pts, levelset.bbox_min, levelset.bbox_max).view(1, -1, 1, 1, 3).float()
                g5 = g5.requires_grad_(True)
                
                with torch.enable_grad():
                    vals = F.grid_sample(sdf5, g5, mode='bilinear', padding_mode='border', align_corners=True).view(-1)
                    gg = torch.autograd.grad(vals.sum(), g5)[0].view(-1, 3)
                
                gw = grid_grad_to_world(gg, levelset.bbox_min, levelset.bbox_max)
                gn = gw.norm(dim=-1, keepdim=True).clamp_min(1e-4 * dx)  # 🔥 평평한 곳 난사 방지
                
                step = lr * (vals.unsqueeze(-1) / gn) * (gw / gn)
                
                # 🔥 (1) |φ|-기반 trust region: 너무 멀리 나가면 축소
                phi_abs = vals.abs().unsqueeze(-1)  # (N,1)
                shrink = (phi_abs / (0.5 * dx)).clamp_min(1.0)  # |φ|>0.5Δx면 축소 시작
                step = step / shrink
                
                # 🔥 (2) Δx-기준 하드 클램프
                step = step.clamp(-max_step, max_step)
                
                pts = torch.clamp(pts - step, min=levelset.bbox_min, max=levelset.bbox_max)
                
                if tol > 0 and vals.abs().max() < tol:
                    break
        
        # 🔥 Straight-Through trick: Forward는 hard snap, Gradient만 alpha 비율로 흘림
        pts_hard = pts.detach()  # φ=0에 정확히 스냅된 좌표
        pts_st = pts_hard + alpha_st * (points - points.detach())  # ST estimator
        
        return torch.clamp(pts_st, min=levelset.bbox_min, max=levelset.bbox_max)

# ------------------------------------------------------
# 2) 접선-전용 스무딩 + 재투영 (미분 제어 가능)
# ------------------------------------------------------

def tangent_smooth_then_project(
    levelset,
    points: torch.Tensor,
    normals: torch.Tensor,
    k: int = 16,
    lambda_smooth: float = 0.3,
    iters: int = 2,
    knn=None,
    require_grad_points: bool = False,
    detach_phi: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    1) 라플라시안 스무딩(접선 성분만)
    2) φ=0 재투영
    3) 노멀 재계산
    """
    if not require_grad_points:
        pts = points.clone()
        nrm_norm = torch.norm(normals, dim=-1, keepdim=True)
        nrm = normals / torch.clamp(nrm_norm, min=1e-6)
        
        with torch.no_grad():
            for _ in range(iters):
                k_eff = min(k, pts.shape[0] - 1)
                idx, w = knn(pts, pts, k_eff)
                neigh = pts[idx]
                centroid = (w.unsqueeze(-1) * neigh).sum(dim=1)
                delta = centroid - pts
                delta_t = delta - (delta * nrm).sum(dim=-1, keepdim=True) * nrm
                pts = pts + lambda_smooth * delta_t
                pts = project_zero(levelset, pts, iters=2, lr=0.8, 
                                 require_grad_points=False, detach_phi=detach_phi)
                nrm = grad_normals(levelset, pts, require_grad_points=False, detach_phi=detach_phi)
        
        return pts.detach(), nrm.detach()
    
    else:
        pts = points.detach().clone()
        normals_d = normals.detach()
        nrm_norm = torch.norm(normals_d, dim=-1, keepdim=True)
        nrm = normals_d / torch.clamp(nrm_norm, min=1e-6)
        
        with torch.no_grad():
            for _ in range(iters):
                k_eff = min(k, pts.shape[0] - 1)
                idx, w = knn(pts, pts, k_eff)
                neigh = pts[idx]
                centroid = (w.unsqueeze(-1) * neigh).sum(dim=1)
                delta = centroid - pts
                delta_t = delta - (delta * nrm).sum(dim=-1, keepdim=True) * nrm
                pts = pts + lambda_smooth * delta_t
                pts = project_zero(levelset, pts, iters=2, lr=0.8, 
                                 require_grad_points=False, detach_phi=detach_phi)
                nrm = grad_normals(levelset, pts, require_grad_points=False, detach_phi=detach_phi)
        
        alpha = 0.03
        pts_final = (1.0 - alpha) * pts.detach() + alpha * points
        pts_final = torch.clamp(pts_final, min=levelset.bbox_min, max=levelset.bbox_max)
        
        return pts_final, nrm

# ------------------------------------------------------
# 3) ∇φ 기반 노멀 계산 (⚡ 최적화)
# ------------------------------------------------------

@torch.no_grad()
def _compute_normals_chunk(
    points_chunk: torch.Tensor,
    sdf5: torch.Tensor,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor
) -> torch.Tensor:
    """
    ⚡ 단일 청크에 대한 노멀 계산 (내부 헬퍼)
    
    최적화:
    - In-place 연산 최대화
    - 불필요한 view/reshape 제거
    - 타입 캐스팅 최소화
    """
    # Grid coordinates (single allocation)
    g5 = world_to_grid(points_chunk, bbox_min, bbox_max).view(1, -1, 1, 1, 3).float()
    g5.requires_grad_(True)
    
    # Compute gradient
    with torch.enable_grad():
        vals = F.grid_sample(sdf5, g5, mode='bilinear', padding_mode='border', align_corners=True)
        gg = torch.autograd.grad(vals.sum(), g5, create_graph=False)[0]
    
    # Convert to world space
    gw = grid_grad_to_world(gg.view(-1, 3), bbox_min, bbox_max)
    
    # ⚡ Fast normalize (rsqrt is faster than norm + div)
    # norm = sqrt(x^2 + y^2 + z^2), so 1/norm = rsqrt(...)
    norm_sq = (gw * gw).sum(dim=-1, keepdim=True)
    inv_norm = torch.rsqrt(norm_sq.clamp_min(1e-12))  # rsqrt = 1/sqrt
    
    return gw * inv_norm


def grad_normals(
    levelset, 
    points: torch.Tensor, 
    chunk: int = 65536,
    require_grad_points: bool = False,
    detach_phi: bool = True
) -> torch.Tensor:
    """
    ∇φ로 노멀 계산 (⚡ 최적화)
    
    최적화:
    - rsqrt 사용 (1/sqrt보다 2배 빠름)
    - 중복 코드 제거 (헬퍼 함수)
    - Pre-allocation
    - 불필요한 detach 제거
    """
    phi_vol = levelset.phi.detach() if detach_phi else levelset.phi
    sdf5 = phi5(phi_vol)
    
    N = points.shape[0]
    
    # ⚡ Pre-allocate output (avoid dynamic growth)
    out = torch.empty_like(points)
    
    # Process in chunks
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        p = points[i:end].detach() if require_grad_points else points[i:end]
        
        out[i:end] = _compute_normals_chunk(p, sdf5, levelset.bbox_min, levelset.bbox_max)
    
    # Return with proper gradient tracking
    return out.detach() if not require_grad_points else out


# ------------------------------------------------------
# 3.5) Local Majority Alignment (⚡ 최적화)
# ------------------------------------------------------

@torch.no_grad()
def align_normals_local_majority(
    normals: torch.Tensor,
    points: torch.Tensor,
    knn,
    k: int = 16,
    verbose: bool = False
) -> torch.Tensor:
    """
    Local k-NN majority voting (⚡ 최적화)
    
    최적화:
    - In-place 연산
    - Boolean 인덱싱 대신 masked assignment
    - 불필요한 clone 제거
    """
    N = points.shape[0]
    if N < 2:
        return normals
    
    k_actual = min(k, N - 1)
    if k_actual < 1:
        return normals
    
    # KNN search
    idx_nn, _ = knn(points, points, k_actual)
    
    # ⚡ Vectorized dot product
    normals_nn = normals[idx_nn]
    dot_prods = (normals.unsqueeze(1) * normals_nn).sum(dim=-1)
    
    # ⚡ Count via boolean ops (faster than separate sums)
    agree = dot_prods > 0
    agree_count = agree.sum(dim=1)
    disagree_count = k_actual - agree_count
    
    # Flip mask
    flip_mask = disagree_count > agree_count
    
    if not flip_mask.any():
        return normals  # ⚡ Early exit (no cloning needed)
    
    # ⚡ In-place flip (clone only if needed)
    normals_aligned = normals.clone()
    normals_aligned[flip_mask] = -normals_aligned[flip_mask]
    
    if verbose:
        num_flipped = flip_mask.sum().item()
        pct = 100.0 * num_flipped / N
        print(f"  [Local Majority] Flipped {num_flipped:,}/{N:,} ({pct:.1f}%) "
              f"to match k={k_actual} neighbors")
    
    return normals_aligned


# ------------------------------------------------------
# 4) 간이 복셀-쿼터 억제 (과밀 버킷 드롭)
# ------------------------------------------------------

@torch.no_grad()
def voxel_quota_drop(points: torch.Tensor, max_per_voxel: int, voxel_size: float) -> torch.Tensor:
    """
    ⚡ Fully vectorized voxel quota drop (no Python loops).
    
    과밀 복셀 드롭용 인덱스 마스크 반환. ~10× faster for 1M points.
    """
    # 1) Voxel key calculation (spatial hash)
    vox = (points / max(voxel_size, 1e-6)).floor().to(torch.int64)
    key = (vox[:, 0] * 73856093) ^ (vox[:, 1] * 19349663) ^ (vox[:, 2] * 83492791)
    
    # 2) Random shuffle + sort by key
    perm = torch.randperm(points.shape[0], device=points.device)
    key_p = key[perm]
    idx_sorted = torch.argsort(key_p)
    key_s = key_p[idx_sorted]
    
    # 3) Group boundaries using unique_consecutive
    uniq, counts = torch.unique_consecutive(key_s, return_counts=True)
    group_first = torch.cumsum(torch.cat([counts.new_zeros(1), counts[:-1]]), dim=0)
    
    # 4) Each element's group id (inverse) via searchsorted
    inv = torch.searchsorted(group_first.cumsum(0), 
                             torch.arange(key_s.numel(), device=key_s.device), 
                             right=True) - 1
    pos_in_group = torch.arange(key_s.numel(), device=key_s.device) - group_first[inv]
    
    # 5) Keep first max_per_voxel in each group
    keep_sorted = pos_in_group < max_per_voxel
    keep_perm = torch.zeros_like(keep_sorted)
    keep_perm[idx_sorted] = keep_sorted
    
    # 6) Restore original order
    mask = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    mask[perm] = keep_perm
    
    return mask

# ------------------------------------------------------
# 5) 진단: φ 잔차/노멀/중복
# ------------------------------------------------------

@torch.no_grad()
def quick_diagnostics(levelset, anchors: torch.Tensor, normals: torch.Tensor, spacing: torch.Tensor) -> Dict:
    sdf5 = phi5(levelset.phi)
    g5 = world_to_grid(anchors, levelset.bbox_min, levelset.bbox_max).view(1, -1, 1, 1, 3)
    phi_vals = F.grid_sample(sdf5, g5, mode='bilinear', padding_mode='border', align_corners=True).view(-1)
    abs_phi = phi_vals.abs()
    nm = normals.norm(dim=-1)
    
    # voxel 중복
    vox_size = float(spacing.mean().item()) * 0.9
    vox = (anchors / max(vox_size, 1e-6)).floor().long()
    key = (vox[:, 0] * 73856093 ^ vox[:, 1] * 19349663 ^ vox[:, 2] * 83492791) & 0x7fffffff
    _, counts = key.unique(return_counts=True)
    
    return {
        "phi_mean": float(abs_phi.mean()),
        "phi_p95":  float(abs_phi.quantile(0.95)),
        "phi_max":  float(abs_phi.max()),
        "n_mag_mean": float(nm.mean()),
        "n_mag_min":  float(nm.min()),
        "voxel_count_mean": float(counts.float().mean()),
        "voxel_count_p95":  float(counts.float().quantile(0.95)),
        "voxel_count_max":  float(counts.max()),
    }


__all__ = [
    "world_to_grid",
    "grid_grad_to_world",
    "phi5",
    "project_zero",
    "tangent_smooth_then_project",
    "grad_normals",
    "align_normals_local_majority",
    "voxel_quota_drop",
    "quick_diagnostics",
]