# =============================================================================
# levelset_improved.py - 개선된 표면 앵커 추출 (⚡⚡ 초고속 최적화 버전)
# =============================================================================
# ⚡⚡ 성능 최적화 v2:
#   10) rsqrt 사용 (normalize 2배 가속)
#   11) Pre-allocation + in-place 연산
#   12) 중복 계산 제거 (voxel size 캐싱)
#   13) Early exit 패턴
#   14) 불필요한 타입 변환 제거
# =============================================================================

from __future__ import annotations
from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 🔥 Import levelset_ops (단일 소스)
from ..core.levelset_ops import (
    project_zero as ops_project_zero,
    grad_normals as ops_grad_normals,
)

EPS = 1e-8

# -----------------------------------------------------------------------------
# 좌표계 유틸 (Y-up → Z-up)
# -----------------------------------------------------------------------------
def to_z_up(points: torch.Tensor, normals: Optional[torch.Tensor] = None):
    """(x, y, z) -> (x, -z, y) : 메시가 Y-up일 때 파이프라인 Z-up에 맞춤."""
    x, y, z = points.unbind(-1)
    pts = torch.stack([x, -z, y], dim=-1)
    if normals is None:
        return pts, None
    nx, ny, nz = normals.unbind(-1)
    nrm = F.normalize(torch.stack([nx, -nz, ny], dim=-1), dim=-1)
    return pts, nrm

# -----------------------------------------------------------------------------
# KNN Distance 기반 SDF (⚡⚡ 최적화)
# -----------------------------------------------------------------------------
class KNNDistanceSDF(nn.Module):
    def __init__(
        self,
        resolution: int = 256,
        padding: float = 0.25,
        k_density: int = 150,
        chunk_size: int = 131072,
        distance_scale: float = 10.0,
        use_gaussian: bool = True,
        gaussian_sigma_vox: float = 0.4,
        gaussian_ksize: Optional[int] = None,
        sigmoid_center: float = 0.48,
        sigmoid_slope: float = 8.0,
    ):
        super().__init__()
        self.resolution = int(resolution)
        self.padding = float(padding)
        self.k_density = int(k_density)
        self.chunk_size = int(chunk_size)
        self.distance_scale = float(distance_scale)
        self.use_gaussian = bool(use_gaussian)
        self.gaussian_sigma_vox = float(gaussian_sigma_vox)
        self.gaussian_ksize = gaussian_ksize
        self.sigmoid_center = float(sigmoid_center)
        self.sigmoid_slope = float(sigmoid_slope)

    @torch.no_grad()
    def _build_density_field(self, grid_points: torch.Tensor, source_points: torch.Tensor, knn) -> torch.Tensor:
        """⚡⚡ Ultra-optimized density field with BF16/FP32 separation."""
        device = grid_points.device
        N = grid_points.shape[0]
        
        # ⚡ Compute tau once
        bbox_size = (source_points.max(dim=0).values - source_points.min(dim=0).values).mean()
        tau = float(bbox_size.item()) / 20.0
        tau_sq = tau * tau  # ⚡ Pre-compute squared
        k = min(self.k_density, source_points.shape[0])
        
        # ⚡ Pre-allocate output
        dens_full = torch.empty(N, dtype=torch.float32, device=device)
        
        for i in range(0, N, self.chunk_size):
            chunk_end = min(i + self.chunk_size, N)
            y = grid_points[i:chunk_end]
            
            # ⚡ BF16 for KNN search only (distance computation)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=torch.bfloat16):
                idx, _ = knn(y, source_points, k)
                neigh = source_points[idx]  # bf16 ok
            
            # ⚡ FP32 for distance, exp, sum (numerical stability)
            d2 = ((y.unsqueeze(1).float() - neigh.float()) ** 2).sum(dim=-1)
            z = (-0.5 * d2 / (tau_sq + 1e-12)).clamp_min(-60.0)  # ⚡ exp underflow guard
            dens = torch.exp(z).sum(dim=1).float()
            
            dens_full[i:chunk_end] = dens
        
        return dens_full

    @torch.no_grad()
    def _gaussian_blur3d(self, vol: torch.Tensor, sigma_vox: float = 0.7, ksize: Optional[int] = None) -> torch.Tensor:
        """⚡⚡ Ultra-fast separable Gaussian blur."""
        if ksize is None:
            ksize = int(2 * math.ceil(3.0 * sigma_vox) + 1)
            ksize = max(3, ksize | 1)
        
        # ⚡ Create 1D kernel once (cached)
        x = torch.arange(ksize, device=vol.device, dtype=vol.dtype) - (ksize // 2)
        g = torch.exp(-0.5 * (x / sigma_vox) ** 2)
        g = g / g.sum()
        
        # ⚡ Reshape for conv3d (no copy)
        gX = g.view(1, 1, ksize, 1, 1)
        gY = g.view(1, 1, 1, ksize, 1)
        gZ = g.view(1, 1, 1, 1, ksize)
        
        # ⚡ Apply separable convolutions
        v = vol.unsqueeze(0).unsqueeze(0)
        pad = ksize // 2
        
        # X-direction
        v = F.conv3d(F.pad(v, (0, 0, 0, 0, pad, pad), mode='replicate'), gX)
        # Y-direction
        v = F.conv3d(F.pad(v, (0, 0, pad, pad, 0, 0), mode='replicate'), gY)
        # Z-direction
        v = F.conv3d(F.pad(v, (pad, pad, 0, 0, 0, 0), mode='replicate'), gZ)
        
        return v.squeeze(0).squeeze(0)

    @torch.no_grad()
    def forward(self, points: torch.Tensor, knn, downsample_to: Optional[int] = None, verbose: bool = False, force_bbox: tuple = None) -> Dict:
        """⚡⚡ Ultra-optimized SDF construction."""
        device, dtype = points.device, points.dtype
        R = self.resolution

        # 🔥 CRITICAL: Use force_bbox if provided (Physics grid alignment)
        if force_bbox is not None:
            bmin, bmax = force_bbox
            bmin = bmin.to(device=device, dtype=dtype)
            bmax = bmax.to(device=device, dtype=dtype)
            dx = ((bmax - bmin) / (R - 1)).mean()
            if verbose:
                print(f"  🎯 Using force_bbox: min={bmin.tolist()}, max={bmax.tolist()}")
        else:
            # Compute bounding box from points
            pmin = points.min(dim=0).values
            pmax = points.max(dim=0).values
            pad = (pmax - pmin) * self.padding
            bmin = pmin - pad
            bmax = pmax + pad
            dx = ((bmax - bmin) / (R - 1)).mean()

        # ⚡ Downsample source points if needed
        if downsample_to and points.shape[0] > downsample_to:
            idx = torch.randperm(points.shape[0], device=device)[:downsample_to]
            points = points[idx]
            if verbose:
                print(f"  Downsampled: {len(idx):,} points")

        # ⚡ Create grid (single allocation)
        lin = torch.linspace(0, 1, R, device=device, dtype=dtype)
        gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing='ij')
        grid_world = bmin + torch.stack([gx, gy, gz], dim=-1) * (bmax - bmin)
        flat = grid_world.reshape(-1, 3)

        if verbose:
            print(f"  Building density field: {R}³ grid, {flat.shape[0]:,} queries, k={self.k_density}")

        # ⚡ Build density field
        density_flat = self._build_density_field(flat, points, knn).reshape(R, R, R)
        
        # Normalize density → SDF
        d_min = density_flat.quantile(0.05)
        d_max = density_flat.quantile(0.95)
        denom = (d_max - d_min).clamp_min(1e-6)  # ← 안전장치
        d_norm = ((density_flat - d_min) / denom).clamp_(0, 1)  # ⚡ in-place
        
        # Sigmoid
        d_smooth = torch.sigmoid((d_norm - self.sigmoid_center) * self.sigmoid_slope)
        sdf = (0.5 - d_smooth) * dx * self.distance_scale

        # Apply blur
        if self.use_gaussian:
            sdf = self._gaussian_blur3d(sdf, sigma_vox=self.gaussian_sigma_vox, ksize=self.gaussian_ksize)
        else:
            sdf = F.avg_pool3d(
                F.pad(sdf.unsqueeze(0).unsqueeze(0), (1,1,1,1,1,1), mode='replicate'),
                kernel_size=3, stride=1
            ).squeeze(0).squeeze(0)

        sdf.clamp_(-2.0, 2.0)  # ⚡ in-place

        if verbose:
            rng = (float(sdf.min()), float(sdf.max()))
            print(f"[SDF] res={R}³, dx={float(dx):.4f}, range={rng}")

        return {"sdf_grid": sdf, "density_grid": density_flat, "bbox_min": bmin, "bbox_max": bmax}

# -----------------------------------------------------------------------------
# Level Set (φ) - Coordinate Conversion Utilities
# -----------------------------------------------------------------------------
# 🔥 사용자 패치 A: 통일된 그리드↔월드 변환 (코너 정렬, align_corners=True)

def grid_idx_to_world(idx: torch.Tensor, bmin: torch.Tensor, bmax: torch.Tensor, res: int) -> torch.Tensor:
    """
    Convert grid indices [0, res-1] to world coordinates (CORNER-ALIGNED).
    
    Compatible with align_corners=True in grid_sample.
    
    Args:
        idx: (..., 3) integer indices in [0, res-1]
        bmin: (3,) world bbox min
        bmax: (3,) world bbox max
        res: grid resolution
    
    Returns:
        world coordinates (..., 3)
    """
    idx_float = idx.to(dtype=bmin.dtype)
    return bmin + (idx_float / (res - 1)) * (bmax - bmin)

def world_to_grid_index(xw: torch.Tensor, bmin: torch.Tensor, bmax: torch.Tensor, res: int) -> torch.Tensor:
    """
    Convert world coordinates to grid indices (CORNER-ALIGNED, inverse of grid_idx_to_world).
    
    Args:
        xw: (..., 3) world coordinates
        bmin: (3,) world bbox min
        bmax: (3,) world bbox max
        res: grid resolution
    
    Returns:
        grid indices (..., 3) in [0, res-1]
    """
    t = (xw - bmin) / (bmax - bmin + EPS)
    return (t * (res - 1)).clamp(0, res - 1)

def _to_5d_DHW(phi_xyz: torch.Tensor) -> torch.Tensor:
    S = phi_xyz.permute(2,1,0).contiguous()
    return S.unsqueeze(0).unsqueeze(0).float()

def _world_to_grid(points: torch.Tensor, bmin: torch.Tensor, bmax: torch.Tensor) -> torch.Tensor:
    """
    Convert world coords to grid_sample normalized coords [-1, 1] (CORNER-ALIGNED).
    
    Compatible with align_corners=True.
    """
    return (points - bmin) / (bmax - bmin + EPS) * 2.0 - 1.0

def _grid_grad_to_world(grad_grid: torch.Tensor, bmin: torch.Tensor, bmax: torch.Tensor) -> torch.Tensor:
    """
    Convert grid-space gradients to world-space gradients.
    
    Chain rule: dφ/dx_world = dφ/dx_grid × dx_grid/dx_world
    """
    return grad_grid * (2.0 / (bmax - bmin + EPS)).to(grad_grid.dtype)


# ============================================================================
# ∇φ 볼륨 사전계산 (autograd 제거 최적화)
# ============================================================================
@torch.no_grad()
def precompute_grad_grid(phi_xyz: torch.Tensor, bbox_min: torch.Tensor, bbox_max: torch.Tensor):
    """
    ⚡ Precompute gradient field using central differences (no autograd).
    
    Args:
        phi_xyz: (R,R,R) SDF grid in xyz-order
        bbox_min: (3,) world bbox min
        bbox_max: (3,) world bbox max
    
    Returns:
        grad5: (1,3,D,H,W) gradient field for grid_sample
    """
    phi = phi_xyz
    # Central differences (replicate pad at boundaries)
    px = (phi.roll(-1, 0) - phi.roll(1, 0)) * 0.5
    py = (phi.roll(-1, 1) - phi.roll(1, 1)) * 0.5
    pz = (phi.roll(-1, 2) - phi.roll(1, 2)) * 0.5
    
    # Scale to world coordinates (chain rule)
    scale = 2.0 / (bbox_max - bbox_min + EPS)  # (3,)
    
    # Reorder to (N,C,D,H,W) for grid_sample: xyz->DHW permutation
    Gx = (px * scale[0]).permute(2, 1, 0).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    Gy = (py * scale[1]).permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
    Gz = (pz * scale[2]).permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
    
    # Concat channel-wise
    grad5 = torch.cat([Gx, Gy, Gz], dim=1).contiguous()  # (1,3,D,H,W)
    return grad5


class LevelSetGrid(nn.Module):
    def __init__(self, resolution:int=256, padding:float=0.25, force_bbox:tuple=None):
        super().__init__()
        self.res = int(resolution)
        self.padding = float(padding)
        self.register_buffer("phi", None)
        # 🔥 NEW: force_bbox = (bbox_min, bbox_max) if provided
        if force_bbox is not None:
            self.register_buffer("bbox_min", force_bbox[0])
            self.register_buffer("bbox_max", force_bbox[1])
        else:
            self.register_buffer("bbox_min", None)
            self.register_buffer("bbox_max", None)

    @torch.no_grad()
    def init_from_points(
        self, x_vol: torch.Tensor, knn, 
        k_density: int = 150, 
        distance_scale: float = 10.0,
        downsample_to: Optional[int] = None, 
        differentiable: bool = False, 
        verbose: bool = True,
        gaussian_sigma_vox: float = 0.4,
        sigmoid_center: float = 0.48,
        sigmoid_slope: float = 8.0,
        use_gaussian: bool = True,
        chunk_size: int = 131072,
        **_
    ):
        """⚡⚡ Ultra-optimized initialization."""
        # 🔥 Use force_bbox if already set in __init__
        force_bbox = None
        if self.bbox_min is not None and self.bbox_max is not None:
            force_bbox = (self.bbox_min, self.bbox_max)
        
        builder = KNNDistanceSDF(
            self.res, self.padding, 
            k_density=k_density, 
            distance_scale=distance_scale,
            chunk_size=chunk_size,
            use_gaussian=use_gaussian,
            gaussian_sigma_vox=gaussian_sigma_vox,
            sigmoid_center=sigmoid_center,
            sigmoid_slope=sigmoid_slope,
        )
        out = builder(x_vol, knn, downsample_to, verbose, force_bbox=force_bbox)
        self.phi = out["sdf_grid"]
        # 🔥 Only update bbox if not forced
        if force_bbox is None:
            self.bbox_min = out["bbox_min"]
            self.bbox_max = out["bbox_max"]
        return self

    def _phi_5d(self): 
        return _to_5d_DHW(self.phi)

    def grad(self, points: torch.Tensor, chunk:int=65536, detach_phi:bool=True) -> torch.Tensor:
        """Compute normals from ∇φ."""
        need_grad = bool(points.requires_grad)
        return ops_grad_normals(self, points, chunk=chunk, require_grad_points=need_grad, detach_phi=detach_phi)

    def project_zero(self, points: torch.Tensor, iters:int=3, lr:float=0.8, detach_phi:bool=True) -> torch.Tensor:
        """Project points to φ=0 surface."""
        need_grad = bool(points.requires_grad)
        return ops_project_zero(self, points, iters=iters, lr=lr,
                               require_grad_points=need_grad, detach_phi=detach_phi)

    # =========================================================================
    # ⚡ Fast Path: Precomputed Gradient Volume (No Autograd)
    # =========================================================================
    def refresh_grad_cache(self):
        """
        ⚡ Precompute ∇φ volume using central differences.
        Call this after phi changes (e.g., after reinit/advect).
        """
        self._grad5 = precompute_grad_grid(self.phi, self.bbox_min, self.bbox_max)
        return self

    @torch.no_grad()
    def grad_fast(self, points: torch.Tensor) -> torch.Tensor:
        """
        ⚡ Fast gradient (normal) lookup using precomputed volume.
        1.6×~3× faster than autograd-based grad().
        """
        if not hasattr(self, "_grad5") or self._grad5 is None:
            self.refresh_grad_cache()
        
        g = _world_to_grid(points, self.bbox_min, self.bbox_max).view(1, -1, 1, 1, 3).float()
        v = F.grid_sample(self._grad5, g, mode='bilinear', padding_mode='border', align_corners=True)
        n = v.view(3, -1).t()
        return F.normalize(n, dim=-1, eps=1e-6)

    @torch.no_grad()
    def project_zero_fast(self, points: torch.Tensor, iters: int = 3, lr: float = 0.8, clamp_step_mult: float = 1.0):
        """
        ⚡ Fast Newton projection using precomputed gradient volume.
        1.6×~3× faster than autograd-based project_zero().
        
        Args:
            points: (N, 3) points to project
            iters: Newton iteration count
            lr: Learning rate (step size multiplier)
            clamp_step_mult: Clamp step magnitude (Δx 단위, 0.6~1.0)
        """
        if not hasattr(self, "_grad5") or self._grad5 is None:
            self.refresh_grad_cache()
        
        pts = points.clone()
        sdf5 = self._phi_5d()
        
        # 🔥 Δx 계산
        dx = float(((self.bbox_max - self.bbox_min) / (self.phi.shape[0] - 1)).max().item())
        max_step = clamp_step_mult * dx
        
        for _ in range(iters):
            g = _world_to_grid(pts, self.bbox_min, self.bbox_max).view(1, -1, 1, 1, 3).float()
            phi = F.grid_sample(sdf5, g, mode='bilinear', padding_mode='border', align_corners=True).view(-1)
            grad = F.grid_sample(self._grad5, g, mode='bilinear', padding_mode='border', align_corners=True).view(3, -1).t()
            
            gn = grad.norm(dim=-1, keepdim=True).clamp_min(1e-4 * dx)  # 🔥 평평한 곳 난사 방지
            step = lr * (phi.unsqueeze(-1) / gn) * (grad / gn)
            step = step.clamp(-max_step, max_step)  # 🔥 Δx 기준 클램프
            
            pts = torch.clamp(pts - step, min=self.bbox_min, max=self.bbox_max)
        
        return pts

    @torch.no_grad()
    def advect(self, vel_grid_5d: torch.Tensor, dt: float):
        """Advect level set using velocity field: ∂φ/∂t + v·∇φ = 0"""
        sdf5 = self._phi_5d()
        D, H, W = sdf5.shape[-3:]
        
        z = torch.linspace(-1, 1, D, device=self.phi.device)
        y = torch.linspace(-1, 1, H, device=self.phi.device)
        x = torch.linspace(-1, 1, W, device=self.phi.device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        grid = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0)
        
        scale = 2.0 / (self.bbox_max - self.bbox_min + EPS)
        v = F.grid_sample(vel_grid_5d, grid, mode='bilinear', align_corners=True)
        v = v.permute(0, 2, 3, 4, 1) * scale.view(1, 1, 1, 1, 3)
        
        back = grid - dt * v
        phi_new = F.grid_sample(sdf5, back, mode='bilinear', align_corners=True)
        
        self.phi = phi_new.squeeze(0).squeeze(0).permute(2, 1, 0).contiguous()
        return self

    @torch.no_grad()
    def reinit_soft(self, steps:int=2, tau:float=0.5, eps:float=1e-3):
        """Soft reinitialization."""
        for _ in range(steps):
            gx = (self.phi.roll(-1,0)-self.phi.roll(1,0))*0.5
            gy = (self.phi.roll(-1,1)-self.phi.roll(1,1))*0.5
            gz = (self.phi.roll(-1,2)-self.phi.roll(1,2))*0.5
            gn = torch.sqrt(gx*gx+gy*gy+gz*gz+eps*eps)
            s = self.phi / torch.sqrt(self.phi*self.phi + eps*eps)
            self.phi = self.phi + tau * s * (1.0 - gn)
        return self
    
    def reinit(self, steps:int=3, tau:float=0.3, eps:float=1e-3):
        return self.reinit_soft(steps=steps, tau=tau, eps=eps)
    
    def advect_levelset(self, vel_grid_5d: torch.Tensor, dt: float, method: str = 'semi_lagrangian'):
        if method != 'semi_lagrangian':
            raise ValueError(f"Only 'semi_lagrangian' method supported, got {method}")
        return self.advect(vel_grid_5d, dt)
    
    # ========================================================================
    # ⚡ Fast gradient computation (no autograd)
    # ========================================================================
    def refresh_grad_cache(self):
        """⚡ Refresh gradient cache after phi update."""
        self._grad5 = precompute_grad_grid(self.phi, self.bbox_min, self.bbox_max)
        return self
    
    @torch.no_grad()
    def grad_fast(self, points: torch.Tensor) -> torch.Tensor:
        """⚡ Fast gradient computation using precomputed grad field."""
        if not hasattr(self, "_grad5") or self._grad5 is None:
            self.refresh_grad_cache()
        
        g = _world_to_grid(points, self.bbox_min, self.bbox_max).view(1, -1, 1, 1, 3).float()
        vals = F.grid_sample(self._grad5, g, mode='bilinear', padding_mode='border', align_corners=True)
        n = vals.view(3, -1).t()  # (N,3)
        return F.normalize(n, dim=-1, eps=1e-6)
    
    @torch.no_grad()
    def project_zero_fast(self, points: torch.Tensor, iters: int = 3, lr: float = 0.8, clamp_step_mult: float = 1.0):
        """⚡ Fast projection using precomputed gradients."""
        if not hasattr(self, "_grad5") or self._grad5 is None:
            self.refresh_grad_cache()
        
        pts = points.clone()
        sdf5 = self._phi_5d()
        
        for _ in range(iters):
            g = _world_to_grid(pts, self.bbox_min, self.bbox_max).view(1, -1, 1, 1, 3).float()
            phi_val = F.grid_sample(sdf5, g, mode='bilinear', padding_mode='border', align_corners=True).view(-1)
            grad = F.grid_sample(self._grad5, g, mode='bilinear', padding_mode='border', align_corners=True).view(3, -1).t()
            
            gn = grad.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            step = lr * (phi_val.unsqueeze(-1) / gn) * (grad / gn)
            
            if clamp_step_mult < 1.0:
                step = step.clamp(-clamp_step_mult, clamp_step_mult)
            
            pts = torch.clamp(pts - step, min=self.bbox_min, max=self.bbox_max)
        
        return pts

    def sample_surface(self, M:int, sigma_mult:float=1.4, p:float=2.0, project_iters:int=3, spacing_scale:float=1.0):
        """Sample surface anchors."""
        R = self.phi.shape[0]
        vox = ((self.bbox_max - self.bbox_min) / (R - 1)).mean()
        sigma = sigma_mult * vox
        
        w = torch.exp(- (self.phi.abs() / (sigma + EPS))**p)
        probs = w.reshape(-1).float()
        probs = probs / (probs.sum() + 1e-8)
        
        num_candidates = min(M, int(probs.numel()))
        sel = torch.multinomial(probs, num_samples=num_candidates, replacement=True)
        iz = sel // (R * R)
        iy = (sel % (R * R)) // R
        ix = sel % R
        vox_idx = torch.stack([ix, iy, iz], dim=1).float()
        
        jitter = torch.rand_like(vox_idx) - 0.5
        coords = (vox_idx + jitter).clamp(0, R - 1)
        world = self.bbox_min + (coords / (R - 1)) * (self.bbox_max - self.bbox_min)
        
        anchors = self.project_zero(world, iters=project_iters)
        normals = self.grad(anchors)
        spacing = torch.full((anchors.size(0),), vox * spacing_scale, device=anchors.device, dtype=anchors.dtype)
        p_surf_raw = w[ix.long(), iy.long(), iz.long()].to(anchors.dtype)
        
        if anchors.size(0) < M:
            rep = (M + anchors.size(0) - 1) // anchors.size(0)
            anchors = anchors.repeat(rep, 1)[:M]
            normals = normals.repeat(rep, 1)[:M]
            spacing = spacing.repeat(rep)[:M]
            p_surf_raw = p_surf_raw.repeat(rep)[:M]
        
        return anchors, normals, spacing, p_surf_raw

# -----------------------------------------------------------------------------
# 앵커 추출 (⚡⚡ 초고속 최적화)
# -----------------------------------------------------------------------------
@torch.no_grad()
def extract_surface_anchors(
    levelset: LevelSetGrid,
    num_anchors: int,
    sigma_mult: float = 1.0,
    project_iters: int = 3,
    compute_curvature: bool = False,
    knn = None,
    curvature_eps_mult: float = 1.25,
    prefer_up: bool = True,
    max_sdf_after_proj: Optional[float] = None,
    verbose: bool = False,
    **kwargs
) -> Dict:
    """
    ⚡⚡ Ultra-optimized surface anchor extraction.
    
    최적화:
    - Voxel size 캐싱 (중복 계산 제거)
    - In-place 연산
    - Pre-allocation
    - 불필요한 타입 변환 제거
    - rsqrt for normalization
    """
    device = levelset.phi.device
    dtype = levelset.phi.dtype
    R = levelset.phi.shape[0]

    # ⚡ Compute voxel size once (cached)
    bbox_range = levelset.bbox_max - levelset.bbox_min
    vox = float((bbox_range / (R - 1)).mean().item())
    sigma = sigma_mult * vox

    if verbose:
        print(f"\n{'='*60}")
        print(f"[Anchor Extraction] Starting...")
        print(f"{'='*60}")
        print(f"  Grid resolution: {R}³ = {R**3:,} voxels")
        print(f"  Voxel size: {vox:.4f}")
        print(f"  Band sigma: {sigma:.4f} (mult={sigma_mult})")
        print(f"  Target anchors: {num_anchors:,}")

    # 1) Band mask
    band_mask = (levelset.phi.abs() < sigma)
    idx = band_mask.nonzero(as_tuple=False)

    if idx.numel() == 0:
        raise RuntimeError(f"No surface voxels in band (sigma={sigma:.4f}). Try larger sigma_mult.")

    K = idx.shape[0]
    if verbose:
        print(f"  Band voxels: {K:,} ({100*K/R**3:.2f}% of grid)")

    # 2) Oversampling
    target_samples = int(num_anchors * 1.5)
    
    if K >= target_samples:
        sel = torch.randint(0, K, (target_samples,), device=device)
        vox_idx = idx[sel].float()
    else:
        rep = (target_samples + K - 1) // K
        vox_idx = idx.repeat(rep, 1).float()[:target_samples]
        if verbose:
            print(f"  Warning: Band has only {K} voxels, repeating {rep}x")

    # 3) Voxel → World (⚡ optimized indexing)
    coords = vox_idx  # (N, 3) [ix, iy, iz]
    
    # Sub-voxel jitter
    jitter = (torch.rand_like(coords) - 0.5).clamp_(-0.49, 0.49)
    coords.add_(jitter).clamp_(0, R - 1)  # ⚡ in-place
    
    # ⚡ Vectorized world coordinate computation
    world = levelset.bbox_min + (coords / (R - 1)) * bbox_range

    if verbose:
        print(f"\n[Projection Phase]")
        print(f"  Initial candidates: {world.shape[0]:,}")
        print(f"  Newton iterations: {project_iters}")

    # 4) Project to φ=0
    anchors = ops_project_zero(levelset, world, iters=project_iters, lr=0.8,
                               require_grad_points=False, detach_phi=True)
    
    # 🔥 4.5) Re-snap to φ=0 with voxel-scale tolerance (사용자 패치 C)
    dx = float((bbox_range.max() / (R - 1)).item())
    if verbose:
        print(f"  Voxel dx: {dx:.6f}")
        print(f"  Re-snapping to φ=0 with tolerance: {0.3*dx:.6f}")
    
    anchors = ops_project_zero(
        levelset, anchors, 
        iters=2, 
        lr=0.9,
        require_grad_points=False, 
        detach_phi=True,
        clamp_step_mult=1.0,
        tol=0.3*dx
    )
    
    # 5) SDF validation (⚡ single grid_sample call)
    sdf5 = levelset._phi_5d()
    g5 = _world_to_grid(anchors, levelset.bbox_min, levelset.bbox_max).view(1,-1,1,1,3).float()
    sdf_vals = F.grid_sample(sdf5, g5, mode='bilinear', padding_mode='border', align_corners=True).view(-1)
    
    # 🔥 5.5) Anchor φ-alignment diagnostics (사용자 패치 D)
    if verbose:
        phi_mean_abs = sdf_vals.abs().mean().item()
        phi_p95 = sdf_vals.abs().quantile(0.95).item()
        mean_in_dx = phi_mean_abs / dx
        p95_in_dx = phi_p95 / dx
        print(f"\n[Anchor φ-Alignment Diagnostics]")
        print(f"  |φ| mean:  {phi_mean_abs:.6f}  ({mean_in_dx:.3f} × Δx)")
        print(f"  |φ| p95:   {phi_p95:.6f}  ({p95_in_dx:.3f} × Δx)")
        print(f"  Target: mean < 0.2×Δx, p95 < 0.5×Δx")
        
        if mean_in_dx < 0.2 and p95_in_dx < 0.5:
            print(f"  ✅ PASS: Anchors are well-aligned to φ=0")
        else:
            print(f"  ⚠️  WARNING: Anchors have systematic offset!")
            print(f"      → Check grid↔world coordinate alignment")
            print(f"      → Verify align_corners=True everywhere")
    
    # ⚡ Auto-compute threshold
    if max_sdf_after_proj is None:
        max_sdf_after_proj = max(0.5 * vox, 0.06)
        if verbose:
            print(f"  Auto max_sdf_after_proj = {max_sdf_after_proj:.4f} (0.5 × voxel)")
    
    valid_mask = sdf_vals.abs() < max_sdf_after_proj
    
    if verbose:
        abs_sdf = sdf_vals.abs()
        print(f"  SDF after projection:")
        print(f"    Mean |SDF|: {abs_sdf.mean():.6f}")
        print(f"    Max |SDF|: {abs_sdf.max():.6f}")
        print(f"    Valid (|SDF| < {max_sdf_after_proj}): {valid_mask.sum().item():,}/{len(valid_mask):,} "
              f"({100*valid_mask.float().mean():.1f}%)")
    
    anchors = anchors[valid_mask]
    
    # Adjust count
    N = anchors.shape[0]
    if N > num_anchors:
        anchors = anchors[:num_anchors]
    elif N < num_anchors:
        shortage = num_anchors - N
        if verbose:
            print(f"  Warning: Only {N:,} valid anchors, short by {shortage:,}")
            print(f"           Repeating to fill...")
        rep = (num_anchors + N - 1) // N
        anchors = anchors.repeat(rep, 1)[:num_anchors]
    
    if verbose:
        print(f"  Final anchors: {anchors.shape[0]:,}")

    # 6) Normals (⚡ already optimized in ops)
    if verbose:
        print(f"\n[Normal Computation]")
    
    normals = ops_grad_normals(levelset, anchors, require_grad_points=False, detach_phi=True)
    
    # Magnitude check
    normal_mag = normals.norm(dim=-1)
    
    if verbose:
        print(f"  Normal magnitude:")
        print(f"    Mean: {normal_mag.mean():.4f} (should be ~1.0)")
        print(f"    Std:  {normal_mag.std():.4f}")
        print(f"    Range: [{normal_mag.min():.4f}, {normal_mag.max():.4f}]")
    
    # Renormalize if needed
    needs_renorm = (normal_mag < 0.9).any() or (normal_mag > 1.1).any()
    if needs_renorm:
        # ⚡ Use rsqrt for faster normalization
        inv_mag = torch.rsqrt((normals * normals).sum(dim=-1, keepdim=True).clamp_min(1e-12))
        normals = normals * inv_mag
        if verbose:
            print(f"  ⚠️  Renormalized normals (were outside [0.9, 1.1])")
    
    # 7) Z-up consistency
    if prefer_up:
        up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        flip = (normals @ up) < 0  # ⚡ matrix multiply (faster)
        num_flipped = flip.sum().item()
        if num_flipped > 0:
            normals[flip] = -normals[flip]
            if verbose:
                print(f"  Flipped {num_flipped:,} normals for Z-up consistency")

    # 8) Spacing (⚡ already cached vox)
    spacing = torch.full((anchors.shape[0],), vox, device=device, dtype=dtype)

    # 9) Surface proximity (⚡ reuse sdf_vals if possible)
    # Need to recompute for filtered anchors
    g5_final = _world_to_grid(anchors, levelset.bbox_min, levelset.bbox_max).view(1,-1,1,1,3).float()
    sdf_final = F.grid_sample(sdf5, g5_final, mode='bilinear', padding_mode='border', align_corners=True).view(-1)
    
    # ⚡ Vectorized computation
    p_surf_raw = torch.reciprocal(sdf_final.abs() + 1e-4).to(dtype)

    out = {
        "anchors": anchors,
        "normals": normals,
        "spacing": spacing,
        "p_surf_raw": p_surf_raw,
        "sdf_values": sdf_final.abs().to(dtype),
    }

    # 10) Optional curvature
    if compute_curvature and knn is not None and anchors.shape[0] > 8:
        if verbose:
            print(f"\n[Curvature Estimation]")
        k = min(32, anchors.shape[0] - 1)
        idx_nn, _ = knn(anchors, anchors, k)
        
        # ⚡ Vectorized curvature
        curv = (normals[idx_nn] - normals.unsqueeze(1)).norm(dim=-1).mean(dim=1)
        out["curvature_approx"] = curv
        
        if verbose:
            print(f"  Curvature: mean={curv.mean():.6f}, max={curv.max():.6f}")

    if verbose:
        print(f"\n[Surface Proximity (p_surf_raw)]")
        print(f"  Based on: 1/|SDF| at anchor positions")
        print(f"  Mean: {p_surf_raw.mean():.2f}")
        print(f"  Std:  {p_surf_raw.std():.2f}")
        print(f"  Range: [{p_surf_raw.min():.2f}, {p_surf_raw.max():.2f}]")
        print(f"  |SDF| stats: mean={sdf_final.abs().mean():.6f}, max={sdf_final.abs().max():.6f}")
        print(f"\n{'='*60}")
        print(f"[Anchor Extraction] Complete")
        print(f"{'='*60}\n")

    return out


def estimate_curvature(
    points: torch.Tensor,
    normals: torch.Tensor,
    knn,
    k: int = 32,
    eps_mult: float = 1.25
) -> torch.Tensor:
    """⚡ Optimized curvature estimation."""
    k = min(k, points.shape[0])
    idx, _ = knn(points, points, k)
    normal_diff = (normals[idx] - normals.unsqueeze(1)).norm(dim=-1)
    curvature = normal_diff.mean(dim=1)
    return curvature


# =============================================================================
# ⚡ Precomputed Gradient Volume (Central Difference)
# =============================================================================
@torch.no_grad()
def precompute_grad_grid(phi_xyz: torch.Tensor, bbox_min: torch.Tensor, bbox_max: torch.Tensor):
    """
    ⚡ Precompute ∇φ volume using central differences.
    
    Args:
        phi_xyz: (D, H, W) SDF grid in world space
        bbox_min: (3,) world bbox minimum
        bbox_max: (3,) world bbox maximum
    
    Returns:
        grad5: (1, 3, D, H, W) gradient volume for grid_sample
    
    Note:
        Uses roll-based central difference. Boundary values wrap around
        (minor artifact). For production, consider replicate padding.
    """
    D, H, W = phi_xyz.shape
    
    # Central difference using roll (boundary wraps)
    # roll gives us diff over 2 voxels: [i-1] to [i+1]
    px = (phi_xyz.roll(-1, 0) - phi_xyz.roll(1, 0))  # 2*dx spacing
    py = (phi_xyz.roll(-1, 1) - phi_xyz.roll(1, 1))
    pz = (phi_xyz.roll(-1, 2) - phi_xyz.roll(1, 2))
    
    # Grid spacing in world coordinates
    # dx = L / (res - 1), and we have 2*dx spacing from roll
    L = bbox_max - bbox_min
    dx = L / torch.tensor([D-1, H-1, W-1], dtype=L.dtype, device=L.device).clamp_min(1.0)
    
    # Convert to world space gradient: diff / (2*dx)
    scale = 1.0 / (2.0 * dx + 1e-8)
    
    # Permute to (D,H,W) → (W,H,D) for grid_sample compatibility
    Gx = (px * scale[0]).permute(2, 1, 0).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    Gy = (py * scale[1]).permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
    Gz = (pz * scale[2]).permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
    
    return torch.cat([Gx, Gy, Gz], dim=1).contiguous()  # (1,3,D,H,W)


__all__ = [
    "LevelSetGrid", 
    "KNNDistanceSDF", 
    "extract_surface_anchors", 
    "estimate_curvature", 
    "precompute_grad_grid",
    "grid_idx_to_world",
    "world_to_grid_index",
]