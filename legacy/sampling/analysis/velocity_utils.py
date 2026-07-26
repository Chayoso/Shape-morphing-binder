"""
Velocity field interpolation utilities for Level Set advection.

Author: CHAYO
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def interpolate_particles_to_grid(
    x_particles: torch.Tensor,
    v_particles: torch.Tensor,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    resolution: int = 128,
    kernel: str = 'trilinear',
    chunk_size: int = 32768
) -> torch.Tensor:
    """
    Interpolate particle velocities to regular grid.
    
    Uses splatting with trilinear weights for smooth interpolation.
    
    Args:
        x_particles: (N, 3) particle positions
        v_particles: (N, 3) particle velocities
        bbox_min: (3,) bounding box minimum
        bbox_max: (3,) bounding box maximum
        resolution: Grid resolution (R³)
        kernel: 'trilinear' or 'gaussian'
        chunk_size: Chunk size for memory efficiency
    
    Returns:
        vel_grid: (1, 3, R, R, R) velocity field in (N,C,D,H,W) format
                  where D,H,W correspond to z,y,x axes
    """
    device = x_particles.device
    dtype = x_particles.dtype
    R = resolution
    
    # Normalize particles to [0, R-1] grid coordinates
    x_norm = (x_particles - bbox_min) / (bbox_max - bbox_min + 1e-12)
    x_grid = x_norm * (R - 1)  # (N, 3) in [0, R-1]
    
    # Initialize grid
    vel_grid = torch.zeros((R, R, R, 3), device=device, dtype=dtype)
    weight_grid = torch.zeros((R, R, R), device=device, dtype=dtype)
    
    # Splat particles to grid
    for i in range(0, x_particles.shape[0], chunk_size):
        x_chunk = x_grid[i:i+chunk_size]
        v_chunk = v_particles[i:i+chunk_size]
        
        # Get 8 corner voxels for trilinear interpolation
        x_floor = x_chunk.floor().long()
        x_frac = x_chunk - x_floor.float()
        
        # Clip to valid range
        x_floor = x_floor.clamp(0, R - 2)
        
        # 8 corners of the cube
        for dx in [0, 1]:
            for dy in [0, 1]:
                for dz in [0, 1]:
                    # Corner indices
                    ix = x_floor[:, 0] + dx
                    iy = x_floor[:, 1] + dy
                    iz = x_floor[:, 2] + dz
                    
                    # Trilinear weights
                    wx = 1.0 - torch.abs(x_frac[:, 0] - dx)
                    wy = 1.0 - torch.abs(x_frac[:, 1] - dy)
                    wz = 1.0 - torch.abs(x_frac[:, 2] - dz)
                    w = wx * wy * wz  # (C,)
                    
                    # Splat weighted velocity
                    for j in range(len(ix)):
                        vel_grid[ix[j], iy[j], iz[j]] += w[j] * v_chunk[j]
                        weight_grid[ix[j], iy[j], iz[j]] += w[j]
    
    # Normalize by weights
    weight_grid = weight_grid.unsqueeze(-1).clamp_min(1e-12)
    vel_grid = vel_grid / weight_grid
    
    # Convert to (1, 3, D, H, W) format
    # (x,y,z,3) → (z,y,x,3) → (3,z,y,x) → (1,3,z,y,x)
    vel_grid = vel_grid.permute(2, 1, 0, 3)  # (z, y, x, 3)
    vel_grid = vel_grid.permute(3, 0, 1, 2)  # (3, z, y, x) = (3, D, H, W)
    vel_grid = vel_grid.unsqueeze(0)         # (1, 3, D, H, W)
    
    return vel_grid


def interpolate_particles_to_grid_weighted(
    x_particles: torch.Tensor,
    v_particles: torch.Tensor,
    weights: torch.Tensor,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    resolution: int = 128,
    kernel: str = 'trilinear',
    chunk_size: int = 32768
) -> torch.Tensor:
    """
    ⚡ Weighted particle-to-grid interpolation for surface-biased advection.
    
    This function prioritizes particles near the surface by applying per-particle
    weights (e.g., exp(-|φ|/bandwidth)). This ensures that level set advection
    uses surface velocity rather than bulk volume-averaged velocity.
    
    Mathematical formulation:
        v(grid_point) = Σ(w_i * k(x - x_i) * v_i) / Σ(w_i * k(x - x_i))
        
        where:
            w_i = exp(-|φ(x_i)|/bandwidth)  # Surface proximity weight
            k(·) = trilinear kernel          # Spatial interpolation weight
            v_i = particle velocity
    
    Args:
        x_particles: (N, 3) particle positions in world coords
        v_particles: (N, 3) particle velocities in world coords
        weights: (N,) per-particle weights (surface proximity)
                 Recommended: exp(-|φ|/bandwidth) where bandwidth ≈ 2*voxel_size
        bbox_min: (3,) bounding box minimum
        bbox_max: (3,) bounding box maximum
        resolution: Grid resolution (R³)
        kernel: 'trilinear' (only option for now)
        chunk_size: Chunk size for memory efficiency
    
    Returns:
        vel_grid: (1, 3, D, H, W) velocity field in (N,C,D,H,W) format
                  where D,H,W correspond to z,y,x axes
    
    Example:
        >>> # Compute surface weights from SDF
        >>> phi_vals = sample_sdf_at_particles(x_particles, levelset)
        >>> bandwidth = 2.0 * voxel_size
        >>> weights = torch.exp(-phi_vals.abs() / bandwidth)
        >>> 
        >>> # Interpolate with surface bias
        >>> vel_grid = interpolate_particles_to_grid_weighted(
        ...     x_particles, v_particles, weights,
        ...     bbox_min, bbox_max, resolution
        ... )
    
    Note:
        This is crucial for morphing applications where surface motion must
        dominate over interior bulk motion. Without weighting, interior particles
        (which are more numerous) would dilute the surface velocity signal.
    """
    device = x_particles.device
    dtype = x_particles.dtype
    R = resolution
    
    # Validate weights
    if weights.shape[0] != x_particles.shape[0]:
        raise ValueError(
            f"Weight count mismatch: got {weights.shape[0]} weights "
            f"for {x_particles.shape[0]} particles"
        )
    
    # Normalize particles to [0, R-1] grid coordinates
    x_norm = (x_particles - bbox_min) / (bbox_max - bbox_min + 1e-12)
    x_grid = x_norm * (R - 1)
    
    # Initialize grids
    vel_grid = torch.zeros((R, R, R, 3), device=device, dtype=dtype)
    weight_grid = torch.zeros((R, R, R), device=device, dtype=dtype)
    
    # Splat particles with weights
    for i in range(0, x_particles.shape[0], chunk_size):
        x_chunk = x_grid[i:i+chunk_size]
        v_chunk = v_particles[i:i+chunk_size]
        w_chunk = weights[i:i+chunk_size]  # 🔥 Per-particle weight
        
        # Get 8 corner voxels for trilinear interpolation
        x_floor = x_chunk.floor().long()
        x_frac = x_chunk - x_floor.float()
        
        # Clip to valid range
        x_floor = x_floor.clamp(0, R - 2)
        
        # 8 corners of the cube
        for dx in [0, 1]:
            for dy in [0, 1]:
                for dz in [0, 1]:
                    # Corner indices
                    ix = x_floor[:, 0] + dx
                    iy = x_floor[:, 1] + dy
                    iz = x_floor[:, 2] + dz
                    
                    # Trilinear weights × particle weights
                    wx = 1.0 - torch.abs(x_frac[:, 0] - dx)
                    wy = 1.0 - torch.abs(x_frac[:, 1] - dy)
                    wz = 1.0 - torch.abs(x_frac[:, 2] - dz)
                    w = wx * wy * wz * w_chunk  # 🔥 Multiply by particle weight
                    
                    # Splat weighted velocity
                    for j in range(len(ix)):
                        vel_grid[ix[j], iy[j], iz[j]] += w[j] * v_chunk[j]
                        weight_grid[ix[j], iy[j], iz[j]] += w[j]
    
    # Normalize by weights
    weight_grid = weight_grid.unsqueeze(-1).clamp_min(1e-12)
    vel_grid = vel_grid / weight_grid
    
    # Convert to (1, 3, D, H, W) format
    vel_grid = vel_grid.permute(2, 1, 0, 3)  # (z, y, x, 3)
    vel_grid = vel_grid.permute(3, 0, 1, 2)  # (3, z, y, x) = (3, D, H, W)
    vel_grid = vel_grid.unsqueeze(0)         # (1, 3, D, H, W)
    
    return vel_grid


def velocity_divergence(
    vel_grid: torch.Tensor,
    voxel_size: float
) -> torch.Tensor:
    """
    Compute divergence of velocity field: ∇·v.
    
    Args:
        vel_grid: (1, 3, D, H, W) velocity field
        voxel_size: Grid spacing
    
    Returns:
        div: (1, 1, D, H, W) divergence field
    """
    vx = vel_grid[:, 0:1]  # (1, 1, D, H, W)
    vy = vel_grid[:, 1:2]
    vz = vel_grid[:, 2:3]
    
    # Central differences
    dvx_dx = (vx.roll(-1, 2) - vx.roll(1, 2)) / (2 * voxel_size)
    dvy_dy = (vy.roll(-1, 3) - vy.roll(1, 3)) / (2 * voxel_size)
    dvz_dz = (vz.roll(-1, 4) - vz.roll(1, 4)) / (2 * voxel_size)
    
    div = dvx_dx + dvy_dy + dvz_dz
    
    return div


def make_incompressible(
    vel_grid: torch.Tensor,
    voxel_size: float,
    iters: int = 10
) -> torch.Tensor:
    """
    Project velocity to divergence-free field (optional).
    
    Solves Poisson equation: ∇²p = ∇·v, then v' = v - ∇p.
    
    Args:
        vel_grid: (1, 3, D, H, W) velocity field
        voxel_size: Grid spacing
        iters: Number of Jacobi iterations
    
    Returns:
        vel_corrected: (1, 3, D, H, W) divergence-free velocity
    """
    device = vel_grid.device
    dtype = vel_grid.dtype
    
    # Compute divergence
    div = velocity_divergence(vel_grid, voxel_size)
    
    # Solve Laplacian(p) = div via Jacobi
    p = torch.zeros_like(div)
    
    for _ in range(iters):
        p_old = p.clone()
        
        # 6-neighbor Laplacian stencil
        laplacian = (
            p_old.roll(-1, 2) + p_old.roll(1, 2) +
            p_old.roll(-1, 3) + p_old.roll(1, 3) +
            p_old.roll(-1, 4) + p_old.roll(1, 4) -
            6 * p_old
        ) / (voxel_size ** 2)
        
        # Jacobi update: p_new = (div - other_terms) / diagonal
        p = p_old + 0.166667 * (div - laplacian)
    
    # Compute gradient of pressure
    dp_dx = (p.roll(-1, 2) - p.roll(1, 2)) / (2 * voxel_size)
    dp_dy = (p.roll(-1, 3) - p.roll(1, 3)) / (2 * voxel_size)
    dp_dz = (p.roll(-1, 4) - p.roll(1, 4)) / (2 * voxel_size)
    
    grad_p = torch.cat([dp_dx, dp_dy, dp_dz], dim=1)  # (1, 3, D, H, W)
    
    # Project: v' = v - ∇p
    vel_corrected = vel_grid - grad_p
    
    return vel_corrected


__all__ = [
    "interpolate_particles_to_grid",
    "interpolate_particles_to_grid_weighted",
    "velocity_divergence",
    "make_incompressible",
]

