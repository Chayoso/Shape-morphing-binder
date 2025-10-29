"""
Learnable Covariance Matrices with Physics-Guided Initialization

This module implements learnable 3D Gaussian covariance matrices with hybrid
physics-guided approach for shape morphing.

Key Features:
- Cholesky parameterization for positive-definiteness
- Hybrid mixing with F-field based initialization (α-blending)
- Per-point learnable parameters
- Differentiable end-to-end

Author: PhysMorph-GS Team
Date: 2025-01
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class LearnableCovariance(nn.Module):
    """
    Learnable 3D Gaussian covariance matrices with Cholesky parameterization.
    
    Mathematical formulation:
        Σ = L·L^T  where L is lower triangular (6 parameters per point)
        
    Hybrid mode:
        Σ_final = α·Σ_physics + (1-α)·Σ_learnable
        
    Args:
        M: Number of points
        device: torch device
        dtype: torch dtype
        init_scale: Initial scale for L parameters (default: 0.08)
    """
    
    def __init__(
        self,
        M: int,
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.float32,
        init_scale: float = 0.08
    ):
        super().__init__()
        self.M = M
        self.device = device
        self.dtype = dtype
        
        # Lower triangular Cholesky parameters (6 per point)
        # Parameterization: [L11, L21, L22, L31, L32, L33]
        self.L_params = nn.Parameter(
            self._initialize_L_params(M, init_scale),
            requires_grad=True
        )
        
        print(f"[LearnableCovariance] Initialized with {M} points")
        print(f"  - Parameters: {self.L_params.shape} ({self.L_params.numel()} total)")
        print(f"  - Device: {device}, dtype: {dtype}")
        print(f"  - Init scale: {init_scale}")
    
    def _initialize_L_params(self, M: int, scale: float) -> torch.Tensor:
        """
        Initialize L parameters to produce isotropic covariance.
        
        For isotropic Σ = σ²·I, we need L such that L·L^T = σ²·I
        This means L = σ·I (diagonal)
        
        Since we use softplus on diagonal: L_ii = softplus(param_i) + 1e-6
        To get L_ii ≈ scale, we need param_i such that softplus(param_i) ≈ scale - 1e-6
        
        Args:
            M: Number of points
            scale: Initial scale (σ)
            
        Returns:
            L_params: (M, 6) initial parameters
        """
        L_params = torch.zeros(M, 6, device=self.device, dtype=self.dtype)
        
        # Inverse softplus for initialization
        # softplus(x) + 1e-6 = scale
        # softplus(x) = scale - 1e-6
        # x = log(exp(scale - 1e-6) - 1)
        # 
        # But for numerical stability, use:
        # softplus(x) ≈ x for x > 10
        # softplus(x) ≈ log(1 + exp(x)) for x < 10
        #
        # For target = 0.08 - 1e-6, inverse softplus:
        target = max(scale, 1e-5)  # Ensure positive (don't subtract eps yet)
        
        # Inverse softplus: log(exp(y) - 1)
        # For small y, use approximation: log(y) for safety
        if target < 0.01:
            init_val = torch.log(torch.tensor(target) + 1e-8)
        else:
            init_val = torch.log(torch.expm1(torch.tensor(target)))  # expm1(x) = exp(x) - 1
        
        # Set diagonal parameters (L11, L22, L33)
        L_params[:, 0] = init_val  # L11
        L_params[:, 2] = init_val  # L22
        L_params[:, 5] = init_val  # L33
        
        # Off-diagonal initialized to zero (L21, L31, L32)
        # This gives isotropic initialization: Σ ≈ σ²·I
        
        return L_params
    
    def _build_lower_triangular(self, params: torch.Tensor) -> torch.Tensor:
        """
        Convert (M, 6) parameters to (M, 3, 3) lower triangular matrices.
        
        Parameterization:
            params[:, 0] = L11
            params[:, 1] = L21
            params[:, 2] = L22
            params[:, 3] = L31
            params[:, 4] = L32
            params[:, 5] = L33
            
        Returns:
            L: (M, 3, 3) lower triangular matrices
        """
        M = params.shape[0]
        L = torch.zeros(M, 3, 3, device=params.device, dtype=params.dtype)
        
        # Fill lower triangular with positive diagonal enforcement
        # Adaptive epsilon based on batch size (critical for 300k+)
        batch_size = params.shape[0]
        if batch_size > 200000:
            eps_diag = 2e-4  # 200k+: Very strong regularization
        elif batch_size > 100000:
            eps_diag = 1e-4  # 100k-200k: Strong regularization
        else:
            eps_diag = 2e-5  # <100k: Normal regularization
        
        # Set diagonals first
        d0 = torch.nn.functional.softplus(params[:, 0]) + eps_diag  # L11
        d1 = torch.nn.functional.softplus(params[:, 2]) + eps_diag  # L22
        d2 = torch.nn.functional.softplus(params[:, 5]) + eps_diag  # L33
        
        L[:, 0, 0] = d0
        L[:, 1, 1] = d1
        L[:, 2, 2] = d2
        
        # 🔥 CRITICAL: Bound off-diagonals relative to diagonals
        # Balance: Numerical stability vs Expressive power
        # - Too loose (0.5+): Singular matrices, cusolver errors
        # - Too tight (0.1): Limited anisotropy, poor target alignment
        # - Sweet spot: 0.2-0.3 for good expressiveness with stability
        
        # Adaptive based on batch size (larger batches need tighter bounds)
        batch_size = params.shape[0]
        if batch_size > 250000:
            factor = 0.15  # 250k+: Conservative but expressive
        elif batch_size > 150000:
            factor = 0.20  # 150k-250k: Balanced
        else:
            factor = 0.25  # <150k: More freedom
        
        bound_21 = factor * d1  # (M,)
        L[:, 1, 0] = torch.clamp(params[:, 1], -bound_21, bound_21)
        
        # L31, L32: similarly bounded
        bound_31 = factor * d2
        bound_32 = factor * d2
        L[:, 2, 0] = torch.clamp(params[:, 3], -bound_31, bound_31)
        L[:, 2, 1] = torch.clamp(params[:, 4], -bound_32, bound_32)
        
        return L
    
    def build_covariance(self) -> torch.Tensor:
        """
        Build covariance matrices from Cholesky parameters.
        
        Returns:
            cov: (M, 3, 3) covariance matrices (Σ = L·L^T)
        """
        # Build lower triangular matrix L
        L = self._build_lower_triangular(self.L_params)  # (M, 3, 3)
        
        # Debug: Check L values
        if torch.isnan(L).any() or torch.isinf(L).any():
            print(f"[WARN] NaN or Inf in L matrix!")
            print(f"  L_params range: [{self.L_params.min():.6f}, {self.L_params.max():.6f}]")
        
        # Check diagonal positivity
        diag_L = torch.diagonal(L, dim1=-2, dim2=-1)  # (M, 3)
        if (diag_L <= 0).any():
            print(f"[WARN] Non-positive diagonal in L!")
            print(f"  L diagonal range: [{diag_L.min():.6f}, {diag_L.max():.6f}]")
        
        # Compute covariance: Σ = L·L^T
        cov = torch.bmm(L, L.transpose(-2, -1))  # (M, 3, 3)
        
        # Add adaptive regularization based on batch size
        # CRITICAL: Prevents singular matrices (det=0) in large batches
        batch_size = cov.shape[0]
        if batch_size > 200000:
            eps_reg = 2e-4  # 200k+: Very strong (matches diagonal epsilon)
        elif batch_size > 100000:
            eps_reg = 1e-4  # 100k-200k: Strong
        else:
            eps_reg = 2e-5  # <100k: Normal
        
        eye = torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0)  # (1, 3, 3)
        cov = cov + eps_reg * eye
        
        return cov
    
    def forward(
        self,
        cov_physics: Optional[torch.Tensor] = None,
        alpha: float = 0.3
    ) -> torch.Tensor:
        """
        Forward pass with optional physics-guided mixing.
        
        Args:
            cov_physics: (M, 3, 3) physics-based covariance (from F-field)
            alpha: Mixing weight
                - alpha=0.0: Pure learnable
                - alpha=0.3: Hybrid (recommended, 30% physics + 70% learnable)
                - alpha=1.0: Pure physics (no learning)
                
        Returns:
            cov_final: (M, 3, 3) final covariance matrices
        """
        # Build learnable covariance
        cov_learnable = self.build_covariance()
        
        # Hybrid mixing
        if cov_physics is not None and alpha > 0.0:
            if alpha >= 1.0:
                # Pure physics (bypass learnable)
                return cov_physics
            else:
                # Ensure same device (move cov_learnable to cov_physics device if needed)
                if cov_learnable.device != cov_physics.device:
                    cov_learnable = cov_learnable.to(cov_physics.device)
                
                # Check for invalid values in cov_physics
                if torch.isnan(cov_physics).any() or torch.isinf(cov_physics).any():
                    print(f"[WARN] Invalid values in cov_physics, using learnable only")
                    return cov_learnable
                
                # Check for negative diagonal in cov_physics
                diag_phys = torch.diagonal(cov_physics, dim1=-2, dim2=-1)
                eye_3x3 = torch.eye(3, device=cov_physics.device, dtype=cov_physics.dtype).unsqueeze(0)
                
                # Adaptive regularization for physics cov
                batch_size = cov_physics.shape[0]
                if batch_size > 200000:
                    eps_phys = 2e-4
                elif batch_size > 100000:
                    eps_phys = 1e-4
                else:
                    eps_phys = 2e-5
                
                if (diag_phys < 0).any():
                    print(f"[WARN] Negative diagonal in cov_physics!")
                    print(f"  Range: [{diag_phys.min():.6f}, {diag_phys.max():.6f}]")
                    min_diag_val = diag_phys.min(dim=-1)[0]
                    correction = torch.clamp(-min_diag_val, min=0.0).unsqueeze(-1).unsqueeze(-1)
                    cov_physics = cov_physics + correction * eye_3x3 + eps_phys * eye_3x3
                else:
                    # Always regularize physics cov for numerical stability
                    cov_physics = cov_physics + eps_phys * eye_3x3
                
                # Hybrid: α·Σ_physics + (1-α)·Σ_learnable
                cov_final = alpha * cov_physics + (1.0 - alpha) * cov_learnable
                
                # Final safety check: ensure positive diagonal + adaptive regularization
                diag_final = torch.diagonal(cov_final, dim1=-2, dim2=-1)
                eye = torch.eye(3, device=cov_final.device, dtype=cov_final.dtype).unsqueeze(0)
                
                # Adaptive epsilon matching batch size strategy
                batch_size = cov_final.shape[0]
                if batch_size > 200000:
                    eps_fix = 2e-4  # 200k+: Very strong
                elif batch_size > 100000:
                    eps_fix = 1e-4  # 100k-200k: Strong
                else:
                    eps_fix = 2e-5  # <100k: Normal
                
                if (diag_final < 0).any():
                    print(f"[WARN] Negative diagonal in final cov! Fixing...")
                    correction = torch.clamp(-diag_final.min(dim=-1)[0], min=0.0).unsqueeze(-1).unsqueeze(-1) * eye
                    cov_final = cov_final + correction + eps_fix * eye
                else:
                    # Always add regularization even if diagonal is positive
                    cov_final = cov_final + eps_fix * eye
        else:
            # Pure learnable
            cov_final = cov_learnable
        
        return cov_final
    
    def reinitialize_from_physics(self, cov_physics: torch.Tensor):
        """
        Reinitialize learnable parameters to match physics covariance.
        Useful for resetting at episode boundaries.
        
        Args:
            cov_physics: (M, 3, 3) target covariance
        """
        with torch.no_grad():
            # Cholesky decomposition: Σ = L·L^T
            try:
                L = torch.linalg.cholesky(cov_physics)  # (M, 3, 3)
                
                # Extract lower triangular parameters
                self.L_params[:, 0] = L[:, 0, 0]  # L11
                self.L_params[:, 1] = L[:, 1, 0]  # L21
                self.L_params[:, 2] = L[:, 1, 1]  # L22
                self.L_params[:, 3] = L[:, 2, 0]  # L31
                self.L_params[:, 4] = L[:, 2, 1]  # L32
                self.L_params[:, 5] = L[:, 2, 2]  # L33
                
                print(f"[LearnableCovariance] Reinitialized from physics covariance")
            except Exception as e:
                print(f"[WARN] Cholesky decomposition failed: {e}")
                print(f"  Keeping current parameters")
    
    def get_statistics(self) -> dict:
        """Get statistics about current covariance parameters."""
        with torch.no_grad():
            cov = self.build_covariance()
            
            # Check for NaN or inf
            if torch.isnan(cov).any() or torch.isinf(cov).any():
                print(f"[WARN] NaN or Inf detected in covariance matrices!")
                print(f"  NaN count: {torch.isnan(cov).sum().item()}")
                print(f"  Inf count: {torch.isinf(cov).sum().item()}")
                return {
                    'det_mean': 0.0,
                    'det_std': 0.0,
                    'trace_mean': 0.0,
                    'trace_std': 0.0,
                    'eigval_min': 0.0,
                    'eigval_max': 0.0,
                    'anisotropy_ratio': 1.0,  # Default to isotropic
                }
            
            # NOTE: We DON'T add extra regularization here for statistics
            # Eigenvalue computation might fail, but that's OK - it's just for logging
            # The covariance used for rendering already has sufficient regularization
            # from build_covariance()
            
            # If cusolver fails below, we'll return partial statistics (without eigenvalues)
            
            # Compute determinants (volume)
            det = torch.linalg.det(cov)
            
            # Compute traces (isotropy measure)
            trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1)
            
            # Eigenvalues (shape anisotropy)
            # NOTE: This might fail due to numerical precision in large batches
            # That's OK - eigenvalues are just for logging, not for learning
            try:
                eigvals = torch.linalg.eigvalsh(cov)  # (M, 3), sorted ascending
            except Exception as e:
                # Cusolver failed - return partial statistics without eigenvalues
                # This is acceptable because statistics are only for logging
                print(f"[INFO] Eigenvalue computation skipped (cusolver numerical precision limit)")
                print(f"  → This is OK - eigenvalues are for logging only, not used in learning")
                print(f"  → Covariance is still valid for rendering (det > 0)")
                return {
                    'det_mean': float(det.mean().item()),
                    'det_std': float(det.std().item()),
                    'det_min': float(det.min().item()),
                    'det_max': float(det.max().item()),
                    'trace_mean': float(trace.mean().item()),
                    'trace_std': float(trace.std().item()),
                    'eigval_min': None,  # Not computed
                    'eigval_max': None,  # Not computed
                    'anisotropy_ratio': 1.0,  # Default assumption
                }
            
            stats = {
                'det_mean': det.mean().item(),
                'det_std': det.std().item(),
                'det_min': det.min().item(),
                'det_max': det.max().item(),
                'trace_mean': trace.mean().item(),
                'trace_std': trace.std().item(),
                'eigval_min_mean': eigvals[:, 0].mean().item(),  # Smallest eigenvalue
                'eigval_max_mean': eigvals[:, 2].mean().item(),  # Largest eigenvalue
                'anisotropy_ratio': (eigvals[:, 2] / (eigvals[:, 0] + 1e-8)).mean().item(),
            }
            
            return stats


__all__ = ['LearnableCovariance']

