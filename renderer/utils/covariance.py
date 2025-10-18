"""Covariance matrix operations."""

from __future__ import annotations
from typing import Tuple
import numpy as np


EPSILON_QUATERNION = 1e-12
EPSILON_EIGENVALUE = 1e-8


def pack_covariance_3x3_to_6d(cov: np.ndarray) -> np.ndarray:
    """
    Pack symmetric 3x3 covariance to 6D format.
    
    Format: [xx, xy, xz, yy, yz, zz]
    
    Args:
        cov: (N, 3, 3) covariance matrices
    
    Returns:
        (N, 6) packed covariances
    """
    if cov.ndim == 2 and cov.shape[1] == 6:
        return cov.astype(np.float32)
    
    if cov.ndim != 3 or cov.shape[1:] != (3, 3):
        raise ValueError(f"Expected (N, 3, 3) covariance, got {cov.shape}")
    
    xx = cov[:, 0, 0]
    xy = cov[:, 0, 1]
    xz = cov[:, 0, 2]
    yy = cov[:, 1, 1]
    yz = cov[:, 1, 2]
    zz = cov[:, 2, 2]
    
    return np.stack([xx, xy, xz, yy, yz, zz], axis=1).astype(np.float32)


def unpack_covariance_6d_to_3x3(cov6: np.ndarray) -> np.ndarray:
    """
    Unpack 6D covariance to symmetric 3x3 format.
    
    Args:
        cov6: (N, 6) packed covariances [xx, xy, xz, yy, yz, zz]
    
    Returns:
        (N, 3, 3) covariance matrices
    """
    N = cov6.shape[0]
    cov = np.zeros((N, 3, 3), dtype=np.float32)
    
    cov[:, 0, 0] = cov6[:, 0]  # xx
    cov[:, 0, 1] = cov6[:, 1]  # xy
    cov[:, 0, 2] = cov6[:, 2]  # xz
    cov[:, 1, 0] = cov6[:, 1]  # xy (symmetric)
    cov[:, 1, 1] = cov6[:, 3]  # yy
    cov[:, 1, 2] = cov6[:, 4]  # yz
    cov[:, 2, 0] = cov6[:, 2]  # xz (symmetric)
    cov[:, 2, 1] = cov6[:, 4]  # yz (symmetric)
    cov[:, 2, 2] = cov6[:, 5]  # zz
    
    return cov


def pack_covariance_torch(cov: "torch.Tensor") -> "torch.Tensor":
    """
    Pack PyTorch covariance tensors while preserving gradients.
    
    Args:
        cov: (N, 3, 3) covariance tensors
    
    Returns:
        (N, 6) packed covariances
    """
    import torch
    
    if cov.shape[-1] == 6:
        return cov
    
    return torch.stack([
        cov[:, 0, 0],  # xx
        cov[:, 0, 1],  # xy
        cov[:, 0, 2],  # xz
        cov[:, 1, 1],  # yy
        cov[:, 1, 2],  # yz
        cov[:, 2, 2],  # zz
    ], dim=1)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to XYZW quaternion.
    
    Args:
        R: (3, 3) rotation matrix
    
    Returns:
        (4,) quaternion [x, y, z, w]
    
    Notes:
        - Uses Shepperd's method for numerical stability
        - Returns normalized quaternion
    """
    trace = np.trace(R)
    
    if trace > 0.0:
        # w is the largest component
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        # x is the largest component
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    
    elif R[1, 1] > R[2, 2]:
        # y is the largest component
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    
    else:
        # z is the largest component
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    # Normalize
    q = np.array([x, y, z, w], dtype=np.float32)
    q /= (np.linalg.norm(q) + EPSILON_QUATERNION)
    
    return q


def decompose_covariance_to_scale_rotation(
    cov: np.ndarray,
    eps: float = EPSILON_EIGENVALUE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose covariance matrices into scale and rotation.
    
    Σ = R diag(s²) Rᵀ → (scales, quaternion)
    
    Args:
        cov: (N, 3, 3) covariance matrices
        eps: Minimum eigenvalue (for numerical stability)
    
    Returns:
        scales: (N, 3) scale factors
        quaternions: (N, 4) quaternions in XYZW format
    
    Notes:
        - Uses eigendecomposition
        - Enforces positive semi-definiteness
        - Sorts eigenvalues in descending order
    """
    N = cov.shape[0]
    scales = np.zeros((N, 3), dtype=np.float32)
    quaternions = np.zeros((N, 4), dtype=np.float32)
    
    for i in range(N):
        # Ensure symmetry
        C = 0.5 * (cov[i] + cov[i].T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        
        # Enforce positive eigenvalues
        eigenvalues = np.clip(eigenvalues, eps, None)
        
        # Compute scales (sqrt of eigenvalues)
        s = np.sqrt(eigenvalues)
        
        # Sort by descending eigenvalue
        idx = np.argsort(-s)
        s = s[idx]
        R = eigenvectors[:, idx]
        
        # Store scales
        scales[i] = s.astype(np.float32)
        
        # Convert rotation matrix to quaternion
        quaternions[i] = rotation_matrix_to_quaternion(R.astype(np.float32))
    
    return scales, quaternions