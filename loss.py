"""
E2E Loss Manager with Silhouette Alignment & Curvature-based Target Covariance
All operations are differentiable and support autograd.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Callable, Union
import numpy as np


# ============================================================================
# Constants
# ============================================================================
DEFAULT_WEIGHTS = {
    'w_alpha': 0.3,
    'w_depth': 0.1,
    'w_photo': 0.0,
    'w_normal': 0.0,
    'w_cov_reg': 0.01,
    'w_normal_smooth': 0.0,
    'w_edge': 0.1,
    'w_cov_align': 0.05,
    'w_coverage': 0.0,  # 🔥 NEW: Coverage loss (hole penalty)
}

SCHEDULE_PARAMS = {
    'linear': {'alpha_range': (0.3, 0.7), 'edge_range': (0.05, 0.20)},
    'cosine': {'alpha_range': (0.3, 0.7)},
}

# Numerical stability constants
EPS_SAFE = 1e-8
EPS_NORMALIZE = 1e-9
CLAMP_MIN_DEPTH = 0.01
CLAMP_GRAD_NORM = (-3.0, 3.0)
CLAMP_CURVATURE = (0.0, 10.0)
# 🔥 Scale clamping removed - trust curvature-based calculation!
# CLAMP_SCALE and CLAMP_NORMAL_SCALE no longer used


# ============================================================================
# Helper Functions
# ============================================================================
def _safe_normalize(tensor: torch.Tensor, dim: int = -1, eps: float = EPS_NORMALIZE) -> torch.Tensor:
    """
    Safely normalize a tensor along the given dimension.
    
    Args:
        tensor: Input tensor to normalize
        dim: Dimension along which to normalize
        eps: Small epsilon for numerical stability
    
    Returns:
        Normalized tensor (differentiable)
    """
    norm = torch.norm(tensor, dim=dim, keepdim=True)
    return tensor / torch.clamp(norm, min=eps)


def _align_tensor_shapes(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Align shapes of prediction and target tensors by removing batch dimension if needed.
    
    Args:
        pred: Predicted tensor
        target: Target tensor
    
    Returns:
        Tuple of (pred, target) with aligned shapes
    """
    if pred.ndim != target.ndim:
        if pred.ndim == 3 and target.ndim == 2:
            pred = pred.squeeze(0) if pred.shape[0] == 1 else pred[0]
        elif pred.ndim == 2 and target.ndim == 3:
            target = target.squeeze(0) if target.shape[0] == 1 else target[0]
    return pred, target


def _safe_device_transfer(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Safely transfer tensor to device if needed (no-op if already on device).
    
    Args:
        tensor: Input tensor
        device: Target device
    
    Returns:
        Tensor on target device
    """
    if tensor.device != device:
        return tensor.to(device)
    return tensor


def _validate_tensor(tensor: torch.Tensor, name: str) -> bool:
    """
    Check if tensor contains NaN or Inf values.
    
    Args:
        tensor: Tensor to validate
        name: Name for logging
    
    Returns:
        True if valid, False if contains NaN/Inf
    """
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"[WARN] {name} contains NaN/Inf values")
        return False
    return True


def _convert_to_torch(array: Union[np.ndarray, torch.Tensor], device: torch.device) -> torch.Tensor:
    """
    Convert numpy array or torch tensor to torch tensor on specified device.
    
    Args:
        array: Input array (numpy or torch)
        device: Target device
    
    Returns:
        Torch tensor on target device
    """
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array.astype(np.float32)).to(device)
    elif isinstance(array, torch.Tensor):
        return array.to(device)
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(array)}")


# ============================================================================
# Loss Manager
# ============================================================================
class E2ELossManager:
    """
    End-to-End Training Loss Manager with comprehensive loss components.
    
    Manages multiple loss terms including:
    - Alpha (silhouette) loss
    - Depth loss
    - Photometric loss
    - Edge alignment loss
    - Covariance alignment loss
    - Covariance regularization
    
    All loss computations are differentiable and support autograd.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize loss manager with configuration.
        
        Args:
            config: Configuration dictionary containing loss weights and parameters
        """
        self.config = config
        self.weights = self._initialize_weights(config)
        self.schedule = config.get('schedule', 'constant')
        self.total_steps = config.get('total_steps', 100)
        self.current_step = 0
    
    def _initialize_weights(self, config: Dict) -> Dict[str, float]:
        """
        Initialize loss weights from config with defaults.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Dictionary of loss weights
        """
        weights = {}
        for key, default_val in DEFAULT_WEIGHTS.items():
            weights[key] = float(config.get(key, default_val))
        return weights
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get a copy of current weights.
        
        Returns:
            Copy of current weight dictionary
        """
        return self.weights.copy()
    
    def update_weights(self, step: Optional[int] = None):
        """
        Update weights based on schedule (linear, cosine, or constant).
        
        Args:
            step: Optional step number to set current step
        """
        if step is not None:
            self.current_step = step
        
        if self.schedule == 'constant':
            return
        
        progress = min(1.0, self.current_step / self.total_steps)
        
        if self.schedule == 'linear':
            self._update_linear_schedule(progress)
        elif self.schedule == 'cosine':
            self._update_cosine_schedule(progress)
    
    def _update_linear_schedule(self, progress: float):
        """
        Apply linear schedule to weights with progressive refinement strategy.
        
        Schedule (recommended):
        - Early (0-10%):  High cov_align (0.10), low edge/coverage (spectrum alignment)
        - Mid (10-60%):   Balanced cov_align (0.05), rising edge (0.10)
        - Late (60-100%): Low cov_align (0.02), high edge (0.12) (detail refinement)
        
        Args:
            progress: Training progress [0, 1]
        """
        # 🔥 Progressive refinement strategy
        if progress < 0.1:
            # Early: Spectrum alignment
            cov_align_mult = 2.0
            edge_mult = 0.5
            coverage_mult = 0.5
        elif progress < 0.6:
            # Mid: Balanced
            cov_align_mult = 1.0
            edge_mult = 1.0
            coverage_mult = 1.0
        else:
            # Late: Detail refinement
            cov_align_mult = 0.4
            edge_mult = 1.2
            coverage_mult = 0.5
        
        # Apply multipliers to base weights
        params = SCHEDULE_PARAMS['linear']
        alpha_min, alpha_max = params['alpha_range']
        edge_min, edge_max = params['edge_range']
        
        self.weights['w_alpha'] = alpha_min + (alpha_max - alpha_min) * progress
        self.weights['w_edge'] = edge_min + (edge_max - edge_min) * progress
    
    def _update_cosine_schedule(self, progress: float):
        """
        Apply cosine schedule to weights.
        
        Args:
            progress: Training progress in [0, 1]
        """
        import math
        params = SCHEDULE_PARAMS['cosine']
        alpha_min, alpha_max = params['alpha_range']
        
        factor = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        self.weights['w_alpha'] = alpha_min + (alpha_max - alpha_min) * (1 - factor)
    
    def compute_render_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        cov: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        view_params: Optional[Dict] = None,
        cov_target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive render loss (DIFFERENTIABLE).
        
        Args:
            pred: Dictionary with keys {'image', 'alpha', 'depth'} - predicted renders
            target: Dictionary with keys {'image', 'alpha', 'depth', 'cov_target'} - ground truth
            cov: (N, 3, 3) covariance matrices
            mu: (N, 3) 3D positions for edge alignment
            view_params: Dictionary with keys {'view_T', 'W', 'H', 'tanfovx', 'tanfovy'}
            cov_target: (N, 3, 3) target covariances from curvature
        
        Returns:
            Dictionary of loss components including 'loss_render_total'
        """
        losses = {}
        total = torch.tensor(0.0, device=self._get_device(pred, cov, mu))
        
        # Compute individual loss components (all differentiable)
        total += self._compute_alpha_loss(pred, target, losses)
        total += self._compute_depth_loss(pred, target, losses)
        total += self._compute_photo_loss(pred, target, losses)
        total += self._compute_edge_loss(pred, target, mu, cov, view_params, losses)
        total += self._compute_cov_align_loss(cov, cov_target, target, losses)
        total += self._compute_cov_reg_loss(cov, losses)
        total += self._compute_coverage_loss(pred, target, losses)  # 🔥 NEW: Hole penalty
        
        losses['loss_render_total'] = total
        return losses
    
    def _get_device(self, pred: Dict, cov: Optional[torch.Tensor], mu: Optional[torch.Tensor]) -> torch.device:
        """
        Get device from available tensors.
        
        Args:
            pred: Prediction dictionary
            cov: Covariance tensor
            mu: Position tensor
        
        Returns:
            Device object
        """
        if 'alpha' in pred:
            return pred['alpha'].device
        if cov is not None:
            return cov.device
        if mu is not None:
            return mu.device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _compute_alpha_loss(
        self, 
        pred: Dict[str, torch.Tensor], 
        target: Dict[str, torch.Tensor],
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute alpha channel loss (DIFFERENTIABLE).
        
        Args:
            pred: Prediction dictionary
            target: Target dictionary
            losses: Dictionary to store loss component
        
        Returns:
            Weighted alpha loss
        """
        if self.weights['w_alpha'] <= 0 or 'alpha' not in pred or 'alpha' not in target:
            losses['loss_alpha'] = torch.tensor(0.0)
            return torch.tensor(0.0)
        
        pred_alpha, target_alpha = _align_tensor_shapes(pred['alpha'], target['alpha'])
        loss_alpha = F.l1_loss(pred_alpha, target_alpha)
        losses['loss_alpha'] = loss_alpha
        return self.weights['w_alpha'] * loss_alpha
    
    def _compute_coverage_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute coverage loss to prevent holes (DIFFERENTIABLE).
        
        Penalizes low alpha values in regions where target has high alpha.
        This prevents sparse particles from creating holes in the rendered surface.
        
        Args:
            pred: Prediction dictionary with 'alpha' key
            target: Target dictionary with 'alpha' key
            losses: Dictionary to store loss component
        
        Returns:
            Weighted coverage loss
        """
        if self.weights.get('w_coverage', 0.0) <= 0 or 'alpha' not in pred or 'alpha' not in target:
            losses['loss_coverage'] = torch.tensor(0.0)
            return torch.tensor(0.0)
        
        pred_alpha, target_alpha = _align_tensor_shapes(pred['alpha'], target['alpha'])
        
        # 🔥 Strategy: Penalize regions where target is opaque but prediction is transparent
        # This forces particles to fill holes even if physics makes them sparse
        
        # Mask: Where target expects opacity (alpha > 0.5)
        target_mask = (target_alpha > 0.5).float()
        
        # Hole penalty: In target regions, pred_alpha should be high
        hole_penalty = target_mask * (1.0 - pred_alpha)  # High when target=1 but pred=0
        
        # Additional: Penalize variance (uniform coverage is better)
        alpha_var = pred_alpha.var()
        
        loss_coverage = hole_penalty.mean() + 0.1 * alpha_var
        
        losses['loss_coverage'] = loss_coverage
        return self.weights.get('w_coverage', 0.0) * loss_coverage
    
    def _compute_depth_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute depth loss with validity masking (DIFFERENTIABLE).
        
        Args:
            pred: Prediction dictionary
            target: Target dictionary
            losses: Dictionary to store loss component
        
        Returns:
            Weighted depth loss
        """
        if self.weights['w_depth'] <= 0 or 'depth' not in pred or target.get('depth') is None:
            losses['loss_depth'] = torch.tensor(0.0)
            return torch.tensor(0.0)
        
        pred_depth = pred['depth']
        target_depth = target['depth']
        device = pred_depth.device
        
        valid_mask = (target_depth > 0) & (pred_depth > 0)
        
        if valid_mask.sum() > 0:
            loss_depth = F.l1_loss(pred_depth[valid_mask], target_depth[valid_mask])
            losses['loss_depth'] = loss_depth
            return self.weights['w_depth'] * loss_depth
        else:
            losses['loss_depth'] = torch.tensor(0.0, device=device)
            return torch.tensor(0.0, device=device)
    
    def _compute_photo_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute photometric loss (DIFFERENTIABLE).
        
        Args:
            pred: Prediction dictionary
            target: Target dictionary
            losses: Dictionary to store loss component
        
        Returns:
            Weighted photometric loss
        """
        if self.weights['w_photo'] <= 0 or 'image' not in pred or 'image' not in target:
            losses['loss_photo'] = torch.tensor(0.0)
            return torch.tensor(0.0)
        
        loss_photo = F.l1_loss(pred['image'], target['image'])
        losses['loss_photo'] = loss_photo
        return self.weights['w_photo'] * loss_photo
    
    def _compute_edge_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        mu: Optional[torch.Tensor],
        cov: Optional[torch.Tensor],
        view_params: Optional[Dict],
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute silhouette edge alignment loss (DIFFERENTIABLE).
        
        Args:
            pred: Prediction dictionary
            target: Target dictionary
            mu: (N, 3) 3D positions
            cov: (N, 3, 3) covariances
            view_params: View parameters dictionary
            losses: Dictionary to store loss component
        
        Returns:
            Weighted edge alignment loss
        """
        if self.weights['w_edge'] <= 0 or mu is None or cov is None or view_params is None:
            losses['loss_edge'] = torch.tensor(0.0)
            return torch.tensor(0.0)
        
        try:
            alpha_target = target.get('alpha', pred.get('alpha'))
            loss_edge, edge_info = edge_align_loss(
                mu, cov, alpha_target,
                view_params['view_T'],
                view_params['W'],
                view_params['H'],
                view_params['tanfovx'],
                view_params['tanfovy']
            )
            losses['loss_edge'] = loss_edge
            losses.update(edge_info)
            return self.weights['w_edge'] * loss_edge
        except Exception as e:
            print(f"[WARN] Edge alignment failed: {e}")
            device = mu.device
            losses['loss_edge'] = torch.tensor(0.0, device=device)
            return torch.tensor(0.0, device=device)
    
    def _compute_cov_align_loss(
        self,
        cov: Optional[torch.Tensor],
        cov_target: Optional[torch.Tensor],
        target: Dict[str, torch.Tensor],
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute covariance spectral alignment loss (DIFFERENTIABLE).
        
        Args:
            cov: (N, 3, 3) predicted covariances
            cov_target: (N, 3, 3) target covariances
            target: Target dictionary (fallback for cov_target)
            losses: Dictionary to store loss component
        
        Returns:
            Weighted covariance alignment loss
        """
        if self.weights['w_cov_align'] <= 0 or cov is None:
            losses['loss_cov_align'] = torch.tensor(0.0)
            return torch.tensor(0.0)
        
        # Get cov_target from argument or target dict
        if cov_target is None and 'cov_target' in target:
            cov_target = target['cov_target']
        
        # 🔥 Guard: Check if cov_target is available
        if cov_target is None:
            print("[WARN] cov_target missing → spectral loss skipped")
            print("  → Make sure target covariance Σ★ is computed in pipeline.py STAGE 6")
            device = cov.device if cov is not None else torch.device('cuda')
            losses['loss_cov_align'] = torch.tensor(0.0, device=device)
            return torch.tensor(0.0, device=device)
        
        # Compute spectral alignment loss
        loss_cov_align = covariance_spectral_loss(cov, cov_target)
        losses['loss_cov_align'] = loss_cov_align
        return self.weights['w_cov_align'] * loss_cov_align
    
    def _compute_cov_reg_loss(
        self,
        cov: Optional[torch.Tensor],
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute covariance regularization loss (DIFFERENTIABLE).
        
        Args:
            cov: (N, 3, 3) covariances
            losses: Dictionary to store loss component
        
        Returns:
            Weighted covariance regularization loss
        """
        if self.weights['w_cov_reg'] <= 0 or cov is None:
            losses['loss_cov_reg'] = torch.tensor(0.0)
            return torch.tensor(0.0)
        
        loss_cov_reg = self._covariance_regularization(cov)
        losses['loss_cov_reg'] = loss_cov_reg
        return self.weights['w_cov_reg'] * loss_cov_reg
    
    def _covariance_regularization(self, cov: torch.Tensor) -> torch.Tensor:
        """
        Regularize covariance matrices (DIFFERENTIABLE).
        
        Args:
            cov: (N, 3, 3) covariances
        
        Returns:
            Regularization loss
        """
        mode = self.config.get('cov_reg_mode', 'frobenius')
        target_scale = self.config.get('target_cov_scale', 0.02)
        device = cov.device
        
        if mode == 'frobenius':
            return self._frobenius_regularization(cov, target_scale, device)
        elif mode == 'trace':
            return self._trace_regularization(cov, target_scale)
        elif mode == 'eigenvalue':
            return self._eigenvalue_regularization(cov, target_scale)
        else:
            print(f"[WARN] Unknown cov_reg_mode '{mode}', returning zero loss")
            return torch.tensor(0.0, device=device)
    
    def _frobenius_regularization(
        self, 
        cov: torch.Tensor, 
        target_scale: float,
        device: torch.device
    ) -> torch.Tensor:
        """
        Frobenius norm regularization (DIFFERENTIABLE).
        
        Args:
            cov: (N, 3, 3) covariances
            target_scale: Target scale parameter
            device: Device
        
        Returns:
            Frobenius regularization loss
        """
        target_cov = (target_scale ** 2) * torch.eye(3, device=device).unsqueeze(0).expand_as(cov)
        return F.mse_loss(cov, target_cov)
    
    def _trace_regularization(self, cov: torch.Tensor, target_scale: float) -> torch.Tensor:
        """
        Trace regularization (DIFFERENTIABLE).
        
        Args:
            cov: (N, 3, 3) covariances
            target_scale: Target scale parameter
        
        Returns:
            Trace regularization loss
        """
        trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1)
        target_trace = 3 * (target_scale ** 2)
        return F.l1_loss(trace, torch.full_like(trace, target_trace))
    
    def _eigenvalue_regularization(self, cov: torch.Tensor, target_scale: float) -> torch.Tensor:
        """
        Eigenvalue regularization (DIFFERENTIABLE).
        
        Args:
            cov: (N, 3, 3) covariances
            target_scale: Target scale parameter
        
        Returns:
            Eigenvalue regularization loss
        """
        eigvals = torch.linalg.eigvalsh(cov)
        target_eig = target_scale ** 2
        return F.mse_loss(eigvals, torch.full_like(eigvals, target_eig))


# ============================================================================
# Silhouette Edge Alignment Loss
# ============================================================================
def compute_projection_jacobian(
    mu: torch.Tensor,
    view_T: Union[np.ndarray, torch.Tensor],
    tanfovx: float,
    tanfovy: float,
    W: int,
    H: int
) -> torch.Tensor:
    """
    Compute projection Jacobian J = ∂(screen_xy)/∂(world_xyz) (DIFFERENTIABLE).
    
    Args:
        mu: (N, 3) world-space positions
        view_T: (4, 4) view matrix (world to camera) - numpy or torch
        tanfovx: Tangent of half horizontal FOV
        tanfovy: Tangent of half vertical FOV
        W: Image width in pixels
        H: Image height in pixels
    
    Returns:
        J: (N, 2, 3) Jacobian matrices
    """
    N = mu.shape[0]
    device = mu.device
    
    # Convert view matrix to torch with type safety
    view_T_torch = _convert_to_torch(view_T, device)
    
    # Transform to camera space (differentiable)
    mu_hom = torch.cat([mu, torch.ones(N, 1, device=device)], dim=1)  # [N, 4]
    mu_cam = (view_T_torch @ mu_hom.T).T  # [N, 4]
    
    x, y, z = mu_cam[:, 0], mu_cam[:, 1], mu_cam[:, 2]
    z_safe = torch.clamp(z, min=CLAMP_MIN_DEPTH)
    
    # Compute Jacobian components (differentiable)
    z2 = z_safe * z_safe
    
    J_screen_cam = torch.zeros(N, 2, 3, device=device)
    J_screen_cam[:, 0, 0] = W / (2 * tanfovx * z_safe)
    J_screen_cam[:, 0, 2] = -W * x / (2 * tanfovx * z2)
    J_screen_cam[:, 1, 1] = -H / (2 * tanfovy * z_safe)
    J_screen_cam[:, 1, 2] = H * y / (2 * tanfovy * z2)
    
    # Chain rule: J = J_screen_cam @ R (differentiable)
    R = view_T_torch[:3, :3]  # [3, 3]
    J = torch.bmm(J_screen_cam, R.unsqueeze(0).expand(N, -1, -1))  # [N, 2, 3]
    
    return J

def _compute_sobel_gradients(
    alpha_target: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Sobel gradients of alpha map (DIFFERENTIABLE).
    
    Args:
        alpha_target: (H, W) or (1, H, W) or (H, W, 1) alpha channel
        device: Target device
    
    Returns:
        Tuple of (grad_x, grad_y, grad_norm) each of shape (H, W)
    """
    # Ensure 2D shape
    if alpha_target.ndim == 3:
        if alpha_target.shape[0] == 1:
            alpha_target = alpha_target[0]
        elif alpha_target.shape[-1] == 1:
            alpha_target = alpha_target.squeeze(-1)
    
    alpha_target = _safe_device_transfer(alpha_target, device)
    alpha_target_expanded = alpha_target.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Create Sobel filters
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
        dtype=torch.float32, 
        device=device
    ).view(1, 1, 3, 3)
    
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
        dtype=torch.float32, 
        device=device
    ).view(1, 1, 3, 3)
    
    # Compute gradients (differentiable)
    grad_x = F.conv2d(alpha_target_expanded, sobel_x, padding=1).squeeze()
    grad_y = F.conv2d(alpha_target_expanded, sobel_y, padding=1).squeeze()
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + EPS_SAFE)
    
    return grad_x, grad_y, grad_norm


def _compute_silhouette_tangents(
    grad_x: torch.Tensor,
    grad_y: torch.Tensor,
    grad_norm: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute silhouette tangent vectors from gradients (DIFFERENTIABLE).
    
    Args:
        grad_x: (H, W) horizontal gradients
        grad_y: (H, W) vertical gradients
        grad_norm: (H, W) gradient magnitudes
    
    Returns:
        Tuple of (tangent_x, tangent_y) each of shape (H, W)
    """
    tangent_x = -grad_y / (grad_norm + EPS_SAFE)
    tangent_y = grad_x / (grad_norm + EPS_SAFE)
    return tangent_x, tangent_y


def _project_points_to_screen(
    mu: torch.Tensor,
    view_T: Union[np.ndarray, torch.Tensor],
    tanfovx: float,
    tanfovy: float,
    W: int,
    H: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D points to screen space (DIFFERENTIABLE).
    
    Args:
        mu: (N, 3) 3D positions
        view_T: (4, 4) view matrix
        tanfovx: Tangent of half horizontal FOV
        tanfovy: Tangent of half vertical FOV
        W: Image width
        H: Image height
        device: Device
    
    Returns:
        Tuple of (screen_x, screen_y) each of shape (N,)
    """
    N = mu.shape[0]
    mu_hom = torch.cat([mu, torch.ones(N, 1, device=device)], dim=1)
    view_T_torch = _convert_to_torch(view_T, device)
    mu_cam = (view_T_torch @ mu_hom.T).T
    
    z = torch.clamp(mu_cam[:, 2], min=CLAMP_MIN_DEPTH)
    ndc_x = mu_cam[:, 0] / z
    ndc_y = mu_cam[:, 1] / z
    
    screen_x = (ndc_x / tanfovx * 0.5 + 0.5) * W
    screen_y = (-ndc_y / tanfovy * 0.5 + 0.5) * H
    
    return screen_x, screen_y

def _sample_tangents_at_points(
    tangent_x: torch.Tensor,
    tangent_y: torch.Tensor,
    grad_norm: torch.Tensor,
    screen_x: torch.Tensor,
    screen_y: torch.Tensor,
    W: int,
    H: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample tangent vectors at projected point locations (DIFFERENTIABLE).
    
    Args:
        tangent_x: (H, W) horizontal tangent components
        tangent_y: (H, W) vertical tangent components
        grad_norm: (H, W) gradient magnitudes
        screen_x: (N,) screen x-coordinates
        screen_y: (N,) screen y-coordinates
        W: Image width
        H: Image height
    
    Returns:
        Tuple of (t_hat, grad_sampled)
            - t_hat: (N, 2) sampled tangent vectors
            - grad_sampled: (N,) sampled gradient magnitudes
    """
    grid_x = 2.0 * screen_x / W - 1.0
    grid_y = 2.0 * screen_y / H - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)
    
    tangent_x_exp = tangent_x.unsqueeze(0).unsqueeze(0)
    tangent_y_exp = tangent_y.unsqueeze(0).unsqueeze(0)
    grad_norm_exp = grad_norm.unsqueeze(0).unsqueeze(0)
    
    # Grid sample is differentiable
    t_x_sampled = F.grid_sample(tangent_x_exp, grid, align_corners=False).squeeze()
    t_y_sampled = F.grid_sample(tangent_y_exp, grid, align_corners=False).squeeze()
    grad_sampled = F.grid_sample(grad_norm_exp, grid, align_corners=False).squeeze()
    
    t_hat = torch.stack([t_x_sampled, t_y_sampled], dim=-1)
    return t_hat, grad_sampled

def edge_align_loss(
    mu: torch.Tensor,
    cov: torch.Tensor,
    alpha_target: torch.Tensor,
    view_T: Union[np.ndarray, torch.Tensor],
    W: int,
    H: int,
    tanfovx: float,
    tanfovy: float
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute silhouette edge alignment loss (DIFFERENTIABLE).
    
    Aligns 2D projected covariance principal axes with silhouette edges
    detected from the alpha channel. All operations support autograd.
    
    Args:
        mu: (N, 3) 3D positions
        cov: (N, 3, 3) 3D covariances
        alpha_target: (H, W) target alpha channel
        view_T: (4, 4) view matrix (world to camera)
        W: Image width
        H: Image height
        tanfovx: Tangent of half horizontal FOV
        tanfovy: Tangent of half vertical FOV
    
    Returns:
        Tuple of (loss, info_dict)
            - loss: Scalar edge alignment loss
            - info_dict: Dictionary with alignment statistics
    """
    device = mu.device
    N = mu.shape[0]
    
    # 1. Compute projection Jacobian (differentiable)
    J = compute_projection_jacobian(mu, view_T, tanfovx, tanfovy, W, H)
    
    # 2. Project covariance to screen space (differentiable)
    cov_2d = torch.bmm(torch.bmm(J, cov), J.transpose(1, 2))  # [N, 2, 2]
    
    # 3. Extract principal axis via eigen-decomposition (differentiable)
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_2d)
        v_max = eigenvectors[:, :, -1]  # [N, 2] - principal axis
    except Exception as e:
        print(f"[WARN] Eigendecomposition failed: {e}")
        return torch.tensor(0.0, device=device), {}
    
    # 4. Compute silhouette tangents from alpha gradients (differentiable)
    grad_x, grad_y, grad_norm = _compute_sobel_gradients(alpha_target, device)
    tangent_x, tangent_y = _compute_silhouette_tangents(grad_x, grad_y, grad_norm)
    
    # 5. Project points to screen and sample tangents (differentiable)
    screen_x, screen_y = _project_points_to_screen(mu, view_T, tanfovx, tanfovy, W, H, device)
    t_hat, grad_sampled = _sample_tangents_at_points(
        tangent_x, tangent_y, grad_norm, screen_x, screen_y, W, H
    )
    
    # 6. Compute alignment score (differentiable)
    alignment = torch.abs((v_max * t_hat).sum(dim=-1))
    
    # 7. Compute edge weighting (differentiable)
    edge_weight = grad_sampled / (grad_sampled.mean() + EPS_SAFE)
    edge_weight = torch.clamp(edge_weight, 0.0, 10.0)
    
    # 8. Compute loss (differentiable)
    loss = (edge_weight * (1.0 - alignment)).mean()
    
    info = {
        'edge_alignment_mean': alignment.mean().item(),
        'edge_weight_mean': edge_weight.mean().item(),
        'edge_grad_norm_mean': grad_sampled.mean().item(),
    }
    
    return loss, info

def covariance_spectral_loss(
    cov_pred: torch.Tensor,
    cov_target: torch.Tensor,
    mode: str = 'eigenvalue'
) -> torch.Tensor:
    """
    Align predicted and target covariance spectra (DIFFERENTIABLE).
    
    Compares eigenvalue spectra or Frobenius norm between predicted
    and target covariance matrices.
    
    Args:
        cov_pred: (N, 3, 3) predicted covariances
        cov_target: (N, 3, 3) target covariances
        mode: 'eigenvalue' or 'frobenius'
    
    Returns:
        loss: Scalar loss
    """
    device = cov_pred.device
    
    # Ensure device alignment
    cov_target = _safe_device_transfer(cov_target, device)
    
    # 🔥 CRITICAL: Detach target to prevent gradient flow
    # Target covariance (from curvature) should guide, not be optimized
    cov_target = cov_target.detach()
    
    # Validate inputs
    if not _validate_tensor(cov_target, "cov_target") or not _validate_tensor(cov_pred, "cov_pred"):
        print(f"[WARN] Skipping spectral loss due to invalid inputs")
        return torch.tensor(0.0, device=device)
    
    # Add regularization for numerical stability
    eps = 1e-6
    eye = torch.eye(3, device=device).unsqueeze(0)
    cov_pred_reg = cov_pred + eps * eye
    cov_target_reg = cov_target + eps * eye
    
    if mode == 'eigenvalue':
        try:
            # Compute eigenvalues (differentiable)
            eig_pred = torch.linalg.eigvalsh(cov_pred_reg)  # [N, 3]
            eig_target = torch.linalg.eigvalsh(cov_target_reg)  # [N, 3]
            
            if not _validate_tensor(eig_pred, "eig_pred") or not _validate_tensor(eig_target, "eig_target"):
                print(f"[WARN] Invalid eigenvalues, using Frobenius fallback")
                return F.mse_loss(cov_pred, cov_target)
            
            # Normalize for scale invariance (differentiable)
            eig_pred_norm = eig_pred / (eig_pred.sum(dim=-1, keepdim=True) + EPS_NORMALIZE)
            eig_target_norm = eig_target / (eig_target.sum(dim=-1, keepdim=True) + EPS_NORMALIZE)
            
            loss = F.l1_loss(eig_pred_norm, eig_target_norm)
        except Exception as e:
            print(f"[WARN] Eigenvalue computation failed: {e}, using Frobenius fallback")
            loss = F.mse_loss(cov_pred, cov_target)
    
    elif mode == 'frobenius':
        loss = F.mse_loss(cov_pred, cov_target)
    
    else:
        print(f"[WARN] Unknown mode '{mode}', returning zero loss")
        loss = torch.tensor(0.0, device=device)
    
    return loss


# ============================================================================
# Advanced Regularization
# ============================================================================
def covariance_regularization_advanced(
    cov: torch.Tensor,
    lambda_vol: float = 0.01,
    lambda_aniso: float = 0.005,
    target_vol: float = 1.0e-4,
    max_aniso_ratio: float = 10.0
) -> torch.Tensor:
    """
    Advanced covariance regularization (DIFFERENTIABLE).
    
    Regularizes volume (det(Σ)) and anisotropy (λ_max / λ_min).
    
    Args:
        cov: (N, 3, 3) covariance matrices
        lambda_vol: Volume regularization weight
        lambda_aniso: Anisotropy regularization weight
        target_vol: Target volume (determinant)
        max_aniso_ratio: Maximum allowed anisotropy ratio
    
    Returns:
        loss: Scalar regularization loss
    """
    # Volume regularization: det(Σ) ≈ target_vol (differentiable)
    det = torch.det(cov)
    loss_vol = lambda_vol * F.mse_loss(det, torch.full_like(det, target_vol))
    
    # Anisotropy cap: λ_max / λ_min < max_aniso_ratio (differentiable)
    eigvals = torch.linalg.eigvalsh(cov)  # [N, 3] ascending
    ratio = eigvals[:, -1] / (eigvals[:, 0] + EPS_SAFE)
    aniso_penalty = torch.clamp(ratio - max_aniso_ratio, min=0.0)
    loss_aniso = lambda_aniso * aniso_penalty.mean()
    
    return loss_vol + loss_aniso # differentiable