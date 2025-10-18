"""
E2E Loss Manager with Silhouette Alignment & Curvature-based Target Covariance
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
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
CLAMP_SCALE = (0.001, 0.1)
CLAMP_NORMAL_SCALE = (0.001, 0.05)


# ============================================================================
# Helper Functions
# ============================================================================
def _safe_normalize(tensor: torch.Tensor, dim: int = -1, eps: float = EPS_NORMALIZE) -> torch.Tensor:
    """Safely normalize a tensor along the given dimension."""
    norm = torch.norm(tensor, dim=dim, keepdim=True)
    return tensor / torch.clamp(norm, min=eps)


def _align_tensor_shapes(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align shapes of prediction and target tensors."""
    if pred.ndim != target.ndim:
        if pred.ndim == 3 and target.ndim == 2:
            pred = pred.squeeze(0) if pred.shape[0] == 1 else pred[0]
        elif pred.ndim == 2 and target.ndim == 3:
            target = target.squeeze(0) if target.shape[0] == 1 else target[0]
    return pred, target


def _safe_device_transfer(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Safely transfer tensor to device if needed."""
    if tensor.device != device:
        return tensor.to(device)
    return tensor


def _validate_tensor(tensor: torch.Tensor, name: str) -> bool:
    """Check if tensor contains NaN or Inf values."""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"[WARN] {name} contains NaN/Inf values")
        return False
    return True


# ============================================================================
# Loss Manager
# ============================================================================
class E2ELossManager:
    """E2E Training Loss Manager with all improvements"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.weights = self._initialize_weights(config)
        self.schedule = config.get('schedule', 'constant')
        self.total_steps = config.get('total_steps', 100)
        self.current_step = 0
    
    def _initialize_weights(self, config: Dict) -> Dict[str, float]:
        """Initialize loss weights from config with defaults."""
        weights = {}
        for key, default_val in DEFAULT_WEIGHTS.items():
            weights[key] = float(config.get(key, default_val))
        return weights
    
    def get_weights(self) -> Dict[str, float]:
        """Get a copy of current weights."""
        return self.weights.copy()
    
    def update_weights(self, step: Optional[int] = None):
        """Update weights based on schedule."""
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
        """Apply linear schedule to weights."""
        params = SCHEDULE_PARAMS['linear']
        alpha_min, alpha_max = params['alpha_range']
        edge_min, edge_max = params['edge_range']
        
        self.weights['w_alpha'] = alpha_min + (alpha_max - alpha_min) * progress
        self.weights['w_edge'] = edge_min + (edge_max - edge_min) * progress
    
    def _update_cosine_schedule(self, progress: float):
        """Apply cosine schedule to weights."""
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
        Compute comprehensive render loss
        
        Args:
            pred: {'image', 'alpha', 'depth'} - predicted renders
            target: {'image', 'alpha', 'depth', 'cov_target'} - targets
            cov: (N,3,3) covariances
            mu: (N,3) 3D positions (for edge alignment)
            view_params: {'view_T', 'W', 'H', 'tanfovx', 'tanfovy'}
            cov_target: (N,3,3) target covariances from curvature
        """
        losses = {}
        total = torch.tensor(0.0, device=self._get_device(pred, cov, mu))
        
        # Compute individual loss components
        total += self._compute_alpha_loss(pred, target, losses)
        total += self._compute_depth_loss(pred, target, losses)
        total += self._compute_photo_loss(pred, target, losses)
        total += self._compute_edge_loss(pred, target, mu, cov, view_params, losses)
        total += self._compute_cov_align_loss(cov, cov_target, target, losses)
        total += self._compute_cov_reg_loss(cov, losses)
        
        losses['loss_render_total'] = total
        return losses
    
    def _get_device(self, pred: Dict, cov: Optional[torch.Tensor], mu: Optional[torch.Tensor]) -> torch.device:
        """Get device from available tensors."""
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
        """Compute alpha channel loss."""
        if self.weights['w_alpha'] <= 0 or 'alpha' not in pred or 'alpha' not in target:
            losses['loss_alpha'] = torch.tensor(0.0)
            return torch.tensor(0.0)
        
        pred_alpha, target_alpha = _align_tensor_shapes(pred['alpha'], target['alpha'])
        loss_alpha = F.l1_loss(pred_alpha, target_alpha)
        losses['loss_alpha'] = loss_alpha
        return self.weights['w_alpha'] * loss_alpha
    
    def _compute_depth_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute depth loss with validity masking."""
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
        """Compute photometric loss."""
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
        """Compute silhouette edge alignment loss."""
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
        """Compute covariance spectral alignment loss."""
        if self.weights['w_cov_align'] <= 0 or cov is None:
            losses['loss_cov_align'] = torch.tensor(0.0)
            return torch.tensor(0.0)
        
        # Get cov_target from argument or target dict
        if cov_target is None and 'cov_target' in target:
            cov_target = target['cov_target']
        
        if cov_target is not None:
            loss_cov_align = covariance_spectral_loss(cov, cov_target)
            losses['loss_cov_align'] = loss_cov_align
            return self.weights['w_cov_align'] * loss_cov_align
        else:
            device = cov.device
            losses['loss_cov_align'] = torch.tensor(0.0, device=device)
            return torch.tensor(0.0, device=device)
    
    def _compute_cov_reg_loss(
        self,
        cov: Optional[torch.Tensor],
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute covariance regularization loss."""
        if self.weights['w_cov_reg'] <= 0 or cov is None:
            losses['loss_cov_reg'] = torch.tensor(0.0)
            return torch.tensor(0.0)
        
        loss_cov_reg = self._covariance_regularization(cov)
        losses['loss_cov_reg'] = loss_cov_reg
        return self.weights['w_cov_reg'] * loss_cov_reg
    
    def _covariance_regularization(self, cov: torch.Tensor) -> torch.Tensor:
        """Regularize covariance matrices."""
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
            return torch.tensor(0.0, device=device)
    
    def _frobenius_regularization(
        self, 
        cov: torch.Tensor, 
        target_scale: float,
        device: torch.device
    ) -> torch.Tensor:
        """Frobenius norm regularization."""
        target_cov = (target_scale ** 2) * torch.eye(3, device=device).unsqueeze(0).expand_as(cov)
        return F.mse_loss(cov, target_cov)
    
    def _trace_regularization(self, cov: torch.Tensor, target_scale: float) -> torch.Tensor:
        """Trace regularization."""
        trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1)
        target_trace = 3 * (target_scale ** 2)
        return F.l1_loss(trace, torch.full_like(trace, target_trace))
    
    def _eigenvalue_regularization(self, cov: torch.Tensor, target_scale: float) -> torch.Tensor:
        """Eigenvalue regularization."""
        eigvals = torch.linalg.eigvalsh(cov)
        target_eig = target_scale ** 2
        return F.mse_loss(eigvals, torch.full_like(eigvals, target_eig))


# ============================================================================
# Silhouette Edge Alignment Loss
# ============================================================================
def compute_projection_jacobian(
    mu: torch.Tensor,
    view_T: np.ndarray,
    tanfovx: float,
    tanfovy: float,
    W: int,
    H: int
) -> torch.Tensor:
    """
    Compute projection Jacobian J = ∂(screen_xy)/∂(world_xyz)
    
    Args:
        mu: (N,3) world-space positions
        view_T: (4,4) view matrix (world to camera)
        tanfovx, tanfovy: tangent of half FOV
        W, H: image dimensions
    
    Returns:
        J: (N,2,3) Jacobian matrices
    """
    N = mu.shape[0]
    device = mu.device
    
    # Convert view matrix to torch
    view_T_torch = torch.from_numpy(view_T.astype(np.float32)).to(device)
    
    # Transform to camera space
    mu_hom = torch.cat([mu, torch.ones(N, 1, device=device)], dim=1)  # [N,4]
    mu_cam = (view_T_torch @ mu_hom.T).T  # [N,4]
    
    x, y, z = mu_cam[:, 0], mu_cam[:, 1], mu_cam[:, 2]
    z_safe = torch.clamp(z, min=CLAMP_MIN_DEPTH)
    
    # Compute Jacobian components
    z2 = z_safe * z_safe
    
    J_screen_cam = torch.zeros(N, 2, 3, device=device)
    J_screen_cam[:, 0, 0] = W / (2 * tanfovx * z_safe)
    J_screen_cam[:, 0, 2] = -W * x / (2 * tanfovx * z2)
    J_screen_cam[:, 1, 1] = -H / (2 * tanfovy * z_safe)
    J_screen_cam[:, 1, 2] = H * y / (2 * tanfovy * z2)
    
    # Chain rule: J = J_screen_cam @ R
    R = view_T_torch[:3, :3]  # [3,3]
    J = torch.bmm(J_screen_cam, R.unsqueeze(0).expand(N, -1, -1))  # [N,2,3]
    
    return J


def _compute_sobel_gradients(
    alpha_target: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Sobel gradients of alpha map."""
    # Ensure 2D shape
    if alpha_target.ndim == 3:
        if alpha_target.shape[0] == 1:
            alpha_target = alpha_target[0]
        elif alpha_target.shape[-1] == 1:
            alpha_target = alpha_target.squeeze(-1)
    
    alpha_target = _safe_device_transfer(alpha_target, device)
    alpha_target_expanded = alpha_target.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    
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
    
    grad_x = F.conv2d(alpha_target_expanded, sobel_x, padding=1).squeeze()
    grad_y = F.conv2d(alpha_target_expanded, sobel_y, padding=1).squeeze()
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + EPS_SAFE)
    
    return grad_x, grad_y, grad_norm


def _compute_silhouette_tangents(
    grad_x: torch.Tensor,
    grad_y: torch.Tensor,
    grad_norm: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute silhouette tangent vectors from gradients."""
    tangent_x = -grad_y / (grad_norm + EPS_SAFE)
    tangent_y = grad_x / (grad_norm + EPS_SAFE)
    return tangent_x, tangent_y


def _project_points_to_screen(
    mu: torch.Tensor,
    view_T: np.ndarray,
    tanfovx: float,
    tanfovy: float,
    W: int,
    H: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project 3D points to screen space."""
    N = mu.shape[0]
    mu_hom = torch.cat([mu, torch.ones(N, 1, device=device)], dim=1)
    view_T_torch = torch.from_numpy(view_T.astype(np.float32)).to(device)
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
    """Sample tangent vectors at projected point locations."""
    grid_x = 2.0 * screen_x / W - 1.0
    grid_y = 2.0 * screen_y / H - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)
    
    tangent_x_exp = tangent_x.unsqueeze(0).unsqueeze(0)
    tangent_y_exp = tangent_y.unsqueeze(0).unsqueeze(0)
    grad_norm_exp = grad_norm.unsqueeze(0).unsqueeze(0)
    
    t_x_sampled = F.grid_sample(tangent_x_exp, grid, align_corners=False).squeeze()
    t_y_sampled = F.grid_sample(tangent_y_exp, grid, align_corners=False).squeeze()
    grad_sampled = F.grid_sample(grad_norm_exp, grid, align_corners=False).squeeze()
    
    t_hat = torch.stack([t_x_sampled, t_y_sampled], dim=-1)
    return t_hat, grad_sampled


def edge_align_loss(
    mu: torch.Tensor,
    cov: torch.Tensor,
    alpha_target: torch.Tensor,
    view_T: np.ndarray,
    W: int,
    H: int,
    tanfovx: float,
    tanfovy: float,
    weight_mode: str = 'gradient',
    tau: float = 0.1
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute silhouette edge alignment loss.
    
    Aligns 2D projected covariance principal axes with silhouette edges
    detected from the alpha channel.
    """
    device = mu.device
    N = mu.shape[0]
    
    # 1. Compute projection Jacobian
    J = compute_projection_jacobian(mu, view_T, tanfovx, tanfovy, W, H)
    
    # 2. Project covariance to screen space
    cov_2d = torch.bmm(torch.bmm(J, cov), J.transpose(1, 2))  # [N,2,2]
    
    # 3. Extract principal axis via eigen-decomposition
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_2d)
        v_max = eigenvectors[:, :, -1]  # [N,2]
    except Exception as e:
        print(f"[WARN] Eigendecomposition failed: {e}")
        return torch.tensor(0.0, device=device), {}
    
    # 4. Compute silhouette tangents from alpha gradients
    grad_x, grad_y, grad_norm = _compute_sobel_gradients(alpha_target, device)
    tangent_x, tangent_y = _compute_silhouette_tangents(grad_x, grad_y, grad_norm)
    
    # 5. Project points to screen and sample tangents
    screen_x, screen_y = _project_points_to_screen(mu, view_T, tanfovx, tanfovy, W, H, device)
    t_hat, grad_sampled = _sample_tangents_at_points(
        tangent_x, tangent_y, grad_norm, screen_x, screen_y, W, H
    )
    
    # 6. Compute alignment score
    alignment = torch.abs((v_max * t_hat).sum(dim=-1))
    
    # 7. Compute edge weighting
    edge_weight = grad_sampled / (grad_sampled.mean() + EPS_SAFE)
    edge_weight = torch.clamp(edge_weight, 0.0, 10.0)
    
    # 8. Compute loss
    loss = (edge_weight * (1.0 - alignment)).mean()
    
    info = {
        'edge_alignment_mean': alignment.mean().item(),
        'edge_weight_mean': edge_weight.mean().item(),
        'edge_grad_norm_mean': grad_sampled.mean().item(),
    }
    
    return loss, info


# ============================================================================
# Curvature-based Target Covariance
# ============================================================================
def _compute_knn_curvature_estimates(
    mu: torch.Tensor,
    normals: torch.Tensor,
    k: int,
    indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute curvature estimates from k-nearest neighbors."""
    # Gather neighbors
    neighbor_points = mu[indices]  # [N,k,3]
    neighbor_normals = normals[indices]  # [N,k,3]
    
    # Compute normal variation
    center_normals = normals.unsqueeze(1)  # [N,1,3]
    normal_diff = neighbor_normals - center_normals  # [N,k,3]
    normal_diff_norm = torch.norm(normal_diff, dim=-1)  # [N,k]
    
    # Compute spatial extent
    center_points = mu.unsqueeze(1)  # [N,1,3]
    spatial_diff = neighbor_points - center_points  # [N,k,3]
    spatial_dist = torch.norm(spatial_diff, dim=-1)  # [N,k]
    
    return normal_diff_norm, spatial_dist


def _weighted_curvature_average(
    curvature_estimates: torch.Tensor,
    spatial_dist: torch.Tensor,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute weighted average curvature with standard deviation."""
    # Compute weights (closer neighbors = more important)
    weights = torch.exp(-spatial_dist / (spatial_dist.mean(dim=1, keepdim=True) + eps))
    weights = weights / (weights.sum(dim=1, keepdim=True) + eps)
    
    # Weighted mean curvature
    mean_curvature = (weights * curvature_estimates).sum(dim=1)  # [N]
    
    # Standard deviation for anisotropy
    curvature_std = torch.std(curvature_estimates, dim=1)  # [N]
    
    return mean_curvature, curvature_std


def estimate_curvature(
    mu: torch.Tensor,
    normals: torch.Tensor,
    k: int = 16
) -> torch.Tensor:
    """
    Estimate principal curvatures using ROBUST normal variation analysis.
    
    Args:
        mu: (N,3) surface points
        normals: (N,3) surface normals
        k: number of neighbors for curvature estimation
    
    Returns:
        curvatures: (N,2) principal curvatures [κ1, κ2]
    """
    N = mu.shape[0]
    device = mu.device
    
    # 1. KNN
    D = torch.cdist(mu, mu)  # [N,N]
    _, indices = torch.topk(D, k=min(k, N), dim=1, largest=False)  # [N,k]
    
    # 2. Compute curvature estimates
    normal_diff_norm, spatial_dist = _compute_knn_curvature_estimates(mu, normals, k, indices)
    
    # 3. Estimate curvature as ratio κ ≈ ||Δn|| / ||Δx||
    eps = 1e-6
    curvature_estimates = normal_diff_norm / (spatial_dist + eps)  # [N,k]
    
    # Remove self (first neighbor is always self with distance 0)
    curvature_estimates = curvature_estimates[:, 1:]  # [N,k-1]
    spatial_dist = spatial_dist[:, 1:]  # [N,k-1]
    
    # 4. Weighted average
    mean_curvature, curvature_std = _weighted_curvature_average(curvature_estimates, spatial_dist, eps)
    
    # 5. Create principal curvatures with slight anisotropy
    k1 = mean_curvature + 0.5 * curvature_std
    k2 = mean_curvature - 0.5 * curvature_std
    
    # 6. Clamp to reasonable range
    k1 = torch.clamp(k1, *CLAMP_CURVATURE)
    k2 = torch.clamp(k2, *CLAMP_CURVATURE)
    
    curvatures = torch.stack([k1, k2], dim=1)  # [N,2]
    
    # 7. Replace invalid values
    if not _validate_tensor(curvatures, "curvatures"):
        invalid_mask = torch.isnan(curvatures) | torch.isinf(curvatures)
        curvatures[invalid_mask] = 0.1
    
    return curvatures


def build_tangent_basis(normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build orthonormal tangent basis from normals (ROBUST version).
    
    Returns two orthonormal tangent vectors perpendicular to the normals.
    """
    N = normals.shape[0]
    device = normals.device
    eps = EPS_SAFE
    
    # Normalize input normals
    normals = _safe_normalize(normals, dim=-1, eps=eps)
    
    # Initial vector (avoid parallel to normal)
    a = torch.tensor([1., 0., 0.], device=device).expand(N, 3).clone()
    parallel = torch.abs(torch.einsum('nd,nd->n', normals, a)) > 0.9
    a[parallel] = torch.tensor([0., 1., 0.], device=device)
    
    # Gram-Schmidt orthogonalization
    t1 = a - torch.einsum('nd,nd->n', a, normals).unsqueeze(-1) * normals
    t1_norm = torch.norm(t1, dim=-1, keepdim=True)
    
    # Handle degenerate cases
    degenerate = (t1_norm.squeeze(-1) < eps)
    if degenerate.any():
        a_alt = torch.tensor([0., 0., 1.], device=device).expand(N, 3).clone()
        t1_alt = a_alt - torch.einsum('nd,nd->n', a_alt, normals).unsqueeze(-1) * normals
        t1[degenerate] = t1_alt[degenerate]
        t1_norm = torch.norm(t1, dim=-1, keepdim=True)
    
    t1 = t1 / (t1_norm + eps)
    
    # t2 = n × t1
    t2 = torch.cross(normals, t1, dim=1)
    t2 = _safe_normalize(t2, dim=-1, eps=eps)
    
    # Final validation
    if not _validate_tensor(t1, "tangent_t1"):
        t1[torch.isnan(t1) | torch.isinf(t1)] = torch.tensor([1., 0., 0.], device=device)
    if not _validate_tensor(t2, "tangent_t2"):
        t2[torch.isnan(t2) | torch.isinf(t2)] = torch.tensor([0., 1., 0.], device=device)
    
    return t1, t2


def create_target_covariance(
    mu: torch.Tensor,
    normals: torch.Tensor,
    curvatures: torch.Tensor,
    base_scale: float = 0.02,
    aniso_factor: float = 2.0
) -> torch.Tensor:
    """
    Create intentional anisotropic target Σ★ from surface geometry.
    
    Args:
        mu: (N,3) surface points
        normals: (N,3) surface normals
        curvatures: (N,2) principal curvatures [κ1, κ2]
        base_scale: base Gaussian scale
        aniso_factor: anisotropy amplification factor
    
    Returns:
        cov_target: (N,3,3) target covariances
    """
    N = mu.shape[0]
    device = mu.device
    
    # Validate inputs
    if not _validate_tensor(curvatures, "curvatures"):
        print(f"[WARN] Using default isotropic covariance")
        cov_default = (base_scale ** 2) * torch.eye(3, device=device).unsqueeze(0).expand(N, 3, 3)
        return cov_default
    
    # Build local frame: [t1, t2, n]
    t1, t2 = build_tangent_basis(normals)  # [N,3], [N,3]
    R = torch.stack([t1, t2, normals], dim=-1)  # [N,3,3]
    
    # Anisotropic scaling based on curvature
    k1, k2 = curvatures[:, 0], curvatures[:, 1]  # [N], [N]
    
    # Scale inversely with curvature: high curvature → narrow Gaussian
    s1 = base_scale / (1.0 + aniso_factor * torch.abs(k1))  # [N]
    s2 = base_scale / (1.0 + aniso_factor * torch.abs(k2))  # [N]
    s3 = torch.full_like(s1, base_scale * 0.3)  # [N]
    
    # Clamp to reasonable range
    s1 = torch.clamp(s1, *CLAMP_SCALE)
    s2 = torch.clamp(s2, *CLAMP_SCALE)
    s3 = torch.clamp(s3, *CLAMP_NORMAL_SCALE)
    
    # Create diagonal scale matrix
    S = torch.diag_embed(torch.stack([s1**2, s2**2, s3**2], dim=-1))  # [N,3,3]
    
    # Σ★ = R @ S @ R^T
    cov_target = torch.bmm(torch.bmm(R, S), R.transpose(1, 2))
    
    return cov_target


def covariance_spectral_loss(
    cov_pred: torch.Tensor,
    cov_target: torch.Tensor,
    mode: str = 'eigenvalue'
) -> torch.Tensor:
    """
    Align predicted and target covariance spectra.
    
    Args:
        cov_pred: (N,3,3) predicted covariances
        cov_target: (N,3,3) target covariances
        mode: 'eigenvalue' or 'frobenius'
    
    Returns:
        loss: scalar loss
    """
    device = cov_pred.device
    
    # Ensure device alignment
    cov_target = _safe_device_transfer(cov_target, device)
    
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
            eig_pred = torch.linalg.eigvalsh(cov_pred_reg)  # [N,3]
            eig_target = torch.linalg.eigvalsh(cov_target_reg)  # [N,3]
            
            if not _validate_tensor(eig_pred, "eig_pred") or not _validate_tensor(eig_target, "eig_target"):
                print(f"[WARN] Invalid eigenvalues, using Frobenius fallback")
                return F.mse_loss(cov_pred, cov_target)
            
            # Normalize for scale invariance
            eig_pred_norm = eig_pred / (eig_pred.sum(dim=-1, keepdim=True) + EPS_NORMALIZE)
            eig_target_norm = eig_target / (eig_target.sum(dim=-1, keepdim=True) + EPS_NORMALIZE)
            
            loss = F.l1_loss(eig_pred_norm, eig_target_norm)
        except Exception as e:
            print(f"[WARN] Eigenvalue computation failed: {e}, using Frobenius fallback")
            loss = F.mse_loss(cov_pred, cov_target)
    
    elif mode == 'frobenius':
        loss = F.mse_loss(cov_pred, cov_target)
    
    else:
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
    Advanced covariance regularization.
    
    Regularizes volume (det(Σ)) and anisotropy (λ_max / λ_min).
    """
    # Volume regularization: det(Σ) ≈ target_vol
    det = torch.det(cov)
    loss_vol = lambda_vol * F.mse_loss(det, torch.full_like(det, target_vol))
    
    # Anisotropy cap: λ_max / λ_min < max_aniso_ratio
    eigvals = torch.linalg.eigvalsh(cov)  # [N,3] ascending
    ratio = eigvals[:, -1] / (eigvals[:, 0] + EPS_SAFE)
    aniso_penalty = torch.clamp(ratio - max_aniso_ratio, min=0.0)
    loss_aniso = lambda_aniso * aniso_penalty.mean()
    
    return loss_vol + loss_aniso