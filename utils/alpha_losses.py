"""
Alpha silhouette loss functions for hard-coupled physics-to-render pipeline.

Implements losses from paper Section 5.1:
  - Binary Cross-Entropy (BCE)
  - Soft IoU
  - Distance Transform (DT)
  - Combined weighted loss with foreground masking
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def compute_dt_map(target: torch.Tensor) -> torch.Tensor:
    """
    Precompute signed distance transform from target silhouette boundary.

    Positive outside target, negative inside.  Result is normalized to [-1, 1].
    Call once per episode (target doesn't change within an episode).

    Args:
        target: (H, W) alpha mask in [0, 1]

    Returns:
        dt_map: (H, W) torch tensor on same device, normalized signed DT
    """
    tgt_np = target.detach().cpu().numpy()
    binary = (tgt_np > 0.5).astype(np.float64)

    dt_outside = distance_transform_edt(1.0 - binary)
    dt_inside = distance_transform_edt(binary)
    dt_signed = (dt_outside - dt_inside).astype(np.float32)

    max_val = np.abs(dt_signed).max()
    if max_val > 1e-8:
        dt_signed /= max_val

    return torch.from_numpy(dt_signed).to(target.device)


def bce_loss(pred: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor = None, eps: float = 1e-7,
             pos_weight: float = 1.0) -> torch.Tensor:
    """
    Weighted binary cross-entropy on alpha masks.

    pos_weight > 1 upweights foreground pixels, addressing class imbalance
    (e.g. pos_weight = bg_count/fg_count ≈ 24 for 4% foreground).
    """
    pred_c = pred.clamp(eps, 1.0 - eps)
    bce = -(pos_weight * target * torch.log(pred_c)
            + (1.0 - target) * torch.log(1.0 - pred_c))
    if mask is not None:
        return bce[mask].mean() if mask.any() else bce.mean()
    return bce.mean()


def soft_iou_loss(pred: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor = None, eps: float = 1e-7) -> torch.Tensor:
    """1 - soft IoU (intersection over union)."""
    if mask is not None and mask.any():
        p, t = pred[mask], target[mask]
    else:
        p, t = pred, target
    intersection = (p * t).sum()
    union = p.sum() + t.sum() - intersection
    return 1.0 - intersection / (union + eps)


def distance_transform_loss(pred: torch.Tensor, dt_map: torch.Tensor,
                            mask: torch.Tensor = None) -> torch.Tensor:
    """
    DT loss: pred weighted by signed distance to target boundary.

    Particles outside target boundary are penalized proportionally to distance.
    """
    weighted = pred * dt_map
    if mask is not None and mask.any():
        return weighted[mask].mean()
    return weighted.mean()


def combined_alpha_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    dt_map: torch.Tensor = None,
    w_bce: float = 1.0,
    w_iou: float = 0.5,
    w_dt: float = 0.1,
    mask_threshold: float = 0.0,
    auto_pos_weight: bool = True,
) -> tuple:
    """
    Combined alpha silhouette loss (paper Section 5.1).

    Args:
        pred: (H, W) predicted alpha
        target: (H, W) target alpha
        dt_map: (H, W) precomputed DT map (from compute_dt_map)
        w_bce, w_iou, w_dt: loss component weights
        mask_threshold: if > 0, compute loss only on foreground pixels
        auto_pos_weight: if True, weight fg pixels by bg/fg ratio (class imbalance fix)

    Returns:
        (total_loss, metrics_dict)
    """
    mask = None
    if mask_threshold > 0:
        mask = (target > mask_threshold) | (pred.detach() > mask_threshold)

    # Auto positive class weight: fg ≈ 4% → w+ = 96/4 = 24
    pos_weight = 1.0
    if auto_pos_weight:
        n_total = target.numel()
        n_fg = float((target > 0.5).sum().item())
        n_bg = n_total - n_fg
        if n_fg > 0:
            pos_weight = n_bg / n_fg  # e.g. 24 for 4% fg

    loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
    metrics = {'pos_weight': pos_weight}

    if w_bce > 0:
        l_bce = bce_loss(pred, target, mask, pos_weight=pos_weight)
        loss = loss + w_bce * l_bce
        metrics['loss_bce'] = float(l_bce.item())

    if w_iou > 0:
        l_iou = soft_iou_loss(pred, target, mask)
        loss = loss + w_iou * l_iou
        metrics['loss_iou'] = float(l_iou.item())

    if w_dt > 0 and dt_map is not None:
        l_dt = distance_transform_loss(pred, dt_map, mask)
        loss = loss + w_dt * l_dt
        metrics['loss_dt'] = float(l_dt.item())

    metrics['loss_total'] = float(loss.item())
    if mask is not None:
        metrics['fg_pixels'] = int(mask.sum().item())
        metrics['total_pixels'] = mask.numel()

    return loss, metrics


# ── Anti-collapse regularizers (paper Section 5.2) ───────────────────────

def volume_preservation_loss(F_t: torch.Tensor) -> torch.Tensor:
    """
    L_vol = mean((det(F) - 1)^2)

    Penalizes volume changes: det(F) should stay near 1.
    """
    det_F = torch.det(F_t)  # (N,)
    return torch.mean((det_F - 1.0) ** 2)


def deformation_penalty_loss(F_t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    L_F = mean(||log(sigma(F))||^2)

    Penalizes extreme stretching/compression via singular values of F.
    """
    # SVD: F = U @ diag(sigma) @ V^T
    _, sigma, _ = torch.linalg.svd(F_t)  # sigma: (N, 3)
    log_sigma = torch.log(sigma.clamp(min=eps))
    return torch.mean(log_sigma ** 2)


def temporal_smoothness_loss(
    x_curr: torch.Tensor,
    x_prev: torch.Tensor,
    x_prev2: torch.Tensor = None,
) -> torch.Tensor:
    """
    L_temp = mean(||x_{t} - 2*x_{t-1} + x_{t-2}||^2)

    Penalizes sudden acceleration. Falls back to velocity penalty
    if only two timesteps available: mean(||x_t - x_{t-1}||^2).
    """
    if x_prev2 is not None:
        accel = x_curr - 2.0 * x_prev + x_prev2
        return torch.mean(accel ** 2)
    else:
        vel = x_curr - x_prev
        return torch.mean(vel ** 2)


def compute_regularization(
    F_t: torch.Tensor,
    x_t: torch.Tensor = None,
    x_prev: torch.Tensor = None,
    x_prev2: torch.Tensor = None,
    w_vol: float = 1.0,
    w_def: float = 0.1,
    w_temp: float = 0.01,
) -> tuple:
    """
    Combined anti-collapse regularization.

    Returns:
        (total_reg_loss, metrics_dict)
    """
    loss = torch.tensor(0.0, device=F_t.device)
    metrics = {}

    if w_vol > 0:
        l_vol = volume_preservation_loss(F_t)
        loss = loss + w_vol * l_vol
        metrics['reg_vol'] = float(l_vol.item())

    if w_def > 0:
        l_def = deformation_penalty_loss(F_t)
        loss = loss + w_def * l_def
        metrics['reg_def'] = float(l_def.item())

    if w_temp > 0 and x_t is not None and x_prev is not None:
        l_temp = temporal_smoothness_loss(x_t, x_prev, x_prev2)
        loss = loss + w_temp * l_temp
        metrics['reg_temp'] = float(l_temp.item())

    metrics['reg_total'] = float(loss.item())
    return loss, metrics
