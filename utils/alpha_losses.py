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
import torch.nn.functional as F
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


def _sobel_edge_magnitude(alpha: torch.Tensor) -> torch.Tensor:
    """Differentiable Sobel edge magnitude for a single alpha image."""
    x = alpha[None, None]
    kx = alpha.new_tensor([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ]).view(1, 1, 3, 3) / 8.0
    ky = alpha.new_tensor([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ]).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-12)[0, 0]


def edge_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
    band_width: int = 5,
    edge_threshold: float = 0.02,
) -> torch.Tensor:
    """
    Align predicted and target silhouette contours.

    The loss is evaluated on a dilated union of target and predicted edge bands
    so that both missed thin structures and spurious protrusions are penalized.
    """
    pred_edge = _sobel_edge_magnitude(pred)
    target_edge = _sobel_edge_magnitude(target.detach())
    diff = (pred_edge - target_edge).abs()

    if band_width > 1:
        pad = band_width // 2
        target_band = F.max_pool2d(
            (target_edge[None, None] > edge_threshold).float(),
            kernel_size=band_width, stride=1, padding=pad
        )[0, 0] > 0
        pred_band = F.max_pool2d(
            (pred_edge.detach()[None, None] > edge_threshold).float(),
            kernel_size=band_width, stride=1, padding=pad
        )[0, 0] > 0
        band = target_band | pred_band
    else:
        band = (target_edge > edge_threshold) | (pred_edge.detach() > edge_threshold)

    if mask is not None:
        band = band | mask

    return diff[band].mean() if band.any() else diff.mean()


def combined_alpha_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    dt_map: torch.Tensor = None,
    w_bce: float = 1.0,
    w_iou: float = 0.5,
    w_dt: float = 0.1,
    w_edge: float = 0.0,
    mask_threshold: float = 0.0,
    auto_pos_weight: bool = True,
    edge_band_width: int = 5,
    edge_threshold: float = 0.02,
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

    loss = pred.new_zeros(())
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

    if w_edge > 0:
        l_edge = edge_alignment_loss(
            pred, target, mask,
            band_width=max(int(edge_band_width), 1),
            edge_threshold=float(edge_threshold),
        )
        loss = loss + w_edge * l_edge
        metrics['loss_edge'] = float(l_edge.item())

    metrics['loss_total'] = float(loss.item())
    if mask is not None:
        metrics['fg_pixels'] = int(mask.sum().item())
        metrics['total_pixels'] = mask.numel()

    return loss, metrics
