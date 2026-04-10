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
             pos_weight: float = 1.0,
             neg_weight: float = 1.0,
             return_map: bool = False) -> torch.Tensor:
    """
    Weighted binary cross-entropy on alpha masks.

    pos_weight > 1 upweights foreground pixels, addressing class imbalance
    (e.g. pos_weight = bg_count/fg_count ≈ 24 for 4% foreground).
    """
    pred_c = pred.clamp(eps, 1.0 - eps)
    bce = -(pos_weight * target * torch.log(pred_c)
            + neg_weight * (1.0 - target) * torch.log(1.0 - pred_c))
    if return_map:
        return bce
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


def gradient_norm(loss: torch.Tensor, tensors: list[torch.Tensor], eps: float = 1e-12) -> float | None:
    """
    Detached gradient norm of ``loss`` with respect to ``tensors``.

    Returns None when gradients are unavailable (e.g. no-grad metric pass).
    """
    if loss is None or (not torch.is_grad_enabled()):
        return None
    req = [t for t in tensors if t is not None and torch.is_tensor(t) and t.requires_grad]
    if not req:
        return None
    grads = torch.autograd.grad(loss, req, retain_graph=True, allow_unused=True, create_graph=False)
    total = 0.0
    any_grad = False
    for g in grads:
        if g is None:
            continue
        any_grad = True
        total += float(g.detach().pow(2).sum().item())
    if not any_grad:
        return None
    return float(np.sqrt(total + eps))


def adaptive_grad_scale(
    anchor_loss: torch.Tensor,
    aux_loss: torch.Tensor,
    tensors: list[torch.Tensor],
    target_ratio: float,
    scale_min: float = 0.0,
    scale_max: float = 100.0,
    eps: float = 1e-12,
) -> tuple[float, dict]:
    """
    Match auxiliary gradient norm to a target fraction of anchor gradient norm.
    """
    target_ratio = float(target_ratio)
    meta = {
        'anchor_grad_norm': 0.0,
        'aux_grad_norm': 0.0,
        'adaptive_scale': float(target_ratio),
    }
    if target_ratio <= 0:
        return 0.0, meta
    g_anchor = gradient_norm(anchor_loss, tensors, eps=eps)
    g_aux = gradient_norm(aux_loss, tensors, eps=eps)
    if g_anchor is None or g_aux is None or g_aux <= eps:
        return float(target_ratio), meta
    scale = target_ratio * g_anchor / max(g_aux, eps)
    scale = float(np.clip(scale, scale_min, scale_max))
    meta.update({
        'anchor_grad_norm': float(g_anchor),
        'aux_grad_norm': float(g_aux),
        'adaptive_scale': float(scale),
    })
    return scale, meta


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


def multiscale_silhouette_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    levels: int = 3,
    fine_weight: float = 1.0,
    coarse_weight: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """
    Laplacian-pyramid-style silhouette loss.

    This explicitly compares high-frequency silhouette bands so that thin
    protrusions and splits are not dominated by the low-frequency outline.
    """
    pred_lvl = pred[None, None]
    target_lvl = target[None, None]

    loss = pred.new_zeros(())
    total_w = 0.0
    metrics: dict[str, float] = {}

    max_levels = max(int(levels), 1)
    for level in range(max_levels - 1):
        if min(pred_lvl.shape[-2:]) < 2:
            break

        low_pred = F.avg_pool2d(pred_lvl, kernel_size=2, stride=2, ceil_mode=True)
        low_target = F.avg_pool2d(target_lvl, kernel_size=2, stride=2, ceil_mode=True)
        up_pred = F.interpolate(low_pred, size=pred_lvl.shape[-2:], mode='bilinear', align_corners=False)
        up_target = F.interpolate(low_target, size=target_lvl.shape[-2:], mode='bilinear', align_corners=False)

        band_pred = pred_lvl - up_pred
        band_target = target_lvl - up_target
        band_loss = (band_pred - band_target).abs().mean()
        band_weight = float(fine_weight) / float(2 ** level)

        loss = loss + band_weight * band_loss
        total_w += band_weight
        metrics[f'loss_ms_band_{level}'] = float(band_loss.item())

        pred_lvl = low_pred
        target_lvl = low_target

    coarse_loss = (pred_lvl - target_lvl).abs().mean()
    loss = loss + float(coarse_weight) * coarse_loss
    total_w += float(coarse_weight)
    metrics['loss_ms_coarse'] = float(coarse_loss.item())

    if total_w > 0:
        loss = loss / total_w
    metrics['loss_ms'] = float(loss.item())
    return loss, metrics


def _masked_box_blur2d(img: torch.Tensor, valid: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Box blur with mask normalization so invalid depth stays neutral."""
    k = max(int(kernel_size), 1)
    if k <= 1:
        return img
    if k % 2 == 0:
        k += 1
    pad = k // 2
    kernel = torch.ones((1, 1, k, k), dtype=img.dtype, device=img.device)
    x = img[None, None]
    m = valid.float()[None, None]
    num = F.conv2d(x * m, kernel, padding=pad)
    den = F.conv2d(m, kernel, padding=pad)
    return (num / den.clamp_min(1e-6))[0, 0]


def depth_curvature_map(
    depth: torch.Tensor,
    valid_mask: torch.Tensor = None,
    smooth_ks: int = 5,
    normalize: bool = True,
    output_quantile: float = 0.95,
) -> torch.Tensor:
    """
    Screen-space curvature proxy from depth via a masked Laplacian response.

    This is a generic depth-based curvature cue: thin protrusions and valleys
    produce larger second-order variation than smooth one-lobe blobs.
    """
    depth = depth.float()
    valid = (depth > 1e-6) if valid_mask is None else valid_mask.bool()
    if not valid.any():
        return torch.zeros_like(depth)

    depth_n = torch.zeros_like(depth)
    vals = depth[valid]
    if normalize:
        mean = vals.mean()
        std = vals.std(unbiased=False)
        depth_n[valid] = (vals - mean) / (std + 1e-6)
    else:
        depth_n[valid] = vals

    depth_s = _masked_box_blur2d(depth_n, valid, kernel_size=smooth_ks)
    lap = depth.new_tensor([
        [0.0, 1.0, 0.0],
        [1.0, -4.0, 1.0],
        [0.0, 1.0, 0.0],
    ]).view(1, 1, 3, 3)
    curv = F.conv2d(depth_s[None, None], lap, padding=1)[0, 0].abs()
    curv = curv * valid.float()

    if normalize:
        vals = curv[valid]
        if vals.numel() > 0:
            if vals.numel() > 8:
                q = torch.quantile(vals, float(output_quantile))
            else:
                q = vals.max()
            if float(q.item()) > 1e-8:
                curv = torch.clamp(curv / q, 0.0, 1.0)
    return curv


def depth_curvature_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    target_alpha: torch.Tensor,
    pred_alpha: torch.Tensor = None,
    target_curvature: torch.Tensor = None,
    mask_threshold: float = 0.01,
    band_width: int = 7,
    smooth_ks: int = 5,
    high_quantile: float = 0.75,
    edge_threshold: float = 0.02,
) -> tuple[torch.Tensor, dict]:
    """
    Curvature-aware depth loss focused on target contour/concavity regions.

    The target curvature comes from the per-view target observation, so the
    loss stays generic and screen-space rather than mesh-correspondence-based.
    """
    valid_t = target_depth > 1e-6
    valid_p = pred_depth > 1e-6
    if pred_alpha is not None:
        valid_p = valid_p & (pred_alpha > mask_threshold)

    if not valid_t.any():
        zero = pred_depth.new_zeros(())
        return zero, {'loss_curv_raw': 0.0, 'loss_curv_region_pixels': 0}

    tgt_curv = target_curvature
    if tgt_curv is None:
        tgt_curv = depth_curvature_map(target_depth, valid_mask=valid_t, smooth_ks=smooth_ks)
    pred_curv = depth_curvature_map(pred_depth, valid_mask=valid_p, smooth_ks=smooth_ks)

    target_edge = _sobel_edge_magnitude(target_alpha.detach())
    if band_width > 1:
        pad = band_width // 2
        edge_band = F.max_pool2d(
            (target_edge[None, None] > edge_threshold).float(),
            kernel_size=band_width, stride=1, padding=pad
        )[0, 0] > 0
    else:
        edge_band = target_edge > edge_threshold

    curv_vals = tgt_curv[valid_t]
    if curv_vals.numel() > 8:
        q = float(torch.quantile(curv_vals, float(high_quantile)).item())
    else:
        q = float(curv_vals.max().item())
    high_curv = tgt_curv >= q if q > 0 else valid_t
    if band_width > 1:
        pad = band_width // 2
        high_curv = F.max_pool2d(
            high_curv.float()[None, None],
            kernel_size=band_width, stride=1, padding=pad
        )[0, 0] > 0

    region = (edge_band | high_curv) & valid_t
    diff = (pred_curv - tgt_curv).abs()
    loss = diff[region].mean() if region.any() else diff[valid_t].mean()
    return loss, {
        'loss_curv_raw': float(loss.item()),
        'loss_curv_region_pixels': int(region.sum().item()),
        'loss_curv_target_q': q,
    }


def combined_alpha_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    dt_map: torch.Tensor = None,
    w_bce: float = 1.0,
    w_iou: float = 0.5,
    w_dt: float = 0.1,
    w_edge: float = 0.0,
    w_ms: float = 0.0,
    mask_threshold: float = 0.0,
    auto_pos_weight: bool = True,
    edge_band_width: int = 5,
    edge_threshold: float = 0.02,
    ms_levels: int = 3,
    ms_fine_weight: float = 1.0,
    ms_coarse_weight: float = 0.5,
    adaptive_grad_match: bool = False,
    grad_match_scale_min: float = 0.0,
    grad_match_scale_max: float = 100.0,
    grad_ratio_iou: float = None,
    grad_ratio_dt: float = None,
    grad_ratio_edge: float = None,
    grad_ratio_ms: float = None,
    bce_weight_map: torch.Tensor = None,
    bce_neg_weight: float = 1.0,
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
    metrics = {'pos_weight': pos_weight, 'bce_neg_weight': float(bce_neg_weight)}

    l_bce = l_iou = l_dt = l_edge = l_ms = None

    if w_bce > 0:
        if bce_weight_map is not None:
            bce_map = bce_loss(
                pred, target, pos_weight=pos_weight, neg_weight=bce_neg_weight, return_map=True
            )
            wmap = bce_weight_map.to(pred.device, dtype=pred.dtype)
            if wmap.shape != pred.shape:
                raise ValueError(f"bce_weight_map shape mismatch: expected {tuple(pred.shape)}, got {tuple(wmap.shape)}")
            active = mask if (mask is not None and mask.any()) else torch.ones_like(pred, dtype=torch.bool)
            active_weights = wmap[active]
            denom = active_weights.sum().clamp_min(1e-8)
            norm = active_weights.numel() / denom
            weighted_map = bce_map * wmap * norm
            l_bce = weighted_map[active].mean()
            metrics['bce_weight_mean'] = float(active_weights.mean().item())
            metrics['bce_weight_min'] = float(active_weights.min().item())
            metrics['bce_weight_max'] = float(active_weights.max().item())
            metrics['bce_weight_norm'] = float(norm.item())
        else:
            l_bce = bce_loss(pred, target, mask, pos_weight=pos_weight, neg_weight=bce_neg_weight)
        loss = loss + w_bce * l_bce
        metrics['loss_bce'] = float(l_bce.item())

    if w_iou > 0:
        l_iou = soft_iou_loss(pred, target, mask)
        metrics['loss_iou'] = float(l_iou.item())

    if w_dt > 0 and dt_map is not None:
        l_dt = distance_transform_loss(pred, dt_map, mask)
        metrics['loss_dt'] = float(l_dt.item())

    if w_edge > 0:
        l_edge = edge_alignment_loss(
            pred, target, mask,
            band_width=max(int(edge_band_width), 1),
            edge_threshold=float(edge_threshold),
        )
        metrics['loss_edge'] = float(l_edge.item())

    if w_ms > 0:
        l_ms, ms_metrics = multiscale_silhouette_loss(
            pred, target,
            levels=max(int(ms_levels), 1),
            fine_weight=float(ms_fine_weight),
            coarse_weight=float(ms_coarse_weight),
        )
        metrics.update(ms_metrics)

    if adaptive_grad_match and l_bce is not None and pred.requires_grad and torch.is_grad_enabled():
        anchor_loss = w_bce * l_bce
        metrics['grad_anchor_alpha'] = 0.0

        def add_aux(name, aux_loss, ratio):
            nonlocal loss
            if aux_loss is None or ratio is None or float(ratio) <= 0:
                return
            scale, meta = adaptive_grad_scale(
                anchor_loss, aux_loss, [pred],
                target_ratio=float(ratio),
                scale_min=float(grad_match_scale_min),
                scale_max=float(grad_match_scale_max),
            )
            loss = loss + pred.new_tensor(scale) * aux_loss
            metrics[f'grad_scale_{name}'] = float(scale)
            metrics[f'grad_norm_{name}'] = float(meta['aux_grad_norm'])
            metrics['grad_anchor_alpha'] = float(meta['anchor_grad_norm'])

        add_aux('iou', l_iou, grad_ratio_iou if grad_ratio_iou is not None else w_iou)
        add_aux('dt', l_dt, grad_ratio_dt if grad_ratio_dt is not None else w_dt)
        add_aux('edge', l_edge, grad_ratio_edge if grad_ratio_edge is not None else w_edge)
        add_aux('ms', l_ms, grad_ratio_ms if grad_ratio_ms is not None else w_ms)
    else:
        if l_iou is not None:
            loss = loss + w_iou * l_iou
        if l_dt is not None:
            loss = loss + w_dt * l_dt
        if l_edge is not None:
            loss = loss + w_edge * l_edge
        if l_ms is not None:
            loss = loss + w_ms * l_ms

    metrics['loss_total'] = float(loss.item())
    if mask is not None:
        metrics['fg_pixels'] = int(mask.sum().item())
        metrics['total_pixels'] = mask.numel()

    return loss, metrics
