"""
Alpha silhouette loss functions for hard-coupled physics-to-render pipeline.

Implements losses from paper Section 5.1:
  - Binary Cross-Entropy (BCE)
  - Soft IoU
  - Distance Transform (DT) — projected manifold distance
  - Combined weighted loss with foreground masking
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


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


def compute_dt_map(target: torch.Tensor, blur_sigma: float = 0.0) -> torch.Tensor:
    """
    Signed distance transform from target silhouette boundary.

    Positive outside target, negative inside.  Optionally Gaussian-blurred
    to widen the basin of attraction (coarse-to-fine curriculum).

    Args:
        target: (H, W) binary target alpha (>0.5 = foreground).
        blur_sigma: if > 0, apply Gaussian blur to the DT map.

    Returns:
        (H, W) signed DT map on the same device as target.
    """
    tgt_np = (target.detach().cpu().numpy() > 0.5).astype(np.float64)
    dt_outside = distance_transform_edt(1.0 - tgt_np)   # distance from fg boundary, outside
    dt_inside = distance_transform_edt(tgt_np)           # distance from fg boundary, inside
    signed_dt = dt_outside - dt_inside                   # positive outside, negative inside
    dt_tensor = torch.from_numpy(signed_dt.astype(np.float32)).to(target.device)

    if blur_sigma > 0:
        ks = max(int(np.ceil(blur_sigma * 3)) * 2 + 1, 3)
        dt_tensor = _gaussian_blur2d(dt_tensor, ks, blur_sigma)

    return dt_tensor


def _gaussian_blur2d(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur to a 2D tensor."""
    half = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - half
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    return F.conv2d(x[None, None], kernel_2d, padding=half)[0, 0]


def dt_loss(pred: torch.Tensor, dt_map: torch.Tensor,
            mask: torch.Tensor = None) -> torch.Tensor:
    """
    Distance transform loss: penalizes predicted alpha weighted by signed
    distance to target boundary.

    L_dt = mean(pred * dt_map)

    - Pred alpha outside target (dt > 0): positive penalty → "move inward"
    - Pred alpha inside target (dt < 0): negative reward → "stay here"
    - Gradient magnitude proportional to distance from boundary.
    """
    loss_map = pred * dt_map
    if mask is not None and mask.any():
        return loss_map[mask].mean()
    return loss_map.mean()


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


def combined_alpha_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    w_bce: float = 1.0,
    w_iou: float = 0.5,
    w_edge: float = 0.0,
    w_dt: float = 0.0,
    dt_blur_sigma: float = 0.0,
    mask_threshold: float = 0.0,
    auto_pos_weight: bool = True,
    edge_band_width: int = 5,
    edge_threshold: float = 0.02,
    adaptive_grad_match: bool = False,
    grad_match_scale_min: float = 0.0,
    grad_match_scale_max: float = 100.0,
    grad_ratio_iou: float = None,
    grad_ratio_edge: float = None,
    grad_ratio_dt: float = None,
    bce_weight_map: torch.Tensor = None,
    bce_neg_weight: float = 1.0,
) -> tuple:
    """
    Combined alpha silhouette loss (paper Section 5.1).

    Args:
        pred: (H, W) predicted alpha
        target: (H, W) target alpha
        w_bce, w_iou: loss component weights
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

    l_bce = l_iou = l_edge = None

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

    if w_edge > 0:
        l_edge = edge_alignment_loss(
            pred, target, mask,
            band_width=max(int(edge_band_width), 1),
            edge_threshold=float(edge_threshold),
        )
        metrics['loss_edge'] = float(l_edge.item())

    l_dt = None
    if w_dt > 0:
        dt_map = compute_dt_map(target, blur_sigma=dt_blur_sigma)
        l_dt = dt_loss(pred, dt_map, mask)
        metrics['loss_dt'] = float(l_dt.item())
        metrics['dt_blur_sigma'] = float(dt_blur_sigma)

    # Determine anchor loss for adaptive gradient matching
    # If DT is primary (w_bce == 0 or very small), use DT as anchor
    use_dt_anchor = (w_dt > 0 and l_dt is not None and (l_bce is None or w_bce < w_dt))
    anchor_loss_term = None
    anchor_name = None

    if adaptive_grad_match and pred.requires_grad and torch.is_grad_enabled():
        if use_dt_anchor and l_dt is not None:
            anchor_loss_term = w_dt * l_dt
            anchor_name = 'dt'
        elif l_bce is not None:
            anchor_loss_term = w_bce * l_bce
            anchor_name = 'bce'

    if anchor_loss_term is not None:
        metrics['grad_anchor_alpha'] = 0.0
        metrics['grad_anchor_type'] = anchor_name

        def add_aux(name, aux_loss, ratio):
            nonlocal loss
            if aux_loss is None or ratio is None or float(ratio) <= 0:
                return
            scale, meta = adaptive_grad_scale(
                anchor_loss_term, aux_loss, [pred],
                target_ratio=float(ratio),
                scale_min=float(grad_match_scale_min),
                scale_max=float(grad_match_scale_max),
            )
            loss = loss + pred.new_tensor(scale) * aux_loss
            metrics[f'grad_scale_{name}'] = float(scale)
            metrics[f'grad_norm_{name}'] = float(meta['aux_grad_norm'])
            metrics['grad_anchor_alpha'] = float(meta['anchor_grad_norm'])

        add_aux('iou', l_iou, grad_ratio_iou if grad_ratio_iou is not None else w_iou)
        add_aux('edge', l_edge, grad_ratio_edge if grad_ratio_edge is not None else w_edge)
        # If DT is anchor, it's already in loss; if BCE is anchor, add DT as aux
        if anchor_name == 'bce' and l_dt is not None:
            add_aux('dt', l_dt, grad_ratio_dt if grad_ratio_dt is not None else w_dt)
        elif anchor_name == 'dt' and l_bce is not None:
            add_aux('bce_aux', l_bce, w_bce)  # BCE becomes auxiliary
    else:
        if l_iou is not None:
            loss = loss + w_iou * l_iou
        if l_edge is not None:
            loss = loss + w_edge * l_edge
        if l_dt is not None:
            loss = loss + w_dt * l_dt

    metrics['loss_total'] = float(loss.item())
    if mask is not None:
        metrics['fg_pixels'] = int(mask.sum().item())
        metrics['total_pixels'] = mask.numel()

    return loss, metrics
