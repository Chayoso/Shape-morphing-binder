
# composite_utils.py
# -----------------------------------------------------------------------------
# Depth-aware compositing of rendered Gaussian splats with a background.
# Supports (a) plain alpha blending with a background image, and
#          (b) Z-test compositing using a background depth map.
# -----------------------------------------------------------------------------

from __future__ import annotations
import numpy as np

def _to_np_img(img):
    img = np.asarray(img)
    if img.dtype != np.float32:
        img = img.astype(np.float32) / (255.0 if img.dtype != np.float32 else 1.0)
    return np.clip(img, 0.0, 1.0)

def resize_to(img, H, W):
    # Pillow-only resize to allow pure-Python envs
    try:
        from PIL import Image
        mode = 'RGB' if img.ndim == 3 and img.shape[2] == 3 else 'L'
        im = Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8)) if img.dtype==np.float32 else Image.fromarray(img)
        im = im.resize((W, H), resample=Image.BILINEAR if mode=='RGB' else Image.NEAREST)
        out = np.asarray(im)
        if out.ndim == 2:
            out = out[..., None]
        if out.shape[2] == 1:
            out = out[..., 0]
        if out.dtype != np.float32:
            out = out.astype(np.float32) / 255.0
        return out
    except Exception:
        # Best-effort nearest neighbor without Pillow
        y = np.linspace(0, img.shape[0]-1, H).astype(np.int32)
        x = np.linspace(0, img.shape[1]-1, W).astype(np.int32)
        return img[np.ix_(y, x)]

def composite_with_background(fg_rgb: np.ndarray,
                              fg_alpha: np.ndarray | None,
                              fg_depth: np.ndarray | None,
                              bg_rgb: np.ndarray | None,
                              bg_depth: np.ndarray | None,
                              eps: float = 1e-6) -> np.ndarray:
    """
    Composite foreground (splat image) with background.
    Shapes:
      - fg_rgb   : (H,W,3) float32 [0,1]
      - fg_alpha : (H,W)   float32 [0,1] or None
      - fg_depth : (H,W)   float32  z>0 in camera space, 0 for invalid/empty
      - bg_rgb   : (H,W,3) float32 [0,1] or None
      - bg_depth : (H,W)   float32  z>0, same space as fg_depth, or None
    Returns:
      - (H,W,3) float32 composited image in [0,1].
    """
    H, W, _ = fg_rgb.shape
    if bg_rgb is None:
        return fg_rgb
    if bg_rgb.shape[:2] != (H,W):
        bg_rgb = resize_to(bg_rgb, H, W)
    bg_rgb = _to_np_img(bg_rgb)
    if bg_rgb.ndim == 2:
        bg_rgb = np.stack([bg_rgb]*3, axis=-1)

    if bg_depth is None or fg_depth is None:
        # Alpha-over compositing
        if fg_alpha is None:
            return fg_rgb * 1.0 + (1.0 - 1.0) * bg_rgb
        a = fg_alpha.astype(np.float32)
        if a.ndim == 3: a = a[...,0]
        a = np.clip(a, 0.0, 1.0)
        out = fg_rgb + (1.0 - a)[...,None] * bg_rgb
        return np.clip(out, 0.0, 1.0)

    # Z-test compositing
    d_fg = fg_depth.astype(np.float32)
    d_bg = bg_depth.astype(np.float32)
    valid_fg = d_fg > eps
    valid_bg = d_bg > eps

    # Default to background
    out = bg_rgb.copy()

    # Where FG is in front of BG (or BG invalid) -> use FG
    front = valid_fg & (~valid_bg | (d_fg <= d_bg - eps))
    out[front] = fg_rgb[front]

    # Where BG is in front (and FG valid), keep BG (already default)
    # Where neither valid, fall back to alpha if present
    neither = (~valid_fg) & (~valid_bg)
    if fg_alpha is not None:
        a = np.clip(fg_alpha.astype(np.float32), 0.0, 1.0)
        if a.ndim == 3: a = a[...,0]
        out[neither] = fg_rgb[neither] + (1.0 - a[neither])[...,None] * bg_rgb[neither]
    return np.clip(out, 0.0, 1.0)