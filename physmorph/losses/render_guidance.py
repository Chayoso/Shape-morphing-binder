"""Render guidance: per-particle displacement from multi-view silhouette loss.

Makes the RENDERER load-bearing. The morph guidance combines a transport term
(3D correspondence) with this render term (multi-view silhouette/appearance
consistency), so render genuinely *drives* the morph — not just visualizes it.

We work in displacement space (d_R = -dD_img/dx) and normalize, so the famous
~45000x magnitude gap vs the 3D term is sidestepped: the relative weight is set
directly by `render_gain`, not by the raw gradient norms.
"""
from __future__ import annotations

import numpy as np
import torch

from .silhouette import _project, d_img, ring_thetas, soft_silhouette, target_silhouettes


# ---------------------------------------------------------------------------
# COLOR render: the renderer's UNIQUE, non-redundant job — appearance matching.
# Geometry transport can place particles correctly yet scramble color; the color
# render loss drives color-consistent correspondence (red -> red region) that
# geometry supervision fundamentally cannot provide.
# ---------------------------------------------------------------------------

def soft_color_image(x, colors, theta, res, extent):
    """Differentiable (res,res,3) color image + (res,res) coverage via CIC splat.
    colors (N,3) are constant; gradients flow through the splat weights -> x."""
    p = _project(x, theta)
    rel = (p + extent) / (2 * extent) * res
    base = torch.floor(rel).long(); frac = rel - base.float()
    cov = x.new_zeros(res * res); col = x.new_zeros(res * res, 3)
    for ox in (0, 1):
        wx = frac[:, 0] if ox else 1 - frac[:, 0]
        for oy in (0, 1):
            wy = frac[:, 1] if oy else 1 - frac[:, 1]
            w = wx * wy
            ii, jj = base[:, 0] + ox, base[:, 1] + oy
            valid = (ii >= 0) & (ii < res) & (jj >= 0) & (jj < res)
            idx = (ii * res + jj).clamp(0, res * res - 1)
            ww = torch.where(valid, w, torch.zeros_like(w))
            cov = cov.index_add(0, idx, ww)
            col = col.index_add(0, idx, ww[:, None] * colors)
    img = col / (cov[:, None] + 1e-4)
    return img.reshape(res, res, 3), cov.reshape(res, res)


def build_target_color_views(target_x, target_colors, n_views, res, extent, device="cuda"):
    thetas = ring_thetas(n_views)
    t = torch.as_tensor(np.ascontiguousarray(target_x, np.float32), device=device)
    c = torch.as_tensor(np.ascontiguousarray(target_colors, np.float32), device=device)
    imgs, alphas = [], []
    with torch.no_grad():
        for th in thetas:
            img, a = soft_color_image(t, c, float(th), res, extent)
            imgs.append(img.detach()); alphas.append(a.detach())
    return imgs, alphas, thetas


def d_img_color(x, colors, target_imgs, target_alphas, thetas, res, extent):
    loss = x.new_zeros(())
    for img_t, a_t, th in zip(target_imgs, target_alphas, thetas):
        img, _ = soft_color_image(x, colors, float(th), res, extent)
        m = (a_t > 0.05).float()[..., None]
        loss = loss + (((img - img_t) ** 2) * m).sum() / (m.sum() + 1.0)
    return loss / len(thetas)


def render_guidance_color(x, colors, target_imgs, target_alphas, thetas, res, extent, device="cuda"):
    """d_R (N,3) = -dD_color/dx : pull each particle so the COLOR render matches."""
    xt = torch.tensor(np.ascontiguousarray(x, np.float32), device=device, requires_grad=True)
    ct = torch.as_tensor(np.ascontiguousarray(colors, np.float32), device=device)
    L = d_img_color(xt, ct, target_imgs, target_alphas, thetas, res, extent)
    g = torch.autograd.grad(L, xt)[0]
    return (-g).detach().cpu().numpy().astype(np.float32)


def _color_gram(img, alpha, thr=0.05):
    """Per-view color statistic: 3x3 color covariance (Gram) + mean over covered pixels."""
    m = (alpha > thr).reshape(-1)
    p = img.reshape(-1, 3)[m]                                # (P,3) covered pixels
    if p.shape[0] < 4:
        return img.new_zeros(3, 3), img.new_zeros(3)
    mu = p.mean(0)
    pc = p - mu
    G = (pc.t() @ pc) / p.shape[0]
    return G, mu


def build_target_color_grams(style_x, style_colors, n_views, res, extent, device="cuda"):
    """Detached target color-Gram + mean per view (the appearance style signal)."""
    imgs, alphas, thetas = build_target_color_views(style_x, style_colors, n_views, res, extent, device)
    grams, means = [], []
    for img, a in zip(imgs, alphas):
        G, mu = _color_gram(img, a)
        grams.append(G.detach()); means.append(mu.detach())
    return grams, means, thetas


def color_gram_grad(x, colors, target_grams, target_means, thetas, res, extent,
                    w_mean=1.0, device="cuda"):
    """d_c (N,3) = -dL_colorGram/dcolors : RENDER restyles appearance. Matches the style's
    color *statistics* (palette/contrast), so render OWNS appearance (method.md S, R1)."""
    xt = torch.as_tensor(np.ascontiguousarray(x, np.float32), device=device)
    ct = torch.tensor(np.ascontiguousarray(colors, np.float32), device=device, requires_grad=True)
    L = ct.new_zeros(())
    for G_t, mu_t, th in zip(target_grams, target_means, thetas):
        img, a = soft_color_image(xt, ct, float(th), res, extent)
        G, mu = _color_gram(img, a)
        L = L + ((G - G_t) ** 2).sum() + w_mean * ((mu - mu_t) ** 2).sum()
    g = torch.autograd.grad(L, ct)[0]
    return (-g).detach().cpu().numpy().astype(np.float32), float(L.item())


def color_psnr(x, colors, target_x, target_colors, n_views=12, res=96, extent=None, device="cuda"):
    """Multi-view color PSNR (render's metric): how well the morph reproduces the
    target's *appearance*. Masked to the target silhouette."""
    if extent is None:
        extent = float(np.abs(np.ascontiguousarray(target_x, np.float32)).max()) * 1.25
    imgs_t, alphas_t, thetas = build_target_color_views(target_x, target_colors, n_views, res, extent, device)
    xt = torch.as_tensor(np.ascontiguousarray(x, np.float32), device=device)
    ct = torch.as_tensor(np.ascontiguousarray(colors, np.float32), device=device)
    mses = []
    with torch.no_grad():
        for img_t, a_t, th in zip(imgs_t, alphas_t, thetas):
            img, _ = soft_color_image(xt, ct, float(th), res, extent)
            m = (a_t > 0.05).float()[..., None]
            mses.append((((img - img_t) ** 2) * m).sum().item() / (m.sum().item() + 1.0))
    mse = float(np.mean(mses))
    return -10.0 * np.log10(mse + 1e-9)


def build_target_views(target_x, n_views, res, extent, device="cuda", k=1.5):
    """Render the target's multi-view silhouettes (the render supervision signal)."""
    thetas = ring_thetas(n_views)
    t = torch.as_tensor(np.ascontiguousarray(target_x, np.float32), device=device)
    return target_silhouettes(t, thetas, res, extent, k), thetas


def render_guidance(x, target_alphas, thetas, res, extent, k=1.5, device="cuda"):
    """d_R (N,3) = -dD_img/dx : the descent direction that improves the multi-view
    silhouette match. This is the render's per-particle 'pull' on the continuum."""
    xt = torch.tensor(np.ascontiguousarray(x, np.float32), device=device, requires_grad=True)
    L = d_img(xt, target_alphas, thetas, res, extent, k)
    g = torch.autograd.grad(L, xt)[0]
    return (-g).detach().cpu().numpy().astype(np.float32)


def silhouette_iou(x, target_x, n_views=12, res=96, extent=None, device="cuda", thr=0.3):
    """Mean multi-view silhouette IoU (floor-free render-space shape match).
    Measures how well the morph reproduces the target's *renders* (render's job)."""
    x = np.ascontiguousarray(x, np.float32); target_x = np.ascontiguousarray(target_x, np.float32)
    if extent is None:
        extent = float(np.abs(target_x).max()) * 1.25
    thetas = ring_thetas(n_views)
    xt = torch.as_tensor(x, device=device); tt = torch.as_tensor(target_x, device=device)
    ious = []
    with torch.no_grad():
        for th in thetas:
            a = (soft_silhouette(xt, float(th), res, extent) > thr)
            b = (soft_silhouette(tt, float(th), res, extent) > thr)
            inter = (a & b).sum().item(); union = (a | b).sum().item()
            ious.append(inter / max(union, 1))
    return float(np.mean(ious))
