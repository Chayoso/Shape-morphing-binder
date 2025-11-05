"""
Visualization Utilities - Episode Visualization & Comparison

Handles visualization, comparison images, and stage exports.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from utils.io_utils import save_image_png, save_depth_png


# ============================================================================
# NOTE: Stage visualizations are handled in sampling/io/export.py
# via save_stage_progression() - more detailed with per-stage properties
# ============================================================================
# Episode Visualization
# ============================================================================

def save_episode_images(
    ep_dir: Path,
    ep: int,
    img_pred: Optional[np.ndarray],
    alpha_pred: Optional[np.ndarray],
    depth_pred: Optional[np.ndarray],
    normal_map_pred: Optional[np.ndarray]
) -> None:
    """
    Save episode rendered images.
    
    Args:
        ep_dir: Episode directory
        ep: Episode number
        img_pred: RGB image
        alpha_pred: Alpha channel
        depth_pred: Depth map
        normal_map_pred: Normal map
    """
    if img_pred is not None:
        save_image_png(ep_dir / f"ep{ep:03d}_render.png", img_pred)
    
    if alpha_pred is not None:
        alpha_vis = np.stack([alpha_pred]*3, axis=-1)
        save_image_png(ep_dir / f"ep{ep:03d}_alpha.png", alpha_vis)
    
    if depth_pred is not None:
        save_depth_png(ep_dir / f"ep{ep:03d}_depth.png", depth_pred, bits=16)
    
    if normal_map_pred is not None:
        save_image_png(ep_dir / f"ep{ep:03d}_normal.png", normal_map_pred)


def save_episode_comparisons(
    ep_dir: Path,
    ep: int,
    img_pred: Optional[np.ndarray],
    img_tgt: Optional[np.ndarray],
    alpha_pred: Optional[np.ndarray],
    alpha_tgt: Optional[np.ndarray]
) -> None:
    """
    Save before/after comparison images.
    
    Args:
        ep_dir: Episode directory
        ep: Episode number
        img_pred: Predicted RGB
        img_tgt: Target RGB
        alpha_pred: Predicted alpha
        alpha_tgt: Target alpha
    """
    if img_pred is None or img_tgt is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # RGB comparison
    axes[0, 0].imshow(img_pred)
    axes[0, 0].set_title(f'Predicted RGB (Episode {ep+1})')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_tgt)
    axes[0, 1].set_title('Target RGB')
    axes[0, 1].axis('off')
    
    # Alpha comparison
    if alpha_pred is not None and alpha_tgt is not None:
        axes[1, 0].imshow(alpha_pred, cmap='gray')
        axes[1, 0].set_title('Predicted Alpha')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(alpha_tgt, cmap='gray')
        axes[1, 1].set_title('Target Alpha')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(ep_dir / f"ep{ep:03d}_comparison.png", dpi=120)
    plt.close()


def create_axis_histogram(
    mu_np: np.ndarray,
    ep_dir: Path,
    ep: int
) -> None:
    """
    Create axis distribution histogram.
    
    Args:
        mu_np: Positions (N, 3)
        ep_dir: Episode directory
        ep: Episode number
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, axis_name in enumerate(['X', 'Y', 'Z']):
        axes[i].hist(mu_np[:, i], bins=50, alpha=0.7, color=['r', 'g', 'b'][i])
        axes[i].set_title(f'{axis_name}-axis Distribution')
        axes[i].set_xlabel(axis_name)
        axes[i].set_ylabel('Count')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ep_dir / f"ep{ep:03d}_axis_hist.png", dpi=120)
    plt.close()


# ============================================================================
# Main Visualization Function
# ============================================================================

def visualize_episode(
    ep: int,
    out_dir: Path,
    cg: Any,
    num_timesteps: int,
    rs_full: Dict,
    ema_state: Dict,
    renderer: Any,
    campos: np.ndarray,
    render_cfg: Dict,
    particle_color: list,
    png_enabled: bool,
    tgt: np.ndarray,
    loss_physics: float,
    seed: int,
    cov_module=None,
    external_levelset=None
) -> None:
    """
    Visualize and save episode results.
    
    Args:
        ep: Episode number
        out_dir: Output directory
        cg: Computation graph
        num_timesteps: Number of timesteps
        rs_full: Upsampling configuration
        ema_state: EMA state
        renderer: Renderer instance
        campos: Camera position
        render_cfg: Rendering configuration
        particle_color: RGB color
        png_enabled: Export PNG images
        tgt: Target positions
        loss_physics: Physics loss
        seed: Random seed
        cov_module: Optional learnable covariance module
    """
    from utils.rendering_utils import upsample_current_state, prepare_rendering_inputs
    from utils.io_utils import save_episode_data, save_episode_summary
    
    # out_dir is already the episode directory (created by run.py)
    ep_dir = out_dir
    ep_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Visualize] Episode {ep+1}")
    
    # Get final state
    pc = cg.get_point_cloud(num_timesteps - 1)
    
    # Upsample (export stages for visualization)
    mu, cov, result = upsample_current_state(
        pc, rs_full, ema_state, seed, cov_module, export_stages=True,
        external_levelset=None,
        current_episode=ep
    )
    
    if mu is None:
        print("  ⚠️ Upsampling failed")
        return
    
    # Convert to numpy
    mu_np = mu.detach().cpu().numpy() if hasattr(mu, 'detach') else np.asarray(mu)
    cov_np = cov.detach().cpu().numpy() if hasattr(cov, 'detach') else np.asarray(cov)
    
    # Save stage progression if available
    if "stage_outputs" in result:
        from sampling.io.export import save_stage_progression
        # Pass episode=-1 since ep_dir is already the episode directory
        save_stage_progression(ep_dir, -1, result["stage_outputs"])
    
    # Save data files
    save_episode_data(ep, ep_dir, mu_np, cov_np, particle_color)
    
    # Save summary
    try:
        x_torch = pc.get_positions_torch(requires_grad=False)
        F_torch = pc.get_def_grads_total_torch(requires_grad=False)
        save_episode_summary(ep, ep_dir, F_torch, mu_np, loss_physics)
    except:
        pass
    
    # Render if available
    if renderer is not None and png_enabled:
        with torch.no_grad():
            rgb = prepare_rendering_inputs(mu, result, campos, render_cfg, particle_color)
            
            pred_render = renderer.render(
                mu, cov, rgb=rgb,
                prefer_cov_precomp=True,
                return_torch=False
            )
            
            img_pred = pred_render.get('image')
            alpha_pred = pred_render.get('alpha')
            depth_pred = pred_render.get('depth')
            normal_map_pred = pred_render.get('normal_map')
            
            # Save images
            save_episode_images(ep_dir, ep, img_pred, alpha_pred, depth_pred, normal_map_pred)
            
            # Save comparisons (if target exists)
            target_dir = out_dir / "target"
            if (target_dir / "target_image.png").exists():
                try:
                    import imageio.v2 as iio
                    img_tgt = iio.imread(target_dir / "target_image.png") / 255.0
                    alpha_tgt = iio.imread(target_dir / "target_alpha.png")[..., 0] / 255.0
                    
                    save_episode_comparisons(ep_dir, ep, img_pred, img_tgt, alpha_pred, alpha_tgt)
                except:
                    pass
            
            # Create histogram
            create_axis_histogram(mu_np, ep_dir, ep)
    
    print(f"  ✅ Saved to {ep_dir}/")


__all__ = [
    'save_episode_images',
    'save_episode_comparisons',
    'create_axis_histogram',
    'visualize_episode',
]

