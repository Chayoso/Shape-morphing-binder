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
    external_levelset=None,
    render_losses=None
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
        render_losses: Optional dict of render loss components
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

    # 🔥 NEW: Generate subdivision visualization (if available)
    if png_enabled:
        try:
            # Check if we have subdivision data (via PLY file)
            ply_files = list(ep_dir.glob("*_surface_*.ply"))
            if len(ply_files) > 0:
                print(f"  🎨 Generating subdivision visualization...")
                import subprocess
                import sys

                # Run subdivision visualization script
                result_viz = subprocess.run(
                    [sys.executable, "visualize_subdivision.py",
                     "--episode", str(ep_dir),
                     "--downsample", "0.1"],  # 10% for speed
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result_viz.returncode == 0:
                    print(f"  ✅ Subdivision visualization saved")
                else:
                    print(f"  ⚠️  Subdivision visualization failed: {result_viz.stderr[:200]}")
        except Exception as e:
            print(f"  ⚠️  Subdivision visualization error: {e}")

    # Save data files
    save_episode_data(ep, ep_dir, mu_np, cov_np, particle_color)

    # Save summary
    try:
        x_torch = pc.get_positions_torch(requires_grad=False)
        F_torch = pc.get_def_grads_total_torch(requires_grad=False)
        save_episode_summary(ep, ep_dir, F_torch, mu_np, loss_physics, render_losses)
    except:
        pass

    # 🔥 NEW: Generate multiscale F-field visualization (if enabled)
    if png_enabled:
        try:
            print(f"  🎨 Generating multiscale F-field visualization...")
            import subprocess
            import sys

            # Run F-field visualization script
            result_ffield = subprocess.run(
                [sys.executable, "visualize_F_field_from_gaussians.py",
                 "--episode_dir", str(ep_dir),
                 "--max_particles", "100000"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result_ffield.returncode == 0:
                print(f"  ✅ F-field visualization saved")
            else:
                print(f"  ⚠️  F-field visualization failed: {result_ffield.stderr[:200]}")
        except Exception as e:
            print(f"  ⚠️  F-field visualization error: {e}")
    
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

                    # Create matplotlib visualization comparison
                    save_matplotlib_comparison(ep_dir, ep, img_pred, img_tgt, alpha_pred, alpha_tgt, depth_pred, render_losses)
                except Exception as e:
                    print(f"  ⚠️ Comparison visualization failed: {e}")
            
            # Create histogram
            create_axis_histogram(mu_np, ep_dir, ep)
    
    print(f"  ✅ Saved to {ep_dir}/")


def save_matplotlib_comparison(
    ep_dir: Path,
    ep: int,
    img_pred: np.ndarray,
    img_tgt: np.ndarray,
    alpha_pred: np.ndarray,
    alpha_tgt: np.ndarray,
    depth_pred: Optional[np.ndarray],
    render_losses: Optional[Dict] = None
) -> None:
    """
    Create and save comprehensive matplotlib comparison visualization with render losses.

    Args:
        ep_dir: Episode directory
        ep: Episode number
        img_pred: Predicted RGB image
        img_tgt: Target RGB image
        alpha_pred: Predicted alpha channel
        alpha_tgt: Target alpha channel
        depth_pred: Predicted depth map (optional)
        render_losses: Render loss components (optional)
    """
    # Load target depth if available
    depth_tgt = None
    target_dir = ep_dir.parent / "target"
    target_depth_path = target_dir / "target_depth.png"

    if target_depth_path.exists() and depth_pred is not None:
        try:
            import imageio.v2 as iio
            depth_tgt = iio.imread(target_depth_path).astype(np.float32) / 65535.0
        except Exception as e:
            print(f"  ⚠️ Failed to load target depth: {e}")

    # Determine grid size based on available data
    has_depth = depth_pred is not None and depth_tgt is not None
    nrows = 4 if has_depth else 3

    # Create comprehensive figure
    fig, axes = plt.subplots(nrows, 3, figsize=(18, nrows * 5))
    fig.suptitle(f'Episode {ep+1} - Render Loss Comparison', fontsize=18, fontweight='bold')

    # ============================================================================
    # Row 1: RGB Comparison
    # ============================================================================
    axes[0, 0].imshow(img_pred)
    axes[0, 0].set_title('Predicted RGB', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_tgt)
    axes[0, 1].set_title('Target RGB', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    rgb_diff = np.abs(img_pred - img_tgt)
    rgb_mae = rgb_diff.mean()
    im_rgb_diff = axes[0, 2].imshow(rgb_diff, cmap='hot', vmin=0, vmax=min(1.0, rgb_diff.max()))
    axes[0, 2].set_title(f'RGB Difference\nMAE: {rgb_mae:.4f}', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im_rgb_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # ============================================================================
    # Row 2: Alpha Comparison
    # ============================================================================
    axes[1, 0].imshow(alpha_pred, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Predicted Alpha', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(alpha_tgt, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Target Alpha', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    alpha_diff = np.abs(alpha_pred - alpha_tgt)
    alpha_mae = alpha_diff.mean()
    im_alpha_diff = axes[1, 2].imshow(alpha_diff, cmap='hot', vmin=0, vmax=1)
    axes[1, 2].set_title(f'Alpha Difference\nMAE: {alpha_mae:.4f}', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im_alpha_diff, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # ============================================================================
    # Row 3: Depth Comparison (if available)
    # ============================================================================
    if has_depth:
        # Predicted depth
        im_depth_pred = axes[2, 0].imshow(depth_pred, cmap='viridis')
        axes[2, 0].set_title('Predicted Depth', fontsize=12, fontweight='bold')
        axes[2, 0].axis('off')
        plt.colorbar(im_depth_pred, ax=axes[2, 0], fraction=0.046, pad=0.04)

        # Target depth
        im_depth_tgt = axes[2, 1].imshow(depth_tgt, cmap='viridis')
        axes[2, 1].set_title('Target Depth', fontsize=12, fontweight='bold')
        axes[2, 1].axis('off')
        plt.colorbar(im_depth_tgt, ax=axes[2, 1], fraction=0.046, pad=0.04)

        # Depth difference (only where both valid)
        valid_mask = (depth_pred > 0) & (depth_tgt > 0)
        if valid_mask.sum() > 0:
            depth_diff = np.zeros_like(depth_pred)
            depth_diff[valid_mask] = np.abs(depth_pred[valid_mask] - depth_tgt[valid_mask])
            depth_mae = depth_diff[valid_mask].mean()

            im_depth_diff = axes[2, 2].imshow(depth_diff, cmap='hot', vmin=0, vmax=depth_diff.max())
            axes[2, 2].set_title(f'Depth Difference\nMAE: {depth_mae:.4f} ({valid_mask.sum()/valid_mask.size*100:.1f}% valid)',
                               fontsize=12, fontweight='bold')
            axes[2, 2].axis('off')
            plt.colorbar(im_depth_diff, ax=axes[2, 2], fraction=0.046, pad=0.04)
        else:
            axes[2, 2].text(0.5, 0.5, 'No valid depth overlap',
                          ha='center', va='center', fontsize=14, color='red')
            axes[2, 2].axis('off')

    # ============================================================================
    # Row 3/4: Error Heatmaps and Loss Breakdown
    # ============================================================================
    last_row = 3 if has_depth else 2

    # Combined error heatmap (RGB + Alpha weighted)
    combined_error = 0.6 * rgb_diff.mean(axis=-1) + 0.4 * alpha_diff
    im_combined = axes[last_row, 0].imshow(combined_error, cmap='hot', vmin=0, vmax=combined_error.max())
    axes[last_row, 0].set_title(f'Combined Error Heatmap\n(60% RGB + 40% Alpha)', fontsize=12, fontweight='bold')
    axes[last_row, 0].axis('off')
    plt.colorbar(im_combined, ax=axes[last_row, 0], fraction=0.046, pad=0.04)

    # Edge difference (Sobel edge detection)
    try:
        from scipy.ndimage import sobel

        # Convert to grayscale for edge detection
        alpha_pred_gray = alpha_pred
        alpha_tgt_gray = alpha_tgt

        # Compute edges
        edges_pred = np.hypot(sobel(alpha_pred_gray, axis=0), sobel(alpha_pred_gray, axis=1))
        edges_tgt = np.hypot(sobel(alpha_tgt_gray, axis=0), sobel(alpha_tgt_gray, axis=1))

        # Edge difference
        edge_diff = np.abs(edges_pred - edges_tgt)
        edge_mae = edge_diff.mean()

        im_edge_diff = axes[last_row, 1].imshow(edge_diff, cmap='hot')
        axes[last_row, 1].set_title(f'Edge Difference\nMAE: {edge_mae:.4f}', fontsize=12, fontweight='bold')
        axes[last_row, 1].axis('off')
        plt.colorbar(im_edge_diff, ax=axes[last_row, 1], fraction=0.046, pad=0.04)
    except:
        axes[last_row, 1].axis('off')

    # Loss breakdown text panel
    axes[last_row, 2].axis('off')

    if render_losses is not None:
        loss_text = "RENDER LOSS BREAKDOWN\n" + "="*40 + "\n\n"

        # Group losses by category
        loss_categories = {
            'Total': ['loss_render_total'],
            'Photometric': ['loss_photo', 'w_photo'],
            'Alpha/Silhouette': ['loss_alpha', 'w_alpha'],
            'Depth': ['loss_depth', 'w_depth'],
            'Edge': ['loss_edge', 'w_edge'],
            'Covariance': ['loss_cov_align', 'loss_cov_reg', 'w_cov_align', 'w_cov_reg'],
        }

        for category, keys in loss_categories.items():
            relevant_losses = {k: v for k, v in render_losses.items() if any(key in k for key in keys)}
            if relevant_losses:
                loss_text += f"{category}:\n"
                for key, val in relevant_losses.items():
                    if isinstance(val, (int, float)):
                        loss_text += f"  {key}: {val:.6f}\n"
                loss_text += "\n"

        # Add computed metrics
        loss_text += f"Computed Metrics:\n"
        loss_text += f"  RGB MAE: {rgb_mae:.6f}\n"
        loss_text += f"  Alpha MAE: {alpha_mae:.6f}\n"
        if has_depth and valid_mask.sum() > 0:
            loss_text += f"  Depth MAE: {depth_mae:.6f}\n"

        axes[last_row, 2].text(0.05, 0.5, loss_text, fontsize=9, family='monospace',
                             verticalalignment='center',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        axes[last_row, 2].text(0.5, 0.5, 'No render loss data available',
                             ha='center', va='center', fontsize=12, color='gray')

    plt.tight_layout()
    save_path = ep_dir / f"ep{ep:03d}_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Saved detailed render comparison to: {save_path.name}")


__all__ = [
    'save_episode_images',
    'save_episode_comparisons',
    'create_axis_histogram',
    'visualize_episode',
    'save_matplotlib_comparison',
]

