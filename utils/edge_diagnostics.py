"""
Edge Diagnostics Utility for Alpha Channel Analysis

This module provides tools to visualize and diagnose edge detection
quality in alpha channels for the edge alignment loss.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional


def compute_sobel_edges(alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Sobel edge gradients for alpha channel.

    Args:
        alpha: (H, W) or (1, H, W) or (H, W, 1) alpha channel tensor

    Returns:
        Tuple of (grad_x, grad_y, grad_norm)
    """
    device = alpha.device

    # Ensure 2D shape
    if alpha.ndim == 3:
        if alpha.shape[0] == 1:
            alpha = alpha[0]
        elif alpha.shape[-1] == 1:
            alpha = alpha.squeeze(-1)

    alpha_expanded = alpha.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

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

    # Compute gradients
    grad_x = F.conv2d(alpha_expanded, sobel_x, padding=1).squeeze()
    grad_y = F.conv2d(alpha_expanded, sobel_y, padding=1).squeeze()
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

    return grad_x, grad_y, grad_norm


def analyze_alpha_edges(alpha: torch.Tensor, verbose: bool = True) -> Dict:
    """
    Analyze edge strength in alpha channel.

    Args:
        alpha: (H, W) alpha channel tensor
        verbose: Print detailed statistics

    Returns:
        Dictionary with edge statistics
    """
    grad_x, grad_y, grad_norm = compute_sobel_edges(alpha)

    # Compute statistics
    stats = {
        'grad_norm_mean': grad_norm.mean().item(),
        'grad_norm_max': grad_norm.max().item(),
        'grad_norm_min': grad_norm.min().item(),
        'grad_norm_std': grad_norm.std().item(),
        'strong_edge_pct_0.01': (grad_norm > 0.01).float().mean().item() * 100,
        'strong_edge_pct_0.05': (grad_norm > 0.05).float().mean().item() * 100,
        'strong_edge_pct_0.1': (grad_norm > 0.1).float().mean().item() * 100,
        'strong_edge_pct_0.5': (grad_norm > 0.5).float().mean().item() * 100,
        'alpha_mean': alpha.mean().item(),
        'alpha_std': alpha.std().item(),
        'alpha_min': alpha.min().item(),
        'alpha_max': alpha.max().item(),
    }

    if verbose:
        print("="*60)
        print("ALPHA CHANNEL EDGE ANALYSIS")
        print("="*60)
        print(f"Alpha statistics:")
        print(f"  Mean: {stats['alpha_mean']:.4f}")
        print(f"  Std:  {stats['alpha_std']:.4f}")
        print(f"  Range: [{stats['alpha_min']:.4f}, {stats['alpha_max']:.4f}]")
        print()
        print(f"Edge gradient statistics:")
        print(f"  Mean: {stats['grad_norm_mean']:.6e}")
        print(f"  Max:  {stats['grad_norm_max']:.6e}")
        print(f"  Std:  {stats['grad_norm_std']:.6e}")
        print()
        print(f"Edge coverage (% of pixels with gradient > threshold):")
        print(f"  > 0.01: {stats['strong_edge_pct_0.01']:.2f}%")
        print(f"  > 0.05: {stats['strong_edge_pct_0.05']:.2f}%")
        print(f"  > 0.10: {stats['strong_edge_pct_0.1']:.2f}%")
        print(f"  > 0.50: {stats['strong_edge_pct_0.5']:.2f}%")
        print("="*60)

        # Diagnosis
        if stats['grad_norm_mean'] < 0.01:
            print("⚠️  WARNING: Very weak edges detected!")
            print("   → Alpha channel may be too smooth/blurry")
            print("   → Edge alignment loss may not be useful")
        elif stats['grad_norm_mean'] < 0.05:
            print("⚠️  WARNING: Weak edges detected")
            print("   → Consider increasing edge loss weight")
        else:
            print("✅ Edge strength looks good")
        print("="*60)

    return stats


def visualize_alpha_edges(
    alpha: torch.Tensor,
    output_path: Optional[str] = None,
    show: bool = False,
    title: str = "Alpha Channel Edge Analysis"
):
    """
    Visualize alpha channel and its edge gradients.

    Args:
        alpha: (H, W) alpha channel tensor
        output_path: Path to save visualization (optional)
        show: Display plot interactively
        title: Plot title
    """
    # Move to CPU and convert to numpy
    alpha_np = alpha.detach().cpu().numpy()

    # Compute edges
    grad_x, grad_y, grad_norm = compute_sobel_edges(alpha)
    grad_x_np = grad_x.detach().cpu().numpy()
    grad_y_np = grad_y.detach().cpu().numpy()
    grad_norm_np = grad_norm.detach().cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)

    # Plot alpha channel
    im0 = axes[0, 0].imshow(alpha_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Alpha Channel')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # Plot grad_x
    im1 = axes[0, 1].imshow(grad_x_np, cmap='RdBu_r')
    axes[0, 1].set_title('Gradient X (Sobel)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # Plot grad_y
    im2 = axes[0, 2].imshow(grad_y_np, cmap='RdBu_r')
    axes[0, 2].set_title('Gradient Y (Sobel)')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Plot gradient magnitude
    im3 = axes[1, 0].imshow(grad_norm_np, cmap='hot')
    axes[1, 0].set_title('Gradient Magnitude')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    # Plot thresholded edges (0.1 threshold)
    edges_01 = (grad_norm_np > 0.1).astype(float)
    im4 = axes[1, 1].imshow(edges_01, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Strong Edges (gradient > 0.1)')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    # Plot histogram of gradient magnitudes
    axes[1, 2].hist(grad_norm_np.flatten(), bins=100, color='blue', alpha=0.7)
    axes[1, 2].axvline(0.01, color='green', linestyle='--', label='0.01')
    axes[1, 2].axvline(0.05, color='orange', linestyle='--', label='0.05')
    axes[1, 2].axvline(0.1, color='red', linestyle='--', label='0.1')
    axes[1, 2].set_xlabel('Gradient Magnitude')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Gradient Distribution')
    axes[1, 2].legend()
    axes[1, 2].set_yscale('log')

    plt.tight_layout()

    # Save or show
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Edge visualization saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def diagnose_edge_alignment_loss(
    alpha_target: torch.Tensor,
    mu: torch.Tensor,
    cov: torch.Tensor,
    view_T: np.ndarray,
    W: int,
    H: int,
    tanfovx: float,
    tanfovy: float,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Full diagnostic pipeline for edge alignment loss.

    Args:
        alpha_target: (H, W) target alpha channel
        mu: (N, 3) 3D positions
        cov: (N, 3, 3) covariances
        view_T: (4, 4) view transformation matrix
        W: Image width
        H: Image height
        tanfovx: Tangent of half horizontal FOV
        tanfovy: Tangent of half vertical FOV
        output_dir: Directory to save diagnostics (optional)

    Returns:
        Dictionary with diagnostic results
    """
    print("\n" + "="*80)
    print("EDGE ALIGNMENT LOSS DIAGNOSTICS")
    print("="*80)

    # 1. Analyze alpha edges
    print("\n[1/3] Analyzing alpha channel edges...")
    edge_stats = analyze_alpha_edges(alpha_target, verbose=True)

    # 2. Visualize edges
    if output_dir:
        print("\n[2/3] Generating edge visualizations...")
        viz_path = Path(output_dir) / "alpha_edge_analysis.png"
        visualize_alpha_edges(
            alpha_target,
            output_path=str(viz_path),
            title=f"Alpha Edge Analysis (mean grad: {edge_stats['grad_norm_mean']:.6e})"
        )

    # 3. Additional diagnostics
    print("\n[3/3] Additional diagnostics...")
    print(f"  Number of Gaussians: {mu.shape[0]}")
    print(f"  Image size: {W} x {H}")
    print(f"  Covariance shape: {cov.shape}")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)

    if edge_stats['grad_norm_mean'] < 0.001:
        print("❌ CRITICAL: Edges are extremely weak (mean < 0.001)")
        print("   → Edge alignment loss will NOT work effectively")
        print("   → Consider setting w_edge: 0.0 in config")
        print("   → Focus on other losses (alpha, cov_align, etc.)")
    elif edge_stats['grad_norm_mean'] < 0.01:
        print("⚠️  WARNING: Edges are very weak (mean < 0.01)")
        print("   → Try edge_loss_mode: 'gradient_boost' with high boost factor")
        print("   → Or reduce w_edge weight")
    elif edge_stats['strong_edge_pct_0.1'] < 1.0:
        print("⚠️  WARNING: Less than 1% of pixels have strong edges")
        print("   → Edge loss may have limited effectiveness")
    else:
        print("✅ Edge strength looks sufficient")
        print("   → Use edge_loss_mode: 'detach_position' (recommended)")
        print("   → If alignment doesn't improve, try 'gradient_boost'")

    print("="*80 + "\n")

    return edge_stats
