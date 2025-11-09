#!/usr/bin/env python3
"""
Example script showing how to use the edge diagnostic tools.

This script demonstrates:
1. Loading a target alpha channel
2. Analyzing edge strength
3. Visualizing edges
4. Running full diagnostics
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.edge_diagnostics import (
    analyze_alpha_edges,
    visualize_alpha_edges,
    diagnose_edge_alignment_loss
)


def example_1_analyze_edges():
    """
    Example 1: Quick edge strength analysis
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Quick Edge Strength Analysis")
    print("="*80 + "\n")

    # Load or create dummy alpha channel
    # In real usage, load from your target render:
    #   alpha_target = torch.load('output/target_alpha.pt')
    #   or extract from rendered image

    # For demo, create a synthetic alpha with a circle
    H, W = 512, 512
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )
    dist = torch.sqrt(x**2 + y**2)
    alpha_target = (dist < 0.5).float()  # Sharp circle edge

    # Analyze edge strength
    stats = analyze_alpha_edges(alpha_target, verbose=True)

    # Interpretation
    print("\nInterpretation:")
    if stats['grad_norm_mean'] > 0.1:
        print("✅ Strong edges - edge loss should work well")
        print("   Recommended: edge_loss_mode: 'detach_position'")
    elif stats['grad_norm_mean'] > 0.01:
        print("⚠️  Moderate edges - edge loss may be useful")
        print("   Recommended: Try 'detach_position', then 'gradient_boost' if needed")
    else:
        print("❌ Weak edges - edge loss may not be effective")
        print("   Recommended: Disable edge loss (w_edge: 0.0)")


def example_2_visualize_edges():
    """
    Example 2: Generate edge visualization
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Generate Edge Visualization")
    print("="*80 + "\n")

    # Create synthetic alpha with soft edges
    H, W = 512, 512
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )
    dist = torch.sqrt(x**2 + y**2)

    # Soft edge (blur)
    alpha_target = torch.sigmoid((0.5 - dist) * 10)

    # Visualize
    output_path = "debug/example_alpha_edges.png"
    visualize_alpha_edges(
        alpha_target,
        output_path=output_path,
        show=False,  # Set True to display interactively
        title="Example: Soft Edge Alpha Channel"
    )
    print(f"Visualization saved to: {output_path}")


def example_3_full_diagnostic():
    """
    Example 3: Full diagnostic with dummy 3D data
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Full Diagnostic Pipeline")
    print("="*80 + "\n")

    # Create dummy data
    H, W = 512, 512
    N = 1000  # Number of Gaussians

    # Dummy alpha with sharp edges
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )
    dist = torch.sqrt(x**2 + y**2)
    alpha_target = (dist < 0.7).float()

    # Dummy 3D positions (random)
    mu = torch.randn(N, 3) * 2.0

    # Dummy covariances (isotropic)
    cov = torch.eye(3).unsqueeze(0).repeat(N, 1, 1) * 0.1

    # Dummy view matrix (identity for simplicity)
    view_T = np.eye(4, dtype=np.float32)

    # Camera parameters
    tanfovx = 0.5
    tanfovy = 0.5

    # Run full diagnostic
    stats = diagnose_edge_alignment_loss(
        alpha_target, mu, cov, view_T, W, H, tanfovx, tanfovy,
        output_dir="debug/"
    )

    print("\nDiagnostic complete! Check debug/ folder for visualizations.")


def example_4_real_world_usage():
    """
    Example 4: How to integrate into your training loop
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Real-World Integration")
    print("="*80 + "\n")

    code_example = """
# In your training loop (e.g., run.py or training_loop.py):

from utils.edge_diagnostics import analyze_alpha_edges, visualize_alpha_edges

# After generating target render
target_render = gaussian_renderer.render(...)
alpha_target = target_render['alpha']

# Run diagnostics on first episode only
if episode == 1:
    print("\\n[DIAGNOSTICS] Analyzing edge strength...")
    edge_stats = analyze_alpha_edges(alpha_target, verbose=True)

    # Save visualization
    visualize_alpha_edges(
        alpha_target,
        output_path=f"{output_dir}/ep{episode:03d}_alpha_edges.png",
        title=f"Episode {episode} - Alpha Edges"
    )

    # Auto-configure edge loss based on edge strength
    if edge_stats['grad_norm_mean'] < 0.001:
        print("⚠️  WARNING: Edges too weak - consider disabling edge loss")
        print("   Set w_edge: 0.0 in your config")
    elif edge_stats['grad_norm_mean'] < 0.01:
        print("⚠️  WARNING: Weak edges detected")
        print("   Recommended: edge_loss_mode: 'gradient_boost'")
    else:
        print("✅ Edge strength sufficient")
        print("   Recommended: edge_loss_mode: 'detach_position'")
"""

    print(code_example)


if __name__ == "__main__":
    # Create debug directory
    Path("debug").mkdir(exist_ok=True)

    print("\n" + "="*80)
    print("EDGE LOSS DIAGNOSTIC EXAMPLES")
    print("="*80)

    # Run examples
    example_1_analyze_edges()
    example_2_visualize_edges()
    example_3_full_diagnostic()
    example_4_real_world_usage()

    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)
    print("\nCheck the following for more information:")
    print("  - configs/edge_loss_examples/EDGE_LOSS_MODES.md")
    print("  - EDGE_LOSS_FIX_SUMMARY.md")
    print("  - debug/ folder for visualizations")
    print("="*80 + "\n")
