#!/usr/bin/env python3
"""
Quick test to verify surface mask calculation logic.
"""
import torch
import numpy as np

# Simulate your scenario
num_total = 150000
grad_mean = 1.23e-03

# Generate synthetic gradient samples (roughly matching your distribution)
torch.manual_seed(42)
grad_sampled = torch.abs(torch.randn(num_total)) * grad_mean * 0.5 + grad_mean * 0.5

# Apply threshold (80% of mean)
edge_threshold = grad_mean * 0.8
surface_mask = grad_sampled > edge_threshold

# Calculate statistics
num_surface = surface_mask.sum().item()
num_total_actual = surface_mask.shape[0]
surface_ratio = num_surface / num_total_actual

print("="*60)
print("Surface Mask Calculation Test")
print("="*60)
print(f"Total particles:        {num_total_actual:,}")
print(f"grad_sampled mean:      {grad_sampled.mean().item():.2e}")
print(f"grad_sampled std:       {grad_sampled.std().item():.2e}")
print(f"grad_sampled max:       {grad_sampled.max().item():.2e}")
print(f"Edge threshold (0.8×):  {edge_threshold:.2e}")
print()
print(f"Surface particles:      {num_surface:,} ({surface_ratio*100:.1f}%)")
print(f"Volume particles:       {num_total_actual - num_surface:,} ({(1-surface_ratio)*100:.1f}%)")
print()
print(f"Calculation: num_surface = (grad_sampled > {edge_threshold:.2e}).sum()")
print(f"Result:      num_surface = {num_surface:,}")
print("="*60)

# Show distribution
percentiles = [10, 25, 50, 75, 90, 95, 99]
print("\nGradient Distribution:")
for p in percentiles:
    val = torch.quantile(grad_sampled, p/100.0).item()
    above_threshold = "✓ SURFACE" if val > edge_threshold else "✗ volume"
    print(f"  {p:2d}th percentile: {val:.2e}  {above_threshold}")
