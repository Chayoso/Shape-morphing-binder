#!/usr/bin/env python3
"""
Visual explanation of how percentile threshold determines surface particle count.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Simulate your gradient distribution
num_total = 150000
grad_mean = 1.23e-03

torch.manual_seed(42)
grad_sampled = torch.abs(torch.randn(num_total)) * grad_mean * 0.5 + grad_mean * 0.5

print("="*70)
print("PERCENTILE THRESHOLD → SURFACE PARTICLE COUNT CALCULATION")
print("="*70)

# ============================================================================
# Step 1: Understanding the gradient distribution
# ============================================================================
print("\n[STEP 1] Gradient Distribution")
print("-"*70)
print(f"Total particles: {num_total:,}")
print(f"grad_sampled shape: {grad_sampled.shape}")
print(f"grad_sampled range: [{grad_sampled.min().item():.2e}, {grad_sampled.max().item():.2e}]")
print(f"grad_sampled mean: {grad_sampled.mean().item():.2e}")
print(f"grad_sampled median: {grad_sampled.median().item():.2e}")

# ============================================================================
# Step 2: What is a percentile?
# ============================================================================
print("\n[STEP 2] What is a Percentile?")
print("-"*70)
print("The Pth percentile is the value below which P% of the data falls.")
print("")
print("Example with 90th percentile:")
sorted_grads = torch.sort(grad_sampled)[0]

percentile_90 = 0.9
index_90 = int(percentile_90 * num_total)
value_90 = sorted_grads[index_90].item()

print(f"  90th percentile = value at index {index_90:,}")
print(f"  90th percentile value = {value_90:.2e}")
print(f"  Meaning: 90% of particles have grad < {value_90:.2e}")
print(f"           10% of particles have grad > {value_90:.2e}")

# ============================================================================
# Step 3: Using percentile as threshold
# ============================================================================
print("\n[STEP 3] Using Percentile as Threshold")
print("-"*70)
print("When we set threshold = 90th percentile:")
print(f"  threshold = {value_90:.2e}")
print("")
print("Count particles with grad > threshold:")

surface_mask = grad_sampled > value_90
num_surface = surface_mask.sum().item()
surface_ratio = num_surface / num_total

print(f"  surface_mask = (grad_sampled > {value_90:.2e})")
print(f"  num_surface = {num_surface:,}")
print(f"  surface_ratio = {num_surface:,} / {num_total:,} = {surface_ratio:.1%}")
print("")
print("✅ Result: Using 90th percentile gives exactly 10% surface particles!")

# ============================================================================
# Step 4: The formula
# ============================================================================
print("\n[STEP 4] The Formula")
print("-"*70)
print("FORMULA: percentile = 1 - desired_surface_ratio")
print("")
print("Examples:")
examples = [
    (0.10, 0.90),  # 10% surface
    (0.15, 0.85),  # 15% surface
    (0.20, 0.80),  # 20% surface
    (0.05, 0.95),  # 5% surface
]

for desired_ratio, percentile in examples:
    thresh = torch.quantile(grad_sampled, percentile).item()
    actual_mask = grad_sampled > thresh
    actual_count = actual_mask.sum().item()
    actual_ratio = actual_count / num_total

    print(f"  Want {desired_ratio*100:4.0f}% surface → Use {percentile*100:4.0f}th percentile")
    print(f"    threshold = {thresh:.2e}")
    print(f"    result = {actual_count:,} particles ({actual_ratio*100:.1f}%)")
    print("")

# ============================================================================
# Step 5: Visual explanation
# ============================================================================
print("\n[STEP 5] Visual Explanation")
print("-"*70)

# Create histogram
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Top plot: Full distribution
ax1.hist(grad_sampled.numpy(), bins=100, alpha=0.7, edgecolor='black')
ax1.axvline(value_90, color='red', linestyle='--', linewidth=2, label=f'90th percentile = {value_90:.2e}')
ax1.fill_between([value_90, grad_sampled.max().item()], 0, ax1.get_ylim()[1],
                  alpha=0.3, color='green', label='Surface particles (10%)')
ax1.set_xlabel('Edge Gradient Strength')
ax1.set_ylabel('Number of Particles')
ax1.set_title('Gradient Distribution with 90th Percentile Threshold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom plot: Cumulative distribution
sorted_vals = torch.sort(grad_sampled)[0].numpy()
cumulative_pct = np.arange(num_total) / num_total * 100

ax2.plot(sorted_vals, cumulative_pct, linewidth=2)
ax2.axhline(90, color='red', linestyle='--', linewidth=2, label='90th percentile')
ax2.axvline(value_90, color='red', linestyle='--', linewidth=2)
ax2.fill_betweenx([90, 100], sorted_vals.min(), sorted_vals.max(),
                   alpha=0.3, color='green', label='Top 10% (surface)')
ax2.set_xlabel('Edge Gradient Strength')
ax2.set_ylabel('Cumulative Percentage (%)')
ax2.set_title('Cumulative Distribution (Shows Percentiles)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_path = 'percentile_explanation.png'
plt.savefig(output_path, dpi=150)
print(f"✅ Saved visualization to: {output_path}")

# ============================================================================
# Step 6: Code implementation
# ============================================================================
print("\n[STEP 6] Code Implementation")
print("-"*70)
print("In utils/rendering_utils.py:")
print("")
print("```python")
print("# Configure desired surface ratio")
print("percentile = render_cfg.get('edge_proximity_percentile', 0.9)  # Default: 90th")
print("")
print("# Compute threshold as Pth percentile of gradient distribution")
print("grad_percentile = torch.quantile(grad_sampled, percentile).item()")
print("")
print("# Mark particles above threshold as surface")
print("surface_mask = grad_sampled > grad_percentile")
print("")
print("# Count result")
print("num_surface = surface_mask.sum().item()")
print("surface_ratio = num_surface / num_total")
print("```")

# ============================================================================
# Step 7: Interactive calculator
# ============================================================================
print("\n[STEP 7] Interactive Calculator")
print("-"*70)
print("Try different percentiles:")
print("")

test_percentiles = [0.80, 0.85, 0.875, 0.90, 0.95]
print(f"{'Percentile':>12} | {'Threshold':>12} | {'Surface Count':>14} | {'Surface %':>10}")
print("-" * 60)

for p in test_percentiles:
    thresh = torch.quantile(grad_sampled, p).item()
    mask = grad_sampled > thresh
    count = mask.sum().item()
    ratio = count / num_total * 100

    print(f"{p*100:>11.1f}th | {thresh:>12.2e} | {count:>14,} | {ratio:>9.1f}%")

print("")
print("="*70)
print("✅ SUMMARY")
print("="*70)
print(f"To get ~10% surface particles:")
print(f"  1. Set edge_proximity_percentile: 0.90 in your config")
print(f"  2. Code computes: threshold = quantile(grad_sampled, 0.90)")
print(f"  3. Result: surface_mask = (grad_sampled > threshold)")
print(f"  4. This gives top 10% of particles by edge gradient strength")
print("="*70)
