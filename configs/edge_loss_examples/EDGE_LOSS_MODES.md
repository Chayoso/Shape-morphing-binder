# Edge Loss Configuration Guide

This document explains the new edge alignment loss options and how to configure them.

## Problem Background

The edge alignment loss was generating two gradient signals:
1. **"Real" signal (dL/dF)**: Gradients to deformation field F (what we want)
2. **"Spurious" signal (dL/dx)**: Gradients to particle positions (23x stronger, drowns out dL/dF)

## Solution: Three Configuration Modes

### Mode 1: `detach_position` (RECOMMENDED - Default)

**What it does**: Blocks the spurious dL/dx gradient path by detaching positions when computing the Jacobian.

**When to use**: First choice for most scenarios. Mathematically clean and preserves gradient correctness.

**Configuration**:
```yaml
optimization:
  loss:
    enabled: true
    w_edge: 0.1
    edge_loss_mode: 'detach_position'  # Block spurious dL/dx path
    debug_edge_gradients: false        # Set to true for diagnostics
```

**Expected behavior**:
- Edge alignment should improve if edges are strong enough
- No gradient instability
- Check logs for `edge_alignment_mean` - should increase toward 1.0

---

### Mode 2: `gradient_boost` (Fallback Option)

**What it does**: Artificially amplifies the dL/dF gradient by a boost factor to overpower the spurious dL/dx signal.

**When to use**:
- If `detach_position` mode doesn't improve alignment
- When you need stronger dL/dF signals
- Trade-off: May cause optimizer instability if boost factor is too high

**Configuration**:
```yaml
optimization:
  loss:
    enabled: true
    w_edge: 0.1
    edge_loss_mode: 'gradient_boost'
    gradient_boost_factor: 1000.0      # Try: 100, 1000, 10000
    debug_edge_gradients: true         # Monitor for instability
```

**Tuning `gradient_boost_factor`**:
- Start with 1000.0
- If alignment doesn't improve: increase to 5000 or 10000
- If you see NaN/Inf or exploding gradients: decrease to 100 or 500
- Monitor `||∂L_render/∂F||` in logs - should increase relative to `||∂L_render/∂x||`

---

### Mode 3: `original` (Legacy - Not Recommended)

**What it does**: Uses the original implementation with both gradient paths active.

**When to use**:
- For comparison/debugging only
- To reproduce old behavior

**Configuration**:
```yaml
optimization:
  loss:
    enabled: true
    w_edge: 0.1
    edge_loss_mode: 'original'
```

**Expected behavior**:
- `||∂L_render/∂x||` will be ~23x larger than `||∂L_render/∂F||`
- Edge alignment likely won't improve

---

## Diagnostics & Debugging

### Enable Debug Prints

```yaml
optimization:
  loss:
    debug_edge_gradients: true
```

**Output example**:
```
[DEBUG EDGE] Using 'detach_position' mode - blocking dL/dx through Jacobian
[DEBUG EDGE] Alpha edge statistics:
  grad_norm mean: 1.105487e-03
  grad_norm max: 2.451234e-02
  grad_norm min: 0.000000e+00
  Percentage of pixels with strong edges (>0.1): 0.00%
[DEBUG EDGE] Alignment statistics:
  alignment mean: 0.006420
  alignment max: 0.845123
  alignment min: 0.000012
```

### Interpreting Edge Statistics

**`grad_norm mean`**: Average edge gradient strength in alpha channel
- `< 0.001`: ❌ Extremely weak - edge loss won't work
- `0.001 - 0.01`: ⚠️ Very weak - try gradient boost or disable edge loss
- `0.01 - 0.1`: ⚠️ Weak but usable
- `> 0.1`: ✅ Strong edges - edge loss should work well

**`edge_alignment_mean`**: How well covariances align with edges
- `< 0.01`: No alignment (random)
- `0.1 - 0.5`: Partial alignment
- `> 0.8`: Good alignment
- Goal: Should increase over training epochs

### Edge Visualization Tool

Use the diagnostic utility to visualize alpha channel edges:

```python
from utils.edge_diagnostics import diagnose_edge_alignment_loss, visualize_alpha_edges

# Visualize edges
visualize_alpha_edges(
    alpha_target,
    output_path="debug/alpha_edges.png",
    title="Alpha Edge Analysis"
)

# Full diagnostic
diagnose_edge_alignment_loss(
    alpha_target, mu, cov, view_T, W, H, tanfovx, tanfovy,
    output_dir="debug/"
)
```

---

## Recommended Workflow

### Step 1: Check Edge Strength
```yaml
optimization:
  loss:
    w_edge: 0.1
    debug_edge_gradients: true  # Enable diagnostics
```

Run 1 episode and check `grad_norm mean` in logs.

### Step 2: Choose Mode Based on Edge Strength

**If `grad_norm mean > 0.01`** (edges are decent):
```yaml
edge_loss_mode: 'detach_position'
gradient_boost_factor: 1000.0  # Won't be used
debug_edge_gradients: false
```

**If `grad_norm mean < 0.01`** (weak edges):

Option A - Try gradient boost:
```yaml
edge_loss_mode: 'gradient_boost'
gradient_boost_factor: 5000.0  # Aggressive boost
debug_edge_gradients: true
```

Option B - Disable edge loss:
```yaml
w_edge: 0.0  # Rely on other losses instead
```

### Step 3: Monitor Training

Watch these metrics in episode summaries:
- `edge_alignment_mean`: Should increase toward 1.0
- `loss_edge`: Should decrease
- Check for NaN/Inf (indicates instability)

### Step 4: Tune if Needed

**If alignment isn't improving**:
- Try increasing `gradient_boost_factor` (2x - 10x)
- Increase `w_edge` weight
- Check edge visualization - edges may be too weak

**If you see gradient explosion (NaN/Inf)**:
- Decrease `gradient_boost_factor` (divide by 2-5)
- Decrease `w_edge` weight
- Switch to `detach_position` mode

---

## Example Configurations

See example config files:
- `1_detach_position.yaml` - Recommended default
- `2_gradient_boost_weak_edges.yaml` - For weak edge scenarios
- `3_gradient_boost_aggressive.yaml` - Maximum boost
- `4_edge_disabled.yaml` - When edges are too weak

---

## Summary Table

| Mode | Pros | Cons | When to Use |
|------|------|------|-------------|
| `detach_position` | Clean, stable, mathematically correct | May be too weak if edges are very faint | Default choice |
| `gradient_boost` | Can amplify weak signals | May cause instability, not mathematically correct | Weak edges, detach_position failed |
| `original` | N/A | Spurious gradients dominate | Debugging only |

---

## Troubleshooting

**Q: Edge alignment stuck at ~0.006?**
- Check `grad_norm mean` - if < 0.001, edges are too weak
- Try `gradient_boost` mode with factor 5000-10000
- Or disable edge loss (`w_edge: 0.0`)

**Q: Getting NaN/Inf in gradients?**
- `gradient_boost_factor` is too high - reduce to 100-500
- Check `w_edge` weight - may be too high
- Switch to `detach_position` mode

**Q: How to know if edge loss is working?**
- `edge_alignment_mean` should increase over episodes
- Visualize edges - should see clear silhouette boundaries
- Compare renders with/without edge loss

**Q: Should I always use edge loss?**
- No! Only if target has sharp silhouette edges
- If `grad_norm mean < 0.001`, disable it
- Other losses (cov_align, alpha, depth) may be sufficient
