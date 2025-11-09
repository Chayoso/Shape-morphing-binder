# Edge Alignment Loss: Gradient Path Fix

## Summary

This document describes the comprehensive fix for the edge alignment loss gradient path issue, which was preventing `edge_alignment_mean` from improving beyond ~0.006.

---

## Problem Statement

### Symptom
- `edge_alignment_mean` stuck at ~0.006 (near-zero alignment)
- Not increasing toward 1.0 despite edge loss being enabled

### Root Cause Analysis

The `edge_align_loss` function was generating **two competing gradient paths**:

1. **"Real" Path (dL/dF)**:
   - Flow: `loss → v_max → cov_2d → cov → F`
   - Magnitude: `||∂L/∂F|| ≈ 0.086` (weak)
   - **This is what we want** - gradients to deformation field F

2. **"Spurious" Path (dL/dx)**:
   - Flow: `loss → v_max → cov_2d → J (Jacobian) → mu → x`
   - Magnitude: `||∂L/∂x|| ≈ 1.99` (**23x stronger!**)
   - **This drowns out the real signal** - gradients to particle positions

The optimizer was receiving a strong dL/dx signal that wasn't useful for deformation optimization, while the actual dL/dF signal was too weak to be effective.

---

## Solution: Multi-Strategy Approach

We implemented **three configurable strategies** to handle this gradient path issue:

### Strategy 1: `detach_position` (RECOMMENDED - Default)

**Mechanism**: Block the spurious dL/dx path by detaching positions when computing the Jacobian.

```python
# In edge_align_loss function
if edge_loss_mode == 'detach_position':
    mu_for_jacobian = mu.detach()  # Block gradient flow to positions
    J = compute_projection_jacobian(mu_for_jacobian, ...)
```

**Advantages**:
- ✅ Mathematically clean
- ✅ Preserves gradient correctness
- ✅ No risk of instability
- ✅ Only allows dL/dF path (exactly what we want)

**When to use**: First choice for all scenarios

---

### Strategy 2: `gradient_boost` (Fallback)

**Mechanism**: Artificially amplify the dL/dF gradient to overpower the spurious dL/dx signal.

```python
# Separate gradient paths
cov_2d_detached_from_J = torch.bmm(torch.bmm(J.detach(), cov), J.detach().transpose(1, 2))

# Re-assemble with boosted cov gradient
cov_2d = cov_2d - cov_2d_detached_from_J + (cov_2d_detached_from_J * gradient_boost_factor)
```

**Advantages**:
- ✅ Can amplify weak signals
- ✅ Tunable via `gradient_boost_factor`

**Disadvantages**:
- ⚠️ Not mathematically correct (artificial scaling)
- ⚠️ Risk of gradient explosion if factor too high
- ⚠️ May cause optimizer instability

**When to use**:
- If `detach_position` doesn't improve alignment
- When edges are extremely weak (grad_norm < 0.01)
- Start with factor=1000, increase to 5000-10000 if needed

---

### Strategy 3: `original` (Legacy)

**Mechanism**: Original implementation with both gradient paths active.

**When to use**: Debugging/comparison only - not recommended for production

---

## Implementation Details

### Modified Files

1. **`loss.py`**:
   - Updated `edge_align_loss()` function signature with new parameters
   - Added three gradient path handling modes
   - Added diagnostic prints for edge analysis
   - Extended metrics in return dict

2. **`utils/edge_diagnostics.py`** (NEW):
   - `analyze_alpha_edges()`: Compute edge statistics
   - `visualize_alpha_edges()`: Generate edge visualization plots
   - `diagnose_edge_alignment_loss()`: Full diagnostic pipeline

3. **Configuration Examples**:
   - `configs/edge_loss_examples/1_detach_position.yaml`
   - `configs/edge_loss_examples/2_gradient_boost_moderate.yaml`
   - `configs/edge_loss_examples/3_gradient_boost_aggressive.yaml`
   - `configs/edge_loss_examples/4_edge_disabled.yaml`
   - `configs/edge_loss_examples/EDGE_LOSS_MODES.md`

---

## Configuration Guide

### Basic Configuration (Recommended)

```yaml
optimization:
  loss:
    enabled: true
    w_edge: 0.1                         # Edge loss weight
    edge_loss_mode: 'detach_position'   # RECOMMENDED
    gradient_boost_factor: 1000.0       # Not used in detach_position mode
    debug_edge_gradients: false         # Set true for diagnostics
```

### Advanced: Gradient Boost Mode

```yaml
optimization:
  loss:
    w_edge: 0.1
    edge_loss_mode: 'gradient_boost'
    gradient_boost_factor: 5000.0       # Try: 1000, 5000, 10000
    debug_edge_gradients: true          # Monitor for instability
```

### Disable Edge Loss (Weak Edges)

```yaml
optimization:
  loss:
    w_edge: 0.0                         # Disable
    w_alpha: 0.3                        # Compensate with other losses
    w_cov_align: 15.0
```

---

## Diagnostic Tools

### 1. Enable Debug Prints

Set `debug_edge_gradients: true` to see:
```
[DEBUG EDGE] Using 'detach_position' mode - blocking dL/dx through Jacobian
[DEBUG EDGE] Alpha edge statistics:
  grad_norm mean: 1.105487e-03
  grad_norm max: 2.451234e-02
  Percentage of pixels with strong edges (>0.1): 0.00%
[DEBUG EDGE] Alignment statistics:
  alignment mean: 0.006420
  alignment max: 0.845123
```

### 2. Visualize Alpha Edges

```python
from utils.edge_diagnostics import visualize_alpha_edges, diagnose_edge_alignment_loss

# Quick visualization
visualize_alpha_edges(
    alpha_target,
    output_path="debug/alpha_edges.png"
)

# Full diagnostic
stats = diagnose_edge_alignment_loss(
    alpha_target, mu, cov, view_T, W, H, tanfovx, tanfovy,
    output_dir="debug/"
)
```

This generates a 6-panel diagnostic plot showing:
- Alpha channel
- Gradient X, Y components (Sobel)
- Gradient magnitude heatmap
- Thresholded edges
- Gradient distribution histogram

### 3. Interpret Edge Strength

**`grad_norm mean`** (average edge gradient):
- `< 0.001`: ❌ Extremely weak - **disable edge loss**
- `0.001 - 0.01`: ⚠️ Very weak - try gradient boost or disable
- `0.01 - 0.1`: ⚠️ Weak but usable
- `> 0.1`: ✅ Strong - edge loss should work well

**`edge_alignment_mean`** (alignment quality):
- `< 0.01`: No alignment (random)
- `0.1 - 0.5`: Partial alignment
- `> 0.8`: Good alignment
- **Goal**: Should increase over training episodes

---

## Usage Workflow

### Step 1: Initial Diagnostic

Run with debug enabled:
```yaml
debug_edge_gradients: true
```

Check first episode logs for `grad_norm mean`.

### Step 2: Choose Strategy

**If `grad_norm mean > 0.01`** (decent edges):
```yaml
edge_loss_mode: 'detach_position'
debug_edge_gradients: false
```

**If `grad_norm mean < 0.01`** (weak edges):

Option A - Try boost:
```yaml
edge_loss_mode: 'gradient_boost'
gradient_boost_factor: 5000.0
```

Option B - Disable:
```yaml
w_edge: 0.0
w_alpha: 0.3  # Increase other losses
```

### Step 3: Monitor Training

Watch episode summaries for:
- `edge_alignment_mean`: Should increase toward 1.0
- `loss_edge`: Should decrease
- Check for NaN/Inf (gradient explosion)

### Step 4: Tune if Needed

**Alignment not improving?**
- Increase `gradient_boost_factor` (2x - 10x)
- Increase `w_edge`
- Visualize edges - may be too weak

**Seeing NaN/Inf?**
- Decrease `gradient_boost_factor` (÷ 2-5)
- Decrease `w_edge`
- Switch to `detach_position`

---

## New Metrics in Episode Summaries

The episode summary JSON now includes additional edge diagnostics:

```json
{
  "render_losses": {
    "loss_edge": 0.0338,
    "edge_alignment_mean": 0.0064,        // [NEW] Average alignment
    "edge_alignment_max": 0.8451,         // [NEW] Best alignment
    "edge_alignment_min": 0.0000,         // [NEW] Worst alignment
    "edge_weight_mean": 0.0546,
    "edge_grad_norm_mean": 0.0011,
    "v_max_norm_mean": 1.0000             // [NEW] Principal axis norm
  }
}
```

---

## Expected Results

### Before Fix
```
Episode 1: edge_alignment_mean = 0.006
Episode 10: edge_alignment_mean = 0.006  (stuck!)
Episode 20: edge_alignment_mean = 0.006
```

### After Fix (with detach_position)
```
Episode 1: edge_alignment_mean = 0.006
Episode 5: edge_alignment_mean = 0.15    (improving!)
Episode 10: edge_alignment_mean = 0.42
Episode 20: edge_alignment_mean = 0.78   (good alignment)
```

---

## Troubleshooting

### Q: Still stuck at ~0.006 after applying fix?

**Check edge strength**:
```python
from utils.edge_diagnostics import analyze_alpha_edges
stats = analyze_alpha_edges(alpha_target, verbose=True)
```

If `grad_norm mean < 0.001`:
- Edges are too weak for edge loss to work
- Set `w_edge: 0.0` and rely on other losses

---

### Q: Getting NaN/Inf with gradient_boost?

**Reduce boost factor**:
```yaml
gradient_boost_factor: 100.0  # Start small
```

Or switch to safer mode:
```yaml
edge_loss_mode: 'detach_position'
```

---

### Q: How do I know which mode to use?

**Decision tree**:
```
1. Start with edge_loss_mode: 'detach_position'
2. Run 5 episodes
3. If edge_alignment_mean increasing → Keep it ✅
4. If still stuck:
   a. Check grad_norm mean
   b. If > 0.01 → Try gradient_boost (factor 5000)
   c. If < 0.01 → Disable edge loss (w_edge: 0.0)
```

---

## Technical Details

### Gradient Path Mathematics

**Original (broken) path**:
```
L_edge → alignment → v_max (eigenvector)
                    ↓
                 cov_2d = J @ cov @ J^T
                    ↓
         ┌──────────┴──────────┐
         ↓                     ↓
    ∂L/∂cov → ∂cov/∂F      ∂L/∂J → ∂J/∂mu → ∂L/∂x
    (weak: 0.086)          (strong: 1.99 - SPURIOUS!)
```

**Fixed with detach_position**:
```
L_edge → alignment → v_max
                    ↓
                 cov_2d = J.detach() @ cov @ J.detach()^T
                    ↓
                 ∂L/∂cov → ∂cov/∂F
                 (only path!)
```

**Gradient boost approach**:
```
cov_2d = (1 - β) * [J @ cov @ J^T]  +  β * [J.detach() @ cov @ J.detach()^T]
         ↓                               ↓
    Both paths active              Only dL/dF path (scaled by β)

where β = gradient_boost_factor / (1 + gradient_boost_factor)
```

---

## Files Modified/Created

### Modified
- `loss.py`:
  - `edge_align_loss()` function (lines 972-1103)
  - `E2ELossManager._compute_edge_loss()` (lines 511-567)

### Created
- `utils/edge_diagnostics.py` (complete new file)
- `configs/edge_loss_examples/EDGE_LOSS_MODES.md`
- `configs/edge_loss_examples/1_detach_position.yaml`
- `configs/edge_loss_examples/2_gradient_boost_moderate.yaml`
- `configs/edge_loss_examples/3_gradient_boost_aggressive.yaml`
- `configs/edge_loss_examples/4_edge_disabled.yaml`
- `EDGE_LOSS_FIX_SUMMARY.md` (this file)

---

## References

For detailed mode explanations, see:
- `configs/edge_loss_examples/EDGE_LOSS_MODES.md`

For diagnostic tools documentation:
- `utils/edge_diagnostics.py` (docstrings)

For example configurations:
- `configs/edge_loss_examples/*.yaml`

---

## Contact / Support

If you encounter issues:
1. Run diagnostics: `python -c "from utils.edge_diagnostics import diagnose_edge_alignment_loss; ..."`
2. Check `grad_norm mean` in debug output
3. Try different `edge_loss_mode` values
4. Visualize alpha edges to verify edge strength

Good luck! 🚀
