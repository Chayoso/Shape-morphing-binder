# Loss Logging Fix: Weighted vs Unweighted Values

## Problem

The loss values logged in episode summaries were **unweighted** (raw loss values before applying config weights), while the actual training used **weighted** values. This caused confusion when tuning loss weights in the config.

### Example of the Problem

**Config**:
```yaml
w_cov_spd: 0.001
```

**Old Logging** (Broken):
```json
{
  "loss_cov_spd": 4.527  // Unweighted - doesn't change when you change w_cov_spd!
}
```

**Actual contribution to training**: `0.001 * 4.527 = 0.00453` (weighted)

---

## Solution

All loss values in `losses` dictionary are now **weighted** (matching what actually contributes to training). Unweighted values are also logged with `_unweighted` suffix for debugging.

### After Fix

**Config**:
```yaml
w_cov_spd: 0.001
```

**New Logging** (Fixed):
```json
{
  "loss_cov_spd": 0.00453,  // Weighted - changes when you tune w_cov_spd ✅
  "loss_cov_spd_unweighted": 4.527  // Raw value for debugging
}
```

Now you can directly see how much each loss contributes to the total!

---

## Changes Made

### All Loss Functions Updated

The following functions now log **weighted** values:

1. **`_compute_alpha_loss`**
   - `loss_alpha` = weighted
   - `loss_alpha_unweighted` = raw

2. **`_compute_depth_loss`**
   - `loss_depth` = weighted
   - `loss_depth_unweighted` = raw

3. **`_compute_photo_loss`**
   - `loss_photo` = weighted
   - `loss_photo_unweighted` = raw

4. **`_compute_edge_loss`**
   - `loss_edge` = weighted
   - `loss_edge_unweighted` = raw

5. **`_compute_cov_align_loss`**
   - `loss_cov_align` = weighted
   - `loss_cov_align_unweighted` = raw

6. **`_compute_cov_reg_loss`**
   - `loss_cov_reg` = weighted
   - `loss_cov_reg_unweighted` = raw

7. **`_compute_coverage_loss`**
   - `loss_coverage` = weighted
   - `loss_coverage_unweighted` = raw

8. **`_compute_cov_spd_regularization`**
   - `loss_cov_spd` = weighted
   - `loss_cov_spd_unweighted` = raw

9. **`_compute_det_barrier_loss`**
   - `loss_det_barrier` = weighted
   - `loss_det_barrier_unweighted` = raw

---

## Benefits

### 1. Easier Weight Tuning

**Before**: You had to mentally multiply unweighted loss by config weight to understand contribution.
```
loss_cov_spd: 4.527 (unweighted)
w_cov_spd: 0.001
Actual contribution: 4.527 * 0.001 = 0.00453 (have to calculate manually)
```

**After**: Weighted value shown directly.
```
loss_cov_spd: 0.00453 (weighted - ready to use!)
```

---

### 2. Verify Config is Applied

You can now verify your YAML config weights are being applied:

**Config**:
```yaml
w_alpha: 0.3
w_depth: 5.0
w_edge: 3.0
```

**Episode Summary**:
```json
{
  "loss_alpha_unweighted": 0.15,
  "loss_alpha": 0.045,  // = 0.3 * 0.15 ✅ Config applied correctly!

  "loss_depth_unweighted": 0.75,
  "loss_depth": 3.75,  // = 5.0 * 0.75 ✅

  "loss_edge_unweighted": 0.034,
  "loss_edge": 0.102  // = 3.0 * 0.034 ✅
}
```

---

### 3. Better Loss Balancing

You can now directly compare weighted losses to understand which ones dominate:

**Episode Summary**:
```json
{
  "loss_alpha": 0.0,      // Not contributing
  "loss_depth": 18.975,   // DOMINANT (too high!)
  "loss_edge": 0.102,     // Very small
  "loss_cov_align": 0.069 // Very small
}
```

**Action**: Depth loss is dominating → reduce `w_depth` or increase other weights.

---

### 4. Debug with Unweighted Values

If you suspect a weight isn't being applied, compare weighted vs unweighted:

```json
{
  "loss_cov_spd": 0.00453,
  "loss_cov_spd_unweighted": 4.527
}
```

**Check**: `0.00453 / 4.527 = 0.001` → Confirms `w_cov_spd: 0.001` is applied ✅

---

## Example: Tuning Loss Weights

### Scenario: Edge alignment not improving

**Step 1**: Check episode summary
```json
{
  "loss_render_total": 2.13,
  "loss_depth": 18.975,     // 89% of total! (too dominant)
  "loss_edge": 0.102,       // 5% of total (too weak)
  "loss_cov_align": 0.069   // 3% of total (too weak)
}
```

**Diagnosis**: Depth loss is drowning out edge and cov_align losses.

**Step 2**: Adjust config
```yaml
# Old config
w_depth: 5.0
w_edge: 3.0
w_cov_align: 10.0

# New config (rebalanced)
w_depth: 1.0    # Reduced 5x
w_edge: 10.0    # Increased 3.3x
w_cov_align: 20.0  # Increased 2x
```

**Step 3**: Verify in next episode
```json
{
  "loss_render_total": 1.85,
  "loss_depth": 3.795,    // 40% of total ✅
  "loss_edge": 0.340,     // 18% of total ✅
  "loss_cov_align": 0.138 // 7% of total ✅
}
```

Much more balanced!

---

## Migration Note

If you have scripts that parse episode summary JSON and expect unweighted values, update them to use:
- `loss_<name>` → weighted value (new default)
- `loss_<name>_unweighted` → raw value (for backward compatibility)

---

## Code Pattern

All loss functions now follow this pattern:

```python
def _compute_xxx_loss(self, ...):
    # Compute raw loss
    loss_unweighted = compute_xxx(...)

    # Apply weight
    loss_weighted = self.weights['w_xxx'] * loss_unweighted

    # Store both
    losses['loss_xxx'] = loss_weighted  # Main value (weighted)
    losses['loss_xxx_unweighted'] = loss_unweighted  # Debug value

    # Return weighted for gradient computation
    return loss_weighted
```

---

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Logged values | Unweighted | Weighted |
| Config verification | Hard (manual calc) | Easy (direct check) |
| Loss balancing | Confusing | Clear |
| Debug info | Missing | Available (`_unweighted`) |
| Weight tuning | Indirect | Direct |

**Bottom line**: Episode summaries now show exactly what contributes to training! 🎯
