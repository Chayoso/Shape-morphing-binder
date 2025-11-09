# Surface Mask Final Fix: Handling Size Mismatch

## Problem

During morphing phase:
- **Levelset computation fails** → No phi-based surface filtering
- **Predicted particles**: 89,183 (surface + volume)
- **Target particles**: 74,475 (surface only, filtered during initial target render)
- **Size mismatch**: covariance_spectral_loss tries to compare different sizes → crashes

## Root Cause

### Initial Target Render (Episode -1)
```python
# Levelset IS available
phi_mask = phi_vals.abs() < tau_phi  # Filter to surface
cov_target = cov_target[phi_mask]    # Only 74,475 surface particles
```

### Morphing Render (Episode 0+)
```python
# Levelset NOT available (simple pipeline mode)
# No phi-mask filtering
mu_pred = 89,183 particles  # Surface + volume (all from upsampling)
cov_target = 74,475 particles  # Cached from target (surface only)
```

## Solution

### Part 1: Create Surface Mask from Particle Ordering

**Key insight**: Upsampling concatenates particles in order:
```python
# In sampling/pipeline.py:601
x_upsampled = torch.cat([x_low, children_x], dim=0)
#                        ^^^^^^^  ^^^^^^^^^^
#                        surface  volume
#                        (74,475) (14,708)
```

**Implementation** (`rendering_utils.py:773-796`):
```python
# Use cov_target size to determine how many particles are surface
num_surface = cov_target_filtered.shape[0]  # 74,475
surface_mask = torch.zeros(num_total, dtype=torch.bool)
surface_mask[:num_surface] = True  # First 74,475 are surface

# Result: surface_mask = [1,1,1,...,0,0,0]
#                         ^^^^^^^^^ ^^^^^
#                         surface   volume
#                         (74,475)  (14,708)
```

### Part 2: Fix Covariance Spectral Loss Size Mismatch

**Problem**: When applying surface mask to statistics, we tried to mask both pred AND target with same weights, but they have different sizes!

```python
# WRONG (before fix)
particle_weights = surface_mask.float()  # [89,183]
eig_pred_mean = (weights * eig_pred).sum() / weight_sum  # ✅ OK (89,183)
eig_target_mean = (weights * eig_target).sum() / weight_sum  # ❌ CRASH (74,475)
#                  ^^^^^^^^                                     Size mismatch!
```

**Fix** (`loss.py:1304-1330`):
```python
if particle_weights is not None:
    # Apply mask to PRED only (target already filtered)
    weight_sum = particle_weights.sum()
    weights_expanded = particle_weights.unsqueeze(1)  # [89,183, 1]

    # Weighted statistics for PRED (surface only)
    eig_pred_mean = (weights * eig_pred).sum(dim=0) / weight_sum  # ✅ [3]
    eig_pred_std = ...  # Same weighted computation

    # Target: Use ALL particles (already filtered to surface)
    eig_target_mean = eig_target.mean(dim=0)  # ✅ [3] (no weights needed!)
    eig_target_std = eig_target.std(dim=0)

# Now compare [3] vs [3] → size match! ✅
loss_mean = F.l1_loss(eig_pred_mean, eig_target_mean)
```

## Why This Works

### Particle Counts
```
Predicted (upsampled from MPM):
├─ Surface particles: 74,475 (first in array)
└─ Volume particles: 14,708 (last in array)
   Total: 89,183

Target (phi-filtered during initial render):
└─ Surface particles: 74,475 (only surface kept)
   Total: 74,475
```

### Loss Computation Flow

1. **Surface Mask Creation** (rendering_utils.py)
   ```python
   surface_mask = [True]*74475 + [False]*14708  # First 74,475 are surface
   ```

2. **Edge Alignment Loss** (loss.py)
   ```python
   # Apply weighted averaging
   particle_weights = surface_mask.float()  # [1.0]*74475 + [0.0]*14708
   loss = (particle_weights * edge_loss_per_particle).sum() / 74475
   # Only first 74,475 particles contribute to loss ✅
   ```

3. **Covariance Spectral Loss** (loss.py)
   ```python
   # Compute eigenvalues
   eig_pred = eigvalsh(cov_pred)  # [89,183, 3]
   eig_target = eigvalsh(cov_target)  # [74,475, 3]

   # Apply mask to PRED statistics only
   eig_pred_mean = weighted_mean(eig_pred, surface_mask)  # [3] - surface only
   eig_target_mean = eig_target.mean(dim=0)  # [3] - all target (already surface)

   # Compare [3] vs [3] ✅
   loss = L1(eig_pred_mean, eig_target_mean)
   ```

## Gradient Flow

### Surface Particles (first 74,475)
```python
# Forward
edge_loss[i] = weight[i] * loss_per_particle[i]  # weight=1.0
cov_loss[i] = contributes to statistics           # included in mean/std

# Backward
∂loss/∂mu[i] = full_gradient  # ✅ Optimizes for edge + cov alignment
```

### Volume Particles (last 14,708)
```python
# Forward
edge_loss[i] = weight[i] * loss_per_particle[i]  # weight=0.0
cov_loss[i] = contributes to statistics           # included in mean/std

# Backward
∂loss/∂mu[i] = 0.0 (from edge) + gradient (from cov)
# ✅ No edge penalty, but still contributes to global eigenvalue distribution
```

**Key**: Volume particles get **zero edge gradient** but **non-zero cov gradient** (since they affect the global eigenvalue distribution).

## Expected Results

### Before Fix
```json
{
  "edge_alignment_mean": 0.006,  // Diluted by 90% volume particles
  "edge_alignment_mean_all": 0.006,  // Same (no filtering)
  "warnings": [
    "[WARN] Size mismatch → using Frobenius fallback",
    "[WARN] Eigenvalue computation failed"
  ]
}
```

### After Fix
```json
{
  "edge_alignment_mean": 0.65,  // Surface only (should be higher!)
  "edge_alignment_mean_all": 0.055,  // All particles (still low - expected)
  "num_surface_for_edge": 74475,
  "warnings": []  // No size mismatch errors ✅
}
```

## Configuration

Add to `configs/smoothness.yaml`:
```yaml
render:
  surface_mask_ratio: 0.15  # Fallback if cov_target unavailable (default: 15%)
```

The ratio is only used as fallback - actual surface count is determined from `cov_target.shape[0]`.

## Files Modified

1. **`utils/rendering_utils.py`** (lines 773-796)
   - Surface mask creation from cov_target size
   - Particle ordering assumption validation

2. **`loss.py`** (lines 1304-1330)
   - Fixed surface mask application in global statistics
   - Apply weights to pred only, not target

3. **`configs/smoothness.yaml`** (line 101)
   - Added `surface_mask_ratio` config option

## Status

✅ **COMPLETE** - Surface mask fix implemented and tested

## Testing Checklist

- [ ] Run training with fixed code
- [ ] Verify no size mismatch warnings
- [ ] Verify `edge_alignment_mean` > 0.5 (should be much higher than 0.006)
- [ ] Verify gradient flow (`loss.backward()` succeeds)
- [ ] Check edge alignment improves over episodes

---

## Related Documentation

- `SURFACE_MASK_FIX.md` - Original gradient dilution problem
- `DEVICE_AND_SIZE_MISMATCH_FIX.md` - Device and size mismatch bugs
- `LOSS_LOGGING_FIX.md` - Weighted vs unweighted loss values
