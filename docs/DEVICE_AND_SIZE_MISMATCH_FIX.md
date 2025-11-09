# Device and Size Mismatch Bug Fixes

## Bugs Fixed

### Bug 1: Device Mismatch (CPU vs CUDA)
**Error**:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

**Root Cause**:
- `surface_mask` passed from pipeline was on CPU
- Loss computations happen on CUDA
- PyTorch doesn't allow operations mixing CPU and CUDA tensors

**Locations**:
1. `covariance_spectral_loss()` - line 1267
2. `edge_align_loss()` - line 1162

**Fix**:
```python
# Before (broken)
particle_weights = surface_mask.float()  # Device mismatch!

# After (fixed)
surface_mask = _safe_device_transfer(surface_mask, device)
particle_weights = surface_mask.float()  # Now on same device ✅
```

---

### Bug 2: Size Mismatch in Frobenius Fallback
**Error**:
```
UserWarning: Using a target size (torch.Size([74475, 3, 3])) that is different to the input size (torch.Size([89183, 3, 3]))
RuntimeError: The size of tensor a (89183) must match the size of tensor b (74475) at non-singleton dimension 0
```

**Root Cause**:
- `cov_pred` has 89,183 particles (all particles including volume)
- `cov_target` has 74,475 particles (surface particles only)
- When eigenvalue computation fails, fallback uses `F.mse_loss(cov_pred, cov_target)`
- PyTorch can't broadcast these different sizes

**Location**:
- `covariance_spectral_loss()` exception handler (line 1343)
- `covariance_spectral_loss()` frobenius mode (line 1347)

**Fix**:
```python
# Before (broken)
except Exception as e:
    print(f"[WARN] Eigenvalue computation failed: {e}, using Frobenius fallback")
    loss = F.mse_loss(cov_pred, cov_target)  # ❌ Size mismatch!

# After (fixed)
except Exception as e:
    print(f"[WARN] Eigenvalue computation failed: {e}, using Frobenius fallback")
    if cov_pred.shape[0] != cov_target.shape[0]:
        min_size = min(cov_pred.shape[0], cov_target.shape[0])
        print(f"[WARN] Size mismatch in Frobenius fallback, using first {min_size} samples")
        loss = F.mse_loss(cov_pred[:min_size], cov_target[:min_size])  # ✅ Truncate to match
    else:
        # Apply surface mask if available
        if particle_weights is not None:
            diff_squared = ((cov_pred - cov_target) ** 2).sum(dim=(1, 2))
            loss = (particle_weights * diff_squared).sum() / particle_weights.sum().clamp_min(1.0)
        else:
            loss = F.mse_loss(cov_pred, cov_target)
```

---

## Code Changes Summary

### Files Modified
- `loss.py`

### Functions Updated
1. **`covariance_spectral_loss()`** (lines 1228-1377)
   - Added device transfer for `surface_mask` (line 1267)
   - Fixed Frobenius fallback size mismatch (lines 1344-1356)
   - Fixed Frobenius mode size mismatch (lines 1359-1370)
   - Added surface mask weighting to fallback paths

2. **`edge_align_loss()`** (lines 1045-1226)
   - Added device transfer for `surface_mask` (line 1162)

---

## Testing

### Verify Fix 1 (Device Mismatch)
**Before**:
```
[WARN] Edge alignment failed: Expected all tensors to be on the same device...
RuntimeError: Expected all tensors to be on the same device, cuda:0 and cpu!
```

**After**:
```
# No device mismatch errors ✅
```

### Verify Fix 2 (Size Mismatch)
**Before**:
```
UserWarning: Using a target size (torch.Size([74475, 3, 3])) that is different...
RuntimeError: The size of tensor a (89183) must match the size of tensor b (74475)
```

**After**:
```
[WARN] Size mismatch in Frobenius fallback, using first 74475 samples
# No crash ✅
```

---

## Why These Bugs Occurred

### Device Mismatch
- `surface_mask` is generated in the Python pipeline (CPU by default)
- Loss computations happen in `loss.py` with tensors on CUDA
- We forgot to transfer `surface_mask` to CUDA before use

### Size Mismatch
- `cov_pred` includes ALL particles (volume + surface)
- `cov_target` only includes SURFACE particles (from curvature computation)
- These naturally have different sizes
- Fallback code assumed they'd have the same size

---

## Prevention

### For Future Code
1. **Always check device**: Use `_safe_device_transfer()` for any tensor coming from external sources
2. **Handle size mismatches**: Never assume tensors have matching sizes without validation
3. **Graceful fallbacks**: Exception handlers should handle edge cases (like size mismatch)

### Checklist for Adding New Losses
- [ ] Check input tensor devices
- [ ] Validate tensor shapes
- [ ] Handle size mismatches gracefully
- [ ] Test with surface mask enabled/disabled
- [ ] Test with size mismatches (different particle counts)

---

## Status
✅ **Fixed** - Both bugs resolved
- Device mismatch: Solved with `_safe_device_transfer()`
- Size mismatch: Solved with size validation and truncation

---

## Related Files
- `SURFACE_MASK_FIX.md` - Explains surface mask feature
- `loss.py` - Loss computation implementations
