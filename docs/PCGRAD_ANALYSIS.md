# PCGrad Implementation Analysis

## Current Status: ✅ Mathematically Correct BUT ⚠️ Potential Gradient Scale Mismatch

---

## Implementation Review

### ✅ What's Working

1. **PCGrad Algorithm (gradient_utils.py:118-189)**
   - ✅ Blockwise whitening (normalizes F and x separately)
   - ✅ Quantile clipping (suppresses outliers)
   - ✅ Conflict detection (cosine < -0.1)
   - ✅ Projection formula: `proj = min(0, dot) / ||g_p||² * g_p`
   - ✅ Scale restoration after projection

2. **Enabled by Default (training_loop.py:803)**
   ```python
   use_pcgrad = rs_full.get('optimization', {}).get('use_pcgrad', True)  ✅
   ```

3. **Test Results**
   - Conflicting gradients (cos=-1.0): Correctly projects to ~0
   - Aligned gradients (cos=1.0): No projection (as expected)
   - Partial conflict (cos=0.3): No projection (threshold is -0.1)

---

## ⚠️ Potential Issues Found

### Issue 1: Gradient Scale Mismatch Before PCGrad

**Location:** `training_loop.py:768-814`

**Problem:** Cosine similarity and PCGrad are computed with inconsistent gradient scales.

**Current Flow:**
```
Session Mode (training_loop.py:88-318):
  1. Compute render loss and backward()
  2. Extract dLdF_render, dLdx_render (raw gradients)
  3. Normalize to unit vectors:
     dLdF_normalized = dLdF_render / ||dLdF_render||  # Line 273
     dLdx_normalized = dLdx_render / ||dLdx_render||  # Line 274
  4. Return (dLdF_normalized, dLdx_normalized)

Legacy Mode (training_loop.py:748-875):
  1. Get physics gradients from C++ backend:
     dLdF_phys, dLdx_phys = cg.get_last_layer_phys_gradients()  # Line 751
     → These are RAW gradients (not normalized!)

  2. Compute cosine similarity:
     # Line 767-769 (approximate location based on context)
     cosine = compute_gradient_cosine_similarity(
         dLdF_phys, dLdx_phys,      # ← RAW physics gradients
         dLdF_render, dLdx_render    # ← UNIT NORM render gradients (from session mode)
     )

  3. Apply PCGrad:
     pcgrad_projection(
         dLdF_render=dLdF_render,    # ← UNIT NORM (from session mode)
         dLdx_render=dLdx_render,    # ← UNIT NORM
         dLdF_physics=dLdF_phys,     # ← RAW (not normalized)
         dLdx_physics=dLdx_phys      # ← RAW
     )
```

**Why This Matters:**

While PCGrad internally normalizes (blockwise whitening), the **cosine similarity** is computed BEFORE PCGrad on mismatched scales:

```python
# compute_gradient_cosine_similarity (gradient_utils.py:41-69)
g_phys = np.concatenate([dLdF_phys.flatten(), dLdx_phys.flatten()])    # RAW
g_render = np.concatenate([dLdF_render.flatten(), dLdx_render.flatten()])  # UNIT NORM

cosine = dot(g_phys, g_render) / (||g_phys|| × ||g_render||)
```

If physics gradients are very small (e.g., 0.01) and render is unit norm (1.0), the cosine will be biased.

**Example:**
```
Physics: g_phys = [0.01, 0, 0]  (||g|| = 0.01)
Render:  g_render = [1, 0, 0]   (||g|| = 1.0, unit norm)

cosine = 0.01 / (0.01 × 1.0) = 1.0  ← Looks aligned!

But actually, they should both be normalized:
g_phys_norm = [1, 0, 0]
g_render_norm = [1, 0, 0]
cosine = 1.0 / (1.0 × 1.0) = 1.0  ✓ Same result

Now with conflict:
Physics: g_phys = [0.01, 0, 0]
Render:  g_render = [-1, 0, 0]

cosine = -0.01 / (0.01 × 1.0) = -1.0  ✓ Correctly detects conflict
```

**Conclusion:** The cosine computation is actually **okay** because it normalizes by magnitudes anyway. The division by `||g_phys|| × ||g_render||` accounts for the scale difference.

---

### Issue 2: Session Mode vs Legacy Mode Inconsistency

**Session Mode (run_e2e_episode_session):**
- Normalizes render gradients to unit norm in callback (line 273-274)
- Returns unit norm gradients

**Legacy Mode (run_e2e_episode):**
- Does NOT normalize render gradients before returning
- Returns raw gradients from backward()

**Impact:** PCGrad behaves differently in session mode vs legacy mode!

**Location:** `training_loop.py:732-734` (legacy mode render grad extraction)

```python
# Legacy mode (training_loop.py:732-734)
dLdF_render = F.grad.detach().cpu().numpy()  # RAW gradients
dLdx_render = x.grad.detach().cpu().numpy()  # RAW gradients
# No normalization here!
```

**Session mode (training_loop.py:273-274):**
```python
# Session mode callback
dLdF_normalized = dLdF_render / (grad_F_norm_raw + eps)  # Unit norm
dLdx_normalized = dLdx_render / (grad_x_norm_raw + eps)  # Unit norm
```

**Fix Needed:** Legacy mode should also normalize to unit norm for consistency.

---

## ✅ Verified Correct Behavior

### 1. PCGrad Projection Formula
```python
# gradient_utils.py:177-178
proj = (min(0.0, dot) / (np.dot(g_p, g_p) + 1e-12)) * g_p
g_r_proj = g_r - proj
```

This is the **correct PCGrad formula** from the paper:
- Only projects if `dot < 0` (conflict)
- Projects onto `g_p` with magnitude `dot / ||g_p||²`
- Removes negative component only

### 2. Blockwise Whitening
```python
# gradient_utils.py:149-150
rF_w, rx_w, stats_r = _blockwise_whiten(dLdF_render, dLdx_render)
pF_w, px_w, stats_p = _blockwise_whiten(dLdF_physics, dLdx_physics)
```

This is **essential** because F and x have different physical units:
- F (deformation gradient): dimensionless, ~O(1)
- x (position): meters, ~O(10)

Without blockwise whitening, x would dominate the projection.

### 3. Quantile Clipping
```python
# gradient_utils.py:157
g_r = _clip_by_quantile(g_r, q=0.999)
```

Clips outliers at 99.9th percentile. This is **good practice** for stability.

### 4. Scale Restoration
```python
# gradient_utils.py:186-187
gF_proj = g_r_proj[:nF].reshape(dLdF_render.shape) * stats_r['nF']
gx_proj = g_r_proj[nF:].reshape(dLdx_render.shape) * stats_r['nx']
```

Restores original scales after projection. ✅ Correct.

---

## 🔧 Recommended Fixes

### Fix 1: Normalize Render Gradients in Legacy Mode (Low Priority)

**File:** `utils/training_loop.py:732-747`

**Current (legacy mode):**
```python
dLdF_render = F.grad.detach().cpu().numpy()
dLdx_render = x.grad.detach().cpu().numpy()
# ← No normalization!
```

**Recommended:**
```python
dLdF_render = F.grad.detach().cpu().numpy()
dLdx_render = x.grad.detach().cpu().numpy()

# Normalize to unit norm for consistency with session mode
grad_F_norm = np.linalg.norm(dLdF_render)
grad_x_norm = np.linalg.norm(dLdx_render)
eps = 1e-12

dLdF_render = dLdF_render / (grad_F_norm + eps)
dLdx_render = dLdx_render / (grad_x_norm + eps)
```

**Impact:** Ensures consistent behavior between session and legacy modes.

---

## 🧪 Testing PCGrad

### Test Case 1: Verify PCGrad is Called
```bash
python run.py -c configs/sp_to_by/exp_test_fixes.yaml --png 2>&1 | grep "PCGrad"
```

**Expected output:**
```
🔥 [PCGrad] Conflict detected (cos=-0.234), applying gradient projection
├─ pcgrad_cosine: -0.234
├─ pcgrad_applied: True
├─ pcgrad_projection_scale: 0.123
```

### Test Case 2: Check Conflict Detection
```bash
python run.py -c configs/sp_to_by/exp_test_fixes.yaml --png 2>&1 | grep "Conflict:"
```

**Expected output:**
```
└─ Conflict: cos(θ) = -0.234 ⚠️ CONFLICT
```

**If no conflicts detected:**
- Check if gradients are actually conflicting (cosine should be < -0.1)
- If cosine is always > -0.1, gradients are not conflicting (this is okay!)

### Test Case 3: Verify PCGrad Improves Convergence

**Without PCGrad:**
```yaml
# configs/sp_to_by/exp_test_no_pcgrad.yaml
optimization:
  use_pcgrad: false  # Disable
```

**With PCGrad:**
```yaml
# configs/sp_to_by/exp_test_fixes.yaml
optimization:
  # use_pcgrad: true (default, no need to specify)
```

**Compare:**
- Episode success rate
- Loss curves smoothness
- Number of line search failures

---

## 📊 Expected Behavior

### When Conflicts Occur (cosine < -0.1):
```
[Gradient Combination]
├─ BEFORE PCGrad:
│  ├─ Conflict: cos(θ) = -0.45 ⚠️
│  └─ Physics and render pulling in opposite directions

🔥 [PCGrad] Applying projection...

├─ AFTER PCGrad:
│  ├─ Conflict resolved: cos(θ) = 0.02 ✓
│  └─ Projection removed 43% of render gradient
```

### When No Conflicts (cosine ≥ -0.1):
```
[Gradient Combination]
├─ Conflict: cos(θ) = 0.32 ✓ aligned
└─ No PCGrad needed (gradients cooperate)
```

---

## 🎯 Summary

### ✅ PCGrad Implementation is Correct
- Algorithm matches the paper
- Blockwise whitening prevents unit mismatch
- Quantile clipping suppresses outliers
- Enabled by default ✓

### ⚠️ Minor Inconsistency
- Session mode normalizes render gradients
- Legacy mode does NOT normalize
- **Recommendation:** Add normalization to legacy mode for consistency

### 🔬 Testing Needed
- Run `exp_test_fixes.yaml` and check logs for PCGrad messages
- Verify conflicts are detected when gradients oppose
- Compare success rate with vs without PCGrad

---

## 📝 Verification Checklist

- [ ] Run test and grep for "PCGrad" in logs
- [ ] Verify conflict detection (cosine < -0.1)
- [ ] Check projection scale (should be 0-100%)
- [ ] Compare episode success rate with/without PCGrad
- [ ] Verify no NaN/Inf in projected gradients

---

**Status:** ✅ PCGrad is mathematically correct and enabled by default. Minor inconsistency between modes can be fixed if needed.
