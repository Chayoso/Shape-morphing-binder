# Render Loss Convergence Analysis

## Problem Statement

When render gradients are injected into physics optimization, the simulation fails to converge. The line search repeatedly rejects steps, preventing the optimizer from making progress.

## Root Cause Analysis

### 1. Gradient Injection Mechanism (`DiffMPMLib3D/CompGraph.cpp:148-195`)

```cpp
// Render gradients are added to physics gradients at the last layer:
pt.dLdF += dF_render;  // Line 189
pt.dLdx += dx_render;  // Line 190
```

**The problem**: Direct addition of gradients from two different objective functions with vastly different scales.

### 2. Gradient Magnitude Mismatch

**Physics gradients**:
- Derived from mass-matching loss: `||x(T) - x_target||²`
- Typical magnitude: ~1e-3 to 1e-2
- Well-behaved, smooth objective

**Render gradients**:
- Derived from image-space losses (alpha, depth, edge alignment, covariance)
- Typical magnitude: ~1e+1 to 1e+3 (**1000x-100000x larger!**)
- Non-smooth due to discrete pixel grid, silhouette discontinuities
- Includes high-frequency noise from edge detection

### 3. Recent "Fixes" That Made Things Worse

#### Config changes (sphere_to_bunny.yaml:55, 77):
```yaml
initial_alpha: 0.01  # ⚡ REDUCED: 100x smaller to allow render gradients to guide!
gain_max: 1000       # ⚡ REDUCED: Prevent render loss degradation!
```

#### Training loop changes (utils/training_loop.py:455):
```python
# ⚡ REDUCED: 1.5→1.0 start to balance physics/render from beginning
phys_w = float(np.interp(progress, [0.0, 0.3, 0.7, 1.0], [1.0, 0.9, 0.85, 0.8]))
```

**Why these don't work**:
- Reducing `initial_alpha` from 1.0 to 0.01 makes optimization **100x slower**
- Even with gain capped at 1000, render grads are still **orders of magnitude larger**
- Starting physics weight at 1.0 means render gradients dominate from the start

### 4. Line Search Failure Pattern

From `CompGraph.cpp:356-383`:
```cpp
for (int ls_iter = 0; ls_iter < max_line_search_iters; ++ls_iter) {
    // Try step with reduced alpha
    float new_loss = eval_loss();

    if (new_loss < gd_loss) {
        // Accept step
        break;
    }

    alpha_try *= 0.5f;  // Halve step size
}

if (!step_accepted) {
    std::cout << "Line search failed." << std::endl;
}
```

**What happens**:
1. Adam computes update direction using combined gradients (physics + render)
2. Line search tries step: `x_new = x + alpha * update_direction`
3. Physics loss **increases** (because render grads push in wrong direction for physics)
4. Line search halves alpha and tries again: 1.0 → 0.5 → 0.25 → ... → 0.0009765625
5. After 10 iterations, gives up: "Line search failed"

### 5. Gradient Conflict Detection

From `utils/training_loop.py:633-643`:
```python
cosine = compute_gradient_cosine_similarity(
    dLdF_phys, dLdx_phys, dLdF_render, dLdx_render
)

conflict_status = (
    '⚠️ CONFLICT' if cosine < -0.3 else
    '✓ aligned' if cosine > 0.3 else
    '~ neutral'
)
```

**Likely seeing**: `cosine < -0.3` (gradients pointing in opposite directions)

## Why Current Solutions Don't Work

### 1. Gradient Scaling (`utils/gradient_utils.py:192-261`)

```python
def ema_gradient_scaling(
    g_render_norm, g_physics_norm, target_ratio,
    ema_state, ema_beta=0.8, power=0.7,
    min_gain=0.1, max_gain=50000.0  # Now 1000 in config
):
    # Compute gain to match target ratio
    target_norm = target_ratio * g_phys
    raw = (target_norm / g_ren) ** power

    # Dynamic cap to prevent explosion
    dyn_cap = max(10.0, 10.0 * (g_phys / g_ren))
    raw = float(np.clip(raw, min_gain, min(max_gain, dyn_cap)))
```

**Issues**:
- EMA smoothing (beta=0.8) is too slow to adapt
- Power function (0.7-0.85) still allows large gains
- Dynamic cap is reactive, not preventive
- **Doesn't fix gradient conflict** - just scales the magnitude

### 2. PCGrad (`utils/gradient_utils.py:118-189`)

```python
if cosine < conflict_threshold:
    # Project out conflicting component
    proj = (min(0.0, dot) / (np.dot(g_p, g_p) + 1e-12)) * g_p
    g_r_proj = g_r - proj
```

**Issues**:
- Only removes conflicting component when `cosine < -0.1`
- **Doesn't work if physics gradient is already small** (projection has no effect)
- Quantile clipping (99.9%) removes outliers but preserves high-frequency noise

### 3. Physics Weight Scheduling (`CompGraph.cpp:222-232`)

```cpp
if (physics_weight_ != 1.0f && has_render_grads_) {
    // Scale physics gradients
    mp.dLdF *= physics_weight_;
    mp.dLdx *= physics_weight_;
}
```

**Critical mistake**: This scales **physics gradients down**, not render gradients!
- `physics_weight=1.0` means physics grads are **not scaled at all**
- `physics_weight=0.8` means physics grads are scaled to 80%
- **Render gradients remain at full strength!**

This is **backwards** - we should be scaling render grads down, not physics grads!

## True Root Cause

**The fundamental issue**: Physics optimization uses a **trust-region method** (line search) that assumes:
1. Gradients point toward a minimum
2. Step sizes can be adjusted to find improvement
3. Objective function is reasonably smooth

When render gradients are added:
- **Assumption 1 violated**: Combined gradient points away from physics minimum
- **Assumption 2 violated**: No step size works (physics vs render trade-off)
- **Assumption 3 violated**: Render loss is non-smooth (edge detection, silhouettes)

## Recommended Solutions

### Option A: Fix Gradient Balancing (Quick Fix)

1. **Reverse physics_weight logic**:
   ```cpp
   // In CompGraph.cpp, line 189-190:
   // Instead of: pt.dLdF += dF_render;
   // Do:
   float render_weight = 1.0f / physics_weight_;  // e.g., 1.0/1.5 = 0.67
   pt.dLdF += render_weight * dF_render;
   pt.dLdx += render_weight * dx_render;
   ```

2. **Increase initial_alpha back to 1.0**:
   ```yaml
   initial_alpha: 1.0  # Trust the scaling, not the step size
   ```

3. **Start with physics-only warmup**:
   ```python
   # Only inject render grads after episode 5
   if ep < 5:
       accumulated_render_grads = None
   ```

### Option B: Separate Physics and Render Optimization (Better)

1. **Decouple objectives**: Don't add gradients together
2. **Alternate optimization**:
   - Pass 1: Optimize physics only → get x(T)
   - Pass 2: Optimize render only → refine x(T) for better appearance
   - Pass 3: Optimize physics with render constraint
3. **Use soft constraint** instead of hard gradient injection

### Option C: Normalize Both Gradients (Safest)

```python
# Before injection:
g_phys_norm = np.linalg.norm([dLdF_phys, dLdx_phys])
g_render_norm = np.linalg.norm([dLdF_render, dLdx_render])

# Normalize to unit vectors
dLdF_phys_unit = dLdF_phys / (g_phys_norm + 1e-12)
dLdx_phys_unit = dLdx_phys / (g_phys_norm + 1e-12)
dLdF_render_unit = dLdF_render / (g_render_norm + 1e-12)
dLdx_render_unit = dLdx_render / (g_render_norm + 1e-12)

# Weighted combination (instead of raw addition)
w_phys = 0.7  # Physics weight
w_render = 0.3  # Render weight

dLdF_combined = w_phys * dLdF_phys_unit + w_render * dLdF_render_unit
dLdx_combined = w_phys * dLdx_phys_unit + w_render * dLdx_render_unit

# Scale back to reasonable magnitude
combined_scale = w_phys * g_phys_norm + w_render * g_render_norm * 0.01
dLdF_final = combined_scale * dLdF_combined
dLdx_final = combined_scale * dLdx_combined
```

## Diagnostic Commands

To verify the issue, check training logs for:

```bash
# 1. Check for line search failures:
grep -i "line search failed" output/*.log

# 2. Check gradient magnitudes:
grep "||g_render||" output/*.log
grep "||g_phys||" output/*.log

# 3. Check for gradient conflicts:
grep "CONFLICT" output/*.log

# 4. Check applied gain values:
grep "Applied gain" output/*.log
```

Expected patterns indicating the issue:
- `||g_render|| >> ||g_phys||` (render grads 100-1000x larger)
- `⚠️ CONFLICT` (negative cosine similarity)
- `Applied gain: [large number] ⚠️ TOO HIGH!`
- `Line search failed` (repeated step rejections)

## Summary

The render loss causes non-convergence because:
1. **Gradient magnitude mismatch**: Render >> Physics (1000x-100000x)
2. **Gradient conflict**: Render and physics point in opposite directions
3. **Wrong scaling logic**: Current code scales physics down instead of render down
4. **Trust-region violation**: Combined gradient doesn't satisfy line search assumptions

**Immediate fix**: Reverse the physics_weight logic to scale render grads down, not physics grads.
