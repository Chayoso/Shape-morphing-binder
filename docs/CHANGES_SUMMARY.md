# Render Loss Boost - All Changes

## Goal
Make render loss more influential on physics optimization (user reported "physics dominates all of them").

---

## Changes Applied to `configs/examples/sphere_to_spot.yaml`

### 1. Magnitude Strategy (Line 62)
```yaml
# BEFORE:
magnitude_strategy: 'physics'  # Conservative - physics magnitude always dominates

# AFTER:
magnitude_strategy: 'weighted'  # Balanced - both physics and render contribute to magnitude
```

**Impact**: This is the **most critical change**. With 'physics' mode, render gradients were always rescaled to physics magnitude, keeping physics dominant. With 'weighted' mode, both losses contribute equally to final gradient magnitude.

---

### 2. Global Render Loss Weight (Line 47)
```yaml
# BEFORE:
render_loss_weight: 500.0

# AFTER:
render_loss_weight: 1000.0  # 2× increase!
```

**Impact**: Render gradients are now 2× stronger before combination with physics gradients.

---

### 3. Individual Component Weights (Lines 51-55)

| Component | Before | After | Change |
|-----------|--------|-------|--------|
| w_alpha   | 0.5    | 1.0   | +100% (2×) |
| w_depth   | 1.0    | 2.0   | +100% (2×) |
| w_photo   | 0.1    | 0.5   | +400% (5×) |
| **w_edge** | **0.1** | **1.0** | **+900% (10×)** ← Physics was ignoring edges! |
| w_cov_align | 5.0  | 8.0   | +60% |

**Impact**:
- **Edge loss** now 10× stronger → physics will respect shape boundaries
- **Depth loss** 2× stronger → better 3D shape matching
- **Photometric loss** 5× stronger → better color/appearance
- **Covariance alignment** stronger → better F-field guidance

---

## Combined Effect

### Gradient Magnitude Estimate

**Before:**
```
Physics:  ||∂L/∂F|| = 2800
Render:   ||∂L/∂F|| = 5.0 × 500 = 2500
Strategy: 'physics' → final magnitude = 2800 (physics wins!)
```

**After:**
```
Physics:  ||∂L/∂F|| = 2800
Render:   ||∂L/∂F|| = 10.0 × 1000 = 10,000  (boosted by higher component weights!)
Strategy: 'weighted' → final magnitude = (2800² + 10000²)/(2800 + 10000) ≈ 7800
```

**Result**: Render gradient magnitude now **dominates** the combined gradient! 🎉

---

## Expected Training Behavior

### Physics Loss
- May plateau **higher** than before (e.g., ~500 instead of ~350)
- This is **GOOD** - optimizer sacrifices physics accuracy for visual quality

### Render Loss
- Should decrease **faster** than before
- Final render loss should be much lower (e.g., ~0.3 instead of ~1.5)

### PCGrad Activation
- More likely to activate due to gradient conflicts
- Look for log messages: "Projecting render gradient..."

### Visual Quality
- **Sharper edges** (edge loss working)
- **Better depth** (depth loss working)
- **Better appearance** (photometric loss working)
- **More accurate shape** (all losses working together)

---

## How to Verify

### 1. Check Logs for Gradient Magnitudes
```
[Episode 5, Pass 1] GRADIENT CHECK:
  ||∂L_render/∂F||: 10500.3  ← Should be LARGER than physics!
  ||∂L_phys/∂F||:   2800.5
  cos(g_phys, g_render): -0.15  ← Negative means conflict!
  PCGrad activated: True  ← Should see this!
```

### 2. Run Convergence Checker
```bash
python check_convergence.py logs/your_new_run.log
```

Look for:
- Render loss decreasing faster
- Physics loss plateau higher (this is OK!)

### 3. Visual Inspection
Compare renderings with previous run:
- Better edge sharpness
- Better silhouette quality
- Better overall appearance

---

## If You Need Even More Render Influence

### Option A: Increase weight further
```yaml
render_loss_weight: 2000.0  # or even 5000!
```

### Option B: Use 'max' strategy
```yaml
magnitude_strategy: 'max'  # Most aggressive - uses max of physics/render
```

### Option C: Increase edge even more
```yaml
w_edge: 2.0  # or 5.0 for very sharp edges
```

---

## Summary

✅ Changed gradient combination strategy (physics → weighted)
✅ Doubled global render weight (500 → 1000)
✅ Boosted all component weights (especially edge: 0.1 → 1.0)

**Expected result**: Render loss now significantly influences physics optimization! Physics will deform to match visual appearance, not just mass distribution.

**Run it and check the logs!** You should see render gradients dominating now.
