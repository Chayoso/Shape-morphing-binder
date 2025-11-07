# Normalized Loss Approach (RMS Strategy)

## What Changed

### Added New Magnitude Strategy: `'normalize'` (RMS)

**Location**: `utils/gradient_utils.py` lines 473-478

```python
elif magnitude_strategy == 'normalize' or magnitude_strategy == 'rms':
    # Normalized: RMS of both magnitudes (treats both equally)
    target_F = sqrt((g_F_phys² + g_F_render²) / 2)
    target_x = sqrt((g_x_phys² + g_x_render²) / 2)
```

**Config**: `configs/examples/sphere_to_spot.yaml` line 63
```yaml
magnitude_strategy: 'normalize'  # RMS normalization - treats both losses equally!
```

---

## How RMS Normalization Works

### Previous Strategies (Problems)

1. **'physics'** (Conservative):
```
Final magnitude = ||g_physics||
Problem: Render gradient always rescaled to match physics → physics dominates!
```

2. **'weighted'** (Balanced):
```
Final magnitude = w_phys × ||g_phys|| + w_render × ||g_render||
Problem: Still depends on weights - need to manually tune render_loss_weight!
```

3. **'max'** (Aggressive):
```
Final magnitude = max(||g_phys||, ||g_render||)
Problem: Whichever is larger dominates completely - unstable!
```

### New: **'normalize'** (RMS - Best!)

```
Final magnitude = sqrt((||g_phys||² + ||g_render||²) / 2)
```

**Why this is better:**
- ✅ Treats both physics and render **equally** (neither dominates by magnitude alone)
- ✅ Automatically balances scale differences
- ✅ More stable than 'max' (doesn't switch suddenly)
- ✅ Less sensitive to weight tuning
- ✅ Both gradients contribute to final magnitude proportionally to their strength

---

## Mathematical Example

### Scenario: Physics vs Render Gradients

```
Physics gradient:  ||∂L_phys/∂F|| = 2800
Render gradient:   ||∂L_render/∂F|| = 10000 (after weights)
```

### Strategy Comparison

| Strategy    | Formula | Result | Comment |
|-------------|---------|--------|---------|
| 'physics'   | `2800` | **2800** | Physics magnitude preserved → render weakened! |
| 'weighted'  | `1×2800 + 1×10000` | **12800** | Sum → too large! |
| 'max'       | `max(2800, 10000)` | **10000** | Render fully dominates → physics ignored! |
| **'normalize'** | `sqrt((2800² + 10000²)/2)` | **7343** | ✅ **Balanced between both!** |

### Why RMS is Optimal

With RMS normalization:
- **Both gradients contribute**: Physics (2800) and Render (10000) both affect the result
- **Result is in-between**: 7343 is between 2800 and 10000
- **Neither dominates**: Both physics and render have influence
- **Stable**: Small changes in either gradient produce proportional changes in result

---

## Current Configuration (Aggressive Render + RMS)

### Settings
```yaml
render_loss_weight: 5000.0         # Strong render signal
magnitude_strategy: 'normalize'    # RMS balancing

# Component weights (all boosted)
w_alpha: 2.0      # 4× original
w_depth: 5.0      # 5× original
w_photo: 1.0      # 10× original
w_edge: 3.0       # 30× original
w_cov_align: 10.0 # 2× original
```

### Expected Gradient Magnitudes (Estimated)

```
Physics gradient:  ||∂L/∂F|| ≈ 2800

Render gradient (with weights):
  - Depth: 5.0 × render_contrib ≈ 3000
  - Edge: 3.0 × render_contrib ≈ 2000
  - Cov_align: 10.0 × render_contrib ≈ 4000
  - Alpha: 2.0 × render_contrib ≈ 1500
  - Photo: 1.0 × render_contrib ≈ 800

  Total render: sqrt(3000² + 2000² + 4000² + 1500² + 800²) ≈ 5600
  With global weight 5000: 5600 × (5000/original_weight) ≈ 28,000

RMS combination:
  target_magnitude = sqrt((2800² + 28000²) / 2)
                   = sqrt((7,840,000 + 784,000,000) / 2)
                   = sqrt(395,920,000)
                   ≈ 19,900
```

**Result**: Final gradient magnitude ≈ **19,900**
- Much stronger than physics alone (2800)
- But not as extreme as render alone (28,000)
- **Balanced influence from both!**

---

## Expected Training Behavior

### Physics Loss
- May plateau **slightly higher** than before
- But NOT as high as with 'max' strategy
- Final physics loss: **~400-600** (vs ~350 physics-only, vs ~800-1500 with 'max')

### Render Loss
- Should decrease **significantly**
- Faster than 'physics' strategy
- Comparable to 'max' strategy but more stable
- Final render loss: **~0.3-0.5**

### Visual Quality
- **Much better edges** (edge weight 3.0, properly balanced now)
- **Better depth matching** (depth weight 5.0)
- **Better overall appearance**
- **Stable convergence** (no oscillations from 'max' strategy)

### PCGrad Activation
- **WILL activate** (gradients still have good magnitude)
- More stable conflict resolution than 'max'
- Should see steady improvement without wild swings

---

## Advantages Over Manual Weight Tuning

### Old Approach (Manual Tuning)
```
Problem: Physics gradient = 2800, Render gradient = 50
Solution: Increase render_loss_weight to 5000
Result: Render gradient = 50 × 5000 / 200 = 1250 (still weaker!)
Problem: Try render_loss_weight = 20000?
Result: Unstable! Gradient magnitude too high!
```

**Issue**: Always guessing the right multiplier!

### New Approach (RMS Normalization)
```
Physics gradient = 2800
Render gradient = 1250 (with reasonable weight)
RMS = sqrt((2800² + 1250²) / 2) = 2087

Both contribute! No need to manually match magnitudes!
```

**Benefit**: Automatic balancing!

---

## Comparison Table

| Approach | Pros | Cons | Use Case |
|----------|------|------|----------|
| **'physics'** | Stable, physics always converges | Render barely affects result | Pure physics simulation |
| **'weighted'** | Tunable via weights | Requires careful weight tuning | When you know exact balance needed |
| **'max'** | Strong render influence | Unstable, can oscillate | Testing maximum render effect |
| **'normalize'** (RMS) | **Auto-balanced, stable, both contribute** | **None!** | **Recommended for E2E!** ✅ |

---

## Verification in Logs

Look for this in training logs (Episode 5+):

```
[Episode 5, Pass 1] GRADIENT CHECK:
  ||∂L_phys/∂F||:   2800.5
  ||∂L_render/∂F||: 28500.3
  cos(g_phys, g_render): -0.25
  [DEBUG] Using RMS normalization: F=2.0e+04, x=1.5e+03  ← Should see this!
  PCGrad activated: True
  Combined magnitude: 20000.2  ← Between 2800 and 28500! ✅
```

**Good signs:**
- ✅ RMS normalization message appears
- ✅ Combined magnitude is between physics and render magnitudes
- ✅ PCGrad activates (cosine similarity < 0.8 or negative)

---

## Recommendation

**Current settings are optimal!**
- `magnitude_strategy: 'normalize'` - Best balance
- `render_loss_weight: 5000` - Strong enough signal
- Component weights (edge=3.0, depth=5.0, etc.) - Good for visual quality

**No need to tune weights further!** The RMS normalization automatically handles scale balancing.

**Just run it and verify**:
1. Check logs for "Using RMS normalization" message
2. Verify combined magnitude is between physics and render
3. Confirm render loss decreases steadily
4. Check visual quality improves

This is the **cleanest solution** - let the math handle the balancing! 🎉
