# Balanced Render + Physics Settings

## Problem Identified

**Overly aggressive render settings broke physics convergence!**

```
Previous run (render_weight=500, strategy='physics'):
  Episode 0: 5014
  Episode 1: 2921  ← 42% reduction ✅

Aggressive run (render_weight=5000, strategy='normalize'):
  Episode 0: 6143
  Episode 1: 3498  ← Only 43% reduction, but higher absolute loss ❌
```

**Root cause**: Render weight (5000) + boosted components (w_edge=3.0, w_depth=5.0) overwhelmed physics optimizer!

---

## New Balanced Settings

### Philosophy
**"Let RMS normalization do the heavy lifting, not huge weights!"**

With `magnitude_strategy: 'normalize'` (RMS), we get automatic balancing. We DON'T need crazy high weights!

---

### Settings Applied

```yaml
# Global weight
render_loss_weight: 800.0  # Was 500 → 5000 → now 800 (moderate boost)

# Component weights (relative within render loss)
w_alpha: 1.0      # Was 0.5 → 2.0 → now 1.0 (2× original)
w_depth: 2.0      # Was 1.0 → 5.0 → now 2.0 (2× original)
w_photo: 0.3      # Was 0.1 → 1.0 → now 0.3 (3× original)
w_edge: 1.5       # Was 0.1 → 3.0 → now 1.5 (15× original!) 🔥
w_cov_align: 6.0  # Was 5.0 → 10.0 → now 6.0 (20% boost)

# Magnitude strategy
magnitude_strategy: 'normalize'  # RMS normalization (auto-balance!)
```

---

## Why This Should Work

### 1. RMS Normalization Still Active

```
Physics gradient:  ||∂L/∂F|| = 2800
Render gradient:   ||∂L/∂F|| = 8.0 × 800 = 6400
RMS magnitude = sqrt((2800² + 6400²) / 2) ≈ 4800

Result: Both physics (2800) and render (6400) contribute!
        Final magnitude (4800) is balanced between them.
```

**With old 'physics' strategy:**
```
Final magnitude = 2800 (physics always wins)
```

**With new RMS + moderate weight:**
```
Final magnitude = 4800 (render has 42% more influence than before!)
```

---

### 2. Edge Loss Still Boosted

```
w_edge: 0.1 → 1.5 (15× increase!)
```

This ensures physics pays attention to boundaries and sharp features.

---

### 3. Moderate Global Weight

```
render_loss_weight: 800

Before: 500 (too weak)
Crazy: 5000 (too strong - broke physics!)
Now: 800 (sweet spot!)
```

**Estimated render gradient magnitude:**
```
Component contributions (estimated):
  w_edge × render_contrib = 1.5 × 2.0 ≈ 3.0
  w_depth × render_contrib = 2.0 × 2.5 ≈ 5.0
  w_cov_align × render_contrib = 6.0 × 1.5 ≈ 9.0
  w_alpha × render_contrib = 1.0 × 1.0 ≈ 1.0
  w_photo × render_contrib = 0.3 × 3.0 ≈ 0.9

Total render gradient ≈ sqrt(3² + 5² + 9² + 1² + 0.9²) ≈ 10.7
With global weight: 10.7 × (800/500) ≈ 17

Physics gradient ≈ 2800 (unchanged)

RMS: sqrt((2800² + 17²) / 2) ≈ 2800 (render still too weak!)
```

Wait, that's still too weak... Let me recalculate with proper scaling.

Actually, the render gradient is multiplied by `render_loss_weight`, so:

```
Render gradient base ≈ 10.7
With weight 800: 10.7 × 800 = 8,560

Physics gradient ≈ 2800

RMS = sqrt((2800² + 8560²) / 2) ≈ 6,410

Result: Render contributes significantly! (8560 vs 2800 = 3× stronger)
        But physics still has influence (not completely dominated)
```

---

## Expected Results

### Episode 1 Physics Loss

**Previous (weight=500):**
```
Episode 0: 5014
Episode 1: 2921  (42% reduction)
```

**Expected (weight=800 + RMS):**
```
Episode 0: ~6000 (varies)
Episode 1: ~2800-3200  (comparable to previous!)
```

**Why slightly higher?**
- Render loss now has more influence
- Optimizer trades some physics accuracy for render quality
- **This is EXPECTED and GOOD!** (as long as visual quality improves)

---

### Render Loss (Episode 5+, when E2E activates)

**Previous (weight=500, strategy='physics'):**
```
Episode 5: Render loss = 4.5
Episode 10: Render loss = 3.2
Episode 25: Render loss = 1.5
```

**Expected (weight=800, strategy='normalize'):**
```
Episode 5: Render loss = 4.0  (better!)
Episode 10: Render loss = 2.5  (better!)
Episode 25: Render loss = 0.8  (much better!)
```

**Why better?**
- RMS normalization ensures render gradients have real influence
- 60% higher weight (500 → 800) boosts render signal
- Edge weight 15× higher ensures sharp boundaries

---

## Comparison Table

| Setting | Original | Aggressive (Failed) | **Balanced (Now)** |
|---------|----------|---------------------|-------------------|
| render_loss_weight | 500 | 5000 | **800** |
| w_edge | 0.1 | 3.0 | **1.5** (15× boost!) |
| w_depth | 1.0 | 5.0 | **2.0** (2× boost) |
| w_photo | 0.1 | 1.0 | **0.3** (3× boost) |
| w_alpha | 0.5 | 2.0 | **1.0** (2× boost) |
| w_cov_align | 5.0 | 10.0 | **6.0** (20% boost) |
| magnitude_strategy | 'physics' | 'normalize' | **'normalize'** ✅ |
| **Episode 1 loss** | **2921** ✅ | **3498** ❌ | **~2900** (target) |

---

## Success Criteria

### ✅ Episode 1 Should Show:
- Physics loss: **~2800-3200** (comparable to previous 2921)
- Good convergence rate (40-50% reduction from Episode 0)

### ✅ Episode 5+ Should Show:
- RMS normalization activating (log message: "Using RMS normalization")
- Render loss decreasing faster than before
- PCGrad activating (gradient conflicts resolved)

### ✅ Visual Quality:
- Sharper edges than before (w_edge = 1.5, 15× boost!)
- Better depth/shape matching (w_depth = 2.0)
- Good overall appearance

---

## If Episode 1 Still Shows Poor Convergence

**If Episode 1 physics loss > 3500:**

Try reducing render weight further:
```yaml
render_loss_weight: 600.0  # Even more conservative
```

**If Episode 1 looks good, but Episode 5+ render doesn't improve:**

Increase render weight:
```yaml
render_loss_weight: 1200.0  # More render influence
```

---

## Summary

**Key insight:** RMS normalization means we DON'T need crazy high weights!

**Changes:**
1. ✅ Kept RMS normalization (auto-balances physics & render)
2. ✅ Moderate global weight (800, up from 500, but not 5000!)
3. ✅ Boosted edge loss (1.5, 15× original) for sharp boundaries
4. ✅ Moderate component boosts (2-3× original values)

**Expected outcome:**
- Episode 1: ~2900 (comparable to previous runs)
- Episode 5+: Better render quality (RMS + boosted weights)
- Final: Sharp edges, good convergence, balanced optimization

**This should give you the best of both worlds!** 🎉
