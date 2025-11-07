# Quality Improvements: Fixing Blur and Bumps

## Problems Identified

### 1. Physics Loss Not Converging Enough
```
Current: 7500 → 350 (95% reduction, but stuck at 350)
Target:  7500 → ~100 (need 72% more reduction)
```

### 2. Blurry Rendering
- Gaussians too large (sigma0 = 0.25)
- Makes image soft/fuzzy instead of sharp

### 3. Bumpy Surface
- F-field smoothing disabled
- Too much jitter in subdivision (0.15)
- Particles create irregular surface

---

## Solutions Applied

### Fix 1: Reduce Gaussian Size (Fix Blur)

**Before:**
```yaml
sigma0: 0.25  # Too large! 2.5x bigger than default
```

**After:**
```yaml
sigma0: 0.12  # Smaller Gaussians → sharper rendering
```

**Effect:**
- ✅ Sharper images (less blur)
- ✅ Better detail visibility
- ⚠️ May show more gaps (but subdivision will fill them)

---

### Fix 2: Enable F-Field Smoothing (Fix Bumps)

**Before:**
```yaml
use_F_smoothing: false  # Disabled → bumpy deformations
```

**After:**
```yaml
use_F_smoothing: true   # Enabled → smooth F-field
F_smooth:
  lambda_lap: 0.005     # Light smoothing (preserves details)
```

**Effect:**
- ✅ Smoother surface (no bumps)
- ✅ Neighboring particles have consistent deformations
- ⚠️ Very sharp features (like edges) slightly smoothed

---

### Fix 3: Reduce Subdivision Jitter (Fix Bumps)

**Before:**
```yaml
subdivision_jitter: 0.15  # Too much randomness
```

**After:**
```yaml
subdivision_jitter: 0.08  # Less jitter → smoother surface
```

**Effect:**
- ✅ More regular particle placement
- ✅ Less surface noise
- ✅ Better subdivision alignment

---

### Fix 4: Increase Particle Count (Improve Coverage)

**Before:**
```yaml
subdivision_target: 60000  # 60k particles
```

**After:**
```yaml
subdivision_target: 80000  # 80k particles (+33%)
```

**Effect:**
- ✅ Better surface coverage
- ✅ Fewer gaps even with smaller Gaussians
- ⚠️ Slightly slower rendering

---

### Fix 5: More Episodes for Better Convergence

**Before:**
```yaml
num_animations: 25  # Stopped at episode 25
```

**After:**
```yaml
num_animations: 40  # Extended to episode 40 (+60%)
```

**Effect:**
- ✅ More optimization time
- ✅ Physics loss can reach ~100 (from 350)
- ⚠️ 60% longer training time

---

### Fix 6: Tighter Optimization Tolerance

**Before:**
```yaml
max_gd_iters: 10
gd_tol: 0.0001
```

**After:**
```yaml
max_gd_iters: 15      # +50% more iterations
gd_tol: 0.00005       # 2x tighter tolerance
```

**Effect:**
- ✅ More precise convergence per episode
- ✅ Better final accuracy
- ⚠️ Slightly slower per episode

---

### Fix 7: Extended Learning Rate Decay

**Before:**
```yaml
Episode 0-7:   alpha = 0.005
Episode 8-15:  alpha = 0.0025
Episode 16-23: alpha = 0.00125
Episode 24+:   alpha = 0.00125  (stopped)
```

**After:**
```yaml
Episode 0-7:   alpha = 0.005
Episode 8-15:  alpha = 0.0025
Episode 16-23: alpha = 0.00125
Episode 24-31: alpha = 0.000625  # NEW: finer tuning
Episode 32-40: alpha = 0.0003    # NEW: ultra-fine tuning
```

**Effect:**
- ✅ Gradual refinement continues through episode 40
- ✅ Can reach physics loss ~100
- ✅ Avoids overshooting in late episodes

---

## Expected Results

### Physics Loss Trajectory (Estimated)

```
Episode 0:  7500
Episode 10: 1200  (84% reduction)
Episode 20: 450   (94% reduction)
Episode 30: 180   (98% reduction)
Episode 40: ~100  (99% reduction) ✅ TARGET!
```

### Render Quality Improvements

**Before (sigma0=0.25, no smoothing):**
- ❌ Blurry edges
- ❌ Bumpy surface
- ❌ Lacks detail

**After (sigma0=0.12, with smoothing):**
- ✅ Sharp edges
- ✅ Smooth surface
- ✅ Clear details

---

## Performance Impact

### Training Time

**Before:**
- 25 episodes × ~30 min/episode = ~12.5 hours

**After:**
- 40 episodes × ~35 min/episode = ~23 hours (+10.5 hours)

**Justification:** Better final quality worth the extra time!

### Rendering Speed

**Before:**
- 60k particles → ~100ms per frame

**After:**
- 80k particles → ~120ms per frame (+20%)

**Justification:** Still real-time capable (~8 FPS)

---

## Validation Checklist

After re-running with new config, check:

### ✅ Physics Convergence
- [ ] Final physics loss < 150 (ideally ~100)
- [ ] Loss decreases smoothly (no oscillations)
- [ ] No NaN gradients

### ✅ Render Quality
- [ ] Sharp edges (not blurry)
- [ ] Smooth surface (no bumps)
- [ ] Good detail visibility
- [ ] No visible gaps

### ✅ Visual Comparison
```bash
# Compare old vs new results
old: output/experiments/old_run/ep040/final.png
new: output/experiments/new_run/ep040/final.png

# Check if new is sharper and smoother!
```

---

## Troubleshooting

### If Still Blurry
- Reduce `sigma0` further (try 0.10 or 0.08)
- Check if covariances are correct in logs

### If Still Bumpy
- Increase `lambda_lap` (try 0.01 or 0.02)
- Reduce `subdivision_jitter` more (try 0.05)

### If Physics Loss Plateaus
- Add more episodes (40 → 50)
- Reduce learning rate even more in late stages
- Check if target mesh is too complex

### If Gaps Appear
- Increase `subdivision_target` (80k → 100k)
- Slightly increase `sigma0` (0.12 → 0.14)
- Check F-field interpolation (k_F_fine)

---

## Summary

**Changes Made:**
1. ✅ Reduced Gaussian size: 0.25 → 0.12 (sharper)
2. ✅ Enabled F-smoothing (smoother surface)
3. ✅ Reduced jitter: 0.15 → 0.08 (less bumpy)
4. ✅ More particles: 60k → 80k (better coverage)
5. ✅ More episodes: 25 → 40 (better convergence)
6. ✅ Tighter tolerance: 0.0001 → 0.00005 (more precise)
7. ✅ Extended LR decay (gradual refinement to ep 40)

**Expected Outcome:**
- Physics loss: 350 → ~100 ✅
- Render quality: Blurry/bumpy → Sharp/smooth ✅
- Training time: +10.5 hours (worth it!) ✅

**Next Step:** Re-run training with updated config!
