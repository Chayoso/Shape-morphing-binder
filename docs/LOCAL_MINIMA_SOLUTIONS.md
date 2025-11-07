# Local Minima Problem: Why Bunny Ears Don't Form

## The Core Issue

**Problem:** Physics optimizer gets stuck in local minimum where particles form a "smooth blob" without sharp features (ears, tail).

**Evidence:**
- Physics loss plateaus (~900)
- Overall shape looks bunny-ish, but no ears
- F-field interpolation can't create features that physics particles never reached

## Root Causes

### 1. **Insufficient Geometric Signal**

```
Current loss weights:
  w_photo: 0.1   → Appearance (color)
  w_depth: 1.0   → Geometry (3D shape)
  w_edge: 0.1    → Boundaries (ears!)

Problem: Photo loss dominates perception
        Edge loss (10x weaker than depth) can't force ears
```

### 2. **Smoothing Pressure from Render Loss**

```
render_loss_weight: 200.0 (very high!)
  → Strong pressure to match target appearance
  → BUT target is smooth in areas without ears
  → Optimizer chooses smooth blob (local minimum)
```

### 3. **All-or-Nothing Training**

```
Episode 0: Apply ALL losses simultaneously
  → Conflicts between physics, depth, photo, edge
  → Optimizer picks "easy" compromise (smooth shape)
  → Never escapes to "hard" solution (with ears)
```

## Solution Strategy: Coarse-to-Fine Training

### **Intuition:**

```
Human sculptor approach:
  1. Rough out body shape (global)
  2. Add major features (ears, legs)
  3. Refine details (face, texture)

Physics optimizer needs same approach!
```

### **Implementation:**

```yaml
# Stage 1 (ep 0-15): GLOBAL SHAPE ONLY
w_depth: 3.0    # Heavy 3D shape emphasis
w_edge: 0.5     # Light boundary
w_photo: 0.0    # Ignore appearance

Result: Particles form body → good foundation

# Stage 2 (ep 16-35): ADD SHARP FEATURES
w_depth: 2.0
w_edge: 1.5     # 🔥 Strong edge → pulls particles to ears!
w_photo: 0.05

Result: Ears start forming → gradient points to ears

# Stage 3 (ep 36+): POLISH DETAILS
w_depth: 1.5
w_edge: 2.0     # Max edge (preserve ears)
w_photo: 0.1

Result: Refine appearance without losing ears
```

---

## Quick Fixes (Ordered by Effectiveness)

### ✅ **Fix 1: Increase Edge Loss** (Highest Impact)

```yaml
w_edge: 1.0  # From 0.1 (10x increase!)
```

**Why it works:**
- Ears have strong edge gradients
- Edge loss provides direct signal: "Move particles HERE (to edge)"
- 10x stronger signal escapes local minimum

**Expected improvement:** 60-80% chance of forming basic ears

---

### ✅ **Fix 2: Progressive Loss Annealing**

Use `sphere_to_bunny_improved.yaml` I just created.

**Why it works:**
- Avoids conflicting gradients early on
- Builds features incrementally (body → ears → details)
- Like curriculum learning for physics

**Expected improvement:** 80-90% chance of good ears

---

### ✅ **Fix 3: Boost Depth Loss Early**

```yaml
episode_schedule:
  0-20:
    optimization:
      loss:
        w_depth: 3.0  # 3x current value
```

**Why it works:**
- 3D shape provides stronger signal than 2D photo
- Ears are primarily a depth feature (stick out!)

**Expected improvement:** 40-60% chance of improvement

---

### ⚠️ **Fix 4: Reduce Render Loss Weight**

```yaml
render_loss_weight: 100.0  # From 200.0
```

**Why it works:**
- Less smoothing pressure
- Physics has more freedom to explore

**Risk:** May reduce visual quality

---

### ⚠️ **Fix 5: Increase Particles**

```yaml
simulation:
  points_per_cell_cuberoot: 4  # From 3
```

**Why it works:**
- More particles in ear region
- Better coverage = better F-field

**Risk:** Slower simulation (1.78x more particles)

---

## Diagnostic: Check If It's Working

### **During Training:**

Watch the depth loss:
```python
# From logs
grep "loss_depth" logs/batch_run_*.log

If decreasing: Good! Ears forming ✓
If plateaued: Still in local minimum ✗
```

### **After Training:**

Visual inspection:
```bash
# Check final render
eog output/sphere/bunny/ep049/render.png

Look for:
  ✓ Ears visible (even if small)
  ✓ Depth variation in top region
  ✗ Smooth blob on top (no ears)
```

---

## Advanced Solutions (If Above Fails)

### **Option 1: Two-Phase Training**

**Phase 1: Physics-only (25 eps)**
```yaml
optimization:
  loss:
    enabled: false  # No rendering
```

**Phase 2: Refinement (25 eps)**
```yaml
optimization:
  loss:
    enabled: true
    render_loss_weight: 150.0
```

---

### **Option 2: Targeted Physics Loss**

Add auxiliary loss that explicitly checks ear coverage:

```python
# Pseudo-code
ear_region_target = bunny.vertices[z > 0.7 * z_max]
ear_region_particles = particles[z > 0.7 * z_max]

loss_ear_coverage = chamfer_distance(
    ear_region_particles,
    ear_region_target
)

total_loss += 100.0 * loss_ear_coverage
```

---

### **Option 3: Restart with Better Initial State**

Pre-stretch sphere toward bunny proportions:

```python
# Create elongated_sphere.obj
vertices = sphere.vertices
vertices[:, 2] *= 1.5  # Stretch vertically
vertices[:, 1] *= 0.8  # Compress horizontally
# → More bunny-like starting shape
```

---

## Testing the Fix

### **Test 1: Run Improved Config**

```bash
python run.py -c configs/Chayo/sphere_to_bunny_improved.yaml --png
```

**Check at ep 20:** Should see basic ear outlines
**Check at ep 35:** Should see clear ears
**Check at ep 49:** Should see refined ears

### **Test 2: Compare Loss Curves**

```bash
python view_losses.py output/sphere/bunny/ --plot

Compare:
  Old: loss_edge stays high (ears not forming)
  New: loss_edge decreases (ears forming!) ✓
```

---

## Expected Results

### **Before (Current Config):**
```
Episode 49:
  Physics loss: ~900
  loss_edge: ~0.08 (still high, edges not matching)
  Visual: Smooth blob, no ears
```

### **After (Improved Config):**
```
Episode 20:
  loss_edge: ~0.05 (starting to decrease)
  Visual: Ear outlines visible

Episode 35:
  loss_edge: ~0.02 (much better!)
  Visual: Clear ears, maybe thin

Episode 49:
  loss_edge: ~0.01 (good!)
  Visual: Well-defined ears ✓
```

---

## Summary

| Fix | Difficulty | Effectiveness | Recommended |
|-----|-----------|---------------|-------------|
| Increase w_edge | ⭐ Easy | ⭐⭐⭐⭐ | ✅ YES |
| Progressive annealing | ⭐⭐ Medium | ⭐⭐⭐⭐⭐ | ✅ YES |
| Boost w_depth early | ⭐ Easy | ⭐⭐⭐ | ✅ YES |
| Reduce render weight | ⭐ Easy | ⭐⭐ | ⚠️ Maybe |
| More particles | ⭐ Easy | ⭐⭐⭐ | ⚠️ If needed |
| Two-phase training | ⭐⭐⭐ Hard | ⭐⭐⭐⭐ | ⚠️ Last resort |

**Recommended approach:** Try `sphere_to_bunny_improved.yaml` first. If that doesn't work, increase particles and try two-phase training.
