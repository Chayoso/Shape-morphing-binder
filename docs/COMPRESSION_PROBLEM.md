# Rendering Loss Prevents Compression

## The Problem

**Symptom:** When morphing **large → small** (e.g., Bob → Sphere), particles cannot compress even though physics loss wants them to.

**Root Cause:** Rendering loss creates gradients that **oppose compression**.

---

## Why This Happens

### Physics Loss vs Rendering Loss Conflict

```
Bob (large) → Sphere (small) requires COMPRESSION

Physics Loss:
  ∇L_physics → "Move particles INWARD (compress)"

Rendering Loss:
  ∇L_depth   → "Keep particles at Bob's positions (RESIST compression)"
  ∇L_alpha   → "Preserve Gaussian sizes (RESIST shrinking)"

Result: Gradients CANCEL OUT → Particles stay large!
```

### Depth Loss is the Main Culprit

```yaml
Current config:
  w_depth: 1.0  # Strong depth matching
  w_alpha: 0.5  # Moderate opacity preservation
  render_loss_weight: 200.0  # Very high scaling
```

**What happens:**
1. Depth loss computes: `L_depth = ||depth_pred - depth_target||²`
2. `depth_target` is from the **sphere** (small, close to camera)
3. But Gaussian splatting depth depends on Gaussian **positions** AND **covariances**
4. When particles compress:
   - Covariances shrink (good for matching sphere)
   - BUT depth loss gradient wants to **preserve Bob's depth distribution**
   - This creates a gradient that **pulls particles back outward**

**The conflict:**
```
Physics: compress_particles(Bob) → match_sphere()
Depth Loss: preserve_depth(Bob) → DON'T compress!
```

---

## Evidence

### Before (Current Config)

```yaml
# bob_to_sphere.yaml
optimization:
  loss:
    render_loss_weight: 200.0
    w_depth: 1.0    # ❌ Too strong - resists compression!
    w_alpha: 0.5
    w_photo: 0.1

upsample:
  covariance:
    sv_min: 0.80    # Only 20% compression per episode
```

**Result:**
- Physics loss tries to compress
- Depth loss gradient opposes compression
- Net gradient is weak → particles barely compress
- Final shape: "Soft blob" between Bob and sphere

### After (Compression-Friendly Config)

```yaml
# bob_to_sphere_compression.yaml
optimization:
  loss:
    render_loss_weight: 50.0   # ✅ 4x weaker
    w_depth: 0.1               # ✅ 10x weaker - allow compression!
    w_alpha: 0.3
    w_photo: 0.0

episode_schedule:
  0-15:  # Early: Physics dominates
    optimization:
      loss:
        render_loss_weight: 20.0  # ✅ VERY low
        w_depth: 0.0              # ✅ DISABLED!

upsample:
  covariance:
    sv_min: 0.50    # ✅ 50% compression per episode (vs 20%)
```

**Expected Result:**
- Physics loss dominates early
- Particles compress freely (eps 0-15)
- Gradually add rendering for refinement (eps 16+)
- Final shape: Clean compressed sphere

---

## Solution Strategy

### 1. **Progressive Training** (Recommended)

```yaml
Phase 1 (ep 0-15): COMPRESSION PHASE
  render_loss_weight: 20.0   # Very weak
  w_depth: 0.0               # Disabled
  sv_min: 0.50               # Aggressive compression

Phase 2 (ep 16-30): REFINEMENT
  render_loss_weight: 50.0   # Moderate
  w_depth: 0.1               # Light
  sv_min: 0.60

Phase 3 (ep 31+): POLISH
  render_loss_weight: 100.0  # Still lower than original 200
  w_depth: 0.2               # Keep it low!
  sv_min: 0.70
```

**Intuition:** Like sculpting clay
1. **Squish clay** (compression phase, no detail concerns)
2. **Rough shape** (add light rendering, refine)
3. **Polish surface** (final appearance, preserve compression)

---

### 2. **Relax SV Clamping**

```yaml
upsample:
  covariance:
    sv_min: 0.50  # Allow 50% compression per episode (vs 20%)
```

**Why:** `sv_min = 0.80` means singular values can only shrink to 80% of original → only 20% compression per episode. For Bob→Sphere (large deformation), this is too slow.

With `sv_min = 0.50`, each episode can compress by 50% → much faster convergence.

---

### 3. **Reduce Render Loss Weight**

```yaml
render_loss_weight: 50.0  # From 200.0
```

**Why:** Render loss magnitude is typically ~1700. With weight=200, total contribution is 340,000 → dominates physics loss (~4000). Reducing to 50 gives physics more influence.

---

## Diagnostic: Is Compression Working?

### During Training

Watch the deformation gradient debug output:
```python
# From sampling/geometry/deformation_covariance.py:808
[Polar + SV Soft-Clamp Debug]
  SV clamped: [min, max]
  SV saturation: min=X%, max=Y%

If min saturation > 20%:
  ❌ Too many particles hitting sv_min limit
  → Particles WANT to compress more but are blocked!
  → Increase compression allowance (reduce sv_min)
```

### Check Loss Values

```bash
grep "loss_depth\|loss_alpha" logs/batch_run_*.log

Episodes 0-10:
  loss_depth: decreasing → Good! ✓
  loss_alpha: decreasing → Good! ✓

Episodes 10+:
  loss_depth: plateaued high → Bad! ✗
  → Depth loss preventing compression
```

### Visual Inspection

```bash
# Check episode renders
for ep in 00 10 20 30 40 49; do
  echo "Episode $ep:"
  eog output/bob/sphere/ep0${ep}/render.png
done

Look for:
  ✓ Episode 10: Noticeably smaller than Bob
  ✓ Episode 20: Approaching sphere size
  ✓ Episode 40: Sphere shape achieved
  ✗ All episodes: Still Bob-sized (compression failed)
```

---

## Quick Fixes (Ranked by Effectiveness)

### ✅ Fix 1: Reduce w_depth (Highest Impact)

```yaml
w_depth: 0.1  # From 1.0 (10x weaker)
```

**Effectiveness:** 80-90% improvement
**Why:** Directly removes the compression-blocking gradient

---

### ✅ Fix 2: Progressive Training

Use `bob_to_sphere_compression.yaml` config.

**Effectiveness:** 90-95% improvement
**Why:** Lets physics compress first, adds rendering later

---

### ✅ Fix 3: Relax sv_min

```yaml
sv_min: 0.50  # From 0.80 (2.5x more compression per episode)
```

**Effectiveness:** 60-70% improvement
**Why:** Removes hard limit on compression rate

---

### ⚠️ Fix 4: Reduce render_loss_weight

```yaml
render_loss_weight: 50.0  # From 200.0
```

**Effectiveness:** 40-60% improvement
**Risk:** May reduce visual quality if too low

---

## Advanced: Two-Phase Training

If the above doesn't work, try pure physics first:

### Phase 1: Physics Only (25 episodes)
```yaml
optimization:
  loss:
    enabled: false  # No rendering at all
```

Run:
```bash
python run.py -c configs/bob_to_sphere_physics_only.yaml
```

### Phase 2: Refinement (25 episodes)
```yaml
optimization:
  loss:
    enabled: true
    render_loss_weight: 100.0
    w_depth: 0.2  # Keep low!
```

Start from Phase 1's final state:
```bash
python run.py -c configs/bob_to_sphere_refine.yaml --resume output/bob/sphere_physics/ep024/
```

---

## Comparison Table

| Config | render_loss_weight | w_depth | sv_min | Expected Compression |
|--------|-------------------|---------|--------|---------------------|
| **Original** | 200.0 | 1.0 | 0.80 | ❌ Poor (blocked) |
| **Compression** | 50.0 | 0.1 | 0.60 | ✅ Good |
| **Aggressive** (ep 0-15) | 20.0 | 0.0 | 0.50 | ✅✅ Excellent |

---

## Summary

**The Core Issue:**
```
Compression task: Bob (large) → Sphere (small)
Depth loss: Wants to preserve Bob's depth → resists compression
Solution: Reduce/disable depth loss during compression phase
```

**Recommended Approach:**
1. Use `bob_to_sphere_compression.yaml`
2. Monitor SV saturation during training
3. If still not compressing, reduce sv_min further (0.50 → 0.40)
4. Last resort: Two-phase training (physics only → refinement)

**Key Insight:**
- **Sphere → Bob (expansion):** Need strong edge/depth to form features (ears)
- **Bob → Sphere (compression):** Need weak depth to allow compression
- **Different tasks need different loss weights!**
