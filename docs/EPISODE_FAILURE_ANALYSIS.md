# Episode Failure Analysis

## Problem Summary

**18 out of 50 episodes (36%) failed during optimization**, leaving empty directories with no renders.

### Failed Episodes
```
ep008, ep014, ep017, ep018, ep019, ep027, ep030-ep038, ep044, ep045, ep049
```

### Failure Pattern

| Training Phase | Failed Episodes | Failure Rate |
|----------------|----------------|--------------|
| Early (ep0-15) | 2/16 | **12.5%** |
| Mid (ep16-31)  | 6/16 | **37.5%** |
| Late (ep32-49) | 10/18 | **55.6%** |

**The failure rate increases dramatically as training progresses!**

---

## Root Cause

Episodes fail when `result.success = False` in the physics optimization. Looking at `utils/training_loop.py:336`:

```python
if png_enabled and result.success:
    # Save visualization
else:
    print(f"Reason: result.success={result.success}")  # Skipped!
```

### Why Do Episodes Fail?

The increasing failure rate (12% → 37% → 55%) suggests **cumulative instability**:

1. **Accumulated Deformation**
   - Each episode builds on the previous state
   - Deformations compound over time
   - Eventually becomes too extreme for physics solver

2. **Numerical Instability**
   - Small errors accumulate across episodes
   - Learning rate decay (ep8, ep16, ep24) may not help
   - Physics gradients become ill-conditioned

3. **Line Search Failure**
   - Line search can't find step size that reduces loss
   - Happens when gradients are too large/small or conflicting
   - Common in late training when already close to target

4. **State Carryover Issues**
   - Grid state carries over between episodes
   - Artifacts accumulate in MPM grid
   - Velocity/deformation fields become unstable

---

## This is NOT a Subgrid Problem

The issue is **optimization convergence**, not rendering or grid resolution. Evidence:

- Empty directories (no files created at all)
- Pattern correlates with learning rate changes
- Increasing failures over time
- `result.success=False` means optimizer gave up

---

## Solutions

### Option 1: Adjust Learning Rate Schedule (Recommended)

The current schedule may decay too aggressively:

```yaml
# Current (may be too aggressive)
episode_schedule:
  8-15:
    optimization:
      initial_alpha: 0.005   # 50% reduction
  16-23:
    optimization:
      initial_alpha: 0.0025  # 75% reduction
  24+:
    optimization:
      initial_alpha: 0.00125 # 87.5% reduction
```

**Try this instead:**

```yaml
# More gradual decay
episode_schedule:
  10-20:
    optimization:
      initial_alpha: 0.008   # 20% reduction
  21-35:
    optimization:
      initial_alpha: 0.006   # 40% reduction
  36+:
    optimization:
      initial_alpha: 0.004   # 60% reduction
```

### Option 2: Increase Line Search Iterations

Give the optimizer more chances to find a valid step:

```yaml
optimization:
  max_gd_iters: 1
  max_ls_iters: 20  # Increased from 10
  gd_tol: 0.001      # Relaxed from 0.0001
```

### Option 3: Reduce Physics Weight for Render Gradients

If render gradients are causing instability:

```yaml
optimization:
  loss:
    render_loss_weight: 100.0  # Reduced from 200.0
```

### Option 4: Disable State Carryover

Reset grid between episodes (may help with accumulated artifacts):

In `run.py:409`, comment out grid carryover:

```python
# cg.promote_last_as_initial(carry_grid=True)
cg.promote_last_as_initial(carry_grid=False)  # Fresh grid each episode
```

### Option 5: Reduce Total Episodes

Train for fewer episodes to avoid late-stage instability:

```yaml
optimization:
  num_animations: 30  # Reduced from 50
```

---

## Quick Fix: Use Only Successful Episodes

Your current video uses all 32 successful episodes, which is fine! You can:

### 1. Accept partial results
The 32 successful episodes show the morphing progression adequately.

### 2. Fill gaps with interpolation

Create smooth video by repeating previous frame:

```python
# In create_video.py, add interpolation option
python create_video.py output/bob/sphere/ --fill-missing
```

### 3. Re-run with better settings

Use recommended settings above and run again:

```bash
python run.py -c configs/Chayo/bob_to_sphere.yaml --png
```

---

## Diagnostic Commands

### Check which episodes failed
```bash
python check_episodes.py output/bob/sphere/
```

### Check optimization convergence
Look for these messages in training output:
```
"Line search failed"
"Gradient too small"
"result.success = False"
```

### Monitor during training
```bash
# Watch for failures in real-time
python run.py -c config.yaml --png 2>&1 | grep -i "success\|failed\|error"
```

---

## Prevention Tips

### 1. Start with fewer episodes
```yaml
num_animations: 25  # Start small, increase if stable
```

### 2. Monitor early episodes
If ep0-10 have failures, stop and adjust parameters.

### 3. Use conservative settings
- Smaller learning rate (`initial_alpha: 0.005`)
- More line search iterations (`max_ls_iters: 20`)
- Lower render loss weight (`render_loss_weight: 100`)

### 4. Test morphing difficulty
Some morphings are harder than others:
- **Easy**: sphere ↔ bob (convex shapes)
- **Medium**: sphere → spot (moderate detail)
- **Hard**: sphere → bunny (high detail, topology change)

---

## Summary

| Issue | Cause | Solution |
|-------|-------|----------|
| Empty episode dirs | Optimization failed (`result.success=False`) | Adjust learning rate |
| Increasing failures | Cumulative instability | Reset grid or reduce episodes |
| Late-stage failures | Extreme deformations | More gradual LR decay |
| Line search fails | Conflicting gradients | Increase `max_ls_iters` |

**This is an optimization convergence issue, not a rendering or subgrid problem.**

The good news: You still got 32/50 successful episodes (64%), which is enough for a decent morphing video!
