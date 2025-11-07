# How to Use PCGrad - Quick Guide

## TL;DR

Add **one line** to your config:

```yaml
optimization:
  use_session_mode: false  # ← Add this!
```

That's it! PCGrad is now enabled by default.

---

## What You'll See in Terminal

### 1. Check Execution Mode

**GOOD (Legacy Mode - PCGrad Available):**
```
✅ [LEGACY MODE] Episode 0 with 1 passes - PCGrad available!
```

**BAD (Session Mode - No PCGrad):**
```
⚠️  [SESSION MODE] Episode 0 - PCGrad NOT available in session mode!
    To use PCGrad, add 'use_session_mode: false' to your config
```

### 2. Check PCGrad Status

During each pass, you'll see:
```
  ├─ PCGrad enabled: True, Cosine: -0.234, Conflict status: ⚠️ CONFLICT
```

**Conflict Status:**
- `⚠️ CONFLICT` - Gradients oppose (cosine < -0.3)
- `~ neutral` - Gradients somewhat independent (-0.3 ≤ cosine ≤ 0.3)
- `✓ aligned` - Gradients cooperate (cosine > 0.3)

### 3. Check PCGrad Activation

When conflict is detected (cosine < -0.1):
```
🔥 [PCGrad] Conflict detected (cos=-0.234), applying gradient projection
```

When no conflict:
```
(No PCGrad message - gradients combined directly)
```

---

## Example Config

```yaml
# configs/sp_to_by/sphere_to_bunny.yaml

input_mesh_path: "assets/isosphere.obj"
target_mesh_path: "assets/bunny.obj"

optimization:
  num_animations: 50
  num_timesteps: 10
  initial_alpha: 0.01

  # 🔥 FORCE LEGACY MODE (enables PCGrad)
  use_session_mode: false

  # PCGrad is enabled by default - no need to specify!
  # To disable: use_pcgrad: false

  loss:
    render_loss_weight: 100.0
    w_depth: 2.0
    w_edge: 2.0
```

---

## Verifying PCGrad Works

### Method 1: Real-Time Logs

```bash
python run.py -c configs/sp_to_by/sphere_to_bunny.yaml --png
```

Watch for:
1. `✅ [LEGACY MODE]` at episode start
2. `PCGrad enabled: True` during passes
3. `🔥 [PCGrad] Conflict detected` when conflicts occur

### Method 2: Save and Grep Logs

```bash
python run.py -c configs/sp_to_by/sphere_to_bunny.yaml --png 2>&1 | tee logs/test.log

# Check mode
grep "LEGACY MODE\|SESSION MODE" logs/test.log | head -5

# Check PCGrad status
grep "PCGrad enabled" logs/test.log | head -10

# Check PCGrad activations
grep "🔥 \[PCGrad\]" logs/test.log
```

### Method 3: Use Verification Script

```bash
python verify_pcgrad.py logs/test.log
```

---

## Troubleshooting

### "I see SESSION MODE in logs"

**Problem:** PCGrad is NOT available in session mode

**Fix:** Add to your config:
```yaml
optimization:
  use_session_mode: false
```

### "PCGrad enabled: False"

**Problem:** PCGrad was explicitly disabled

**Fix:** Remove this line from config:
```yaml
optimization:
  use_pcgrad: false  # ← Remove this!
```

Or change to:
```yaml
optimization:
  use_pcgrad: true
```

### "PCGrad enabled: True but no conflict messages"

**This is OKAY!** It means:
- Gradients are aligned (cosine ≥ -0.1)
- No conflicts to resolve
- PCGrad is active but not needed

This is **expected** in:
- Early episodes (warmup phase, ep < 5)
- Late episodes (optimization converging)

### "ImportError: No module named 'diff_gauss'"

**Problem:** Missing dependency

**Fix:** Install diff-gaussian-rasterization:
```bash
cd submodules/diff-gaussian-rasterization
pip install -e .
```

---

## Expected Behavior by Phase

### Early Episodes (0-5)
```
✅ [LEGACY MODE] Episode 2 with 1 passes - PCGrad available!
[Warmup] Episode 2 < 5: Physics-only (skipping render grads)
```
- No render gradients → No conflicts → No PCGrad needed

### Mid Episodes (5-20)
```
✅ [LEGACY MODE] Episode 8 with 1 passes - PCGrad available!
  ├─ PCGrad enabled: True, Cosine: -0.234, Conflict status: ⚠️ CONFLICT
🔥 [PCGrad] Conflict detected (cos=-0.234), applying gradient projection
```
- Conflicts common as render loss activates
- PCGrad resolves conflicts

### Late Episodes (20+)
```
✅ [LEGACY MODE] Episode 35 with 1 passes - PCGrad available!
  ├─ PCGrad enabled: True, Cosine: 0.456, Conflict status: ✓ aligned
```
- Gradients become aligned as optimization converges
- PCGrad rarely needed (but still active)

---

## Performance Note

**Trade-off:** Legacy mode is **10-15x slower** than session mode

- Session mode: Fast but **NO PCGrad** ❌
- Legacy mode: Slower but **PCGrad works** ✅

**Recommendation:** Use legacy mode until PCGrad is implemented in session mode.

---

## All Experiment Configs Ready

All configs in `configs/sp_to_by/` have been updated with `use_session_mode: false`:

- ✅ exp_test_fixes.yaml
- ✅ exp_lr_gentle.yaml
- ✅ exp_lr_constant.yaml
- ✅ exp_sv_relaxed.yaml
- ✅ exp_sv_tight.yaml
- ✅ exp_ls_patient.yaml
- ✅ exp_progressive_annealing.yaml
- ✅ exp_particles_high.yaml
- ✅ exp_best_practices.yaml
- ✅ exp_aggressive_stable.yaml
- ✅ exp_bunny_render_20.yaml
- ✅ exp_bunny_render_50.yaml
- ✅ exp_bunny_render_100.yaml
- ✅ exp_bunny_render_150.yaml

**You can run any of these and PCGrad will work!**

---

**Ready to test!** 🚀
