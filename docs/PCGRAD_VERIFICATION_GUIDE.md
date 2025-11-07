# PCGrad Verification Guide

## ✅ PCGrad Implementation Status: CORRECT

**Analysis:** The PCGrad implementation is mathematically correct and enabled by default.

**Full Analysis:** See `docs/PCGRAD_ANALYSIS.md`

---

## Quick Verification

### Method 1: Check Logs During Training

Run training and grep for PCGrad messages:

```bash
python run.py -c configs/sp_to_by/exp_test_fixes.yaml --png 2>&1 | tee logs/test.log
grep "PCGrad" logs/test.log
```

**Expected output if PCGrad is working:**
```
🔥 [PCGrad] Conflict detected (cos=-0.234), applying gradient projection
├─ pcgrad_cosine: -0.234
├─ pcgrad_applied: True
├─ pcgrad_projection_scale: 0.123
```

**If you see this:** ✅ PCGrad is working correctly!

**If you don't see this:** Either:
- ✅ No conflicts detected (gradients are aligned, PCGrad not needed)
- ⚠️ PCGrad disabled (check config)

---

### Method 2: Use Verification Script

After training completes:

```bash
python verify_pcgrad.py logs/test.log
```

**This checks:**
1. ✅ PCGrad is enabled
2. ✅ Conflict detection is working
3. ✅ PCGrad is applied when needed
4. ✅ Episode success rate
5. ✅ Gradient magnitude balance

**Example output:**
```
======================================================================
PCGrad Verification Report
======================================================================

✅ Check 1: PCGrad Enabled
   Status: ✓ YES

✅ Check 2: Conflict Detection
   Found 30 conflict checks
   Average cosine: -0.123
   Conflicts detected (cos < -0.1): 12/30
   Range: [-0.456, 0.234]

✅ Check 3: PCGrad Application
   Times applied: 12
   Projection scale range: [0.034, 0.234]
   Average projection scale: 0.123

✅ Check 4: Episode Success
   Successful episodes: 10
   Failed episodes: 0
   Success rate: 100.0%

✅ Check 5: Gradient Magnitudes
   ||g_render|| range: [1.000e+00, 1.000e+00]
   ||g_phys||   range: [8.234e-03, 1.234e-02]
   Ratio (render/phys): [81.1, 121.5]
   Average ratio: 97.3
   ✓ Gradients are reasonably balanced

======================================================================
Summary
======================================================================

✅ All checks passed! PCGrad is working correctly.
```

---

### Method 3: Visual Inspection

Check the training loop logs for conflict status:

```bash
grep "Conflict:" logs/test.log
```

**Example output:**
```
└─ Conflict: cos(θ) = -0.234 ⚠️ CONFLICT
└─ Conflict: cos(θ) = 0.456 ✓ aligned
└─ Conflict: cos(θ) = -0.023 ~ neutral
```

**Interpretation:**
- `cos < -0.3`: Strong conflict (⚠️ CONFLICT)
- `cos > 0.3`: Strong alignment (✓ aligned)
- `-0.3 ≤ cos ≤ 0.3`: Neutral (~ neutral)

**PCGrad triggers when:** `cos < -0.1`

---

## What PCGrad Does

### Without PCGrad:
```
Physics gradient:  [1,  0, 0] (wants to move +X)
Render gradient:   [-1, 0, 0] (wants to move -X)

Combined: [0, 0, 0]  ❌ STUCK!
```

### With PCGrad:
```
Physics gradient:  [1,  0, 0] (wants to move +X)
Render gradient:   [-1, 0, 0] (wants to move -X)

PCGrad projects render gradient:
  Projection = [-1, 0, 0]  (remove conflicting component)
  Render_proj = [-1, 0, 0] - [-1, 0, 0] = [0, 0, 0]

Combined: [1, 0, 0]  ✅ Follows physics!
```

---

## Expected Behavior

### Early Episodes (ep 0-5):
```
[Warmup] Episode 2 < 5: Physics-only (skipping render grads)
```
- No render gradients → No PCGrad needed

### Mid Episodes (ep 5-20):
```
└─ Conflict: cos(θ) = -0.234 ⚠️ CONFLICT
🔥 [PCGrad] Conflict detected (cos=-0.234), applying gradient projection
├─ pcgrad_applied: True
├─ pcgrad_projection_scale: 0.123
```
- Conflicts common as render loss starts influencing
- PCGrad resolves conflicts

### Late Episodes (ep 20+):
```
└─ Conflict: cos(θ) = 0.456 ✓ aligned
├─ No PCGrad needed (gradients cooperate)
```
- Gradients become aligned as optimization converges
- PCGrad rarely needed

---

## Troubleshooting

### Issue 1: PCGrad Not Firing (But Conflicts Exist)

**Symptom:**
```
└─ Conflict: cos(θ) = -0.234 ⚠️ CONFLICT
[No PCGrad message]
```

**Possible Causes:**
1. PCGrad disabled in config:
   ```yaml
   optimization:
     use_pcgrad: false
   ```
   **Fix:** Remove this line (default is `True`)

2. Conflict threshold too strict:
   ```python
   # training_loop.py:805
   if use_pcgrad and cosine < -0.1:  # ← Check this threshold
   ```
   **Fix:** Lower threshold to -0.2 if needed

3. Legacy mode (not session mode):
   - PCGrad only works in legacy mode (run_e2e_episode)
   - Session mode has different gradient handling

### Issue 2: Too Many Conflicts

**Symptom:**
```
Conflicts detected: 45/50 (90%)
```

**Interpretation:**
- This is **okay** if PCGrad is resolving them
- High conflict rate = render and physics have different goals
- Should decrease as training progresses

**If persists late in training:**
- render_loss_weight may be too high
- Try reducing: `render_loss_weight: 50` (from 100)

### Issue 3: No Conflicts Ever

**Symptom:**
```
Conflicts detected: 0/50 (0%)
All cosine > 0.0
```

**Interpretation:**
- Gradients are aligned (good!)
- PCGrad not needed
- This is **expected** in later episodes

**If happens in early episodes:**
- Render gradients may be too weak
- Check w_depth and w_edge values
- Consider increasing render_loss_weight

### Issue 4: Episodes Still Failing

**Symptom:**
```
Episode success rate: 60%
PCGrad is working correctly
```

**Interpretation:**
- PCGrad fixes gradient conflicts
- But other issues may exist (learning rate, SV clamping, etc.)
- See: `docs/EPISODE_FAILURE_ANALYSIS.md`

---

## Advanced: Testing PCGrad Effectiveness

### Experiment 1: Compare With/Without PCGrad

**Config A: PCGrad Enabled (default)**
```yaml
# configs/sp_to_by/exp_pcgrad_enabled.yaml
optimization:
  # use_pcgrad: true (default, don't need to specify)
```

**Config B: PCGrad Disabled**
```yaml
# configs/sp_to_by/exp_pcgrad_disabled.yaml
optimization:
  use_pcgrad: false
```

**Run both:**
```bash
python run.py -c configs/sp_to_by/exp_pcgrad_enabled.yaml --png
python run.py -c configs/sp_to_by/exp_pcgrad_disabled.yaml --png
```

**Compare:**
- Episode success rate
- Loss curve smoothness
- Number of line search failures

**Expected:** PCGrad should improve stability when conflicts exist.

---

## Summary

### ✅ PCGrad is Working If:
1. Log shows "PCGrad applied" when conflicts detected
2. Conflict resolution visible (cosine changes after projection)
3. Episode success rate > 90%
4. No line search failures

### ⚠️ Check Configuration If:
1. Conflicts detected but PCGrad never applied
2. PCGrad firing too often (>80% of passes)
3. Episodes still failing despite PCGrad

### 🎯 Key Metrics:
- **Conflict rate (early):** 20-50% is normal
- **Conflict rate (late):** <10% expected
- **Projection scale:** 0.1-0.3 typical
- **Episode success:** >90% with proper config

---

## Files

- **Implementation:** `utils/gradient_utils.py:118-189`
- **Integration:** `utils/training_loop.py:801-828`
- **Analysis:** `docs/PCGRAD_ANALYSIS.md`
- **Verification:** `verify_pcgrad.py`

---

**Ready to test?**

```bash
# Run test config
python run.py -c configs/sp_to_by/exp_test_fixes.yaml --png 2>&1 | tee logs/test.log

# Verify PCGrad
python verify_pcgrad.py logs/test.log
```

Good luck! 🚀
