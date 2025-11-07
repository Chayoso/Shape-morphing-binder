# Sphere → Bunny Convergence: Experiments & Fixes

## 🎯 Objective
Fix sphere→bunny morphing divergence issues through systematic parameter exploration and pipeline fixes.

---

## ✅ Critical Pipeline Fixes Applied

### 1. Fixed Double render_loss_weight Application
- **Issue:** 200x weight became 40,000x due to double application
- **Fix:** Removed from loss.py, now only applied during gradient combination
- **File:** `loss.py:298-303`

### 2. Fixed Gradient Magnitude Mismatch
- **Issue:** Hardcoded normalization (target=0.05) didn't adapt
- **Fix:** Normalize to unit vectors, scale dynamically based on physics
- **File:** `utils/training_loop.py:266-281`

### 3. Applied render_loss_weight Correctly
- **Issue:** Config render_loss_weight had no effect
- **Fix:** Now properly incorporated into w_render during combination
- **File:** `utils/training_loop.py:764-799`

### 4. Enabled PCGrad by Default
- **Issue:** Gradient conflicts ignored → stuck optimization
- **Fix:** PCGrad now enabled by default (projects conflicting components)
- **File:** `utils/training_loop.py:801-803`

**📄 Details:** See `docs/PIPELINE_FIXES_APPLIED.md`

---

## 🔬 Experiment Suite

Created **14 systematic configs** in `configs/sp_to_by/`:

| Config | Group | Tests | Episodes |
|--------|-------|-------|----------|
| `exp_test_fixes.yaml` | ✅ **TEST** | Verify pipeline fixes | 10 |
| `exp_bunny_render_20.yaml` | Render Loss | Very low (physics-dominant) | 25 |
| `exp_bunny_render_50.yaml` | Render Loss | Low (physics-strong) | 25 |
| `exp_bunny_render_100.yaml` | Render Loss | Medium (balanced) | 25 |
| `exp_bunny_render_150.yaml` | Render Loss | High (render-strong) | 25 |
| `exp_lr_gentle.yaml` | Learning Rate | Gradual decay (80%→60%→40%) | 10 |
| `exp_lr_constant.yaml` | Learning Rate | No decay (constant 0.008) | 10 |
| `exp_sv_relaxed.yaml` | SV Clamping | Conservative (30% compression) | 10 |
| `exp_sv_tight.yaml` | SV Clamping | Very tight (15% compression) | 20 |
| `exp_ls_patient.yaml` | Line Search | 2x iterations (20 vs 10) | 10 |
| `exp_progressive_annealing.yaml` | Loss Schedule | Coarse-to-fine (depth→edge→polish) | 10 |
| `exp_particles_high.yaml` | Particles | 2.37x more (64 vs 27 per cell) | 10 |
| `exp_best_practices.yaml` | Combined | All stabilization techniques | 10 |
| `exp_aggressive_stable.yaml` | Combined | High render + safeguards | 10 |

**📄 Details:** See `configs/sp_to_by/EXPERIMENT_GUIDE.md`

---

## 🚀 Quick Start

### 1. Test Pipeline Fixes (RECOMMENDED FIRST)
```bash
python run.py -c configs/sp_to_by/exp_test_fixes.yaml --png
```

**Expected results:**
- ✅ All 10 episodes complete successfully (no failures)
- ✅ Smooth loss curves (no jumps)
- ✅ Gradient logs show balanced magnitudes
- ✅ Early signs of ear formation

### 2. Run Best Practices Config
```bash
python run.py -c configs/sp_to_by/exp_best_practices.yaml --png
```

### 3. Check Results
```bash
# Episode success rate
python check_episodes.py output/experiments/test_fixes/

# Loss curves
python view_losses.py output/experiments/test_fixes/ --plot

# Visual inspection
eog output/experiments/test_fixes/ep009/render.png
```

---

## 📊 Expected Improvements

### Before Fixes:
```
Episode Success: 64% (18/50 failed)
Failure Pattern: 12% → 37% → 55% (increasing)
Gradient Ratio: render/physics = 10,000x
Line Search: Frequent failures
Features: Smooth blob (no ears)
```

### After Fixes:
```
Episode Success: >90% expected
Failure Pattern: Stable across episodes
Gradient Ratio: render/physics = ~1x (balanced!)
Line Search: Stable convergence
Features: Ears form by episode 10-15
```

---

## 📁 Project Structure

```
Shape-morphing-binder/
├── configs/
│   ├── Chayo/                      # Original configs
│   └── sp_to_by/                   # 🔥 NEW: Sphere→bunny experiments
│       ├── exp_test_fixes.yaml     # ✅ Test pipeline fixes first!
│       ├── exp_best_practices.yaml # Combined fixes
│       ├── exp_*.yaml              # 12 other experiments
│       ├── EXPERIMENT_GUIDE.md     # Detailed experiment descriptions
│       └── README.md               # Quick reference
│
├── docs/                           # 🔥 NEW: Documentation
│   ├── PIPELINE_FIXES_APPLIED.md   # ✅ What was fixed
│   ├── PIPELINE_ANALYSIS.md        # Mathematical issues found
│   ├── EPISODE_FAILURE_ANALYSIS.md # Failure pattern analysis
│   ├── LOCAL_MINIMA_SOLUTIONS.md   # Feature formation solutions
│   ├── COMPRESSION_PROBLEM.md      # SV clamping issues
│   └── TROUBLESHOOTING_ERRORS.md   # Common errors
│
├── loss.py                         # ✅ FIXED: Removed double weight
├── utils/
│   ├── training_loop.py            # ✅ FIXED: 3 critical issues
│   └── gradient_utils.py           # Gradient combination logic
│
└── README_EXPERIMENTS.md           # 📄 This file
```

---

## 🔍 Diagnostic Tools

### Check Episode Success
```bash
python check_episodes.py output/experiments/test_fixes/
```

### View Loss Curves
```bash
python view_losses.py output/experiments/test_fixes/ --plot
```

### Monitor Training (Real-time)
```bash
python run.py -c config.yaml --png 2>&1 | grep -E "Episode|success|CONFLICT|PCGrad"
```

---

## 📝 What to Look For in Logs

### ✅ Good Signs (Pipeline Working):
```
[Weight Calculation]
│  ├─ render_loss_weight (config): 100.0
│  ├─ w_render_base (schedule): 0.300
│  ├─ w_render (final): 0.300
│  └─ w_physics: 1.000

🔥 [Normalized Gradient Combination] Pass 1
├─ BEFORE normalization:
│  ├─ ||g_render|| = 1.000e+00  ✅ Unit norm
│  ├─ ||g_phys||   = 8.234e-03
│  ├─ Ratio (render/phys) = 121.5  ✅ Not 10000x!

├─ AFTER combination:
│  ├─ ||g_combined|| = 8.145e-03  ✅ Physics magnitude!
│  └─ Ratio (combined/phys) = 0.989 ✅

[Episode 9] Session Results:
  Success: ✅
```

### ⚠️ Warning Signs (Still Issues):
```
Line search failed  ❌
result.success = False  ❌
Ratio (render/phys) = 10000  ❌ (gradients still mismatched)
Episode failure rate > 20%  ❌
```

---

## 🐛 Troubleshooting

### If exp_test_fixes.yaml still fails:

1. **Check gradient logs** - Are they balanced (~1x ratio)?
   - If not: PCGrad may not be working, check line 803

2. **Check episode success rate**
   ```bash
   python check_episodes.py output/experiments/test_fixes/
   ```
   - If < 80%: Try exp_sv_tight.yaml or exp_ls_patient.yaml

3. **Check for line search failures**
   ```bash
   grep "line search" logs/*.log
   ```
   - If frequent: Increase max_ls_iters to 25

4. **Visual inspection**
   - If ears not forming: Try exp_progressive_annealing.yaml
   - If smooth blob persists: Increase w_edge to 3.0+

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| `docs/PIPELINE_FIXES_APPLIED.md` | **START HERE** - What was fixed & why |
| `docs/PIPELINE_ANALYSIS.md` | Deep dive into mathematical issues |
| `docs/EPISODE_FAILURE_ANALYSIS.md` | Why episodes fail (cumulative instability) |
| `docs/LOCAL_MINIMA_SOLUTIONS.md` | Why ears don't form (local minima) |
| `configs/sp_to_by/EXPERIMENT_GUIDE.md` | Detailed experiment descriptions |
| `configs/sp_to_by/README.md` | Quick reference for experiments |

---

## ⏱️ Time Estimates

- **exp_test_fixes.yaml:** ~1 hour (10 episodes)
- **exp_best_practices.yaml:** ~1 hour (10 episodes)
- **exp_progressive_annealing.yaml:** ~1 hour (10 episodes)
- **exp_particles_high.yaml:** ~2 hours (10 episodes, 2x slower)
- **All experiments:** ~15 hours total

**Tip:** Run 3-4 in parallel on different GPUs if available.

---

## 🎯 Success Criteria

### Primary Goal: Stability
- ✅ Episode success rate > 90%
- ✅ No increasing failure pattern
- ✅ Smooth loss curves

### Secondary Goal: Quality
- ✅ Bunny ears visible by episode 10
- ✅ loss_edge decreases over time
- ✅ Final render shows clear features

---

## 🚦 Next Steps

1. **✅ Run exp_test_fixes.yaml** to verify pipeline fixes
2. **✅ Check results** (success rate, gradient logs, visuals)
3. **If successful:** Run exp_best_practices.yaml with more episodes (25-50)
4. **If still failing:** Try individual experiments to isolate issue
5. **Once stable:** Create production config with best parameters

---

## 📞 Support

- **Issues:** Check `docs/TROUBLESHOOTING_ERRORS.md`
- **Failures:** Check `docs/EPISODE_FAILURE_ANALYSIS.md`
- **Local minima:** Check `docs/LOCAL_MINIMA_SOLUTIONS.md`
- **Pipeline bugs:** Check `docs/PIPELINE_ANALYSIS.md`

---

**Status:** ✅ **Pipeline fixes applied, ready to test!**

**Recommended first command:**
```bash
python run.py -c configs/sp_to_by/exp_test_fixes.yaml --png
```

Good luck! 🚀
