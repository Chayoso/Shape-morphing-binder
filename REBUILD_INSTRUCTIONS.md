# 🔧 PCGrad Implementation - Rebuild Required

## ❌ Current Issues (from logs)

The test runs show the OLD code is still being used:

1. **C++ Issue**: `Number of Iteration Passes: 3` (should be 1)
2. **C++ Issue**: `Max GD iters: 1` (should be 10)
3. **Python Issue**: `[Warmup] Episode 0 < 5: Physics-only (skipping render grads)`

## ✅ Source Code Status

The source code **IS correct**:
- `DiffMPMLib3D/CompGraph.cpp:288` has `totalTemporalIterations = 1` ✅
- `configs/sp_to_by/exp_test_fixes.yaml:25` has `max_gd_iters: 10` ✅
- `utils/training_loop.py:673` computes render loss for Pass 1 ✅

## 🚨 Problem

The compiled binary `diffmpm_bindings.cpython-310-x86_64-linux-gnu.so` is OLD and needs to be rebuilt.

Also, there's still a warmup skip in the Python code at line ~781-785.

## 📋 Steps to Fix

### Step 1: Clean old binaries
```bash
rm -f diffmpm_bindings.*.so
rm -rf build/
```

### Step 2: Rebuild C++ with clean cache
```bash
# Activate conda environment first
source ~/anaconda3/etc/profile.d/conda.sh
conda activate diffmpm_v2.3.0

# Clean and rebuild
python setup.py clean --all
python setup.py build_ext --inplace --force
```

### Step 3: Fix Python warmup skip

Edit `utils/training_loop.py` around line 781-790:

**REMOVE this block:**
```python
if ep < 5:
    # Warmup: Physics-only (skip render grads)
    print(f"\n├─ [Warmup] Episode {ep} < 5: Physics-only (skipping render grads)")
    accumulated_render_grads = None
    continue
```

**REPLACE with:**
```python
if ep < 5:
    # Early episodes: Lower render weight but STILL calculate similarity
    w_render_base = 0.05  # Low but non-zero
    print(f"\n├─ [Early Training] Episode {ep} < 5: Low render weight ({w_render_base}), similarity calculation active")
```

### Step 4: Verify the rebuild worked
```bash
python run.py -c configs/sp_to_by/exp_test_fixes.yaml --png 2>&1 | tee logs/test_final_rebuild.log
```

Look for these in the log:
- ✅ `Number of Iteration Passes: 1`
- ✅ `Max GD iters: 10`
- ✅ `[Render] Computing loss for Pass 1...`
- ✅ `🎯 GRADIENT SIMILARITY:` in Pass 2 & 3
- ❌ NO `[Warmup]` skip messages

## 📊 Expected Output Structure

```
Episode 0
├─ Pass 1 (90 iters)
│  ├─ Physics optimization (9 timesteps × 10 iters)
│  └─ Compute render loss → Extract gradients
│
├─ Pass 2 (90 iters)
│  ├─ Inject Pass 1 render grads
│  ├─ Physics optimization (9 timesteps × 10 iters)
│  ├─ Compute render loss
│  └─ PCGrad similarity calculation → Combine gradients
│
└─ Pass 3 (90 iters)
   ├─ Inject Pass 2 combined grads
   ├─ Physics optimization (9 timesteps × 10 iters)
   ├─ Compute render loss
   └─ PCGrad similarity calculation → Combine gradients
```

**Total: 270 iterations per episode (90 × 3 passes)**
