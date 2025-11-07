# ✅ PCGrad Final Implementation - Correct Pass Structure

## Flow Per Episode

```
══════════════════════════════════════════════════════════════════
EPISODE 0 START
══════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────
Pass 1/3 (Physics-only)
──────────────────────────────────────────────────────────────────
1. Physics Optimization (NO render grads injected)
   └─ 9 timesteps × 10 GD iterations = 90 iterations

2. Compute Render Loss
   └─ Forward: MPM → Upsample → Render
   └─ Loss: Compare with target
   └─ Backward: Extract ∂L/∂F, ∂L/∂x (render gradients)

3. Store render gradients for Pass 2

──────────────────────────────────────────────────────────────────
Pass 2/3 (Physics + Render)
──────────────────────────────────────────────────────────────────
1. Inject Pass 1 render gradients into C++

2. Physics Optimization (WITH Pass 1 render grads)
   └─ 9 timesteps × 10 GD iterations = 90 iterations
   └─ Gradients: ∂L_total = ∂L_physics + ∂L_render(Pass1)

3. Compute Render Loss
   └─ Forward: MPM → Upsample → Render
   └─ Loss: Compare with target
   └─ Backward: Extract ∂L/∂F, ∂L/∂x (NEW render gradients)

4. **PCGrad Similarity Calculation**
   ├─ Get physics gradients from C++ backend
   ├─ Calculate cosine similarity(phys_grads, render_grads)
   ├─ If conflict (cos < -0.1): Apply PCGrad projection
   └─ Combine: phys_grads + projected_render_grads

5. Store COMBINED gradients for Pass 3

──────────────────────────────────────────────────────────────────
Pass 3/3 (Physics + Render)
──────────────────────────────────────────────────────────────────
1. Inject Pass 2 COMBINED gradients into C++

2. Physics Optimization (WITH Pass 2 combined grads)
   └─ 9 timesteps × 10 GD iterations = 90 iterations

3. Compute Render Loss
   └─ Forward: MPM → Upsample → Render
   └─ Loss: Compare with target
   └─ Backward: Extract ∂L/∂F, ∂L/∂x (NEW render gradients)

4. **PCGrad Similarity Calculation**
   ├─ Get physics gradients from C++ backend
   ├─ Calculate cosine similarity
   ├─ If conflict: Apply PCGrad projection
   └─ Combine gradients

5. Save PNG outputs (final pass)

══════════════════════════════════════════════════════════════════
EPISODE 1 START (repeat)
══════════════════════════════════════════════════════════════════
```

## Key Implementation Details

### C++ Changes:
- `totalTemporalIterations = 1` (removed internal 3-pass loop)
- `max_gd_iters = 10` (10 iterations per timestep)
- Added `GetLastLayerPhysGradients()` to expose physics gradients

### Python Changes:
- Removed duplicate `run_physics_optimization()` call
- Render loss computed after EVERY pass (including Pass 1)
- PCGrad similarity calculation in Pass 2 & 3
- No warmup skip - similarity from Episode 0

### Total Iterations Per Episode:
- Pass 1: 90 iterations (physics-only)
- Pass 2: 90 iterations (physics + Pass 1 render)
- Pass 3: 90 iterations (physics + Pass 2 combined)
- **Total: 270 iterations per episode**

## Files Modified

1. `DiffMPMLib3D/CompGraph.cpp:288` - Set `totalTemporalIterations = 1`
2. `DiffMPMLib3D/PointCloud.cpp` - Added gradient getters
3. `DiffMPMLib3D/CompGraph.cpp` - Added `GetLastLayerPhysGradients()`
4. `bind/bind.cpp` - Exposed `get_last_layer_phys_gradients()` to Python
5. `utils/training_loop.py:625-628` - Removed duplicate optimization call
6. `utils/training_loop.py:667-673` - Removed incorrect Pass 1 skip
7. `utils/training_loop.py:746-747` - Fixed numpy/device compatibility
8. `utils/training_loop.py:781-785` - Removed warmup skip
9. `configs/sp_to_by/exp_test_fixes.yaml:25` - Set `max_gd_iters: 10`

## Status

✅ **COMPLETE** - All changes implemented and code is ready for testing!

To test, run with conda environment:
```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate diffmpm_v2.3.0
python run.py -c configs/sp_to_by/exp_test_fixes.yaml --png
```
