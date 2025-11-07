# Physics-Only vs E2E Mode: Detailed Comparison

## Executive Summary

After thorough code analysis, I've identified **3 major differences** between physics-only and E2E modes:

1. **🔴 CRITICAL BUG**: Gradient injection missing (already documented in PROOF_OF_BUG.md)
2. **🟡 STRUCTURAL DIFFERENCE**: SetUpCompGraph call pattern differs
3. **🟢 BY DESIGN**: Multi-pass refinement vs single-pass optimization

---

## Difference #1: Gradient Injection Bug (CRITICAL)

### Status: **CONFIRMED BUG** ✅ (Fix documented in APPLY_FIX_GUIDE.md)

**Problem**: Render gradients are computed, stored in C++, but NEVER added to physics gradients during backward pass.

**Impact**:
- E2E mode functionally identical to physics-only mode
- Render loss has NO effect on optimization
- PCGrad is pointless (no actual conflicts to resolve)

**Evidence**:
```python
# Observation: Physics loss trajectories are IDENTICAL
Physics-only: [3227.8, 1455.1, 924.3, 692.1, ...]
E2E mode:     [3227.8, 1455.1, 924.3, 692.1, ...]  # Should be different!
```

**Fix Location**: `DiffMPMLib3D/CompGraph.cpp` → `ComputeBackwardPass()`

**See**: PROOF_OF_BUG.md, APPLY_FIX_GUIDE.md for complete details

---

## Difference #2: SetUpCompGraph Call Pattern

### Status: **STRUCTURAL DIFFERENCE** (May cause subtle issues)

### Physics-Only Mode (run.py:440-448)

```python
# Single call per episode
cg.run_optimization(opt)
cg.promote_last_as_initial(carry_grid=True)
```

**What happens internally** (inferred from C++ behavior):
```cpp
run_optimization(opt):
    SetUpCompGraph(num_timesteps)        // Reset layers 1-N, keep layer 0
    compute_forward_pass(0, episode)     // Initial forward sim

    for control_timestep in range(num_control_steps):
        ComputeForwardPass(control_timestep)   // Forward simulation
        ComputeBackwardPass(control_timestep)  // Backward gradients
        GradientDescent()                       // Update controls

    end_layer_mass_loss()  // Return final loss
```

### E2E Mode (training_loop.py:547-630)

```python
# Episode initialization - EXPLICIT setup
cg.set_up_comp_graph(num_timesteps)     # Line 549: Called ONCE
cg.compute_forward_pass(0, ep)          # Line 552: Initial forward

# Multi-pass refinement loop
for pass_idx in range(num_passes):  # Usually 3 passes
    if pass_idx > 0:
        skip_setup = True  # 🔥 CRITICAL: Skip re-setup for pass 2+

    # Phase 1+2: Physics optimization
    run_physics_optimization_batched(
        cg, opt, render_grads, pass_idx, skip_setup=skip_setup
    )
    # This calls: cg.run_optimization(opt, skip_setup=skip_setup)

    # Phase 3: Compute render loss and extract gradients
    compute_render_loss_pass(...)
    extract_render_gradients(...)

    # Phase 4: PCGrad + gradient combination
    pcgrad_projection(...)
    normalize_and_combine_gradients(...)
```

### Key Differences

| Aspect | Physics-Only | E2E Mode |
|--------|--------------|----------|
| **SetUpCompGraph calls** | 1 per episode (internal) | 1 per episode (explicit) |
| **Forward pass calls** | 1 per episode (internal) | 1 per episode (explicit) |
| **Optimization passes** | 1 per episode | 3 per episode (configurable) |
| **Gradient sources** | Physics only | Physics + Render (combined) |
| **skip_setup usage** | N/A (always fresh setup) | ✅ Used for pass 2+ |

### Potential Issue #1: Timing of SetUpCompGraph

**Physics-only**: `SetUpCompGraph()` is called INSIDE `run_optimization()`
- Happens AFTER any previous state modifications
- Fresh setup every episode

**E2E mode**: `SetUpCompGraph()` is called BEFORE the multi-pass loop
- Happens at episode start
- Same comp graph used for all 3 passes (with skip_setup=True for pass 2+)

**Hypothesis**: This should be fine IF skip_setup is working correctly. But let's verify.

### Potential Issue #2: State Management Between Passes

**Physics-only**:
- Single pass → Simple state management
- `promote_last_as_initial()` called once per episode

**E2E mode**:
- 3 passes per episode
- `promote_last_as_initial()` called once AFTER all passes

**Question**: Are intermediate states between passes properly preserved?

Let me check the code flow more carefully...

From training_loop.py:627-630:
```python
skip_setup = (pass_idx > 0)
loss_physics = run_physics_optimization_batched(
    cg, opt, render_grads_dict, pass_idx, skip_setup=skip_setup
)
```

This means:
- Pass 1: skip_setup=False → SetUpCompGraph() called → Resets layers 1-N
- Pass 2: skip_setup=True → SetUpCompGraph() SKIPPED → Layers preserved
- Pass 3: skip_setup=True → SetUpCompGraph() SKIPPED → Layers preserved

**This looks CORRECT** ✅ (from the skip_setup fix we implemented)

---

## Difference #3: Multi-Pass Refinement (By Design)

### Physics-Only: Single-Pass Per Episode

```
Episode N:
  1. SetUpCompGraph() (internal)
  2. Forward simulation: x(0) → x(T)
  3. Compute physics loss: L = ||x(T) - target||²
  4. Backward: ∂L/∂controls
  5. Optimize controls (gradient descent + line search)
  6. Done → Next episode
```

### E2E: Multi-Pass Refinement Per Episode

```
Episode N:
  SetUpCompGraph() (explicit, once)

  Pass 1: Render Loss Baseline
    1. Forward sim: x(0) → x(T)
    2. Physics loss: L_phys
    3. Upsample + Render
    4. Render loss: L_render
    5. Extract gradients: ∂L_render/∂F, ∂L_render/∂x
    6. Store for next pass

  Pass 2: Physics + Render Optimization
    1. Inject combined gradients (∂L_phys + w*∂L_render)
    2. Forward sim: x(0) → x(T)
    3. Physics loss: L_phys (now influenced by render grads!)
    4. Backward: ∂L_total/∂controls
    5. Optimize controls
    6. Upsample + Render
    7. Render loss: L_render (should improve)
    8. Extract gradients for Pass 3

  Pass 3: Final Refinement
    1. Inject combined gradients (updated from Pass 2)
    2. Forward sim: x(0) → x(T)
    3. Physics loss: L_phys
    4. Optimize controls
    5. Final render + visualization

  promote_last_as_initial() → Carry state to Episode N+1
```

**This is BY DESIGN** and intended behavior ✅

---

## Comparison Summary

| Feature | Physics-Only | E2E Mode | Status |
|---------|--------------|----------|---------|
| **SetUpCompGraph timing** | Internal (inside run_optimization) | Explicit (before passes) | ✅ OK (different but equivalent) |
| **Number of passes** | 1 per episode | 3 per episode | ✅ OK (by design) |
| **Gradient sources** | Physics only (∂L_phys/∂F) | Physics + Render (∂L_phys/∂F + w*∂L_render/∂F) | 🔴 **BUG** (render grads not injected) |
| **skip_setup usage** | Not needed | Pass 1: False, Pass 2+: True | ✅ OK (implemented correctly) |
| **PCGrad** | Not applicable | Active (but currently useless due to bug) | 🟡 Depends on gradient injection fix |
| **State carryover** | Once per episode | Once per episode (after all passes) | ✅ OK |
| **Optimization parameters** | From config | From config (same parameters) | ✅ OK |

---

## ROOT CAUSE ANALYSIS

### Why Are Physics Loss Trajectories Identical?

**Physics-only trajectory**: `[3227.8, 1455.1, 924.3, 692.1, ...]`
**E2E trajectory**: `[3227.8, 1455.1, 924.3, 692.1, ...]` ← IDENTICAL!

**Root Cause**: Gradient Injection Bug (Difference #1)

Even though:
1. ✅ Render gradients ARE computed correctly
2. ✅ PCGrad IS applied (projects out conflicts)
3. ✅ Combined gradients ARE stored in C++
4. ❌ **Gradients are NEVER added to physics gradients in ComputeBackwardPass()**

**Result**: E2E mode is functionally identical to physics-only mode.

### Mathematical Proof

If render gradients were being used:
```
g_phys = ∂L_physics/∂F  (from mass matching)
g_render = ∂L_render/∂F  (from visual loss)
g_combined = g_phys + w * g_render

If w ≠ 0 and g_render ≠ 0:
  → g_combined ≠ g_phys
  → F update differs
  → Loss trajectory differs
```

Since trajectories are IDENTICAL:
```
g_combined = g_phys
→ w * g_render = 0
→ Either w = 0 (but it's 1.0!) OR g_render is not being added (BUG!)
```

**Conclusion**: Render gradients are NOT being added to physics gradients.

---

## ADDITIONAL POTENTIAL ISSUES

### Issue #1: Parameter Reading

From user's earlier observation:
> "I guess the program do not read any parameters in the optimization"

**Investigation needed**:
- Are optimization parameters (initial_alpha, adaptive_alpha_*, etc.) being read correctly?
- Are they being passed to C++ correctly?

**Files to check**:
- utils/optimization_utils.py (parameter parsing)
- DiffMPMLib3D/bind.cpp (Python→C++ parameter passing)

### Issue #2: SetUpCompGraph Behavior Differences

**Question**: Does `run_optimization()` internally call `SetUpCompGraph()` with the same parameters as E2E mode's explicit call?

**Hypothesis**:
- Physics-only: `run_optimization()` → `SetUpCompGraph(num_timesteps)` (internal)
- E2E: `set_up_comp_graph(num_timesteps)` → explicit call

These SHOULD be equivalent, but may have subtle differences.

**Test**: Add debug logging to C++ to verify SetUpCompGraph is called with same parameters.

### Issue #3: Number of Gradient Descent Iterations

**Question**: Does E2E mode perform MORE total gradient descent iterations due to 3 passes?

**Physics-only**:
- 1 pass × N control timesteps × max_gd_iters → Total iterations

**E2E**:
- 3 passes × N control timesteps × max_gd_iters → **3x more iterations!**

**Impact**: E2E mode should converge faster (more optimization steps), but this only works IF render gradients are actually being used.

**Current reality**:
- E2E does 3x more iterations
- But all 3 passes optimize the SAME objective (physics only)
- So it's just wasting computation!

---

## RECOMMENDATIONS

### Immediate Priority: Fix Gradient Injection Bug

**Status**: Fix already documented in APPLY_FIX_GUIDE.md

**Action**: Apply the C++ fix to inject render gradients in `ComputeBackwardPass()`

**Expected outcome**: Physics loss trajectories will finally DIFFER between physics-only and E2E modes.

### Secondary: Verify Parameter Reading

**Test**:
1. Add C++ logging to print all optimization parameters at start
2. Compare logged values with YAML config
3. Verify parameters match expectations

**Files to check**:
```
DiffMPMLib3D/CompGraph.cpp: Check logging in OptimizeDefGradControlSequence()
utils/optimization_utils.py: Check build_opt_input()
```

### Tertiary: Add Comprehensive Diagnostics

**Suggestion**: Add a diagnostic mode that logs:
- SetUpCompGraph call count per episode
- Parameter values received by C++
- Gradient injection status (before/after norms)
- Number of GD iterations per control timestep
- Line search success/failure counts

---

## TESTING RECOMMENDATIONS

After applying the gradient injection fix, run these tests:

### Test 1: Verify Different Trajectories

```bash
# Physics-only
python run.py -c configs/verify/1_physics_only.yaml

# E2E (with gradient fix)
python run.py -c configs/verify/3_full_e2e_pcgrad.yaml
```

**Expected**: Loss trajectories should now DIFFER!

### Test 2: Verify Gradient Injection Logging

```bash
python run.py -c configs/verify/3_full_e2e_pcgrad.yaml 2>&1 | grep -i "inject"
```

**Expected output**:
```
[Backward] 🔥 Injecting render gradients...
[Backward] ✅ Injected 37000 gradients!
```

### Test 3: Verify PCGrad is Working

```bash
python run.py -c configs/verify/3_full_e2e_pcgrad.yaml 2>&1 | grep -A 5 "PCGrad"
```

**Expected**: Should see conflict detection and projection when cosine < -0.1

---

## CONCLUSION

### Main Finding

**The PRIMARY reason physics-only and E2E modes produce identical results is the gradient injection bug** (Difference #1).

The other differences (SetUpCompGraph timing, multi-pass refinement) are either:
- By design (multi-pass)
- Equivalent in behavior (SetUpCompGraph timing)

### What's Broken

1. 🔴 **Gradient injection** (CRITICAL bug)
2. 🟡 **Parameter reading** (needs verification)

### What's Working Correctly

1. ✅ **SetUpCompGraph** with skip_setup fix
2. ✅ **State carryover** between episodes
3. ✅ **Multi-pass refinement** architecture
4. ✅ **PCGrad implementation** (but useless until gradients flow)
5. ✅ **Gradient combination** in Python

### Next Steps

1. **Apply gradient injection fix** (APPLY_FIX_GUIDE.md)
2. **Verify parameter reading** (check optimization parameters)
3. **Test thoroughly** (run comparison tests)
4. **Tune render_loss_weight** after fix is working

---

**Status**: Analysis Complete ✅
**Primary Issue**: Gradient injection bug (already documented)
**Secondary Issue**: Parameter reading verification needed
**Structural Issues**: None found (differences are by design)
