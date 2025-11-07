#!/usr/bin/env python3
"""
Verify the gradient injection bug.

Run this with: conda activate diffmpm_v2.3.0 && python verify_gradient_bug.py
"""

import sys
import os

# Add project to path
sys.path.insert(0, '/home/chayo/Desktop/Shape-morphing-binder')

print("="*70)
print("GRADIENT INJECTION BUG VERIFICATION")
print("="*70)

# Step 1: Check if diffmpm_bindings can be imported
print("\n[1/3] Checking diffmpm_bindings...")
try:
    import diffmpm_bindings
    print("✅ diffmpm_bindings imported successfully")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

# Step 2: Check CompGraph methods
print("\n[2/3] Checking CompGraph gradient methods...")
cg_methods = [m for m in dir(diffmpm_bindings.CompGraph) if not m.startswith('_')]

methods_to_check = {
    'set_render_gradients': 'Set render gradients (legacy)',
    'run_e2e_pass_batched': 'Run E2E pass with gradients (batched)',
    'has_render_gradients': 'Check if render gradients are stored',
    'GetLastLayerPhysGradients': 'Get physics gradients for PCGrad',
}

print("\nGradient-related methods:")
for method, desc in methods_to_check.items():
    exists = method in cg_methods
    status = "✅" if exists else "❌"
    print(f"  {status} {method:30s} - {desc}")

# Step 3: The smoking gun test
print("\n[3/3] The Smoking Gun Test:")
print("-" * 70)
print("""
To prove gradients aren't being used, compare these two runs:

  A) Physics-only:  configs/verify/1_physics_only.yaml
  B) E2E + PCGrad:  configs/verify/3_full_e2e_pcgrad.yaml

If physics loss trajectories are IDENTICAL:
  → ❌ Render gradients are NOT being injected into physics updates
  → Bug confirmed!

If physics loss trajectories are DIFFERENT:
  → ✅ Gradients ARE working
  → No bug (your observation was mistaken)

Recommended test:
  1. Run both configs for 5 episodes
  2. Compare loss_physics values:
     - If same → BUG EXISTS
     - If different → NO BUG
""")

# Step 4: Check the actual bug location
print("\n" + "="*70)
print("LIKELY BUG LOCATION")
print("="*70)

print("""
The bug is most likely in DiffMPMLib3D/CompGraph.cpp:

  Function: ComputeBackwardPass()
  Problem:  Render gradients are stored but NEVER added to physics gradients

Expected code (MISSING):
  ```cpp
  if (has_render_grads_) {
      // Add render grads to final layer physics grads
      for (size_t i = 0; i < final_points.size(); i++) {
          points[i].dLdF += stored_render_grad_F_[i];  // ← MISSING!
          points[i].dLdx += stored_render_grad_x_[i];  // ← MISSING!
      }
  }
  ```

To fix:
  1. Find ComputeBackwardPass() in DiffMPMLib3D/CompGraph.cpp
  2. Add render gradient injection at final timestep
  3. Rebuild C++ bindings: cmake --build build
  4. Test: physics loss should change when render loss is enabled!
""")

print("="*70)
print("VERIFICATION COMPLETE")
print("="*70)
print("""
Next steps:
  1. Run the smoking gun test (compare physics-only vs E2E)
  2. If bug confirmed, read GRADIENT_INJECTION_BUG.md for fix
  3. Apply the C++ fix and rebuild
  4. Retest to verify gradients now affect physics loss
""")
