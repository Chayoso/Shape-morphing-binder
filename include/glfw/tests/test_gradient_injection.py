#!/usr/bin/env python3
"""
Test if render gradients actually affect physics optimization.

This script runs a simple test to verify gradient injection is working.
"""

import sys
import numpy as np
import diffmpm_bindings

# Create minimal test setup
print("="*70)
print("GRADIENT INJECTION TEST")
print("="*70)

# Check if set_render_gradients method exists
cg_methods = [m for m in dir(diffmpm_bindings.CompGraph) if not m.startswith('_')]
print(f"\nCompGraph methods: {cg_methods}\n")

has_set_render_grads = 'set_render_gradients' in cg_methods
has_run_e2e_batched = 'run_e2e_pass_batched' in cg_methods

print(f"✓ set_render_gradients exists: {has_set_render_grads}")
print(f"✓ run_e2e_pass_batched exists: {has_run_e2e_batched}")

if not has_set_render_grads and not has_run_e2e_batched:
    print(f"\n❌ ERROR: Neither gradient injection method found!")
    print(f"   This means render gradients CANNOT be injected!")
    sys.exit(1)

print(f"\n✅ Gradient injection interface exists")
print(f"\nAvailable methods for gradient injection:")
if has_set_render_grads:
    print(f"  - set_render_gradients (legacy)")
if has_run_e2e_batched:
    print(f"  - run_e2e_pass_batched (batched, preferred)")

# Check run_e2e_pass_batched signature
if has_run_e2e_batched:
    import inspect
    try:
        sig = inspect.signature(diffmpm_bindings.CompGraph.run_e2e_pass_batched)
        print(f"\nrun_e2e_pass_batched signature:")
        print(f"  {sig}")
    except Exception as e:
        print(f"\n  (Could not get signature: {e})")

print("\n" + "="*70)
print("NEXT STEP: Check if C++ actually USES the gradients")
print("="*70)
print("""
To verify gradients are being used:

1. Run TWO experiments:
   A) Physics-only (no render loss)
   B) E2E with render loss

2. Compare physics loss trajectories:
   - If IDENTICAL: Gradients are NOT being used ❌
   - If DIFFERENT: Gradients ARE working ✅

3. Check training logs for these messages:
   - "Batched E2E mode"
   - "Render grads: ||∂L/∂F||=..."
   - "PCGrad projection"
   - "Gradient Combination Summary"
""")
