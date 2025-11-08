#!/usr/bin/env python3
"""
Simple test for gradient normalization (no renderer dependencies).
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_normalize_and_combine():
    """Test the core gradient normalization function."""
    print("="*70)
    print("Testing normalize_and_combine_gradients()")
    print("="*70)

    # Import directly to avoid renderer dependencies
    from utils.gradient_utils import normalize_and_combine_gradients

    # Create test gradients with HUGE magnitude mismatch
    N = 100
    dLdF_physics = np.random.randn(N, 3, 3).astype(np.float32) * 0.01  # Small
    dLdx_physics = np.random.randn(N, 3).astype(np.float32) * 0.01

    dLdF_render = np.random.randn(N, 3, 3).astype(np.float32) * 100.0  # HUGE!
    dLdx_render = np.random.randn(N, 3).astype(np.float32) * 100.0

    # Compute norms before
    g_phys = np.sqrt(np.linalg.norm(dLdF_physics)**2 + np.linalg.norm(dLdx_physics)**2)
    g_render = np.sqrt(np.linalg.norm(dLdF_render)**2 + np.linalg.norm(dLdx_render)**2)

    print(f"\nTest Setup:")
    print(f"  Physics gradient norm:  {g_phys:.6e}")
    print(f"  Render gradient norm:   {g_render:.6e}")
    print(f"  Ratio (render/physics): {g_render/g_phys:.6e}")

    # Test with 'physics' strategy
    print(f"\n{'-'*70}")
    print(f"Testing strategy: physics (conservative)")
    print(f"{'-'*70}")

    dLdF_combined, dLdx_combined, info = normalize_and_combine_gradients(
        dLdF_physics=dLdF_physics,
        dLdx_physics=dLdx_physics,
        dLdF_render=dLdF_render,
        dLdx_render=dLdx_render,
        w_physics=0.7,
        w_render=0.3,
        magnitude_strategy='physics'
    )

    print(f"  Before: ratio = {info['ratio_before']:.6e}")
    print(f"  After:  ratio = {info['ratio_after']:.4f}")
    print(f"  Combined norm: {info['g_combined_norm']:.6e}")

    # Verify results
    if abs(info['ratio_after'] - 1.0) < 0.1:
        print(f"  PASS: Ratio normalized to ~1.0")
        return True
    else:
        print(f"  FAIL: Ratio should be ~1.0")
        return False

def test_edge_cases():
    """Test edge cases."""
    print("\n" + "="*70)
    print("Testing Edge Cases")
    print("="*70)

    from utils.gradient_utils import normalize_and_combine_gradients

    N = 10
    all_passed = True

    # Test 1: Zero physics gradients
    print("\nTest 1: Zero physics gradients")
    try:
        dLdF_phys = np.zeros((N, 3, 3), dtype=np.float32)
        dLdx_phys = np.zeros((N, 3), dtype=np.float32)
        dLdF_render = np.random.randn(N, 3, 3).astype(np.float32)
        dLdx_render = np.random.randn(N, 3).astype(np.float32)

        dLdF_comb, dLdx_comb, info = normalize_and_combine_gradients(
            dLdF_phys, dLdx_phys, dLdF_render, dLdx_render
        )
        print(f"  PASS: Handled gracefully")
    except Exception as e:
        print(f"  FAIL: {e}")
        all_passed = False

    # Test 2: Zero render gradients
    print("\nTest 2: Zero render gradients")
    try:
        dLdF_phys = np.random.randn(N, 3, 3).astype(np.float32)
        dLdx_phys = np.random.randn(N, 3).astype(np.float32)
        dLdF_render = np.zeros((N, 3, 3), dtype=np.float32)
        dLdx_render = np.zeros((N, 3), dtype=np.float32)

        dLdF_comb, dLdx_comb, info = normalize_and_combine_gradients(
            dLdF_phys, dLdx_phys, dLdF_render, dLdx_render
        )
        print(f"  PASS: Handled gracefully")
    except Exception as e:
        print(f"  FAIL: {e}")
        all_passed = False

    return all_passed

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GRADIENT NORMALIZATION FIX - SIMPLE TEST")
    print("="*70 + "\n")

    try:
        test1_passed = test_normalize_and_combine()
        test2_passed = test_edge_cases()

        print("\n" + "="*70)
        if test1_passed and test2_passed:
            print("ALL TESTS PASSED!")
            print("="*70)
            print("\nThe gradient normalization fix is working correctly.")
            print("\nNext steps:")
            print("  1. Run training: python run.py configs/Chayo/sphere_to_bunny.yaml")
            print("  2. Check logs for 'Line search failed' (should be NONE)")
            print("  3. Verify 'Ratio (combined/phys) ~ 1.0' in output")
        else:
            print("SOME TESTS FAILED")
            print("Please review the errors above.")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
