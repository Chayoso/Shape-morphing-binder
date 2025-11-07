#!/bin/bash
# Diagnostic script for gradient injection issue

echo "======================================================================"
echo "GRADIENT INJECTION DIAGNOSTIC"
echo "======================================================================"

cd /home/chayo/Desktop/Shape-morphing-binder

# Activate conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate diffmpm_v2.3.0

echo ""
echo "Step 1: Check if gradient injection methods exist in C++ bindings"
echo "----------------------------------------------------------------------"

python3 << 'EOF'
import sys
sys.path.insert(0, '/home/chayo/Desktop/Shape-morphing-binder')

try:
    import diffmpm_bindings

    cg_methods = [m for m in dir(diffmpm_bindings.CompGraph) if not m.startswith('_')]

    print("Available CompGraph methods:")
    for m in sorted(cg_methods):
        print(f"  ✓ {m}")

    print("\n" + "="*70)
    print("Gradient Injection Interface Check:")
    print("="*70)

    has_set_render = 'set_render_gradients' in cg_methods
    has_run_batched = 'run_e2e_pass_batched' in cg_methods
    has_has_render = 'has_render_gradients' in cg_methods

    print(f"  set_render_gradients:  {'✅ YES' if has_set_render else '❌ NO'}")
    print(f"  run_e2e_pass_batched:  {'✅ YES' if has_run_batched else '❌ NO'}")
    print(f"  has_render_gradients:  {'✅ YES' if has_has_render else '❌ NO'}")

    print("\n" + "="*70)
    if has_run_batched or has_set_render:
        print("✅ Gradient injection interface EXISTS")
        print("   Problem is likely in HOW gradients are being used in C++")
    else:
        print("❌ Gradient injection interface MISSING")
        print("   C++ bindings need to be rebuilt with gradient support!")

except ImportError as e:
    print(f"❌ Error importing diffmpm_bindings: {e}")
    sys.exit(1)

EOF

echo ""
echo "Step 2: Check C++ source code for gradient usage"
echo "----------------------------------------------------------------------"

# Check if gradient storage variables exist in CompGraph.h
if grep -q "stored_render_grad_F_\|has_render_grads_" DiffMPMLib3D/CompGraph.h 2>/dev/null; then
    echo "✅ Gradient storage variables found in CompGraph.h"
    grep -n "stored_render_grad\|has_render_grads_" DiffMPMLib3D/CompGraph.h | head -5
else
    echo "⚠️  Gradient storage variables NOT found in CompGraph.h"
fi

echo ""
echo "Step 3: Check if gradients are actually USED in backward pass"
echo "----------------------------------------------------------------------"

# Look for where render gradients are added to physics gradients
if grep -q "stored_render_grad" DiffMPMLib3D/CompGraph.cpp 2>/dev/null; then
    echo "✅ Render gradients referenced in CompGraph.cpp"
    echo ""
    echo "Key locations:"
    grep -n "stored_render_grad" DiffMPMLib3D/CompGraph.cpp | head -10
else
    echo "❌ Render gradients NOT referenced in CompGraph.cpp"
    echo "   This means gradients are stored but NEVER USED!"
fi

echo ""
echo "======================================================================"
echo "DIAGNOSTIC COMPLETE"
echo "======================================================================"
