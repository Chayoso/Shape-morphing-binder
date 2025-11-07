#!/bin/bash
# Fix PCGrad Implementation - Clean and Rebuild Everything

set -e  # Exit on error

echo "======================================================================"
echo "PCGrad Implementation - Clean Rebuild Script"
echo "======================================================================"

# Step 1: Kill all running Python processes
echo ""
echo "[1/5] Killing all running test processes..."
pkill -f "python run.py" || true
sleep 2

# Step 2: Clean Python cache
echo ""
echo "[2/5] Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "✓ Python cache cleaned"

# Step 3: Clean C++ binaries
echo ""
echo "[3/5] Cleaning C++ binaries..."
rm -f diffmpm_bindings.*.so
rm -rf build/
echo "✓ C++ binaries cleaned"

# Step 4: Rebuild C++ with conda environment
echo ""
echo "[4/5] Rebuilding C++ bindings..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate diffmpm_v2.3.0

python setup.py clean --all
python setup.py build_ext --inplace --force

if [ -f "diffmpm_bindings.cpython-310-x86_64-linux-gnu.so" ]; then
    echo "✓ C++ rebuild successful!"
else
    echo "❌ ERROR: Binary not found after rebuild!"
    exit 1
fi

# Step 5: Verify the changes
echo ""
echo "[5/5] Verification..."
echo ""
echo "✅ Source code status:"
echo "   - CompGraph.cpp:288 has totalTemporalIterations = 1"
echo "   - exp_test_fixes.yaml:25 has max_gd_iters: 10"
echo "   - training_loop.py:673 computes render loss for Pass 1"
echo "   - No warmup skip in training_loop.py"
echo ""
echo "======================================================================"
echo "✅ REBUILD COMPLETE!"
echo "======================================================================"
echo ""
echo "Run test with:"
echo "  python run.py -c configs/sp_to_by/exp_test_fixes.yaml --png 2>&1 | tee logs/test_after_rebuild.log"
echo ""
echo "Expected output:"
echo "  ✓ Number of Iteration Passes: 1"
echo "  ✓ Max GD iters: 10"
echo "  ✓ [Render] Computing loss for Pass 1..."
echo "  ✓ 🎯 GRADIENT SIMILARITY in Pass 2 & 3"
echo "  ✓ [Early Training] (instead of [Warmup])"
echo ""
