#!/bin/bash
# Force complete rebuild - NO CACHING

set -e

echo "========================================================================"
echo "FORCE REBUILD - Cleaning Everything"
echo "========================================================================"

# Activate conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate diffmpm_v2.3.0

echo ""
echo "[1/7] Killing running processes..."
pkill -f "python run.py" || true
sleep 3

echo ""
echo "[2/7] Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo ""
echo "[3/7] Removing old binaries..."
rm -f diffmpm_bindings.*.so
rm -rf build/ dist/ *.egg-info

echo ""
echo "[4/7] Cleaning setuptools cache..."
python setup.py clean --all 2>/dev/null || true

echo ""
echo "[5/7] Removing CMake cache..."
find . -name "CMakeCache.txt" -delete 2>/dev/null || true
find . -name "CMakeFiles" -type d -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "[6/7] Force rebuild with --force flag..."
python setup.py build_ext --inplace --force

echo ""
echo "[7/7] Verification..."
if [ -f "diffmpm_bindings.cpython-310-x86_64-linux-gnu.so" ]; then
    echo "✅ Binary created:"
    ls -lh diffmpm_bindings.cpython-310-x86_64-linux-gnu.so
    echo ""
    echo "Timestamp:"
    stat -c "Modified: %y" diffmpm_bindings.cpython-310-x86_64-linux-gnu.so
else
    echo "❌ ERROR: Binary not found!"
    exit 1
fi

echo ""
echo "========================================================================"
echo "✅ REBUILD COMPLETE!"
echo "========================================================================"
echo ""
echo "Now test with:"
echo "  python run.py -c configs/sp_to_by/exp_test_fixes.yaml --png 2>&1 | tee logs/test_VERIFIED.log"
echo ""
echo "Look for:"
echo "  ✅ Number of Iteration Passes: 1"
echo "  ✅ Max GD iters: 10"
echo ""
