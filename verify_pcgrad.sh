#!/bin/bash
# Verify PCGrad Implementation is Working

set -e

echo "========================================================================"
echo "PCGrad Implementation Verification Test"
echo "========================================================================"

# Activate conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate diffmpm_v2.3.0

# Run the test
echo ""
echo "Running test..."
echo ""
python run.py -c configs/sp_to_by/exp_test_fixes.yaml --png 2>&1 | tee logs/VERIFY_PCGRAD.log

echo ""
echo "========================================================================"
echo "Test complete! Analyzing log..."
echo "========================================================================"
echo ""

# Check for success markers
echo "Checking for correct values:"
echo ""

if grep -q "Number of Iteration Passes: 1" logs/VERIFY_PCGRAD.log; then
    echo "✅ C++ Fix: Number of Iteration Passes = 1"
else
    echo "❌ C++ Issue: Still showing 'Number of Iteration Passes: 3'"
fi

if grep -q "Max GD iters: 10" logs/VERIFY_PCGRAD.log; then
    echo "✅ Config: Max GD iters = 10"
else
    echo "❌ Config Issue: Max GD iters != 10"
fi

if grep -q "\[Render\] Computing loss for Pass 1" logs/VERIFY_PCGRAD.log; then
    echo "✅ Pass 1: Render loss computed"
else
    echo "❌ Pass 1: Render loss NOT computed (still skipping)"
fi

if grep -q "🎯 GRADIENT SIMILARITY" logs/VERIFY_PCGRAD.log; then
    echo "✅ PCGrad: Similarity calculation active"
else
    echo "❌ PCGrad: Similarity calculation not found"
fi

if grep -q "\[Early Training\]" logs/VERIFY_PCGRAD.log; then
    echo "✅ Python: No warmup skip (using early training)"
else
    if grep -q "\[Warmup\]" logs/VERIFY_PCGRAD.log; then
        echo "❌ Python: Still showing [Warmup] skip"
    else
        echo "⚠️  Python: Could not determine warmup status"
    fi
fi

echo ""
echo "Full log saved to: logs/VERIFY_PCGRAD.log"
echo ""
