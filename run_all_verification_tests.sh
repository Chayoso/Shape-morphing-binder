#!/bin/bash
# Run all verification tests in sequence

set -e

echo "========================================================================"
echo "Running All Verification Tests"
echo "========================================================================"
echo ""
echo "This will run 3 tests to compare:"
echo "  1. Physics-only (single-scale F-field)"
echo "  2. Physics-only (multi-scale F-field)"
echo "  3. Full E2E (physics + render + PCGrad)"
echo ""
echo "Each test runs 30 episodes. Total time: ~30-60 minutes"
echo ""

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate diffmpm_v2.3.0

# Timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGDIR="logs/verify_${TIMESTAMP}"
mkdir -p "$LOGDIR"

echo "Logs will be saved to: $LOGDIR"
echo ""

# Test 1: Physics-only (single-scale)
echo "========================================================================"
echo "[1/3] Running: Physics-Only (Single-Scale)"
echo "========================================================================"
python run.py -c configs/examples/verify/1_physics_only.yaml --png 2>&1 | tee "$LOGDIR/1_physics_only.log"
echo ""
echo "✅ Test 1 complete!"
echo ""

# Test 2: Physics-only with multi-scale
echo "========================================================================"
echo "[2/3] Running: Physics-Only (Multi-Scale)"
echo "========================================================================"
python run.py -c configs/examples/verify/2_physics_multiscale.yaml --png 2>&1 | tee "$LOGDIR/2_physics_multiscale.log"
echo ""
echo "✅ Test 2 complete!"
echo ""

# Test 3: Full E2E
echo "========================================================================"
echo "[3/3] Running: Full E2E (Physics + Render + PCGrad)"
echo "========================================================================"
python run.py -c configs/examples/verify/3_full_e2e_pcgrad.yaml --png 2>&1 | tee "$LOGDIR/3_full_e2e_pcgrad.log"
echo ""
echo "✅ Test 3 complete!"
echo ""

echo "========================================================================"
echo "All Tests Complete!"
echo "========================================================================"
echo ""
echo "Logs saved to: $LOGDIR/"
echo ""
echo "Output images:"
echo "  - output/verify/1_physics_only/"
echo "  - output/verify/2_physics_multiscale/"
echo "  - output/verify/3_full_e2e_pcgrad/"
echo ""

# Quick comparison
echo "========================================================================"
echo "Quick Loss Comparison (First 10 Episodes)"
echo "========================================================================"
echo ""

echo "Test 1 (Physics-Only, Single-Scale):"
grep "Final loss" "$LOGDIR/1_physics_only.log" | head -10 | nl

echo ""
echo "Test 2 (Physics-Only, Multi-Scale):"
grep "Final loss" "$LOGDIR/2_physics_multiscale.log" | head -10 | nl

echo ""
echo "Test 3 (Full E2E):"
grep "Final loss" "$LOGDIR/3_full_e2e_pcgrad.log" | head -10 | nl

echo ""
echo "========================================================================"
echo "For detailed analysis, see logs in: $LOGDIR/"
echo "========================================================================"
