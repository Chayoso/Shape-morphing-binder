#!/bin/bash
# Run all ablation study experiments overnight
# Total: 12 experiments (6 bunny + 6 spot)

echo "========================================="
echo "Starting Ablation Study - Overnight Run"
echo "========================================="
echo "Total experiments: 12 (6 bunny + 6 spot)"
echo "Estimated time: ~4-6 hours"
echo ""

# Bunny experiments
echo "========================================="
echo "BUNNY EXPERIMENTS (Sphere → Bunny)"
echo "========================================="

for exp in 1_baseline 2_equal_contribution 3_geometric_dominance 4_depth_led_edge_assisted 5_alpha_led 6_geometric_only_control; do
    echo ""
    echo "┌─────────────────────────────────────────"
    echo "│ Running: Bunny - $exp"
    echo "└─────────────────────────────────────────"
    python3 run.py configs/ablation_study/bunny/${exp}.yaml

    if [ $? -ne 0 ]; then
        echo "❌ ERROR: Bunny $exp failed!"
    else
        echo "✅ Completed: Bunny $exp"
    fi
done

# Spot experiments
echo ""
echo "========================================="
echo "SPOT EXPERIMENTS (Sphere → Spot)"
echo "========================================="

for exp in 1_baseline 2_equal_contribution 3_geometric_dominance 4_depth_led_edge_assisted 5_alpha_led 6_geometric_only_control; do
    echo ""
    echo "┌─────────────────────────────────────────"
    echo "│ Running: Spot - $exp"
    echo "└─────────────────────────────────────────"
    python3 run.py configs/ablation_study/spot/${exp}.yaml

    if [ $? -ne 0 ]; then
        echo "❌ ERROR: Spot $exp failed!"
    else
        echo "✅ Completed: Spot $exp"
    fi
done

echo ""
echo "========================================="
echo "✅ ALL ABLATION EXPERIMENTS COMPLETE!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  output/ablation/bunny/1_baseline/"
echo "  output/ablation/bunny/2_equal_contribution/"
echo "  output/ablation/bunny/3_geometric_dominance/"
echo "  output/ablation/bunny/4_depth_led_edge_assisted/"
echo "  output/ablation/bunny/5_alpha_led/"
echo "  output/ablation/bunny/6_geometric_only_control/"
echo ""
echo "  output/ablation/spot/1_baseline/"
echo "  output/ablation/spot/2_equal_contribution/"
echo "  output/ablation/spot/3_geometric_dominance/"
echo "  output/ablation/spot/4_depth_led_edge_assisted/"
echo "  output/ablation/spot/5_alpha_led/"
echo "  output/ablation/spot/6_geometric_only_control/"
