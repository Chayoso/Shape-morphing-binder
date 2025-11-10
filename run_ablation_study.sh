#!/bin/bash
# Ablation Study Runner
# Runs all ablation experiments sequentially

set -e  # Exit on error

echo "========================================="
echo "  Ablation Study: Loss Component Analysis"
echo "========================================="
echo ""
echo "Running 5 experiments:"
echo "  1. Physics Only (baseline)"
echo "  2. Covariance Only"
echo "  3. Alpha Only"
echo "  4. Depth Only"
echo "  5. All Combined (full model)"
echo ""

# Track start time
start_time=$(date +%s)

# Array of config files
configs=(
    "configs/ablation_study/01_physics_only.yaml"
    "configs/ablation_study/02_covariance_only.yaml"
    "configs/ablation_study/03_alpha_only.yaml"
    "configs/ablation_study/04_depth_only.yaml"
    "configs/ablation_study/05_all_combined.yaml"
)

# Array of experiment names
names=(
    "Physics Only"
    "Covariance Only"
    "Alpha Only"
    "Depth Only"
    "All Combined"
)

# Run each experiment
for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    name="${names[$i]}"
    
    echo "========================================="
    echo "Experiment $((i+1))/5: $name"
    echo "Config: $config"
    echo "========================================="
    
    # Run experiment
    python run.py --config "$config" 2>&1 | tee "logs/ablation_$(printf "%02d" $((i+1)))_$(echo $name | tr ' ' '_' | tr '[:upper:]' '[:lower:]').log"
    
    echo ""
    echo "✓ Completed: $name"
    echo ""
done

# Calculate total time
end_time=$(date +%s)
total_time=$((end_time - start_time))
hours=$((total_time / 3600))
minutes=$(((total_time % 3600) / 60))
seconds=$((total_time % 60))

echo "========================================="
echo "  Ablation Study Complete!"
echo "========================================="
echo ""
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
echo ""
echo "Results saved to:"
echo "  - output/ablation/01_physics_only/"
echo "  - output/ablation/02_covariance_only/"
echo "  - output/ablation/03_alpha_only/"
echo "  - output/ablation/04_depth_only/"
echo "  - output/ablation/05_all_combined/"
echo ""
echo "Logs saved to:"
echo "  - logs/ablation_*.log"
echo ""
echo "Next steps:"
echo "  1. Compare results: python utils/plot_summaries.py --ablation configs/ablation_study"
echo "  2. Review README: configs/ablation_study/README.md"
echo ""
