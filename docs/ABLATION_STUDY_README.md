# Ablation Study: Loss Weight Balance

## Overview

This ablation study tests 6 different loss weight configurations on 2 tasks:
- **Bunny**: Sphere → Bunny (20 episodes)
- **Spot**: Sphere → Spot (20 episodes)

**Total experiments**: 12 (6 × 2)

## Hypothesis

Current settings suffer from "Depth Drowning" where:
- `w_depth × loss_depth ≈ 8.7` (dominant)
- `w_edge × loss_edge ≈ 0.1` (negligible)
- `w_cov_align × loss_cov_align ≈ 0.06` (negligible)

This prevents geometric losses (edge, covariance) from properly guiding the optimization.

## Experiment Configurations

### 1. Baseline (Current State - "Depth Drowning")
**File**: `1_baseline.yaml`

```yaml
w_alpha: 0.0
w_depth: 5.0      # Dominant contribution ≈ 8.7
w_photo: 0.0
w_edge: 3.0       # Weak contribution ≈ 0.1
w_cov_align: 10.0 # Weak contribution ≈ 0.06
```

**Expected**: Similar to current results (depth dominates)

---

### 2. Equal Contribution (Balanced - **Recommended**)
**File**: `2_equal_contribution.yaml`

```yaml
w_alpha: 0.0
w_depth: 0.5       # Contribution ≈ 0.88
w_photo: 0.0
w_edge: 27.0       # Contribution ≈ 1.0
w_cov_align: 160.0 # Contribution ≈ 0.96
```

**Expected**: Best balance - all losses contribute equally (~1.0)

---

### 3. Geometric Dominance
**File**: `3_geometric_dominance.yaml`

```yaml
w_alpha: 0.0
w_depth: 0.2       # Auxiliary ≈ 0.35
w_photo: 0.0
w_edge: 54.0       # Dominant ≈ 2.0
w_cov_align: 330.0 # Dominant ≈ 2.0
```

**Expected**: Strong geometric guidance, minimal depth

---

### 4. Depth-Led, Edge-Assisted
**File**: `4_depth_led_edge_assisted.yaml`

```yaml
w_alpha: 0.0
w_depth: 3.0       # Co-dominant ≈ 5.25
w_photo: 0.0
w_edge: 135.0      # Co-dominant ≈ 5.0
w_cov_align: 160.0 # Auxiliary ≈ 0.96
```

**Expected**: Depth and edge work together

---

### 5. Alpha-Led (Silhouette "GPS")
**File**: `5_alpha_led.yaml`

```yaml
w_alpha: 35.0      # GPS ≈ 1.0
w_depth: 0.0       # Disabled
w_photo: 0.0
w_edge: 27.0       # Contribution ≈ 1.0
w_cov_align: 160.0 # Contribution ≈ 0.96
```

**Expected**: Silhouette provides global guidance instead of depth

---

### 6. Geometric Only (Control - "No GPS")
**File**: `6_geometric_only_control.yaml`

```yaml
w_alpha: 0.0
w_depth: 0.0       # No global guidance
w_photo: 0.0
w_edge: 10.0       # Local geometry only
w_cov_align: 20.0  # Local geometry only
```

**Expected**: Likely to fail - no global shape guidance

---

## Running the Study

### Run All Experiments (Overnight)
```bash
./run_ablation_study.sh
```

This will run all 12 experiments sequentially (~4-6 hours total).

### Run Individual Experiments
```bash
# Bunny experiments
python3 run.py configs/ablation_study/bunny/1_baseline.yaml
python3 run.py configs/ablation_study/bunny/2_equal_contribution.yaml
# ... etc

# Spot experiments
python3 run.py configs/ablation_study/spot/1_baseline.yaml
python3 run.py configs/ablation_study/spot/2_equal_contribution.yaml
# ... etc
```

## Output Structure

Results will be saved to:
```
output/ablation/
├── bunny/
│   ├── 1_baseline/
│   ├── 2_equal_contribution/
│   ├── 3_geometric_dominance/
│   ├── 4_depth_led_edge_assisted/
│   ├── 5_alpha_led/
│   └── 6_geometric_only_control/
└── spot/
    ├── 1_baseline/
    ├── 2_equal_contribution/
    ├── 3_geometric_dominance/
    ├── 4_depth_led_edge_assisted/
    ├── 5_alpha_led/
    └── 6_geometric_only_control/
```

Each directory contains:
- Episode images (`ep###_render.png`, `ep###_alpha.png`, `ep###_depth.png`)
- Episode summaries (`ep###_summary.json`)
- Training losses (`training_losses.json`)
- Training summary plot (`training_summary.png`)

## Metrics to Compare

After running, compare these metrics across experiments:

1. **Physics Loss** (`loss_physics_final`)
   - Lower is better
   - Shows how well particles match target density

2. **Render Losses**
   - `loss_edge`: Edge alignment quality
   - `loss_cov_align`: Covariance alignment quality
   - `loss_depth`: Depth map alignment
   - `loss_alpha`: Silhouette alignment

3. **Visual Quality**
   - Final rendered shape (episode 19)
   - Convergence speed (how quickly losses decrease)

## Analysis Scripts

Generate comparison videos:
```bash
# For each experiment
python3 create_bunny_video.py output/ablation/bunny/2_equal_contribution --type comparison --fps 10
python3 create_bunny_video.py output/ablation/spot/2_equal_contribution --type comparison --fps 10
```

Plot training summaries:
```bash
python3 utils/plot_summaries.py output/ablation/bunny/2_equal_contribution
python3 utils/plot_summaries.py output/ablation/spot/2_equal_contribution
```

## Key Changes from Current Config

- **Increased** `render_loss_weight`: `1e2` → `1e4` (100× stronger)
- **Rebalanced** loss weights to equalize contributions
- **Fixed** gradient chain bug (if applied)

## Expected Winner

**Experiment 2: Equal Contribution** is expected to perform best because:
- All losses contribute equally (~1.0)
- No single loss dominates
- Both global (depth) and local (edge, cov) guidance are balanced
