# Ablation Study: Loss Component Analysis

This directory contains configurations for systematically evaluating the contribution of each loss component in the physics-guided 3D Gaussian Splatting shape morphing pipeline.

## Study Design

| Experiment | Config | Loss Components | Purpose |
|------------|--------|-----------------|---------|
| **1. Physics Only** | `01_physics_only.yaml` | Physics only (`render_loss_weight=0`) | Baseline - pure mass/momentum conservation |
| **2. Covariance Only** | `02_covariance_only.yaml` | Physics + Covariance (`cov_align=1.0`) | Test local shape/curvature matching |
| **3. Alpha Only** | `03_alpha_only.yaml` | Physics + Alpha (`alpha=1.0`) | Test global silhouette matching |
| **4. Depth Only** | `04_depth_only.yaml` | Physics + Depth (`depth=1.0`) | Test 3D structure via depth maps |
| **5. All Combined** | `05_all_combined.yaml` | Physics + All (`α=1.0, d=0.3, c=0.5`) | Full model with balanced weights |

## Key Parameters (Standardized Across All Configs)

### Training
- **Episodes**: 50
- **Save frequency**: Every 10 episodes
- **Mode**: E2E rendering with session mode
- **Target**: Isosphere → Bunny

### Physics
- **Timesteps**: 10 per episode
- **Time step (dt)**: 0.00833333 (120 fps)
- **Drag**: 0.5
- **Smoothing**: 0.955 (APIC)

### Optimization
- **Initial alpha**: 0.01
- **Adaptive alpha**: Enabled (target_norm=2500.0, min_scale=0.1)
- **Max GD iterations**: 1
- **Max line search**: 15
- **PCGrad**: Enabled (threshold=-0.1)

### Upsampling & Covariance
- **Multi-scale F-field**: Enabled
  - Coarse: 64 neighbors
  - Fine: 16 neighbors
  - Standard: 32 neighbors
  - Blend mode: Adaptive
- **Subdivision**: Enabled (target=250K particles, jitter=0.4)
- **Curvature-based covariance**: Enabled for target
- **F-smoothing**: Disabled (per user request)

## Loss Component Details

### 1. Physics Loss (Always Active)
- **Type**: Hard constraint from MPM simulation
- **Function**: Ensures mass/momentum conservation, physical plausibility
- **Gradient source**: Backpropagation through MPM solver

### 2. Covariance Alignment (`cov_align`)
- **Type**: Gaussian covariance matrix matching
- **Function**: Matches local surface curvature and orientation
- **Strength**: Excellent for fine surface details
- **Limitation**: May not capture global shape without other losses

### 3. Alpha Loss (`alpha`)
- **Type**: L1 loss on alpha channel (opacity)
- **Function**: Matches silhouette and object boundaries
- **Strength**: Strong global shape signal
- **Limitation**: No depth information, can't distinguish front/back

### 4. Depth Loss (`depth`)
- **Type**: L1 loss on depth maps
- **Function**: Matches 3D structure and depth ordering
- **Strength**: Provides explicit 3D geometry signal
- **Limitation**: View-dependent, may need multiple views

### 5. Regularizers (Always Active)
- **`cov_reg`** (0.01): Prevents covariance collapse
- **`det_barrier`** (0.1): Prevents degenerate Gaussians

## Running the Study

### Option 1: Run All Experiments (Recommended)
```bash
./run_ablation_study.sh
```
This will:
- Run all 5 experiments sequentially
- Save logs to `logs/ablation_*.log`
- Save outputs to `output/ablation/*/`
- Display total runtime

### Option 2: Run Individual Experiments
```bash
# Baseline
python run.py --config configs/ablation_study/01_physics_only.yaml

# Individual losses
python run.py --config configs/ablation_study/02_covariance_only.yaml
python run.py --config configs/ablation_study/03_alpha_only.yaml
python run.py --config configs/ablation_study/04_depth_only.yaml

# Full model
python run.py --config configs/ablation_study/05_all_combined.yaml
```

### Option 3: Quick Test (First Episode Only)
```bash
# Modify num_episodes to 1 in any config, then run
python run.py --config configs/ablation_study/01_physics_only.yaml
```

## Expected Results

| Experiment | Expected Shape Match | Expected Detail | Expected Convergence |
|------------|---------------------|-----------------|---------------------|
| 1. Physics Only | Poor | Poor | Fast (no render loss) |
| 2. Covariance Only | Moderate | Good | Moderate |
| 3. Alpha Only | Good (silhouette) | Poor (interior) | Moderate |
| 4. Depth Only | Moderate-Good | Moderate | Moderate |
| 5. All Combined | **Best** | **Best** | Slowest (more constraints) |

## Analysis

### Evaluation Metrics
After running experiments, compare:
1. **Visual quality**: Rendered images at episodes 10, 20, 30, 40, 50
2. **Loss curves**: Track physics loss, render losses, total loss
3. **Convergence rate**: How quickly does each approach reach good results?
4. **Final Chamfer distance**: Measure geometric accuracy (if ground truth available)

### Key Questions
1. **Which single loss is most important?** Compare experiments 2-4
2. **Is the full model better than the sum of parts?** Compare experiment 5 vs individual components
3. **What does physics alone achieve?** Experiment 1 baseline
4. **Are losses complementary?** Does combining them improve over best single loss?

## Output Structure
```
output/ablation/
├── 01_physics_only/
│   ├── ep000/
│   ├── ep010/
│   ├── ...
│   └── ep050/
├── 02_covariance_only/
├── 03_alpha_only/
├── 04_depth_only/
└── 05_all_combined/
```

Each episode directory contains:
- `render_*.png` - Rendered images
- `particles_*.obj` - Particle positions (if saved)
- Loss values logged in terminal/logs

## Notes

### Important Design Decisions
1. **Physics-only uses renderer**: Even experiment 1 enables E2E rendering for visualization, but sets `render_loss_weight=0` so gradients don't affect optimization
2. **Fair comparison**: All experiments use identical physics, optimization, and upsampling parameters
3. **Balanced weights in combined model**: The weights in experiment 5 (`α=1.0, d=0.3, c=0.5`) were tuned for this specific task
4. **F-smoothing disabled**: User requested this be turned off across all experiments

### Customization
To modify for different tasks:
- Change `input_mesh` and `target_mesh` in all configs
- Adjust loss weights in `05_all_combined.yaml` based on target shape complexity
- Increase `num_episodes` for harder morphing tasks
- Modify `subdivision_target` based on available GPU memory

## References
See main project documentation for:
- Loss function mathematical definitions
- PCGrad gradient combination details
- Multi-scale F-field interpolation
- Curvature-based covariance computation
