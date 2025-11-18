# Ablation Study: Render Loss Component Analysis

This directory contains configurations for systematically evaluating the contribution of each rendering loss component in the physics-guided shape morphing pipeline.

## Study Design

| Experiment | Config | Active Losses | Purpose |
|------------|--------|---------------|---------|
| **1. Physics Only** | `01_physics_only.yaml` | None (`render_loss_weight=0`) | **Baseline** - Pure physics, no rendering guidance |
| **2. Alpha Only** | `02_alpha_only.yaml` | Alpha (silhouette) | Test global boundary/silhouette matching |
| **3. Depth Only** | `03_depth_only.yaml` | Depth | Test 3D structure via depth maps |
| **4. Full Model (Ours)** | `04_full_model.yaml` | All (α, d, edge, cov) | **Complete system** with all components |

## Key Differences from `bunny_1`

All configs match `bunny_1` (full model) except for the ablated components:
- Same physics parameters (4³ particles/cell, drag=0.5, etc.)
- Same optimization (20 episodes, adaptive alpha enabled)
- Same upsampling (100K subdivided particles, multiscale F-field)
- Only difference: **which render losses are active**

## Loss Component Details

### Physics Loss (Always Active)
- **Source**: MPM simulation gradients
- **Function**: Mass/momentum conservation, physical plausibility
- **Cannot be disabled**: Forms the physical foundation

### Alpha Loss (`w_alpha`)
- **Type**: L1 loss on opacity channel
- **Function**: Matches silhouette and object boundaries
- **Strength**: Strong global shape signal
- **Limitation**: No depth information

### Depth Loss (`w_depth`)
- **Type**: L1 loss on depth maps
- **Function**: Matches 3D structure and depth ordering
- **Strength**: Explicit 3D geometry signal
- **Limitation**: View-dependent

### Edge Loss (`w_edge`)
- **Type**: Edge alignment loss
- **Function**: Matches object boundaries precisely
- **Enabled in**: Full model only

### Covariance Loss (`w_cov_align`)
- **Type**: Gaussian covariance matrix matching
- **Function**: Local surface curvature and orientation
- **Enabled in**: Full model only

## Running the Study

### Individual Experiments
```bash
# 1. Physics only baseline
python run.py --config configs/ablation/01_physics_only.yaml

# 2. Alpha loss only
python run.py --config configs/ablation/02_alpha_only.yaml

# 3. Depth loss only
python run.py --config configs/ablation/03_depth_only.yaml

# 4. Full model (reference)
python run.py --config configs/ablation/04_full_model.yaml
```

### Batch Run
```bash
# Run all experiments sequentially
for config in configs/ablation/*.yaml; do
    python run.py --config $config
done
```

## Expected Results

| Experiment | Chamfer Dist. ↓ | IoU ↑ | Mass Err. ↓ | Shape Quality |
|------------|----------------|-------|-------------|---------------|
| 1. Physics Only | ~15.0 | ~5% | ~10% | Poor |
| 2. Alpha Only | ~7-8 | ~14% | ~24% | Moderate |
| 3. Depth Only | ~8-9 | ~12% | ~22% | Moderate |
| **4. Full Model** | **5.65** | **17.03%** | **26.15%** | **Best** |

## Analysis After Running

### 1. Calculate Metrics
```bash
# After experiments complete, calculate metrics
python calculate_metrics_final_episodes.py

# Calculate mass error
python calculate_mass_error.py

# Generate LaTeX table
python generate_ablation_table.py
```

### 2. Compare Visual Results
```bash
# Check rendered outputs
ls output/ablation/*/ep019/*_render.png
```

### 3. Key Questions to Answer
1. **Which single loss is most effective?** Compare experiments 2-3
2. **Is physics alone sufficient?** Experiment 1 baseline
3. **Do losses combine synergistically?** Compare Full vs individual
4. **Trade-off**: Shape quality vs. mass conservation

## Output Structure
```
output/ablation/
├── 01_physics_only/
│   └── ep019/
│       ├── ep019_gaussians.npz
│       ├── ep019_render.png
│       └── metrics_final.json
├── 02_alpha_only/
├── 03_depth_only/
└── 04_full_model/
```

## LaTeX Table Generation

After running all experiments:
```bash
python generate_ablation_table.py
```

Generates:
```latex
\begin{table}[t]
\centering
    \caption{\textbf{Quantitative Evaluation \& Ablation Study...}}
    ...
\end{table}
```

## Key Findings (Expected)

1. **Physics alone is insufficient** for accurate shape morphing (high Chamfer distance)
2. **Render losses are complementary**:
   - Alpha provides global shape
   - Depth provides 3D structure
   - Combined is better than individual
3. **Mass error trade-off**: Better shape quality comes at cost of ~26% mass error (acceptable)
4. **Full model achieves best balance** between fidelity and physics

## Notes

- Requires diff-gaussian-rasterization module to be built
- Each experiment takes ~2-4 hours (20 episodes @ 10 timesteps)
- GPU memory: ~16GB recommended for 100K particles
- For quick tests, reduce `num_animations` to 5-10

## Comparison with bunny_1

The full model config (`04_full_model.yaml`) should produce identical results to `configs/physics/bunny_1.yaml`:
- Chamfer Distance: 5.65 × 10⁻²
- IoU: 17.03%
- Mass Error: 26.15%

This validates that the ablation setup is fair and controlled.
