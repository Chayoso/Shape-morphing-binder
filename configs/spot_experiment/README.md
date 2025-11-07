# Spot Experiment Suite - E2E Render Loss Study

Comprehensive experimental study of E2E training for sphere→spot morphing.

**All experiments: 60 episodes, Full 4K resolution, RMS normalization enabled**

---

## Experiment Categories

### 1. Baseline (Physics Only)
- **01_physics_only.yaml** - No render loss (pure physics optimization)
  - `loss.enabled: false`
  - Baseline to compare E2E improvements

### 2. Learning Rate (Alpha) Variations
All with `render_loss_weight = 1e4`, component weights: `w_alpha=2.0, w_depth=5.0, w_photo=1.0, w_edge=3.0, w_cov_align=10.0`

- **02_alpha_0.005_render_1e4.yaml** - Conservative (baseline E2E)
  - `initial_alpha: 0.005`
  - Expected: Stable, slow convergence

- **03_alpha_0.01_render_1e4.yaml** - Moderate
  - `initial_alpha: 0.01` (2× faster)
  - Expected: Faster convergence, slight instability risk

- **04_alpha_0.075_render_1e4.yaml** - Aggressive
  - `initial_alpha: 0.075` (15× faster)
  - Expected: Very fast convergence, potential instability

### 3. Render Weight Variations
All with `initial_alpha = 0.005`, standard component weights

| Experiment | Render Weight | Expected Behavior |
|------------|---------------|-------------------|
| **05_render_1e2.yaml** | 100 | Weak render, physics-dominant |
| **06_render_1e3.yaml** | 1,000 | Balanced physics/render |
| **07_render_5e3.yaml** | 5,000 | Strong render influence |
| **08_render_1e4.yaml** | 10,000 | Very strong render (baseline) |

**Expected effective weights (Episode 15+):**
- 1e2: `w_render = 30` (weak)
- 1e3: `w_render = 300` (moderate)
- 5e3: `w_render = 1,500` (strong)
- 1e4: `w_render = 3,000` (very strong)

### 4. Component Weight Variations
All with `initial_alpha = 0.005, render_loss_weight = 1e4`

| Experiment | Key Parameter | Purpose |
|------------|---------------|---------|
| **09_edge_heavy.yaml** | `w_edge = 10.0` | Sharper boundaries |
| **10_depth_heavy.yaml** | `w_depth = 15.0` | Better shape accuracy |
| **11_cov_heavy.yaml** | `w_cov_align = 20.0` | Stronger F-gradient alignment |
| **12_balanced.yaml** | All weights = 5.0 | Equal component importance |

---

## Quick Start

### Run Single Experiment
```bash
python run.py -c configs/spot_experiment/02_alpha_0.005_render_1e4.yaml --png
```

### Run All Experiments (Batch)
```bash
for i in {01..12}; do
    config=$(ls configs/spot_experiment/${i}_*.yaml 2>/dev/null | head -1)
    if [ -f "$config" ]; then
        echo "Running experiment $i..."
        python run.py -c "$config" --png 2>&1 | tee logs/spot_exp_${i}.log
    fi
done
```

---

## Expected Outputs

Each experiment creates:
- `output/spot_exp/{exp_name}/ep{num}_*.png` - Rendered frames
- `output/spot_exp/{exp_name}/ep{num}_summary.json` - Loss metrics
- Logs with **RENDER LOSS** tracking per episode

### Key Metrics to Track

**Physics Loss** (in summary JSON):
```json
{
  "loss_physics_final": 6500.2
}
```

**Render Loss** (from logs):
```
├─ Render loss: 24.853937
│  ├─ loss_alpha: 0.037436
│  ├─ loss_edge: 0.047123
│  ├─ loss_cov_align: 0.001000
│  ├─ loss_det_barrier: 0.001710
```

**Gradient Combination** (from logs):
```
🔥 [Gradient Combination Summary]
├─ BEFORE normalization:
│  ├─ ||g_render|| = 2.420123e+00
│  ├─ ||g_phys||   = 7.814635e+01
│  ├─ Ratio (render/phys) = 3.096912e-02
├─ WEIGHTS:
│  ├─ w_physics = 1.00
│  ├─ w_render  = 5000.00
│  └─ Strategy  = normalize
├─ AFTER combination:
│  ├─ ||g_combined|| = 2.764228e+05
│  ├─ Ratio (combined/phys) = 3537.2447 ✅
```

---

## Analysis Scripts

### Extract Render Loss Evolution
```bash
# Extract render loss from logs
grep "Render loss:" logs/spot_exp_02.log | awk '{print NR-1, $4}' > render_loss_exp02.txt
```

### Compare Physics Loss Across Experiments
```python
import json
import glob

for exp_dir in sorted(glob.glob('output/spot_exp/*/')):
    summaries = glob.glob(f'{exp_dir}ep*_summary.json')
    if summaries:
        with open(summaries[-1]) as f:
            data = json.load(f)
            print(f"{exp_dir}: Physics Loss = {data['loss_physics_final']:.1f}")
```

### Plot Render vs Physics Convergence
```python
import matplotlib.pyplot as plt
import re

# Parse log file
render_losses = []
physics_losses = []

with open('logs/spot_exp_02.log') as f:
    for line in f:
        if 'Render loss:' in line:
            render_losses.append(float(re.search(r'Render loss: ([\d.]+)', line).group(1)))
        if 'Physics] Pass' in line and 'Final loss:' in line:
            physics_losses.append(float(re.search(r'Final loss: ([\d.]+)', line).group(1)))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(render_losses, label='Render Loss')
plt.xlabel('Episode')
plt.ylabel('Render Loss')
plt.title('Render Loss Evolution')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(physics_losses, label='Physics Loss')
plt.xlabel('Episode')
plt.ylabel('Physics Loss')
plt.title('Physics Loss Evolution')
plt.legend()

plt.tight_layout()
plt.savefig('loss_comparison.png')
```

---

## Expected Results Summary

| Experiment | Physics Loss (Final) | Render Loss (Final) | Convergence Speed |
|------------|---------------------|---------------------|-------------------|
| 01 (Physics Only) | ~350 | N/A | Slow |
| 02 (α=0.005, rlw=1e4) | ~600-800 | ~0.5-1.0 | Moderate |
| 03 (α=0.01, rlw=1e4) | ~700-900 | ~0.3-0.7 | Fast |
| 04 (α=0.075, rlw=1e4) | ~800-1200 | ~0.2-0.5 | Very Fast (may oscillate) |
| 05 (rlw=1e2) | ~400-500 | ~2.0-3.0 | Slow (render weak) |
| 06 (rlw=1e3) | ~500-700 | ~1.0-2.0 | Moderate |
| 07 (rlw=5e3) | ~700-900 | ~0.5-1.0 | Fast |
| 08 (rlw=1e4) | ~800-1000 | ~0.3-0.7 | Very Fast |
| 09 (Edge Heavy) | ~700-900 | ~0.4-0.8 | Sharp edges |
| 10 (Depth Heavy) | ~700-900 | ~0.3-0.6 | Better shape |
| 11 (Cov Heavy) | ~700-900 | ~0.5-1.0 | Better deformation |
| 12 (Balanced) | ~600-800 | ~0.6-1.2 | Well-rounded |

**Note:** Higher physics loss with E2E is EXPECTED - trading physics accuracy for visual quality!

---

## Troubleshooting

### Render Loss Not Decreasing
- Check `render_loss_weight` is being read correctly (look for debug output)
- Verify `w_render (final)` in logs is large (>100)
- Ensure `magnitude_strategy: 'normalize'` is set

### Training Unstable
- Reduce `initial_alpha` (try 0.005 → 0.0025)
- Reduce `render_loss_weight` (try 1e4 → 5e3)
- Check for NaN gradients in logs

### Physics Loss Increasing
- Render weight too strong - reduce `render_loss_weight`
- Learning rate too high - reduce `initial_alpha`
- Add stronger det(F) barrier: `w_det_barrier: 1.0`

---

## File Structure

```
configs/spot_experiment/
├── README.md (this file)
├── 01_physics_only.yaml
├── 02_alpha_0.005_render_1e4.yaml
├── 03_alpha_0.01_render_1e4.yaml
├── 04_alpha_0.075_render_1e4.yaml
├── 05_render_1e2.yaml
├── 06_render_1e3.yaml
├── 07_render_5e3.yaml
├── 08_render_1e4.yaml
├── 09_edge_heavy.yaml
├── 10_depth_heavy.yaml
├── 11_cov_heavy.yaml
└── 12_balanced.yaml
```

Generated configs: 12 total
All experiments: 60 episodes each = 720 total episodes
Estimated runtime: ~2-3 hours per experiment (24-36 hours total for full suite)
