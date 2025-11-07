# Spot Experiment Suite - Quick Start Guide

## 🚀 Generate All Configs

```bash
cd configs/spot_experiment
chmod +x create_remaining.sh
./create_remaining.sh
```

This will create all 12 experiment configs (06-12).

---

## 📋 Experiment Overview

| # | Name | Parameter Variation | Purpose |
|---|------|---------------------|---------|
| **01** | physics_only | No E2E | Baseline |
| **02** | alpha_0.005_render_1e4 | α=0.005 | Conservative (baseline E2E) |
| **03** | alpha_0.01_render_1e4 | α=0.01 | 2× faster learning |
| **04** | alpha_0.075_render_1e4 | α=0.075 | 15× faster learning |
| **05** | render_1e2 | rlw=100 | Weak render |
| **06** | render_1e3 | rlw=1,000 | Moderate render |
| **07** | render_5e3 | rlw=5,000 | Strong render |
| **08** | render_1e4 | rlw=10,000 | Very strong render |
| **09** | edge_heavy | w_edge=10.0 | Sharp boundaries |
| **10** | depth_heavy | w_depth=15.0 | Shape accuracy |
| **11** | cov_heavy | w_cov_align=20.0 | F-gradient alignment |
| **12** | balanced | All weights=5.0 | Equal components |

**All experiments: 60 episodes, 4K resolution, RMS normalization**

---

## ▶️ Run Experiments

### Single Experiment
```bash
python run.py -c configs/spot_experiment/02_alpha_0.005_render_1e4.yaml --png
```

### Run All (Batch)
```bash
for config in configs/spot_experiment/0*.yaml configs/spot_experiment/1*.yaml; do
    if [ -f "$config" ]; then
        exp_name=$(basename "$config" .yaml)
        echo "====================================="
        echo "Running experiment: $exp_name"
        echo "====================================="
        python run.py -c "$config" --png 2>&1 | tee "logs/${exp_name}.log"
    fi
done
```

### Run Specific Category
```bash
# Alpha variations (02-04)
for i in 02 03 04; do
    python run.py -c configs/spot_experiment/${i}_*.yaml --png 2>&1 | tee logs/exp_${i}.log
done

# Render weight variations (05-08)
for i in 05 06 07 08; do
    python run.py -c configs/spot_experiment/${i}_*.yaml --png 2>&1 | tee logs/exp_${i}.log
done
```

---

## 📊 Track Render Loss

### Extract from Logs
```bash
# Extract render loss per episode
grep "Render loss:" logs/02_alpha_0.005_render_1e4.log | \
    awk '{print NR-1, $4}' > render_loss_02.txt

# Extract all component losses
grep -E "loss_(alpha|edge|depth|photo|cov_align):" logs/02_alpha_0.005_render_1e4.log > \
    components_02.txt
```

### Plot Render Loss Evolution
```python
import matplotlib.pyplot as plt
import re

def extract_render_loss(log_file):
    losses = []
    with open(log_file) as f:
        for line in f:
            if 'Render loss:' in line:
                match = re.search(r'Render loss: ([\d.]+)', line)
                if match:
                    losses.append(float(match.group(1)))
    return losses

# Plot
exp_02 = extract_render_loss('logs/02_alpha_0.005_render_1e4.log')
exp_05 = extract_render_loss('logs/05_render_1e2.log')
exp_08 = extract_render_loss('logs/08_render_1e4.log')

plt.figure(figsize=(10, 6))
plt.plot(exp_02, label='Exp 02 (rlw=1e4, α=0.005)', linewidth=2)
plt.plot(exp_05, label='Exp 05 (rlw=1e2)', linewidth=2)
plt.plot(exp_08, label='Exp 08 (rlw=1e4)', linewidth=2)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Render Loss', fontsize=12)
plt.title('Render Loss Evolution Across Experiments', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('render_loss_comparison.png', dpi=300)
plt.show()
```

### Compare All Experiments
```python
import json
import glob
import pandas as pd

results = []
for exp_dir in sorted(glob.glob('output/spot_exp/*/')):
    exp_name = exp_dir.split('/')[-2]
    summaries = sorted(glob.glob(f'{exp_dir}ep*_summary.json'))

    if summaries:
        # Get final episode
        with open(summaries[-1]) as f:
            data = json.load(f)

        # Extract render loss from logs
        log_file = f'logs/{exp_name}.log'
        render_losses = []
        try:
            with open(log_file) as f:
                for line in f:
                    if 'Render loss:' in line:
                        match = re.search(r'Render loss: ([\d.]+)', line)
                        if match:
                            render_losses.append(float(match.group(1)))
        except:
            pass

        results.append({
            'Experiment': exp_name,
            'Physics Loss (Final)': data.get('loss_physics_final', 0),
            'Render Loss (Final)': render_losses[-1] if render_losses else 0,
            'Render Loss (Initial)': render_losses[0] if render_losses else 0,
            'Improvement': (render_losses[0] - render_losses[-1]) if render_losses else 0
        })

df = pd.DataFrame(results)
df.to_csv('experiment_results.csv', index=False)
print(df.to_string())
```

---

## 🔍 Monitor During Training

Watch render loss in real-time:
```bash
tail -f logs/02_alpha_0.005_render_1e4.log | grep --line-buffered "Render loss:"
```

Check gradient combination:
```bash
tail -f logs/02_alpha_0.005_render_1e4.log | grep --line-buffered -A 15 "Gradient Combination Summary"
```

---

## 📈 Expected Results

### Render Loss Convergence

| Experiment | Initial | Episode 30 | Episode 60 | Improvement |
|------------|---------|------------|------------|-------------|
| 02 (baseline) | ~25 | ~2.0 | ~0.5-1.0 | 96-98% |
| 03 (α=0.01) | ~25 | ~1.5 | ~0.3-0.7 | 97-99% |
| 04 (α=0.075) | ~25 | ~1.0 | ~0.2-0.5 | 98-99% |
| 05 (rlw=1e2) | ~25 | ~4.0 | ~2.0-3.0 | 88-92% |
| 08 (rlw=1e4) | ~25 | ~1.5 | ~0.3-0.7 | 97-99% |

### Physics vs Render Trade-off

**Expected pattern:**
- Higher render weight → Higher physics loss, Lower render loss
- Lower render weight → Lower physics loss, Higher render loss

**Sweet spot:** Experiment 06-07 (rlw=1e3-5e3) should give best balance

---

## ⚠️ Troubleshooting

### Render Loss Not Decreasing
```bash
# Check if weight is being read correctly
grep "render_loss_weight READ:" logs/*.log
# Should show 100.0, 1000.0, 10000.0, etc.

# Check final weight
grep "w_render (final):" logs/*.log
# Should show values like 5000.0, 30000.0, etc.
```

### Training Diverges
- Try lower alpha: 0.075 → 0.01 → 0.005
- Try lower render weight: 1e4 → 5e3 → 1e3
- Check for NaN in logs: `grep -i nan logs/*.log`

---

## 📁 Output Structure

```
output/spot_exp/
├── 01_physics_only/
│   ├── ep000_*.png
│   ├── ep000_summary.json
│   └── ...
├── 02_alpha_0.005_render_1e4/
│   └── ...
└── ...

logs/
├── 01_physics_only.log
├── 02_alpha_0.005_render_1e4.log
└── ...
```

---

## ⏱️ Estimated Runtime

- **Per experiment**: 2-3 hours (60 episodes)
- **Full suite (12 experiments)**: 24-36 hours
- **Parallel (4 GPUs)**: 6-9 hours

Run in parallel on different GPUs:
```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python run.py -c configs/spot_experiment/01_*.yaml --png

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python run.py -c configs/spot_experiment/02_*.yaml --png

# ...
```

---

## 📚 Next Steps

1. **Generate configs**: Run `./create_remaining.sh`
2. **Run experiments**: Start with 01, 02, 05 to validate setup
3. **Monitor**: Watch logs for render loss convergence
4. **Analyze**: Compare results using provided Python scripts
5. **Visualize**: Create plots of render loss evolution
6. **Report**: Document findings in experiment notes

**Good luck with your experiments!** 🚀
