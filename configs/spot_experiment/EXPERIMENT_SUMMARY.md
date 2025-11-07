# Spot Experiment Suite - Created Configs

## ✅ Created Configs (Ready to Run)

1. **01_physics_only.yaml** - Physics-only baseline
2. **02_alpha_0.005_render_1e4.yaml** - Conservative α, strong render
3. **03_alpha_0.01_render_1e4.yaml** - Moderate α, strong render
4. **04_alpha_0.075_render_1e4.yaml** - Aggressive α, strong render
5. **05_render_1e2.yaml** - Weak render influence

## 📝 To Be Created (Use Template Below)

6. **06_render_1e3.yaml** - Moderate render (rlw=1e3)
7. **07_render_5e3.yaml** - Strong render (rlw=5e3)
8. **08_render_1e4.yaml** - Very strong render (rlw=1e4)
9. **09_edge_heavy.yaml** - Edge-focused (w_edge=10.0)
10. **10_depth_heavy.yaml** - Depth-focused (w_depth=15.0)
11. **11_cov_heavy.yaml** - F-gradient focused (w_cov_align=20.0)
12. **12_balanced.yaml** - Equal components (all=5.0)

---

## Quick Creation Template

### For Render Weight Variations (06-08)

**Copy `05_render_1e2.yaml` and modify:**
- **06**: Change `1e2` → `1e3` everywhere
- **07**: Change `1e2` → `5e3` everywhere
- **08**: Change `1e2` → `1e4` everywhere
- Update `output_dir` and file comment

### For Component Weight Variations (09-12)

**Copy `02_alpha_0.005_render_1e4.yaml` and modify component weights:**

**09 (Edge Heavy):**
```yaml
w_edge: 10.0  # Changed from 3.0
```

**10 (Depth Heavy):**
```yaml
w_depth: 15.0  # Changed from 5.0
```

**11 (Covariance Heavy):**
```yaml
w_cov_align: 20.0  # Changed from 10.0
```

**12 (Balanced):**
```yaml
w_alpha: 5.0      # Changed from 2.0
w_depth: 5.0      # Same
w_photo: 5.0      # Changed from 1.0
w_edge: 5.0       # Changed from 3.0
w_cov_align: 5.0  # Changed from 10.0
```

**Remember to update in BOTH sections:**
- `optimization.loss` section
- `upsample` section

---

## Automated Creation Script

```bash
# Create remaining render weight configs
for rlw in 1e3 5e3 1e4; do
  num=$([ "$rlw" = "1e3" ] && echo "06" || [ "$rlw" = "5e3" ] && echo "07" || echo "08")
  sed "s/1e2/$rlw/g; s/05_render_1e2/${num}_render_$rlw/g" \
    configs/spot_experiment/05_render_1e2.yaml > \
    configs/spot_experiment/${num}_render_$rlw.yaml
done

echo "Created configs 06-08!"
```

---

## Run Experiments

### Single Experiment
```bash
python run.py -c configs/spot_experiment/02_alpha_0.005_render_1e4.yaml --png
```

### Batch Run (All Created Configs)
```bash
for config in configs/spot_experiment/0[1-5]_*.yaml; do
  echo "Running $config..."
  python run.py -c "$config" --png 2>&1 | tee logs/spot_$(basename $config .yaml).log
done
```

---

## Render Loss Tracking

All configs automatically track render loss. Look for these in logs:

```
[Render] Computing loss for Pass 3...
├─ Render loss: 24.793589
│  ├─ loss_alpha: 0.039214
│  ├─ loss_edge: 0.042638
│  ├─ loss_cov_align: 0.001000
│  ├─ loss_det_barrier: 0.001844
```

### Extract Render Loss to CSV
```bash
grep "Render loss:" logs/spot_02_alpha_0.005_render_1e4.log | \
  awk '{print NR-1, $4}' | \
  sed 's/ /,/g' > render_loss_exp02.csv
```

### Extract All Component Losses
```bash
grep -E "loss_(alpha|edge|depth|photo|cov_align):" logs/spot_02_alpha_0.005_render_1e4.log | \
  awk '{print $2, $3}' > component_losses.txt
```

---

## Expected Timeline

- **Created configs (01-05)**: ~10-15 hours total
- **Full suite (01-12)**: ~24-36 hours total
- **Per experiment**: ~2-3 hours (60 episodes × 2-3 min/episode)

---

## Analysis Checklist

For each experiment, track:

- [ ] Physics loss convergence (final episode)
- [ ] Render loss evolution (plot over episodes)
- [ ] Gradient combination ratio (render/physics)
- [ ] Visual quality of final render
- [ ] Training stability (any NaN/divergence?)
- [ ] Convergence speed (episodes to reach threshold)

Compare across:
- Physics-only vs E2E methods
- Different learning rates (alpha variations)
- Different render weights (weak → strong)
- Different component emphases (edge, depth, cov)
