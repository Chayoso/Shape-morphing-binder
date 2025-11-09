# Edge Loss Quick Reference Card

## 🚀 Quick Start

### Default Configuration (Recommended)
```yaml
optimization:
  loss:
    w_edge: 0.1
    edge_loss_mode: 'detach_position'
    debug_edge_gradients: false
```

---

## 🔧 Configuration Options

### `edge_loss_mode`

| Mode | When to Use | Stability | Effectiveness |
|------|-------------|-----------|---------------|
| `detach_position` | **Default** - Use first | ✅ Stable | Good if edges > 0.01 |
| `gradient_boost` | Weak edges (0.001-0.01) | ⚠️ Can be unstable | Stronger signal |
| `original` | Debugging only | ✅ Stable | ❌ Doesn't work |

### `gradient_boost_factor`

Only applies when `edge_loss_mode: 'gradient_boost'`

| Factor | Use Case | Risk |
|--------|----------|------|
| 100 | Conservative | Low |
| 1000 | **Default** - Moderate | Medium |
| 5000 | Weak edges | Medium-High |
| 10000 | Very weak edges | High (watch for NaN) |

### `debug_edge_gradients`

- `true`: Print detailed diagnostics (use for first run)
- `false`: Quiet mode (use for production)

---

## 📊 Interpreting Metrics

### `grad_norm_mean` (Edge Strength)

Shown in debug output when `debug_edge_gradients: true`

| Value | Interpretation | Action |
|-------|----------------|--------|
| > 0.1 | ✅ Strong edges | Use `detach_position` |
| 0.01 - 0.1 | ⚠️ Moderate edges | Try `detach_position`, then `gradient_boost` |
| 0.001 - 0.01 | ⚠️ Weak edges | Use `gradient_boost` (factor 5000+) |
| < 0.001 | ❌ Too weak | **Disable edge loss** (`w_edge: 0.0`) |

### `edge_alignment_mean` (Training Progress)

Shown in episode summaries

| Value | Meaning | Expected Trend |
|-------|---------|----------------|
| < 0.01 | No alignment | Should increase if edges are strong |
| 0.1 - 0.5 | Partial alignment | Improving |
| > 0.8 | Good alignment | Target achieved ✅ |

---

## 🎯 Decision Tree

```
START
  ↓
Run 1 episode with: edge_loss_mode: 'detach_position', debug_edge_gradients: true
  ↓
Check grad_norm mean in logs
  ↓
  ├─ > 0.1 → ✅ Keep detach_position, disable debug
  ├─ 0.01-0.1 → Run 5 more episodes
  │             ├─ edge_alignment increasing → ✅ Keep detach_position
  │             └─ edge_alignment stuck → Try gradient_boost (factor 5000)
  ├─ 0.001-0.01 → Switch to gradient_boost (factor 5000-10000)
  └─ < 0.001 → ❌ Disable edge loss (w_edge: 0.0)
```

---

## 🛠️ Diagnostic Tools

### Quick Analysis (Python)
```python
from utils.edge_diagnostics import analyze_alpha_edges

stats = analyze_alpha_edges(alpha_target, verbose=True)
# Check stats['grad_norm_mean']
```

### Visualization
```python
from utils.edge_diagnostics import visualize_alpha_edges

visualize_alpha_edges(alpha_target, output_path="debug/edges.png")
```

### Command Line Example
```bash
python examples/diagnose_edge_loss.py
```

---

## ⚠️ Troubleshooting

### Problem: edge_alignment_mean stuck at ~0.006

**Solution 1**: Check edge strength
```bash
# Set debug_edge_gradients: true, run 1 episode, check logs
# Look for "grad_norm mean: X.XXXe-XX"
```

**Solution 2**: If grad_norm < 0.01, try gradient boost
```yaml
edge_loss_mode: 'gradient_boost'
gradient_boost_factor: 5000.0
```

**Solution 3**: If grad_norm < 0.001, disable edge loss
```yaml
w_edge: 0.0
w_alpha: 0.3  # Compensate with other losses
```

---

### Problem: NaN/Inf gradients

**Cause**: `gradient_boost_factor` too high

**Solution**: Reduce factor or switch mode
```yaml
gradient_boost_factor: 500.0  # Reduce by 2-10x
# or
edge_loss_mode: 'detach_position'  # Switch to safer mode
```

---

### Problem: Don't know if edge loss is helping

**Solution**: Run ablation test
```bash
# Run with edge loss
python run.py --config config_with_edge.yaml

# Run without edge loss
python run.py --config config_no_edge.yaml

# Compare final renders
```

---

## 📝 Example Configs

See `configs/edge_loss_examples/`:
- `1_detach_position.yaml` - Default (recommended)
- `2_gradient_boost_moderate.yaml` - Weak edges
- `3_gradient_boost_aggressive.yaml` - Very weak edges
- `4_edge_disabled.yaml` - Edges too weak
- `EDGE_LOSS_MODES.md` - Full documentation

---

## 📚 Full Documentation

- **Summary**: `EDGE_LOSS_FIX_SUMMARY.md`
- **Configuration Guide**: `configs/edge_loss_examples/EDGE_LOSS_MODES.md`
- **Example Script**: `examples/diagnose_edge_loss.py`
- **Source Code**: `loss.py` (lines 972-1103), `utils/edge_diagnostics.py`

---

## 🎓 Key Concepts

### The Problem
- Edge loss was generating two gradient paths:
  1. **dL/dF** (real signal) - weak (0.086)
  2. **dL/dx** (spurious signal) - strong (1.99) - **23x stronger!**
- Spurious signal drowned out real signal

### The Solutions
1. **detach_position**: Block spurious dL/dx (clean, recommended)
2. **gradient_boost**: Amplify dL/dF to overcome dL/dx (fallback)
3. **Disable**: When edges too weak (< 0.001)

### Success Criteria
- `edge_alignment_mean` increases from ~0.006 to > 0.5
- No NaN/Inf in gradients
- Visual improvement in renders

---

## ✅ Checklist for First Run

- [ ] Set `edge_loss_mode: 'detach_position'`
- [ ] Set `debug_edge_gradients: true`
- [ ] Run 1 episode
- [ ] Check logs for `grad_norm mean`
- [ ] Decide: keep detach / try boost / disable
- [ ] Set `debug_edge_gradients: false` for production
- [ ] Monitor `edge_alignment_mean` in episode summaries
- [ ] Visualize edges if uncertain: `python examples/diagnose_edge_loss.py`

---

**Last Updated**: 2025
**Version**: 1.0
