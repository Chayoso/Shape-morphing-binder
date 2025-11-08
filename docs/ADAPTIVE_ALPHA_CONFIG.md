# Configurable Adaptive Alpha Parameters

## Overview

The adaptive alpha (learning rate) feature now supports full YAML configuration! You can control whether adaptive scaling is enabled and tune the threshold parameters.

## YAML Configuration

Add these parameters to your config file under `optimization:`:

```yaml
optimization:
  # Base learning rate
  initial_alpha: 0.01

  # Adaptive alpha parameters (optional - defaults shown)
  adaptive_alpha_enabled: true      # Enable/disable adaptive learning rate
  adaptive_alpha_target_norm: 2500.0  # Target gradient norm threshold
  adaptive_alpha_min_scale: 0.1     # Minimum alpha scale (10% of base)
```

## Parameters Explained

### `adaptive_alpha_enabled` (boolean)
- **Default:** `true`
- **Purpose:** Master switch for adaptive learning rate
- **When to use:**
  - `true`: Let the optimizer automatically reduce learning rate when gradients are too large (recommended)
  - `false`: Use fixed `initial_alpha` throughout optimization

### `adaptive_alpha_target_norm` (float)
- **Default:** `2500.0`
- **Purpose:** Target gradient norm for stable optimization
- **How it works:** When gradient norm exceeds this threshold, `alpha` is scaled down proportionally
- **Formula:** `alpha_scale = min(1.0, target_norm / current_grad_norm)`
- **When to adjust:**
  - **Increase** (e.g., 3000.0) if optimization is too conservative
  - **Decrease** (e.g., 2000.0) if you're getting line search failures

### `adaptive_alpha_min_scale` (float)
- **Default:** `0.1` (10% of base)
- **Purpose:** Prevents alpha from becoming too small
- **Range:** [0.0, 1.0]
- **Example:** With `initial_alpha=0.01` and `min_scale=0.1`, alpha will never go below `0.001`

## Examples

### Example 1: Disable Adaptive Alpha (Fixed Learning Rate)

```yaml
optimization:
  initial_alpha: 0.005  # Lower base rate since no adaptation
  adaptive_alpha_enabled: false
```

**Output:**
```
[Fixed Alpha] alpha=0.005 (adaptive disabled)
```

### Example 2: More Aggressive Adaptation

```yaml
optimization:
  initial_alpha: 0.01
  adaptive_alpha_enabled: true
  adaptive_alpha_target_norm: 2000.0  # Lower threshold = more adaptation
  adaptive_alpha_min_scale: 0.05     # Allow more reduction
```

**Output (when grad_norm=3000):**
```
[Adaptive Alpha] grad_norm=3000, target=2000, scale=0.6667, alpha=0.006667 (reduced from 0.01)
```

### Example 3: Conservative Adaptation

```yaml
optimization:
  initial_alpha: 0.01
  adaptive_alpha_enabled: true
  adaptive_alpha_target_norm: 3500.0  # Higher threshold = less adaptation
  adaptive_alpha_min_scale: 0.3      # Don't reduce below 30%
```

**Output (when grad_norm=3000):**
```
[Adaptive Alpha] grad_norm=3000, target=3500, scale=1.0, alpha=0.01 (no reduction)
```

## How It Works

**During optimization, for each control timestep:**

1. **Check if adaptive alpha is enabled:**
   - If `false`: Use fixed `initial_alpha`
   - If `true`: Continue to step 2

2. **Compute current gradient norm:**
   ```cpp
   float current_grad_norm = Compute_dLdF_Norm();
   ```

3. **Scale alpha if gradients too large:**
   ```cpp
   float alpha_scale = min(1.0, target_norm / current_grad_norm);
   alpha_scale = max(alpha_scale, min_scale);  // Clamp to minimum
   float alpha = initial_alpha * alpha_scale;
   ```

4. **Print debug info:**
   - Shows when scaling occurs
   - Displays actual gradient norm
   - Shows computed scale factor

## Default Behavior (Backward Compatible)

**If you don't specify these parameters in your YAML:**
- Uses the original hardcoded values
- `adaptive_alpha_enabled = true`
- `adaptive_alpha_target_norm = 2500.0`
- `adaptive_alpha_min_scale = 0.1`

**This ensures backward compatibility** - existing configs will work exactly as before!

## When to Tune These Parameters

### Increase `target_norm` if:
- Optimization is too conservative
- Learning is very slow
- Gradients are consistently high but stable

### Decrease `target_norm` if:
- Getting line search failures
- Optimization is unstable
- Seeing NaN or inf values

### Decrease `min_scale` if:
- Need more aggressive reduction for very high gradients
- Complex geometry requires careful steps

### Disable adaptive alpha if:
- You want complete control over learning rate
- Using custom learning rate schedules in `episode_schedule`
- Debugging convergence issues

## Technical Details

**Files Modified:**
- `bind/bind.cpp` - Added parameters to OptInput and E2EConfig structs
- `DiffMPMLib3D/CompGraph.h/.cpp` - Added parameters to OptimizeDefGradControlSequence
- `DiffMPMLib3D/E2ESession.h/.cpp` - Added parameters to SessionConfig
- `utils/physics_utils.py` - Read parameters from YAML
- `run.py` - Pass parameters to session mode

**Performance:**
- No performance impact when enabled (backward pass already computed)
- Slightly faster when disabled (skips one backward pass per timestep)

## Example Config Files

See `configs/verify/3_full_e2e_pcgrad.yaml` for a complete example with all adaptive alpha parameters configured.
