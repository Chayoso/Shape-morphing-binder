# Configurable Number of Passes per Episode

## Problem: E2E Local Minima from Multiple Passes

**Root Cause:** E2E mode calls `SetUpCompGraph()` multiple times per episode (once per pass), resetting the simulation state each time. This creates discontinuities and local minima.

**Physics-Only:** 1 optimization call per episode ✅
**E2E (3 passes):** 3 optimization calls per episode, each resetting simulation ❌

## Solution: Configure `num_passes`

Add to your YAML config:

```yaml
optimization:
  num_passes: 1  # Reduce from default 3 to avoid resets
```

## Comparison

### Default (3 passes):
```
Episode 0:
  Pass 1: SetUpCompGraph() → Optimize (physics only)
  Pass 2: SetUpCompGraph() → Optimize (physics + render) ← RESET!
  Pass 3: SetUpCompGraph() → Optimize (physics + render) ← RESET!
Total: 3× simulation resets per episode
```

### With num_passes: 1
```
Episode 0:
  Pass 1: SetUpCompGraph() → Optimize (physics + render)
Total: 1× simulation reset per episode (same as physics-only!)
```

## Recommended Configurations

### Option 1: Single Pass with PCGrad (Recommended)
```yaml
optimization:
  use_session_mode: false  # Required for PCGrad
  use_pcgrad: true         # Resolve gradient conflicts
  num_passes: 1            # Avoid simulation resets
  
  loss:
    enabled: true
    render_loss_weight: 100.0  # Can be higher with PCGrad
```

**Benefits:**
- No simulation resets (like physics-only)
- PCGrad resolves gradient conflicts
- Faster than 3-pass mode

### Option 2: Session Mode (Fastest)
```yaml
optimization:
  use_session_mode: true   # 10-15x faster!
  num_passes: 1            # Single pass is enough
  use_pcgrad: false        # Not available in session mode
  
  loss:
    enabled: true
    render_loss_weight: 10.0  # Lower weight without PCGrad
```

**Benefits:**
- Maximum performance
- No simulation resets
- Session mode handles state properly

### Option 3: Multi-Pass for Refinement (Advanced)
```yaml
optimization:
  use_session_mode: false
  use_pcgrad: true
  num_passes: 2  # First pass: physics, Second pass: refinement
  
  loss:
    enabled: true
    render_loss_weight: 50.0
```

**Use when:** You want progressive refinement and accept the overhead

## How It Works

**Legacy Mode (`use_session_mode: false`):**
- Each pass calls `cg.run_optimization(opt)` from Python
- This calls `OptimizeDefGradControlSequence()` in C++
- Which calls `SetUpCompGraph()` → resets simulation layers

**Session Mode (`use_session_mode: true`):**
- Calls `SetUpCompGraph()` ONCE in `InitializeEpisode()`
- All passes share same simulation state
- No resets between passes!

## Migration Guide

**If you're getting local minima in E2E:**

1. Try single pass first:
   ```yaml
   optimization:
     num_passes: 1
   ```

2. If still having issues, switch to session mode:
   ```yaml
   optimization:
     use_session_mode: true
     num_passes: 1
     use_pcgrad: false  # Not available yet
     loss:
       render_loss_weight: 10.0  # Reduce weight
   ```

3. Monitor convergence - you should see improvement!

## Backward Compatibility

- **Default:** `num_passes: 3` (maintains existing behavior)
- **Existing configs:** Will work unchanged
- **New configs:** Can optimize by setting `num_passes: 1`
