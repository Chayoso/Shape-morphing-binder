# SetUpCompGraph Fix: Enable Multi-Pass Without Resets

## Problem

In legacy E2E mode with multiple passes, `SetUpCompGraph()` was called for **every pass**, resetting simulation layers 1-N each time. This caused:
- Discontinuities between passes
- Wasted computation
- Local minima in optimization

## Solution

Added `skip_setup` parameter to conditionally call `SetUpCompGraph()`:

### C++ Changes

**CompGraph.h** - Added parameter:
```cpp
void OptimizeDefGradControlSequence(
    // ... existing params ...
    bool skip_setup = false  // Set true to skip SetUpCompGraph (for pass 2+)
);
```

**CompGraph.cpp** - Conditional setup:
```cpp
if (!skip_setup) {
    std::cout << "[Setup] Initializing computation graph..." << std::endl;
    SetUpCompGraph(num_steps);
} else {
    std::cout << "[Setup] Skipping SetUpCompGraph (using existing state)" << std::endl;
}
```

**bind.cpp** - Python binding:
```cpp
.def("run_optimization", [](CompGraph& self, const OptInput& opt, bool skip_setup = false) {
    // ...
    self.OptimizeDefGradControlSequence(
        // ... params ...
        skip_setup  // 🔥 MULTI-PASS FIX
    );
}, py::arg("opt"), py::arg("skip_setup") = false)
```

### Python Changes

**utils/physics_utils.py** - Accept parameter:
```python
def run_physics_optimization(
    cg, opt, num_timesteps, control_stride, ep, pass_idx=0,
    skip_setup=False  # 🔥 Skip SetUpCompGraph for pass 2+
):
    cg.run_optimization(opt, skip_setup=skip_setup)
```

**utils/training_loop.py** - Use for pass 2+:
```python
# Skip SetUpCompGraph for pass 2+ to preserve simulation state
skip_setup = (pass_idx > 0)
loss_physics = run_physics_optimization_batched(
    cg, opt, render_grads_dict, pass_idx, skip_setup=skip_setup
)
```

## How It Works

### Before (3 passes with resets):
```
Pass 1:
  SetUpCompGraph()  ← Reset simulation
  Optimize

Pass 2:
  SetUpCompGraph()  ← Reset AGAIN! ❌
  Optimize

Pass 3:
  SetUpCompGraph()  ← Reset AGAIN! ❌
  Optimize
```

### After (3 passes without resets):
```
Pass 1:
  SetUpCompGraph()  ← Setup once
  Optimize

Pass 2:
  [Skip setup]      ← Preserve state ✅
  Optimize

Pass 3:
  [Skip setup]      ← Preserve state ✅
  Optimize
```

## Usage

### Option 1: Use num_passes=1 (Simplest)
```yaml
optimization:
  num_passes: 1  # No resets needed!
```

### Option 2: Use Multi-Pass with Skip (Advanced)
```yaml
optimization:
  num_passes: 3  # Full refinement
```
The `skip_setup` parameter is automatically used for pass 2-3!

### Option 3: Session Mode (Recommended)
```yaml
optimization:
  use_session_mode: true  # Handles this internally!
```

## Backward Compatibility

- `skip_setup` defaults to `false`
- Existing code works unchanged
- Physics-only mode (single optimization call) unaffected

## Limitations

- **Batched mode**: Still calls `SetUpCompGraph()` on each pass (hardcoded)
- **Workaround**: Use `num_passes: 1` or switch to session mode

## Testing

The fix is automatically applied when running with multiple passes.
You should see in the logs:

```
Pass 1:
  [Setup] Initializing computation graph...
  
Pass 2:
  [Setup] Skipping SetUpCompGraph (using existing state)
  
Pass 3:
  [Setup] Skipping SetUpCompGraph (using existing state)
```

## Recommendations

1. For PCGrad + E2E: Use `num_passes: 1`
2. For maximum performance: Use session mode
3. For multi-pass refinement: This fix enables it properly!
