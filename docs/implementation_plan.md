# Implementation Plan: Render-Guided Physics Morphing

## Status (as of 2026-03-27)

### What we know from 40ep HC run

| Metric | Value |
|--------|-------|
| phys_ctrl_norm (late) | 2~3 |
| inject_F_norm | 0.2~0.3 |
| cos(phys, render)_x | -0.001~-0.003 |
| HC phys vs PO phys | +4% worse |
| alpha_mse final | 0.080 |

**Root cause**: render injection scaled by physics grad norm → vanishes as physics converges.
Render gradient direction nearly orthogonal to physics in F-space.

---

## Phase 1: Diagnostic Experiments

### Exp 1 — Fixed-norm injection

**What changes**: remove phys_ctrl_norm-based scaling.

```
# BEFORE (bad)
lambda_eff = alpha_balance × phys_ctrl_norm / dLdF_norm
g_inject = g_render × lambda_eff

# AFTER (Exp 1)
g_inject = alpha_fixed × (g_render / (||g_render|| + eps))
```

**Config parameter**: `hard_coupling.alpha_fixed` (replaces alpha_balance)
**Sweep**: α ∈ {0.1, 0.5, 1.0}, each 10ep test

**What to measure**:
- inject_F_norm stays non-zero late in optimization
- physics_loss vs PO baseline
- alpha_mse improvement rate

**Configs to create**:
- `configs/exp1_a01.yaml` (α=0.1)
- `configs/exp1_a05.yaml` (α=0.5)
- `configs/exp1_a10.yaml` (α=1.0)

---

### Exp 4 — Final-state render gradient field visualization

**What it does**: visualize dL/dx_T as 3D vector field on particles.
Check if gradient points toward missing silhouette regions (e.g. bunny ears).

**Script**: `tools/visualize_render_grad.py`
- loads ep particle positions
- overlays dLdx vectors (colored by magnitude)
- saves as HTML (plotly) or PNG (matplotlib 3D)

**What to check**:
- coherence: do nearby particles have similar gradient direction?
- semantics: do vectors point toward target shape missing regions?
- noise: is the field globally smooth or locally chaotic?

---

## Phase 2: Option A — Target-State Proximal Guidance

### Concept

Instead of injecting render gradients directly into F-space:

```
# OLD (direct injection, broken)
dLdF_inject = alpha × normalize(dLdF_render)   # F-space, misaligned
dLdx_inject = alpha × normalize(dLdx_render)

# NEW (Option A)
x_target = x_T - η × clip(dLdx_render / ||dLdx_render||, τ)  # smoothed target
dLdx_attr = 2 × λ_attr × (x_T - x_target)     # attractor gradient (x-space only)
dLdF_inject = 0                                 # no F injection
dLdx_inject = dLdx_attr
```

### Why this works

- render says **where to go** (x_target)
- physics decides **how to get there** (backprop through simulation)
- attractor magnitude = distance to target → naturally large when far, small when close
- no direct F-space injection → avoids orthogonality problem

### Stabilization

1. **Normalize render gradient**: `g_hat = dLdx / (||dLdx|| + eps)`
2. **Clip step**: `η_clipped = clip(η, τ_max)` to prevent extreme x_target
3. **Smooth x_target**: KNN neighbor averaging over particle positions
4. **Late activation**: `attractor_start_ep` (default 5)

### Config parameters

```yaml
hard_coupling:
  mode: option_a          # 'fixed_norm' for Exp1, 'option_a' for Phase 2
  render_start_ep: 5
  # Exp 1 params
  alpha_fixed: 0.5        # fixed inject norm (Exp 1 only)
  # Option A params
  eta: 0.1                # target step size
  lambda_attr: 1.0        # attractor loss weight
  tau: 0.5                # clip threshold for x_target step
  smooth_k: 8             # KNN neighbors for x_target smoothing
  attractor_start_ep: 5
```

### Implementation

**Python-side only** (no C++ changes needed):

```python
# In extract_render_gradients() → returns dLdx_render
# In run_episode():

# 1. compute x_target
g_hat = dLdx_render / (np.linalg.norm(dLdx_render) + 1e-8)
g_hat_clipped = np.clip(g_hat, -tau, tau)
x_target = x_T - eta * g_hat_clipped  # (N, 3)

# optional: smooth x_target via KNN
x_target = smooth_positions(x_target, x_T, k=smooth_k)

# 2. attractor gradient
dLdx_attr = 2.0 * lambda_attr * (x_T - x_target)  # (N, 3)

# 3. inject (dLdF = 0)
new_render_grads = {
    'dLdF': np.zeros_like(F_e),   # no F injection
    'dLdx': np.ascontiguousarray(dLdx_attr),
}
```

---

## Code Refactoring Plan

### Files to clean

| File | Changes |
|------|---------|
| `utils/training_loop.py` | Remove: `get_lambda_alpha()`, `compute_smoothing_loss()` dead code, phys_ctrl_norm scaling. Add: `compute_inject_fixed_norm()`, `compute_attractor_grads()` |
| `run.py` | Remove: smooth_cfg, ema_state dead code. Simplify episode loop. |
| `utils/alpha_losses.py` | Keep as-is (clean already) |
| `utils/physics_utils.py` | Check for dead code |
| `configs/sphere_to_bunny.yaml` | Remove: schedule section. Clean up comments. |
| `configs/sphere_to_bunny_physics_only.yaml` | Remove: schedule section. |

### New files

| File | Purpose |
|------|---------|
| `configs/exp1_a01.yaml` | Exp 1, α=0.1 |
| `configs/exp1_a05.yaml` | Exp 1, α=0.5 |
| `configs/exp1_a10.yaml` | Exp 1, α=1.0 |
| `configs/option_a_v1.yaml` | Option A first version |
| `tools/visualize_render_grad.py` | Exp 4 visualization |

### training_loop.py structure after refactor

```
render_hard_coupled()         # unchanged
extract_render_gradients()    # unchanged (returns raw dLdF, dLdx)
_compute_pca_normals()        # unchanged

# NEW injection strategies
compute_inject_fixed_norm(dLdF, dLdx, alpha)   # Exp 1
compute_attractor_grads(x_T, dLdx, eta, lambda_attr, tau, smooth_k)  # Option A

run_episode()                 # dispatches based on mode
```

---

## Execution Order

1. **Write this doc** ✓
2. **Refactor code** (remove dead code, add mode dispatch)
3. **PO 40ep baseline** (background, configs/sphere_to_bunny_physics_only.yaml)
4. **Exp 1**: α sweep 10ep each → pick best α
5. **Exp 4**: visualization script
6. **Option A v1**: implement + 10ep test
7. **Full 40ep validation**: PO vs Exp1-best vs OptionA

---

## Comparison Table (to fill in)

| Method | phys_loss@ep39 | alpha_mse@ep39 | inject_F_norm | cos(p,r) |
|--------|---------------|----------------|---------------|----------|
| PO baseline | 278.4 | TBD | 0 | - |
| HC (current, phys-scaled) | 290.5 | 0.080 | 0.2~0.3 | ≈0 |
| Exp1 α=0.1 | TBD | TBD | TBD | TBD |
| Exp1 α=0.5 | TBD | TBD | TBD | TBD |
| Exp1 α=1.0 | TBD | TBD | TBD | TBD |
| Option A v1 | TBD | TBD | TBD | TBD |
