# Surface Mask Strict Threshold Fix

## Problem

`edge_alignment_mean` was always 0.0 even though the edge loss was being computed with a surface_mask containing 30,000 particles.

## Root Cause Analysis

### Symptoms
```json
{
  "num_surface_particles": 30000,
  "num_total_particles": 150000,
  "edge_alignment_mean": 0.0,           // ❌ Zero (surface particles)
  "edge_alignment_mean_all": 0.006417,  // ✅ Non-zero (all particles)
  "edge_grad_norm_mean": 1.3e-05        // ❌ Near-zero edge gradients
}
```

### Investigation
1. Target alpha **does have edges**: grad_norm mean = 3.13e-03, max = 2.56
2. But edge loss sees: grad_norm mean = 1.3e-05 (240x weaker!)
3. This means: **Surface particles are NOT projecting onto edge regions**

### Root Cause
In `utils/rendering_utils.py:701`, the code was marking **ALL phi-filtered particles as surface**:

```python
# ❌ BEFORE (WRONG)
surface_mask = torch.ones(mask_phi.sum().item(), dtype=torch.bool, device=mu.device)
```

**Why this is wrong:**
- `tau_phi = 0.12 * dx` is a **loose filter** to remove outliers
- After phi-filtering, you still have BOTH:
  - True surface particles (|φ| ≈ 0)
  - Near-surface volume particles (0 < |φ| < 0.12*dx)
- Marking ALL as "surface" includes many volume particles
- These volume particles project to **interior regions** (not edges)
- Result: edge_grad_norm ≈ 0 → edge_alignment = 0

## Solution

Use a **stricter SDF threshold** to identify TRUE surface particles:

```python
# ✅ AFTER (CORRECT)
# Apply loose filter first (remove outliers)
mask_phi = phi_vals.abs() < tau_phi  # tau_phi = 0.12*dx

# Then apply STRICT threshold for surface identification
phi_vals_filtered = phi_vals[mask_phi]
tau_surface = 0.015 * dx  # Much stricter (8x stricter than tau_phi)
surface_mask = phi_vals_filtered.abs() < tau_surface
```

**Key parameters:**
- `tau_phi = 0.12 * dx`: Loose filter to remove far-field particles
- `tau_surface = 0.015 * dx`: Strict threshold for TRUE surface (8x stricter)
- Ratio: `tau_surface / tau_phi = 0.125` (12.5%)

## Expected Results

### Before Fix
```
φ-mask: 150,000 → 30,000 particles (tau_phi=0.12*dx)
Surface mask: ALL 30,000 particles marked as surface  ❌
  → Many are volume particles
  → They project to interior (no edges)
  → edge_alignment_mean = 0.0
```

### After Fix
```
φ-mask: 150,000 → 30,000 particles (tau_phi=0.12*dx)
Surface mask (strict): 3,000 / 30,000 = 10% (tau_surface=0.015*dx) ✅
  → Only TRUE surface particles (|φ| < 0.015*dx)
  → They project to actual silhouette edges
  → edge_alignment_mean > 0.0 (expected: 0.3-0.7)
```

## Code Changes

**File:** `utils/rendering_utils.py`

**Location:** Lines 696-729

**Changes:**
1. Added `phi_vals_filtered = phi_vals[mask_phi]` to extract SDF values after filtering
2. Added `tau_surface = 0.015 * dx` for strict surface threshold
3. Created `surface_mask_strict` using the strict threshold
4. Updated logging to show surface ratio and threshold values

## Verification

After running training, check the episode summary:

```json
{
  "num_surface_particles": 3000,      // Should be 10-20% of filtered
  "num_total_particles": 30000,       // After phi-filter
  "surface_ratio": 0.10,              // Should be ~10-20% (not 1.0)
  "edge_alignment_mean": 0.45,        // Should be > 0.0 (target: 0.3-0.7)
  "edge_alignment_mean_all": 0.045,   // All particles (for comparison)
  "edge_grad_norm_mean": 0.0031       // Should match target (~3.13e-03)
}
```

**Success criteria:**
- ✅ `edge_alignment_mean > 0.0` (surface particles)
- ✅ `edge_alignment_mean >> edge_alignment_mean_all` (10x higher)
- ✅ `edge_grad_norm_mean ≈ 3.13e-03` (matches target alpha edges)
- ✅ `surface_ratio ≈ 0.1-0.2` (10-20%, not 100%)

## Tuning

If `edge_alignment_mean` is still low after fix, adjust `tau_surface`:

```python
# More strict (fewer surface particles, higher quality)
tau_surface = 0.01 * dx   # 1% threshold → ~5% of filtered particles

# Less strict (more surface particles, lower quality)
tau_surface = 0.02 * dx   # 2% threshold → ~15% of filtered particles

# Current (recommended)
tau_surface = 0.015 * dx  # 1.5% threshold → ~10% of filtered particles
```

**Rule of thumb:**
- Stricter threshold → fewer "surface" particles → higher edge alignment quality
- Too strict → may miss some actual surface regions
- Target: 10-20% of phi-filtered particles should be marked as surface

## Related Files

- `utils/rendering_utils.py:696-729` - Main fix location
- `loss.py:1158-1204` - Surface mask usage in edge loss
- `SURFACE_MASK_FIX.md` - Previous surface mask documentation
- `EDGE_LOSS_FIX_SUMMARY.md` - Edge loss gradient fix documentation
