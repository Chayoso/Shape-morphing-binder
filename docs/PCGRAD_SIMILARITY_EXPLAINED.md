# PCGrad Cosine Similarity Explained

## 🎯 What Similarity Do We Calculate?

We calculate the **cosine similarity** between **two gradient vectors**:

1. **Physics Gradients** (from MPM simulation)
2. **Render Gradients** (from 3D Gaussian Splatting)

---

## 📊 The Two Gradient Vectors

Each gradient vector contains **TWO components**:

### Physics Gradients:
```
∂L_physics/∂F  →  How physics loss changes with deformation gradient
∂L_physics/∂x  →  How physics loss changes with particle positions
```

### Render Gradients:
```
∂L_render/∂F   →  How render loss changes with deformation gradient
∂L_render/∂x   →  How render loss changes with particle positions
```

---

## 🔢 How Similarity is Computed

### Step 1: Concatenate Components

```python
# Flatten and concatenate both components into single vectors
g_physics = [∂L_physics/∂F, ∂L_physics/∂x]  # Single long vector
g_render  = [∂L_render/∂F,  ∂L_render/∂x]   # Single long vector
```

**Example shapes:**
- `dLdF`: (N_particles, 3, 3) → flatten to (N_particles * 9,)
- `dLdx`: (N_particles, 3)    → flatten to (N_particles * 3,)
- Combined: (N_particles * 12,) dimensional vector

### Step 2: Compute Cosine Similarity

```python
cosine = dot(g_physics, g_render) / (||g_physics|| × ||g_render||)
```

Where:
- `dot(a, b)` = inner product (sum of element-wise products)
- `||g||` = L2 norm (Euclidean length)

### Step 3: Interpret Result

```
cosine = +1.0  →  Perfect alignment (same direction)
cosine =  0.0  →  Orthogonal (independent)
cosine = -1.0  →  Perfect opposition (opposite directions)
```

---

## 🧮 Concrete Example

Let's say we have 2 particles:

### Physics Gradients:
```python
∂L_physics/∂F = [[1, 0, 0],    # Particle 0: stretch in X
                  [0, 1, 0],
                  [0, 0, 1],

                  [1, 0, 0],    # Particle 1: stretch in X
                  [0, 1, 0],
                  [0, 0, 1]]

∂L_physics/∂x = [[1.0, 0.0, 0.0],   # Particle 0: move +X
                  [1.0, 0.0, 0.0]]   # Particle 1: move +X
```

### Render Gradients:
```python
∂L_render/∂F = [[-1, 0, 0],   # Particle 0: compress in X (opposite!)
                 [0, -1, 0],
                 [0, 0, -1],

                 [-1, 0, 0],   # Particle 1: compress in X (opposite!)
                 [0, -1, 0],
                 [0, 0, -1]]

∂L_render/∂x = [[-1.0, 0.0, 0.0],  # Particle 0: move -X (opposite!)
                 [-1.0, 0.0, 0.0]]  # Particle 1: move -X (opposite!)
```

### Computation:

```python
# Step 1: Flatten and concatenate
g_physics = [1,0,0, 0,1,0, 0,0,1, 1,0,0, 0,1,0, 0,0,1,  1,0,0, 1,0,0]
            └────────── ∂L/∂F (18 elements) ──────────┘  └─ ∂L/∂x (6) ─┘

g_render  = [-1,0,0, 0,-1,0, 0,0,-1, -1,0,0, 0,-1,0, 0,0,-1,  -1,0,0, -1,0,0]
            └────────── ∂L/∂F (18 elements) ──────────┘  └─ ∂L/∂x (6) ─┘

# Step 2: Dot product
dot(g_physics, g_render) = (1×-1) + (1×-1) + (1×-1) + (1×-1) + (1×-1) + (1×-1) + (1×-1) + (1×-1)
                          = -1 - 1 - 1 - 1 - 1 - 1 - 1 - 1
                          = -8

# Step 3: Norms
||g_physics|| = sqrt(1² + 1² + 1² + 1² + 1² + 1² + 1² + 1²) = sqrt(8) = 2.828
||g_render||  = sqrt(1² + 1² + 1² + 1² + 1² + 1² + 1² + 1²) = sqrt(8) = 2.828

# Step 4: Cosine
cosine = -8 / (2.828 × 2.828) = -8 / 8 = -1.0

# Result: Perfect opposition! ⚠️ STRONG CONFLICT
```

**Interpretation:** Physics wants to stretch and move +X, but render wants to compress and move -X. They're fighting!

---

## 🎯 What Does Each Loss Want?

### Physics Loss (MPM Simulation):
- Wants **physically plausible deformations**
- Minimizes strain energy, enforces material properties
- Gradients point toward valid elastic deformations

### Render Loss (3DGS):
- Wants **visual similarity to target**
- Minimizes depth error, edge error, photo error
- Gradients point toward matching target appearance

---

## 🔥 When Do Conflicts Occur?

### Example 1: Bunny Ears
```
Physics:  "Don't stretch too much! (sv_min constraint)"
           → Gradient: compress back

Render:   "Need to form tall ears to match target!"
           → Gradient: stretch upward

Similarity: NEGATIVE (conflict)
```

### Example 2: Surface Alignment
```
Physics:  "Minimize volume change"
           → Gradient: maintain current volume

Render:   "Match target surface depth"
           → Gradient: expand/compress surface

Similarity: NEGATIVE or NEUTRAL (depends on alignment)
```

### Example 3: Converged State
```
Physics:  "Already at equilibrium"
           → Gradient: ~0 (small adjustments)

Render:   "Already matching target"
           → Gradient: ~0 (small adjustments)

Similarity: POSITIVE (aligned)
```

---

## 🧮 Mathematical Details

### Complete Formula:

```
Given:
  g_phys  = [∂L_physics/∂F₁, ..., ∂L_physics/∂Fₙ, ∂L_physics/∂x₁, ..., ∂L_physics/∂xₙ]
  g_render = [∂L_render/∂F₁,  ..., ∂L_render/∂Fₙ,  ∂L_render/∂x₁,  ..., ∂L_render/∂xₙ]

Cosine similarity:
  cos(θ) = (g_phys · g_render) / (||g_phys|| × ||g_render||)

Where:
  g_phys · g_render = Σᵢ (g_phys[i] × g_render[i])  (dot product)
  ||g_phys||        = √(Σᵢ g_phys[i]²)              (L2 norm)
  ||g_render||      = √(Σᵢ g_render[i]²)             (L2 norm)
```

### Properties:

- **Range:** `[-1, +1]`
- **Scale invariant:** Only measures direction, not magnitude
- **Symmetric:** `cos(g_phys, g_render) = cos(g_render, g_phys)`

---

## 🎨 Visual Interpretation

```
         g_physics
              ↑
              │
              │
     -1.0 ←───┼───→ +1.0
              │
              │
              ↓
          g_render

cos = +1.0:  Same direction      (↑↑)  ✅ Aligned
cos =  0.0:  Perpendicular       (↑→)  ~ Neutral
cos = -1.0:  Opposite direction  (↑↓)  ⚠️  Conflict
```

---

## 🔧 Why Concatenate F and x?

**Question:** Why not compute similarity separately for F and x?

**Answer:** We want to know if the **overall optimization direction** conflicts:

### Combined View (Current):
```
Physics: [F_grads, x_grads] → Single direction in parameter space
Render:  [F_grads, x_grads] → Single direction in parameter space
Similarity: Do these directions conflict?
```

### Separate View (Not used):
```
Similarity_F = cos(∂L_phys/∂F, ∂L_render/∂F)
Similarity_x = cos(∂L_phys/∂x, ∂L_render/∂x)

Problem: What if F conflicts but x aligns? How to combine?
```

**By concatenating:** We get a single scalar that tells us if the full parameter update will conflict.

---

## 📊 Logging Example

When you run training, you'll see:

```
├─ [PCGrad Status]
│  ├─ 🎯 GRADIENT SIMILARITY:
│  │   ├─ Cosine: -0.2345 ⚠️ CONFLICT
│  │   └─ Range: -1.0 (opposite) → 0.0 (orthogonal) → +1.0 (aligned)
```

This means:
- Physics gradient vector: `g_phys`
- Render gradient vector: `g_render`
- Angle between them: `cos⁻¹(-0.2345) ≈ 103.5°`
- Interpretation: They point in somewhat opposite directions

---

## 🎯 Summary

### What similarity?
**Cosine similarity between full physics and render gradient vectors**

### What gradients?
```
g_physics = [∂L_physics/∂F, ∂L_physics/∂x]  (concatenated)
g_render  = [∂L_render/∂F,  ∂L_render/∂x]   (concatenated)
```

### How computed?
```
cos(θ) = dot(g_physics, g_render) / (||g_physics|| × ||g_render||)
```

### What does it mean?
- `+1.0`: Physics and render want the same changes ✅
- `0.0`: Physics and render want independent changes ~
- `-1.0`: Physics and render want opposite changes ⚠️

### When does PCGrad activate?
**When `cos(θ) < -0.1` (default threshold)**

---

## 🔍 Code Reference

**Implementation:** `utils/gradient_utils.py:41-69`

```python
def compute_gradient_cosine_similarity(
    dLdF_physics, dLdx_physics,
    dLdF_render, dLdx_render
) -> float:
    # Flatten all gradients
    g_phys = np.concatenate([dLdF_physics.flatten(), dLdx_physics.flatten()])
    g_render = np.concatenate([dLdF_render.flatten(), dLdx_render.flatten()])

    # Compute cosine similarity
    dot = np.dot(g_phys, g_render)
    norm_phys = np.linalg.norm(g_phys) + 1e-12
    norm_render = np.linalg.norm(g_render) + 1e-12

    cosine = dot / (norm_phys * norm_render)
    return float(np.clip(cosine, -1.0, 1.0))
```

**Now you know exactly what similarity we're calculating!** 🎯
