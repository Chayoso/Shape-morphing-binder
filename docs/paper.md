# Joint Optimization of 3D Physical Simulation and Alpha Silhouette Matching

## 1. Overview

We consider the problem of jointly optimizing a **3D physical simulation** and an **alpha silhouette matching objective** in a single end-to-end differentiable pipeline.

The goal is to optimize a shared parameter set \(\theta\) such that:

1. the simulated motion/deformation remains **physically plausible**, and  
2. the rendered alpha silhouettes match the target silhouettes as closely as possible.

Formally, we want:

\[
\min_{\theta} L_{\text{total}}(\theta)
\]

where the total loss combines silhouette matching, physical consistency, and regularization terms.

---

## 2. Problem Setup

### 2.1 Optimization Variables

The optimization variable \(\theta\) may include:

- **initial states**: particle positions, initial velocities
- **control variables**: external forces, actuator parameters
- **material parameters**: Young's modulus \(E\), Poisson's ratio \(\nu\), yield stress, damping
- **shape parameters**: latent rest-shape parameters, canonical deformation parameters
- **bridge parameters**: parameters mapping physics states to rendering states

A key principle is:

> The rendering representation should not be independently optimized in a way that breaks consistency with physics.

Instead, the rendering state should ideally be a deterministic function of the physical state.

---

### 2.2 Physical State

Let the physical state at time \(t\) be

\[
x_t = \{p_i^t, v_i^t, F_i^t, C_i^t, m_i, \rho_i, \dots\}
\]

where, for example:

- \(p_i^t\): particle position
- \(v_i^t\): particle velocity
- \(F_i^t\): deformation gradient
- \(C_i^t\): APIC affine matrix
- \(m_i\): particle mass
- \(\rho_i\): density

For MPM-style simulations, this parameterization is natural.

---

### 2.3 Silhouette Observation

Let the predicted alpha silhouette at time \(t\) and camera \(c\) be

\[
\hat M_{t,c} = \mathcal{R}_{\alpha}(x_t)
\]

and let the ground-truth silhouette be

\[
M_{t,c}^{\text{gt}}.
\]

The renderer \(\mathcal{R}_{\alpha}\) must be differentiable with respect to the physical state.

---

## 3. End-to-End Pipeline

The full pipeline is:

\[
\theta \xrightarrow{\text{simulate}} x_{1:T}
\]

\[
x_t \xrightarrow{\text{render bridge}} g_t
\]

\[
g_t \xrightarrow{\text{differentiable alpha rendering}} \hat M_{t,c}
\]

where:

- \(\theta\) are the optimization variables
- \(x_t\) is the simulation state
- \(g_t\) is the rendering proxy representation (e.g., Gaussian parameters, mesh, occupancy)
- \(\hat M_{t,c}\) is the rendered alpha silhouette

The simulator evolves the state as:

\[
x_{t+1} = \mathcal{P}(x_t; \theta)
\]

and the rendering bridge is defined as:

\[
g_t = h(x_t).
\]

Then the renderer produces:

\[
\hat M_{t,c} = \mathcal{R}_{\alpha}(g_t, \Pi_c)
\]

where \(\Pi_c\) denotes camera parameters.

---

## 4. Core Design Principle: Strong Coupling Between Physics and Rendering

The most important design choice is how to couple the rendering representation to the physical state.

### 4.1 Hard Coupling

The preferred approach is to define the rendering representation as a deterministic function of the physical state:

\[
g_t = h(x_t).
\]

For example, if we use Gaussian splats derived from particles, then for particle \(i\):

### Mean
\[
\mu_i^t = p_i^t
\]

### Covariance
A physically meaningful choice is:

\[
\Sigma_i^t = F_i^t \Sigma_{0,i} (F_i^t)^T
\]

or more generally,

\[
\Sigma_i^t \propto F_i^t \Sigma_{0,i} (F_i^t)^T
\]

where \(\Sigma_{0,i}\) is the canonical covariance in the rest configuration.

### Opacity
Opacity may also be tied to physical quantities, e.g.,

\[
\alpha_i^t = f(\rho_i^t, \det F_i^t).
\]

This ensures that silhouette gradients flow back into physically meaningful variables such as positions, deformation gradients, and densities.

---

### 4.2 Why Hard Coupling Matters

If rendering parameters such as Gaussian centers, covariances, or opacities are treated as fully independent optimization variables, the optimizer may "cheat" by:

- distorting only the rendering representation
- improving silhouette overlap without changing the physical state appropriately
- producing visually acceptable masks but physically implausible motion

Therefore, true joint optimization requires strong coupling:

\[
g_t = h(x_t).
\]

---

## 5. Loss Function Design

We decompose the total loss into three main parts:

\[
L_{\text{total}} =
\lambda_{\alpha} L_{\alpha}
+
\lambda_{\text{phys}} L_{\text{phys}}
+
\lambda_{\text{reg}} L_{\text{reg}}.
\]

---

## 5.1 Alpha Silhouette Loss

### (a) Binary Cross-Entropy Loss

A standard choice is binary cross-entropy on alpha masks:

\[
L_{\text{BCE}}
=
\sum_{t,c,u}
-
\left[
M_{t,c}^{\text{gt}}(u)\log \hat M_{t,c}(u)
+
(1-M_{t,c}^{\text{gt}}(u))\log \left(1-\hat M_{t,c}(u)\right)
\right]
\]

where \(u\) indexes image pixels.

---

### (b) Soft IoU Loss

Shape overlap can be more directly measured using a soft IoU term:

\[
L_{\text{IoU}}
=
1 -
\frac{\sum \hat M M^{\text{gt}}}
{\sum \hat M + \sum M^{\text{gt}} - \sum \hat M M^{\text{gt}}}.
\]

A Dice variant may also be used:

\[
L_{\text{Dice}}
=
1 -
\frac{2\sum \hat M M^{\text{gt}}}
{\sum \hat M + \sum M^{\text{gt}}}.
\]

---

### (c) Distance Transform / Boundary-Aware Loss

Silhouette gradients can become weak when masks are far apart. A distance-transform-based loss helps provide a more informative gradient field:

\[
L_{\text{dt}} = \sum_{t,c,u} \hat M_{t,c}(u)\, D_{t,c}^{\text{gt}}(u)
\]

where \(D_{t,c}^{\text{gt}}(u)\) denotes the distance to the ground-truth silhouette boundary.

This helps especially during early optimization when there is limited overlap.

---

### Recommended Combined Silhouette Loss

A practical combination is:

\[
L_{\alpha}
=
\lambda_{\text{bce}} L_{\text{BCE}}
+
\lambda_{\text{iou}} L_{\text{IoU}}
+
\lambda_{\text{dt}} L_{\text{dt}}.
\]

---

## 5.2 Physics Consistency and Regularization

Silhouette matching alone does not guarantee physical plausibility. We therefore include physics-related losses or regularizers.

### (a) Dynamics Consistency

If needed, one may include an explicit dynamics residual:

\[
L_{\text{dyn}}
=
\sum_t \|x_{t+1} - \mathcal{P}(x_t;\theta)\|^2.
\]

If the simulator itself is already differentiable and trusted, this term may be less critical.

---

### (b) Constitutive Consistency

For continuum materials, one may enforce stress-strain consistency:

\[
L_{\text{const}}
=
\sum_{i,t}
\|\sigma_i^t - \sigma(F_i^t)\|^2.
\]

---

### (c) Volume Preservation

To prevent physically implausible volume changes:

\[
L_{\text{vol}} = \sum_t \left( \mathrm{Vol}(x_t)-\mathrm{Vol}_0 \right)^2.
\]

---

### (d) Deformation Regularization

To discourage extreme local stretching or compression, we may penalize singular values of deformation gradients:

\[
L_F = \sum_{i,t} \|\log \Sigma(F_i^t)\|^2
\]

where \(\Sigma(F_i^t)\) denotes the vector of singular values of \(F_i^t\).

---

### (e) Temporal Smoothness

To regularize unrealistic temporal jitter:

\[
L_{\text{temp}} = \sum_t \|x_{t+1} - 2x_t + x_{t-1}\|^2.
\]

---

### Example Total Loss

A practical objective may be:

\[
L_{\text{total}} =
\lambda_{\text{BCE}}L_{\text{BCE}}
+
\lambda_{\text{IoU}}L_{\text{IoU}}
+
\lambda_{\text{dt}}L_{\text{dt}}
+
\lambda_{\text{vol}}L_{\text{vol}}
+
\lambda_F L_F
+
\lambda_{\text{temp}}L_{\text{temp}}
+
\lambda_{\text{ctrl}}L_{\text{ctrl}}
\]

where \(L_{\text{ctrl}}\) regularizes control magnitudes when control variables are optimized.

---

## 6. Gradient Flow

The key gradient path for silhouette supervision is:

\[
\frac{\partial L_{\alpha}}{\partial \theta}
=
\sum_{t,c}
\frac{\partial L_{\alpha}}{\partial \hat M_{t,c}}
\frac{\partial \hat M_{t,c}}{\partial g_t}
\frac{\partial g_t}{\partial x_t}
\frac{\partial x_t}{\partial \theta}.
\]

This means:

1. the silhouette loss produces gradients in image space,
2. the differentiable renderer maps them to the rendering proxy,
3. the rendering bridge maps them to the physical state,
4. the simulator backpropagates them to the optimized parameters \(\theta\).

If any link in this chain is broken or too weak, joint optimization will fail.

---

## 7. Why Naive Joint Optimization Often Fails

### 7.1 Silhouette Gradients Are Highly Local

Alpha silhouette losses mainly act near object boundaries. This often causes the optimizer to:

- align only boundaries
- ignore internal mass distribution
- produce unrealistic stretching
- inflate thin structures into blobs

Thus, image-space supervision alone is insufficient.

---

### 7.2 Scale Mismatch Between Physics and Rendering Gradients

Typically:

- rendering gradients are sharp, local, and noisy
- physics gradients are smoother, more global, but sometimes weaker or stiffer

Naively summing them may cause one term to dominate the other.

---

### 7.3 Early Optimization Is Especially Fragile

If silhouette loss is applied too strongly from the start, the optimizer may converge to bad local minima by exploiting easy but non-physical solutions, such as:

- collapsing shape volume
- inflating covariance
- overly flattening geometry
- distorting only rendering parameters

Therefore, optimization schedules matter.

---

## 8. Recommended Optimization Schedule

A staged strategy is much more stable than applying all losses equally from the beginning.

### Stage 1: Physics-First Warm-Up

Use:

\[
L = L_{\text{phys}} + \epsilon L_{\alpha}
\]

with very small \(\epsilon\).

Goal:

- stabilize the simulation
- obtain reasonable coarse motion
- ensure the rendering bridge is functioning

---

### Stage 2: Joint Optimization

Gradually increase silhouette weight:

\[
\lambda_{\alpha}(t) \uparrow
\]

Goal:

- refine image-space alignment
- improve boundary matching
- preserve physical plausibility through existing regularizers

---

### Stage 3: Fine-Tuning

Late in optimization, silhouette terms may be weighted more strongly, but physical regularization should remain active.

---

### Example Schedule

| Stage | \(\lambda_{\alpha}\) | \(\lambda_{\text{phys}}\) | Description |
|---|---:|---:|---|
| Early | 0.01 | 1.0 | physics warm-up |
| Middle | 0.1 ~ 0.5 | 1.0 | joint fitting |
| Late | 0.5 ~ 1.0 | 0.5 ~ 1.0 | silhouette refinement |

These values are task-dependent and should be tuned.

---

## 9. MPM + Gaussian Splatting Instantiation

For an MPM-based simulator, let the state be:

\[
x_t = \{p_i^t, v_i^t, F_i^t, C_i^t\}_{i=1}^N.
\]

We then define a rendering bridge that converts MPM states into Gaussian splats.

### Gaussian Mean
\[
\mu_i^t = p_i^t
\]

### Gaussian Covariance
A physically grounded choice is:

\[
\Sigma_i^t = F_i^t \Sigma_{0,i} (F_i^t)^T.
\]

In practice, this may need stabilization. A more robust parameterization is:

\[
\Sigma_i^t = R_i^t \, \mathrm{diag}(s_i^t)\, (R_i^t)^T
\]

with clipped scales:

\[
s_{\min} \le s_i^t \le s_{\max}.
\]

---

### Gaussian Opacity

Opacity may be fixed or adjusted according to deformation/density:

\[
o_i^t = o_0 \cdot f(\det F_i^t).
\]

Too much freedom here may again allow rendering-only cheating.

---

### Alpha Rendering

The differentiable renderer then produces:

\[
\hat M_{t,c} = \mathcal{R}_{\alpha}(\{\mu_i^t,\Sigma_i^t,o_i^t\}_{i=1}^N; \Pi_c).
\]

---

## 10. Spatially Weighted Silhouette Supervision

In shape morphing or inverse simulation, source and target silhouettes may have little overlap early on. In that case, full-image silhouette loss may produce noisy or unhelpful gradients.

A better strategy is to weight silhouette supervision spatially.

### Pixel-Space Weighting

\[
L_{\alpha}
=
\sum_{t,c,u}
w_{t,c}(u)\,
\ell(\hat M_{t,c}(u), M_{t,c}^{\text{gt}}(u))
\]

where \(w_{t,c}(u)\) may be chosen such that:

- larger near target boundaries
- larger in overlap regions
- smaller in uncertain interior regions
- smaller in distant background regions

---

### Particle-Space Weighting

One may also define weights on particles or Gaussians, emphasizing those whose projections lie near target boundaries or likely correspondence zones:

\[
L_{\alpha} = \sum_i w_i L_{\alpha,i}.
\]

This supports the idea:

> Use silhouette supervision where it is informative, and rely more heavily on physics where overlap is weak or ambiguous.

---

## 11. Practical Pseudo-Code

```python
theta = init_parameters()
optimizer = Adam([theta], lr=lr)

for it in range(num_iters):
    # 1. differentiable simulation
    states = run_mpm(theta)   # states[t] contains p, v, F, ...

    # 2. build rendering proxy from physics states
    gaussians = []
    for t in range(T):
        g_t = build_gaussians_from_mpm_state(states[t])
        gaussians.append(g_t)

    # 3. render alpha masks
    alpha_preds = []
    for t in range(T):
        pred_views = []
        for cam in cameras:
            alpha = render_alpha(gaussians[t], cam)
            pred_views.append(alpha)
        alpha_preds.append(pred_views)

    # 4. silhouette losses
    loss_bce = 0.0
    loss_iou = 0.0
    loss_dt  = 0.0
    for t in range(T):
        for c in range(num_cams):
            pred = alpha_preds[t][c]
            gt   = alpha_gt[t][c]

            loss_bce += bce_loss(pred, gt)
            loss_iou += soft_iou_loss(pred, gt)
            loss_dt  += distance_transform_loss(pred, gt)

    # 5. physics / regularization losses
    loss_vol  = volume_preservation_loss(states)
    loss_def  = deformation_penalty(states)
    loss_temp = temporal_smoothness_loss(states)

    # 6. scheduled weights
    w_alpha = alpha_schedule(it, num_iters)

    loss = (
        w_alpha * (
            lambda_bce * loss_bce +
            lambda_iou * loss_iou +
            lambda_dt  * loss_dt
        )
        + lambda_vol  * loss_vol
        + lambda_def  * loss_def
        + lambda_temp * loss_temp
    )

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_([theta], 1.0)
    optimizer.step()