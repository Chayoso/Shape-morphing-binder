# Geometric dFc control contract — 2026-09-14

Status: approved design, implemented gradient-flow milestone. This does not certify
a finished sphere-to-bunny morph, absence of floaters, or the native raster adjoint.
The prior experimental optimizer edits were reverted before this implementation.

## State semantics

The original `Trajectory.F` is retained as **F_model**, the constitutive/control
state. It is not the spatial deformation gradient of the actual particle motion:
its recurrence includes dFc and the legacy smoothing factor. Its stress remains
`Fe = (F_model + dFc) @ inverse(Fp)`. Retaining this recurrence preserves the
baseline forward solver; it does not prove that its constitutive evolution is
kinematically consistent. That remains a separate physical-model decision.

**F_geom** carries the original source reference into the current geometric state:

```
u(x_p) = sum_i w_i(x_p) v_i
G_p = sum_i outer(v_i, grad_x_p w_i(x_p))
F_geom_next = (I + dt G_p) F_geom
x_next = x + dt u(x)
```

For the speed-capped map, `G` is premultiplied by the cap Jacobian
`v_max/|u| * (I - u_hat u_hat^T)` on the active branch. Grid contact is already
included in the grid velocities. The derivative remains piecewise smooth at
contact/cap changes. The geometry path uses `Trajectory`; `mpm_step` has an extra
particle floor projection and is not an interchangeable replay implementation.

This is the material gradient obtained by composing the reconstructed discrete
velocity fields, holding each solved grid field fixed for its spatial derivative.
It is not the Jacobian of the entire particle simulation with respect to moving
one initial particle independently (that perturbation changes the solved field).
Its **control adjoint**, however, does differentiate those grid fields and the
position dependence of their interpolation weights.

F_geom starts at I only at the original source. Restart requires explicit
cumulative F_geom0 and the original vol0. It is never smoothed, plastic-assimilated,
conditioned, or incremented directly by dFc. Constitutive F and geometric F are
archived under distinct names. New replay/commit code carries x/v/C/F_model/F_geom
together and checks all substeps against the accepted rollout.

## Image observation and control path

```
dFc -> constitutive stress -> P2G -> grid velocity -> {x, F_geom}
                                                 -> Gaussian rendering -> image loss
```

Surface Gaussian means are `x_parent + F_geom @ frozen_rest_offset`; covariances
are `F_geom @ Sigma0 @ F_geom.T`. Both endpoint covectors participate in the
control adjoint. No independently optimized centers, covariance, opacity or
child offsets exist in this path. Opacity remains 0.9. Target and source images
use the same raster API and covariance path. Explicit resolutions above 384 are
now respected; the driver defaults to 512 loss / 1024 display pixels.

The surface set is a boolean material mask frozen from the source; the initial
boundary estimator is approximate, not a topological surface extraction. Target
observations have their own frozen target mask. All particles participate in
mass matching and MPM. Images contain only the surface set.

Endpoint masking is insufficient: an observed surface particle depends on
interior controls through MPM. Therefore the **complete** image adjoint is pulled
back first, then its dFc contribution is masked in control space:

```
g_direction = g_physics + lambda * surface_mask * g_render
g_objective = g_physics + lambda * g_render
```

Interior dFc receives only the physical/3D gradient contribution. Its motion still
responds to surface stress through the shared MPM grid. The global line search
and accepted time step depend on the joint objective; an interior update is not
claimed to be invariant to the presence of image supervision.

## One joint update

`pipeline/geometric.py` uses one rollout, one candidate dFc sequence, and one
joint scalar decision. An RMS diagonal preconditioner acts on the combined
direction. There is no alternating optimization, PCGrad projection, separate
appearance optimization, or render/physics norm-ratio amplification.

Because the surface restriction changes the search direction, Armijo uses the
unmasked implemented AD derivative of the actual joint scalar dotted with the
actual candidate direction. This derivative is mathematically exact only to the
extent the participating operators' adjoints pass their individual checks.
A non-descent restricted direction stops and reports the conflict. It does not
silently fall back to a physics-only update. Failed candidates leave the last
accepted control and state intact. Every candidate and replay checks finite
states, full grid stencils, positive constitutive/geometric determinants and
geometric conditioning at every substep. These checks do not certify solidness.

The diagnostic driver keeps the original **sum** mass loss, terminal mean kinetic
loss, and original control cost. Lambda is fixed at 1 (0 for the identical
physics-only path). L2 image residual is the initial smooth-loss diagnostic;
L1 is selectable for the separate derivative comparison. No measured gradient
ratio is used to choose weights. A remaining norm gap is reported, not claimed
to have been solved by this change. Objective units/discretization and a better
joint preconditioner must be studied after the gradient gates.

## Verification and execution

CPU tests in `tests/test_geometric_deformation.py` and `test_geometric_joint.py`
cover affine/translation maps, active cap spatial and control derivatives,
dt=0/zero-stiffness control invariance, x and F_geom finite differences, material
bridge compatibility, repeated backward, cumulative restart, control masking,
stress transmission into the interior, joint descent and rejection, and replay
state corruption invisible to the scalar loss. CPU gates are not a native
Gaussian-raster correctness claim.

`scripts/probes/audit_geometric_control.py` runs on hyde06 and records exact source
and raster extension hashes. It compares full image-loss differences, frozen
pixel-covector differences, and frozen endpoint-covector MPM differences. L1 and
L2 are both measured. No branch is declared correct solely because the end-to-end
loss decreases.

Server commands from the isolated staged tree:

```bash
env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python \
  scripts/probes/audit_geometric_control.py --res 512 --out native512.json

env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python \
  scripts/geometric_pipeline.py --out smoke --port 8774
```

The live page is a diagnostic monitor. During optimization it explicitly labels
terminal candidate previews; afterward it replays the physical substep frames.
All substeps are saved at display resolution for per-frame visual review.
No opacity fading, target clipping, or particle deletion hides defects.

Local viewing (no local simulation):

```powershell
ssh -N -L 8774:127.0.0.1:8774 chayo@hyde06.dabh.io
```

Open `http://127.0.0.1:8774` while the tunnel is active. Training and the native
raster run on the server. A diagnostic viewer is not a final morph deliverable.

## Deferred gates

Native renderer derivative accuracy, constitutive F_model consistency, material
surface attachment under large deformation, connectivity/holes/floaters, shared
physical conditioning, and grid/substep convergence remain explicit gates.
VBD/Chebyshev and large-scale long-horizon optimization follow these gates;
neither is used to bypass a wrong derivative or to edit surface positions.

## Measured milestone

The full CPU suite passes 146 tests. The following native CUDA numbers have
different scopes and must not be conflated:

| Check | Discretization | Observed relative FD error |
|---|---|---|
| Frozen x/F_geom covectors through MPM | N=256, T=3, dt=1/120, dx=.75, 16^3 grid, lambda=800, mu=400; 512px, 4 cameras, sigma0=.16, one splat/parent | maximum 0.00995% |
| Full L2 image loss through MPM | same state, epsilon=.003/.01/.03 | 6.09% / 2.51% / 1.44% |
| Full L1 image loss through MPM | same state and epsilons | 9.51% / 4.75% / 3.30% |
| Native raster, single splat interior ROI | N=1, 512px, sigma0=.16, opacity=.9; center 32x32 pixels, minimum contributing alpha=.403 | maximum 0.0155% across x/F_geom |
| Native compositing, three splats interior ROI | N=3, otherwise same ROI; depth order separated, minimum remaining transmittance=.002286 | maximum 0.0244% across x/F_geom |

These ROI tests support the implemented mean/covariance derivatives inside smooth
raster branches. They do not certify active visibility/alpha transitions in full
images. Full-scene frozen-pixel-covector tests still disagree with the infinitesimal
AD slope at the tested finite steps, so L1 sign crossings alone cannot explain the
full-scene gap. Branch transitions are a remaining hypothesis, not a proven sole
cause or a reason to declare a CUDA backward bug.

Both dt=0 and zero-stiffness native tests give exactly zero displacement, zero
F_geom change and zero image-to-control gradient. The full state includes F_model,
which may change in those cases; it no longer changes image geometry directly.

The joint smoke uses **N=2048, two windows of T=8, dt=1/120, dx=.5, 16^3 grid,
lambda=800, mu=400, loss grid 24^3 with dx=1/3, four 512px cameras, one splat/parent,
1024px display**. All 12 proposed updates were accepted by the same joint scalar;
the masked interior render-control norm was exactly zero. Raw-state maximum
displacement is .0045698 world units, and minimum cumulative det(F_geom)=.9777214.
The raw physical/render L2 gradient norm ratio is still **786.6–831.1**. This is
evidence of a functioning path, not evidence that conditioning or morph quality
has been solved. No norm-ratio gain was applied.

All 17 physical substep frames were visually inspected in the saved camera. No
detached splat or white through-hole was observed in this small test; the silhouette
is continuous. The image is flat gray with a blurred splat boundary, displacement
is small, and texture transport is not assessed. Final morph quality is **not
approved**. A 1024px buffer by itself is not a claim of detailed surface quality.

Artifacts: `output/geometric_native512.json`,
`output/geometric_smooth_raster512.json`,
`output/geometric_smooth_raster_multi512.json`, and
`output/geometric_smoke_v1/{metadata,history,qa_manifest}.json` with `trajectory.npz`
and every PNG frame. Each native probe records its executed script hash and raster
binary hash. The single-splat probe revision is preserved as
`scripts/probes/audit_raster_smooth_branch.py` in the isolated server audit tree;
the subsequent three-splat revision is preserved there as
`scripts/probes/audit_raster_smooth_branch_multi.py`. Later local docstring changes
do not change these retained, at-run script versions.

The implemented entry point is **`scripts/geometric_pipeline.py`**, with the joint
solver in `physmorph/pipeline/geometric.py`. Existing `pipeline_run.py` experimental
arms retain their original semantics and are not the new F_geom path.
The verified server source snapshot is
`~/physmorph_v2/output/geometric_verified_20260914`; the completed smoke and all PNGs
are in `~/physmorph_v2/output/geometric_pipeline_20260914/smoke_v1`. The original
server production source remains at its restored pre-task state.

At this audit milestone, the 8774 monitor served this saved trajectory through the corrected atomic
snapshot protocol. A local HTTP check decoded a 1024x1024 PNG and verified bytewise
pixel equality to the saved PNG for the frame number in that same snapshot.
The local SSH tunnel PID is recorded in `output/geometric_tunnel_20260914.pid`.
The subsequent surface-response experiment now occupies this same monitor;
see `response_control.md` for its current result and constraints.
To serve an existing run without repeating simulation:

```bash
/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python scripts/geometric_pipeline.py \
  --replay /home/chayo/physmorph_v2/output/geometric_pipeline_20260914/smoke_v1 --port 8774
```

## Does the image contribution actually change physical motion?

An additional causal intervention freezes the initial state, source vol0 and
the realized physical/image adjoints **from a single backward evaluation per
loss**. It then toggles only the surface render contribution in the first RMS
update. Both trials use zero initial control and moments. This avoids attributing
differences between separately repeated optimizer backward passes to rendering.
The current replay source hashes are checked against the reference run.

Discretization: N=2048, one window with T=8, dt=1/120, dx=.5, 16^3 grid,
Lamé lambda=800 and mu=400, mass=1/particle, smoothing=.955; 24^3 mass-loss grid
with dx=1/3. The L2 image loss uses 4 cameras at 512px, sigma0=.07467026946,
819 fixed surface particles and one splat/parent. The step size is .003, RMS
epsilon=1e-5, and render objective weight is 1 or 0. Both full-step candidates
pass their respective Armijo decisions and all raw-state validity checks.

The switch changes surface dFc (maximum component difference .000252275), while
the interior dFc difference and interior direct image gradient are **exactly
zero**. Stress, velocity, position and cumulative geometric deformation all
change, including internal particle motion:

| Raw quantity | L2 of render-on minus render-off | Largest L2 difference in two fixed-control repeats per arm |
|---|---:|---:|
| Position x | 6.5529e-6 | 1.4934e-7 |
| Velocity v | 1.4349e-4 | 2.2043e-8 |
| F_geom | 2.0947e-5 | 3.3718e-7 |
| Total PK1 P | 2.1210 | .0017472 |

These norms aggregate time, particles and components, rather than describing a
terminal particle alone. Maximum position component difference is 4.1723e-7 world
units. Interior position-difference L2 is 3.7099e-6 despite identical interior
controls: this is the shared MPM stress/velocity coupling. The two repeats report
observed forward variation, not a statistical noise upper bound.

This confirms an actual image-adjoint-to-physical-motion influence in this
one-step fixture. It does **not** establish practically useful morph guidance,
long-horizon improvement, or native raster derivative accuracy at visibility
transitions. The effect on position here is small.

The initial independently executed two-window on/off comparison was rejected as
a strict causal comparison because recomputing source vol0 on CUDA produced
non-bit-identical volumes. Its files remain preserved; those two runs are not
the basis of the result above.

Reproduce from the matching original smoke source snapshot on hyde06:

```bash
cd ~/physmorph_v2/output/geometric_pipeline_20260914
env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python \
  scripts/probes/compare_render_influence.py --reference smoke_v1 \
  --out render_influence_intervention.json
```

Local result: `output/geometric_render_influence_intervention.json`. The probe
records the source archive and executed script hashes; physical metrics do not
consume rendered images.
