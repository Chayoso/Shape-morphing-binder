# Passive material surface for render-guided MPM

Status: experimental implementation, 2026-09-14. This is an explicit observation
variant; the production server tree and earlier experiment snapshots are retained.
No 20% quality result or no-floater deliverable is established by this document.

## Why change the observation

Development v1-v5 used one Gaussian per selected volume-shell particle. Both
transported material covariances and fixed isotropic kernels failed the registered
sphere/bunny quality criterion. Inspection of v4's native 1024px endpoint revealed
floating-looking lobes and incomplete ears. Even its sparse target image contains
an isolated-looking dot. Higher image resolution alone does not repair sampling
or enforce a continuous material boundary.

## Physical dependency

Weld the original source sphere's OBJ normal/UV seams into a closed, oriented
physical mesh. Subdivide it three times **before simulation**. Each new vertex is an
independent passive material marker; it is advected every substep, not interpolated
from the original coarse hull after a rollout. The topology stays fixed.

For a marker q, use the same cubic B-spline velocity map as an MPM mass particle:

\[
u(q_t)=\sum_i w_i(q_t)v_i^{grid},\qquad
q_{t+1}=q_t+\Delta t\,\operatorname{cap}(u(q_t)).
\]

The marker's cumulative geometric gradient obeys
\(F^s_{t+1}=(I+\Delta t\,D_q\operatorname{cap}(u))F^s_t\), including the exact
speed-cap derivative. Markers add no mass, momentum, stress, or control variables.
They read the grid produced by the same controlled MPM body.

\[
dFc_{surface}\to stress\to grid\ velocity\to q_t\to
triangle\ Gaussians\to L_{image}.
\]

The Warp tape includes marker transport. The Torch geometry bridge optionally
returns six outputs: x, F_model, v, F_geom, surface_x, surface_F. The backward pass
seeds both marker outputs and pulls their covectors through the MPM grid. The
reduced response optimizer also measures this entire physical path. There is no
independent correction to marker positions or Gaussian parameters. Interior dFc
remains zero in this reduced experiment; interior mass particles still move.

## Surface quadrature

Each current triangle contributes a Gaussian with its centroid and vertex
population covariance, plus fixed .001 world-unit thickness along its normal.
The population covariance is intentionally **four times** the covariance of a
uniform triangular lamina; it is an overlapping rendering footprint, not a fitted
material moment. Color .35 and opacity .9 are fixed. Loss and display share this
representation. Degenerate triangles cannot be approved merely because normal
normalization uses a numerical denominator floor.

Mesh-face parameterization is related to
[GaMeS (2024), section 3](https://arxiv.org/html/2402.01459v3).
Unlike that paper's trainable barycentric and scale parameters, this experiment
fixes every Gaussian parameter except its dependence on the advected vertices.
It is not a reproduction of GaMeS or a claim that attaching centers alone proves
physical or visual correctness.

The target is a derived, repaired bunny mesh; the original OBJ remains unchanged.
Direct interior rejection sampling supplies mass particles and a deterministic
interior grid supplies dense 3-D target quadrature. Both objective arms receive
that same geometry and dense volume. All target representations receive one
translation set by the dense mass quadrature's center, matching the source
particle center. The random evaluation sample is not independently recentered.
Final transformed float32 surface arrays are checked before saving the fixture.
The raw particle-quality metric and held-out view protocol remain
unchanged. This changes the representation and target quadrature relative to v1-v5;
comparisons across these families are development observations, not a single-factor
causal ablation. Within a pair, fixture arrays and implementation hashes must match.

## Checks and remaining limits

CPU tests verify unchanged mass-particle dynamics with passive markers, coincident
marker/particle trajectory equality with and without a speed cap, marker AD against
finite differences, cumulative restart state, and zero marker-control derivative
when stiffness is zero. Separate tests check initial closed connectivity, triangle
Gaussian rigid-motion behavior and vertex derivatives.

Every candidate retains the original MPM health checks. Markers additionally need
finite states, complete grid stencils, positive geometric determinants, condition
number <=20, and grid-sampled mass density >=10% of their own initial density.
Final-frame support is resampled from a mass grid rebuilt at final MPM positions.
Mesh faces need area >=1% of original area, edge length <=4 times original length,
and agreement of their normal orientation with locally transported tangents.
The condition response in the QP includes both mass-particle and marker geometry.
Actual nonlinear trajectory checks remain authoritative.

Initial, accepted-candidate, and final replay trajectories also run Open3D's
global intersection query. A float64 separating-axis check removes a candidate
only when a positive separation is independently certified; contact, degeneracy,
and numerically uncertain gaps remain rejected. This was needed because the v8
zero-control rollout produced nearly coplanar false positives after a maximum
marker displacement of 1.83e-7 at substep 15 (N=12000, dx=.25, dt=1/120, T=64).
The added test does not prove completeness of Open3D's candidate generation.

These checks do **not** prove global injectivity, mesh/particle containment,
or absence of unsupported bridges at finite resolution. An offline all-state
audit additionally measures particle containment; per-frame visual inspection remains required
before a finished animation or no-floater claim. Original marker positions and
density references persist across restarts. Expected face normals use cumulative
geometric gradients without inverting an average of rotations; a degenerate
expected normal is explicitly rejected. CPU regressions cover final-only loss of
support, cumulative reference bounds, and a singular average of proper rotations.

## Input geometry findings and retained development failures

The v6/v7 preparation used a subdivided sampled-point hull and an axis-filled
voxel target. At the original bunny pitch max-extent/110, 256499 occupied voxels
enclosed 13518 empty voxels. Sealing those voids still left artificial exterior
bridges and tunnels (Euler -158). Native target inspection showed straight bars
between the ears and body. This target is unsuitable for a bunny-quality claim.

The replacement uses PyMeshFix 0.18.1, installed in an isolated output directory
without replacing environment dependencies. Repair explicitly preserves components
and disables component joining. Its warning that it could not fix everything is
retained in the log. Independent inspection of the actual repaired float32 asset
found one watertight, consistently oriented component, Euler 2, 70372 triangles,
positive triangle areas, and zero reported self-intersections. This validation,
and subsequent final-coordinate validation, is the acceptance evidence; the
repair routine's return is not itself a certificate. See
[PyMeshFix's API](https://pymeshfix.pyvista.org/api.html).

The source mesh has 2562 vertices / 5120 faces after welding and refinement.
Both closed meshes are sampled directly without axis filling or voxel jitter
outside their boundary. N=12000, dx=.25, dt=1/120, T=64, lambda=5000, mu=3000,
source volume=3.8406127884, particle mass=.3200510657, and a 32^3 mass grid at
dx=.25 characterize the v8/v9 family. v8 failed its initial intersection gate
and also had a measured dense-target COM offset of about .00818. v9 corrects
the common target translation and diagnoses the intersection false positives.
Neither input repair nor an optimizer smoke establishes the 20% quality claim.

## Surface support in the response subproblem

v9's third accepted updates approached the existing 10% mass-support bound:
image minimum ratio .101412, physics .107000, with trust radii .0006 and .0024.
These are development diagnostics at the discretization above, not final scores.
The optional v10 response model adds a row for **each marker and each time**:

`c[t,p] = 1 + minimum_ratio - rho[t,p]/rho_original[p] <= 1`.

It uses the same actual support boundary as trajectory health. Separate time
rows avoid the derivative cancellation caused by taking a nonsmooth time minimum
before central finite differences. The QP can therefore propose another physical
control direction while preserving modeled support. Finite probes may cross this
bound; actual candidates and final replay must still satisfy the unchanged 10%
threshold. No position correction, density clamp, or surface-only forward solve
is added. The model is still local; actual nonlinear acceptance remains required.
Rejected candidates now retain their health reasons and attempted radii.
