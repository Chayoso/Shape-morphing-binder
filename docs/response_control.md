# Surface response control (2026-09-14)

The next control milestone uses an image objective constrained by 3-D target
matching and the existing physical regularizers. It is implemented in
`physmorph/pipeline/response_control.py`; the server experiment entry point is
`scripts/response_pipeline.py`. The audited MPM and F_geom path is unchanged.
This is a new constrained formulation, not a preconditioner claimed to preserve
the previous weighted-sum objective.

## Why this addresses the gradient imbalance

`D_vol` is mass matching to a target, not a measure of whether Newton's laws were
obeyed. Every candidate already satisfies the implemented discrete MPM update.
Adding an image loss to an unnormalized volumetric sum makes their units and
reductions decide the optimization priority. A larger image multiplier cannot
tell us which physical actuation can actually produce the desired image change.

We instead measure that response in a small physical control space. With shell
basis B, coefficients z and fixed initial state s0, the forward dependency is:

```text
z -> surface dFc = Bz -> constitutive stress -> grid momentum -> particle motion
  -> (x, F_geom) -> fixed material Gaussians -> surface image residual
```

For each residual r, a local approximation is r(z + delta) = r(z) + J delta.
The trust-region subproblem is:

```text
minimize    0.5 ||r_image + J_image delta||^2
subject to  0.5 ||r_mass  + J_mass  delta||^2 <= L_mass(z)
            0.5 ||r_core  + J_core  delta||^2 <= L_core(z)
            ||delta|| <= radius
            ||z + delta|| <= max_control_rms
```

This residual linearization and comparison of actual versus predicted reduction
follow the standard trust-region construction described in the
[Ceres nonlinear least-squares documentation](https://ceres-solver.readthedocs.io/latest/nnls_solving.html).
Our implementation uses the small dense Gram matrices J^T J and
[SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)
for the constrained subproblem, rather than Ceres. Positive scalar normalization
of the objective and inequalities improves numerical units; it does not introduce
an image/physics tradeoff coefficient. Numerical floors and stopping tolerances
still limit exact scale invariance near zero.

The local model only proposes a control. We then run one complete MPM candidate,
evaluate all losses together, and require actual objective reduction, mass/core
nonincrease within recorded numerical tolerance, and an actual/predicted gain
ratio of at least 0.1. A rejected candidate shrinks the trust radius. All substeps
must pass finite-state, grid-support, orientation and geometric-conditioning
checks. No physics-then-image alternating state commits are made. If no local
feasible descent exists, the method stops; neither PCGrad nor a larger multiplier
can create missing physical controllability.

## Meaning of the controls and residuals

Six symmetric modes cover constant xx, yy, zz, xy, xz and yz actuation on a fixed
material shell. The optional linear spatial basis has 24 modes. The shell mask
has a hard boundary: the field is coherent within the shell, not globally smooth.
Whitening uses mean surface Frobenius squared, so ||z|| is RMS dFc per particle
and per substep. Total RMS is bounded by 0.3, and actual maximum per-particle
Frobenius control and increment are checked against 0.4. These are declared
control bounds, not a claim that all such controls are materially valid.

The same Bz is injected on **every substep**. Increasing T changes elapsed time
and accumulated constitutive-control injection; it cannot be described as the
same actuation with a longer observation horizon. The geometric deformation is
still transported by the actual spatial velocity derivative, never by directly
adding dFc. See `geometric_control_contract.md` for the F_model/F_geom distinction.

This reduced diagnostic fixes interior dFc to zero in both arms, while retaining
full interior MPM motion. It isolates surface actuation and is narrower than the
eventual surface-image/interior-3D optimization requested by the user. It does not
replace that full control parameterization or the existing C++ baseline.

Residual definitions preserve the earlier physical terms:

```text
r_mass = log(1 + mass_grid(x_T)) - log(1 + target_mass_grid)
L_mass = 0.5 sum(r_mass^2)
L_core = L_mass + 0.5 mean_p |v_T,p|^2 + 0.001 sum(dFc^2)/(N T)
L_image = 0.5 mean_{camera,pixel,channel} (RGB - target_RGB)^2
```

Only shell particles are rendered. Opacity, color, primitive count and material
attachment are fixed; no free Gaussian translation or covariance optimization
can satisfy the image loss independently of MPM. Source and target cameras are
identical. A sphere enclosing source/target positions plus three rest Gaussian
standard deviations and a 10% margin avoids the cropped target found during
visual QA. This framing is not a guarantee for arbitrary later deformations.

The target is translated to the source center of mass before either experiment;
pure internal stress cannot translate a free body's center of mass. The source
state itself is unchanged. Both arms reuse byte-identical source vol0, target,
shell masks and basis from the same saved reference.

## Derivative status and cost

The new controller uses central finite differences of the full MPM/native-render
residual response in these six modes, initially at +/-0.005 RMS dFc. It is a
numerical reduced Jacobian, **not exact AD Gauss-Newton**. The audited autograd
bridge remains available, but this experiment does not assume a working JVP or
second backward through that bridge. It does not validate visibility-boundary
derivatives merely by producing a successful update.

One fixed mixed direction at half the probe size checks the residual prediction;
20% relative error is the configured rejection threshold. This checks one
direction, not the whole Jacobian. Image reliability gates the image arm only;
the physics objective depends on mass/core reliability. Both arms still evaluate
images for comparison and their costs are counted. Image evaluation errors can
therefore still fail this diagnostic script. The final nonlinear candidate
checks remain necessary even when the direction check passes.

Observed scalar variation in two initial replays sets numerical acceptance
tolerances; these are not statistical bounds. Final replay compares x, v, C,
F_model and F_geom as well as scalar losses. Gram eigenvalues are logged, and
there is no unbounded inverse of a weak response mode. A finite trust region does
not establish that all numerical columns are accurate.

## Matched server experiment

Results below use the final v3 source snapshot on hyde06 GPU 0. Discretization:
N=2048, one T=32 window, dt=1/120 (duration 0.266667), dx=0.5, 16^3 MPM grid,
grid_min=(-4,-4,-4), mass=1 per particle, Lame lambda=800 and mu=400, smoothing
0.955, drag=0.9, no external force or active floor/speed cap. Mass observation:
24^3 grid, dx=1/3. There are 819 shell particles, six constant symmetric modes,
four image cameras at **1024px**, 1024px frame exports, one Gaussian per shell
particle, sigma0=0.07467026946, opacity=0.9 and fixed gray color.

Both arms start at zero control with the same basis, horizon, constraints, bounds
and six-iteration limit. The image arm minimizes L_image; the comparison minimizes
L_core. Thus their objective-specific acceptance differs, and actual rollout
counts can differ. This is not a comparison with the old T=8 full-field RMS path.

| Quantity | Initial | Image criterion | 3-D/core criterion |
|---|---:|---:|---:|
| Image objective | 0.006213454 | 0.0009939612 | 0.001001196 |
| Mass objective | 21.489371 | 5.513704 | 5.493586 |
| Core objective | 21.489371 | 5.592459 | 5.574548 |
| Raw Chamfer, world units | 0.06447826 | 0.04349378 | 0.04342726 |
| Maximum particle displacement, world units | 0 | 0.1029619 | 0.1053430 |
| Accepted updates | - | 6 | 6 |
| Rollout evaluations | - | 87 | 87 |
| Optimization seconds | - | 5.045 | 5.531 |

The image criterion reduces its objective by about 84.0% from zero control, but
its advantage over the matched 3-D/core criterion is only about 0.72%. The 3-D/core
criterion has slightly better raw Chamfer. This is evidence of image-directed
physical actuation, not a strong render-guidance quality win. Different numerical
optimization paths and limited iteration budgets also limit that interpretation.
Shared source/target/vol0/mask/basis arrays and source provenance were verified
byte-identical; scalar reductions still exhibit small CUDA replay variation.

Interior dFc is exactly zero in both runs. For the image criterion, all-step
minimum det(F_geom) is 0.909406, maximum geometric condition number is 1.244965,
and maximum center-of-mass drift is 8.04e-8 world units. Its one-direction image
response check is 15.23-16.65% relative error: acceptance of useful
candidates must not be mistaken for an exact native-raster Jacobian audit.

Loss values are optimization diagnostics. Reported geometric metrics use raw
simulation positions and never use the renderer. Chamfer here is the mean of
the two mean nearest-neighbor distances, in world units, not squared distance.
Timing runs from optimizer entry through final replay, including response
estimation, rollout checks and any first-use module loading/JIT inside them.
It excludes CLI setup and final frame export, and does not establish production
real-time performance.

## Validation, artifacts and limits

The CPU suite passed 153 tests with three existing warnings. The seven new tests
cover physical basis units/support, feasible image correction, loss-unit
rescaling, a genuinely conflicting one-dimensional control, real CPU MPM motion
with zero interior actuation, and a physics-only correction that must survive an
unreliable synthetic image response. Adversarial review caught and prompted the
last comparison-gate fix.

Every one of the 33 exported physical substep frames was visually inspected in
the saved camera, with additional 1024px inspection of the endpoint and target.
No detached splat, white through-hole or crossfade ghost was observed; silhouette
continuity is maintained. The corrected target has visible camera margin. The
surface remains flat gray with a blurred boundary, texture transport is not
assessed, and final morph quality is not approved. This is one-camera diagnostic
QA, not a universal no-floater guarantee. `qa_manifest.json` records that scope.

Artifacts are in `output/response_control_v3/{image_t32,physics_t32}` locally and
`~/physmorph_v2/output/response_control_v3_20260914/` on hyde06. Each run records
script, Python source, raster-binary and reference-archive hashes. Earlier v1/v2
results and source snapshots are preserved, but their 512px/camera conditions
are not used as the final matched comparison. The main server source tree was
not overwritten.

This is a sphere-to-ellipsoid control fixture, not a finished textured bunny
morph. The RMS bound becomes active, so residual error must not be described as
full convergence. Native raster finite-step response, temporal control semantics,
constitutive consistency, discretization dependence, denser surface coverage and
large-deformation attachment remain separate open questions. VBD/Chebyshev is not
implemented here: changing the mechanical solve would require validating its
response derivatives against the same coupled objective first. A next expansion
should add spatial/time control modes and compare at fixed effort and horizon,
rather than raise the image gradient gain or control bound without evidence.

To reproduce from the matching isolated source directory on hyde06:

```bash
env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python scripts/response_pipeline.py \
  --reference ../geometric_pipeline_20260914/smoke_v1 --out NEW_IMAGE_DIRECTORY \
  --steps 32 --iterations 6 --frames
```

Use a fresh directory and `--objective physics` for the matched 3-D criterion.
The existing `scripts/geometric_pipeline.py --replay RESULT_DIRECTORY --port 8774`
serves saved physical frames without repeating the optimization.

The current local monitor is http://127.0.0.1:8774 and replays the v3 image
criterion's 33 physical frames at 0.2 seconds per displayed frame. The physical
time is still 32/120 seconds; playback is deliberately slower. An atomic HTTP
snapshot was decoded as 1024x1024 and verified pixel-identical to its indexed
saved frame. The server viewer PID at handoff is 775338; the existing local SSH
tunnel remains in use. See `output/response_control_v3/image_t32/viewer_verification.json`.
