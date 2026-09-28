# CUDA execution boundary (2026-09-26)

The CUDA execution boundary covers numerical particle state, loss setup, nearest-neighbor
queries, commit operators, plastic assimilation, and accepted-state bookkeeping. Torch
owns differentiable leaves; Warp owns MPM trajectories; CuPy owns nondifferentiable
arrays. A Warp-owned stream is shared by all three libraries so graph capture and buffer
handoffs have an explicit ordering. Archive frames transfer to host at the output boundary
instead of accumulating an entire movie in device memory.

`PipelineConfig.compute_backend="legacy"` retains the comparison implementation.
`compute_backend="cuda"` requires CuPy 14+, CUDA 12 NVRTC, and a CUDA device. It must
raise when a required GPU dependency is absent. The existing denoised Poisson shading
reference is an immutable input asset, prepared separately by
`scripts/prepare_pipeline_input.py`; point hashes and reference discretization are checked
when loading it. The exact loaded asset bytes are SHA-256 hashed and recorded under
each arm's `input_assets.target_reference`; file path, schema, target point hash,
reference factor and spacing are included. This preparation still uses CPU reconstruction.
Source/target sampling and initial discretization reports are input preparation on the
host. File I/O, scalar decisions, logging and video/image transport also remain host tasks.
Accepted physics state, loss preparation, state commits, raw quality metrics and the
archive renderer do their numerical array work on CUDA. This is not a claim that mesh
asset preparation or every historical research branch has been ported.

On hyde06, isolated packages live under `/data/relcfd/chayo/physmorph_v2/deps/gpu_pipeline`.
The original Python environment is unchanged. Set `CUDA_HOME` and `CUDA_PATH` to
`/usr/local/cuda-12.8`, put its lib64 first in `LD_LIBRARY_PATH`, and launch with
`scripts/ops/cuda_python.py SCRIPT ...` so CuPy initializes CUDA 12 NVRTC before legacy
packages load their additional CUDA 11 library. The package applies `WARP_CACHE_PATH`
before `wp.init()`; setting the environment variable alone does not configure Warp.
All server caches remain under /data.

Warp1.16 or newer is required by the constitutive custom adjoint. The existing
hyde06 environment already uses1.16.0. Warp1.9's code generator orders nested
custom adjoints incorrectly and cannot compile this path; the package fails
early on older versions. Local CPU validation can use an isolated1.16 install
without changing the machine's global environment. See [P300](constitutive_adjoint_p300.md).

The tensor rasterizer keeps covariance eigendecomposition, quaternion conversion,
camera matrices, raster buffers and deferred shading on CUDA. `render_splat_gpu.py`
encodes using NVENC. PNG/file transport and labels remain host output operations.
The frozen NumPy rasterizer remains a comparison interface.

## Render influence on the existing mixed60 run

Discretization: N=300000, dx=0.3062907544 wu, dt=1/240, T=20, loss grid36 cubed,
39 committed windows. Render loss uses six azimuths and three elevations at64x64;
its silhouette and shading terms are optimized with lambda_auto=0.5 and render-side PCGrad.

| Windows | Median nominal render gradient share | Raw physics/render cosine |
| --- | ---: | ---: |
| 1-6 | 43.7% | 0.302 |
| 7-20 | 43.2% | 0.026 |
| 21-39 | 35.5% | -0.056 |
| All39 | 40.5% | -0.030 |

The share is `lambda*||projected_render||/(||physics_core||+lambda*||projected_render||)`
at the first optimizer iteration of each window. It is neither displacement share nor
a causal improvement estimate. W1 enters after this balancing; normalization, clipping,
RPROP/Adam and line search further transform the update. Late conflict components are
removed by PCGrad. New `render_channels` telemetry records norms/cosines separately for
stress, body, surface_u and material leaves without changing the gradient.

CUDA integration has passed a six-window mixed60 prefix (8 optimizer iterations per window).
The matched render-off prefix changes the final positions by median1.535 native spacings
(p95=3.617); rendering therefore changes the optimization trajectory. This alone does not
establish improved shape quality. Native source spacing is0.0349885366wu. In the first
window the nominal shares per control channel are stress38.82%, body30.23%, surface_u86.16%.
Only4.1% of the outer layer passes the u transport gate initially; that86.16% is not a
whole-object movement share. Do not treat this document as evidence that every optional
research branch has passed CUDA validation.

## Current CUDA compatibility boundary

The active mixed60 path is the migration target. Experimental `reattach`,
`pace_front_geo`, `local_dress_iters`, `use_gauss_loss` and `assim_consensus`
currently fail explicitly in CUDA mode: their separate reconstruction/graph/consensus
contracts have not passed CUDA gates. They are not silently sent to CPU or disabled.
The CLI also rejects `--live_port` and `--live_dir` in CUDA mode: the historical viewer
still computes covariance/support on CPU. Use the CUDA archive renderer for this path.

`scripts/ops/run_body_control.sh` is now the active recipe entry and defaults to CUDA.
Prepare an immutable bundle containing source/target points, their volumes and target
shading, then pass `--input-reference PATH` (or `PHYSMORPH_INPUT_REFERENCE`). The same
bundle supplies shading unless a separate `--target-reference PATH` is given. The launcher
never falls back to legacy execution. An explicit `--legacy-comparison` selects the old numerical
backend. Relative reference paths are resolved before the launcher changes directories.
Use a new run name, because the launcher refuses to overwrite existing run evidence:

```bash
source scripts/ops/hyde06_env.sh
CUDA_VISIBLE_DEVICES=-1 "$PY" scripts/prepare_pipeline_input.py \
  /data/relcfd/chayo/physmorph_v2/work/bunny300k_input.npz \
  --tgt assets/bunny.obj --n 300000 --seed 1 --sampler stratified --disc_ref
bash scripts/ops/run_body_control.sh 0 c291_bunny_cuda_new bunny 300000 60 body_terminal \
  --input-reference /data/relcfd/chayo/physmorph_v2/work/bunny300k_input.npz
```

Preparation refuses to overwrite an asset. The input loader validates finite float32
Nx3 arrays, count, positive finite volumes, exact source/target point hashes, mesh file
hashes and seed/sampler/sample options. Mesh files remain identity checks; execution uses
the bundle's exact points and volumes. Only volume sampling is supported by this bundle
format; shell sampling needs its separate per-particle volume contract.

Fresh mesh sampling in the CUDA dependency environment did not reproduce the historical
archived target byte-for-byte; the old shading asset was correctly rejected. Bundles avoid
resampling under a different environment. Historical archive shading assets remain usable
with their exact archived points; `prepare_target_reference.py` retains that interface.

For direct CLI work use `scripts/ops/run_gpu_pipeline.sh GPU ... --sampler stratified
--input_reference PATH` with matching mesh/N/seed flags; its final argument
forces `--compute_backend cuda`, so earlier arguments cannot silently override it.
G1 checks and post-run raw metrics use the selected backend too. Direct historical
`pipeline_run.py` defaults remain available for explicit comparison work.

High-resolution splat appearance improvement is deferred at the user's request; the
[motion-versus-appearance diagnosis](motion_vs_appearance_20260926.md) is complete.
Next priority after execution parity is raw-state thin-region supply and per-particle
settlement during morphing. The exported4K image is not the current optimization image:
`use_gauss_loss=False` in mixed60, and its low-resolution silhouette/density-normal
loss does not directly penalize individual anisotropic splat boundaries.

## Validation so far

Local CPU reference suite: 305 passed, 22 skipped (GPU/reference requirements), 69.58 s.
The final entry/metric suite then passed 27 checks in 6.24 s, including immutable bundle
validation, launcher argument order, viewer rejection and trailing-null timing.
Hyde06 GPU checks initially passed 20 cases, covering duplicate/self kNN,
small and empty sets, large covariance batches, exact bounded cross queries, source
lattice index parity, small pin-cohort assimilation and independent raw metrics/audits.
At N300000/T20/dt1/240/dx0.3062907544/loss36 cubed, the CUDA six-window prefix
completed in102.09s; the legacy prefix completed in119.46s. Both had zero guards.
This is an indicative pair, not a controlled performance benchmark. CUDA active-pin
motion was exactly zero in every tested window. Six-window positions differ from the
legacy path: median1.012sp vs0.478sp between two legacy repeats. The line search chooses
different steps from window3. Exact trajectory parity is not claimed.

On the actual source AND target300k clouds, the CPU/GPU33-neighbor indices are identical
for every particle; maximum distance difference2.78e-17wu. The small-cohort assimilation
comparison (511 particles, eta1, isochoric on/off) passes atol5e-6/rtol5e-5. These comparisons
exclude those tested deterministic mismatches, not all possible floating-point divergence.

Four actual300k one-window runs in one process verify allocator reuse: Torch live17.04MB
and reserved7.881GB stay constant; CuPy live10.800MB stays constant, reserved1.576GB grows
to1.792GB on the second call and remains there for calls3/4. A per-device Warp-owned stream
prevents each context from creating another allocator stream pool. Reference identities
are device-keyed, and RPROP neighborhoods are owned by each pipeline invocation.

The production-size raw audit uncovered a CuPy 14 defect missed by small tests:
`KDTree.query_ball_point` faults in its native CUDA radius kernel on the actual N=300000
cloud. `CUDA_LAUNCH_BLOCKING=1` localized this to the radius kernel launch. CUDA radius
counts now use validated exact kNN queries, doubling neighbor count only for unresolved
queries. Radii are inclusive; temporary query rows are bounded and there is no fixed
neighbor-count cap or CPU fallback. CUDA list-returning radius queries, non-Euclidean
distance and approximate radius queries fail explicitly.

`scripts/probes/gpu_radius_counts.py` compared all 300000 source and all 300000 target
queries against SciPy at radius 0.0689865998 wu: every count matched. The CUDA compute
and raw-audit suites passed 18 tests in 6.45 s, including duplicate points, zero radius,
boundary inclusion, dense neighborhoods beyond the initial capacity and empty sets.
The full raw audit subsequently completed successfully with this implementation.

Raw summary metrics and `morph_raw_qa.py --compute-backend cuda` now use the same explicit
device context without consuming the renderer. Archive reads remain host I/O. The audit
checks all delivered frames for admitted-pin movement, with one frame upload per check.
Density is a neighborhood-count diagnostic, not a watertightness proof.

Adversarial review found two inherited reporting mistakes while inspecting this work:
the CLI supplied T-1 as an accepted-window stride, and supplied an incorrect held-tail count
after delivery truncation or a final null commit. Layer breathing now consumes actual accepted `frame_end`
boundaries, including unequal archive strides/null commits; held count is intersected
with the delivered slice using the last accepted boundary. Null frames between accepted
windows remain in the archive. Historical fixed-lag calls are labelled `fixed_archive_lag`.
Old `layer_*` numbers must not be presented as accepted-window reversal measurements.

## Full active-path result and remaining work

At N=300000, T=20, dt=1/240, dx=0.3062907544 wu and a 36³ loss grid, the full CUDA mixed60
run completed in 589.05 s: 39 accepted windows from 42 attempts, 782 delivered frames,
all trajectory guards zero. Its independent raw metrics give silhouette IoU 0.97181042,
minimum det(F) 0.77342945 and a highest-tip neighborhood count of 7.2 after scaling by
40000/N (radius 0.25 wu about the target's highest point). These are execution and shape
measurements, not a no-hole or convergence certificate.

The raw audit checked all delivered frames after pin admission: all 253911 checked
particles had exactly zero movement. A fixed cohort of 1959 particles that were surface
and unpinned at delivery still moved across the audited tail: median normal/tangential
movement 0.01490/0.01240 native spacings per archived step, p95 0.06944/0.07260, reversal
fraction 0.08635. Sparse growing regions and free-particle rest remain unresolved.
This GPU migration does not change that physical objective or establish improved quality.

The actual prepared reference used by the full run was separately revalidated against
the archived target: schema 1, N=300000, reference factor 1.9574338205844317;
asset SHA-256 `9c1f205945b4b68b330b902f59570b3f9ccb9d8eca8e0ebc553e035f642620a7`.
Reordering target points or changing the factor was correctly rejected. Hash recording
was added after the full run, so this is a retrospective input validation; subsequent
CLI arms record the asset identity directly.

The actual strict CLI was also tested from the active `run_body_control.sh` entry using
a newly prepared immutable bundle. At N=300000, T=20, dt=1/240, dx=0.3063033045 wu and a
36³ loss grid, a one-window cap retained the 300-window schedule and ran all 8 optimizer
iterations. Both CUDA G1 checks passed; all trajectory guards were zero; 21 raw frames
and post-run metrics were written. The arm took 11.33 s, excluding input preparation and
G1 checks. Both archived source/target arrays exactly equal the prepared arrays, and
geometry/shading provenance records the same bundle hash
`c77bb9d7a91e4fb9d3823cf336c6f64d187167990b7d3e321dc965c8cfdc7032`.
This is an execution smoke on fresh inputs, not a matched historical quality experiment;
its in-transit rest gate failed as expected for the one-window prefix.

The full-run summary's final duplicate frame was subsequently excluded from jitter by
the accepted-boundary timing helper (782 delivered, 781 simulated, one held suffix frame).
Only jitter fields were recomputed on CUDA: jitter_abs 0.0001493908134 wu,
jitter_rel 1.517466891e-5, jitter_max_abs 0.0009164229268 wu. Static shape metrics were
retained. The original summary survives as `cuda_mixed60_metrics_before_tailfix.json`;
the corrected file records the timing basis and source-file hashes.
