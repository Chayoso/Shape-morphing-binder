# P333: derivative of actual elastic assimilation

Status: opt-in primitive and frozen old/new-pin composition implemented; 46 CPU
checks and all 23 CUDA tests pass. Frozen `1457340acac8d98f0602ba65fc41dd2b72262385`
passes the extended production-size FP32 graph and composed-map gates in 2.73s.
No optimizer, runner or existing withdrawal default uses this derivative yet.

The next coast must respond to the plastic state the runner really commits.
`plasticity/assimilation_adjoint.py` differentiates the existing elastic
assimilation order: Fe=F inv(Fp), right stretch power, optional isochoric
increment normalization, multiplication by incoming Fp, cumulative singular
clamp, then optional fixed-target log-band projection. Physical F enters this
map; F+dFc does not. Growth, consensus, pin yield/release/follow/KKT and successor
layer/bond/reference preparation remain outside this primitive.

Forward float32 uses the existing Torch algebra. CPU production uses NumPy;
actual pin admission also subsets newly pinned particles before calling the
legacy function, whose SVD backend changes below 20000 rows. Thus primitive
large-CUDA equality and composed-subset numerical parity are separate gates.
Float64 exists for independent derivative checks, not a new production mode.

The increment pullback differentiates the symmetric right stretch using
pairwise singular sums and spectral divided differences. Isochoric diagonal
terms include all three singular values. The cumulative U diag(q) V-transpose
pullback retains both the symmetric divided differences and the skew factor
(q_i+q_j)/(s_i+s_j), including incoming Fp rotation. Repeated singular values
use their analytic smooth limits without perturbing the forward spectrum.

For isochoric cumulative projection, the pullback applies, in reverse order,
the exponential, the fixed-active-set projector diag(f)-f f^T/sum(f), the log,
and the FIRST singular clamp. The determinant target retains the existing
feasible-range clamp. Differentiating bisection decisions or omitting the first
clamp would give a different derivative. At an exact floor/clamp/health boundary,
no unique derivative is claimed; finite-difference tests stay within branches.
Only first derivatives are supported.

`cuda_svd.py` uses the installed PyTorch 2.8 small-matrix settings: scalar-type
epsilon tolerance, 400 sweeps, sorted singular values, packed column-major
inputs and owned Torch buffers. The existing linalg.svd checks convergence on
the host. CuPy14 also prohibits cuSOLVER calls during capture in its wrappers;
the helper invokes the installed Linux CUDA12 solver through typed C pointers,
with owned Torch buffers and explicit API status checks. It requires aligned
Torch/CuPy streams and checks info on
the device. A nonconverging matrix fails explicitly; it does not silently take
PyTorch's host-directed fallback. Thus parity is scoped to converged inputs.
See [PyTorch's cuSOLVER implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp),
[its sweep setting](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.h),
and [NVIDIA's gesvdjBatched contract](https://docs.nvidia.com/cuda/cusolver/#cusolverdn-t-gesvdjbatched).

Invalid elastic determinant sets the increment to identity but still applies
the cumulative projection. Eta<=0 bypasses both, as in production. The pin
wrapper uses disjoint frozen masks. `settle_pin_assim=True` restores incoming
Fp on old pins and performs the second eta=1/noniso assimilation on new pins,
consuming the first call's result. False does neither pin-specific operation.
This wrapper maps Fp only; it does not implement v/C zeroing or pin admission.

Tests compare the existing forward, independent identity/isotropic/rigid and
twofold-spectrum derivatives, noncommuting F/Fp directional differences, active
bands, skipped increments, singular floor, old/new/free pin composition and
repeated backward calls. The CUDA gate additionally checks both subset backend
sizes, actual GPU graph replay and absence of explicit numerical host copies.

Next: connect a gradient-bearing coast Fp and v/C pin projection to the actual
controlled-head boundary, preserving direct x/v/C/F/Fg paths. Validate tiny
head-assimilation-coast finite differences with detached-Fp/C negative controls
before any production candidate search. A conditional derivative with frozen
successor preparation is not a complete derivative of preparation policies.

Rendering influence: unchanged; these are algebra/derivative tests without a
morph, render objective or candidate adoption. No new render influence number,
natural-rest, hole-removal or 4K appearance claim follows.

## Recorded verification

These are matrix-map tests, without an MPM grid or time step. The final hyde06
run is `work/p303/p333_cuda5`, code `work/p303/code_assimilation5`, Torch2.8.0+cu128,
CuPy14.0.0, CUDA12.8 libraries, RTX6000 Ada. All 23 tests pass, no skips. CPU
reference/derivative checks pass 46/46 (37 new plus 9 legacy). Independent review
checked the spectral formulas, branches, typed ABI, stream and buffer ownership.

- N20000 float32 ordinary forwards equal production bit-for-bit for eta .35/1
  with isochoric off/on. The original inputs remain unchanged.
- Actual N24000 runner-style composition preserves old pins and calls production
  on only the newly pinned subset. N128 new pins has maximum absolute difference
  1.728535e-6 (0.126858 of the registered 64-FP32-epsilon scaled allowance);
  N20000 new pins is exact. Smaller subsets take the production CuPy SVD branch.
- Float64 directional differences cover noncommuting F/Fp, cumulative clamps and
  the singular floor at two radii. Repeated spectra and the independent identity
  derivative include both F and Fp paths.
- CUDA graph forward/backward replays changed F, Fp and cotangent seeds at N1
  float64 and N20000 float32. The composed N20000 case includes 1024 old and 1024
  new pins, verifies their distinct forward/gradient behavior and a nonzero
  second-call effect. These are Torch graph gates, not a Warp joint-rollout gate.

Three initial attempts each pass 18 numerical gates but expose, in order,
the inverse's host check, SVD's host convergence check, and CuPy's capture wrapper
restriction. Their failures are retained. The fourth passes the original 20
gates; the fifth passes all 23 extended gates. No gate threshold was relaxed.
The final test warning is a scalar conversion in a post-replay test assertion,
outside numerical capture; it does not signal a failed check.

The monitor sampled 718MiB process peak in the final test (not an exact peak).
Project usage after collection is 77355478828 bytes, below 100GB. No result
cleanup or retained before/corrected video change was needed. Source/output
hashes and complete test reports are in `evidence/p333/source_receipt.json`.
That receipt binds artifacts; it is not an independent rerun of GPU calculations.
