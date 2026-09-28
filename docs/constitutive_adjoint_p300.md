# P300: stable rotation adjoint for fixed-corotated stress

Status: local CPU counterexample independently verified and reproduced on
server CUDA; the same-forward derivative fix passed10 focused Warp1.16 CPU
tests. The first CUDA suite has9 passes and1 chain finite-difference failure,
now isolated to the float32 forward difference by a separate all-channel oracle.
The original strict position-sequence probe passes on the new derivative.
Test-oracle revision and actual morph-quality checks remain pending. This is a constitutive differentiation
correction, not a new morph loss or evidence that holes/rest are repaired.

The current rotation is R=UV^T from a signed SVD, with the existing proper-
rotation correction. Generic differentiation of the individual U/V factors
uses squared singular-value differences. The product R can remain smooth when
those separate factors are not uniquely determined. In particular, repeated
positive singular values are not a singularity of the full-rank positive-
determinant polar rotation.

For the smooth positive-determinant map, writing F=RS and dR=R Omega gives
S Omega+Omega S=R^T dF-dF^T R. This is the polar derivative's Lyapunov equation;
see [Gawlik and Leok, equation66](https://arxiv.org/pdf/1608.04491).
For incoming rotation cotangent B, solve
S Z+Z S=R^T B-B^T R and return R Z. This avoids differences of nearly equal
singular values. The planned implementation retains the forward rotation and
uses a custom backward function, supported by
[Warp's custom-gradient interface](https://nvidia.github.io/warp/v1.12/user_guide/differentiability.html).
The installed versions themselves must be tested; the documentation version
does not establish execution compatibility.

In three dimensions the skew-matrix solve can use the equivalent3x3 matrix
trace(S)I-S. Its eigenvalues are pairwise signed-stretch sums. For positive
full-rank F these are positive even at an isotropic state. Inverted inputs
require a separate signed-stretch/unique-proper-rotation contract; ties that
make the proper rotation nonunique cannot be fixed by silently clamping a
denominator. The fix must not change forward inversion handling or insert a
new material regularizer.

## Local counterexample

The preserved one-kernel diagnostic evaluates54 cases with Warp1.9.0 CPU,
float32 matrices, lambda80, mu40, Fp=I and dFc=0. Cases cover separated,
nearly repeated, repeated-pair and isotropic positive spectra, rotated
equivalents, and symmetric/skew/random cotangents. An independent float64
SVD/Sylvester reference uses the exact float32 input. This is not a pipeline
run and has no time/grid discretization.

At F=I, the independent linear-elastic stress tangent is
DP*(B)=mu(B+B^T)+lambda trace(B)I. A skew cotangent therefore has zero response.
The current kernel instead gives directional AD93.6587668, while the analytic
value is0 and the actual float32 forward's centered difference at epsilon1e-3
is-0.0009728. At diag(1,1,1.4), the same diagnostic gives AD152.5857382 versus
analytic67.4414027 and forward centered difference67.4414795. Rotated repeated
spectra also fail. The discrepancies exceed finite-difference rounding in these
counterexamples and establish a local constitutive VJP defect.

Evidence: output/p300/stress_rotation_cpu.py and stress_rotation_cpu.json.
The source, inputs, seeds, all54 results, analytic values and three finite-
difference steps are preserved. Independent review verified the polar and
volumetric VJP formulas and identity-elasticity check. A separate gradient-
inventory inspection found no missing persistent tape gradient or uncleared
gradient in the small CPU case; this does not prove all CUDA bridges correct.

The server has Warp1.16.0. Its unchanged frozen constitutive kernel must be
evaluated by the same diagnostic before any claim about that backend. A later fix requires
unchanged forwards, full stress-chain dFc/F/Fp/material checks, ordinary and
captured rollout checks, and original strict tolerances. The prior P297 CUDA
failures remain failed evidence, even if this local cause is repaired.

## Unchanged server CUDA counterexample

The same54-case oracle ran on hyde06 GPU2 using Warp1.16.0 and the frozen
e39e0d6 constitutive/kernels sources. Actual kernel execution is CUDA; the
small independent analytical oracle remains float64 host arithmetic. There is
no MPM rollout, rendering or production CPU fallback. Source hashes/version,
inputs and all finite-difference values are retained in the result.

At identity with the skew seed, stress directional AD is70.9092652 versus
analytic0 and actual CUDA-forward centered difference0.00714867 at epsilon1e-3.
At diag(1,1,1.4), AD173.8718187 differs from analytic67.4414027 and CUDA-forward
FD67.4254652. Thus the derivative problem is also present on the actual server
backend. Its contribution to the prior bridge mismatch or morph quality is
still unmeasured; no whole-pipeline cause claim follows.

Evidence: output/p300/stress_rotation_cuda.{py,json}; remote work/p300/ under
/data/relcfd/chayo/physmorph_v2. Result SHA256:
38b44e4c706cc7d589ac38ab5c7a53363f82026c8a053d3b6ed6dc6ef9e38ce7.

The implementation registers a custom rotation adjoint and leaves the original
proper-SVD forward function unchanged. The focused10-test CPU suite passed on
an isolated Warp1.16.0 installation; no global local package was replaced. Tests
cover exact forward equality, positive repeated spectra, a distinct inverted
branch, finite differences, dFc/F/Fp/lambda/mu stress-chain derivatives and the
computed-singular NaN path. The latter is not a claim to detect every nonunique
inverted tie. Warp1.9 cannot compile a nested custom adjoint because of function
declaration ordering; the dependency minimum is now1.16 with an early import
check. The server already has1.16.0. No tolerance in P297 was changed.

The first frozen7437f1e CUDA suite executed10 tests with no skips:9 passed and
the stress-chain directional finite-difference case failed. Its AD value
-12.427041947996095 differs from FD-12.453812206315954 by0.0267702583,
above the unchanged0.0249076244 allowance. That loop stops at its first failure,
so its later channels are not validated by the nine other passing cases.
The failed XML/JSON are retained as output/p300/run_cuda_tests.{xml,json}.
Do not relabel this as a full CUDA pass or loosen the allowance. A separate
all-channel analytical check is needed to distinguish a pullback error from
float32 forward finite-difference error. The unchanged-tolerance rollout probe
can supply separate diagnostic evidence but cannot clear this chain failure.

## Independent chain diagnosis and original rollout probe

The frozen7437f1e all-channel oracle ran on hyde06 GPU0, Warp1.16.0 CUDA,
using the failed test's exact float32 inputs, cotangent and seeded directions.
It compares complete F/dFc/Fp/lambda/mu gradients to a separate float64
SVD/Sylvester chain reference. Both Fp paths are included: the elastic argument
(F+dFc)Fp^-1 and the final stress multiplication by Fp^-T. No MPM discretization
applies to this single-particle constitutive check. The fixed matrix epsilon
sweep(.02,.01,.005,.003,.001,.0003) and scalar sweep(.4,.2,.1,.05,.02) were
recorded before CUDA execution; no best epsilon or new tolerance is selected.

Full-gradient relative errors are3.53e-7(F and dFc),4.64e-7(Fp),1.00e-6(lambda)
and4.32e-6(mu). For the failed F direction, analytic-12.4270501958 and actual
AD-12.4270419480 differ by8.25e-6. The original epsilon.003 CUDA FD remains
-12.4538122063 and still fails its original bound. At that same epsilon,
float64 FD is-12.4274397988, and rounding only the perturbed input to float32
gives-12.4243706362. CUDA forward loss error contributes another-0.0294415701
to the FD, accounting for the discrepancy. The other four original directional
FD readouts pass. This diagnoses the test's float32 forward subtraction;
it does not retroactively label the original9/10 suite a pass.

Evidence: output/p300/chain_diagnostic.py, chain_reference.json, chain_cuda.json.
CUDA result SHA256:
4d75f83841a6c3e9983e87b7d46cba0b414c750dfc8cae8d5f172c922f56751e.
Diagnostic script SHA256:
766f29aa9d51309320361731913aae68d5a03eadb98c776bd979bf1cbfc2d14a.

Separately, the unchanged scripts/probes/position_sequence.py passed all52
pass-bearing checks on frozen7437f1e, hyde06 GPU2. It retains the original
N64 subset of the300k source, T3 plus T1 edge case, dt1/240,
dx0.3062907543956724wu, seeds, masks, controls and all tolerances. Ordinary
and captured rollouts, full/terminal/velocity/merged cotangents, first-step
gradients, pinned particles, nontrivial layer relaxation, ownership, reset
and all six finite-difference cases pass. Independent review checked all62
dependency hashes and exact input/tolerance parity with the failed old-core
probe. This clears that small bridge test only, not all300k behavior or holes.
Evidence: output/p300/position_sequence.json; SHA256:
c7993926569d44ac53b2816b1b574aa72db9ace904090f74472ff18bfa5ca805.

The full local Warp1.16 CPU suite returned658 passed,22 skipped and2 failures
in421.64s. Both failures were the same mock Warp module lacking __version__,
newly read by the dependency guard. The mock was corrected and two explicit
old-version early-rejection cases added; tests/test_gpu_entry.py then passed
all19 tests. This is a recorded full run plus targeted repair, not an assertion
that the initial full run was green. No global local package was replaced.

The chain test was subsequently revised after independent review of that error
decomposition. It now compares the complete actual CUDA/CPU VJP for all five
inputs against the analytic chain with the same64-float32-epsilon,
coefficient/cotangent-scaled allowance as the existing spectral tests. It also
checks actual forward P against the independent reference. The reference's
directional derivatives are checked by float64 forward differences at the
original epsilon.003/.1 and original relative.002/absolute.02 bounds.
Every other actual-kernel finite-difference test is unchanged. This changes
the chain oracle, not the production code or its derivative tolerances, and
retains the original failed float32-FD evidence above. The revised focused
CPU module passed all10 tests in8.99s; a fresh CUDA run remains required.
