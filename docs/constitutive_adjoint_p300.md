# P300: stable rotation adjoint for fixed-corotated stress

Status: local CPU counterexample independently verified and reproduced on
server CUDA; the same-forward derivative fix passed10 focused Warp1.16 CPU
tests, with full CPU and CUDA validation pending. This is a constitutive differentiation
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
