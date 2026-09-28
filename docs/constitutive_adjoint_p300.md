# P300: stable rotation adjoint for fixed-corotated stress

Status: local CPU counterexample independently verified and reproduced on
server CUDA; the same-forward derivative fix passed10 focused Warp1.16 CPU
tests. The first CUDA suite has9 passes and1 chain finite-difference failure,
now isolated to the float32 forward difference by a separate all-channel oracle.
The original strict position-sequence probe passes on the new derivative.
The revised focused suite passes10/10 on CPU and CUDA. Matched cap6/cap24
morph checks show mixed changes and do not establish repaired holes or rest.
This is a constitutive differentiation
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
CPU module passed all10 tests in8.99s. A fresh frozen0402e8e CUDA execution
passed the same10 tests with zero errors or skips (pytest0.899s; wrapper1.692s).
The test, constitutive and kernel hashes were independently verified against
the reviewed source. The original9/10 execution remains separately recorded.
Evidence: output/p300/run_cuda_tests_v2.{json,xml}; result SHA256:
c43c22adccd0aef4f589772ba972bc8c242d92464342b3bb4f3282101b414365.

A fresh full CPU run on frozen0402e8e with isolated Warp1.16 completed:
662 passed,22 skipped,0 failed in313.63s. Log:
output/p300/full_cpu_116_v2.log. The skips remain recorded and are not GPU
passes. The separate CUDA unit and position-sequence runs above cover their
explicit scopes. No full-morph quality inference follows from unit validation.

## Matched cap6 morph comparison

Frozen9706141 old core versus0402e8e corrected core use exactly the original
bunny300k inputs, T20, dt1/240, dx0.3062907543956724wu, loss36^3, eight inner
iterations and cap6. Both retain shared PIC, physical temporal variance,
motion accounting and the promoted-state outer render gate; external shift,
geometric variance and geometric rest are off. The two approved physmorph
byte changes are the rotation adjoint and the Warp version guard, inert on
the common1.16.0 server. The driver and numerical metric helpers are identical.

Both runs accept6/6 windows with zero guards, in107.314/107.271s. The independent
CUDA audit binds each result to its own complete frozen source digest, exact
old/new blob pair, inputs, prepared reference, configurations and MPM parameters.
It checks all121 physical position frames, excludes the copied suffix, and uses
raw simulation states without a renderer. Audit source491bf2c; its66 focused
provenance/scope tests also passed independent CPU review.

| At accepted W6 | Old | Corrected |
|---|---:|---:|
| Silhouette IoU | 0.895936 | 0.894246 |
| Chamfer (wu) | 0.0663149 | 0.0664017 |
| Fixed upper-target coverage | 0.609979 | 0.609653 |
| Fixed6712-source-ID density | 0.618687 | 0.619245 |
| Current top-region density | 0.613066 | 0.606212 |
| Arrived fraction | 0.942377 | 0.943667 |
| Pinned fraction | 0.071837 | 0.098207 |
| Minimum accepted trajectory det(F) | 0.905836 | 0.903680 |

On20000 identical sampled IDs out of21658 particles free in both arms at the
common endpoint, W2..6 saved-step RMS changes0.257236 to0.257005 native spacings
(-0.090%). Native spacing is0.03498853660707278wu. Phase1..19 RMS changes
0.250454 to0.250126sp, while phase20 RMS changes0.362758 to0.363777sp (+0.281%).
Phase20 includes the final physical/layer step plus PIC. Raw-step reversal
fraction changes0.015636 to0.015378; accepted-commit reversal instead increases
0.07719 to0.08495. These are direction changes, not proof of periodic oscillation.
The cohort is selected at the common endpoint, not by individual arrival time.
Both progress crossings occur at the same measured accepted commits. The larger
pin fraction must be retained in any later interpretation of reduced motion.

Previously admitted pins show exact zero motion on4165/5357 checked IDs.
Total admissions are21551/29462: the newest admissions have no later physical
observation and are excluded from that denominator. The two-view point-projection
hole fraction is9.54e-5/0, but this does not certify watertightness or the absence
of visible Gaussian holes. Neither arm reaches the highest tip at W6.

This is a small mixed early result, not a quality repair. It justifies observing
the corrected derivative later in the same recipe with a bounded cap24 pair;
there is no new gain, geometric-variance promotion or new renderer result.
The extended audit keeps exact provenance gates, uses an explicit6/24 cap and
reports the last10 common accepted displacements separately from each retained
endpoint and full progression. Its92 focused CPU tests passed independent review.

Evidence: output/p300/{control6,candidate6,quality6}.json. Quality SHA256:
f8bfe1385892b98bcb27c9a6aec6fcf1db2dee1fbb1344bd14690ad0013c471c.
All eight original NPZ/JSON/log files were copied, fully hashed and fsynced to
C:/dev/physmorph_archives/p300_cap6_pair_20260928T181241Z. After audit completion,
the four NPZ duplicates were locally rehashed and removed from the server under
the reviewed read-lease/hash/durable-receipt procedure, freeing1,127,856,850bytes.
JSON/logs and snapshots remain remote. Archive receipt SHA256:
7dd35308c9185305a6d4a3ae315b279985f5e5e00df64976185b76a71626f0ad.
Removal record: output/p300/cap6_removed_receipt.json. Project usage after removal
was96,939,061,771bytes. No unarchived source or output was deleted.

## Matched cap24 result: better progress, residual motion persists

The same frozen9706141/0402e8e simulation pair was rerun with cap24 only;
N300k, T20, dt1/240, dx0.3062907543956724wu, loss36^3, eight inner iterations
and every other recipe setting remain as above. Both accept24/24 with zero
guards and retain481 physical frames without holds or delivery trimming.
Runtime is423.928/420.734s. The frozen32756ba CUDA audit passed provenance and
state checks; independent review verified both complete source manifests,
run hashes and all reported scopes. These are capped runs, not a final gallery.

| At accepted W24 | Old | Corrected |
|---|---:|---:|
| Silhouette IoU | 0.968238 | 0.969364 |
| Chamfer (wu) | 0.0588659 | 0.0587629 |
| Fixed upper-target coverage | 0.956766 | 0.964668 |
| Highest-tip neighbors, raw count within0.25wu | 49 | 51 |
| Fixed6712-source-ID density | 0.712642 | 0.711096 |
| Same source IDs with density<0.5 | 0.229291 | 0.232569 |
| Current top-region density | 0.901699 | 0.898589 |
| Arrived fraction | 0.999197 | 0.999563 |
| Pinned fraction | 0.779923 | 0.799700 |
| Minimum accepted trajectory det(F) | 0.854025 | 0.872049 |

The5568 identical IDs still free in both arms at W24 are measured over the
same accepted endpoint interval14..24, i.e. displacements in W15..24. Their
source-native spacing remains0.03498853660707278wu. Saved raw-step RMS rises
0.088178 to0.090628sp (+2.78%), tangent RMS0.049521 to0.053239sp (+7.51%),
and phases1..19 RMS0.043159 to0.050097sp (+16.1%). Phase20 RMS falls0.346577
to0.341441sp (-1.48%), but includes final physical/layer motion plus PIC.
Tangent bases use each arm's frozen endpoint normals, not a common normal
basis; the saved vector RMS does not have that decomposition ambiguity.
The phase20-to-next-phase1 reversal fraction falls0.611111 to0.561901;
accepted-commit RMS instead rises0.705277 to0.858705sp (+21.75%) and commit
reversal rises0.033246 to0.055995. No rest or individual post-arrival conclusion
follows from that mixed motion redistribution.

Previously admitted pins remain exactly still on231985/232154 checked IDs.
Total admissions are233977/239910, with final admissions excluded from later
physical observation. The fixed source cohort includes MORE pinned IDs in the
candidate (74.76% to76.61%); any smaller aggregate motion on that mixed cohort
must not be presented as a matched free-particle improvement.

Progress also differs: the0.225-times-initial-Chamfer crossing occurs at oldW21
and correctedW18. There, IoU is0.967285/0.967470 and upper coverage
0.949909/0.950104, while fixed-source density is0.704093/0.693795. Thus the
same-W24 shape improvement partly accompanies faster progress, and does not
remove the sparse-support tradeoff. Both W24 point-projection hole fractions
are0; the accepted-endpoint audit alone cannot rule out transient gaps between
commits or visible splat coverage problems.

Across all accepted rows, the median nominal render-gradient share is
0.433711/0.435868 and median adaptive lambda0.059384/0.058539. These are scalar
history diagnostics before Adam in mixed control coordinates, not causal
fractions of motion or a render-on/off ablation. Geometry, density and material
motion are computed on CUDA; scalar history summaries and provenance I/O use
the host. The fixed renderer and its high-resolution appearance are unchanged.

The derivative correction is retained for its independent numerical evidence.
It is not a solution to the remaining holes, drift or oscillation. No geometric
variance weight/default or new render result is adopted. Next, inspect saved
intermediate physical frames for endpoint-supported regions that disappear
and return; an individual arrival-time audit needs additional per-ID records.

Evidence: output/p300/{control24,candidate24,quality24}.json. Quality SHA256:
447afcfe743087d7495955a9adb18f2da7e15f454c1332a23aa9225719159aed.
All cap24 raw archives remain under server work/p300. Project usage after the
audit is99,063,384,101bytes; do not start another large archive without headroom.
