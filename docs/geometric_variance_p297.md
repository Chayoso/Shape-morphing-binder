# P297: let temporal variance observe the saved positional path

Status: optional implementation passed CPU review; both CUDA derivative probes
retain strict failures. Repeated adjoints support a bounded T20 one-window
integration diagnostic, not acceptance of the primitive gate or a quality claim.
P295 resolves an explicit window-end position jump and a smaller immediate
layer-relaxation response. P296 screens whether removing parts of that endpoint
filter preserves geometry before any such split can be selected. P297 keeps
the current shared endpoint map and changes a different observable.

The current w_kin_var penalizes population variance of the stored physical V.
Direct layer/bond position changes and the PIC endpoint jump are not all visible
to V. With geometric_variance, define y0=x0, yt=xt for t<T, yT=the exact shared
promoted endpoint, and Ut=(yt-y(t-1))/dt. Replace only the existing variance input
by mean_p mean_t ||Ut-mean_t(U)||^2. The same w_kin_var and density-unit multiplier
apply. The existing physical terminal and running kinetic, contact and continuity
terms retain their physical velocity inputs. The old momentum variance is no
longer penalized by this one term; record both variances explicitly.

The bridge returns an independently owned post-layer X=[x1,...,xT] and seeds every
actual trajectory position in the backward pass. X's last seed is merged with
the separate xT seed, just as the terminal V seed is merged with vT. The ordinary
and captured CUDA implementations have separate explicit APIs. Legacy outputs
and defaults remain unchanged; previous-position and full-sequence bridge modes
are mutually exclusive. Start state is constant. Q and the exact H^T pullback of
the promoted endpoint remain in the endpoint map.

Gradient evaluation, line search, warm-start checks, replay-noise calibration
and final replay merit all use the same observable. A reused accepted path must
equal the validated trajectory arrays before promotion. No x/v/C/F/Fp update,
extra pin admission, renderer edit or interpolation is introduced. Configuration
requires shared PIC, T>=2, finite positive w_kin_var, geometric_rest off and
endpoint-only KKT pin admission off. Defaults remain off.

This is a policy experiment, not a guarantee of rest. For a fixed endpoint,
Var(U)=sum_t||dy_t||^2/(T dt^2)-||yT-y0||^2/(T^2 dt^2), so an evenly distributed
path can reduce variance and constant drift has zero variance. Record every
phase's raw/saved RMS displacement, separate raw-final and remap movement, net
displacement and total path length. A smaller scalar alone cannot pass a motion
gate. At T20, a single endpoint jump has variance coefficient19/400 times its
squared geometric speed. With w_kin_var200 that is9.5, versus P294's remap weight5
before common unit conversion; unchanged nominal weight is not equal strength.

Validation sequence:

1. CPU derivative and ownership tests for both bridges, actual post-layer X,
   mixed control channels, pins, T1, simultaneous/repeated/missing seeds and
   stale contexts. CPU integration compares independently reconstructed archive
   rates and accepted/replay merits in legacy and density units.
2. A small hyde06 CUDA graph/FD gate on the fixed mixed60 source subset, using
   the original dx/dt, followed by a cap1 integration pair. These are correctness
   checks, not physical-quality evidence.
3. Before interpretation of an isolated mechanism solve, hold actual entering
   state, prepared targets/layer/plan/arrival data, initial controls and positive
   render lambda fixed in both arms. A copied argument list alone does not prove
   prepared references equal. Admit only an originally outer-accepted snapshot.
4. Any full adaptive-lambda A/B is instead a whole-policy comparison. Preserve
   identical source/configuration except this flag, original bunny300k/T20,
   dt1/240, dx0.3062907543956724wu, loss36^3, eight inner iterations, shared PIC,
   no subcell shift, corrected committed-state outer render merit, cap60 under
   animations300. Report actual acceptance/stopping, not just the cap. Match
   material IDs and report all phases so earlier movement or a larger remap
   cannot masquerade as rest. Guard/pin failure or loss of thin-region supply
   and target fit precludes promotion. A successful prefix cannot replace full
   trajectory, hold, gallery and per-frame rendered QA.

The automatic lambda balancer uses the changed physics gradient. Its response
is part of a whole-policy comparison and must not be described as fixed-lambda
causal evidence. No new rendering experiment is included here.

## Initial validation, September 28

Source5a3cd5f preserves defaults off. Independent review passed67 focused CPU
cases; the full CPU suite passed558 with22 skipped in161.63s. This includes
accepted-buffer and forced-replay merits in both loss unit systems. New Warp
position interfaces retain all existing public output meanings. The CLI rejects
the endpoint-only KKT combination because it cannot certify full-path sensitivity.

The first CUDA primitive probe uses64 particles from the immutable original
300k source, T3, dt1/240, dx0.3062907543956724wu, fixed nontrivial layer relaxation,
nonzero initial state and mixed dFc/u/body controls. It launched16:01:05UTC on
GPU0. Forward X and xT agree exactly, other forward fields and u/body derivative
comparisons pass, and independent finite differences at both preregistered eps
pass for all three channels. However dFc coordinatewise derivative comparisons
fail: max discrepancies2.86e-6 to5.71e-6, strict tolerance ratios1.09 to1.83.
The stored path-seed repeat also differs by5.71e-6, after intervening seed types.

This does not yet distinguish ordinary backward variability from seed-reset
contamination. Keep `position_sequence.json`/log and frozen source unchanged;
the run is failed, not reclassified. A separate v2 measures consecutive identical
seeds on both implementations before held-out cross/seed-cycle comparisons, and
reports float64 vector statistics alongside every unchanged strict criterion.
No noise allowance is inferred from the disputed cross difference and no strict
pass is silently substituted. Integration remains pending its interpretation.

Evidence: server `/data/relcfd/chayo/physmorph_v2/work/p297/position_sequence.json`,
local `output/p297/position_sequence.json`. The T1 edge case keeps the T3 frozen
layer relaxation fraction while disabling body control; it tests API/adjoint
behavior and is not a one-step physical experiment.

## Repeat diagnostic and bounded integration scope

The v2 diagnostic (source48725a3, simulation core unchanged from5a3cd5f)
preserves the first failed result and every strict criterion. Eight consecutive
identical-seed adjoints precede held-out seed-cycle comparisons. For the T3 path
dFc gradient, maximum within-implementation pair RMS differences are6.0381e-7
(ordinary) and5.7909e-7 (captured); cross-implementation maximum RMS is6.1235e-7.
The difference of sample means has RMS2.2790e-7, relative L2 1.9443e-5 and
angle0.001114 degrees. The held-out dFc path deviations show no seed-cycle
growth. All five dFc sample-mean comparisons meet the original coordinate limits.
These observations are consistent with backward numerical variability; they do
not prove an atomics-only cause. Very small u/body velocity-seed differences can
exceed their own tiny repeated-adjoint envelopes, so no all-channel noise-bound
claim is made. Forward outputs remain unchanged and the independent finite
differences pass at both original epsilons for all three control channels.

The v2 strict gate still fails. In addition to T3 dFc comparisons, the T1 merged
oracle comparison reaches2.863e-6 (tolerance ratio1.058); the T3 repetition does
not calibrate T1. T1 remains unresolved and is outside the next T20 scope.
Evidence is preserved as `output/p297/position_sequence_v2.json` locally and
the corresponding server work/p297 file. No derivative tolerance is relaxed.

Independent review permits a new, explicitly bounded integration pair using the
actual original bunny300k recipe, T20, dt1/240, dx0.3062907543956724wu, eight
inner iterations, shared PIC, no subcell shift and one outer window. Both arms
retain the real objective's other terms; only geometric_variance differs.
This pair must reconstruct the saved positional variance independently in
float64, show a nonzero weighted objective contribution, check finite returned
X and equality of X[-1] with xT, and verify owned/promoted/archive and F/v/C/pin
state contracts. Zero initial pins make that particular first-window check
vacuous; this cannot establish late-window pin behavior or physical quality.

An optional read-only `on_objective` observer exposes a callback-lifetime scalar
evaluator of the same prepared final objective with only variance substituted.
The default path incurs no additional objective evaluations. Subtraction of
float32 scalar merits uses an explicit rounding bound based on their magnitudes;
this is separate from the unchanged derivative criteria. Inner validation is
not outer acceptance. No full-run, gallery or render promotion follows from a
successful one-window integration alone.

## Actual T20 cap1 integration, September 28

Frozen sourcefc07572 passed independent code review and17 CPU tests covering
accepted-buffer and forced-replay paths in both observable modes. GPU0 control
and GPU2 geometry each accepted all8 inner iterations and one outer window,
with no guards, in14.99s and16.22s respectively. Both used the accepted buffer.
Emitted protocols have identical source/target/reference hashes, MPM parameters
and code hashes; geometric_variance is the only configuration difference.
The original strict v1/v2 failures remain attached by hash.

| Observable check | Control | Geometric variance |
|---|---:|---:|
| Independently reconstructed geometric variance |0.651513681|0.617452826|
| Independently reconstructed physical variance |0.565108990|0.566999004|
| Selected term's expected weighted contribution |0.00342455436|0.00374175743|
| Scalar subtraction absolute error |7.08e-10|1.35e-9|
| Recorded merit equals recomputed merit |exact|exact|
| Gradient forwards returning full X |0|8|

Variances have units(wu/s)^2 at T20/dt1/240; dx0.3062907543956724wu and N300k
are unchanged. The effective weight is200/33003.359375=0.00605998916.
The geometric arm has finite full-X seeds and nonzero earlier-position gradient
participation. Returned X is owned, finite and terminal-exact; accepted/promoted
positions and every archived position match the evaluated saved path. F/v/C are
checked at optimizer return; outer commit independently checks x/F/v, not C.
Both initial pin counts are zero and that pin check is explicitly vacuous.

This passes the new bounded integration check only. The lower first-window
geometric variance is not a rest, hole or fit improvement claim. Before any
full-run decision, the conditional next experiment compares8 attempted windows
with motion accounting in both arms, retaining all original controls/objective
terms and adaptive render balancing. It must report progress and fixed-ID
motion/coverage alongside the variance; all strict primitive failures remain.

Evidence: server work/p297/integration_control and integration_geometry;
local output/p297 holds both protocol.json and integration.json. Each server
accepted_path.npz is105,601,458bytes. Control SHA256 startsd089effaec802e3d;
geometry startsc7f38157dc065b7f; full hashes are in each integration result.
