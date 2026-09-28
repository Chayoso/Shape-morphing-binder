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

## Eight-window policy comparison

Frozen04c1ddc, original bunny300k/T20/dt1/240/dx0.3062907543956724wu/loss36^3,
eight inner iterations. Both arms add the same read-only motion accounting;
geometric_variance remains the only A/B configuration difference. Both accepted
8/8 outer attempts without holds or guards, in154.34s/168.86s. Pin-motion checks
passed. This is an adaptive-lambda policy comparison, not fixed-lambda causality.

| At accepted window8 | Control | Geometric variance |
|---|---:|---:|
| Silhouette IoU |0.914117066|0.914021493|
| Chamfer(wu) |0.061726091|0.061657956|
| Fixed upper-target coverage |0.718129572|0.730668757|
| Arrival fraction |0.957273333|0.958303333|
| Pin fraction |0.334090000|0.360056667|
| Minimum detF in this window |0.892302096|0.907784820|
| Fixed source-ID density |0.633566746|0.641556168|
| Moving-top conditional density |0.724574669|0.705320409|
| Tip particles |0|0|

The fixed source cohort contains6712 pre-treatment upper-boundary IDs, including
IDs now below y2.3 or pinned. Its pin fraction rises0.278010->0.318087, so its
aggregate motion cannot establish matched-free rest. Moving-top membership
changes7935->8021; its lower density and higher under-half fraction
(0.253308->0.267548) cannot alone establish newly opened holes. Fixed upper-target
coverage uses the same15312 target IDs. First Chamfer-threshold crossings occur
at the same windows1/2/7 in both arms; there is no clear early progress slowdown.

The matched common-endpoint-free cohort contains15372 IDs (none pinned in either
arm there), selected from the union of sparse endpoint boundaries. W1 is excluded
from this paired phase audit; its interval is windows2..8, raw archive frames20..160.
In source-native spacings0.03498853660707278wu:

| Same-ID movement over windows2..8 | Control | Geometric variance |
|---|---:|---:|
| All raw-step RMS(sp) |0.235458124|0.234673322|
| Phases1..19 RMS(sp) |0.227720968|0.229349141|
| Phase20 RMS(sp) |0.351469227|0.319392887|
| 19->20 reversal fraction |0.208068473|0.200308539|
| 20->next1 reversal fraction |0.208734496|0.190888195|
| Interior reversal fraction |0.000299968|0.000448662|
| Commit-to-commit reversal fraction |0.060803192|0.067850638|

Phase20 includes the last raw step and PIC remap. Its RMS falls9.13%, while
interior RMS rises0.715% and total raw RMS falls only0.33%. This supports less
concentrated boundary motion with redistribution into earlier steps, not rest.
The per-arm motion-accounting split does not use the same frozen material cohort
and is descriptive, not a causal isolation of PIC. Both tips remain empty and
the late failure regime is unobserved. No prefix quality or repair pass is claimed.

Independent review supports ONE unchanged cap60 exploratory pair to examine
that missing late regime. No coefficient/tolerance tuning between arms. Record
each final endpoint and a common accepted-time interval, progress crossings,
fixed source IDs, shared free IDs, raw/interior/final-phase/tangential drift,
pin admission and observed-pin denominators, target fit and thin-region supply.
More pins or a lower variance scalar cannot establish improvement. Final fit,
supply and persistent motion remain required before any promotion; strict CUDA
primitive failures, full gallery and per-frame render QA remain outstanding.

Evidence: local output/p297/variance_{control,geometry}8.json,
variance_quality8.json and variance_phase8.json. The candidate stride-one NPZ
remains on hyde06 under work/p297; the completed control stride-one NPZ was
subsequently archived locally as recorded below. The phase audit is bound to the exact quality
JSON hash and identical cohort ID hash. Source/code/config equality is checked
by the quality driver; it compares paired input arrays, not a reported source hash.

Storage: one already archived P295 smoke NPZ(692,411,244bytes) was removed from
the server after renewed local and server full SHA256 verification, an inode
read lease and durable receipts. All companion evidence and its local archive
remain; usage after that action was98,568,868,826bytes. No other deletion is
implied by that individual receipt.

Before full60, all nine previously verified P294/P295 raw duplicates and the
completed P297 prefix-control raw archive were removed with the same renewed
local full-hash/fsync and server lease/full-hash/receipt procedure. Total removed
logical bytes:11,360,768,568. Every payload remains in its verified local archive;
JSON/log/compact evidence stays on the server. The prefix-control archive and
its companions are at C:/dev/physmorph_archives/p297_prefix_control_20260928T163844Z.
Other local archive roots are listed in output/p297/archived_cleanup_manifest.json.
Each exact removal has a local output/p297/remove_*.json and durable server
prepared/completed receipt; no recursive deletion was used.

The cap60 worst-case bound is1201 position frames,61 F samples and60 compact
commits at N300k/T20. A conservative5.30GB per arm plus250MB reserve requires
10.85GB headroom. After the final duplicate removal, project use was
88,644,736,788bytes, leaving11,355,263,212bytes below the user's100GB limit.
The full pair runs from frozene0ca68d with no physics changes after cap1.

## Full60 exploratory result: promotion rejected

The unchanged cap60 pair ran from e0ca68d on September28, with original
bunny300k, T20, dt1/240, dx0.3062907543956724wu, loss36^3, eight inner
iterations, shared PIC, shift off, corrected outer render merit and motion
accounting in both arms. Only geometric_variance differs. The numerical source
hash is c9c622efa148287a66c31b361b25c94d565ab6a05bfe803403895b8028e333b1.
State guards are zero in both runs. The candidate reduces measured late motion
but loses silhouette fit and thin-region supply. It does not pass the combined
quality/rest gate; defaults and renderer deliverables remain unchanged.

Delivery, actual stopping and physical rest are different scopes:

| Run scope | Control | Geometric variance |
|---|---:|---:|
| Actual accepted / attempted windows |35 /35|34 /37|
| Accepted windows retained in delivery |35|30|
| Runtime(s), descriptive |544.2853|637.7387|
| Actual last accepted frame index |700|680|
| Delivered frame count |702|601|
| Archived frame count |702|682|
| Minimum detF over delivered accepted windows |0.862778664|0.847918212|

Control W31..35 are accepted but have improved=0 and stale=1..5. With tol0.003
and patience5, runner.py:1145 and1395..1406 set frozen after insufficient track
improvement. The candidate ends after three consecutive outer rejections at
attempts35..37; best-state delivery selects W30 and excludes accepted W31..34.
Its actual W34 metadata reports detF minimum0.842885017 and v_mean0.055535905wu/s,
but that state's independent geometric metrics were not sampled by these audits.
The flag converged=True returns frozen (runner.py:1900), not equilibrium or rest.

The current quality loader filters accepted records by frame_end<=deliver_n
(scripts/probes/render_influence.py:34). Thus its inherited analysis_scope phrase
"final accepted states at each arm stopping point" must be read here as the last
delivered accepted state: control W35 versus candidate W30. The original JSON is
preserved; variance_summary60.json explicitly adds both endpoint scopes and keeps
candidate W34 raw_geometry=null. The common phase interval is W21..30, starting
at accepted endpoint20/frame400 and ending at endpoint30/frame600.

At the same accepted W30, source-native spacing is0.03498853660707278wu and
target-native spacing0.03493084911867985wu:

| Raw geometry at W30 | Control | Geometric variance |
|---|---:|---:|
| Silhouette IoU |0.969559302|0.967885164|
| Chamfer(wu) |0.058820392|0.058786497|
| Fixed upper-target coverage |0.962317137|0.956178161|
| Fixed source-ID density |0.704875596|0.680125149|
| Fixed source-ID under-half fraction |0.233611442|0.245530393|
| Moving-top conditional density |0.918641821|0.889664916|
| Moving-top under-half fraction |0.119756744|0.134829672|
| Tip-ball particle count(target89) |45|55|
| Arrival fraction |0.999640000|0.999883333|
| Pin fraction |0.898543333|0.753083333|

Fixed source density uses the same6712 pre-treatment upper-boundary IDs and
the historical target median r8 radius0.06898659982768912wu; counts exclude self
and divide by8. This cohort includes pinned particles and IDs outside y>2.3.
Upper-target coverage uses the same target IDs in y>2.3, whereas moving-top
membership differs(12826/12564). The candidate's additional tip particles do not
cancel the lower fixed-target coverage or fixed-ID density. Neither density nor
binary projection proves watertightness. Own delivered-endpoint IoU is
0.969392301/0.967885164; both miss0.971. The0.225-initial-Chamfer crossing moves
from W18 toW20, so some late progress is slower despite slightly lower W30 Chamfer.

The matched free cohort contains3601 material IDs, unpinned in both arms over
W21..30. It is selected from the common-endpoint sparse-boundary union, not by
each particle's arrival time. Normals are separately frozen at each arm's common
endpoint; vector RMS is independent of those normal bases. All motion below uses
source-native spacings per saved physical step at dt1/240:

| Same-ID late motion | Control | Geometric variance |
|---|---:|---:|
| All saved-step RMS(sp) |0.086603403|0.065759785|
| Tangential saved-step RMS(sp) |0.049024267|0.028934023|
| Phases1..19 vector RMS(sp) |0.031073246|0.022855344|
| Phase20 vector RMS(sp) |0.362846526|0.276698423|
| Phase20 path share |0.330740631|0.330483705|
| Raw-path net/path median |0.388131499|0.560677826|
| Commit-path net/path median |0.859447062|0.924351513|
| Raw-step reversal fraction |0.066352311|0.056256637|
| Commit reversal fraction |0.127433738|0.040297448|
| 19->20 reversal fraction |0.668258817|0.564121077|
| 20->next1 reversal fraction |0.711592459|0.602085840|
| Interior reversal fraction |0.000651054|0.000749801|

Overall RMS falls24.07% and tangent RMS40.98%. Both earlier phases and phase20
fall(26.45%/23.74%), while the phase20 path share remains about33%. This is not
a selective removal of the boundary pattern: into/out-of-boundary reversals
remain56.41%/60.21%. Their denominators are36010/32409 pairs in each arm;
interior counts are422/648180 and486/648172. Phase20 includes the raw final step
plus PIC. The per-arm motion-accounting decomposition uses changing cohorts,
so its components are descriptive, not a matched-ID causal operator split.
No post-arrival rest test or periodic-oscillation frequency was measured here.

Pin observations also require an archive-clock qualification. The control's
frozen branch appends one copied frame701 after physical endpoint700, then
breaks without incrementing n_held when dressing is absent(runner.py:540..548).
Thus n_held=0 coexists with one nonphysical held copy. Raw QA checks274258 pins
and reports zero drift, but the final1065 admissions have only that copied row
as a subsequent observation. The history-derived count with a later physical
step is273193. Candidate delivery has no suffix:225925 admitted pins versus
223417 checked, excluding2508 final admissions. Zero drift is observed over the
reported archive rows, not evidence of continued physical rest for unobserved
final admissions. The matched phase audit excludes held suffixes. Own-tail raw
QA uses different free cohorts(2925/6668) and is not an A/B matched-ID rest test.

Render lambda changes alongside this policy: at W20 it is0.0374227/0.0892803,
W24 0.0293259/0.0891700, and W30 0.0217697/0.2109393. This association does not
establish that lambda caused the fit loss; entering state, controls and objective
gradients also differ. If further mechanism work is authorized, an identical
prepared late state/target/control snapshot with fixed positive lambda would
separate the combined regularizer-plus-PCGrad effect from lambda feedback.
It would not isolate the regularizer alone: the changed physics gradient also
changes PCGrad's reference direction. Before a solve, a same-state gradient audit
should separately record delta physics gradient, projected render-gradient
direction and lambda from a cloned balancer. This is a proposed diagnostic,
not a causal conclusion from the adaptive pair.
No new weight is selected here. Both strict primitive CUDA failures, including
the unresolved T1 comparison, remain open; this full run does not relax them.
No renderer run, gallery pass or default promotion follows from this result.

Evidence is local output/p297/variance_{control,geometry}60.json and logs,
variance_quality60.json, variance_phase60.json and variance_summary60.json.
The phase report binds exact quality bytes and the cohort ID hash. The stdlib-only
selector preserves all nulls, both original run hashes and the distinction between
delivered and actual last-accepted metadata; it performs no geometry recomputation.
Summary SHA256:026eb50fd1e461769318eac19f3f599748cd6c1e95c7548062e0fa06e187eec8.
Quality SHA256:7b9d67d5387f8a17ef11eb24b79f36b25e1252cab40871c61a2a8ff47180e605.
Phase SHA256:a199692972ee398586b05ee2d2dbe924293b4a9c20c790a371eebcbc799485a6.
