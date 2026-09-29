# P316: observe actual ordinary stopping before another local braking change

P304-P315 examine W20, which still rewards shape fitting. Coarse transport
arrival there is not individual convergence. P314's local speed reductions
remain rejected, and P315's fixed histogram is only a numerically equivalent
GPU optimization. Neither is a physical rest policy.

Run the P303 comparison at its configured300-window horizon from the immutable
mixed60 source. Keep N300k,T20,dt1/240,dx.3062907543956724wu,loss36^3,iters8,
ordinary pin admission, elastic assimilation, outer rejection and stopping.
Keep animations300 and the ordinary C2F schedule (64->96 at zero-based animation
150, the151st attempted solve, not the150th accepted commit), rather than
compressing the schedule into a short run. Baseline uses shared commit-PIC;
raw disables commit_pic and commit_pic_objective. Both use no shift_sub,
outer_render_committed and read-only motion accounting, as in P303. No GS loss,
new stopping threshold, archive restart or repaired-candidate adoption.

This is a comparison of the two existing formulations and stopping behavior,
not a declaration that either fixes the morph. Execute the arms serially to
avoid simultaneous host frame-stack peaks. Reserve30GB per arm before launch:
up to6001 position frames (~21.60GB),301 F samples (~3.25GB), compact commit
archive, plan/cohort telemetry and metadata. In-memory full F_frames add
~64.8GB per live run, plus positions and output-stack copies. Recheck project
usage before each arm; if current usage plus reservation exceeds100GB, clean
only verified obsolete project results under the existing user authorization
before launching. Existing before/corrected evidence is retained.

The observer wraps the original optimize_window call, adds no forward or
gradient, and returns its exact tuple. Own window-start pin_init and x0 hash
before calling it, then own the returned full plan, start-arrived mask and
radius immediately. Retain per-ID optimizer-terminal |v|^2 in float64 computed
on the active device; this is before outer commit operators, not a claim about
promoted or post-pin velocity. Each attempted window gets a separate archive,
including null, gradient-stop and rejected attempts. Missing plans are allowed
only for solves with no accepted inner history.
Also own the returned raw optimizer endpoint before the runner replaces the
archived terminal frame with a promoted/PIC position. This permits SAME-ID
raw terminal step versus final position-correction comparisons. At300 attempts
the extra telemetry ceiling is3.06GB:1.08GB plans,.18GB two boolean masks,
.72GB float64 per-ID terminal speed-squared and1.08GB raw endpoint positions,
before compression. The complete worst-case disk estimate is about29.1GB/arm;
the30GB reservation must stay available during the run.

After the ordinary run, bind each attempt's start hash to its actual archived
frame and map accepted intervals using outer acceptance, null/reject status,
frame_end and the actual solver frame count. Distinguish final attempted/last
accepted state, delivery-selected endpoint and held suffix. Held/pinned zero
motion is imposed, not evidence of natural equilibrium. Bind the complete
source/input bytes before/after and every per-attempt sidecar.
The C2F history entry is a schedule event sharing an animation number with its
solve, not another physical attempt. Retain it separately. If final telemetry
mapping fails, export the completed physical result first, then retain the
trace error and fail the diagnostic; never lose the expensive physical archive
because an observer could not map a history event.

Later GPU analysis derives end arrival from the actual promoted endpoint under
the SAME window's plan and radius. A next-window plan change at unchanged
positions is a relabel event, not physical departure. Follow fixed material IDs
through all accepted positional paths, excluding null/rejected/held rows and
accounting for first pin admission. Next-attempt pins plus final archive pins
close the post-admission state only while release policies are off; the driver
rejects release flags. Existing motion_accounting cohorts change by window, so
their stored-velocity summaries remain descriptive until evaluated on a fixed
cohort. The per-ID terminal scalar supports that later check without full V.

Report all-phase multi-view raw holes, coverage and per-ID path/terminal motion;
do not infer absence of holes from only two views or a final-frame score.
Separate continued fitting, reference changes, pin-imposed rest and ordinary
global stopping. Record render losses, adaptive lambda and direction telemetry
through the actual end; this pair is not a render-on/off causal ablation.
If full convergence is not reached by the cap, report that outcome explicitly.
No rendered deliverable is shipped without all-frame QA.

Before launch require actual two-window CPU observer-on/off parity for both
PIC and raw branches, complete state/archive mapping, owned captures, and
null/reject/held/gradient-stop regressions plus independent refutation review.

Implementation gate closes on four independent CPU cases. Both actual two-window
branches cross a C2F event with exact observer-on/off state/merit equality. The
PIC fixture has a nonzero raw/promoted endpoint difference; the raw fixture
matches exactly. Output raw/compact archives and rendering reports are bound
by final hashes, in addition to source/input and per-attempt telemetry bindings.
No CUDA horizon result is implied by this preparation.

## Baseline result (frozen9cab095)

N300k,T20,dt1/240,dx.3062907543956724wu,loss36^3,iters8; configured300 windows.
The ordinary baseline stops after43 attempts, with40 accepted windows/320 inner
updates. Attempts41-43 are outer-rejected; three consecutive rejections cause
the stop. There are802 raw/delivered frames: initial +800 physical steps +one
held row. No C2F event was reached. Guard counters are all0. Final pin count is
272864; the remaining27136 material IDs are free. `reported_converged=true`
means the runner froze; it does not establish individual or natural rest.

Independent `refute_p316.py --hash-large` passes the source/input/output hashes,
complete attempted/accepted/null/held mapping, raw/compact/F endpoints, delivery
identity and every pin-admission chronology. It does not recompute arrival,
all-phase finiteness/holes or geometry. The raw alternative is launched serially
from the SAME frozen code; no baseline quality or policy promotion follows.

Rendering:18 views at64pixels throughout this baseline. Median nominal weighted
render-direction share.365489 (320 recorded updates in accepted windows),
adaptive lambda median.0301579, observed image-loss change median-3.55800e-6.
These are observational optimizer quantities, not a36.5% displacement share or
a render-on/off causal estimate. Full report and independent audit are retained
in `docs/evidence/p316/full_baseline1.*`; physical-motion analysis is P317.

## Raw alternative completed (same frozen9cab095)

Same N300k,T20,dt1/240,dx.3062907543956724wu,loss36^3,iters8.30 attempted
windows,28 actual commits/224 inner updates,562 raw/delivered frames including
one held row. No C2F, no guards. Last accepted attempt28 has stale3; the next
two non-brake outer rejections consume patience5 and freeze the runner. This
is a plateau/rejection stop, not evidence of natural rest. Final273393 pins,
26607 free IDs. Independent full-hash mapping/chronology audit passes.

Rendering uses18 views/64pixels. Nominal render-direction share median.486167,
lambda.0462864, recorded accepted image-loss change median-7.09023e-5. These
own-duration observations do not provide a matched causal rendering fraction.
Audit/protocol/trace/render report retained in `docs/evidence/p316/full_raw1.*`.
No full-horizon physical policy is adopted; P317 motion and P319 all-phase
shape observations precede any change to acceptance or rest behavior.
