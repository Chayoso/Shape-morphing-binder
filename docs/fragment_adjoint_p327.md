# P327: production effect of retained fragment activity

P326 found an incorrect lifetime for the per-step fragment mask in reverse. Its
small changing-mask fixture demonstrates an opposite-sign old gradient with an
unchanged forward path. P327 measures the effect in the actual N300k morph before
claiming any improvement in holes, residual flow, arrival oscillation or fitting.

Both arms use one frozen checkout and the unchanged full-horizon raw recipe:
N300k,T20,dt1/240,dx.3062907543956724wu,loss36^3,iters8, maximum300 attempts,
ordinary stopping, no endpoint PIC, no shift_sub. Rendering guidance is18views
at64px with GS loss off. No withdrawal objective or new rest policy is enabled.

`fragment_adjoint_compare.py --mode legacy` aliases every gradient trajectory's
fragment-mask slots to one buffer before capture, reproducing the old reverse
lifetime. `--mode retained` uses the corrected per-step arrays. The original
allocation remains owned in both arms; no-grad evaluation is unchanged. Both
arms separately record actual masks into identical per-step observation buffers
inside the captured forward. Reading the legacy array list after the rollout
would incorrectly report the final mask at every time, so that is not used.

After every actual persistent-adjoint forward, device reductions report per-phase
active and dynamic-only counts, differences from the final mask and temporal
flips, for all/free/layer-free cohorts. One scalar packet is copied to the host.
These are attempted gradient evaluation points, including repeats; they are not
automatically accepted inner updates or committed outer endpoints. Warmup/capture
construction is not counted as an executed optimizer forward. Diagnostics add
24MB per N300k,T20 bonded trajectory plus temporary reductions and small JSON;
do not treat measured runtime as uninstrumented performance.

Runs are serial, with the shared GPU launch lock/interval and30GB reservation
per arm. The wrapper/mode protocol and all completed observer data are hashed
against the full-horizon trace, whose input/physical-code/output bindings remain
unchanged. An observer failure does not grant quality or provenance clearance.
Both arms have identical physical code/config digests because the intervention
is a runtime wrapper. Pair analysis must consume each activity report's bound
fragment protocol and validate legacy then retained modes; ordinary digests alone
do not identify this intervention.

Use existing P317 chronology/arrival/free-ID analysis and P319 every-phase raw
shape observations. Compare the common physical clock separately from each own
actual endpoint and held padding. Preserve original acceptance/quality gates;
no new rest-speed threshold. Projected openings alone are not3D watertightness.
Use per-arm rendering-influence reports: norms, balance, accepted update dot
products and reference-local image loss changes. A changed gradient from this
fix may affect both physics and rendering paths; norm share is not a causal
motion fraction. A pair alone cannot establish universal improvement or replace
the19-case gallery and all-frame visual QA before a new deliverable is adopted.

The producer frozen6a525c3 passed3 CPU observer cases (including full C parity)
and2 actual CUDA graph cases. The same actual GPU test launch additionally passed
all6 stronger P326 derivative/stream/fragment checks, for8 passed and0 skipped
in2.21s. The serial production pair is `p327_legacy1` then `p327_retained1`, GPU0,
from `work/p303/code_fragment_adjoint1`. Outputs remain diagnostic; the retained
before/corrected4K exports are unchanged.

The pair comparator now supports `fragment_adjoint_retained_full` with identical
configs bound to the exact native raw recipe, explicit ordered modes, full
producer/wrapper/input hashes, every attempt's disposition and archive frame
span, exact cohort basename, and before/after output/log checks. Existing full
raw metrics/thresholds are reused. Final40 metadata cases passed independently;
broader existing quality/phase/layer checks also passed. This does not independently
recompute the activity packet counts. For analysis, clone the frozen6a525c3
producer and overlay only `quality_compare.py` and `raw_phase.py`; its producer
ops/docs/physical files must remain byte-identical. An external launch wrapper
may supply the ordinary GPU lock/interval/storage guard without replacing those
bound producer files. No analysis job may silently relabel the two equal-config
arms from code hashes alone.
