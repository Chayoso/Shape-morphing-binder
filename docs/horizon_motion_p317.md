# P317: distinguish motion, endpoint correction and reference changes over P316

Read only the completed P316 archives; no MPM replay, loss evaluation, state
replacement or new stopping/pin policy. Keep each arm's discretization in the
report. Numerical geometry runs on CUDA on hyde06; archive decoding, hashing and
JSON output are explicit host I/O. Read the raw NPZ in bounded accepted-window
intervals, never materialize the full trajectory repeatedly or extract a second
22GB copy. Validate the P316 trace with the independent auditor first.

For each actual accepted interval, use the promoted saved path and separately
retain raw last displacement, endpoint correction, and promoted last displacement.
The correction is part of the last saved step, not an extra dt. Stored terminal
speed comes from the optimizer before outer operations and pin admission; it is
not promoted velocity or proof of equilibrium. Its vector direction and affine C
cannot be reconstructed from speed squared. Check that already pinned IDs have
exactly unchanged positions throughout every accepted interval.

Use the recorded window-start arrival mask as the controller's classification.
Compute endpoint arrival under that SAME frozen full plan/radius. Reconstructed
start masks may differ at the FP32 roundtrip boundary; count those differences
explicitly. For reference relabels compare geometric arrival at the previous
accepted endpoint against the next accepted start, requiring exactly identical
positions. Rejected/null references are not physical departures or relabels in
this accepted-path report. Coarse arrival is not convergence or rest.

Track fixed material IDs. Final-free, eventually-pinned, initial-free and the
first accepted window's start-arrived set are retrospective/initial cohorts,
not populations automatically matched across arms. First accepted endpoint
arrival qualifies SUBSEQUENT accepted windows only. Keep escapees and reentries
in this chronology. Save per-ID observation-window/step counts and path/squared
displacement/stored-speed sums separately before and after pin admission.
Unequal follow-up durations must remain visible; imposed zeros are not rest.
Report accumulated sums under explicit sum names, and divide by observed
particle-step/window counts before deriving pooled RMS. Also expose per-ID
normalized distributions; empty denominators produce null, not a measured zero.
The reversal totals below cover the entire accepted path, not post-arrival only.

Retain consecutive saved-step and window-net reversal counts with their eligible
denominators: both displacement lengths must exceed1e-4 source-native spacing,
as in the existing raw-phase analysis. Include adjacent accepted-window pairs
using the actual saved last displacement, not the raw optimizer endpoint. These
negative-dot statistics describe reversals; they do not certify oscillation.
Null/rejected/held rows add no physical steps. Distinguish whole accepted path,
last actual accepted window, delivery-selected endpoint, and held suffix.

Hash-bind source telemetry, completed archive, sidecars and render report before
and after analysis; parse each JSON from the same bytes that were hashed.
Require actual backend/arrival dependencies to match the producer's hashes.
Independently compare trace ordering, acceptance, delivery metadata and actual
last accepted index with the original result and raw archive.
Copy the render-influence report and per-window recorded
loss/lambda/direction observations. This baseline/raw pair is not a render-on/off
causal ablation; nominal direction share is not a motion fraction. Image losses
or these motion diagnostics cannot establish closure or4K visual quality.

CPU fixtures must cover a nonzero PIC jump, pin invariance, adjacent-window
reversal, null/rejected/held exclusion, actual/delivery scope, post-first-arrival
follow-up, pin chronology and input mutation. CUDA parity must exercise the whole
archive analysis with geometry on device. Independent refutation precedes use.
All-phase holes/coverage, actual whole-horizon physical decisions and rendered
frame QA remain separate required work. This analysis promotes no physical fix.
