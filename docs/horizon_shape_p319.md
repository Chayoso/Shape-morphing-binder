# P319: all accepted phases, raw terminal versus promoted endpoint

P316's completed baseline/raw archives stop under ordinary global policies;
P317 confirms a substantial terminal PIC jump in the baseline. Neither a saved
endpoint nor a two-view hole metric establishes closure through the motion.
Observe every accepted position state before choosing a physical change.

Use P317's hash-bound accepted intervals and their P316 sidecars. For each arm:
observe source frame0, then every archived accepted step1..T. Additionally read
the original optimizer xT immediately before the archived promoted/PIC endpoint.
Label raw xT with no archive-frame index and the paired terminal archive index:
these two observations share a time, not two physical steps. They remain
separate even in the raw arm where their positions are exactly identical.
Omit null/rejected/held duplicates. Baseline has801 accepted archive positions
and40 raw-terminal samples; raw has561 positions and28 raw-terminal samples.

For each observation, on CUDA:

- Binary3x3 point-footprint masks at128 and256 pixels,24 azimuth/elevation views,
  one fixed target-derived extent. Retain per-view body/hole pixel counts,
  hole fractions, target IoU and new/removed hole pixels relative to the previous
  labelled observation. Also retain hole pixels outside the target's hole mask.
- Target-ID coverage by nearest source within2 native TARGET spacings. Separately
  record source IDs outside that target support. Target undercoverage early in
  transport is expected; it is not by itself a hole in the current material.
- Axis-aligned3D extent-box exclusions and per-view projected-center exclusions
  as separate counts. The former cannot certify absence of rotated-view clipping.

No renderer or image loss is consumed. Pack full body masks and target-ID coverage
bits into small chunks with explicit sample-index tables and hashes. Save target
masks and fixed view/extent/spacing definitions. Bind the completed P317 report,
its per-ID archive and all original dependencies, plus the current analysis code,
before and after the run. The same raw input bytes must remain in place.

Numerical geometry/nearest-neighbor/mask/fill operations run on CUDA; file decoding,
hashing, metadata and packed output are host I/O. Stream individual archive frames;
do not load or extract a second complete trajectory. Reserve2GB under the shared
launch lock: at the configured maximum6301 observations, two packed mask sets
plus300k target coverage bits require about1.785GB before compression/metadata.
The driver rejects non300k/T20 inputs or more than6301 samples. The2GB check is
launch headroom rather than an active quota; the packed-size bound applies to
these explicit inputs and fixed view/resolution counts.
Recheck the100GB project cap; retain before/corrected deliverables and active
comparison evidence. Run arms on otherwise free GPUs only.

CPU tests include a transient interior hole whose endpoints are identical,
fixed extent under ejecta, excluded hold rows, and separate raw/promoted terminal
labels. The CUDA gate compares complete observation rows and all packed bits to
CPU on changing shapes and views. Independent review precedes archive execution.

The first actual CUDA gate (accee2d) failed before archive execution because the
installed CuPy does not support `packbits(axis=...)`. The follow-up packs a flat
buffer after padding each row to a byte boundary; it preserves independent row
tail bits. An odd17-pixel test exercises padding in addition to aligned rows.
The failed gate is retained and is not counted as a successful shape audit.

The corrected gate, frozen632e1f6, passes1 actual CUDA case (0 skips,5.95s) and
5 independent CPU cases. It includes odd-width packing, a transient hole,
axis-box ejecta and rotated-view-only clipping. Complete observation rows and
packed bytes match the CPU reference. Gate XML/log and source hashes are in
`docs/evidence/p319/shape_verify2.*`. The full baseline/raw observers completed
as `work/p303/baseline_shape1` and `raw_shape1` from the same frozen
`code_horizon_shape2`. The implementation gate and archive audit remain distinct.

This is a finite-view/discrete-support diagnostic: projected openings can be
genuine topology, and finer masks can expose sampling sparsity. Even zero holes
would not prove3D watertightness or4K Gaussian quality. Source shape/target topology,
within-morph supply and final fit must be interpreted together. No raw/PIC policy
is promoted by the observer. Render-influence telemetry is copied with its
discretization and non-causal limits; this pair does not isolate render on/off.

## Completed observations and independent CUDA audit

N300k,T20,dt1/240,dx.3062907543956724wu,loss36^3,iters8. Baseline contains841
labelled samples (801 archived positions plus40 raw optimizer endpoints); raw
contains589 (561 plus28). Additional endpoint labels share physical time.
The independent CUDA auditor passes13026/9246 checks with zero violations.
It hashes all bound files, independently reduces every packed coverage/body mask,
checks row padding, hole/IoU/transition counts, and reprojects all300k source
points at seven selected states per arm across24views and128/256 resolutions.
All selected masks agree exactly. At those states, up to128 deterministic target
IDs also pass direct FP64 nearest-distance coverage checks against all source
points. This is not regeneration of every nearest-distance query; the same
external CuPy hole-fill library is reused. Source-outside-target-support counts
are not independently recomputed. Receipts and the executed verifier are retained
in `docs/evidence/p319`, along with a hash-bound compact result summary.

| Own accepted archive, excluding additional raw endpoint labels | Baseline W40 | Raw W28 |
|---|---:|---:|
| Samples containing projected holes,128 pixels |424/801|448/561|
| Samples containing projected holes,256 pixels |728/801|419/561|
| Maximum single-view hole pixels,128 |64 (frame299/view2)|58 (frame222/view22)|
| Maximum single-view hole pixels,256 |209 (frame177/view21)|307 (frame176/view2)|
| Final target-covered IDs within2target spacings |298426/300000|297825/300000|

These are different durations and optimized policies, not paired causal effects.
The target itself has one256-pixel hole in view10; final source holes at that
resolution lie outside that target-hole mask, so their locations matter.
Within each baseline window's SAME raw/promoted endpoint pair, the PIC map
adds106..755 covered target IDs (median140.5), while the24-view hole-pixel sum
can either increase or decrease (256-pixel delta -230..195, median1.5).
Raw pairs are identical. The map can help endpoint support yet cause the P317
terminal path spike; removing it alone is not a joint solution.

Rendering remains18views at64pixels; neither arm reaches the scheduled96-pixel
stage. Baseline/raw nominal direction-share medians.365489/.486167 and adaptive
lambda medians.0301579/.0462864 describe their own accepted updates, not a
render-caused displacement fraction. No physical default or renderer is promoted.
