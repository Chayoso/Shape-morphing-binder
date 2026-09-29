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

This is a finite-view/discrete-support diagnostic: projected openings can be
genuine topology, and finer masks can expose sampling sparsity. Even zero holes
would not prove3D watertightness or4K Gaussian quality. Source shape/target topology,
within-morph supply and final fit must be interpreted together. No raw/PIC policy
is promoted by the observer. Render-influence telemetry is copied with its
discretization and non-causal limits; this pair does not isolate render on/off.
