# P320: remove both existing layer position paths in a full raw-endpoint run

P317 proves that shared PIC can dominate the last saved displacement. P319
also shows that it improves same-endpoint target support, while neither existing
policy eliminates intermediate projected openings. PIC removal alone is not
promoted. Raw W28's own final-free geometric and stored terminal speed norms are
already close (.106927569/.106885890wu/s); scalar norm agreement does not prove
vector consistency or implicate hidden layer movement as its remaining cause.

Before changing the integrator, test the existing stress/body formulation with
both direct layer position paths disabled. The optional `raw-no-layer` arm of
`full_horizon.py` starts from the immutable mixed60 source; it is not an archive
restart or candidate replacement. Relative to raw, only `layer_ctrl` and
`layer_relax` change from true to false. Their dormant gates/coefficients stay
recorded. Stress/body controls, material, bonds, pins, assimilation, loss weights,
accepted buffers, line search, outer acceptance and ordinary stopping remain.
PIC and subgrid shifting remain off. Bonds can still project disconnected
particles, so this arm is not automatically a pure-force or exact x/v model.

Use N300k,T20,dt1/240,dx.3062907543956724wu,loss36^3,iters8,300 configured windows,
18 render views/64pixels with the existing96-pixel schedule. Run a fresh raw
reference from the SAME frozen checkout and the no-layer arm serially with30GB
reserved at each launch, preserving the100GB project cap. Retain both outcomes
and the old P316 raw result; a single repeat is not a noise distribution.
The read-only RestTrace owns start pins, references, raw endpoints and stored
terminal speed squared. Rejected/held rows are not physical movement.

This changes the optimized policy, including later gates and pin decisions;
it is not a same-control instantaneous decomposition of the layer's work.
Compare common accepted physical frames first, with matching material IDs and
pin-follow-up denominators; show each own endpoint separately. Reuse the
unchanged P317/P319 physical metrics and all-phase support observations. Keep
zero guard/inversion requirements, whole-horizon shape/supply gates, independent
verification and eventual all-frame4K visual QA. No threshold relaxation or
default adoption follows from faster stopping or lower motion alone.

The experiment can refute that these layer projections are necessary for the
observed motion or holes; success would motivate a force-path design, not certify
natural rest. Do not convert arbitrary layer displacements to v/C/F as a
bookkeeping fix: that feeds new momentum into P2G and needs its own impulse/work
and deformation contract. No such recoupling is implemented here.

Report rendering's lambda, raw and combined direction norms, accepted control
updates, image-loss changes and independent raw evidence. Direction-norm share
is not a fraction of displacement. This pair is not render-on/off causality and
does not supervise the exported4K appearance.

The comparator's explicit `layer_projection_off_full` mode and phase report pass
104 CPU contract/regression cases and independent refutation. It validates the
full actual and delivery-retained accepted clocks separately, forbids compressed
raw frames for bounded mmap access, hash-binds artifacts before and after
analysis, and reports the same source cohort at actual/delivered endpoints.
Zero common path gives an undefined (null) fraction rather than an invented0.
Only the probe files may be overlaid onto a clone of the frozen producer core;
the comparator rejects a different physical source aggregate. Actual CUDA
comparison results are recorded below.

## Completed pair, 2026-09-29 UTC

Frozen producer1ff004d, analysis-probe overlay0ce63ea. Fresh raw accepted31/31
attempts in435.99s. No-layer accepted61/65 attempts in576.81s; its last accepted
state is attempt62. Both have one copied held row and no delivery truncation;
622/1222 archive frames contain621/1221 actual physical positions. Guards are0.
Raw ends on accepted-track plateau; no-layer ends after3 consecutive rejections.
Neither stopping rule certifies individual rest.

At the same31 accepted windows, under the discretization above:

| Raw-state observation | Raw with layers | No layers |
| --- | ---: | ---: |
| silhouette IoU | .9618841 | .9594799 |
| target support within2 target spacings | .9927100 | .9909933 |
| upper-target support | .9354754 | .9254180 |
| fixed6712-ID upper source density | .9702771 | .7785496 |
| tip particles | 75 | 33 |
| same1778 free IDs, W22–31 step RMS(sp) | .01643544 | .01658742 |
| same IDs, phase20 RMS(sp) | .01703085 | .01974626 |

First19-phase RMS is essentially unchanged(.01640350/.01640433sp), while the
last phase worsens15.94%. Window-boundary reversals remain exactly91/16002 in
both arms; interior reversals increase210/320040 to663/320038. These are negative
dot products, not independently certified oscillations. The common cohort is
selected from both outcomes, not a full-population causal estimator.

No-layer's own endpoint has99.6093% pins versus93.8303% at raw's earlier endpoint.
Its remaining1172 free IDs still have terminal geometric speed RMS.133116wu/s;
raw's distinct18509 free IDs have.074350wu/s. Final312/353 new pin admissions have
only copied-held follow-up. Lower whole-cloud mean motion at a later endpoint
cannot establish natural rest.

P319 all-phase observers and bounded independent CUDA audits pass:652/1282
labelled observations,10191/19589 reduction checks and7/6 selected source-state
reprojections. In the common621 physical positions,128-pixel projected openings
occur in509/504 positions;256-pixel openings occur in477/576. Summed24-view hole
pixels increase18626→46268 at128 and114798→277469 at256. These are projected
interior-mask openings, which can include legitimate silhouette concavities or
sampling artifacts; they are not proven3D cavities. No all-frame hole remedy is
established.

Rendering influence remains active: own accepted-update nominal direction-share
medians.449727/.366573, adaptive-lambda medians.0356187/.0401626,248/488 accepted
inner updates. These unequal-duration summaries are not rendering's causal
movement share. Both use18views64pixels,GSoff. See evidence/p320 for receipts,
image-loss changes and exact source/report hashes.

Decision: reject combined layer removal as the requested joint remedy. Retain
the experiment and original policy. Independent refutation checked physical
source/input identity, artifact receipts, clock/cohort definitions and phase
scalar aggregation; it did not independently regenerate every raw geometry
query. No stopping, admission or quality threshold is relaxed.
