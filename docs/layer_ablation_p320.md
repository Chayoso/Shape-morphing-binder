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
comparison results remain pending.
