# Root analysis — one spectral gap, three symptoms (2026-09-02)

The oscillation, the floating gaussians, and the 1000x-small render gradient are
one statement projected onto three measurements.

**Root question: are the visually important modes of the particle state —
surface-local, fine-scale, thin-feature — observable by the loss stack and
controllable through the dynamics? Measured answer: barely either.**

The state is N x 3 particle positions; what the losses measure and what the
adjoint lets the control move is only the LOW-FREQUENCY subspace of it.

## 1. Unobservable modes -> floating gaussians

D_vol (CIC), DT-W1 and the silhouette are low-pass aggregates. Our own
forensics: CIC is structurally blind to sub-cell arrangement; the silhouette
sees boundary rays only. A viewer-visible floater is a particle living in the
LOSS NULL SPACE - zero restoring gradient; drag + plasticity freeze it in
place. Evidence that this is the mechanism and not a tuning issue: the nn-band
term made exactly one null-space mode observable (per-particle assignment) and
that floater class (fork halo 326->78) died, after every grid-side tightening
had failed.

## 2. Uncontrollable modes -> render gradient 1000x small

The render gradient's UNIQUE content is its high-frequency component (surface
detail, floater positions, ears); its low-frequency component is redundant
with the physics terms. To reach dFc that covector must pass the MPM adjoint:
cubic B-spline P2G/G2P is a low-pass filter, its adjoint is the same filter,
and F-smoothing s=0.955 is an explicit low-pass. So the channel's unique
information is precisely what the control path attenuates most. lambda (650-
1300) compensates norms, not direction; what survives is the redundant part.
Work telemetry quantifies it: render_work 1e-6..1e-3 vs phys_work 0.03..33.
Scale separation: grid dx=0.5 vs splat sigma~0.04 - a 12x gap the signal was
asked to cross twice.

## 3. Low-stiffness valley -> near-optimum oscillation

The late-run residual error lives exactly in those soft modes (ears = thin
features = high frequency). Curvature there is ~0, so every noise source -
lambda drift, per-window Adam restarts, assimilation ratchets - makes the
iterate wander the flat valley. Forensic match: strong oscillators sit ON the
surface at 1.2x spacing; the ear is not special, the global pattern is merely
visible there. v1's global damping suppressed the wandering (and the detail);
v2 removed noise sources one by one (drivers #1-#5); the valley stayed flat
because flatness is the objective's property, not the optimizer's.

## What the frame dictates

1. **Observe at the visual scale**: the viewer's forward model as loss (gauss
   channel), supervision sampling matched to splat size (Nyquist res floor),
   surface-aligned representation (2DGS disks) so that what is seen is what is
   differentiated.
2. **Route fine corrections around the low-pass**: high-frequency correction
   must not ride dFc through the dynamics. The existing bypass is the
   assimilation->Fp channel (terminal state made permanent directly); the
   retired local surface pass was groping at the same idea. Explicit scale
   division of labor: coarse transport = MPM + volumetric losses; fine surface
   dressing = render-driven, surface-local.

Every prior fix (caps, gates, priors, pacing) repaired the well-observed
subspace and was necessary; none of them could touch the gap itself. The
dossier forensics are the evidence ledger for this diagnosis.

## REVISION (2026-09-02, transfer-function probe — docs/probes/transfer_function.md)

§2 was MEASURED and is largely refuted as the driver:

- The adjoint's fine-scale attenuation exists but is small: white per-particle
  covectors reach dFc at 0.14 of a 2-dx-smooth covector (7x), 0.1 wu bands at 0.31,
  but the ACTUAL silhouette covector (62–65% of its energy below 0.1 wu, 90% on ~1500
  rim particles) transmits at gain 0.89 vs the physics covector's 1.03 — ratio 0.86.
- The "render gradient is 1000x smaller" is LOSS SCALE (d_render ≈ 0.03 vs d_vol ≈
  240 → ||g_p||/||g_r|| ≈ 6500 in x-space, 7600 in dFc space); the adjoint changes
  that ratio by ≤1.65x. λ ≈ 5e3 (at lambda_cap) compensates it in the objective.
- Consequently the render_work SHARE (~0.02%) that motivated "the render channel
  barely steers" is scale-dominated and uninformative: even a step perfectly aligned
  with the render descent direction would score ~1/6500. Steering is now measured
  by scale-free cosines (`render_cos`, `phys_cos` in optimizer stats).
- F-smoothing s is a TEMPORAL EMA on F, not a spatial filter: s=1.0 freezes F
  (F-path gain exactly 0), s=0 disables it; 0.955 costs the x-path 4.8x and the
  F-path 23–34x uniformly across scales. Not a spectral gap.
- Alignment: cos(g_r, g_p) rises 0.17 (x) → 0.63 (dFc); 78% of the transmitted
  render norm stays orthogonal to physics — the render channel's unique content
  does reach the control.

What stands from §1/§3: the OBSERVATION null space (silhouette/CIC blind to interior
and sub-cell arrangement — floaters) and the flat-valley soft modes (oscillation) are
unaffected by this revision. What changes: the "route around the low-pass" program
(§"What the frame dictates", item 2) loses its premise; the remaining levers are the
observation model (what the loss can see — gauss/Tier D, per-particle terms), the
λ cap binding at the balance point, and the optimizer's handling of the soft modes.

## REVISION 2 (2026-09-03) — the target was a SHELL

Found by the particle-relaxation design pass and verified: `assets/bunny.obj` is not
watertight (Euler −3), `mesh.voxelized().fill()` adds zero interior voxels, so
`sample_volume` returned SURFACE samples (0% of samples deeper than 3% of the bbox
diagonal; the isosphere source is 67% interior). Every bunny run morphed a solid sphere
into a hollow shell target. On the best 20k state (r3b): 98% of particles lie within 0.1
of the shell, 0% deeper than 0.3, and the median NN spacing collapsed from 0.0756
(source volume) to 0.0293 — the body is a thin layer. Consequences for this dossier:
- "floaters" (out_nn 8–12%) were the layer's outer thickness, not stray particles;
- the "clustering ratio 2–2.7" compared a layer against a 4× finer shell spacing;
- the "under-filled ears" were a layer failing to thin enough; the KDE losses minimised
  layer thickness (their outward expulsion mode) — all three variants were fighting
  the sampler, not physics;
- chamfer 0.070–0.071 is the shell-vs-shell floor, i.e. the runs had converged.
The optimizer-side findings (mom_carry cascade, pace cap, gate v3, best-commit delivery,
E4/berth ownership) stand on their own evidence. Armadillo is watertight but its fill is
partial (11% interior). Fix: robust volumetric fill for non-watertight meshes + target
volume matched to the source (isochoric particles cannot change total volume), then
re-run the ladder on real solids.
