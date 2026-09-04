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

## REVISION 3 (2026-09-04) — the stored F does not track the real deformation

On the first real-volume run (v1 solid bunny, converged): stored det F ≈ 1.00 everywhere,
but the Lagrangian volume ratio measured from particle positions (kNN spacing now vs in
the source, cubed) is **1.32 in the ears and 0.94 in the body**; 75% of particles have
|J_stored − J_true| > 0.2. The ear slab holds 81% of the target mass; the D_vol gradient in
the ears is LARGER than in the body and points toward the tips — the optimizer sees the
deficit, the physics does not realize it. Cause: `k_update`'s F-smoothing (s = 0.955,
inherited from v1) is a temporal EMA that integrates 4.5% of each increment, so the stored
F lags the true deformation; the sKL volume prior, the isochoric assimilation and the
corotated elastic response all act on that fictitious F. The ears were not filled by
transport; they were stretched thin (true J 1.32) while the body compressed. The earlier
"s ≤ 0.8 falsified (J collapses)" probe result is the same fact from the other side: with
an honest F the constitutive model feels the true strain and the current control/stiffness
regime cannot carry it. Remedy under test: a volume prior on the DENSITY-measured J
(ρ0/ρ from the CIC mass grid, differentiable through the rasterization) so mass is pushed
from the over-dense body into the under-dense ears regardless of what F says.

## REVISION 3, amendment — the density prior is not the lever; the objective is LOCAL

v5 (prior calibrated at the source, J≡1: zero gradient → froze at a9) and v5b (calibration
deferred until the prior gradient is measurable: from a2 every candidate is rejected — the
prior's descent direction is "undo the last step" because the source is the one state where
J≡1, and at equal norm it cancels the data term; reversal 0.93). A penalty on local density
cannot route mass from the body into the ears, whatever its weight.

The measurement that reframes the ear deficit (v3 solid 40k): the d_vol gain per accepted
commit decays geometrically (0.69 → 0.26 → 0.10 over commits 40-80 / 80-120 / 120-160), so
the run does not converge slowly — it converges to a WRONG fixed point (extrapolated floor
≈ 54 vs 30 at 20k; ear-region fraction 0.069 vs 0.100). Every term in the objective is local:
D_vol is an L2 mismatch of rasterized mass (its gradient reaches one cell width beyond a
mismatch), d_dt vanishes inside the target, d_sil sees silhouettes only, the density prior
acts at the particle. A body particle that is 6% over-compressed sits in a region with no
mismatch and feels nothing; the ear deficit's gradient acts only on the particles already in
the ears (REVISION 3: "points to the tips, unrealized"). Transport from body to ears must
be carried by material coupling through P2G/G2P — a few cells per commit — which is
precisely the "dilution" the user named. The remedy is a NON-LOCAL mass-balance signal
(Poisson/W1 deficit potential: ∇²φ = ρ − ρ_target, push along −∇φ; the standard
projection/shifting route in the particle literature), not another local penalty.

**Measured (band_push probe, v3 40k delivered state, loss grid 64³/0.5 wu).** Descent
component OUTWARD from the surplus band's centre along the long axis, divided by the mean
gradient magnitude, for the band's 13.4k particles (|y|<0.5): D_vol (L2) +0.05 (core +0.15;
28% of band particles outward-dominant; the densest slab [−0.5,0) points INWARD, −0.12);
H⁻¹ +0.46 (core +0.47; 53% outward-dominant), and +0.3…+0.87 in every interior slab from
−2.5 to +2.0. The L2 term is blind inside the uniform surplus; the H⁻¹ term drives every
band particle poleward. The gradient's mass share in the band: 0.20 (L2) vs 0.41 (H⁻¹).
Ear width is 1.27 wu = 2.5 cells at dx 0.5 — a separate resolution suspect, deferred.
