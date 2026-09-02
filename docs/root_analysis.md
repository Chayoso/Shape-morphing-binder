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
