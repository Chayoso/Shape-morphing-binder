# D175 next measurements: stated before any run (2026-10-29, after the 208 FAIL; coordinator's GO)

## (a) Does a blurred-picture comparison carry the render's pull into the bulk?

Comparisons, each read on the body's exterior discs exactly as the run reads them (a field found at the state, its
discs, the soft silhouette), against the keyframe's pictures P_k (the mean of 8 samples):

- **current**: the run's term, silhouette (w_hole 2, w_spray 1) + shading;
- **blur σ**, σ ∈ {2, 4, 8} px of the 96 px pictures: mean (G_σ α − G_σ α_k)²;
- **pyramid**: Σ over σ ∈ {0, 1, 2, 4, 8} of the blur-σ term over its own value at the state (Treuille's multi-scale blur;
  each level weighs equally at the state);
- **sdf**: mean α · SDF_k, with SDF_k the keyframe silhouette's signed distance in pixels / res (positive outside). It pulls
  the outline outward where the keyframe lies beyond it and inward where it does not, with a strength growing with the gap;
- **reference, 3D target**: the transport + proximity toward the keyframe sample as a 3D target pack (the physics
  objective; what the bulk would get from a 3D keyframe).

Two readings:

1. **Position pull** on S97's dumped segment states (windows 1-5, job 208_dump): the gradient of each comparison with
   respect to the particle positions.
2. **Through the dynamics**, at the source state (window 1's start, exact): the method's segment gradient (soundness +
   λ held × PCGrad(render)) with each comparison in place of the render term. Then Adam's first step
   g / (|g| + eps), the run's step length and clip, and one rollout. Δx = x_T(step) − x_T(0) is what that comparison
   actually moves.

Per comparison:
- the share of |pull|² (reading 1) and of |Δx|² (reading 2) per pitch layer below the body's convex hull: 0-1, 1-2, 2-3,
  3-5, > 5 pitches;
- the cosine with the keyframe's displacement field (A − I)(x − c), over all particles, per layer and below 3 pitches;
- the share explained by one linear field M (least squares), and cos(M, A − I);
- reading 2: the one-step aspect gain.

**"Reaches the bulk"** (decided on reading 2, the step the optimiser takes) means both:
- REACH: the share of |Δx|² below 3 pitches is ≥ max(0.25, 2 × the current comparison's);
- ALIGN: cos(Δx, stretch) over the particles below 3 pitches is ≥ 0.3 and ≥ the current comparison's + 0.1.

Only a comparison that passes both gets a pre-registered arm. The arm would be: the blur inside the keyframe comparison
only, the final-target render term unchanged (gate bit-SAME), 40k, seeds 97 / 99, the 208 bars. If several pass, the one
with the largest one-step aspect gain is chosen.

## (b) S99's replay nulls

Keyframe runs now record, at every commit, E_final − E_accept split into
- Δlv (the end drift; in a segment it carries no transport),
- Δstab,
- λ Δrender,
- the rest (cleanup),
against the tolerance (10 × replay_rel × |E|), and whether the commit read the render on a disc set found anew.

The record is off the frozen path: gate G1-G4 bit-SAME again. Reruns: S99 × 2, S97 × 1 (the 208 flags).

## (c) Is 1 s physically enough for a 1.48 aspect?

- dF_c alone (`--render_weight_scale 0 --u_off`), the same Hu flags, sphere → the stretched sphere as the 3D target
  (assets: the isosphere's vertices × diag(1/√1.3, 1.3, 1/√1.3), renormalised by prepare as any target is). 40k, seed 97.
- Read: the body's second-moment aspect at every kept frame, and the frame at which it reaches 1.40 and 1.48.
- Enough = aspect ≥ 1.40 by frame 240.
