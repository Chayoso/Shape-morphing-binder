# Observability probe — which viewer-visible defects does each render loss SEE?

Script: `scripts/probes/observability.py`. States: the final frames of
`b2_flag_paced` (flagship window, 300 paced commits, chamfer 0.108) and
`b4_paced450` (450 paced, ended mid-cycle at 0.146). N=20k, gauss res 384 (Nyquist
floor), gauss parents = frozen source-surface subset (10,234 of 20,000), 4 children.
Floater = target-NN distance > 2× target spacing.

## Per-particle gradient energy and coverage

| subset (b2 / b4) | n | silhouette: energy share · seen(>0.1·median) · zero | gauss: energy share · seen · zero |
|---|---|---|---|
| all | 20000 | 1.00 · 0.57 · 0.07 | 1.00 · 0.51 · 0.49 |
| floaters >2sp | 4452 / 8077 | 0.24 / 0.46 · 0.57 / 0.62 · 0.02 / 0.03 | 0.34 / 0.65 · 0.49 / 0.52 · 0.51 / 0.48 |
| far floaters >3sp | 1743 / 3972 | 0.13 / 0.26 · 0.54 / 0.64 · 0.00 / 0.01 | 0.14 / 0.45 · 0.46 / 0.53 · 0.54 / 0.47 |
| surface parents | 10234 | 0.92 · 0.65 · 0.06 | 1.00 · 1.00 · 0.00 |
| interior parents | 9766 | 0.08 · 0.48 · 0.09 | **0.00 · 0.00 · 1.00** |
| floater & interior | 2261 / 3843 | 0.04 · 0.49 / 0.52 · 0.02 / 0.05 | **0.00 · 0.00 · 1.00** |

## Direction on floaters (cos between −grad and the vector to the nearest target point)

| loss | floaters with nonzero grad | median cos | frac cos > 0.5 |
|---|---|---|---|
| silhouette (b2 / b4) | 0.98 / 0.97 | +0.26 / +0.38 | 0.36 / 0.43 |
| gauss (b2 / b4) | 0.49 / 0.52 | +0.22 / +0.39 | 0.35 / 0.44 |

## Reading

1. **Floaters are NOT in the silhouette's null space.** 97–98% of floaters carry a
   nonzero silhouette gradient and they hold 24–46% of its energy (over-represented
   relative to their 22–40% population share). root_analysis §1's "floater = loss
   null space" is false for the silhouette.
2. **The gauss loss with surface-only parents is blind to half of the floaters by
   construction**: every interior-parent floater (2261 / 3843) gets exactly zero
   gradient. The viewer renders EVERY particle, so the objective's observation model
   was deliberately narrower than the truth standard.
3. **The problem is DIRECTION, not presence.** On floaters the image-space losses
   point only weakly toward the surface (median cos ≈ 0.3; only ~40% of floaters
   have cos > 0.5). A projected-coverage loss pushes a floater wherever the ray's
   alpha mismatch shrinks — sideways in image space — not toward its nearest target
   point. The per-particle assignment term (nn-band) is the one mechanism whose
   direction is cos ≈ 1 by construction, which is why it was the only thing that
   ever killed a floater class (fork halo 326 → 78).

Combined with docs/probes/transfer_function.md (adjoint attenuation small),
sobolev_precond.md (render term descends at the same relative rate as physics per
accepted step; preconditioning a no-op; s ≤ 0.8 falsified) and material_carrier.md
(Tier M dead): the render channel reaches the control, steers at parity, and sees the
floaters — the residual defects are a composite-objective LOCAL MINIMUM (weakly
directed image gradients balanced against physics) plus an observation model that
omits interior parents, not a transfer failure.
