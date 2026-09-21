# The last implementation — plan (2026-09-21, after the render × u factorial)

Written at the user's request ("prove everything unproven first, then organise how the last
implementation should go"). Evidence: `docs/surface_gradient.md` §6–§10, `docs/experiments.md`
2026-09-19 … 21. Everything below is at 40k; 150k stays excluded until the user re-opens it.

## 0. Where the evidence stands

- **The render channel changes the control and the outcome** (§10, current recipe, three
  targets, against an identical-configuration re-run): a third of every accepted control
  update is the render direction (g_share 0.35–0.40, cosine 0.01–0.15 to the physics
  gradient — deterministic per window); silhouette IoU +0.6 … +2.0 points, 2–8 spreads;
  det F less compressed on dragon and bob; chamfer unchanged; the gain is lost when the
  channel is switched off at a third of the run. It does not lower the level the density
  term reaches (only on bunny, −12 %), and its trajectory signature is below the chaos floor
  (the cut twin diverges exactly as an identical re-run does, 0.4–1.0 spacings by the end).
- **Through the physics (the stress control) the channel carries the outline and the
  features down to the cell** (3.6 spacings at 40k): the whole outline gain survives with
  the u channel off; the dragon's ridges improve with u off (dcorr +0.28 → +0.35).
- **Below the cell the actuator is u**, the per-window normal displacement of the outer
  layer (§7): necessary on bunny (without it a twentieth of the surface stays five spacings
  from the true mesh — the base the cell sum cannot resolve), neutral on dragon, harmful on
  bob (normal error 5.6° → 7.0°, detail correlation +0.32 → +0.26, a visibly bumpy ring), and
  the main source of outer-layer roughness on all three (morph-mean layer RMS 0.19–0.22
  spacings without it, 0.25–0.33 with it; the render channel alone adds 0.01).
- **The surface pipeline is settled** (§10.12): outer layer → same-side PCA pulling → screened
  Poisson → mass rule; QA passes on 19 targets' videos; stratified sampling removed the
  target shot noise (§9); the relaxation holds the layer at 1/T per step (§6).

## 1. The one open structural problem: the sub-cell actuator is not physical

u is a position projection the constitutive model never sees: `k_layer_project` moves
x by (u/T) n per step and F is not updated, so a rough u costs no strain energy — nothing in
the physics resists it, the relaxation removes only the part that is rough on the local
plane, and both gradients feed it: the cell sum at sub-cell granularity (the same signal
that caused the ejections, method.md §10.6–10.9) and the half-rough render covector (§6).
"Rendering controls physics" is therefore complete for the outline and incomplete for the
surface: the sub-cell path bypasses the physics instead of driving it.

What the actuator must satisfy: (a) it is part of the physics — the material resists it
and the adjoint reaches it through F; (b) its driver has real sub-cell content; (c) it
still finishes what the cell sum cannot (bunny's base). Every constant derived from the
discretisation (no per-shape tuning).

## 2. Candidates

**P3 — u through the deformation gradient (recommended first).** After the projection,
F_p ← (I + ∇δ_p) F_p on the layer, where δ_p = (u_p/T) n_p is the step's u displacement and
∇δ_p its least-squares gradient over the layer neighbourhood already built per window
(`layer_relax_data`: K = 24 neighbours, weights w, radius 2 spacings; M_p = Σ_a w_a r_a ⊗ r_a
inverted on the host per window — no division in kernels, the neighbourhood is frozen per
window anyway). A rough u then produces strain, the stress resists it in the following
steps, the stress control and the assimilation respond, and the adjoint passes through F
(the tape already differentiates F). The relaxation projection stays outside F (a constraint,
like contact). Cost: one kernel per step on 8–16 % of the particles. Predictions (fx spread:
IoU 0.5, layer RMS 0.02, hp_res 0.012, dcorr 0.03): layer RMS with u within 0.02 of the u-off
value (bunny morph 0.27 → ≤ 0.22, bob ≤ 0.22 and end ≤ 0.15), bob n_dev ≤ 6.0° and dcorr ≥
+0.31, bunny d_95 ≤ 1.5 and hp_res ≤ 0.17, silIoU within 0.5 of `fx_11`. Falsifiers: the
layer stays ≥ 0.25 (the assimilation erases the strain within the window, the resistance is
too short-lived) or bunny's d_95 returns above 3 (the resistance blocks the finishing).

**P1 — u driven by the render channel only (the cheap ablation, run alongside).** Detach
the physics loss from u: the cell-sum loss lives on the cell grid and has no legitimate
sub-cell content, so its u-gradient is the granularity signal. Prediction: bob's u roughness
falls (`fx_01_bob` end 0.44 → ≤ 0.2 spacings). Risk, measured by d_95: bunny's base finish
weakens — the factorial shows the physics gradient through u does finish it (λ = 0: d_95 1.82
with u, 5.37 without).

**P2 — the saturation gate (fallback).** u only where the cell containing the particle has
a cell-sum residual below one particle mass — the smallest change the cell sum can
register, derived from ppc — i.e. where the physics path has nothing left to say.
Prediction: bob (a thin ring the cell sum satisfies early) receives little u; bunny's base
(a persistent residual) keeps it.

**P5 — drop u.** Rejected unless P1–P3 all fail: bunny's base stays unfinished (d_95 5.4).

Falsified already and not revisited: a particle force for the layer (2.4 % of a bump per
window, averaged away by P2G/G2P), the hard per-step constraint (diverges), the u-step
projection through W (removes the channel's structure with its noise), the quadratic splat
kernel (smoother covector, rougher surface).

## 3. Test protocol (pre-registered when launched)

- The fx design at 40k: each candidate on bunny / dragon / bob, one run each (~10 min),
  read against `fx_11` / `fx_11c` (the spread) and `fx_10` (u off) with the same readings
  (`fx_summary.py`, `layer_rms.py`, `surface_gt.py --gt_all`, QA columns). Three candidates ×
  three targets = 9 runs, ~40 min on two GPUs.
- Decision: adopt the candidate that meets its predictions on all three targets; if two do,
  the one with the lower layer RMS; a candidate that loses bunny's base (d_95 > 3) is out
  regardless of smoothness.
- Then the λ = 0 twin of the adopted recipe on the three targets — the standing proof column
  (g_share + outcome against the spread) so the final page carries the render-controls-
  physics evidence under the final recipe.

## 4. The deliverable phase (after the u decision)

1. Freeze the recipe in `scripts/ops/hyde06_env.sh`; `docs/method.md` §10.13 for the u channel
   as a physical surface actuator (equation, discretisation numbers, the adjoint); §10.12
   already covers the surface.
2. The 40k gallery of the 19 targets with the final recipe: per run the QA columns, the
   Poisson video (`photoreal_batch.sh`), the per-window g_share; λ = 0 twins for bunny /
   dragon / bob (from §3).
3. The report page: gallery + QA + the proof column + the surface metrics against the true
   mesh; `docs/experiments.md` summary; published.
4. The cleanup sweep the user asked for at this milestone: archive logs, delete the
   falsified-ladder outputs on hyde06 and locally, list what was deleted in
   `docs/experiments.md`.
5. 150k stays excluded until the user re-opens it; when re-opened, the same recipe
   (`--cell_diag 26` gives ppc 91 at 150k) and the same surface pipeline; the Poisson videos
   are the slow step.

## 5. What this plan does not do

- No per-shape constants: the u clip (one spacing), the layer thresholds, the iso level and
  the gate in P2 are all derived from the discretisation.
- No metric consumes the renderer: `surface_gt`, `layer_rms` and the QA columns are geometric.
- No trajectory-divergence claims: the chaos floor equals the intervention's signature at
  40k and 150k; the proof is the per-window share plus the outcome against the spread.
