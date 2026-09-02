# v3 rationale — why this should work, and what would falsify it

*Pre-registered 2026-09-01, BEFORE the v3 A/B ran. Predictions and risks below are
commitments; `experiments.md` records what actually happened.*

## 1. The claim under test

Two additions to the validated v2 dynamic pipeline, and one reformulation:

- **`render_ws`** — warm-start each window's control sequence from the previous solution.
- **`render_gs`** — replace the raw image-space render pull with a Sobolev-metric
  (grid-smoothed) direction before the adjoint pullback.
- **`vbd` / `vbd_phys`** — replace the dynamic rollout entirely: each commit is a grid
  equilibrium of `E_elastic + D_vol + λ_R·D_render`, solved by colored block descent.

## 2. Why each should work (premises → mechanism)

**Warm start.** Premise (measured): λ_R and the gradient field vary smoothly across
commits (λ 1130→61 monotone; loss components monotone). Consecutive windows therefore
solve nearby problems, and dFc=0 restarts discard a near-valid solution every commit —
8 Adam iterations currently re-derive what the previous window already knew. Mechanism:
initialization in the basin ⇒ the same budget spends on *refinement*.
Prediction: same-budget loss strictly below `render` on most commits, or equal quality
in fewer accepted iterations. Falsifier: warm-started windows REJECT more (stale controls
fight the assimilated state).

**Grid-GS render direction.** Premises (measured): (i) the render pull is 8–16×
surface-concentrated with dead zones elsewhere — a Jacobi signal with holes; (ii)
cos(∇render, ∇vol) → −0.74 late, so the render direction increasingly needs to be
trusted on its own; (iii) v1's tearing came from many particles answering the same pixel.
Mechanism: screened diffusion `(I+κL)⁻¹` is an SPD metric change — it fills dead zones
with neighbour-consistent pull and damps the double-counting mode, i.e. exactly the two
known defects of the raw signal. The pullback still goes through the true MPM adjoint, so
physics claims are untouched; the line search bounds the worst case at "wasted iteration".
Prediction: fewer line-search rejections and/or better sil_iou at equal budget; sharper
hole-closure early. Falsifier: smoothing *blurs the thin-feature signal* (ear tips) —
sil_iou flat but ear region visually worse than `render`.

**VBD-MPM.** Premises: (a) the dynamic path already runs quasi-statically — w_kin=5
suppresses momentum and kin ends at 0.002, so the T-step rollout is an expensive way to
find a near-equilibrium; (b) implicit MPM as grid-DOF energy minimisation is established
(Gast15/HOT), VBD's node-block descent is established on meshes — the transplant to grid
nodes inherits both; (c) our assimilation ratchet (`F_e → R_e S_e^{1−η}`) provides the
progress mechanism between equilibria; (d) the solve's energy IS the objective, so the
"render feedback → physics" coupling is exact by construction, not balanced through an
adjoint. Mechanism: each commit finds the elastic-coherent configuration that best
satisfies mass + silhouette; plasticity bakes it in; repeat.
Prediction: zero oscillation/ejection by construction (no velocities), G2/G3 trivially
clean, competitive chamfer/sil_iou at a fraction of the wall-clock, and monotone
per-commit energy traces. Falsifier: equilibria stall (local minima without momentum) —
D_vol plateaus early at worse chamfer than `phys`.

## 3. Risk register (watch → mitigation)

| # | risk | symptom to watch | mitigation / fallback |
|---|---|---|---|
| R1 | GS smoothing washes out thin-feature signal | ear/paw region in G6 strips; sil_iou flat vs `render` | κ↓, iters↓; coarse-to-fine κ schedule |
| R2 | norm-preserving rescale inflates a near-zero smoothed field | line-search rejection spike in `render_gs` | skip smoothing when ‖g‖ below floor |
| R3 | smoothed direction not a descent direction for the COMBINED loss | rejections; loss stalls | line search already bounds this; drop to raw g |
| R4 | warm-started dFc stale after assimilation changed the state | first-iteration rejections in `render_ws` | decay init (dFc·0.5) or re-zero on rejection |
| R5 | CIC (C0) kinematics → checkerboard/hourglass grid modes | F aniso banding, visual ripple in `vbd` | quadratic B-spline stencil; mild u-Laplacian tax |
| R6 | frozen stencil invalid for large per-commit motion | `move` per commit ≫ dx; detF outliers | data terms bound equilibrium offset; cap via λ_R |
| R7 | 2-color parity ≠ true GS for the 2-cell CIC coupling | slow ‖∇E‖ decay, sweeps hit cap | more colors; Chebyshev/multigrid (HOT) later |
| R8 | uniform V_p biases elastic energy where density varies | none expected at ppc≈8 | per-particle V_p from mass P2G (eq 10) |
| R9 | quasi-static local minima (no momentum to escape) | early plateau freeze at high D_vol | anneal λ_R; one dynamic "shake" commit; accept |
| R10 | per-commit λ at u=0 overweights render right after assimilation (elastic grad ≈ 0 there) | λ spikes in `vbd` history | EMA already damps; floor the phys norm |
| R11 | cross-family comparison unfair (budgets differ) | — | report wall-clock per arm alongside metrics |

## 4. Round-1 outcome (2026-09-01, full scale, hyde06 — predictions vs observed)

| item | predicted | observed | verdict |
|---|---|---|---|
| baselines | reproduce | phys 0.1785/0.887, render 0.1371/0.9605, all gates PASS | ✓ |
| warm start | R4: stale controls fight assimilated state | **R4 confirmed, worse than predicted**: dFc is an ABSOLUTE control → verbatim reuse double-applies (Jmin 0.71→0.37→1e-4, loss →177279, 9 inversions ACCEPTED because det(F) was invisible to acceptance) | falsified as implemented |
| grid-GS | fewer rejections / better sil_iou | **untestable** — arm config coupled warm_start=True (design error), identical death signature | confounded, rerun |
| VBD | competitive chamfer, clean gates | **total freeze**: gnorm bit-identical every commit, move=0. Fringe nodes with Σw≈0 → diag=floor → runaway d → every color candidate rejected (small-node-mass pathology, Steffen 08); plus at young=1.4e5 the equilibrium offset is ~1e-3/commit — could not have progressed anyway | falsified as implemented, 2 root causes fixed |

New risks recorded: **R12** small-weight fringe nodes poison per-color acceptance
(fix: weight-thresholded active set + relative diag floor + per-node trust radius);
**R13** quasi-static stiffness scale — elasticity must be a coherence regulariser
(`vbd_young≈2e3`), not the dynamic material, or equilibria cannot move. Fixes applied the
same day: safeguarded decayed warm start (compare-to-cold + orientation check),
det(F)>0 added to line-search ACCEPTANCE (defect existed in v2 too — any arm could have
committed an inversion the loss cannot see), render_gs decoupled from warm start.

## 5. v4 main-line reinforcement (pre-registered 2026-09-01, before the batch ran)

Target: PhysMorph-GS **improved** — current code is the PhysMorph baseline. Every quantity
and gradient must stay logically consistent; the win must show in metrics AND visuals AND
the trajectory (no 2-commit snap).

| axis | mechanism | papers | prediction | falsifier |
|---|---|---|---|---|
| curvature-blind silhouettes (round ear tips) | **PBR-lite**: normals from mass-field gradient (n=−∇ρ/&#124;∇ρ&#124;), headlight-Lambertian shaded-image L2 in the balanced render scalar | [SDFDiff](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_SDFDiff_Differentiable_Rendering_of_Signed_Distance_Fields_for_3D_Shape_CVPR_2020_paper.pdf), [nvdiffrast](https://arxiv.org/abs/2011.03277), SoftRas | ear/paw sharpness up at equal budget | shading noise destabilises λ or detF margin |
| render-vs-phys gradient conflict (cos→−0.74) | **PCGrad one-sided** projection of the render grad | [PCGrad](https://arxiv.org/abs/2001.06782), [CAGrad](https://arxiv.org/pdf/2110.14048) | fewer rejections late-run, sil_iou ≥ render | projection strips the very correction that beats phys |
| thin-feature resolution | **coarse-to-fine** render targets (64→96px at half-run) | classic multiresolution | sharper extremities, no early-run cost | D_render rescale confuses freeze (mitigated: track reset) |
| λ runaway at render saturation (measured 1.77e5) | **λ cap** 5e3 | — (own finding) | late-run λ bounded, no volume oscillation | oscillation persists ⇒ needs proximal/AA instead |
| trajectory snap (D_vol mostly spent in 3 commits) | **pacing**: per-window loss cut is a TRUE upper bound — overshooting line-search candidates are rejected and halved (the break-after-accept form was refuted: one step could still snap); move_cv + first3 metrics, `render_pace` isolates it | (deliverable-driven) | move spread across commits, endpoint quality intact | paced arm loses fidelity or trips freeze early |
| particle/volume oscillation at optimum | λ cap + λ-free freeze (existing); **guarded Anderson on CONTROLS** in backlog if needed | [AA for physics (TOG18)](https://arxiv.org/abs/1805.05715) + guarded variants | cap alone suffices | else implement guarded AA (line search = the guard) |
| Gaussian-native render loss | backlog: diff_gauss alpha/depth in the loss (server build exists) | GIC, 3DGS-LM | — | — |
| PLAN-B if tranche falsifiers fire | render gradient as a **generalized external force** (per-step body force through grid momentum; force cap tied to material strength, paired with physical damping; window-frozen f keeps the dFc adjoint clean) — or as feedback **internal stress** (dFc ← γ·J_render, "active material"). Design space: {force vs stress} × {adjoint-optimised vs per-step feedback}; the main line is stress×adjoint, VBD was force×equilibrium; plan-B fills force×feedback. Honesty line vs v1: forces pass through the momentum equation (legitimate physics); v1's sin was velocity SERVO + geometric surgery | [Treuille 03 (force keyframe control)](https://grail.cs.washington.edu/projects/control/) vs [McNamara 04 (adjoint)](https://dl.acm.org/doi/10.1145/1015706.1015744) — the exact dichotomy; eigenstrain/growth-tensor lineage for the stress form | dense-in-time feedback gives natural pacing + no chaotic-horizon adjoint | tearing at high gain (v1 regime) — the force cap is the guard |
| adaptive/high-res grid | backlog after c2f verdict | SPGrid/adaptive MPM lineage | — | — |

**Adversarial round 2 (v4 tranche, Opus, 13 findings — all accepted before the batch):**
headlight vector was exactly inverted (l·d=−1, backlit); render_full bundled 5 features
(render_pace arm added; dfc_clip remains only in render_full — noted); move_cv scored the
2-commit snap as perfect (now inf-sentinel); pace was a lower bound (now enforced in
acceptance); PCGrad de-weighted render 33% via raw-norm λ (now λ from projected grad);
"surface-dominant" refuted on solid clouds (normals now carry a surface WEIGHT used in the
splat); depth-blindness measured at 1.2× noise floor (soft front-bias visibility added —
approximate, claim softened); pbr_ambient wired; d_render/d_pbr telemetry split; cfg
snapshot before run; c2f resets stale + logs its switch. Tests 45 (headlight regression,
surface-weight discrimination, PCGrad math, pace floor).

**v4 arm adoption rules (pre-committed):** each arm adopts only if all gates clean AND
chamfer within 2% of `render` AND its OWN axis improves: `render_pbr` — ear/paw region
visibly sharper in G6 (silhouette metrics may tie); `render_pc` — late-run rejected steps
strictly fewer at equal sil_iou; `render_c2f` — sil_iou up; `render_pace` — first3 ≤ 35%
and move_cv ≤ 0.8 with chamfer within 5% of `render`; `render_full` — must beat `render`
on ≥2 axes with none regressed.

## 6. Decision rule (pre-committed)

- `render_ws`/`render_gs` **adopt** if: G2–G4 clean AND (sil_iou ≥ `render` − 0.002 with
  fewer rejections, or sil_iou > `render`). Ear-region visual regression vetoes (R1).
- `vbd` **promote to co-headline** if: Gc convergence every commit, G2 clean, chamfer
  within 10% of `render` at ≤ wall-clock parity, visuals hole/flicker-free. Otherwise it
  remains a solver-research arm and the dynamic family stays the deliverable.

## 7. Fringe tranche pre-registration (2026-09-01, after hero1/hero2)

**Finding chain:** hero-scale stray_max is flat under both lg (propagation) and w_creg
(creation-side smoothing) — 0.367/0.370/0.367%. Root cause isolated to the LOSS, not the
control: α = 1−exp(−k·w) saturates ~k·w for sparse mass, so the spray penalty
relu(α−α_t)² fades quadratically with sparsity. A lone stray between the ears is
asymptotically invisible to every render view AND to D_vol (sub-cell mass at loss_res 64).

**Mechanism (structural, not a weight change): asymmetric saturation.** Hole side keeps
the saturated α (presence detection — a hole is a hole regardless of how much mass is
missing). Spray side adds an UNSATURATED, mass-linear penalty supported strictly outside
the target silhouette: constant per-unit-mass gradient independent of sparsity, so one
particle feels the same pull as a clump. w_creg (adopted) then propagates that pull to
its frozen-topology neighbours instead of letting one particle be yanked alone.

**Adoption rule:** hero3 (`render_full_creg` + linear spray) adopts iff stray_max < 0.2%
(the original G4 gate) with chamfer within 2% of hero2 (0.0940), no gate regression, and
the between-ears region visibly clean in G6. Falsifier: if linear spray drags legitimate
thin-feature mass (ear tips) inward — visible as ear erosion / silIoU drop — the fix is
wrong and plan-B's force-form cleanup is next, NOT a weight retune.

**Literature verdict (4+ papers read, 2026-09-01) — design refined to DT-weighted W1:**
the adopted form is mass x **distance-transform** outside target support (not flat-linear):
[DRWR (ICML 20)](http://proceedings.mlr.press/v119/han20b/han20b.pdf) states our exact
failure ("points far outside the silhouette receive zero gradients") and fixes it with a
per-point unsigned-DT loss, flat inside / linear outside — our asymmetry, exactly;
[3DGS-as-MCMC (NeurIPS 24)](https://arxiv.org/abs/2404.09591) kills residual floaters with
an L1 opacity term because a CONSTANT gradient works where the photometric gradient has
saturated away (their lambda=0.01 — start small); [Mip-NeRF 360](https://arxiv.org/abs/2111.12077)'s
distortion loss gives an isolated lobe a gradient set by DISTANCE, not magnitude;
[Sinkhorn divergences (AISTATS 19)](https://arxiv.org/abs/1810.08278) prove density-blurred
losses screen extreme support points while OT-type costs give them full displacement
gradients — mass x DT IS a pointwise W1 cost. Notably, PhysMorph-GS **v1 already had
L_DT = alpha_p * DT(p)** (linear in alpha, signed-DT); the v2 rewrite lost the
sparsity-invariance when it replaced it with the saturated asymmetric silhouette. This
tranche restores the v1 mechanism in its theoretically clean form (unsigned, clamped,
outside-only, fixed weight OUTSIDE the lambda channel — lambda->cap x constant gradient
would be [arXiv:2409.15746](https://arxiv.org/html/2409.15746v1)'s documented mass-ejection
mode, hence the cap-independence and the dt_clamp handoff to w_box).

**Propagation verdict:** [Chamfer structural-failure (arXiv:2603.09925)](https://arxiv.org/html/2603.09925)
Corollary 1 proves LOCAL regularizers (kNN/Laplacian/repulsion) cannot rescue a
neighbour-less particle — non-local coupling is required. Our adjoint already provides it:
the DT pull enters through the MPM tape, so P2G/G2P spreads a lone particle's correction
through grid nodes. w_creg is hereby demoted to a bulk-regularity term (its isolation win
stands); it is NOT the fringe mechanism, and no extra propagation machinery is added.

### §7.1 2D→3D pivot (forensics, 2026-09-01) + Codex round answers

**2D multi-view DT falsified before adoption:** on hero3's final state, only **1.8%** of
ear-region strays had DT>0 in ANY of the 18 views, at every mask threshold (0.05/0.2/0.5)
— the concavity between thin features lies INSIDE the visual hull, so no silhouette-based
term can see it. The 3D loss-grid DT sees **100%** (meanDT 1.5–2 cells). DRWR/v1 used 2D
because their supervision was silhouettes-only; we have a volumetric target — ported to
3D (`losses/volumetric.py: target_dt_grid, d_w1`), same W1 theory, no hull blindness.
hero3 (2D term): stray 0.367% = hero1 exactly, confirming inertness end-to-end.

**Codex (gpt-5.6-sol xhigh) round — 15 findings, all answered:**
1,2 (target self-force via sub-threshold alpha / boundary subgradient): moot in 2D form;
in 3D, support = the SAME CIC stencil the sampler gathers, so loss AND grad vanish on the
target by construction — pinned by `test_no_target_self_force` (rim subgradient bounded
< 1 cell, <5% of particles). 3 (Lipschitz bound is √2·w_dt not w_dt): correct; in 3D the
bound is √3·w_dt — documented, not load-bearing. 4 (force-free gap between DT clamp and
box leash): correct and ported — dt_clamp_frac 0.25→2.0, no interior plateau
(`test_no_force_free_gap_inside_box`); beyond the box the quadratic leash dominates the
linear tail (intentional overlap, not double-counting). 5 (pace floor counts a DT
plateau constant): with clamp 2.0 the plateau lives outside the box (negligible mass);
early-window DT is genuinely reducible — accepted as residual, documented. 6 (freeze
blind to DT): correct — d_dt now computed on every ARCHIVED state, logged, and included
in phys_track. 7 (lg pass can undo DT progress, assimilation ratchets it): correct —
lg_sweeps>0 with w_dt>0 now raises (the exact-quadratic local solve cannot host a
non-quadratic term); lg is parked anyway. 8 (post-local composite stale): subsumed by 7's
guard. 9 (DT inflates λ numerator + PCGrad reference): correct — gradient assembly split
into phys_core / dt_term; λ and PCGrad use phys_core only, W1 joins after projection.
10 (c2f stale balancer EMA): DT is now built from the loss grid — render-res switches
no longer touch it; the balancer-EMA-across-c2f staleness predates this tranche, logged
as a known issue. 11 (outside-viewport clamp kills outward gradient): moot — no
projection in 3D; the DT grid spans the MPM domain. 12 (w_dt silently off when λ=0):
fixed — dt3 builds independently of the render channel
(`test_w1_independent_of_render_channel`). 13 (masses unused; "W1" naming): masses now
applied; the term is the one-sided W1 transport bound (no target-capacity constraint —
capacity is D_vol's job; documented). 14 (weak tests): ported suite `test_w1_cleanup.py`
= self-force, fixed-N invariance, direction, gap, monotonicity + causal d_dt/λ-free/lg
guards in the smoke tests. 15 (C3 pixel convention): survived — moot in 3D regardless.

### §7.2 Opus round (REFUTE, 15 findings) — the two that mattered, all answered

Opus caught what Codex and the forensics both missed — the term was structurally right
and NUMERICALLY OFF: (1) the mean normalisation gave each particle authority w_dt/N,
measured 300-3270x below lambda*D_render / d_vol at the shipped weight — every "the DT
term didn't move the hero" verdict (hero3/hero4) was an artefact of an accidental no-op,
not of the mechanism; and (2) on the coarse loss grid (ldx~1 world unit) CIC dilation +
the flat trilinear cell created a ~1-unit DEAD RADIUS: 72-90% of the production fringe
band (0.1-0.6 units off-surface) felt zero gradient — my forensic had conflated DT
*value* visibility with *gradient* visibility. Fixes: d_w1 is now a SUM (per-particle
pull = w_dt exactly, N-invariant, smoke-to-production transferable; 3DGS-MCMC's
L1-opacity is also a sum) with default w_dt 0.2 (Opus gradient-parity estimate
3000-5000 in mean form / 20000 particles); the DT lives on its OWN fine target-fitted
grid (dt_res=160, cell ~0.019*extent, spanning 1.5x extent so the whole leash interior
is on a live slope), decoupled from loss_res. Sub-cell fringe-regime gradient is now a
test. Remaining Opus items: separate best_dt freeze track (f4 refinement — folded-in
form sat at the tolerance noise floor); d_dt streamed to the live viewer with its own
series (f8); empty-support assert (f11); scalars() no longer builds a second graph
(f14, and the +1 adjoint replay per iter from the phys_core split is documented cost);
border-clamp direction distortion (f13) accepted — beyond the DT cube the leash owns
the far field; sil_k/mask coupling (f15) moot in 3D (no alpha threshold). C2 (ear
erosion) survived both reviews: support = the sampler's own CIC stencil, self-force
zero by construction (tested). hero4/p3_* (2026-09-01 evening batch) are struck from
the record as no-op-weighted + dead-zoned; v7 reruns supersede them.

### §7.3 Dose-response falsification of the UNGATED sum → complementarity gate

Calibration ladder (N=20k, bare render_dt): w_dt 0.05 / 0.2 / 1.0 →
stray 0.77 / 4.03 / 7.82%, G3 drift FAIL at all three, 4 trajectory inversions at 0.2,
first3 45/69/100%. Monotone dose-response = the mechanism itself, not the dose: at
window 1 roughly a third of the source lies outside target support, and a constant
per-particle force on that COHERENT mass double-drives bulk transport — snap,
compression inversions, and (per the porosity forensics) violent transport CREATES the
very fringe the term exists to remove. Notably w_dt=0.05 delivered chamfer 0.1012
(28% under the render baseline) — the pull does carry shape information — but through
the wrong channel. Fix is structural, not a weight: a per-particle COMPLEMENTARITY
GATE exp(-rho/rho_iso) (loss-grid CIC density, frozen per window, detached), so the
constant pull reaches ONLY mass the density losses are blind to — bulk outside mass
(rho~40+/cell) is silenced, lone fringe (1-3/cell) keeps ~full pull. Anchor: Floaters
No More (EGSR 23) reconditions gradients by a region property instead of changing the
objective; complementarity with d_vol is the same move in density space. rho_iso=4
(particles/cell at the porosity-forensics isolation scale). Falsifier unchanged
(stray < 0.2% at hero, chamfer within 2%, no ear erosion); if the GATED term still
fails it, the mechanism is wrong and plan-B is next.

### §7.4 Gate autopsy → kNN-scale gate (v8)

Gated v7 batch: stability fully restored (no inversions, first3 12%) but endpoint
unchanged — hero5 out-of-support (fine-DT>2cells, the honest target-based metric)
2.14% vs hero1 2.18%, ear region 0.400% both. Yet d_dt fell 97% — resolved by autopsy:
the DESCENT WAS THE GATE DYING, not the fringe moving. The loss-grid CIC density gate
read median (and p90) 0.0000 on 100% of the out-of-support particles: fringe between
thin features shares coarse cells with the features, so it never looks isolated at
grid scale. Silenced exactly where needed. Measured separation at kNN scale:
out-of-support ratio median 1.69 (p10 1.12) vs bulk 0.99; ratio>1.5 catches 64% with
8.3% bulk false positives — which cost NOTHING because in-support particles sit at
DT=0. Gate v2 = ramp(d_kNN/median, 1.2→1.8), frozen per window; early dense outside
bulk still silenced (dose-response protection intact). ALSO recorded: stray_frac/
stray_max are SELF-REFERENTIAL (body-kNN isolation, no target reference) — they
conflate interior porosity with ejecta and cannot credit out-of-support cleanup; the
fine-DT out-of-support fraction is the honest endpoint metric for this tranche and
should join metrics.py (queued). Verdict rule unchanged, now measured on out-of-support
frac + ear visual.
