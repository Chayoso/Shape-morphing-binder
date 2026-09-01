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
| trajectory snap (D_vol mostly spent in 3 commits) | **pacing**: window loss cut ≤ 12%/window + dFc clip; move_cv + first3 metrics | (deliverable-driven) | move spread across commits, endpoint quality intact | paced arm loses fidelity or trips freeze early |
| particle/volume oscillation at optimum | λ cap + λ-free freeze (existing); **guarded Anderson on CONTROLS** in backlog if needed | [AA for physics (TOG18)](https://arxiv.org/abs/1805.05715) + guarded variants | cap alone suffices | else implement guarded AA (line search = the guard) |
| Gaussian-native render loss | backlog: diff_gauss alpha/depth in the loss (server build exists) | GIC, 3DGS-LM | — | — |
| adaptive/high-res grid | backlog after c2f verdict | SPGrid/adaptive MPM lineage | — | — |

## 6. Decision rule (pre-committed)

- `render_ws`/`render_gs` **adopt** if: G2–G4 clean AND (sil_iou ≥ `render` − 0.002 with
  fewer rejections, or sil_iou > `render`). Ear-region visual regression vetoes (R1).
- `vbd` **promote to co-headline** if: Gc convergence every commit, G2 clean, chamfer
  within 10% of `render` at ≤ wall-clock parity, visuals hole/flicker-free. Otherwise it
  remains a solver-research arm and the dynamic family stays the deliverable.
