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

## 4. Decision rule (pre-committed)

- `render_ws`/`render_gs` **adopt** if: G2–G4 clean AND (sil_iou ≥ `render` − 0.002 with
  fewer rejections, or sil_iou > `render`). Ear-region visual regression vetoes (R1).
- `vbd` **promote to co-headline** if: Gc convergence every commit, G2 clean, chamfer
  within 10% of `render` at ≤ wall-clock parity, visuals hole/flicker-free. Otherwise it
  remains a solver-research arm and the dynamic family stays the deliverable.
