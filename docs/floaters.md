# Problem dossier 1 — Floating Gaussians (부유 가우시안)

Status: **SOLVED (2026-09-02)** — closed by visual audit on the converged flagship
(h19): ear-region closeups show a fully closed surface with no visible scatter;
fork lo-band 326→78 (0.2% of N), ear out-of-support 0-2 particles, deep strays
0.013%. The ear-interior cov<0.3 criterion (<15%) was RETIRED as metric-driven
over-spec after five mechanism falsifications (fill v1-v4, growth v1) shared one
root — any channel forcing mass into under-covered regions harms transport or
manufactures fringe — and the closeups showed the 19.9% statistic does not manifest
at the renderer's operating point (density inhomogeneity below visual resolution).
The audit trail stays: if a future viewer/loss resolves it (real-3DGS loss tranche),
the criterion returns with that evidence.

## 1. Phenomenon

Sparse Gaussians hovering off the body surface, most visibly between/below the ears
(the "fork", y∈[1.4, 2.3]) and along the lower rim. They dominate the perceptual
quality of any 3DGS render long before they register in aggregate metrics.

## 2. What they turned out to be (session forensics, 2026-09-01/02)

Four DISTINCT populations, each with a different mechanism — the single word
"floaters" hid four problems:

| population | forensic signature | root cause | fix |
|---|---|---|---|
| far ejecta (>4 fine cells) | outside every view's gradient | fast transport + λ runaway | box leash + λ cap + pacing (pre-session) |
| squeeze-ejecta | J = 0.52–0.84 on the gap particles | **volumetric plastic ratchet**: assimilation baked compression permanently; \|J−1\|>0.3 grew 0→41% over 120c | isochoric assimilation (det Fp = 1) + sKL volume prior |
| lone stragglers | kNN-isolated, DT>2 cells | α-saturation: soft-silhouette AND log-mass D_vol gradients vanish with sparsity | DT-W1 cleanup (sum form, kNN gate) |
| **fork halo** (the user-visible one) | 0.05–0.10 wu off support, semi-dense clumps, 90% kNN-gate-closed, **zero force in the fine-DT dilation dead band** | no loss term had ANY gradient at that distance | grid-free near-band W1 (nearest-target-particle assignment, berth 1.5×spacing) |

Key measurement journey: metric blindness was half the problem. `stray_frac` is
self-referential (body kNN, no target term — conflates porosity with ejecta);
`out_dt_frac` has a ~3.5-cell dilated dead radius (a 1%-of-body halo at 0.03–0.10 wu
reads 0.000%). The honest metric is `tgt_nn_metrics`: distance to the nearest target
particle in units of the target's own median NN spacing, fraction + tail.

## 3. Mechanism history (what was tried, what falsified it)

1. **w_creg** (kNN-Laplacian on dFc): real in isolation (−21% strays), inert at hero
   scale — fringe is *residue*, not *creation*. Kept as bulk regularity.
2. **2D multi-view DT** (DRWR/v1-style): falsified — the visual hull hides interior
   concavities; 1.8% of ear strays visible at ANY mask threshold. → 3D.
3. **Ungated 3D W1 sum**: dose-response catastrophe (stray 0.8/4.0/7.8% at
   w 0.05/0.2/1.0) — constant pull on the early coherent outside mass double-drives
   transport, and violent transport CREATES fringe.
4. **Grid-density gate**: silenced 100% of its own targets (fringe shares coarse
   cells with the features it hugs).
5. **kNN gate**: the honest-metric winner, but blind to 3–10-particle clumps
   (LOF-class scores are at chance on clustered outliers — DROD) and its inversions
   needed the trajectory-det guard.
6. **Transport-budget gate**: measured no-op (mid-run partial rub on every out
   particle: no cleanup + rim damage). Lesson: selectivity in WHO beats scheduling
   of HOW MUCH.
7. **Near-band W1** (grid-free): the fork-halo owner. Fork lo-band 326→97 (−70%),
   plus an unexpected bonus — it REMOVES the late-run phys/render gradient conflict
   (g_cos min −0.86 → −0.00), because boundary-mass placement competition was the
   conflict's source.
8. **Fill v1/v2** (ear interior): v1's mask was 95–100% OUTSIDE support (an outward
   fringe *factory*, Opus F1); v2 fixed the mask but its constant weight dominated
   late 30:1 and froze the run early. **v3 (implemented 2026-09-02): fill weight from
   the norm balancer** (fill_λ = α·‖∇phys‖/‖∇fill‖, EMA, cap) — dominance
   structurally impossible. Open falsifier: ear cov<0.3 must drop <15% at α ≤ 0.1.

## 4. Related papers (organized by sub-problem)

**Sparsity-blind losses / floater suppression**
- DRWR (Han et al., ICML 2020) — states the failure verbatim ("points far outside
  the silhouette receive zero gradients"); flat-inside/linear-outside per-point DT.
- 3DGS-as-MCMC (Kheradmand et al., NeurIPS 2024) — L1 opacity: a CONSTANT gradient
  works where the photometric gradient has saturated away; loss-neutral relocation
  as the escape hatch we keep documented but unused (physics-honesty).
- Mip-NeRF 360 (Barron et al., CVPR 2022) — distortion loss: an isolated lobe's
  gradient set by distance to the main mass, not its own magnitude.
- AbsGS (Ye et al., ACM MM 2024) — gradient collision: signed residuals cancel at
  fine structures; one-signed accumulation is the accepted repair (our one-signed
  W1/deficit terms).
- Floaters No More (Philip & Deschaintre, EGSR 2023) — recondition gradients by a
  region property rather than changing the objective (our gate lineage).

**Transport / assignment view**
- Sinkhorn divergences (Feydy et al., AISTATS 2019) — density-blurred losses screen
  extreme support points; OT-type costs give isolated points full displacement
  gradients (the W1 sum form's backbone).
- Unbalanced OT (Séjourné et al., 2023) — the transport-range theorem behind every
  locality gate (beyond the range, a deficit is mathematically destroyed mass).
- Partial OT for point clouds (Bai et al., 2025) — warning: symmetric partial OT
  *discards* outliers rather than filling deficits; only source-soft/target-hard
  relaxation fills.
- Chamfer structural failure (arXiv:2603.09925) — local regularizers provably cannot
  rescue neighbour-less particles; non-local coupling (our MPM grid adjoint) is
  required. Also why the near-band term uses a BERTH + band (not raw Chamfer).
- Fattal & Lischinski (SIGGRAPH 2004) — the gathering term: locality-gated mass
  attraction into deficits; pull must be a potential that VANISHES at coverage (the
  fill v2→v3 lesson).
- Tampubolon et al. (SIGGRAPH 2017) §4.3.4 — the log-volume ledger (queued: the
  principled home for control-injected volume drift).
- DROD (arXiv:2603.12847) — clusterlier detection: why fixed-k kNN gating is blind
  to small clumps by construction.

## 5. v1 (PhysMorph-GS C++) vs v2 — the same enemy, different weapons

v1 fought floaters with **global, symptom-side controls**: outlier removal/clamps,
kNN displacement-field smoothing (k=64, 3 iters), and its L_DT silhouette term
(which we initially mis-ported to 2D before the hull forensics). v2's position:
identify each population's creation/retention mechanism and give it a dedicated,
falsifiable term — leash (far field), isochoric+sKL (squeeze source), gated DT-W1
(lone stragglers), near-band W1 (dead-band halo), norm-balanced fill (interior
deficit) — with the honest target-referenced metric so cleanup cannot be
metric-flattered.

## 6. Current state & open work

Converged flagship (`render_full_dt_iso_nn`): fork lo-band 97 (from 326), ear
out-of-support 0–2 particles, deep strays 0.013%. OPEN: ear interior coverage
(cov<0.3 = 26.5% — the near-band term drags halo to the surface but does not push
mass INTO thin features) — fill v3 verification batch is the closing move; if its
falsifier fires, the growth-tensor channel (render-commanded det(G), morphoelastic)
is pre-registered next.

## 7. Code changes made for this problem (file → change → intent)

| where | what changed | why (intent) |
|---|---|---|
| `losses/volumetric.py: target_dt_grid` | fine TARGET-FITTED EDT grid (dt_res=160, 1.5×extent cube), support = the sampler's own CIC stencil | give the spray term a force field with no dead radius over the fringe band (the loss grid's ~1-unit cells had one), and make target self-force vanish by construction |
| `losses/volumetric.py: d_w1` | SUM form `Σ m·gate·DT`, trilinear sample | per-particle pull = w_dt exactly — N-invariant, sparsity-invariant; the mean form was a measured 300–3000× no-op |
| `losses/volumetric.py: isolation_gate` | kNN-ratio ramp (1.2→1.8×median), frozen per window | full pull on true singletons all run long while dense rim mass is untouched — selectivity in WHO gets pulled (the winning gate) |
| `losses/volumetric.py: w1_budget` | scalar transport budget (KEPT, falsified) | A/B honesty: the budget conflated early-safety with mid-run power; retained so the falsification is reproducible |
| `losses/volumetric.py: deficit_field` | one-signed blurred-ratio deficit mask **AND unblurred target occupancy**, returns deficit mass | fill must never fire outside true support (v1's mask was 95–100% outside = an outward fringe factory) |
| `losses/volumetric.py: nn_band_assign / d_nn_band` | frozen nearest-target-particle assignment; pull only in the berth..far band (1.5–4.5× target NN spacing) | the fork halo lives INSIDE the DT grid's dilation dead band — this term uses the target's raw geometry, so no grid, no dilation, no dead band; the berth keeps legitimate rim mass untouched |
| `losses/volumetric.py: coverage_shortfall` | mask-free continuous shortfall statistic | fill progress must be comparable across windows (per-window binary EDT energies are not — their masks change) |
| `pipeline/optimizer.py` | `phys_core`/`dt_term` split; cleanup terms OUTSIDE the λ channel; fill v3 weight from a dedicated norm balancer at it==0 | cleanup gradients must not inflate the λ numerator or PCGrad's reference (Codex f9); λ→cap × constant gradient is the documented mass-ejection mode; the balancer bounds fill ≤ α×physics gradient so late-run dominance (v2's failure) cannot form |
| `pipeline/runner.py` | fine-grid/pts/spacing in build_target; archived-state `d_dt`/`d_fill` telemetry; per-term freeze tracks; lg×W1 incompatibility guard | the freeze must SEE cleanup progress (commits that only retrieved fringe looked stale); the quadratic local pass would undo accepted W1 steps and assimilation would ratchet the regression |
| `metrics.py: tgt_nn_metrics` | target-NN distance in units of the target's own median spacing, fraction + tail | the honest floater metric: `stray_frac` is self-referential and `out_dt_frac` has a ~3.5-cell dead radius — cleanup was being metric-flattered ("ear count 1" hid the 0.07–0.15 wu halo) |
| `pipeline/config.py` | every knob above, each with its measured justification in the comment | knobs must carry their evidence — a falsified mechanism stays labeled falsified |
| `tests/test_w1_cleanup.py` | self-force, fixed-N sparsity invariance, berth/band eligibility, no-gap, mask-leak regression | pin exactly the claims the adversarial rounds attacked |

