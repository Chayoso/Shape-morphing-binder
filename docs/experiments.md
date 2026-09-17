# Experiments — gates, metrics, result log

## Gates (what "done" means)

| gate | test | threshold |
|---|---|---|
| G1a | constant dFc sequence ≡ shared control rollout | max&#124;Δx&#124; ≤ 1e-6·scale |
| G1b | dL/ds (material leaves + v_T adjoint) vs central FD | rel err < 0.25, finite & nonzero |
| G2 | guard counters over the full run (incl. any-step F inversion) | **all zero** |
| G3 | tail jitter over SIMULATED frames (held padding excluded) AND terminal drift v̄·dt·T/diag (dynamic family; ≡0 for VBD) | both < 0.3% bbox diag |
| G4 | hole_frac (binary 3×3 splat, FIXED target extent) | ≤ 2% AND ≤ physics arm |
| G5 | each render-driven arm vs its physics baseline, same seed/budget | sil_iou ↑, chamfer ≤ +2%, hole ↓ |
| G6 | per-frame visual QA (quicklook strips + gif over the FULL range) | closed solid, no ghost/floaters/flicker |
| Gc (VBD) | solver ‖∇E‖ ≤ tol·‖∇E₀‖ every commit | convergence = gradient-validity gate |

Metric independence: metrics share **no operator with any loss** (binary point splat, one
fixed target-derived extent, held-aware jitter, `outside_frac` ejecta telemetry); raw
simulation state only — the renderer is never consumed.

## Verification protocol

1. `pytest` (42 CPU/warp-CPU tests, incl. end-to-end smokes of both families).
2. Adversarial gate: Codex (gpt-5.6-sol, xhigh) + Claude Opus, REFUTE mode, findings cite
   file:line, implementer answers every one. Round 1 (2026-09-01): 26 findings, all fixed;
   both reviewers cleared the autograd bridge, line-search restore, projection math.
3. hyde06 runs via `scripts/pipeline_run.py`; every number states its discretisation.

## Run commands (hyde06)

```bash
cd ~/physmorph_v2
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
CUDA_VISIBLE_DEVICES=<free> $PY scripts/pipeline_run.py \
    --arms phys,render,render_ws,render_gs,vbd --w_kin 5.0 --out output/v3_ab
$PY scripts/quicklook.py --npz output/v3_ab_render_gs.npz --frames 0,120,240,360,480,600 --out …
$PY scripts/make_gif.py --npz output/v3_ab_vbd.npz --out …          # VBD: 1 frame/commit
CUDA_VISIBLE_DEVICES=<free> $PY scripts/grad_analysis.py --out output/grad_analysis
```

## Why this implementation outruns the C++ oracle (DiffMPMLib3D)

*(corrected after adversarial review — the first version of this section wrongly called
the C++ serial: it is OpenMP-parallel throughout, 62 `#pragma omp parallel for`, and
`bind.cpp` releases the GIL so OpenMP can use all cores.)*

Timed artifact on OUR side: `output/local_ab2.json` (n=5000, T=10, 12 commits ≈ 0.25
min/arm on an RTX 4090 Laptop) and the hyde06 logs for full scale (N=20000/T=20/30
commits ≈ 0.5 min/arm, RTX 6000 Ada, `output/v4_ab.log`). **No committed like-for-like
C++ timing exists** (legacy logs are empty and legacy configs ran 90 animations) — the
"hours" folklore is remembered, not sourced; treat cross-implementation speedups as
unquantified until someone re-runs the C++.

Factors, honestly ranked:

1. **GPU parallelism (dominant).** Every kernel (P2G/G2P/stress) is data-parallel over
   20000 particles / 262k nodes on an SM array vs OpenMP threads on CPU cores; the losses
   (CIC splats, silhouettes, shading) are torch GPU ops in the same device memory.
2. **Adjoint schedule.** The C++ backward does produce all-layer gradients per pass; the
   difference is the OPTIMISATION schedule: CompGraph descends Gauss-Seidel over layers
   (each `ComputeForwardPass(c)` re-runs the suffix c→T−1, an O(T²) shape), where our
   single-tape window is Jacobi over layers, O(T) per iteration. At the repo's own C++
   configs (`max_gd_iters: 1`) this is only ~1.2×/2.4× in step-kernels — a real but
   secondary factor, and a different algorithm rather than a pure speedup.
3. **No binding-boundary crossings in the hot loop** — mainly relevant to the legacy
   python-extension workflows that marshalled numpy per call; the pure-C++ path largely
   avoided it.
4. **JIT-cached SoA kernels** (warm start ~10 ms per the hyde06 logs) and the no-tape
   `eval_terms` path, which keeps adjoint memory/bookkeeping off the line-search probes.

## Result log (discretisation with every number — AGENTS rule 4)

### 2026-09-01 — v2 dynamic family, FULL SCALE, hyde06 RTX 6000 Ada (all gates PASS)

`dx=0.5, dt=1/240, 64³, smoothing 0.955, loss_res 32; N=20000 isosphere→bunny, T=20,
iters=8, 30 commits, w_kin=5, w_box=10, assim=0.5(elastic), α_λ=0.5, 18 views @64px`;
0.5 min/arm.

| arm | chamfer | sil_iou | hole | jitter_rel | guards | commits |
|---|---|---|---|---|---|---|
| phys | 0.1786 | 0.8885 | 0.05% | 0.00003 | all 0 | 30/30 |
| render | **0.1439** | **0.9536** | **0.00%** | 0.00003 | all 0 | 30/30 |

λ_R self-anneals 1130→61; kin 8.97→0.002 monotone. G6: closed solid every sampled frame,
ears+paw form (tips rounder than target). Smoke-scale tuning history and ablations
(w_kin=0.5 momentum snowball; displacement-assimilation instability): git history
`2607972` and method.md §8.

### 2026-09-01 — gradient measurements (hyde06 GPU1, N=5000 T=10 probes; grad_analysis.json)

- v1 fixed-λ render contribution through the MPM adjoint: **0.04–0.3%** of the update
  (raw norm gap 150–1200×) — the "render did nothing" era, quantified. v2 norm balancing
  pins it at α_λ.
- cos(∇D_render, ∇D_vol) on dFc: **+0.68 → −0.74** across commits — render carries rim
  information the coarse mass grid opposes late; usable only under norm balancing.
- Image-space render pull is **8–16× surface-concentrated** (D_vol: 1.1–3.4×) — the
  surface-only-feedback premise, measured.
- v1 forensics (archived npz, v2 metrics): greedy tail jitter 0.00485 (fails G3); F aniso
  max 2.7 / detF 0.32 *with* the silent clamp. v2 full scale, no clamp: 1.70 / 0.51.

### 2026-09-01 — grid-GS differentiability pre-check (CPU toy, float64)

Colored block-GS solve differentiated three ways vs FD: at ‖∇E‖=3.6e-5 — unrolled 0.1%,
IFT adjoint 0.04% error; at 10 sweeps — 17%/54%. Conclusion: both routes valid at
convergence; solve tolerance is a correctness gate. `scripts/probe_gs_differentiability.py`.

### 2026-09-01 — v3 round 1, full scale hyde06 (same discretisation as above)

| arm | chamfer | sil_iou | hole | G2 | note |
|---|---|---|---|---|---|
| phys | 0.1785 | 0.8869 | 0.03% | PASS | baseline reproduced |
| render | **0.1371** | **0.9605** | 0.00% | PASS | v2 headline holds |
| render_ws | 0.3610 | 0.7758 | 0.24% | **FAIL** (9 inv) | R4: absolute-control double-application |
| render_gs | 0.3628 | 0.7363 | 0.07% | **FAIL** (6 inv) | confounded (warm_start coupled) |
| vbd / vbd_phys | 0.5220 | 0.7112 | 0.00% | PASS* | *frozen at undeformed: R12 fringe-node poisoning + R13 stiffness scale |

Full post-mortem + fixes: `rationale.md` §4. Notable v2-era latent defect found by the
failure: line-search acceptance never checked det(F)>0 (inversions are invisible to the
data terms) — now part of `_state_ok` for every dynamic arm.

### 2026-09-01 — v3 round 2/3 + coherence verdict; VBD arm retired to deprecated/

Round 2 (fixes): `render_ws` healed (0.1404/0.9592, all gates PASS — but ties plain
`render`, so warm start stays optional); `render_gs` no longer dies but REGRESSES
(0.2379, detFmin 0.0011 → R1/R3 falsified, not adopted); `vbd` moves but crawls.
Round 3 (8-color decoupling): no change — the limiter was never the coloring.
Deep diagnosis (`vbd_diagnose`): per-term exit gradients 67/51/26 nearly cancelling =
the EQUILIBRIUM is the limiter (solver fine); η=1.0/E=300 unlocks speed (D_vol 225→61
in 19 commits) at the cost of material memory.

**Coherence table (the "is it Chamfer-like?" question, measured):**

| arm | nbr_overlap(16) | disp_rough | move | chamfer / sil_iou | time |
|---|---|---|---|---|---|
| dynamic render | 0.387 (p10 0.19) | 0.070 | 0.614 | **0.137 / 0.954** | **0.5 min** |
| vbd η=0.5 E=2e3 | 0.962 | 0.044 | 0.106 | 0.44 / 0.77 | 8.8 min |
| vbd η=1.0 E=300 | **0.798** | **0.041** | 0.429 | 0.316 / 0.859 | 8.2 min |

Reading: NO arm is Chamfer-like (a NN flow would show rough ≫0.3, overlap <0.1). At
comparable displacement the quasi-static arm preserves ~2× more neighbourhoods than the
dynamic arm (whose plastic shear physically mixes 60% of 16-NN sets) — but loses on every
fidelity metric and on wall-clock, and its render coupling is a load, not a physics
modification. Decision (pre-committed rule): **dynamic family = the deliverable; VBD
retired to deprecated/** with this analysis as its legacy. The dynamic arm's improvement
axes exposed here: neighbour mixing + inversion margin (detFmin 0.46–0.51; one transient
mid-window inversion observed under atomic nondeterminism) → assimilation-rate/stiffness
sweep pending.

### 2026-09-01 — stray forensics + the loss-grid resolution verdict

Forensics on the v4 npz (kNN strays located per frame): the "ejection" is **interior
porosity, not flight** — strays sit at radius 1.2–1.3 vs body r95 2.6–2.8 (0% beyond),
identities persist 0.6→0.88, created during the fast-transport commits (~f80) and then
RATCHETED PERMANENT by plastic assimilation; invisible to every loss term (D_vol cell 1.0
holds ~160 particles; the silhouette cannot see the interior). Pacing prevents creation
(render_full: none until f400).

loss_res 64 A/B (cell 0.5 = dx): **v1's "coarse grid for stability" fear is REAL** —
unpaced phys EXPLODES (detF −939, dead at commit ~2) and unpaced render snaps (first3
100%, strays 1.96%): the fine grid sees the pockets but also amplifies the violence that
tears them. **Paced render_full + loss_res 64 is a synergy: chamfer 0.1098 (best ever,
−21% vs prior best), silIoU 0.9515, holes 0.00%, detF 0.55, first3 22%, G2/G3 PASS**
(stray 0.265% marginally over the 0.2% gate — thin-feature sparsity vs real porosity to be
adjudicated). Flagship config = render_full @ loss_res 64; D_vol coarse-to-fine remains
the fallback if any unpaced arm must run fine grids.

### 2026-09-01 — v4 tranche-1 batch (8 arms, full scale, same discretisation)

| arm | chamfer | sil_iou | detF_min | first3 | move_cv | stray_max | gates |
|---|---|---|---|---|---|---|---|
| phys | 0.1784 | 0.8913 | 0.59 | 56% | 1.66 | 0.29% | ejection ✗ |
| render | 0.1384 | 0.9624 | 0.41 | 52% | 1.56 | 0.46% | ejection ✗ |
| render_mat | 0.1419 | 0.9569 | 0.44 | 54% | 1.63 | 0.42% | ejection ✗ |
| render_pbr | 0.1838 | 0.9256 | −0.22 | 90% | 0.55 | 0.26% | **G2 ✗ (died early)** |
| render_pc | 0.1472 | 0.9478 | 0.51 | 57% | 1.73 | 0.45% | ejection ✗ |
| render_c2f | 0.1434 | 0.9541 | 0.50 | 58% | 1.75 | 0.44% | ejection ✗ |
| render_pace | 0.2227 | 0.8335 | 0.74 | **15%** | **0.43** | **0.13%** | G3 drift ✗, G5 ✗ |
| **render_full** | 0.1655 | 0.9364 | **0.75** | **20%** | 0.58 | **0.20%** | **ALL PASS** |

Verdicts (pre-registered rules, rationale §5): `render_pbr` falsifier FIRED (standalone
shading destabilises — inversion, early stop; inside the paced bundle it survives) — not
adopted standalone, needs separate sil/pbr balancing (structural, queued). `render_pc`
clean but strictly worse than `render` — the projection discards useful conflict
information (the user's prior); superseded by the local-global arm pending review.
`render_c2f` ≈ tie at this budget. `render_pace` trajectory metrics exactly as designed
(first3 52→15%) but endpoint short at 30 commits — pacing pays commits for smoothness by
construction; rerun at 60. **`render_full` is the only arm passing every gate** including
the NEW whole-trajectory ejection check, with the best inversion margin (detF 0.75) and an
even trajectory; fidelity gap vs `render` (0.166 vs 0.138) is the 30-commit budget under
pacing. **Headline discovery: every unpaced arm produces 0.26–0.46% transient mid-run
strays that endpoint metrics never saw** — pacing reduces them structurally (gentler
transport, lower |v|max). Flagship candidate: render_full @ 60 commits, pc→lg swap
pending the local-global review.

### 2026-09-01 — hero1 + lg isolation + w_creg (fringe tranche, hyde06)

**hero1** `render_full` N=40k, 60 commits, loss_res 64: chamfer **0.0943**, silIoU 0.948,
first3 12% (paced trajectory), jitter 6e-5, stray_max 0.367% — the residual is the
thin-feature fringe between the ears/paws, the flagship's dominant visual defect.

**lg isolation** (`render` vs `render_lg`, N=20k, lr32): chamfer 0.1405 vs 0.1421,
silIoU tie, stray 0.435 vs 0.450 — a tie at 3.3x wall-clock, with λ_loc pinned at the
cap (5000) the whole run. Verdict: on this benchmark the global window already exhausts
the silhouette signal; the band has nothing left to descend, and the fringe is invisible
to it for the same reason it is invisible to the global term (α-saturation, below).
**lg parked** — not adopted into the flagship; re-evaluate on hard pairs where the rim
residual should be under-resolved by the global step.

**w_creg** (kNN-Laplacian penalty on dFc, frozen window-start topology, w=100 k=8):
- isolation (`render` vs `render_creg`, lr32): stray 0.495→**0.390% (−21%)** with
  chamfer 0.1403→0.1381, silIoU 0.958→0.960, jitter 1.8e-4→1.1e-4, detFmin 0.40→0.46,
  and G3_rest recovered (drift 0.0041→0.0024). Every metric co-improves — the
  lone-particle-actuation mechanism is real. **Adopted** (it is nearly free).
- hero2 (`render_full_creg`, N=40k/60c/lr64): stray 0.370% vs hero1 0.367%, chamfer
  0.0940 vs 0.0943 — **tie**. At flagship scale the fringe is NOT created by lone
  actuation; it *survives* because the spray side of D_render saturates:
  α = 1−exp(−k·w) ≈ k·w for sparse mass, so relu(α−α_t)² gives a gradient that
  vanishes quadratically with sparsity. A lone stray is asymptotically invisible.
  Next mechanism (pre-registered in rationale.md): unsaturated (mass-linear) spray
  term outside target support — hole side stays saturated (presence detection),
  spray side becomes linear so per-unit-mass pull is sparsity-independent.
  Literature check in progress before implementation (standing rule).

### 2026-09-01 — fringe tranche VERDICT: gated W1 cleanup ADOPTED (v8)

Mechanism that survived three falsification rounds (2D DT: visual-hull-blind, measured
1.8% gradient visibility; ungated 3D sum: dose-response catastrophe; grid-density gate:
silenced 100% of its own targets): **SUM_p m_p·gate_p·DT₃D(x_p)** on a fine
target-fitted grid (dt_res 160), gate = kNN-isolation ramp (1.2→1.8 × median), fixed
weight w_dt=0.2 outside the λ channel. Verdict on the HONEST metric (out_dt_frac:
target-referenced, now in metrics.py — stray_frac is self-referential and read a tie
throughout):

| bench (render_full → render_full_dt v8) | chamfer | sil_iou | out>2cell | ear out>2cell |
|---|---|---|---|---|
| sphere→bunny hero (N=40k, 60c, lr64) | 0.0943 → **0.0917** | 0.9483 → **0.9591** | 2.18 → **1.71%** | 0.400 → **0.087%** (−78%) |
| sphere→armadillo | 0.0930 → **0.0921** | 0.9081 → **0.9196** | — | — |
| spot→bunny | 0.0953 → **0.0911** | 0.9546 → **0.9597** | — | — |
| A→C letters | 0.2654 → **0.2561** | 0.9455 → **0.9515** | — | hole 5.12→5.51% (watch) |

Deep strays (>4 cells) 0.258→0.105% (−59%). Visual (ear_v8.png): between-ears scatter
21 → 4 particles — the tranche's target defect is visually gone. Every bench improved
chamfer AND sil_iou; trajectory pacing intact (first3 12%). Remaining known residues:
one compact interior porosity clump (different mechanism — real-3DGS-loss tranche),
A→C hole coverage (W1 is spray-side by design; holes need the w_hole channel), and the
G4_ejection gate still keyed to self-referential stray_max (metric replacement
pre-registered, not silently swapped).

### 2026-09-02 — v11 verdict: hero10 flagship (iso + kNN-W1 + traj-guard); photoreal V1

**hero10 `render_full_dt_iso` (N=40k, 120c, lr64): chamfer 0.0851, silIoU 0.9647 (both
all-time best), G2/G3/G4holes PASS, EAR out-of-support 1 particle / 40000** (hero1: 21),
deep strays (>4 fine cells) 0.013%. The trajectory-det acceptance guard eliminated the
kNN-W1 inversion AND improved quality (rejecting inverting candidates steers to better
minima). Pairs: armadillo 0.0885/0.9129, A→C 0.2446/0.9530 — A→C's hole 5.85% vs the
TARGET's own 5.76% at this metric = at ceiling; G4_holes' absolute 2% is unattainable
for this pair (gate fix pre-registered: compare vs hole_frac_tgt). Fill term v1 did NOT
improve ear coverage (23.1→23.5%): its budget counts porosity deficits everywhere, the
same disease that neutered budget-W1 — redo pre-registered (surface-deficit-only mask).
G4_ejection still keyed to self-referential stray_max (replacement pre-registered).

**Photoreal 3DGS V1** (`scripts/render_photoreal.py`, diff_gauss): F→cov3Ds_precomp,
kNN-PCA normals (density-field normals were sampling-noise mottled; PCA + field-sign
orientation), interior particles blended to ambient albedo by surface weight (random
interior normals showed as dark dapples), camera-relative 3-light studio rig, COLMAP
y-down convention. output/photoreal_hero10.png. Art direction (face-on framing; the
morph's swept-back ears) queued.

### 2026-09-02 — v12: fill v2 verdict (mechanism works, dominance fails); disk incident

hyde06 root hit 100% (output/ = 77G of trajectory npz) mid-batch — hero11/v12_arma/
v12_AC completed their sims (gate lines logged) but lost npz/json saves; superseded
npz purged (64G freed); h12_j0 re-runs the fixed flagship as the reference.

**hero11f `render_full_fill_iso` (fill v2, w_fill=0.2, 120c budget): d_fill (coverage
shortfall) 0.694 → 0.195 (−72%) — the support-ANDed deficit mechanism genuinely
fills.** But: chamfer 0.1102 (flagship era 0.0845 — catastrophic), converged at 51
commits, G3 drift FAIL (0.0038), early move 0.022→0.079 (unpaced-scale transport).
Root cause: LATE-STAGE DOMINANCE — once deficit mass < budget·N the scalar hits 1 and
the constant 0.2 pull outweighs the data gradients ~30:1 (the hole-side twin of the
§7.3 dose-response lesson). Fattal's construction says the pull must be a POTENTIAL
that vanishes at coverage; our mask-exit is bang-bang (hysteresis was the documented
residual). Fill v3 pre-registered: continuous demand weighting (pull ∝ local
shortfall fraction, smoothly → 0 at coverage) — not a weight retune. Flagship remains
fill-free. v12_AC (fill v2): hole 5.94% vs target-ceiling 5.76% — fill does not move
the AC hole (it is at the metric ceiling; gate fix to target-relative comparison
still queued).

### 2026-09-02 — hero12 w_jvol ladder: sKL volume prior ADOPTED (w_jvol=50); new flagship

`render_full_dt_iso` + w_jvol ∈ {0,10,50}, N=40k, 120c, lr64 (arms identical otherwise;
j0 = the post-review-stack baseline):

| w_jvol | chamfer | silIoU | detFmin | |J-1|>0.3 | J p1/p99 | drift | out_nn p95 |
|---|---|---|---|---|---|---|---|
| 0 | 0.0958 | 0.9450 | 0.0005 | 13.7% | 0.33/1.48 | 0.0020 | 0.127 |
| 10 | 0.0781 | 0.9693 | 0.108 | 1.0% | 0.74/1.24 | 0.0010 | 0.102 |
| **50** | **0.0778** | **0.9696** | **0.497** | **0.0%** | **0.92/1.09** | 0.0010 | 0.105 |

Monotone dose-response in the RIGHT direction on every axis, no trade-off anywhere:
the volume-spring pathology is eliminated (essentially incompressible morph), the tail
creep ("흔들림") halves, and chamfer/silIoU set all-time records — beating hero10
(0.0851/0.9647) by 8.6%/0.5pt. Armadillo generalization (w_jvol=10): 0.0873/0.9239,
both best-ever for the pair. Reading: the sKL prior PREVENTS in the energy what the
trajectory-det guard could only REJECT (the F5 literature's exact prediction —
Smith18/ThinShellLab lineage); with the spring defused, the strengthened guard stack
stops costing quality (j0's regression vs hero10 was that cost, now moot).
**Flagship = render_full_dt_iso @ w_jvol=50 (h12_j50). --w_jvol CLI default set to 50.**

### 2026-09-02 — PCGrad removed from the flagship (h13 ablation)

`render_full_dt_iso` ± grad_project, everything else identical (w_dt 0.2, w_jvol 50,
iso, 120c): bunny 0.0778/0.9696 → 0.0777/**0.9706** (tie+), armadillo 0.0873/0.9239 →
**0.0863/0.9315** with detFmin 0.108→0.403. Pre-registered rule (removal >= tie ->
remove) fires. History: standalone render_pc was falsified in v4 (the user's original
skepticism); the in-bundle contribution was never isolated until now — it measures
<= 0 under the current stack (W1 outside lambda, sKL volume prior). Reading: the
late-run phys/render conflict PCGrad existed for (cos -0.74, v4-era) appears resolved
by the channel separation; the g_cos/g_share telemetry (now in every commit rec) will
verify directly. The parked local-global pass stays parked — its re-activation
condition (a conflict-handling gap after PCGrad removal) did not materialise; results
IMPROVED without either mechanism.

### 2026-09-02 — h15 (200c, live-streamed): near-band W1 adopted; oscillation closed

4-GPU batch with the new --live_port streaming (quad dashboard). All arms w_dt 0.2,
w_jvol 50, iso; nn arms add w_nn 0.2:

| arm | chamfer | silIoU | out_nn | fork lo-band | note |
|---|---|---|---|---|---|
| h15_nn (bunny) | **0.0693** | 0.9697 | **10.1%** | **97** | flagship |
| h15_base (bunny, no nn) | 0.0735 | **0.9758** | 19.5% | 326 | g_cos min **-0.86** |
| h15_spot (nn) | **0.0682** | **0.9725** | 11.8% | — | pair best-ever |
| h15_AC (nn) | **0.1300** | **0.9677** | 0.8% | — | chamfer HALVED (0.2446→) |

**Near-band W1 ADOPTED** (flagship = render_full_dt_iso_nn): the user-visible fork
floaters drop 326→97 (-70%) with chamfer -5.7%; the -0.6pt silIoU is soft-edge
rearrangement, accepted. Bonus (g_cos telemetry): the base arm still hits late-run
phys/render conflict (cos min -0.86); WITH the near-band term the conflict vanishes
(min -0.00) — boundary-mass competition was the conflict's source; PCGrad removal and
lg's permanent parking are both re-confirmed. g_share median 0.34 across all benches:
the render channel steadily drives a third of the update (the paper's core claim,
continuously measured). **Oscillation CLOSED as a pathology**: at 200c every track
still improves 9-27% per 40 commits (freeze correctly withheld); tail move halved to
0.003-0.004 by w_jvol. Remaining micro-motion = ongoing descent; a run-to-actual-
convergence probe (400c) is queued to find the true resting point.

### 2026-09-02 — 400c convergence probe: the oscillation thread is CLOSED

h16_conv (`render_full_dt_iso_nn`, N=40k, 400c budget): **froze at commit 282**
(phys=54.85) and held perfectly still for the remaining 118 commits (jitter 7e-5,
hole 0.00%, drift within G3). Final: chamfer 0.0709, silIoU 0.9660. Verdict: the
user-visible micro-motion at production budgets (60-200c) was honest unfinished
descent, not a pathology; the system has a true resting point at ~282 commits and
rests there. Combined with w_jvol (tail move halved) this closes the oscillation
complaint end-to-end. Also: G4_holes gate is now TARGET-RELATIVE (pass if hole <=
max(2%, target's own hole + 0.5pt)) — the pre-registered fix for the A->C metric
ceiling (target itself measures 5.76%).

### 2026-09-02 — h17/h18: oscillation criteria met; fill v3 works, verdict at convergence

**h17 (4 pairs, 300c): ALL freeze in budget** — bunny 259, spot 235, armadillo 133,
teapot 61 — with held tails at jitter 6-8e-5 (G3 x40 margin). Oscillation dossier's
quantitative criteria are met; replay artifact updated with the converged run (the
tail is a true rest). New pair records: spot 0.0672/0.9702.

**h18 (full stack + fill v3, 300c): chamfer 0.0662 / silIoU 0.9732 — both all-time
records** (fill v3 helps globally, not just the ears). Ear cov<0.3: 26.5→18.7%
(target <15%), cov<0.1 halved to 6.0%, fork lo-band 97→64. fill_lam self-anneals
0.062→0.009 exactly as designed (demand-driven weight: dominance impossible).
Falsifier threshold NOT yet met — but the run did NOT converge in 300c (d_fill still
descending at 0.206); the pre-registered threshold presumes the resting state, so
the verdict moves to h19 (same config, 400c). No parameter was changed.

## 2026-09-02 (late) — pacing, gate v1→v3, Tier D ladder, root-cause probes

All N=20k, T=20, dx=0.5, dt=1/240, 64³, smoothing 0.955, loss_res 64, flagship stack
(w_dt 0.2, w_nn 0.2, w_jvol 50, assim_iso) unless noted; results under
/data/relcfd/chayo/physmorph_v2/output (b8/e4 on hyde01 ~/physmorph_v2/output).

| run | arm / change | anims | chamfer | out_nn>2sp | far>3sp | note |
|---|---|---|---|---|---|---|
| g2_anneal | flagship + anneal 0.7 (40k) | 300 | **0.0701** | ~11% | — | conv 226; rev-cos −0.523→−0.345 vs h17 |
| g3_ref / g3_mix / g1_pilot | flagship / +gauss_mix / pure gauss | 120 | 0.089 / 0.108 / 0.099 | 9.8 / 19.1 / 15.0% | 520 / 1318 / 881 | gauss in objective loses at matched N |
| s1..s5 | render_stable_gauss bundle | 300 | 0.14–0.18 | 40–57% | — | froze at a27–a69: gate latch (3 forensics), w_cov (a17–19 regression), state-poison from ~a60 |
| s2_main / s2_cov25 | stable, w_cov 0 vs 25 | 300 | 0.0950 / 0.1026 | 12.8 / 17.9% | 860 / 1363 | w_cov guilty → retired |
| b0 / b1 | flagship+pace 0.01 / +gauss_mix | 300 | 0.234 / 0.190 | 68 / 59% | — | froze a70/a61: plateau tol vs glidepath → pace_bound exemption |
| b2 / b3 | same, pace_bound fix | 300 | **0.1078** / 0.2415 | 22.3 / 66.5% | 1743 / 9983 | b2 = first full-timeline paced run; b3 gauss_mix blew up late |
| b4 / b5 | 450@ρ0.01 / 300@ρ0.003 | 450 / 300 | 0.146 / 0.120 | 40.4 / 27.0% | 3972 / 2255 | best d_vol 69@a381 then limit cycle (λ antiphase) |
| b6 / b7 | + outer gate v1 / v2 | 450 | 0.151 / 0.138 | 47.5 / 41.8% | 5737 / 4677 | v1 froze a94 (patience); v2 pinned by 333 brake rejects (kinetic merit) |
| t1_b0 / t1_d1 | stable + dressing 0/20 | 120 | 0.216 / 0.175 | 67.6 / 56.7% | — | INVALID: 101/88 brake rejects (objective≠gate) |
| t2_b0 / t2_d1 | render_flag_dress 0/20 | 120 | 0.155 / 0.131 | 48.9 / 36.1% | 5526 / 3251 | Tier D K-D2 FAIL (d_gauss −3.6..5.1%); raw spread = path divergence |
| e2_assim08 / e2_assim10 | flagship+pace, η 0.8 / 1.0 | 300 | 0.134 / 0.1125 | 38.8 / 25.5% | 3464 / 2135 | spring-back 0.20 at all η — not a lever |
| b8_gate_v3 | gate v3 (shape-only merit) + best-commit truncation | 450 | 0.163 | — | — | 354 brake rejects, pinned a89: momentum-carrying paced commits → quasi-static rule (b9) |
| e4_nnfar | nn_far_k 4.5 → 1000 (own far clumps) | 300 | 0.1067 | 20.5% | 1511 | vs b2 tie/−8%/−13%, max_dt 23.6sp, guards 0 → ADOPTED; truncation fired at a291 |

Probes (docs/probes/): transfer_function (adjoint gain 0.89 vs 1.03 — §2 refuted),
sobolev_precond (no-op; s≤0.8 falsified; render descends at parity), material_carrier
(Tier M NO-GO: dFc reproduces material motion at 0.9% residual / 1/1700 cost),
observability (97% of floaters see the silhouette; median cos to target 0.3; surface-parent
gauss blind to the interior half). Ops: hyde06 key rejected from ~12:10; b8/e4 moved to
hyde01 GPU1 (warp-lang 1.16 installed into miniconda3/envs/diffmpm_v2.3.0).
| r1_record / r2_nomom | 40k, anneal 0.7, E4, pace 0.12, mom_carry 0.9 / 0 | 300 | 0.0725 / **0.0730** | 12.1 / 12.4% | 1731 / 1733 | r1 cascaded 82→489 after a91 (rescued by truncation); r2 clean, conv 104 → mom_carry RETIRED |
| r3_unpaced_gate | 20k, **--pace 0**, anneal 0.7, mom 0, E4, gate v3 brake | 300 | 0.0879* | 8.6%* | 417* | **best d_vol 23.4 @a217** (prev best 62), 221 commits, 2 brake rejects, guards 0; *metrics at the (wrongly) truncated a150 — the c2f res change made d_sil jump; a217 frames lost → truncation fixed (no frame drops, d_sil excluded) |
| r4_record40 | 40k, pace 0.12 (arm default), anneal, mom 0, E4, gate | 300 | 0.0731 | 12.2% | 1665 | conv 162, best d_vol 84.8, brake 34; pace 0.12 itself caps convergence depth (r2/r4 freeze at ~100-160) |
| r3b_p0_full | 20k, pace 0 recipe (re-run, full archive) | 300 | 0.0872 | **8.1%** | **419** | REPRODUCES r3: best d_vol 23.8@a181, conv 187, brake 3, guards 0, delivered a180 (3522/3643 frames, none dropped) |
| r5_record40_p0 | 40k, pace 0 recipe | 300 | **0.0714** | 11.0% | 1443 | ≈ g2 (0.0701) with brake 0 / guards 0 / no runaway; conv 162; best d_vol 72.9 |

**Recipe adopted 2026-09-03:** `render_full_dt_iso_nn --pace 0 --anneal 0.7 --mom_carry 0
--nn_far_k 1000 --outer_merit` (now the arm defaults: pace 0, outer_merit on).
| g_arma_p0 | sphere→armadillo, 20k, pace-0 recipe | 300 | 0.1074 | **1.6%** | **35** | GENERALIZES: max_dt 5.8sp, hole 0.00%, conv 129, brake 1, guards 0 (bunny ears are the hard case) |
| x3_kde20 | 20k, pace-0 recipe + one-sided KDE density term (w_kde 1) | 300 | 0.1016 | 21.9% | 1812 | FALSIFIED: cluster ratio 2.0→1.25 ✓ but out_nn ×2.7, chamfer +16% — exterior particles see rho_t≈0 and get repelled; → two-sided form (x5) |
| x1_lossres96 | 40k, pace-0 recipe, loss_res 64→96 | 300 | 0.0749 | 16.8% | 2659 | FALSIFIED: +5% chamfer, floaters worse, froze at a77 (sparse CIC is noisier); cluster 2.7→2.0 only. Finer grids are N-limited. |
| x5_kde2_20 | 20k, two-sided KDE (w_kde 1) | 300 | 0.0911 | 15.7% | 1241 | interior ✓ (cluster 1.25, unfilled 27%, best d_vol 17.9) but fringe still scattered (out_nn 8→16%, chamfer +4.5%) → v3 support-weighted particle side (x6) |
| x2_berth1 | 40k, pace-0 recipe, nn_berth_k 1.5→1.0 | 300 | **0.0706** | **9.5%** | 1318 | ADOPTED: vs r5 tie+/−14%/−9%, max_dt 18.0, jitter 5e-5, guards 0, conv 191; clustering unchanged (42%) |
| x6_kde3_20 | 20k, KDE v3 (support²-weighted particle side) | 300 | 0.0898 | 14.3% | 1043 | KDE FAMILY FALSIFIED for the fringe (3 variants: 21.9 / 15.7 / 14.3% vs 8.1%); interior de-clustering real (2.0→1.25-1.33) but the target-side term treats exterior particles near full faces as surplus. Clustering needs an in-simulator remedy. |

> **SHELL-ERA WARNING (2026-09-03):** every bunny/armadillo result above this line was
> produced against a SURFACE-SHELL target (sampler bug, see docs/root_analysis.md
> REVISION 2): the converged bodies were thin layers, "out_nn" measured layer thickness,
> and chamfer 0.070 was the shell-vs-shell floor. Optimizer-side conclusions (pace 0,
> mom_carry retired, gate v3, best-commit delivery, E4/berth ownership) stand; every
> geometric number must be re-established on real volumes (v-series below).

## Real-volume era (sampler fixed: axis fill + interior check + target volume matched to source)

| run | arm / change | anims | chamfer | out_nn>2sp | far>3sp | note |
|---|---|---|---|---|---|---|
| v1_solid20 | 20k bunny solid, adopted recipe | 300 | 0.1591* | **1.1%** | 28 | max_dt 3.9sp, uncovered@1.5sp 16.1% (floor 11.2%) @2sp 3.3% (floor 1.0%), silIoU 0.965, hole 0.14%, guards 0, brake 0, conv 139 — floaters at the sampling floor on a real solid |
| v2_solid_arma | 20k armadillo solid | 300 | 0.1752* | **1.6%** | 12 | first real-volume result: silIoU 0.967, hole 0.02%, max_dt 4.1sp, guards 0, brake 0, conv 128, body spacing 0.071 vs target 0.080; uncovered@1.5sp 27.7% (@2sp 7.9%) — *volumetric chamfer, not comparable to shell-era numbers |
| v4_solid20_rev | 20k solid, unlatched reversal REJECT | 300 | 0.1619 | 1.7% | 41 | FALSIFIED: reversal rejects fed patience → froze at a56 mid-descent (d_vol 46 vs v1 30); tail rev-cos +0.27 only because it stopped early → v6 sign-change step control instead |
| v5_solid20_jdens | 20k solid + density-J prior (w_jdens 1) | 300 | 0.389 | — | — | CALIBRATION BUG: equal-norm at the source where J≡1 → sKL gradient 0 → scale astronomical → every step rejected, froze a9. Fixed: calibration deferred to the first window with a live prior gradient, scale capped 1e3 → v5b |
| v3_solid40 | 40k solid bunny, adopted recipe | 300 | 0.1298* | **1.5%** | 100 | HERO BASELINE on a real volume: silIoU 0.968, hole 0.00%, jitter 1e-5, max_dt 5.1sp, uncovered 19.9%/5.1%, guards 0, brake 0, conv 164; body 7% over-dense; **ear-region fraction 0.069 vs 0.100 (−31%)** — thin-feature transport is the remaining geometric defect |
| v6_solid20_annrev | 20k solid, anneal ×0.5 on commit reversal | 300 | 0.1619 | 1.6% | 29 | FALSIFIED as a schedule: ×0.5 on reversal vs ×1.15 recovery with a 0.05 floor collapses α (froze a71, d_vol 43 vs v1 30); rev-cos −0.11 only by stopping early. With v4 this falsifies the implemented reversal controllers, not reversal as a signal (REFUTE Opus F7); v4's untried fix would be the reject-type split. v4/v6 chamfers 0.16195/0.16194 are different states (frames 1022/1305) at the 20k volumetric chamfer floor. |
| v5b_solid20_jdens | 20k solid, density-J prior, deferred equal-norm calibration | 300 | — | — | — | FALSIFIED (killed at a92): every candidate from a2 rejected (gain −1.13, reversal 0.93). Deferred calibration fired at a near-zero prior gradient (scale capped 1e3) and the term then dominated: candidates continued the previous step (reversal_cos +0.93 = aligned) and doubled the merit; the cold restart replayed the identical candidate, brake rejects did not count stale → infinite loop (gate bug, fixed: REPLAYED rejects count as stale). Falsifies parity-at-zero-gradient calibration, not the prior class (REFUTE F7). |

### 2026-09-04 — the solid hero's residual is a vertical under-stretch; v7 pre-registration

y-slab census of the delivered v3 (solid bunny 40k) state, current/target particle ratio:
bottom y<-2: 0.68 · y -2..-1.5: 0.7 · y -1.5..-0.5: ~1.0 · **y -0.5..+0.5: 1.67 / 1.35 / 1.22** ·
y +1..+2.5: 0.81 / 0.73 / 0.72 / 0.58. The sphere's equatorial band never stretched to the
bunny's poles: 30% surplus in the middle, 20-40% deficits at BOTH the ears and the feet, all
inside a silhouette that is already right (silIoU 0.968). D_vol sees it (127 cells with a
>50% deficit, 94 with a >50% surplus at 0.5 wu cells, 54 ppc) but its CIC gradient is the
difference of neighbouring cell residuals: zero inside a uniform surplus band, non-zero only
at its edges — the band erodes like diffusion. The per-commit gain decays geometrically from
a40 while anneal is still 1.0 and every commit is accepted, so neither the step schedule nor
the gate is the cap: the LOCAL data term is. This is the "differential view" failure of
Kugelstadt et al. 2021 (Implicit Density Projection): the solver never sees the accumulated
density residual as a field; the documented cure is a Poisson solve with that residual as
source. Ear width: 1.27 wu = 2.5 cells at dx 0.5 (5 at 0.25) — resolution is a second,
separate suspect, held for later.

**v7 (pre-registered): H⁻¹ mass balance, `--w_h1 1`** (parity with D_vol's gradient norm at
the source; docs/method.md). Arms: v7a solid bunny 20k (baseline v1: d_vol 30, ear frac
0.078, conv 139, out_nn 1.08%), v7b solid bunny 40k (baseline v3: 57.8, 0.069, 164, 1.49%).
Success: band ratio (y −0.5..0.5) ≤ 1.15 and pole ratios ≥ 0.85; ear frac ≥ 0.09 (20k) /
≥ 0.085 (40k); best d_vol below baseline; out_nn ≤ baseline + 0.3 pt; guards 0; no freeze
before a100. Failure: out_nn up > 0.5 pt, or a brake-reject streak (gate now ends such runs).

### 2026-09-04 — v7a verdict (H⁻¹ mass balance, 20k solid bunny, in-core, uncorrected)

| metric | v1 baseline | v7a | criterion | |
|---|---|---|---|---|
| best d_vol (commit) | 29.95 (a134) | **4.69 (a103)** | below baseline | ✓ |
| band ratios bottom/mid/top | (v3 40k: 0.70/1.46/0.68) | **1.01 / 1.04 / 1.04** | mid ≤ 1.15, poles ≥ 0.85 | ✓ |
| ear-region frac (target 0.100) | 0.078 | **0.105** | ≥ 0.09 | ✓ |
| J_true ears / body | 1.32 / 0.94 | **0.91 / 1.07** | — | |
| uncovered @1.5sp / @2sp (floor 11.2 / 1.0) | 16.1 / 3.3 | **12.5 / 1.7** | — | |
| out_nn>2sp / far>3sp / max_dt | 1.08% / 28 / — | 1.31% / 20 / 4.1 sp | out_nn ≤ 1.38 | ✓ |
| chamfer / silIoU / hole | 0.1591 / 0.9653 / 0.14% | 0.1571 / 0.9654 / 0.00% | — | |
| anims / accepted / rejects / brake / guards | 139 / 135 / — / 0 / 0 | 107 / 101 / 4 / 0 / 0 | no freeze < a100 | ✓ |
| tail rev-cos / osc% / move / frac>0.5sp | −0.44 / 61% / — / 0 | −0.53 / 74% / 0.094 sp / 0.22% | reopen rule >1% | holds |

The vertical under-stretch is gone (band 1.46 → 1.04 at the same geometry class), the ears
are filled to the target fraction, coverage is at the Poisson floor + 1.3 pt, floaters within
tolerance. d_h1 fell 205k → 264 (780×). v7a ran commit 7ed1337 (H⁻¹ inside the core, no
self-energy correction, c2f recalibration live): the REFUTE-Opus F1 test (sub-cell clustering
at equal d_vol vs v1 — `paired_census.py`) decides whether the uncorrected term already
traded sub-cell quality; v8 (self-corrected) is the recipe candidate either way.

**Paired census (REFUTE Opus F1/g, `paired_census.py`; chamfer here = mean NN both ways/2):**

| state | d_vol | chamfer | out_nn | far | uncovered | ear | bands | cluster ratio | CIC centre excess |
|---|---|---|---|---|---|---|---|---|---|
| v1 best a135 | 29.95 | 0.0797 | 1.20% | 28 | 15.9 / 3.2 | 0.078 | 0.79/1.32/0.76 | 1.23 | +0.006 |
| v7a @ d_vol≤29.95 (a13) | 29.12 | 0.0823 | 3.48% | 157 | 13.2 / 1.8 | 0.105 | 1.02/1.00/1.05 | 1.17 | +0.003 |
| v7a best a103 | 4.69 | **0.0787** | 1.32% | **22** | **12.5 / 1.7** | **0.105** | **1.01/1.04/1.05** | **1.18** | +0.019 |

Reading: the feared trade (cell-scale mass bought with sub-cell quality, the x3 signature)
did not occur — the cluster ratio IMPROVED (1.23 → 1.18) and every geometric metric but
out_nn (+0.12 pt, within the pre-registered +0.3) is better at best-vs-best. The equal-d_vol
row is confounded by commit age (a13 is early-morph spray, 120 commits before the cleanup
terms act); it still shows the mass distribution right from the start. The lattice attractor
leaves a faint fingerprint (+1.9% CIC centre excess vs +0.6%); v8a (self-corrected) should
remove it. Tail amplitude in v7a is larger than v3's (p2p p99 0.71 sp, ears 0.61) and
marginally trips the reopening rule — held for the v8a census before reopening.
**H⁻¹ ADOPTED provisionally (v8 = self-corrected, in-core) pending v8a; v7b (40k) running.**


## 2026-09-14 — render-controls-physics ladder (PRE-REGISTERED, NOT RUN)

Design and falsifiers: `docs/render_controls_physics.md` §10. Server access failed all day
(`ssh -J chayo@hyde01.dabh.io` → `Permission denied (publickey,password)` at the jump host;
the ed25519 key must be re-registered in JumpCloud). Nothing below has a number yet.
Local verification: 128 → 186 CPU tests green (`python -m pytest tests/ -q`), after the REFUTE round `docs/reviews/refute_rcp_opus_20260915.md` (12 findings, all answered).

Run commands (hyde06, after `cd ~/physmorph_v2` and a fresh deploy of this branch;
thread caps + staggered launches per the ops rules):

```bash
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
OUT=/data/relcfd/chayo/physmorph_v2/output
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
# persistent viewer (once): serves every run under $OUT/live on port 8765
setsid nohup $PY scripts/viewer_serve.py --root $OUT/live --port 8765 > $OUT/viewer_serve.log 2>&1 < /dev/null &
# baseline vs the contract arms, 20k, pace-0 recipe, live-published
CUDA_VISIBLE_DEVICES=<free> setsid nohup $PY scripts/pipeline_run.py \
    --arms render_full_dt_iso_nn,render_ctrl,render_ctrl_gauss,render_ctrl_first \
    --n 20000 --animations 300 --loss_res 64 --pace 0 --anneal 0.7 --mom_carry 0 \
    --nn_far_k 1000 --live_dir $OUT/live --out $OUT/rcp_20k > $OUT/rcp_20k.log 2>&1 < /dev/null &
# density units and the discretisation contract (separate GPUs, staggered 45 s)
... --arms render_ctrl --loss_units density --out $OUT/rcp_20k_density
... --arms render_ctrl --ppc 8 --loss_units density --out $OUT/rcp_20k_ppc8
# oscillation triage on every archive
$PY scripts/probes/oscillation_triage.py --npz $OUT/rcp_20k_render_ctrl.npz --json $OUT/rcp_20k.json \
    --arm render_ctrl --out $OUT/rcp_20k_render_ctrl_triage.json --png $OUT/rcp_20k_render_ctrl_triage.png
```

Local: `python scripts/viewer_tunnel.py --open` keeps the tunnel and opens
http://127.0.0.1:8765 (run selector, replay scrub, /quad, /compare).

| arm | falsifier (pre-registered) |
|---|---|
| render_ctrl | chamfer > flagship +2 %, or hole_frac ↑, or any G2 guard |
| render_ctrl_gauss | d_gauss not below render_ctrl's at equal d_vol |
| render_ctrl_first | any G2 guard, or chamfer +2 % vs render_ctrl |
| render_ctrl --loss_units density | λ trace not O(1), or loss_res 32→64 changes d_vol by >1.6× |
| render_ctrl --ppc 8 --loss_units density (and at 5k) | hole_frac not ≤ the fixed-dx run's at 5k |
| triage probe | driver C must vanish under w_kin_running; B/A per the rules |


### 2026-09-15 — render-controls-physics ladder, batches a–c (hyde06, GPU 0/2)

Discretisation for every row unless noted: N=20000 sphere→bunny (real-volume sampler, source
volume 50.82 wu³, target matched), T=20, dt=1/240, dx=0.5, 64³ MPM grid, smoothing 0.955,
loss_res 64, pace 0, anneal 0.7, mom_carry 0, nn_far_k 1000, w_kin 0.5 (the CLI default),
w_dt 0.2, w_nn 0.2, w_jvol 50, dfc_clip 0.02, 300-commit budget; every run froze on the
plateau rule at the listed commit count (3–6 min each on an RTX 6000 Ada). Code: commits
`ce39df1`…`3f1a0dd`. Metrics from raw state (metrics.py); gates as in the table header.

| arm | commits | chamfer | silIoU | hole | detFmin | G2 | G3 (drift) | note |
|---|---|---|---|---|---|---|---|---|
| `render_full_dt_iso_nn` (flagship baseline) | 111 | **0.1599** | **0.9655** | 0.04% | 0.745 | 0 | FAIL 0.0035 | λ med 1150 (cap never binds), kin_T 0.33, best d_vol 30.2 |
| `render_ctrl` (basis 12³×4, F_g render, kin_run 1, Chebyshev, w_creg 0) | 102 | 0.1617 | 0.9533 | 0.09% | 0.884 | 0 | FAIL 0.0031 | λ 430, kin_T 0.19, kin_run 0.079, best d_vol 33.9 |
| `render_ctrl_gauss` (+ hybrid 3DGS L1, Charbonnier 0.02) | 90 | 0.1621 | 0.9566 | 0.03% | 0.907 | 0 | FAIL 0.0032 | |
| `render_ctrl_first` (+ physics projected off render) | 73 | 0.1626 | 0.9569 | 0.04% | 0.910 | 0 | FAIL 0.0035 | best d_vol 41.8 — the render-first cone slows the mass descent |
| `render_ctrl --loss_units density` | 116 | 0.1660 | 0.8974 | 0.20% | 0.879 | 0 | PASS 0.0027 | λ 0.02–0.09 (O(1) ✓) but silIoU −6.8 pt: the source-calibrated weight conversion drifts along the morph — FALSIFIED as configured |
| `render_ctrl --ppc 8 --loss_units density` (dx 0.271, 118³, loss_res 118, CFL 0.305, measured ppc median 7.0) | 74 | **0.1552** | 0.9625 | 0.02% | **0.937** | 0 | FAIL 0.0036 | best chamfer of the ladder (−3 %); the discretisation contract, not the objective, moved the number |
| `render_ctrl --control_grid 6` (144/216 nodes empty) | — | 0.1704 | 0.9483 | 0.12% | 0.978 | 0 | PASS | too coarse |
| `render_ctrl --control_grid 24` | — | 0.1597 | 0.9570 | 0.07% | 0.818 | 0 | PASS | recovers the baseline chamfer; basis resolution IS a lever (6 < 12 < 24 monotone) |
| `render_ctrl --control_tknots 20` (per-step in time) | — | 0.1616 | 0.9513 | 0.11% | 0.899 | 0 | PASS | time knots are not the lever |
| baseline `--dfc_clip 0` | — | 0.2060 | 0.8461 | 0.21% | 0.693 | 0 | FAIL jitter 1.4e-3 | froze in 0.5 min; the clip is part of the recipe for BOTH families (REFUTE F11 answered: reported clipped AND unclipped) |
| `render_ctrl --dfc_clip 0` | — | 0.2080 | 0.8455 | 0.38% | 0.664 | 0 | FAIL | same |

Verdicts against the pre-registered falsifiers (`docs/render_controls_physics.md` §10):
- `render_ctrl`: chamfer +1.1 % (inside the +2 % bound) but hole 0.09 % vs 0.04 % — the
  hole clause fires on paper while both sit far below the 2 % gate and the target's own
  0.01 %; silIoU −1.2 pt. Verdict: a TIE on shape with a markedly better inversion margin
  (detFmin 0.75 → 0.88) and half the terminal kinetic energy. NOT adopted as flagship;
  kept as an arm. Quicklook QA (frames 0/400/800/1200/1600/end, 2 views): all three
  render arms closed solids, no ghost/floaters; render_ctrl's ear tips slightly blunter
  than the baseline, `_gauss` restores them.
- `render_ctrl_gauss`: the d_gauss criterion is unmeasurable against a baseline that has
  no Gaussian term; on the shared metrics it ties `render_ctrl` (+0.3 pt silIoU).
- `render_ctrl_first`: no guard, chamfer +1.7 % vs baseline, but the mass objective
  converges worse (41.8 vs 30.2) — the render-first projection is not adopted.
- density units: falsified as configured (silIoU −6.8 pt) even though λ is O(1) as
  predicted; the weight conversion is exact only at the source.
- `--ppc 8`: chamfer 0.1552 is the best number in the ladder and holes 0.02 %; G3 drift
  0.0036 just over the 0.003 gate (as for every dx-0.5 render arm here). The
  discretisation contract is the one change that improved the shape metric.
- Grid sweep 6/12/24 is monotone in chamfer → the basis resolution is a lever (the
  pre-registered "non-monotone ⇒ not the lever" did not fire); 24³ recovers the baseline.

**Oscillation triage (all six a/b/density/ppc archives, `scripts/probes/oscillation_triage.py`):**
VISIBLE (10–69 % of particles with tail excursion > 0.5 sp) with driver **C_control** on
every arm — inside every 20-step window the mean speed goes 0.47 → 0.10 → 0.45 (turning
point mid-window, continuous across the boundary), 95 % of the speed power at period T;
elastic period 132 steps, J peak-to-peak 0.008, CFL 0.24 (0.37 at ppc 8) rule out
stiffness and volume. The first probe version missed it (sag 0, jump 0.93); the rule now
includes the intra-window modulation max/min (measured 2.7–4.1 vs threshold 2). This is
the answer to the user's "진동" question: a control limit cycle sustained by the
terminal-only objective with a weak terminal kinetic weight (w_kin 0.5), not a physics
instability. Remedies pre-registered and running as batch d: `w_kin_var` 10/50
(velocity variance over the window), `w_kin 5`, `w_kin_running 10`; falsifier = the
window-locked power fraction must drop below 0.5 and the visible fraction below 1 %
without a chamfer regression > 2 %.


### 2026-09-15 — batch d: oscillation remedies (same discretisation as batches a–c)

Falsifier (pre-registered above): the window-locked speed power fraction must drop below
0.5 and the visible fraction (tail excursion > 0.5 sp) below 1 % without a chamfer
regression > 2 %. Triage columns: modulation = median intra-window max/min speed,
power = spectral power fraction at period T, visible = fraction of particles.

| arm | chamfer | silIoU | hole | G3 (drift) | s̄ (wu/s) | kin_T | kin_var | modulation | power | visible | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (w_kin 0.5, batch a) | 0.1599 | 0.9655 | 0.04% | FAIL 0.0035 | 0.308 | 0.33 | — (build predates kin_var; kin_run 0.128) | 4.1 | 0.95 | 9.6 % | C_control |
| baseline `--w_kin 5` | 0.1593 | 0.9663 | 0.04% | PASS 0.0018 | 0.206 | 0.12 | 0.042 | 2.9 | 0.81 | 5.9 % | C_control (amplitude ↓, cycle intact) |
| baseline `--w_kin_running 10` | 0.1586 | 0.9668 | 0.11% | FAIL 0.0030 | 0.247 | 0.21 | 0.064 | 3.4 | 0.85 | 7.3 % | C_control |
| baseline `--w_kin_var 10` | 0.1589 | 0.9644 | 0.10% | PASS 0.0024 | 0.248 | 0.18 | 0.057 | 3.2 | 0.84 | 10.5 % | C_control |
| baseline `--w_kin_var 50` | 0.1596 | 0.9655 | 0.03% | PASS 0.0012 | **0.134** | **0.033** | **0.0083** | **2.1** | **0.58** | **2.5 %** | C_control, at the threshold |
| `render_ctrl --w_kin 5` | 0.1618 | 0.9574 | 0.10% | PASS 0.0024 | 0.241 | 0.13 | 0.038 | 2.5 | 0.87 | 23.7 % | C_control |
| `render_ctrl --w_kin_var 10` | 0.1616 | 0.9562 | 0.08% | PASS 0.0027 | 0.260 | 0.15 | 0.043 | 2.5 | 0.85 | 44.9 % | C_control |

Reading. The velocity-variance term is the mechanism-matched lever: at w_kin_var 50 the
mean speed falls 2.3×, the terminal kinetic energy 10×, the window-locked power 0.95 →
0.58 and the visible fraction 9.6 → 2.5 %, with chamfer/silIoU/holes UNCHANGED (0.1596 /
0.9655 / 0.03 %) and G3 passing with 2.5× margin. The classic terminal weight (w_kin 5)
and the running kinetic term (10) only shave the amplitude (power 0.81–0.85). The
dose-response 10 → 50 is monotone; the falsifier is not yet met (power 0.58 > 0.5,
visible 2.5 % > 1 %), so batch f continues the ladder at w_kin_var 200 and batch e tests
the cause side (control continuity across windows, `--warm_start`). The basis arms
oscillate MORE visibly (excursion p99 1.0–1.3 sp vs 0.6–0.8): a coarse control basis
makes the cycle spatially coherent, so the same energy moves whole regions — a second
reason the per-particle flagship stays.


### 2026-09-15 — batches e/f: cause test (warm start) and the w_kin_var dose-response

Same discretisation as batches a–d (20k, T=20, dt=1/240, dx=0.5, loss_res 64, pace 0).
All rows are the flagship baseline arm `render_full_dt_iso_nn` unless marked.

| change | commits | chamfer | silIoU | hole | G3 (drift) | s̄ | kin_T | kin_var | modulation | power @T | visible | triage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `--warm_start --w_kin 5` | 125 | 0.1588 | 0.9655 | 0.01% | PASS 0.0020 | 0.222 | 0.140 | 0.048 | 2.9 | 0.90 | 10.5 % | C_control — continuity across windows does NOT remove the cycle |
| `render_ctrl --warm_start --w_kin 5` | 126 | 0.1614 | 0.9556 | 0.01% | PASS 0.0027 | 0.210 | 0.109 | 0.033 | 2.7 | 0.87 | 11.1 % | C_control |
| `--w_kin 5 --w_kin_var 50` | 127 | 0.1591 | 0.9651 | 0.00% | PASS 0.0007 | 0.113 | 0.019 | 0.0053 | 1.97 | 0.50 | 2.1 % | at the threshold |
| `--warm_start --w_kin 5 --w_kin_var 50` | 141 | **0.1584** | 0.9646 | 0.09% | PASS 0.0007 | 0.107 | 0.015 | 0.0048 | 2.02 | 0.50 | **0.3 %** | **INVISIBLE (sub-spacing)** — visible criterion met |
| `--w_kin_var 200` (w_kin 0.5) | 121 | 0.1592 | 0.9639 | 0.02% | PASS 0.0001 | 0.089 | 0.005 | 0.0008 | 1.27 | **0.06** | 1.6 % | window lock GONE (power 0.95 → 0.06); the residual 1.6 % is the known flat-valley random walk, not window-locked |
| `--warm_start --w_kin 5 --w_kin_var 200` (batch g) | 116 | 0.1596 | 0.9647 | 0.05% | PASS 0.0003 | 0.091 | 0.005 | 0.0010 | 1.23 | **0.04** | 1.9 % | window lock gone; residual = flat-valley random walk |
| `render_ctrl --control_grid 24 --w_kin_var 50` (batch f2) | 106 | 0.1597 | 0.9589 | 0.04% | PASS 0.0015 | 0.210 | 0.098 | 0.0286 | 2.3 | 0.81 | 23.7 % | C_control — on the coarse basis the same weight buys 3× less variance reduction (coherent motion is cheap per node): the basis arms need a higher w_kin_var; flagship stays per-particle |

Reading against the pre-registered falsifier (power < 0.5, visible < 1 %, chamfer within
+2 %): `w_kin_var 200` removes the window-locked component outright (power 0.06,
modulation 1.27, drift 1e-4) at chamfer −0.4 % / silIoU −0.16 pt; `warm start + w_kin 5 +
w_kin_var 50` drives the visible fraction to 0.3 % at the best chamfer of the family
(0.1584). Neither single row meets both criteria at once; the combined point
(`--warm_start --w_kin 5 --w_kin_var 200`, batch g) and the basis arm at 24³ with
w_kin_var 50 (batch f2) are the last two rungs. The cause is settled: a per-window
terminal-only objective admits push-and-return trajectories for free, and pricing the
in-window velocity variance is the term that targets exactly that, with no measurable
shape cost at 20k.


### 2026-09-15 — ladder verdict and recommended recipe (NOT yet adopted as a default)

Fourteen arms, all 20k / T=20 / dt=1/240 / dx=0.5 / loss_res 64 unless noted, one seed,
3–6 min each; every number carries its discretisation in the tables above.

1. **The in-simulation vibration is a window-locked control limit cycle**, measured, not
   inferred: speed V-shaped inside every window, 95 % of the power at period T, elastic
   period and J variation uninvolved. Cold start is not the cause (warm start: power
   0.90). The velocity-variance term is the matched lever: `w_kin_var 200` removes the
   window lock outright (power 0.95 → 0.04–0.06, modulation 4.1 → 1.2, drift 3e-4 →
   1e-4–3e-4) with chamfer −0.2…−0.4 % and silIoU −0.1…−0.2 pt; `warm_start + w_kin 5 +
   w_kin_var 50` reaches visible 0.3 % (INVISIBLE by the dossier rule) at the family's
   best chamfer 0.1584 while leaving half the window-locked power. The two pre-registered
   criteria (power < 0.5 AND visible < 1 %) are met by different rows, not by one; the
   residual visible 1.6–1.9 % under kv200 is the flat-valley random walk already
   characterised in docs/oscillation.md (Addendum 7), not window-locked motion.
   **Recommended recipe for the next replicate:** flagship + `--w_kin 5 --w_kin_var 50
   --warm_start` (visibility first) or `--w_kin_var 200` (spectral cleanliness first);
   adoption as a CLI default waits for the 40k replicate and a REFUTE round.
2. **The render-controls-physics arms tie the flagship on shape** (chamfer +1.1…+1.7 %,
   silIoU −0.9…−1.2 pt), with a far better inversion margin (detFmin 0.75 → 0.88–0.94)
   and half the terminal kinetic energy; basis resolution is a lever (6 < 12 < 24, 24³
   recovers the baseline chamfer); the coarse basis makes the limit cycle spatially
   coherent, so those arms need a larger `w_kin_var`. Not adopted as flagship.
3. **The discretisation contract moved the shape number**: `--ppc 8` (dx 0.271, 118³,
   loss grid following dx in density units) gave the ladder's best chamfer 0.1552 with
   holes 0.02 %; density units at the legacy dx were falsified (silIoU 0.897).
4. `dfc_clip 0` collapses both families (0.206 / 0.846 in 16 commits): the clip is part
   of the recipe, so basis-vs-flagship comparisons at equal clip are the fair ones.


**Caveat added 2026-09-15 (after photoreal QA):** the 20k `render_ctrl_gauss` and
`render_ctrl_first` arms rendered their Gaussian loss from the unrelaxed total `F_g`,
which reaches singular values 3–5 on the ears late in the morph (needle splats). The
runner now relaxes `F_g` at every accepted commit with the physics assimilation rule
(`relax_stretch`); those two rows are not re-run here — their silhouette/chamfer numbers
stand (the silhouette term does not use F), their d_gauss telemetry does not.


### 2026-09-15 — 40k replicate (batch h, hyde06 GPU 0/2)

N=40000 sphere→bunny (real-volume sampler, source volume 50.82 wu³), T=20, dt=1/240,
dx=0.5 / 64³ (rows 1–3, 6) or dx=0.2712 / 149³ with the loss grid following dx in density
units (rows 4–5; measured unit ratios at the source: D_vol legacy/density 4.16e4, gradient
3.88e4, n_support 8712, m_ref 4.59), loss_res 64 unless noted, pace 0, anneal 0.7,
mom_carry 0, nn_far_k 1000, dfc_clip 0.02, 300-commit budget, one seed, 6–9 min per arm.
Triage columns as in batch d. The F_g relaxation (`relax_stretch`) was deployed while
batch h ran; it only affects `render_ctrl` archives (row 6) and no objective here.

| arm | commits | chamfer | silIoU | hole | detFmin | G3 (drift) | s̄ | kin_var | modulation | power @T | visible | triage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline `render_full_dt_iso_nn` | 126 | 0.1302 | 0.9626 | 0.01% | 0.622 | PASS 0.0005 | 0.167 | 0.0243 | 3.7 | 0.87 | 1.6 % | C_control |
| `--warm_start --w_kin 5 --w_kin_var 50` | 138 | **0.1296** | **0.9676** | 0.02% | 0.661 | PASS 0.0006 | 0.083 | 0.0020 | 1.7 | 0.29 | 2.6 % | no driver |
| `--w_kin_var 200` | 143 | 0.1304 | 0.9621 | 0.01% | 0.622 | PASS 0.0001 | 0.067 | 0.0004 | 1.3 | **0.07** | 1.4 % | no driver |
| `--ppc 8 --loss_units density` (dx 0.271, 149³) | 73 | **0.1164** | 0.9620 | 0.38% | 0.729 | PASS 0.0017 | 0.350 | 0.0694 | 3.3 | 0.79 | 38.9 % | C_control |
| `--ppc 8 --loss_units density --warm_start --w_kin 5 --w_kin_var 50` | 86 | 0.1169 | 0.9590 | **0.00%** | **0.756** | PASS 0.0010 | 0.277 | 0.0384 | 3.1 | 0.62 | 13.1 % | C_control — the same relative weight is too weak at the finer dx (2× the kinetic energy); needs a larger w_kin_var in density units |
| `render_ctrl --control_grid 24 --w_kin_var 200` | 132 | 0.1301 | 0.9570 | 0.04% | 0.676 | PASS 0.0009 | 0.099 | 0.0023 | 1.8 | 0.46 | 5.2 % | no driver; ties the baseline chamfer, silIoU −0.6 pt |
| `--ppc 8 --loss_units density --warm_start --w_kin 5 --w_kin_var 200` (batch i) | 128 | **0.1160** | 0.9645 | 0.47% | 0.733 | PASS 0.0013 | 0.212 | 0.0250 | 3.3 | 0.64 | 2.3 % | C_control (lock still present at the finer dx) |
| `--ppc 8 --loss_units density --warm_start --w_kin 5 --w_kin_var 500` (batch i) | 119 | **0.1160** | 0.9538 | 0.42% | 0.742 | PASS 0.0005 | 0.148 | 0.0039 | 2.0 | **0.10** | **0.6 %** | INVISIBLE — lock removed at the finer dx; silIoU −1 pt is the cost |

Verdicts (single seed; differences of ±0.5 % chamfer / ±0.5 pt silIoU are within what one
seed can resolve — see the REFUTE note below):
- The 20k oscillation verdict REPLICATES at 40k: baseline power 0.87 / modulation 3.7;
  `w_kin_var 200` removes the lock (0.07) and `warm_start + w_kin 5 + w_kin_var 50` reaches
  0.29 with chamfer −0.5 % and silIoU +0.5 pt vs baseline. Both keep every gate.
- `--ppc 8` (dx from N, loss grid following dx in density units) is the largest shape
  effect in the whole ladder: chamfer 0.1302 → 0.1164 (−10.6 %) at equal silIoU, with
  holes 0.38 % (0.00 % with the recipe). Its oscillation is 2× larger at the finer dx and
  the recipe at "50" only halves the power; the density-unit weight conversion is at the
  source only, so the next rung is `--ppc 8` with w_kin_var 200–500.
- The basis arm at 24³ with w_kin_var 200 ties the baseline chamfer at 40k (0.1301) with
  the lock removed; silIoU −0.6 pt remains the basis's cost.

**Recommended (pending REFUTE): flagship + `--warm_start --w_kin 5 --w_kin_var 50` at dx 0.5;
with the discretisation contract (`--ppc 8 --loss_units density`, chamfer −11 %) the same
recipe needs `w_kin_var` 200 (visible 2.3 %, silIoU 0.9645) or 500 (invisible, silIoU
0.9538) — the density-unit weight conversion holds at the source only, so the weight is
not transferable across dx without this re-tuning. Holes at ppc 8 sit at 0.4–0.5 % (baseline
0.01 %; gate 2 %), a finer-dx effect to watch at the deliverable stage.** No CLI default
has been changed.


### REFUTE-2 corrections to the 2026-09-15 sections (evening)

Review: `docs/reviews/refute_rcp2_opus_20260915.md` (19 findings). What changes in the
reading of the tables above:

- **Commit counts** in the batch a–c table are the number of records with a `d_vol`
  (accepted + outer-rejected); accepted-only counts are 3–5 lower. `truncation` and
  `deliver_n` are now written into the JSON as well.
- **Single seed, no replicates.** Every arm above is seed 1, and no two rows share a
  config. The 40k dx-0.5 family (base / ws_kv50 / kv200) spans chamfer 0.1296–0.1304
  (0.6 %) and silIoU 0.9621–0.9676 (0.55 pt) with a non-monotone dependence on
  `w_kin_var`, so the recipe's "−0.5 % / +0.5 pt" is a tie within noise, not an
  improvement; at 20k chamfer correlates −0.70 with the number of accepted commits. Seed
  replicates run in batch j.
- **Code hashes.** The 40k baseline and the ppc8 arm ran under an earlier build than the
  remedies (`code_hash` 55650941 vs ffae3933; the difference is the since-withdrawn F_g
  relaxation, inert for these arms); the baseline is re-run under the current build in
  batch j and every archive now carries the git sha (VERSION file in tarball deploys).
- **Visible < 1 % at 40k** is met by no arm, including the baseline (1.6 %); the
  window-locked component (power 0.87 → 0.29 / 0.07, tortuosity) is the replicating
  effect. `power_frac` is convention-dependent by 3–5×; both conventions are now stored.
- **`--ppc 8` arms (batches h, i)**: in density units the conversion was measured on the
  run's own 149³ grid, so every fixed weight (`w_kin`, `w_kin_var`, `w_jvol`, `w_dt`,
  `w_nn`, `w_box`, …) was ≈0.31× its legacy-64 meaning — the reason `w_kin_var` had to go
  to 200–500 there, and one of four simultaneous changes (dx, loss grid 149³, units,
  weights) behind the −10.6 % chamfer. Batch j decouples them (`--ppc 8` in legacy units
  at loss_res 64; density units with the reference-grid calibration). Requested ppc 8
  measures 7.0 on the source and 6.0 on the target (24–27 % of target cells below 4).
  Holes at ppc 8 spanned 0.00–0.47 % across four arms differing only in kinetic weights:
  noise at this scale.
- **G4_ejection** (`stray_max < 2e-3`, self-referential) fails on every 40k arm
  (stray_max 0.0026–0.0037) and passes on every 20k arm; it is omitted from the tables
  because its replacement was pre-registered earlier, but it is not passing.
- **G3 drift** was read from the last history record rather than the delivered slice;
  fixed in `pipeline_run.eval_gates` (affects truncated arms by one commit).
- The **F_g commit-time relaxation** of `502b543` is withdrawn (REFUTE-2 F11): it
  changed the image without motion. The render forward model now saturates the stretch
  (`gauss_cov_sat`), and the `render_ctrl_gauss`/`_first` rows above still stand on their
  silhouette metrics only.


### 2026-09-15 — batch j part 1: attribution of the window-locked cycle (code e234248)

20k flagship baseline, dt=1/240, dx=0.5, loss_res 64, pace 0, seed 1; triage under the
REFUTE-2 probe (lock band 6 %, tortuosity, both power conventions):

| change | chamfer | silIoU | speed period | locked | power (excl. / all bins) | modulation | tortuosity | visible |
|---|---|---|---|---|---|---|---|---|
| T = 10 | 0.1599 | 0.9510 | **10.00** | yes | 0.36 / 0.06 | 1.78 | 1.28 | 3.4 % |
| T = 20 (batch a) | 0.1599 | 0.9655 | 20.02 | yes | 0.95 / 0.89 | 4.12 | 2.80 | 9.6 % |
| T = 40 | 0.1675 | 0.9578 | **40.16** | yes | 0.97 / 0.94 | 3.81 | 2.54 | 100 % |
| `--assim 0` (no plastic reset) | 0.1588 | 0.9679 | 20.03 | yes | 0.93 / 0.80 | 4.59 | 2.87 | 46.8 % |

The period FOLLOWS the window length (10.00 / 20.02 / 40.16 substeps) — a fixed physical
time (the F-smoothing constant dt/(1−s) = 22.2 steps, the elastic period 132 steps and
its harmonics) cannot do that, and removing plastic assimilation leaves the cycle intact
(power 0.93, tortuosity 2.87). REFUTE-2 F1/F2 are therefore answered by measurement: the
window-locked driver is the per-window control re-optimisation under the terminal-only
objective, and the label `C_window` may be read as "control" for this pipeline.
Side result: T = 10 windows nearly suppress the cycle by themselves (power 0.36,
tortuosity 1.28, visible 3.4 %) at the same chamfer (0.1599) with silIoU −1.5 pt and
half the wall-clock — a horizon-side candidate to combine with `w_kin_var`.


### 2026-09-15 — batch j part 2: seed replicates, code-matched 40k pair, `--ppc 8` decoupled (code e234248)

All rows `render_full_dt_iso_nn`, dt=1/240, pace 0, anneal 0.7, mom_carry 0, nn_far_k 1000,
dfc_clip 0.02, 300-commit budget; commits = accepted records; triage under the REFUTE-2
probe (power = excl.-low-bins / all-bins conventions; tortuosity = path/net per window).
Recipe = `--warm_start --w_kin 5 --w_kin_var 50`. Every archive (43) was re-triaged with
`--json` under the current probe (REFUTE-2 F19).

**Seed replicates, 20k, dx 0.5, loss_res 64**

| arm | seed | commits | chamfer | silIoU | hole | power | tortuosity | visible |
|---|---|---|---|---|---|---|---|---|
| baseline | 1 (batch a) | 108 | 0.1599 | 0.9655 | 0.04% | 0.95 / 0.89 | 2.80 | 9.6 % |
| baseline | 2 | 128 | 0.1585 | 0.9664 | 0.08% | 0.95 / 0.87 | 2.57 | 11.8 % |
| baseline | 3 | 112 | 0.1589 | 0.9660 | 0.14% | 0.95 / 0.89 | 3.30 | 5.8 % |
| recipe | 1 (batch e) | 138 | 0.1584 | 0.9646 | 0.09% | 0.50 / 0.11 | — | 0.3 % |
| recipe | 2 | 125 | 0.1585 | 0.9651 | 0.09% | 0.44 / 0.05 | 1.27 | 2.2 % |
| recipe | 3 | 107 | 0.1583 | 0.9665 | 0.05% | 0.47 / 0.08 | 1.31 | 3.6 % |

Baseline seed spread: chamfer 0.1585–0.1599 (std 0.0007 = 0.45 %), silIoU 0.9655–0.9664
(0.09 pt). Recipe mean 0.1584 / 0.9654 vs baseline mean 0.1591 / 0.9660: a **tie within
seed noise** (REFUTE-2 F8 answered by measurement). The oscillation effect is seed-
invariant: power 0.95 → 0.44–0.50, tortuosity 2.6–3.3 → 1.27–1.31 on every seed.

**40k, code-matched (all under e234248)**

| arm | commits | chamfer | silIoU | hole | detFmin | G3 | G4_ej | power | tortuosity | visible |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline dx 0.5, loss_res 64 | 165 | 0.1298 | 0.9644 | 0.00% | 0.466 | PASS | FAIL | 0.97 / 0.92 | 3.44 | 2.4 % |
| recipe, dx 0.5 | 139 | 0.1297 | 0.9661 | 0.00% | — | PASS | — | 0.36 / 0.04 | 1.25 | 3.9 % |
| `--ppc 8` (dx 0.215, 149³ MPM), LEGACY units, loss_res 64 | 75 | 0.1294 | **0.9757** | 0.00% | 0.770 | PASS | FAIL | 0.94 / 0.87 | 2.04 | 17.5 % |
| `--ppc 8 --loss_units density` (loss grid 149³, reference-64 calibration: unit_ratio 1.29e4) + recipe | 127 | **0.1142** | 0.9735 | 0.52% | 0.755 | PASS | PASS | 0.83 / 0.51 | 1.90 | 3.3 % |

Reading:
- The recipe's shape numbers are a tie under matched code (0.1297 / 0.9661 vs 0.1298 /
  0.9644); its window-locked component drops 0.97 → 0.36 (tortuosity 3.44 → 1.25).
- **The `--ppc 8` decoupling (REFUTE-2 F13) separates two effects:** the finer MPM dx at
  fixed loss grid moves silIoU (+1.1 pt, 0.9757, the best 40k silhouette so far) at tie
  chamfer; the finer LOSS grid (149³, density units) moves chamfer (0.1142, −12 % vs the
  matched baseline) at +0.9 pt silIoU with the finer-dx hole level (0.5 %, gate 2 %). With
  the reference-64 calibration the ppc8/density arm's weights are 3.2× the batch-h/i
  meaning, as predicted, and `w_kin_var 50` now reaches power 0.83/0.51, tortuosity 1.9
  (still above 1.5 — the finer dx needs ~200 as batch i showed).
- G4_ejection (self-referential stray metric) fails on the dx-0.5 40k arms and the ppc8
  legacy arm, passes on the ppc8/density arm; it stays reported and unreplaced.

**Verdict and recommendation (2026-09-15, end of ladder).** Adopt for the next flagship
candidate, pending one more REFUTE pass on this section: `--ppc 8 --loss_units density
--warm_start --w_kin 5 --w_kin_var 200` at 40k (the discretisation contract with the
reference-grid calibration plus the window-cycle remedy at the weight batch i found
necessary at this dx). What is established with replicates: the window-locked control
cycle and its removal by `w_kin_var` (seed-invariant, T-tracking, assimilation-independent);
what is established once: the −12 % chamfer of the fine loss grid and the +1.1 pt silIoU
of the fine MPM grid. No CLI default has been changed in this session.

### 2026-09-15 — batches k/l: thin-feature transport ("scatter then return"; code 519617f)

Question raised by the user on the 40k GIFs: the ears fill with a spray that later
thickens, a few strays stay at the feet — volume, or something else? Dossier:
`docs/thin_feature_transport.md` (probe definition, literature, pre-registration §4, results
§5). Same discretisation as batch j (20k, dx 0.5, loss grid 64³, T 20, 300 commits).

- **Batch k (ablation of the cleanup pulls):** `w_nn 0`, `w_dt 0`, `w_creg 1000` leave the
  thin-region sparse PEAK at 0.58–0.61 (baseline 0.58) → the spray is not produced by the
  nn-band / W1 pulls or by control roughness; it is the core mass-matching descent acting on
  individual surface particles (Xu et al. TVCG 2025's own diagnosis of their loss) followed
  by numerical fracture (a particle beyond an empty grid cell has no neighbours to pull it
  back, Yue 2015). Volume is not involved (J p2p 0.008, isochoric assimilation, `w_jvol`
  active). The 20k/40k targets are the same mesh; the "joined ears" are the orthographic
  overlap at az 0.6.
- **Batch l (`w_coh` 3/30/100; 24³ basis + coh30):** `w_coh` falsified (peak 0.577–0.581,
  end 0.36–0.37 vs 0.38, chamfer 0.1591–0.1603 vs 0.1599). The coarse control basis is the
  lever: 24³ + coh30 delivered the full thin-target mass share (0.194/0.194), end sparsity
  0.226 vs 0.379, chamfer 0.1604, silIoU 0.9573 (−0.8 pt). Deconfounding arm (24³ alone,
  36³) = batch n.
- **Batch m (`w_bond` 10/100, `vol_frontier`, both, 40k `--ppc 8` + bond):** all falsified
  on the spray peak (0.580–0.586 vs 0.578); `w_bond 100` at 40k `--ppc 8` is harmful (chamfer
  0.1296 vs 0.1164, delivered thin mass 0.148 vs 0.188). `vol_frontier` shows the far-field
  D_vol is not the driver (peak 0.586 with the pull removed). Dossier §5b.
- **Support-gated APIC implemented** (Yao–Zhao 2026; `--gate_lo/--gate_hi`, `MPMParams.gate_*`,
  `kernels.k_cell_count/k_support_gate`, `tests/test_support_gate.py` incl. gated adjoint vs
  FD): opt-in forward-model remedy, batch p.
- **Batch n (24³ alone, 36³, 24³ + bond100, 24³ + frontier) + batch-c archives re-probed:**
  the spray PEAK is set by the kind of control, not its resolution — 0.50 for every basis
  from 6³ to 36³ (per-particle 0.58), unchanged by any add-on; END sparsity / delivered thin
  mass are monotone in basis resolution and saturate: 6³ 0.37/0.138, 12³ 0.28/0.180, 24³
  0.25–0.26/0.183–0.191 (two replicates), 36³ 0.24/0.195 (chamfer 0.1601, silIoU 0.9582).
  The add-ons on the basis (coh30 0.226, frontier 0.231, bond100 0.246, kin_var50 0.232) are
  within 2–3 noise bands of 24³ alone with no replicate — not adopted. Dossier §5c.
- **Gate-coverage probe (dossier §5d):** the vanguard's 3³-cell count is 0.31 (20k, 37 ppc)
  and 0.39–0.43 (40k `--ppc 8`) of the interior median — the sparse particles are NOT
  grid-isolated; the "numerical fracture" reading in the dossier §1 is struck out. The
  spray is a sub-loss-cell density deficit of a connected stream.
- **Batch p — both driver hypotheses falsified (dossier §5e):** support gate (0.05, 0.3) /
  (0.1, 0.6): peak 0.569/0.569 (baseline 0.578) with the vanguard's affine transfer halved
  (ω 0.43) — not an APIC artefact; chamfer 0.1593/0.1588, silIoU +0.1/+0.25 pt, harmless,
  stays opt-in. Physics-only (`--lambda_auto 0`, render OFF): peak 0.560 — the driver is the
  D_vol mass-matching descent itself; the render loss fills the ears more (0.165 vs 0.159)
  at a higher end sparsity (0.379 vs 0.324). Batch o2 (40k `--ppc 8` + 24³, legacy units)
  froze at commit 46 (chamfer 0.1337; peak 0.436, end 0.245 on the delivered slice).
- **Closing verdict:** not volume; the ears are filled first by the nearest particles at
  ~2× spacing because the loss cannot see spacing below its cell and a per-particle control
  can move single particles. Levers: control basis (peak 0.50, end 0.24 at 36³) and the
  discretisation contract (`--ppc 8`: peak 0.44–0.48, end strays 0.01–0.2 % = the mass ejection fix; density units: peak 0.36).
- **Batch o1 (40k `--ppc 8` + density + recipe + 24³ basis):** froze at commit 135 (outer-merit
  guard, reversal −0.55 on consecutive windows); chamfer 0.1175 / silIoU 0.9712 / strays
  0.22 % vs 0.1142 / 0.9735 / 0.07 % for the per-particle recipe; spray peak 0.424 → 0.319
  vs 0.444 → 0.329 (noise). The levers do not add at fine dx (dossier §5f). Flagship
  candidate unchanged: per-particle `--ppc 8 --loss_units density --warm_start --w_kin 5
  --w_kin_var 200`; the basis is the 20k lever.
- **Batch q1 (20k, density units, 128³ loss grid = 0.25 wu cell, per-particle):** peak
  0.493 → end 0.300, delivered 0.176 (> baseline 0.165), chamfer 0.1553 (−2.9 %), silIoU
  0.9634 — "what the loss can see" confirmed in direction (pre-registered < 0.36 not
  reached). Dossier §5g.
- **Batch q3 (20k, density/128 + 36³ basis):** peak 0.487 → end 0.311, delivered 0.177,
  chamfer 0.1558 — the combination does not add; both levers hit the same ~0.49 floor
  (they remove the same single-particle actuation).
- **Batch q2 (40k `--ppc 8` + density + recipe + 36³ basis):** froze at commit 119; chamfer
  0.1151, silIoU 0.9738, hole 0.00 %, spray 0.433 → 0.344, delivered 0.173, strays 0.11 % —
  within noise of the per-particle recipe (dossier §5f). Flagship candidate unchanged.
- **Ear–head "connection" (user report, 40k flagship frame 1746):** measured as projection —
  the wedge between the lower ear and the head is covered by the target's own az-0.6
  projection (the far ear edge-on); morph-only pixels are a ≤ 0.2 wu fringe (1 % of the
  target area), zero particles beyond 0.25 wu of the target surface, the notch preserved at
  az 1.4. The real residual is ear UNDER-fill (3–4 % of the projected ear area, 1.2 % of ear
  target points uncovered at 0.15 wu). Dossier §6; probes `scripts/probes/{web_probe,
  ear_views, cover_diff}.py`; gallery `40k_ppc8_loss_notch.gif` + `cover_diff` sheet.

### 2026-09-16 — target streak forensic (user report: "the thin line above the ear should not exist")

The user is right, and it is the TARGET, not the morph. `sampling/mesh._fill_centers` took
trimesh's axis **'base'** fill first; on the non-watertight bunny (Euler −3) that fill draws
1-voxel columns between unrelated surface voxels — 485 interior streak voxels in 5 clusters
of 28–67 voxels along one index axis (`scripts/probes`-style forensic, local), plus 4,870
base-only voxels of streaks and axis-aligned slabs. Sampled with replacement from the
256,499 centres, the streaks carry ~0.2 % of the particles: at 20k they look like scattered
points around the ear, at 40k (twice the points on the same columns) they read as a dotted
LINE from the ear tip to the head — the "upper thin line". Both the 20k and 40k targets came
from the same 110³ fill (the finer fallback never triggered), so the artefact was always
there; only its visibility scaled with n.

Fix (commit below): `'orthographic'` (a voxel counts as interior only if it is enclosed in all
three axis projections) is tried first — it has zero line-like interior voxels and 267,451
centres (+4.3 %: it also fills interior layers the base fill missed) — and any interior voxel
with ≤ 2 of 6 filled neighbours is stripped and counted (`STREAK_REPORT`). `filled_volume`
uses the same fill, so source/target volume matching stays consistent.
`tests/test_sampler_fill.py`: the bunny fill has no streaks, the base fill has > 100 (the
guard), the 20k/40k samples share one fill, and the filled volume equals the sampled one.

Consequences: every bunny result in this log up to 2026-09-15 was measured against a target
with ~0.2 % streak particles and a ~4 % smaller filled volume. Chamfer/silIoU shifts from the
target change are expected at the few-percent level and are re-measured below on the two
deliverable arms; the thin-feature findings (dossier §5–§6: peak set by control kind and
loss granularity, no over-mass, ear under-fill) do not rest on the streak voxels, but the
"upper ear as a sparse band" reading in §6 partly did — the band was the streak plus the
far ear, and is re-read after the rerun.

### 2026-09-16 — batch r: the deliverable arms on the FIXED target (code 0ccc43a, hyde06 GPU 0/2)

Same recipes and discretisation as batches a / n / j; only the target changed (orthographic
fill, no streaks, +4.3 % filled volume → the source is matched to it). "before" = the
streaky-target value from the 2026-09-15 log.

| arm | before: chamfer / silIoU / hole | FIXED target | spray peak → end, before | FIXED | delivered thin mass, before | FIXED | strays > 2 sp, before | FIXED | commits before | FIXED |
|---|---|---|---|---|---|---|---|---|---|---|
| 20k baseline `render_full_dt_iso_nn` | 0.1599 / 0.9655 / 0.04 % | 0.1568 / 0.9650 / 0.03 % | 0.578 → 0.379 | 0.608 → 0.361 | 0.165 / 0.194 | 0.157 / 0.191 | 1.27 % | 1.09 % | 108 | 87 |
| 20k `render_ctrl --control_grid 36` | 0.1601 / 0.9582 / 0.06 % | 0.1566 / 0.9594 / 0.07 % | 0.501 → 0.240 | 0.521 → 0.228 | 0.195 / 0.194 | 0.191 / 0.191 | 1.03 % | 1.04 % | (n) | 85 |
| 40k `--ppc 8` + density + recipe (flagship candidate) | 0.1142 / 0.9735 / 0.52 % | 0.1135 / 0.9541 / 0.00 % | 0.444 → 0.329 | 0.427 → 0.297 | 0.172 / 0.193 | 0.174 / 0.192 | 0.07 % | 0.06 % | 300 | 113 |

Reading: chamfer −2 % on both 20k arms (the streak particles were target points nothing
could match); silIoU, holes and strays unchanged within noise; the thin-feature picture is
the same on the clean target — per-particle control 0.61 → 0.36 with 0.157 delivered, the
36³ basis 0.52 → 0.23 with the full ear mass (0.191 / 0.191). The dossier's verdicts
(§5–§6) do not change. G3_rest fails on the fixed-target baseline as it did before
(drift; reported, not gated).

Ear–head check on the fixed-target 40k run (`cover_diff.py`, `ear_views.py`): particles beyond 0.15 wu of the target 0.015 % at the end frame; ear target points uncovered at 0.15 wu 0.74 %; projected morph-only pixels 0.73 % of the target area, target-only 4.80 % (az 0.6). The dossier §6 reading (no over-mass, ear under-fill) stands on the clean target; the 'sparse band' above the lower ear at az 0.6 was the far ear plus the streak, and with the streak gone it is the far ear alone.

Visual QA of the fixed-target 40k GIF (contact sheet, frames 0/110/330/774/1436/2209): both ears form as
lobes, no line, but 6 particles (0.015 %) leave the body downward from commit ~100 and stay visible
as dots under the bunny at the end — G4_ejection FAIL on this run (the streaky-target run had 0.07 %
strays > 2 sp but none this far). Reported, not gated; the far-stray census belongs to
`docs/floaters.md`.

### 2026-09-16 — speed profile and the allocation fix (hyde06 GPU 2, 40k `--ppc 8` density recipe, 12 commits, cProfile)

Where a window's time went (108 s in `run_pipeline`, 97.8 s in 12 `optimize_window` calls):

| item | time | share |
|---|---|---|
| `Trajectory.__init__` — 291 rollouts × 189 per-step arrays built from numpy zeros / identity (host → device copy, 57,252 `wp.array` constructions, `warp.context.copy` 30.4 s) | 43.5 s | 40 % |
| `warp_mpm_ext` forward + torch `run_backward` (MPM adjoint through the tape) | 17.4 + 16.5 s | 31 % |
| `.item()` syncs in `_norm` / `scalars` (line search bookkeeping) | 7.9 + 6.1 s | 13 % |
| PBR-lite shading channel `d_pbr` (5,184 shaded views) | 8.3 s | 8 % |
| silhouette loss `d_render` | 5.0 s | 5 % |
| one-off: `savez_compressed` 5.4 s, metrics 6.9 s (stray trajectory census 3.8 s) | | |

Fix (commit 5a3b8ee): the t > 0 arrays are allocated on the device (`wp.zeros`) and the
F-type lists are `wp.clone`d from a cached device identity — same values, no host traffic
(micro-benchmark: 15.6 → 5.2 ms per rollout's zero arrays, 3.4 → 1.4 ms for the identities).
Re-profiled with the same 12-commit run while the same batch shared the GPU: `run_pipeline`
108.3 → 78.7 s (−27 %), `optimize_window` 97.8 → 67.3 s (−31 %), the arm 1.8 → 1.3 min.
Results are bit-identical on the CPU and inside CUDA's atomic-add run-to-run noise on the
GPU (`tests/test_traj_alloc.py`: the SAME code differs run-to-run by 7e-9 on x_T and 5e-7 on
the dFc gradient; the change stays inside that band). Deployed mid-batch-h; later runs of
that batch use it.

Not done (candidates, in order of expected gain): (1) fewer `.item()` syncs — batch the
line-search scalars into one tensor and read once per iteration (~10 %); (2) the PBR-lite
channel evaluates 18 shaded views per loss call — half the views at the coarse phase; (3) a
CUDA-graph capture of the T-step rollout + adjoint (launch overhead is small at T = 20, so
the gain is uncertain); (4) the ejection census at the end (`metrics.ejection_trajectory`)
walks every frame — sample every 4th.

- Pass 3 (commit c542572): persistent per-window no-grad `Trajectory` rolled out as a CUDA
  graph (`Trajectory.capture/run`; the control is copied into a persistent buffer its dFc
  sequence views; grid accumulators re-zeroed per step) replaces the per-candidate
  construction (~200 allocations + ~250 Python launches at 150k); plastic assimilation in
  torch on the GPU (same maths, float32; test vs numpy 2e-4); the remaining numpy dets
  batched; `PHYSMORPH_TIMING=1` logs a per-window eval/terms/grad/final breakdown with
  device syncs. Tests: persistent rollouts bit-identical to fresh ones on CPU, allclose on
  CUDA (atomics). Result: see the `graph` profile below.
- Pass 3 timing (clean, no cProfile, 150k `--domain auto`, 6 windows, commit 7fab3cf):
  3.0–4.0 s/window = eval rollouts 14 ms × 37–40 + losses 10 ms × 37–40 + det check 4 ms ×
  37–40 + tape forward 43 ms × 8 + gradient section 236 ms × 8 (three adjoint passes per
  iteration with PCGrad, plus the render backward) + commit rollout 0.1 s + 0.4 s Python.
  Two structural wastes found: (a) 3 of 8 iterations per window ended in an exhausted line
  search (10 rollouts each = 30 of the 37 rollouts) and the next iteration re-tested
  already-rejected step sizes from an unchanged point — the window now ends at the first
  exhausted search (commit 37e6629; no change to any accepted step); (b) the tape rollout
  rebuilt a Trajectory (~500 allocations with grads) and launched ~120 forward + ~120 adjoint
  kernels from Python per pass — now `function.PersistentAdjoint` (commit 07c4ff2): one tape
  trajectory per window, forward and zero+adjoint as CUDA graphs, seeds through persistent
  buffers (bit-identical to the plain bridge on CPU).
- Pass 4 timing (commit 9ff24a9, clean, 150k `--domain auto`): 2.3–2.9 s/window; 17–22
  rollouts (28 ms each); gradient section 1.25–1.67 s = three adjoint passes per iteration
  at 60–70 ms each (`adj_bench.py`: forward graph replay 19 ms, adjoint replay 75 ms, no-grad
  graph rollout 24 ms vs 32 ms from Python). The adjoint kernels are now the floor: 300
  windows ≈ 12–14 min at 150k. Below 10 min needs hand-written adjoints of P2G/G2P (the
  automatic adjoint of the atomic scatter is ~4× the forward) — not done in this pass.
- `--domain auto` calibration fix (commit 6cf1a23): the density-unit reference cell now
  reaches the runner (0.5 wu on any box); the auto domain's speed gain stands, the
  earlier auto-domain results do not (see the ejection ladder for the root cause).

### 2026-09-16 — high-resolution gallery (batch h): 10 targets × {render, physics-only}

User request: judge what remains (mass ejection, high-res + diverse examples, speed), then a
document with ~10 high-res examples, PBR stills, gradient reach (% of surface, magnitudes,
heatmaps), loss curves + wall-clock, the physics-only comparison, hi-res GIFs; every result
viewable in the 3D viewer later.

Discretisation (all runs): 40k particles, `--ppc 8` (dx 0.2148, MPM 149³, loss grid 149³ in
density units with the reference-64 calibration), T 20, dt 1/240, `--warm_start --w_kin 5
--w_kin_var 200`, 300-commit budget with the patience freeze, `--pace 0 --anneal 0.7
--mom_carry 0 --nn_far_k 1000`, source isosphere, target volume matched to the source, FIXED
sampler (0ccc43a). Arms: `render_full_dt_iso_nn` (λ_auto 0.5) and the same with
`--lambda_auto 0` (physics-only, same code path). Targets (asset survey: 14 of 15 meshes fill
with the orthographic voxeliser; `car.obj` is a surface shell and is refused): bunny,
armadillo, dragon, spot, bob, teapot, heart, A, C, V — GPU 0 takes the first five, GPU 2 the
rest, render then physics-only per target; the speed fix 5a3b8ee was deployed after the first
two render runs started, so later runs are faster (s/commit is reported per run).

Measured per run (raw state): chamfer, silIoU, hole, gates (G4_ejection), delivered commits,
wall-clock and s/commit (live-packet mtimes), the window loss and D_vol curves, thin-feature
sparse peak → end and delivered thin mass (`scatter_probe2`), strays > 2 sp. Gradient probe
(`scripts/probes/grad_field.py`) at commits 5 % / 35 % / delivered of the render run and the
delivered frame of the physics-only run: share of all / surface / interior particles with
|∂D_render/∂x|, |∂D_vol/∂x|, |∂D_render/∂dFc| (through the MPM adjoint over one window from
the archived state with v = C = 0) above 1e-3 of the max, magnitude means, reach by depth
decile, heatmaps. Display: hi-res GIFs (`make_gif --res 320`), PBR stills
(`scripts/render_pbr.py`: surface splatting + GGX; the 3DGS photoreal path smears solids —
docs/floaters.md 2026-09-04 — so this is screen-space surface reconstruction, raw state only).

Pre-registered expectations: (i) render vs physics-only: silIoU +0.5–1 pt and more thin-feature
mass for render on every target, chamfer within ±3 % (the 20k finding); (ii) the image
gradient on x touches 5–10 % of particles (15–25 % of the surface set) and, through the
adjoint, 30–55 % of all particles; (iii) G4_ejection fails on some targets with a handful of
far particles — the count per target decides the next ejection ladder; (iv) s/commit drops by
~25 % on the runs started after 5a3b8ee. Results: `docs/highres_report.md` (written from
`output/report/`, the artifact page is the deliverable).

**Batch h results (2026-09-16 09:10 CDT, 20 runs, 07:20–09:08).** Full tables: `docs/highres_report.md`; page with
GIFs / PBR stills / heatmaps: https://claude.ai/code/artifact/3a1a66fe-6100-4ccf-9e54-21c02a1d0c8a.

| target | render: chamfer / silIoU / hole / commits / min | phys: chamfer / silIoU / hole / commits / min | thin mass r / p (tgt) | strays > 0.5 wu r / p |
|---|---|---|---|---|
| bunny | 0.1142 / 0.9570 / 0.02 % / 115 / 19.1 | 0.1145 / 0.9528 / 0.01 % / 129 / 7.6 | 0.172 / 0.163 (0.192) | 5 / 0 |
| teapot | 0.1111 / 0.9743 / 0.00 % / 112 / 18.2 | 0.1111 / 0.9624 / 0.00 % / 108 / 6.6 | 0.222 / 0.218 (0.212) | 0 / 0 |
| armadillo | 0.1159 / 0.9356 / 0.28 % / 79 / 8.6 | 0.1172 / 0.8992 / 0.36 % / 74 / 4.3 | 0.152 / 0.137 (0.205) | 13 / 15 |
| heart | 0.1112 / 0.9819 / 0.00 % / 117 / 12.6 | 0.1111 / 0.9755 / 0.00 % / 118 / 7.5 | 0.197 / 0.196 (0.193) | 0 / 0 |
| dragon | 0.1325 / 0.8078 / 1.00 % / 139 / 14.7 | 0.1296 / 0.7886 / 0.24 % / 90 / 5.5 | 0.134 / 0.120 (0.176) | 274 / 177 |
| A | 0.1133 / 0.9768 / 0.00 % / 107 / 11.4 | 0.1133 / 0.9599 / 0.00 % / 143 / 8.8 | 0.157 / 0.150 (0.172) | 1 / 0 |
| C (ring) | 0.5515 / 0.7656 / 0.02 % / 20 / 2.3 | 0.4918 / 0.6677 / 0.22 % / 16 / 1.2 | 0.085 / 0.083 (0.155) | 40 % / 26 % |
| spot | 0.1136 / 0.9768 / 0.00 % / 74 / 7.8 | 0.1150 / 0.9574 / 0.00 % / 114 / 7.2 | 0.158 / 0.139 (0.177) | 0 / 0 |
| V | 0.1248 / 0.8817 / 0.23 % | 0.1147 / 0.9322 / 0.01 % | 0.163 / 0.191 (0.185) | 123 / 42 |
| bob (ring) | 0.1496 / 0.8011 / 2.06 % | 0.1220 / 0.8654 / 2.57 % | 0.195 / 0.195 (0.171) | 580 / 127 |

Verdicts against the pre-registration:
- (i) render vs physics-only: CONFIRMED on 6/10 (teapot, heart, spot, bunny, A, armadillo):
  silIoU +0.4 to +3.6 pt, thin mass +0.001–0.019, chamfer within ±1 %. FALSIFIED on the
  two targets that need a hole to open or a sharp re-entrant corner (bob ring, V): the render
  arm is worse (bob 0.1496/0.80 vs 0.1220/0.87; V 0.1248/0.88 vs 0.1147/0.93) and ejects 3–5×
  more particles. dragon: both arms silIoU 0.79–0.81. C (ring): both arms stall at commit 9 —
  the outer-merit guard rejects every candidate window (gain −0.08, reversal 0.96) and the
  hole never opens; a topology-change limit of the mass-matching descent, not of the render
  channel.
- (ii) gradient reach: CONFIRMED — |∂D_render/∂x| active on 6–9 % of particles (15–25 % of
  the surface set, ≤ 2.5 % interior; surface/interior magnitude 30–80×); D_vol on 100 %
  (surface/interior 1.0–2.2×); through the adjoint the image gradient reaches 50–90 % of all
  particles and 47–90 % of the interior (heart 90 %, teapot 56 %, bunny 52 %, armadillo 50 %).
- (iii) ejection: target-driven — 0 on the smooth targets in both arms, 13–15 on the
  armadillo in both, 177–274 on the dragon in both, and the render channel adds single
  digits on bunny/A but multiplies it 3–5× on bob/V. Ejected particles are already out at the
  midpoint (100 %) — they do not return. Next ladder: `v_max` (needs a CLI flag), near-band
  re-coupling weighted to the ejected set, a λ ramp in the first windows, and a `w_spray 0`
  arm on bob/V to isolate the silhouette excess term.
- (iv) speed: CONFIRMED — render runs started before 5a3b8ee 9.8–10.0 s/commit, after
  6.3–6.7 s/commit (−34 %); physics-only 3.5–3.8 s/commit, so the render channel costs
  ~2.8 s per commit at 40k / 18 views.

Display: `scripts/render_pbr.py` (surface splatting + GGX) replaces the 3DGS photoreal path
for solids in the report; `scripts/probes/stray_census.py` is the ejection census. The
`--live_dir` packets of all 20 runs are served by `viewer_serve.py` on hyde06
(`/runs` lists them; `/compare` pairs render vs phys).

### 2026-09-16 — mass-ejection ladder (algorithmic fix), 40k dragon / armadillo / bob

User request: solve mass ejection completely, algorithmically (no state editing), then all 10
targets at ≥ 150k particles with surface videos. Measured first (batch-h archives, dt 1/240,
speeds from consecutive archived steps): the body's p95 speed is 0.25–0.28 wu/s over the run
(max 2.6–2.8 in the first windows); the particles that end > 0.5 wu from the target run at
5–6 wu/s and are above 2× the body p95 on 90 % of the frames; they leave in windows 2–3
(archived frame ~45) and never return. Ejection is a small set of particles at 2–20× the
body speed, launched during the fast initial descent.

Mechanisms (all opt-in; commits e72728f, 4816d61, 03cbed4):
- `eject_veto` (runner, outer-merit stage): a candidate window that INCREASES the number of
  isolated particles (nearest neighbour > `eject_iso_k` target spacings) is rejected like a
  brake reject (step shrink + cold restart; patience charged only on a replay). Hard
  guarantee on accepted commits: the isolated count is monotone non-increasing.
- `w_esc` (optimizer): hinge on the window-end velocity relative to the frozen source
  neighbours, `mean relu(|v_i − mean_j v_j| − esc_k·median|v|)² / thr²`, through the MPM
  adjoint (steers the descent away from launching single particles).
- `--v_max` (forward model, existed in MPMParams, now on the CLI): G2P speed cap.

Ladder (render arm, 40k `--ppc 8` density recipe, targets with the worst batch-h ejection:
dragon 274, bob 580, armadillo 13 far particles):
- k = 3 (veto + hinge): FAILED — every early window rejected ("EJECTION 3→71" from anim 2):
  the sphere's surface dilutes to ~3 spacings legitimately during the initial descent; runs
  froze at anim 14 (chamfer 0.75–0.81). Default raised to 6.
- k = 6 (veto + hinge, no cap): FAILED — dragon rejects "0→1" from anim 2 and replays the same
  candidate after the cold restart (frozen at anim 16, chamfer 0.57); armadillo 15 rejects by
  commit 20. A single launched particle per window is enough to veto, and the line search
  cannot find a non-launching step by shrinking alone: the launch happens inside the window,
  the veto only sees it afterwards.
- v_max 3 wu/s + veto k 6 + hinge: FAILED — dragon froze at 2.0 min with 17 % of particles
  > 0.5 wu from the target and 15 veto rejects: the isolation veto fires on legitimate
  stretching (neighbours moving away), the cold restart replays the same candidate, the
  patience freeze follows. **User ruling at this point: no parameter-only fixes** (caps,
  thresholds, weights) — the mechanism must remove the defect by construction.
- **Discrete-continuity line-search feasibility** (`--continuity`, method.md eq. 23, commit
  29d2704): a step is accepted only if no particle's window-end velocity relative to its
  frozen material neighbours exceeds one local spacing per window, sp_i/(T·dt), or the
  iteration's reference. Scale = discretisation; enforced by rejection + α halving inside the
  line search, so no accepted commit can launch a particle. Pre-registered on dragon / bob /
  armadillo (render arm, 40k): 0 far particles at the end, chamfer within ±3 % of batch h,
  silIoU within ±1 pt, line-search exhaustion no more frequent than batch h. Falsifier: any
  far particle at the end, or a freeze before commit 60.
- Continuity v1 (reference = warm-started rollout): dragon 170 far (was 274), chamfer 0.1276,
  silIoU 0.837 — reduced, not solved; armadillo 15 (was 13).
- **Continuity v2 FALSIFIED** (reference = free rollout, max over steps): armadillo 12 far,
  bob 717 (was 580). Probe on the archives: the far particles are not a clump (7–10 % of
  their source neighbours are also far), their velocity RELATIVE to their material
  neighbours is 0.5–1.1 wu/s (limit 1.3) while the absolute speed is 2–5 wu/s — the
  neighbours move too; the particle then keeps going after the neighbourhood stops, having
  left every other particle's stencil. Numerical fracture, not a launch: a rule on the
  per-window relative velocity cannot see a slow, sustained drift of a decoupled particle.
- **Material bonds for decoupled particles** (`--bonds`, method.md eq. 24, commit below):
  one-sided tension bonds to the frozen source neighbours, rest length re-based at the
  window start, stiffness (6/K)(λ+2μ) r, acting only on particles with no other particle in
  their 3³ cells; force into P2G momentum with the reaction on the neighbour. Bit-identical
  when nothing is decoupled; adjoint checked. Pre-registered on dragon / bob / armadillo:
  0 far particles at the end, chamfer within ±3 % of batch h, silIoU within ±1 pt.
  Falsifier: any far particle at the end (a decoupled particle the bonds could not hold).
- Explicit bond spring FAILED (unstable): dragon froze at anim 23 with 9 % of particles
  > 0.5 wu from the target — a linear spring with a multi-wu extension integrated explicitly
  explodes (ω Δt ≈ 1.3, extension 30× the rest length). Replaced by **material re-coupling**
  (same flag `--bonds`, method.md eq. 24 rewritten): material-PIC velocity in P2G and a
  position projection toward the rest lengths in the advection step, for decoupled particles
  only. Bit-identical when coupled; adjoint checked; pre-registration as above.
- Re-coupling v2 FALSIFIED: dragon 735 far (was 274), chamfer 0.1596. Census on the archive:
  ~125 particles become decoupled at frame ~100 in both runs and never return, because the
  rest lengths were re-based at every window start — the separation was accepted.
- Re-coupling v3 (rest lengths carried as state, refreshed only for coupled particles;
  projection at 1/T per step) FALSIFIED: dragon 772 far, chamfer 0.1666; armadillo 46 far,
  silIoU 0.885, froze at commit 46. Diagnosis: the per-particle test "no other particle in
  my 3³ cells" flags legitimate thin-feature TIPS (dragon spines, armadillo claws are
  single-particle cells at 40k) and misses the actual ejecta, which leave in groups of 2–3
  particles 0.4–0.5 wu apart — they share grid nodes with each other, not with the body.
- **Re-coupling v4 — fragments** (commit 038d043): decoupled ⇔ the particle's occupied grid
  cell lies in a connected component (26-connectivity) of occupied cells other than the
  largest one (the body). `runner.fragment_mask` (scipy.ndimage.label, once per window, a
  few ms) → `bond_frag`; the kernels apply the material-PIC velocity and the bond projection
  to fragment particles only; rest lengths are refreshed for body particles and frozen for
  fragments (they are pulled back INTO the body). Thin features stay connected through
  occupied cells and are never touched. No thresholds. Pre-registration: 0 far particles at
  the end on dragon / bob / armadillo, chamfer and silIoU within noise of batch h.
- Re-coupling v4 FALSIFIED on dragon: chamfer 0.1762, silIoU 0.784, 886 far; the mask flagged
  732 particles from commit ~47 on — a whole dragon spine whose occupancy has one-cell gaps at
  dx 0.215, so the raw-occupancy components split legitimate thin material and the projection
  yanked it toward stale rest lengths.
- **Re-coupling v5 — stencil connectivity** (commit c6f5bbf): the components are taken on the
  occupancy DILATED by one cell (two particles couple through shared grid nodes when their
  cells are within the B-spline support, so a one-cell gap is still one body). Debris more
  than two cells away from the body is still a fragment. Test: a thin feature with a one-cell
  occupancy gap is not flagged. Pre-registration unchanged (0 far, quality within noise).
- **Implementation bug found 2026-09-16 12:50 (commit c542572)**: the no-grad rollouts —
  every line-search candidate, the warm-start comparison and the COMMIT rollout — built
  their Trajectory without the bonds; only the tape rollout carried them. So v2–v5 were
  verdicts on a broken implementation: the loss was evaluated on plain physics, the
  gradient described the re-coupled system, and the committed frames never contained the
  projection (which is also why the quality dropped: gradient/loss mismatch). Fixed by
  construction — one persistent no-grad Trajectory per window (speed pass) is built from
  the same RolloutSpec as the tape rollout, bonds included. v5 relaunched with the fix.
- **v5 with the fix, dragon** (commit c542572): chamfer 0.1225, silIoU 0.893 (better than the
  plain run's 0.1325 / 0.86 — the projection now acts on the committed trajectory), but
  G4_ejection still FAIL: 121 particles > 0.5 wu at the end (was 274), 82 flagged fragments.
  `scripts/probes/fragment_trace.py` on the archive (every step archived): the end
  fragments' mean distance to their 8 source neighbours grows SMOOTHLY from 1.0× rest at
  frame 0 to 2× by frame 80, 10× by frame 420 and 17× at the end — a continuous drift of
  ~2 % per window from the very first windows, not a launch; they are first flagged at a
  median frame of 763 (ratio ≈ 10); after the flag the ratio stays at 1.000 / 1.004 (+1 /
  +20 frames), i.e. the projection cancels only the one-window excess over the frozen rest
  lengths, because the rest lengths were re-based every window while the particle was
  still "coupled" — the slow separation was accepted window by window. 0 of 82 ever came
  back. Conclusion: a rest length measured from the current configuration cannot separate
  legitimate plastic flow (neighbour distances grow 2–5× in a sphere→dragon morph) from
  ejection (17×); the grid-connectivity flag fires only after the material is already two
  cells away. Both the detection and the return are the wrong primitives for this defect.
  Armadillo v5: chamfer 0.1143, silIoU 0.939, 12 far (max 1.8 wu), 6 fragments at the end;
  bob v5: chamfer 0.1425, silIoU 0.822, hole 2.3 %, 507 far (max 4.7 wu) — bob is the worst
  ejector of the three under per-particle control.
- **Control basis — the by-construction candidate** (census on the existing 40k bunny
  archives, `stray_census.py`): per-particle control (`render_full_dt_iso_nn`, batch h
  recipe) ends with 8 particles > 0.5 wu (max 3.8–4.3 wu, all out by mid-run); the SAME
  recipe on the 24³ and 36³ control basis (`render_ctrl --control_grid 24/36`, batches o1/q)
  ends with 0 particles > 0.25 wu (max 0.15–0.17 wu). Mechanism: a control field on a coarse
  node grid cannot vary inside a B-spline stencil, so no particle can be pushed differently
  from its material neighbours — the sub-cell differential push that drives the slow drift
  does not exist. No threshold, no detector, no return force. Pre-registration for the
  trio (`ejb_*`, 40k `--ppc 8`, recipe, 24³, `--domain auto`, no bonds): 0 particles
  > 0.5 wu at the end on dragon / bob / armadillo (per-particle: 274 / ? / ?); chamfer within
  5 % of the per-particle runs (bunny o1: 0.1175 vs 0.1142). Falsifier: any far particle.
- **Basis FALSIFIED on the auto domain** (13:20): `ejb_armadilo` (render_ctrl, 24 nodes on
  the 14 wu auto box = 0.58 wu = 2.7 dx, tknots 4): chamfer 0.1258, 98 far (max 5.7 wu),
  froze at 77; `ejb_dragon`: chamfer 0.312, 2594 far (6.5 %); `ejc_armadilo` (flagship arm
  + `--control_grid 24`, per-step knots): chamfer 0.147, silIoU 0.755, 579 far. All worse
  than per-particle control (armadillo 12 far). The fixed-domain bunny runs had a 4–6 dx
  node spacing; at 2.7 dx the basis does not remove the sub-cell push and its coarser
  optimisation freezes early. Not pursued further: a spacing would be a tuned parameter.
- **Where the ejecta come from** (dragon v5 archive): 81 % of the end-far particles start in
  the outer 10 % of the source sphere's radius (17 % of all particles do); their source
  cells hold 6 particles vs 8 — they are the source SURFACE layer. Mechanism (from the
  P2G/G2P algebra): a particle's own control stress acts on its stencil nodes with force
  f_i = −V P_c ∇w_i and comes back as Σ_i w_i f_i / m_i; for an interior particle the node
  masses are uniform and Σ_i w_i ∇w_i ≈ 0 cancels the self-term, for a surface particle the
  outward nodes carry only its own mass and the self-term does not cancel — a surface
  particle can propel itself outward with its own control stress, and every commit's
  plastic assimilation (η = 0.5) forgives half the elastic stretch that would pull it back.
  That is the 2 %/window drift the fragment trace measured.
  Generalises: armadillo v5 — 100 % of the 12 far particles start in the outer 10 % of the
  source radius; bob v5 — 45 % of 125 (all particles: 17 %). And the separation is EARLY:
  the end-far set's neighbour-distance ratio is 6 (armadillo) / 4.6 (bob) at 1/12 of the
  run, 11–14 by 1/4, then flat to the end — they detach during the initial expansion of the
  sphere toward the target's extent (v_absmax 2.5–5.5 wu/s in that phase) and never move
  relative to their old neighbours again.
- **Neighbourhood-consensus plastic assimilation** (commit after 9ff24a9, `--assim_consensus`):
  plasticity is a continuum property — the plastic increment at a commit follows the
  elastic stretch of the particle's STENCIL NEIGHBOURHOOD with its own contribution removed
  (`plasticity.consensus_elastic`: the P2G/G2P cubic-B-spline transfer of F_e, self term
  subtracted at every node, nodes carrying no other mass do not vote, a particle with no
  voting node keeps its own F_e). A particle stretching with its neighbours is assimilated
  exactly as before; a particle stretching AWAY from neighbours that are not stretching keeps
  the excess as elastic strain, whose restoring stress grows every window until it pulls
  the particle back — and the adjoint sees that cost, so lone-particle pushes stop paying.
  No threshold, no detector, no return force: only WHICH strain is forgiven changes. Tests:
  uniform stretch identical to per-particle; a lone 3× stretched particle keeps F_e ≈ 3
  (per-particle would forgive half); an isolated particle uses its own strain.
  Pre-registration (trio `ejd_*`, flagship arm, per-particle control, no bonds, auto
  domain): 0 particles > 0.5 wu at the end on dragon / bob / armadillo (per-particle v5:
  121 / 507 / 12); chamfer within 5 % of v5 (0.1225 / 0.1425 / 0.1143). Falsifier: any far
  particle, or chamfer worse by > 5 %.
- **Consensus assimilation FALSIFIED** (trio `ejd_*`, 13:22): dragon chamfer 0.253, silIoU
  0.577, detFmin 0.26, 2475 far (6.2 %), stopped at 172; armadillo 0.145 / 0.631, 685 far;
  bob froze at 15 commits with 7509 far (19 %). Retaining the excess as elastic strain does
  not pull surface particles back — it makes neighbouring particles' plastic states
  inconsistent (a surface particle has few voting nodes) and the accumulated elastic
  mismatch tears the body apart in chunks. Worse than every earlier attempt.
- **What the control actually does to the ejecta** (dragon v5 history): `dfc_absmax` is
  0.02 in every window — the per-particle control sits at `dfc_clip` — and the measured
  drift is ~2 % of the neighbour distance per window: the ejecta are surface particles the
  rasterised gradient pushes outward at the clip, window after window, toward target regions
  the body never fills (the physics-only twin ejects the same way). The self-propulsion
  term is negligible at this magnitude (`selfprop_probe.py`: one particle at dFc = −0.3 I
  moves 1e-4 wu in a window); it is the DIFFERENTIAL push between material neighbours that
  the per-particle gradient produces at cell granularity.
- **Sobolev (H1) descent direction** (commit after c059934, `--grad_h1`): the total control
  gradient (after λ / PCGrad / W1) is replaced by the converged solution of
  (I + κ (I − A)) u = g on the frozen material kNN graph (A = neighbour mean, κ = 2 as the
  existing render-only `control_h1`, Jacobi to a 1e-4 relative change, norm preserved).
  The descent direction cannot differ between material neighbours at sub-stencil scale, so
  a lone surface particle cannot be pushed away from its neighbourhood — the whole
  neighbourhood moves or nothing does. Standard Sobolev gradient descent (shape
  optimisation), no threshold, no detector. Pre-registration (`eje_dragon`, flagship,
  per-particle, no bonds): 0 far particles at the end (v5: 121), chamfer within 5 % of v5
  (0.1225). Falsifier: any far particle, or chamfer > 0.129.

### 2026-09-16 — 150k gallery v2 (batch `h150_*`, launched 13:50)

Configuration: the v5 recipe with the no-grad bonds fix — `render_full_dt_iso_nn`, 150k
`--ppc 8`, density units, warm start, `--w_kin 5 --w_kin_var 200`, 300 animations,
`--loss_res 64 --pace 0 --anneal 0.7 --mom_carry 0 --nn_far_k 1000`, `--bonds` (material
re-coupling with the dilated-occupancy fragment mask), `--domain auto`, `--archive_stride 8`,
live packets for the viewer (`live/h150_<T>_render_full_dt_iso_nn`). One run per GPU:
k1 = bunny armadilo dragon spot bob (GPU 0), k2 = teapot heart A C V (GPU 2). Code
c059934+ (persistent no-grad + tape graphs, line-search break, batched dets, torch
assimilation). Pre-registration: ≤ 14 min per target unshared (2.3–2.9 s/window × 300);
chamfer ≤ the 40k batch-h values; ejection reported by the census, not gated away.
Post-processing per target (`hr150_post.sh` via `hr150_watch.sh`): surface video (GPU
z-buffer splats, two azimuths, target outline), particle GIF, PBR stills (delivered +
target, az 35/215), scatter probe, stray census, loss curves → `report150/<T>/`.
- Results as they land: teapot — 77 commits (outer-merit patience stop), 7.6 min, chamfer
  0.0931, silIoU 0.888, hole 0 %, 81 fragments (0.05 %); heart — FROZE at 18 commits
  (1.9 min): from anim 6 every candidate was a brake reject (gain −0.052 < −0.05,
  reversal +0.90, replay of the same candidate after the cold restart), chamfer 0.150,
  move_cv 0.17. At 40k (batch h) heart ran 123 commits with rejects only at the end, so
  this is a 150k/auto-domain-specific plateau at the very start; diagnostics
  `dbg_heart_auto` / `dbg_heart_fixed` (40k, 60 anims, current code) and `dbg_dragon_v5`
  (v5 config on the current code — regression check for the ejb/ejc/ejd/eje verdicts).
  Bunny and A show no outer-merit rejects (129 / 42 commits at 14:05).
  Diagnosis of the heart freeze (history json): the outer merit is Σ component/scale with
  the scales fixed at the first commit; heart's `d_dt` (isolation-gated distance-transform
  term) GROWS 8308 → 10704 (+29 %) over the first six windows while d_vol (0.107 → 0.052)
  and d_sil (0.142 → 0.101) fall — the expanding body crosses outside the target's DT
  band before it takes the heart's shape — so the merit rises 5.2 %/window, the brake
  fires (gain < −0.05), the cold restart replays the same candidate and patience ends the
  run. The window objective (which weights the same terms with their optimiser weights)
  still decreased every window (0.138 → 0.0845). 40k heart with the current code
  (`dbg_heart_auto/fixed`, 60 anims) shows no rejects and reproduces batch h's first
  window to 4 digits, so this is not a code regression but the gate's first-window
  normalisation meeting a 150k transient.
  Side finding from the same pair (40k heart, 60 anims, current code): `--domain fixed`
  chamfer 0.1124 / silIoU 0.983 / 0 rejects vs `--domain auto` 0.1254 / 0.892 / 3 rejects.
  The auto domain changes the density-unit calibration cell (`unit_ref_res` follows the
  box) and therefore the effective weights — a quality confound of the speed change that
  must be checked target by target before the auto domain is kept (heart2 at 150k runs
  with `--no_outer_merit`; the fixed-vs-auto question is open).
- **Root cause found (14:15)**: the auto-domain block set `args.unit_ref_res` (a 0.5 wu
  reference cell) but `cfg.unit_ref_res` never received it, so the density-unit
  calibration ran its legacy cell sum on a 64³ grid over the SMALLER auto box — a 0.2 wu
  reference cell — and every converted weight was ~3× off: dragon 40k ratio 4.36e4 (auto)
  vs 1.45e4 (fixed). Every run launched with `--domain auto` (ejb, ejc, ejd, eje, the 150k
  batch, the heart diagnostics) carried this. Fixed in `pipeline_run.py` (cfg receives the
  reference cell); the affected mechanism verdicts (basis, consensus assimilation, Sobolev
  direction) are VOID and must be re-run; the v5 trio (fixed domain) stands.
- Code-regression check (14:20): `dbg_dragon_v5fixed` — the v5 configuration on the FIXED
  domain with the current code (persistent trajectories, tape graphs, line-search break,
  shared grid, torch assimilation): chamfer 0.1224, silIoU 0.887, 62 fragments, 204
  commits, 6.8 min vs v5's 0.1225 / 0.893 / 82 / 141 / 9.0 min. No regression from the
  speed work; the catastrophes were the calibration bug alone.
- **Corrected auto domain, v5 reference (`ejf2_dragon`, 14:26)**: chamfer 0.1199, silIoU
  0.901, hole 0.34 %, 83 far (0.21 %, max 2.5 wu), 24 fragments, 251 commits, 8.5 min — on
  par with or slightly better than the fixed-domain v5 (0.1225 / 0.893 / 121 far). The auto
  domain with the 0.5 wu calibration cell is a valid speed lever; the mechanism re-tests
  (`eje2` Sobolev, `ejd2` consensus, `ejc2` basis 24) run against this reference.
- **Sobolev direction re-test FALSIFIED** (`eje2_dragon`, corrected domain): chamfer
  0.1240, silIoU 0.882, 97 far (max 5.2 wu), froze at 100 commits (12 rejects) vs the
  reference 0.1199 / 0.901 / 83 far / 251 commits. Smoothing the descent direction over
  the material graph neither reduces the far set nor keeps quality — the differential
  push the ejecta receive is not a sub-stencil gradient artefact that a κ = 2 graph
  smoothing removes.
- **Consensus assimilation re-test FALSIFIED** (`ejd2_dragon`): chamfer 0.1294, silIoU
  0.824, 219 far (max 4.6 wu), froze at 79 commits — worse than the reference on every
  count (the earlier catastrophe was the calibration bug; the mechanism itself still
  loses: retained elastic mismatch between neighbours degrades the body).
- **Control basis re-test FALSIFIED for ejection** (`ejc2_dragon`, flagship + 24³ basis):
  chamfer 0.1324, silIoU 0.931, 192 far (max 3.6 wu), froze at 66 commits. Higher
  silhouette IoU but more than twice the far particles of per-particle control; the
  basis moves chunks coherently and detaches them.
- Ladder verdict (corrected domain): per-particle control + material re-coupling v5 with
  the no-grad bonds fix is the best configuration measured — dragon 83 far (0.21 %),
  chamfer 0.1199; bunny 150k 98 far (0.065 %), chamfer 0.0765 — and ejection is NOT
  solved. The by-construction direction the literature points to is a permanent
  reference connectivity (total-Lagrangian MPM: shape functions on the undeformed grid,
  so neighbours can never lose each other — de Vaucorbeil et al.; "Simulating Brittle
  Fracture with Material Points" constrains particle domains to the cell size), at the
  price of forbidding the topology changes (holes in A, the teapot handle) that the
  updated-Lagrangian formulation creates by the same numerical fracture — a hybrid with
  a discretisation-defined bond range is the open design.

- **150k batch v2, corrected domain** (14:27 →): bunny — 101 commits (10 rejects), 11.7 min,
  chamfer 0.0765, silIoU 0.920, hole 0.09 %, 47 fragments, 98 far (0.065 %, max 1.8 wu).
  teapot — 176 commits, 20.2 min, chamfer 0.0716, silIoU 0.969, hole 0 %, 1 fragment,
  stray 0.115 % (the void auto-calibration run: 0.093 / 0.888 / 81 fragments).
  armadillo — 190 commits (23 rejects/nulls), 19.0 min, chamfer 0.0814, silIoU 0.794, hole
  0.35 %, 86 fragments, stray 1.24 %. heart (corrected) — 101 commits, 11.1 min, chamfer
  0.0728, silIoU 0.983, hole 0 %, 0 fragments, stray 0.035 %, **G4_ejection PASS** (the
  18-commit freeze was the calibration bug).
  dragon — 111 commits, 11.0 min, chamfer 0.1435, silIoU 0.642, hole 2.37 %, 128 fragments,
  stray 2.98 % — the weakest 150k result (40k corrected reference: 0.1199 / 0.901); the run
  stopped at 111: a slow plateau (L 0.2159 → 0.2147 over the last windows, acc 8/0) ended
  by the low-gain latch (gain 1.3e-5 < tol, then a −3e-4 reject) and patience; every
  late window also reports GUARD clamp=71 — 71 particles pinned at the auto box edge, the
  ejecta that reached the leash. The dragon's thin spines and legs are where the 150k
  discretisation loses most (hole 2.4 %).
  A — 240 commits, 23.0 min, chamfer 0.0781, silIoU 0.848, hole 0.39 %, 179 fragments,
  stray 2.38 %. spot — 92 commits, 8.7 min, chamfer 0.0769, silIoU 0.951, hole 0 %, 21
  fragments, stray 0.80 %.
  C — FROZE at 20 commits (2.1 min): brake rejects from anim 16 (gain −0.054, reversal
  0.98, replay) with the calibration fixed — the outer-merit gate's first-window
  normalisation genuinely misfires on C at 150k (the 40k gallery also noted "C stalls");
  chamfer 0.571. Re-run queued as `h150_C2` with `--no_outer_merit` after V.
  bob — 112 commits, 10.9 min, chamfer 0.1621, silIoU 0.577, hole 0.78 %, 134 fragments,
  stray 1.91 % (40k v5: 0.1425 / 0.822) — with dragon the second target that loses at
  150k: both are the thin-feature/ring shapes where the 150k material clumps.
  C re-run without the gate (`h150_C2`): the loss RISES every window (L 0.947 → 1.057,
  D_vol 0.23 → 0.26, kin 1.8, |v|max 4.7 wu/s) while `GUARD clamp` grows 2136 → 4738 —
  thousands of particles pinned at the auto box edge: the sphere's initial expansion
  toward the C ring overshoots the leash box and the clamp turns it into a runaway. The
  gate's brake in the first run was therefore correct. `h150_C3` re-runs C on the fixed
  32 wu domain (no clamping) — a configuration difference, not a tuned constant.
  `h150_C3` (fixed domain, gate on): the same brake rejects from anim 16 (gain −0.053,
  reversal 0.986) — C at 150k regresses in the merit after ~15 windows on either domain:
  the sphere→ring expansion overshoots and the gate correctly stops it. C at 150k is a
  FAILURE (reported with the gated run's numbers: chamfer 0.571, silIoU 0.70, 20 commits).
  V — 224 commits, 20.7 min, chamfer 0.0766, silIoU 0.792, hole 1.03 %, 260 fragments,
  stray 1.58 %.


### 2026-09-16 — diverse-mesh batch (`n40_*`, launched 16:40)

User request: try meshes beyond the ten in the gallery. Nine public test models
(alecjacobson/common-3d-test-models: cow, homer, max-planck, nefertiti, ogre, fandisk, beast,
cheburashka, bimba; beast / max-planck / ogre are not watertight — the axis-based fill still
produces a solid) added to `assets/` and run at 40k `--ppc 8` with the v5 recipe on the
corrected auto domain (`run_new40_20260916.sh`; GPU 0: cow homer maxplanck nefertiti
fandisk, GPU 2: ogre beast cheburashka bimba), post-processed per target into
`report_new40/<T>/` (surface video, particle GIF, PBR stills, census, fragments, loss).
Pre-registration: chamfer ≤ 0.13 and silIoU ≥ 0.85 on the blob-like models (cow, homer,
cheburashka, bimba, nefertiti, max-planck); the sharp-edged fandisk and the thin-limbed
beast/ogre are the expected weak cases; ejection counted by the grid fragment mask.

Results (40k, corrected auto domain, v5 recipe): cow — 87 commits, 4.7 min, chamfer 0.1135,
silIoU 0.956, hole 0 %, 2 fragments; ogre — 117 commits, 3.8 min, 0.1196 / 0.916 / 0.13 %,
17 fragments; homer — 117 commits, 6.4 min, 0.1195 / 0.956 / 0 %, 7 fragments; beast — 136
commits, 4.4 min, 0.1475 / 0.738 / hole 2.05 %, 76 fragments (thin limbs, the expected weak
case); cheburashka — 137 commits, 4.3 min, 0.1147 / 0.942 / 0.01 %, 1 fragment; max-planck —
101 commits, 5.7 min, 0.1113 / 0.978 / 0.16 %, 0 fragments; bimba — 81 commits, 2.9 min,
0.1113 / 0.979 / 0 %, 0 fragments; nefertiti — 145 commits, 8.1 min, 0.1173 / 0.959 / 0.01 %,
29 fragments; fandisk — 97 commits, 5.6 min, 0.1130 / 0.974 / 0 %, 1 fragment (the sharp-edged
case held up). Report: https://claude.ai/code/artifact/006c79b6-4b7e-4880-8810-d167b924ffe8


### 2026-09-16 — particle re-attachment (`--reattach`), user directive "mass ejection must be gone"

Mechanism (commit `pipeline/runner.py`, after assimilation): a particle whose grid cell is not
connected to the body — `fragment_mask` on the occupancy dilated by one cell, i.e. it shares
no grid node with any other material point — is no longer a continuum element, only a stray
mass. It is merged back onto the nearest body particle (position + half a particle spacing of
jitter; v, C, F, Fp, Fg copied). No particle is deleted (mass conserved), the window dynamics
and every loss are untouched, and every commit ends with zero fragments BY CONSTRUCTION. This
is conservative resampling of a degenerate sampling, the remedy the MPM/PIC literature applies
to under-sampled regions (arXiv 2603.03860 §resampling), not a force, a threshold or a weight.
Honest caveat: it is a discretisation repair at the commit, so a re-attached particle jumps
(≈ 2 cells) at that frame; the drift that produced it is not prevented, only undone, and the
same particle can drift again — the log counts every merge (`reattached`).
Pre-registration (40k trio dragon / bob / armadillo, `--bonds --reattach`): 0 fragments and
0 far particles at every commit; chamfer / silIoU within noise of the v5 reference
(0.1199 / 0.901 dragon); re-attachments per window logged. Then the 150k gallery is re-run
with the flag.

- `ejr_armadilo` (17:06): end census **0 particles > 0.25 wu, max 0.13 wu** (v5: 12 far,
  max 1.8 wu); chamfer 0.1147, silIoU 0.961 (v5: 0.1143 / 0.939); ~100 commits, 2.3 min to
  the gate stop; 1–2 particles re-attached per window from anim 20 on. G4_ejection still
  reads FAIL because `stray_max` is the maximum over ALL archived frames (the drift inside a
  window before its commit merges it); the delivered state is clean.
- `ejr_bob` (17:29): end census **0 particles > 0.25 wu, max 0.13 wu** (v5: 507 far); chamfer
  0.1122, silIoU 0.980 (v5: 0.1425 / 0.822), hole 2.68 % (v5 2.3 %); 324 particles
  re-attached over 21 commits, 7.3 min.
- `ejr_dragon` (17:33): end census **0 particles > 0.5 wu** (17 at 0.25–0.45 wu — within two
  cells, stretched surface material, not detached), 0 fragments; chamfer 0.1183, silIoU
  0.965 (v5: 0.1199 / 0.901); 227 particles re-attached over 27 commits, 7.3 min.
- Verdict: the pre-registration holds on all three (0 fragments and 0 far > 0.5 wu at the
  end; chamfer within noise, silIoU better on all three). The 150k gallery is re-run with the
  flag (v3 below).


### 2026-09-16 — 150k gallery v3 (`h150r_*`, launched 17:36): the v2 recipe + `--reattach`

Same configuration as v2 (150k `--ppc 8`, density units, `--bonds`, corrected auto domain,
archive stride 8, live packets) plus `--reattach`, run from the `/data` repo with
`scripts/ops/run_batch.sh` (GPU 2: teapot heart A C V; GPU 0 after `ejr_dragon`: bunny
armadilo dragon spot bob), post-processed by `post_run.sh` into `report150/<T>/`
(overwriting v2's media; v2's numbers stay in the log above). Pre-registration: 0 fragments
and 0 far particles at the end on every target; chamfer / silIoU within noise of v2 or
better (bob improved at 40k); C is expected to stop early again (its runaway is not an
ejection problem).

Results (v3): teapot — 121 commits, 16.1 min, chamfer 0.0721, silIoU 0.980 (v2: 0.0716 /
0.969), hole 0 %, 0 fragments, end census 0 far (max 0.09 wu), 32 particles re-attached over
3 commits.
bunny — 111 commits, 19.9 min, chamfer 0.0757, silIoU 0.974 (v2: 0.0765 / 0.920), hole 0 %,
0 fragments, 142 particles re-attached.
heart — 101 commits, 11.3 min, chamfer 0.0732, silIoU 0.983, 0 fragments, 0 re-attached
(it never ejected), end census 0 far.
armadillo — 131 commits, 14.3 min, chamfer 0.0801, silIoU 0.925 (v2: 0.0814 / 0.794), hole
0.32 %, 0 fragments, 619 particles re-attached; census 483 (0.32 %) — NOTE the census "far"
is the distance to the nearest TARGET point (off-target material), not isolation: on the
same frame only 4 particles have an 8th neighbour beyond 0.5 wu, each within 0.29 wu of the
body. The report columns are relabelled (off-target vs fragments).
dragon — 101 commits, 11.6 min, chamfer 0.1335, silIoU 0.776 (v2: 0.1435 / 0.642), hole
0.10 % (v2 2.37 %), 0 fragments, 3753 particles re-attached (2.5 % — the spines shed
material every window; the merges keep the body whole).
A — 201 commits, 20.6 min, chamfer 0.0743, silIoU 0.984 (v2: 0.0781 / 0.848), hole 0 %,
0 fragments, 1585 particles re-attached.
C — froze at 21 commits again (2.5 min; 6 brake rejects, gain ≈ −0.05): chamfer 0.570,
silIoU 0.708, 0 fragments, 6 re-attached — the sphere→ring overshoot is not an ejection
problem and the gate stops it correctly; C stays a 150k failure.
spot — 81 commits, 8.6 min, chamfer 0.0763, silIoU 0.976 (v2: 0.0769 / 0.951), hole 0 %,
0 fragments, 42 particles re-attached.
bob — 101 commits, 9.9 min, chamfer 0.1695, silIoU 0.581 (v2: 0.1621 / 0.577), hole 1.44 %,
0 fragments, 770 particles re-attached — bob stays the weak 150k case (ring + thin limbs) but
without strays. V — 171 commits, 17.0 min, chamfer 0.0737, silIoU 0.975 (v2: 0.0766 / 0.792),
hole 0 %, 0 fragments, 1074 particles re-attached.

v3 verdict (10/10 ran; batch 17:36–19:08): every delivered state has **0 grid-disconnected
particles**; chamfer within noise or better on all nine converged targets; silIoU improved on
bunny (0.920→0.974), armadillo (0.794→0.925), dragon (0.642→0.776), A (0.848→0.984),
V (0.792→0.975), spot, teapot; C stays a failure (gate stop at 21, not ejection). Re-attached
particles per target: heart 0, teapot 32, spot 42, bunny 142, armadillo 619, bob 770, V 1074,
A 1585, dragon 3753 — the count is the price of the drift the recipe still produces.


### 2026-09-16 — new-mesh 150k batch (`n150_*`, launched 21:58, unattended)

The nine public meshes at 150k with the v3 recipe (`--bonds --reattach`, corrected auto
domain, archive stride 8): GPU 0 cow homer maxplanck nefertiti fandisk, GPU 2 ogre beast
cheburashka bimba (`run_batch.sh`), post-processed into `report_n150/<T>/`.
Pre-registration: 0 fragments at the end on every mesh; chamfer / silIoU at or above the 40k
values for the eight blob-like meshes; beast remains the weak case.


### 2026-09-16 — five hypotheses for the remaining defects (user: visible ejection / pops, particle look, render-feedback doubt)

Symptoms on the v3 gallery: (a) strays still visible in the videos and the commit-time merge
makes them vanish in one frame; (b) during the morph the object reads as spheres/ellipsoids
(disk splats) in sparse regions; (c) the render channel carries a third of every update
(`g_share` 0.33, λ 0.03–0.05) yet `g_cos ≈ 0` and `render_work ≈ 0` — the render pull is
orthogonal to the physics pull and does no work on the accepted step.

| # | hypothesis | literature (method it borrows) | experiment | falsifier |
|---|---|---|---|---|
| H1 | Ejection is born at the free surface from a control field that varies below the grid stencil; a control resolved ON the MPM grid (node spacing = dx, the same B-spline weights) cannot push one particle against its stencil-mates, and permanent reference connectivity removes numerical fracture altogether. | de Vaucorbeil et al. 2020, Total-Lagrangian MPM (no numerical fracture) [S1]; Sadeghirad et al. 2011, CPDI particle domains [S2]; Su et al. 2022, A-ULMPM — adaptively updated reference configuration, "without numerical fracture" [S3] | `ejg_dragon`: 40k, v3 recipe, `--control_grid <grid_n>` (the auto-domain grid itself); then an A-ULMPM-style rule: reference updated only at commits (the window rollout already is; the trial is the control resolution) | fragments before re-attachment not reduced vs v3 (dragon: 227 merges) or chamfer > 0.13 |
| H2 | The per-commit relaxation (η = 0.5 of ALL elastic stretch) is rate plasticity with no yield surface, so a 2 %/window drift is forgiven exactly like the body's flow; a yield criterion keeps sub-yield strain elastic and pulls leaders back. | Stomakhin et al. 2013, snow MPM (clamped singular values, hardening) [S4]; Klár et al. 2016, Drucker–Prager return mapping [S5] | `--assim_yield`: von Mises return mapping on the Hencky elastic strain, ε_y from the material (the strain at which the fixed-corotated stress equals the control's clip stress) | drift rate of the end-fragment set unchanged (fragment_trace ratio slope) — expected, since the body's own flow is also ~2 %/window |
| H3 | The cell-sum density loss rewards a lone particle in an empty target cell (largest marginal gain), which is what pulls surface leaders out; a transport loss (EMD / Sinkhorn) moves mass as a flow and has no such reward. | Feydy et al. 2019, Sinkhorn divergences / GeomLoss [S6]; Huang et al. 2021, PlasticineLab — the PRT-EMD loss is "the most ideal choice" for filled targets [S7] | `--phys_loss ot`: debiased Sinkhorn divergence on 20k subsamples replaces D_vol (same units calibration), 40k dragon | merges/fragments not below v3, or chamfer worse |
| H4 | The particle look is a rendering artefact: disk splats sized by the local spacing separate wherever the sampling is sparse; an implicit surface (blurred density isosurface, anisotropic kernels) reads as one object and, below the resolvable density, a stray is not a surface. | Yu & Turk 2013, anisotropic kernels [S8]; van der Laan et al. 2009, screen-space fluid rendering [S9]; conservative resampling to restore sampling in depleted cells [S10] | `render_surface_video.py --mode iso`: ray-marched isosurface of the blurred density on the GPU; metrics untouched | reviewers still see spheres/ellipsoids or stretched thin features vanish |
| H5 | The render gradient is real but high-frequency (per-particle silhouette edges) and orthogonal to the smooth physics gradient; with a norm-balanced λ it costs a third of the step budget and does nothing. Either it must be smoothed onto the material (Sobolev pull) or the renderer's support widened (soft rasteriser / Gaussian splats). | Liu et al. 2019, SoftRas — probabilistic soft aggregation gives non-local gradients [S11]; Laine et al. 2020, nvdiffrast — analytic silhouette gradients [S12]; Kerbl et al. 2023, 3DGS — view-space position gradients as the densification signal [S13] | (i) physics-only twins `p40_*` vs render twins `r40_*` on dragon/bunny/armadillo (40k, v3 recipe) — does the render channel change chamfer/silIoU at all; (ii) `--render_gs_iters 4` (grid-GS Sobolev pull) on the same targets — does `g_cos`/`render_work` leave zero and does silIoU rise | twins identical within noise AND the Sobolev pull does not raise `render_work` above zero |

Sources: [S1] https://www.sciencedirect.com/science/article/abs/pii/S0045782519306759 ·
[S2] https://www.semanticscholar.org/paper/2ab3da08bbbd0fd86642bd5435a7b9189617a172 ·
[S3] https://onlinelibrary.wiley.com/doi/10.1111/cgf.14477 ·
[S4] https://www.academia.edu/3847240/A_Material_Point_Method_for_Snow_Simulation ·
[S5] https://math.ucdavis.edu/~jteran/papers/KGPSJT16.pdf ·
[S6] https://www.kernel-operations.io/geomloss/ ·
[S7] https://arxiv.org/pdf/2104.03311 ·
[S8] https://dl.acm.org/doi/10.1145/2421636.2421641 ·
[S9] https://dl.acm.org/doi/10.1145/1507149.1507164 ·
[S10] https://arxiv.org/html/2603.03860 ·
[S11] https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Soft_Rasterizer_A_Differentiable_Renderer_for_Image-Based_3D_Reasoning_ICCV_2019_paper.pdf ·
[S12] https://nvlabs.github.io/nvdiffrast/ ·
[S13] https://arxiv.org/pdf/2308.04079
Order of experiments: H5(i) and H4 first (cheapest, decide whether the render channel and the
renderer are the problem), then H1, H3, H2.

- **H5(i) dragon** (22:12): physics-only twin `p40_dragon` (λ = 0, same recipe incl.
  re-attachment) vs the render twin `r40_dragon`: D_vol at commit 20 0.0511 vs 0.0450, at the
  end 0.0259 vs 0.0206; final chamfer 0.1230 vs 0.1191, silIoU 0.920 vs 0.966. The render
  channel lowers even the physics term and adds 4.6 silIoU points — the feedback is real,
  modest in chamfer, large in silhouette. (armadillo, bunny twins pending.)
- **H4 result**: `scripts/render_iso_video.py` (ray-marched isosurface, blur 1.5 spacings,
  iso 0.5 × source bulk density) shows one continuous object at every frame; lone particles
  fall below the iso-level and do not render, dense detached chunks still do. With the true
  particle spacing the bunny's ears are revealed as sparse (most of the ear region is below
  half the bulk density) — a physics fact the splat renderer had hidden with oversized disks.
  The post-processing now writes the isosurface video as `<T>_surface.gif` and keeps the
  splat video as `<T>_splat.gif`; the ten v3 targets are being re-rendered.
- **H5(i) complete** (22:21): physics-only vs render twin (40k, v3 recipe, same stopping
  rules) — dragon 0.1230 / 0.920 → 0.1191 / 0.966; armadillo 0.1161 / 0.929 → 0.1146 / 0.961;
  bunny 0.1145 / 0.953 → 0.1135 / 0.967. The render channel raises silIoU by 1.4–4.6 points
  and lowers chamfer by 1–3 % on every target: the feedback is real. The near-zero
  `g_cos` / `render_work` therefore mean the render pull acts on directions the physics pull
  is indifferent to (silhouette-edge material), not that it is ignored. H5(ii) (`r40gs_*`,
  Sobolev render pull) tests whether smoothing that pull onto the material adds more.
- **H5(ii) dragon** (22:31): `r40gs_dragon` (grid-GS Sobolev render pull, 4 sweeps):
  chamfer 0.1175, silIoU 0.970 vs `r40_dragon` 0.1191 / 0.966; telemetry g_share 0.334 vs
  0.338, render_cos 0.034 vs 0.028, g_cos 0 in both. Smoothing the render pull onto the
  material changes almost nothing: the render channel's contribution is already in the
  accepted steps (H5(i)), its orthogonality to the physics pull is structural (silhouette
  edges vs bulk density), not a conditioning defect. H5 verdict: feedback real, modest;
  no remedy needed beyond what the recipe already does.
- **H3 first attempt**: the full Sinkhorn plan at every line-search evaluation (40k × 8192,
  10 sweeps) cost minutes per window — killed; replaced by one plan per window (barycentric
  targets of the window's start positions, per-particle L2 inside the window), calibrated
  once to D_vol's gradient norm at the source. `ejo_dragon` relaunched 22:32.
- **H5(ii) bunny**: `r40gs_bunny` 0.1134 / 0.968 vs `r40_bunny` 0.1135 / 0.967 — identical
  within noise. The Sobolev render pull is not a lever; H5 closed (feedback real, modest,
  structurally orthogonal to the density pull).
- **H3 dragon, first valid run** (`ejo_dragon`, 22:42; one plan per window, 8192 target
  samples, ε = dx²): **1 particle re-attached in 141 commits** (the density loss: 227) and 0
  fragments — the transport loss removes the drift that produced the ejecta, exactly as the
  hypothesis states. Quality is behind: chamfer 0.1558, silIoU 0.957, hole 0.95 % (density
  loss: 0.1191 / 0.966 / 0.08 %) — the 8192-sample plan at ε = dx² is a blurry coverage
  target. Next: 32768 samples with ε = (0.5 dx)² and ε = dx² (`ejo2`, `ejo3`).
- **H3 variants** (overnight): `ejo2_dragon` (32768 samples, ε = (0.5 dx)²): chamfer 0.1779,
  silIoU 0.954, hole 1.77 %, 3 re-attached, 211 commits; `ejo3_dragon` (32768, ε = dx²):
  0.1519 / 0.956 / 0.88 %, 2 re-attached, 181 commits; `ejo` (8192, dx²): 0.1558 / 0.957 /
  0.95 %, 1 re-attached. Ejection is gone in every variant (1–3 merges vs 227); more samples
  do not help and a sharper ε hurts, and every variant leaves a HOLE (0.9–1.8 % vs 0.08 %):
  the entropic barycentric map shrinks toward the interior (the known Sinkhorn bias), so
  thin features are under-filled. Remedy by construction: the DEBIASED Sinkhorn divergence
  S_ε = OT_ε(α,β) − ½OT_ε(α,α) − ½OT_ε(β,β) (Feydy et al. 2019) whose self-term cancels the
  shrinkage — per window the target displacement becomes T_i − T_i^self (`ejo4`).

New-mesh 150k batch (`n150_*`, v3 recipe, complete 23:41): cow 0.0755 / 0.944 (83 re-attached),
ogre 0.0840 / 0.879 (hole 1.28 %, 981), beast 0.1217 / 0.829 (hole 1.69 %, 3443), homer
0.0782 / 0.969 (271), cheburashka 0.0773 / 0.972 (3206), max-planck 0.0734 / 0.981 (14; rerun
after a deploy race crashed the first start), nefertiti 0.0896 / 0.868 (559), fandisk 0.1106 /
0.804 (600), bimba 0.0798 / 0.963 (1491) — every delivered state with 0 fragments; runs stop at
26–94 commits (outer-merit gate). Report: `output/report_n150/` (isosurface videos).
- **H3 debiased** (`ejo4_dragon`, 06:28): chamfer 0.1443, silIoU 0.965, hole 0.47 %, 2
  re-attached, 91 commits, 3.3 min. Debiasing halves the hole (0.95 → 0.47 %) and brings
  silIoU to the density loss's level (0.966); chamfer still lags (0.144 vs 0.119). The
  remaining gap is the ε-blur of the per-window targets (thin features under-resolved) —
  next: ε-scaling inside the solve so a smaller ε converges (`ejo5`).


### 2026-09-17 — the cause test across every mesh (`ot40_*`, launched 06:37)

User directive: confirm the cause in parallel and fix it — no ejection on any mesh. If H3 is
the cause, the transport loss must leave zero grid-disconnected particles on ALL 19 meshes
WITHOUT the re-attachment safety net. Sweep: the v3 recipe minus `--reattach`, with
`--phys_loss ot --ot_debias` (8192 target samples, ε = dx², 10 sweeps), 40k, four sequential
chains (GPU 0: bunny armadilo dragon spot bob | teapot heart A C V; GPU 2: cow homer maxplanck
nefertiti fandisk | ogre beast cheburashka bimba). Measure: the last `fragments N` line of
each log (grid-connectivity fragments at the end, no merges), the end census, chamfer /
silIoU / hole against the density-loss runs (v5 trio, batch h, n40). Pre-registration:
fragments ≤ 3 on every mesh (density loss: 1–76 on the new meshes, 12–507 far on the trio);
falsifier: any mesh with > 3 end fragments under OT.

06:42 relaunch (the first four runs killed after 3 min): ejo5_dragon finished at 06:37 with the
sharper plan — ε = (0.5 dx)², 20 sweeps with ε-scaling — beating ejo4 on every metric
(chamfer 0.1348 vs 0.1443, silIoU 0.969 vs 0.965, hole 0.25 % vs 0.47 %, 1 merge vs 2, and
17 commits vs 27: the sharper plan converges in fewer windows). 0.5 dx is not a tuned number:
it is the particle spacing at ppc 8 (dx / 8^(1/3)), i.e. the plan resolves to the resolution
of the cloud itself instead of the loss cell; the code default is now ε = (nn spacing)² and
20 sweeps (three ε-scaling stages need ≥ 6 sweeps each). The sweep runs with that setting.

**Density baselines (d40, same recipe, no re-attach, 40k), all 10 targets — complete 07:40.**
End fragments (grid connectivity): bunny 3, teapot 0, armadilo 12, heart 0, A 0, dragon 41,
C 0 (frozen at chamfer 0.54), V 33, spot 0, bob 85. Chamfer / silIoU / hole: bunny
0.1136/0.962/0.03 %, teapot 0.1110/0.975/0.01, armadilo 0.1144/0.935/0.27, heart
0.1114/0.981/0, A 0.1129/0.976/0, dragon 0.1277/0.854/1.15, C 0.5419/0.768/0.01, V
0.1224/0.902/0.01, spot 0.1145/0.977/0, bob 0.1239/0.849/2.49. New meshes (n40, same
recipe): beast 76, ogre 17, nefertiti 29, homer 7, cow 2, cheburashka 1, fandisk 1, bimba 0,
maxplanck 0. So 11 of 19 meshes eject under the density loss without the safety net.

**The OT sweep was stopped after six meshes (07:20) — two defects found in the OT code
by a synthetic ball → ball + thin-spike test (20k particles, spike = 1.7 % of the mass,
run on hyde06):**

1. `barycentric_targets` divided the plan rows by a_i = 1/N, which assumes converged row
   marginals; the solver returned after a g-sweep, so the rows were not normalised and the
   targets were scaled OUTSIDE the target's convex hull — the map reached x = 4.18 on a
   spike whose tip is at 2.40. Fixed: T_i = Σ_j π_ij y_j / Σ_j π_ij (row-normalised
   projection, always a convex combination of target samples).
2. 20 sweeps at ε = spacing² leave the plan far from converged: the row-marginal error was
   0.83 (83 %) and only 57 % of the spike's mass was reached (190 of 332 particles); the
   converged plan (marginal error < 1 %) reaches 87–88 % (290–294) with every map
   (barycentric, debiased, argmax). Convergence, not the map, was the thin-feature loss.
   Naive convergence at the target ε costs ~2000 sweeps (200 s at 20k); geometric
   ε-scaling from the squared target diameter, halving per level, each level iterated to the
   tolerance (Schmitzer 2019 / Feydy 2019), reaches the same 1 % in 115 sweeps (3.1 s at
   20k), 37 sweeps for 10 %, 65 for 3 %. The reach saturates by 3 %; 1 % is the stopping
   rule (a numerical convergence criterion, no shape constant). Warm re-solves after a
   coherent 0.04 wu move (a 2 % window) still need 64–181 sweeps at the target ε — the
   fixed-point iteration is intrinsically slow at small ε — so the per-window cost, not the
   cold start, sets the OT recipe's speed (measured next at 40k and 150k).

The ot40 runs that finished before the stop (row-scaled, unconverged plans) still had 0–2
end fragments (bunny 0, teapot 0, heart 0, cow 2) against 3 / 0 / 0 / 2 for the density
loss: even a bad transport plan does not reward leaving the body. Their quality numbers
(bunny 0.154 / 0.925 / 0.50 %) are not the OT recipe's and are discarded; the sweep is
re-run with the fixed solver.

**Per-window cost of the converged plan (hyde06, tol 1 %):** full-cloud Sinkhorn, 40k:
cold 108 sweeps 11 s (main + self plan); a warm re-solve after a 0.04 wu coherent move at
the target ε needs 570 sweeps / 32 s — MORE than the anneal — and re-annealing from 4
levels up 95 sweeps / 9 s; 150k: cold 52.7 s. Sweeps are O(N·M) (memory-bound blocks), so
the full plan is not affordable per window. Structural fix (commit after 4efa440): solve the
dual on a fixed uniform subsample of 8192 particles against the 8192 target samples (every
solve anneals through all levels) and evaluate the out-of-sample entropic map — the
row-normalised barycentric projection with the subsample's potentials (Pooladian &
Niles-Weed 2021) — for all N particles in one O(N·M) pass. Ball-to-spike, debiased: 40k
in-spike 537/628 (85.5 %) vs the converged full plan 560/628 (89 %), 2.6 s vs 12 s per
window; 150k 2586/2385 (an 8 % over-fill; the extension conserves mass only on the
subsample) at 4.0–4.7 s per window. Subsample fraction is what changes between 40k (20 %)
and 150k (5.5 %); both within ±10 % of the thin feature's mass share.

**First pipeline runs with the subsampled map (ot40b, 07:32, ε = particle spacing) were
still slow:** the real source→bunny problem needed 268–400 sweeps for the main plan (the
400 cap hit, marginal error up to 4.7 %) and 176–388 for the self plan, 5–18 s per window
with two runs per GPU. Stopped at 07:45. Three changes (commit after ae5313f): (1) the
plan's blur resolves the SAMPLE set it is computed on — √ε = particle spacing ×
(N/8192)^(1/3) (1.7 spacings at 40k, 2.6 at 150k), a derived rule; synthetic 40k:
in-spike 87 % (vs 85 % at the particle spacing), 1.2 s vs 1.6 s per window, 100–108
sweeps; (2) the cost block is computed once per solve (the subsample block fits one
chunk); (3) the self plan uses the symmetric averaged Sinkhorn update (f = g), converging
to 5e-4 in 80 sweeps. Sweep relaunched as ot40b with these defaults.

**ot40b (OT-only, fixed solver) — stopped after four meshes (07:44):** all runs end in
3–4 min (outer-merit patience once the map is reached) with holes. bunny 0 frag,
0.1278 / 0.952 / 0.16 % (density 3 frag, 0.1136 / 0.962 / 0.03 %); cow 0 frag, 0.1313 /
0.896 / 0.13 % (2 frag, 0.1135 / 0.956 / 0); ogre 0 frag, 0.1334 / 0.887 / **2.66 %** (17
frag, 0.1196 / 0.916 / 0.13 %); armadilo **5 frag**, 0.1417 / 0.889 / **3.54 %** (12 frag,
0.1144 / 0.935 / 0.27 %). Whole-trajectory stray_max 0.10–0.80 %. Verdict: the transport
map alone removes most ejection but not all (armadillo 5) and cannot fill at the particle
scale — the entropic map's image sits 0.92 spacings from the nearest target point on the
real bunny whether 8192 or 16384 target samples are used (offline test); the sample count
is not the limit, the entropic blur is. Convergence criterion switched to the standard L1
mass error (84 sweeps vs 152 for the max-row criterion, identical map).

**ot_leash (commit after 7e8f99f): the cell sum stays the fill term; the plan is a
leash.** L = D_vol + s · mean relu(|x_i − T_i| − √ε)², with T_i the debiased map image
of the window's start position and √ε the plan's own blur radius (the sample spacing);
s from the one-shot gradient parity. Inside the radius the density loss refines freely;
a particle that leaves the body by more than the plan's resolution is pulled back to
where the plan puts its mass; in the first windows every particle is outside the radius
(the image is the whole transport away), so the term is the OT pull until the cloud
arrives. Pre-registration (ot40c, 40k, no re-attach, armadilo ogre dragon bob = the
worst density-loss ejectors, 12 / 17 / 41 / 85 fragments): end fragments ≤ 3 on all four
AND chamfer / silIoU / hole within noise of the density runs (0.1144/0.935/0.27,
0.1196/0.916/0.13, 0.1277/0.854/1.15, 0.1239/0.849/2.49). Falsifier: fragments > 3 on any,
or a hole increase > 0.5 pt.

**ot40c v1 FALSIFIED (07:50):** armadilo 1 frag, 0.1396 / 0.822 / 4.32 %, ogre 2 frag,
0.1452 / 0.807 / 1.97 %; both stopped by the outer-merit gate at 26–31 windows (candidates
rejected with gain −0.04…−0.06 and reversal 0.93: the two terms oscillate). Two causes:
(1) the entropic image sits ~0.9 spacings INSIDE the target (blur; p90 1.4, leash radius
1.7), so for the tail of particles the leash and the cell sum pull toward different places;
(2) the window-1 global-norm calibration (all particles outside the radius, |g_ot| large)
makes the per-particle hinge on a stray ~1e-6, i.e. no restraint. v2 (commit after
aa9a970): anchors = map images projected onto the target point set (KD-tree; both terms
now want the same support, the plan still decides the region), Huber hinge (quadratic r…2r,
linear beyond), and per-particle parity — the scale is set so that a particle at 2r feels
the pull the density loss exerts on its most-pulled particle at the source; beyond 2r the
pull is constant. Trials ot40d on the same four meshes.

**ot40d v2 FALSIFIED (07:56), worse:** armadilo 5 frag, 0.2016 / 0.730 / 1.48 % (12
rejects, stop at 28 plans); dragon 10 frag, 0.2020 / 0.775 / 0.88 %. A strong per-particle
pull toward NOISY anchors (the sampled map's 0.9-spacing fuzz, re-drawn every window)
drags particles across the body and creates strays. Two structural facts learned:

1. The outer-merit gate reads the fixed cell sum (`rec["d_vol"]`, runner.py:698) while
   the inner objective of every OT variant is (partly) a per-window transport surrogate:
   when the inner descent does not lower the cell sum the gate sees a regression, rejects,
   and patience stops the run — this is why EVERY OT run ended at 3–4 min. An OT term in
   the inner objective needs either a consistent merit (the window-independent Sinkhorn
   divergence S_ε as a merit component) or no gate.
2. A leash needs anchors that are as smooth as the continuum map; v3 (commit 4ebbb04)
   averages the sampled displacement over the k material neighbours inside one blur
   radius (k from the blur volume, ~20 at 40k) before projection.

Trials (08:00): T1 `ot40e` — OT-only with a sharper plan (16384 target samples, √ε = one
particle spacing = 0.28 loss cells) and the gate off (`--no_outer_merit`: the transport
loss has no runaway mechanism, so the brake is not needed), armadilo + ogre. T2 `ot40f` —
leash v3, dragon + bob. Pre-registration unchanged (fragments ≤ 3, quality within noise of
d40).

**Leash v3 FALSIFIED and the leash line closed (07:57):** dragon 13 frag, 0.2063 / 0.776 /
1.98 %, gate stop at 23 plans — denoising the anchors changed nothing; a per-particle pull
strong enough to hold a stray is strong enough to tear the bulk when its anchor disagrees
with the cell sum, and a pull weak enough not to tear holds nothing (v1). Added T3 `ot40g`:
OT-only with the default plan (8192 samples, √ε = sample spacing) and the gate off, on
armadilo + ogre, to separate the gate effect from the plan sharpness (T1 `ot40e` = sharp
plan + gate off; `ot40b` = default plan + gate on).

**ot40g (gate off) stopped at 20 windows anyway** (ogre 2 frag, 0.1333 / 0.885 / 2.17 %,
"converged at anim 20 (phys=0.0131); holding still"): the stop is the convergence tracker
(runner.py:920, `stale >= patience` on the fixed cell sum + kinetic track), not the gate.
The cell sum ROSE from window 15 on (0.009 → 0.013) while the OT surrogate kept falling:
the blurred map pulls the surface inward (fuzz), which the cell sum reads as holes. Mid-run
the OT runs also clamp 9–11 particles per window at the domain box (fast strays, |v|max
4.5) — OT-only is not stray-free during the run either. The sharp-plan trial (ot40e) was
stopped: 27 s per plan (16384² blocks) and the same tracker would stop it.

**ot_pace (commit after d8414f4): the density loss keeps the objective, the transport
plan paces its TARGET.** Each window the current cloud is advected toward its debiased,
material-smoothed map image by at most one blur radius per particle
(x_int,i = x0,i + min(1, √ε/|d_i|) d_i) and rasterised with the loss's own CIC splat;
that grid is the window's target density (McCann displacement interpolation of the plan,
one plan per window). The cell sum then only asks for local moves along the plan — no
cell far from a particle rewards it for leaving the body, which is the H3 mechanism —
and once every particle is within a blur radius of its image the target is the image
cloud itself. The fixed cell-sum merit, gate and tracker are untouched (they read the
true target). No new constant: the pace is the plan's blur radius. Pre-registration
(ot40h, 40k, no re-attach, armadilo ogre dragon bob): fragments ≤ 3 on all four and
chamfer / silIoU / hole within noise of d40 / n40; falsifier as before.

**ot40h RESULTS (08:20) — ot_pace passes on three of three finished (ogre pending):**

| mesh (40k, no re-attach) | density: frag / chamfer / silIoU / hole | ot_pace: frag / chamfer / silIoU / hole | time |
|---|---|---|---|
| bob | 85 / 0.1239 / 0.849 / 2.49 % | **1** / 0.1188 / **0.975** / 2.78 % (target ring) | 6.6 min |
| dragon | 41 / 0.1277 / 0.854 / 1.15 % | **2** / 0.1296 / **0.967** / **0.45 %** | 9.3 min |
| armadilo | 12 / 0.1144 / 0.935 / 0.27 % | **0** / 0.1227 / **0.965** / 0.23 % | 12.5 min |

| ogre | 17 / 0.1196 / 0.916 / 0.13 % | **3** / 0.1257 / **0.957** / 0.78 % | 23.1 min (3 runs/GPU) |

Fragments 85/41/12/17 → 1/2/0/3 and silIoU up 3–13 points; chamfer within ±0.008; the
runs go to 97–243 windows (no tracker stop: the fixed cell sum keeps falling).
Whole-trajectory stray_max 0.59–0.86 % (transient isolated particles mid-run; the end
fragments are what remain). Against the pre-registration: fragments ≤ 3 on all four ✓;
quality within noise on bob, dragon, armadilo ✓; ogre's hole 0.13 → 0.78 % exceeds the
0.5-pt falsifier (its silIoU is 4 points better) — a partial pass, recorded as such. The
remaining 15 meshes launched as `op40_<T>` (08:17); first in: C (the letter, which the
density recipe never morphs at 40k: chamfer 0.54) — ot_pace 8 fragments, 0.1710 / 0.927 /
6.76 %, gate stop at 31 windows (13 rejects): a much better shape but not fragment-free.
Then fandisk 1 → 0 (0.1201 / 0.977 / 0.02 %), V 33 → **0** (0.1186 / **0.979** / 0; density
0.902), cow 2 → 2 (0.1216 / 0.961 / 0.01), bunny 3 → **0** (0.1225 / 0.972 / 0), beast 76 →
14 (0.1699 / 0.878 / 1.50 %; density 0.1475 / 0.738 / 2.05 %, 15 gate rejects) — the hardest
new mesh improves by 14 silIoU points but is not fragment-free. spot 0 → 0 (0.1193 / 0.975 /
0), teapot 0 → 0 (0.1180 / 0.971 / 0.22 %), cheburashka 1 → 0 (0.1209 / 0.973 / 0.09 %),
heart 0 → 0 (0.1145 / 0.981 / 0), **homer 7 → 14** (0.1294 / 0.943 / 0.02 %; density 0.1195 /
0.956 / 0) — the one counter-example so far: thin arms and hands, the plan sends mass along
them and the tips detach (stray_max 0.74 %, 0 gate rejects; fragments appear in a burst at
window 49, +12 at once, and stay). A 0 → 0 (0.1175 / 0.980 / 0.01 %). 150k bunny v2 (the
arrival-projection code, 09:20): 0 fragments, **19 re-attachments** (v3 142), silIoU 0.974
(= v3), hole 0.18 %, chamfer 0.0949 (v3 0.0757), 21.4 min.

**150k with re-attachment (h150p, EXTRA `--archive_stride 8 --reattach --phys_loss
ot_pace --ot_debias`), first run bunny (08:46):** 0 fragments, **20 re-attachments** over
the run (v3 density recipe: 142), silIoU 0.970 (v3 0.974), hole 0.15 %, 25.7 min — but
chamfer 0.0976 vs 0.0757: the end target was the rasterised IMAGE cloud, whose entropic
fuzz sits ~0.9 spacings inside the target, so the surface ended slightly inside and
fuzzy. Fix (commit f68df6c): a particle within one blur radius of its image contributes
the nearest TARGET point to the paced grid instead of the image (particles still in
flight keep the advected position), so on the support the end target is the target. The
h150p batch was stopped and relaunched on this code (the first bunny moved to
void_h150p_v1/); the 40k sweep already in flight keeps the pre-fix code for runs that had
started (fragments are the sweep's verdict, unaffected by the end-state projection).

150k teapot v2 (09:00): 0 fragments, **2 re-attachments** (v3 32), silIoU 0.977 (v3 0.980),
hole 0.36 %, 16.6 min — chamfer 0.0826 vs 0.0721 still. The 150k bunny v2 log shows why:
at anim 121 only 26 % of the particles were "arrived" (|d_i| ≤ blur radius) — at 150k the
map noise IS the blur radius (both are the sample spacing, 0.1 wu = 2.6 spacings), so the
arrival test was a coin flip and the paced grid stayed a blurred copy of the cloud. Fix
(commit after 8df1527): every paced position within one blur radius of a target POINT is
snapped to it (removes the normal blur component, keeps the tangential transport);
positions still in flight stay advected. Deployed mid-batch: bunny and teapot (v2 code)
are queued for a re-run after the new-mesh batches; armadilo onward run on the final code.

**The snap was wrong (09:40):** 150k cow on the snap code — 0 fragments but 104
re-attachments (density recipe n150: 83), silIoU 0.920 (0.944), hole 1.54 % (0), chamfer
0.111 (0.076): where the source sphere overlaps the target body, 99 % of the paced positions
lie "on the support" and snapping them to their own nearest target point removes the
tangential transport — the pacing degenerates into a projection. Reverted (commit after
211565b) to the arrival projection (f68df6c). A full-resolution map (potentials
c-transformed to every target point and every particle, N × N passes, 7 s per call at 150k)
was then measured on the real 150k bunny end state: the plan displacement is p50 2.0 blur
radii, p90 7, with EITHER map (arrived 26 vs 28 %), while the end state is 0.99 spacings
from the target — the large displacements are interior density redistribution (the MPM
cloud is not uniformly dense; the target sampling is), not sample noise. The cell sum's
log form tolerates that; the plan does not, and the paced target keeps asking for interior
moves that perturb the surface — the source of the +0.02 wu chamfer at 150k. Accepted for
now (silIoU equal, re-attachments 7–70× fewer); the 150k batches were stopped and relaunched
on one consistent code (bunny / teapot / heart, already on that code, kept).

**40k cause-test sweep COMPLETE (09:45), all 19 meshes, no re-attachment, end fragments
density → ot_pace:** bunny 3→0, teapot 0→0, armadilo 12→0, heart 0→0, A 0→0, dragon 41→2,
C 0(frozen)→8, V 33→0, spot 0→0, bob 85→1, cow 2→2, homer 7→14, maxplanck 0→0, nefertiti
29→0, fandisk 1→0, ogre 17→3, beast 76→14, cheburashka 1→0, bimba 0→0. Sum 307 → 44;
fragment-free on 13 of 19, ≤ 3 on 16 of 19; worse only on homer (thin arms); C morphs
(silIoU 0.77 → 0.93) but the gate stops it with 8. silIoU up on every ejecting mesh
(+3…+14 points); chamfer +0.005…+0.01 on most (the paced surface).

**Visual QA of the 150k ot_pace stills (09:45) — a regression the metrics understated:**
bunny and teapot PBR stills against v3: the ot_pace surface is bumpy at the particle scale
everywhere and the thin features (ears, feet, spout, handle) are visibly sparser — single
particles instead of a solid. Same silIoU, chamfer +0.02 wu (0.5 spacings): that is what
"the paced target keeps asking for interior moves" looks like. Not shippable as the gallery.

**ot_resid (commit 99b3753): pace only the cell sum's RESIDUAL.** Per window, on the loss
grid: excess = (cloud − target)+, deficit = (target − cloud)+ (equal totals). Particles are
drawn ∝ their excess fraction e_i = excess/cloud at their cell, target points ∝ the deficit
at their cell; the Sinkhorn plan is solved between those two samples (uniform after the
importance draw), the debiased map gives every particle a displacement d_i, smoothed over
the material neighbours with excess weights, and the paced position is
x0,i + e_i · min(1, h/|d_i|) · d_i (arrived excess particles project onto the target).
Particles in satisfied cells contribute their own position: the paced target differs from
the current occupancy only where the cell sum itself wants mass to move, and there it asks
for one blur radius along a coherent flow. No far-cell reward for a lone particle, and no
interior redistribution beyond the cell sum's own residual. No new constant (same residual,
same blur radius, same neighbour rule). Trials `or40_bunny`, `or40_dragon` (40k, no
re-attach; excess mass at the start 38 %, plan 120 sweeps, 3.5 s per window with three
runs per GPU). Pre-registration: fragments ≤ the ot_pace values (0 / 2) AND chamfer within
0.005 of the density runs (0.1136 / 0.1277) with a smooth surface in the stills.

**ot_resid FALSIFIED on bunny (10:08):** 0 fragments, but chamfer 0.1248 (density 0.1136,
ot_pace 0.1225), silIoU 0.962, hole 0.16 %, 35 gate rejects in 207 windows — worse than
ot_pace. 150k V (ot_pace) meanwhile: 0 fragments, 81 re-attachments (v3 1074), silIoU 0.972
(v3 0.975), chamfer 0.104 (v3 0.074), hole 0.96 %: the same pattern on every 150k target.

**ot_shape (commit after 1381877):** ot_pace with the particle subsample drawn with
probability inverse to the cloud's cell mass at the particle, so the plan's source measure
is uniform over the occupied cells (the support) like the uniformly sampled target — the
plan transports the SHAPE and asks for no interior redistribution (the cell sum tolerates
the interior; ot_pace paid for equalising it with rough surfaces). Redrawn every window.
Trials `os40_bunny`, `os40_dragon`; same pre-registration as ot_resid.

**ot_shape FALSIFIED on bunny (10:20):** 0 fragments, chamfer 0.1301, silIoU 0.968, hole
0.27 %, 16 rejects, stop at 95 windows — worse than ot_pace.

**Hand-off (commit 7fa1ab8, part of ot_pace):** the paced target is used while some target
cell with mass lies beyond one cell of any occupied cell; once every deficit cell is
adjacent to the body (3³ dilation of the occupancy on the loss grid), the window target is
the FIXED target grid — the cell sum's CIC gradient already reaches every deficit from the
body, so no far cell can reward a lone particle, and the fixed target fills thin features
at full strength (the paced target left them sparse at 150k). Re-evaluated every window;
no constant (cell adjacency). Trials `oh40_bunny`, `oh40_dragon` (1712 far target cells
at the start on bunny). Pre-registration: fragments ≤ ot_pace (0 / 2), chamfer within
0.005 of density (0.1136 / 0.1277), smooth stills.

**Hand-off FALSIFIED (10:28):** bunny 0 fragments but chamfer 0.1279, 14 rejects right
after the switch (14 hand-off windows), stop at 136 windows in 6 min — changing the inner
objective mid-run reads as a merit regression to the gate. Kept opt-in (`cfg.ot_handoff`),
off by default; deployed 2a7d9b3 (no 150k run started on the hand-off code). The
structural search stops here for today: four variants that tried to remove the
interior-redistribution cost (snap, ot_resid, ot_shape, hand-off) all lost to plain
ot_pace.

**150k ot_pace + re-attachment so far (v3 → v4: re-attachments, silIoU, chamfer):**
bunny 142 → 19, 0.974 → 0.974, 0.076 → 0.095; teapot 32 → 2, 0.980 → 0.977, 0.072 → 0.083;
heart 0 → 0, 0.983 → 0.978, 0.073 → 0.083; V 1074 → 81, 0.975 → 0.972, 0.074 → 0.104;
spot 42 → 34, 0.951 → 0.981, 0.077 → 0.087; armadilo 619 → 153, 0.925 → 0.968, 0.080 →
0.091. All 0 fragments. Re-attachments 4–13× fewer, silIoU equal or better (+3…+4 on spot,
armadilo), chamfer +0.01…+0.03 wu everywhere (the paced surface).

**Where this leaves the day (10:35).** The cause is confirmed on 19 meshes and the
structural fix works for what it was built for: without any safety net the end fragments
fall from 307 to 44 (0 on 13 meshes), and with the re-attachment net the 150k runs merge
4–13× fewer particles, so the visible pops in the videos drop accordingly. Its cost is
visible too: the PBR stills of V, bunny and teapot show porous thin extremities and a
particle-scale roughness that the v3 density recipe does not have (v3 shows ejecta blobs
instead — V's stray at the bottom of the v3 still). The cause of the cost is measured (the
plan asks for interior redistribution the cell sum tolerates; the paced target is a blurred
copy of the target wherever the plan has not arrived), and four attempts to remove it in
one day failed (snap, residual-only plan, support-uniform plan, hand-off). Deliverable: the
v3 gallery stays; the 150k ot_pace runs go to a separate v4 page for comparison; the open
item is a paced target that is exact on the arrived support without changing the inner
objective mid-run.

**150k, all 10 targets (11:40) — v3 (density + re-attach) → v4 (ot_pace + re-attach):
re-attachments / chamfer / silIoU / hole**

| target | v3 | v4 |
|---|---|---|
| bunny | 142 / 0.0757 / 0.974 / 0 | 19 / 0.0949 / 0.974 / 0.18 % |
| teapot | 32 / 0.0721 / 0.980 / 0 | 2 / 0.0826 / 0.977 / 0.36 % |
| heart | 0 / 0.0732 / 0.983 / 0 | 0 / 0.0825 / 0.978 / 0.13 % |
| spot | 42 / 0.0763 / 0.976 / 0 | 34 / 0.0869 / 0.981 / 0 |
| A | 1585 / 0.0743 / 0.984 / 0 | 90 / 0.0894 / 0.983 / 0 |
| V | 1074 / 0.0737 / 0.975 / 0 | 81 / 0.1036 / 0.972 / 0.96 % |
| armadilo | 619 / 0.0801 / 0.925 / 0.32 % | 153 / 0.0912 / 0.968 / 0.25 % |
| dragon | 3753 / 0.1335 / 0.776 / 0.10 % | 134 / 0.1206 / 0.969 / 0.55 % |
| bob | 770 / 0.1695 / 0.581 / 1.44 % | 253 / 0.1020 / 0.958 / 3.01 % |
| C (fails in both) | 6 / 0.5698 / 0.708 / 0 | 63 / 0.3906 / 0.748 / 1.26 % |

All 0 end fragments in both (the net). Re-attachments 8023 → 829 (−90 %). silIoU: dragon
+19, bob +38, armadilo +4, spot +0.5, the rest within 0.006. Chamfer: worse by 0.010–0.030
on the seven easy targets (the paced surface), better on dragon and bob. The PBR stills:
v4 bob is a clean ring with a porous lower part; v3 bob is a disc with chunks in flight.

**User (12:00): is the LOG in the loss right, given ejection persists? — and try a larger
mesh size.** The log residual r = log(1+m_t/m_ref) − log(1+m/m_ref) has gradient
2r/(m_ref + m) per unit cell mass: largest at an EMPTY cell (m = 0) and collapsing as the
cell fills, so a surface particle is rewarded more for reaching a far empty cell than for
finishing an adjacent nearly-full one — H3 amplified by the log. A linear residual
(m − m_t)/m_ref has a gradient proportional to the deficit (no empty-cell amplification).
Mesh size: a coarser MPM grid (ppc 8 → 27, cell = 3 spacings) makes the numerical
fracture (a particle sharing no grid node with its neighbours) far rarer; a coarser loss
grid (loss_res 64 → 32) halves the number of empty cells a leader can be pulled to and
doubles the CIC reach. Three trials on the worst density-loss ejectors (dragon 41, bob 85,
armadilo 12 fragments; 40k, no re-attach, otherwise the recipe): `lin40` `--dvol_form
linear` (commit after a9ea680), `pp40` `--ppc 27`, `lr40` `--loss_res 32`.
Pre-registration: fragments vs d40 (41 / 85 / 12) and chamfer / silIoU / hole vs d40
(0.1277/0.854/1.15, 0.1239/0.849/2.49, 0.1144/0.935/0.27); the mechanism claim (log
empty-cell amplification) is supported if `lin40` alone cuts the fragments by more than
half on all three without a quality loss.

**Results (11:25):**

| trial (40k, no re-attach) | dragon frag / chamfer / silIoU / hole | bob | armadilo |
|---|---|---|---|
| density, ppc 8, loss_res 64 (d40) | 41 / 0.1277 / 0.854 / 1.15 % | 85 / 0.1239 / 0.849 / 2.49 % | 12 / 0.1144 / 0.935 / 0.27 % |
| ot_pace (ot40h) | 2 / 0.1296 / 0.967 / 0.45 % | 1 / 0.1188 / 0.975 / 2.78 % | 0 / 0.1227 / 0.965 / 0.23 % |
| linear residual (lin40) | 25 / 0.4451 / 0.531 / 2.14 % (stops at 5 min) | 0 / 0.4148 / 0.702 (gate stop at 1.3 min) | — |
| loss_res 32 (lr40) | 23 / 0.1204 / 0.903 / 0.06 % | pending | pending |
| **ppc 27 (pp40)** | **0** / 0.1264 / **0.955** / 0.02 % (7.5 min) | **2** / 0.1176 / **0.958** / 2.89 % (8.9 min) | pending |

The linear residual is not a clean test: without the log's saturation the interior
over-density of the sphere dominates the objective while the merit, tracker and unit
calibration are still the log form, so the gate stops the run at once (a fair test needs
the whole pipeline in linear units). The coarser loss grid halves dragon's fragments. The
coarser MPM grid — 27 particles per cell, a cell of three spacings — removes them on dragon
and nearly on bob with the plain log density loss, and lifts silIoU by 10 points: the
numerical fracture (a particle that shares no grid node with its neighbours) needs a
one-spacing gap at ppc 8 and a three-spacing gap at ppc 27. 27 = 3³ is the standard
high-quality MPM particle count per cell, a discretisation choice, not a per-shape constant.

**Complete (11:25):** ppc 27 armadilo **0** / 0.1198 / **0.955** / 0.28 % (4.0 min) — all
three worst ejectors at 0 / 2 / 0 with the plain log density loss, silIoU +10 / +11 / +2,
chamfer within 0.006, and the runs are faster (grid 41³ instead of 59³ at 40k). loss_res 32:
bob 87 (no help), armadilo 5. Linear residual: armadilo 38 / 0.685 (broken as a drop-in, as
above). The mesh-size lever the user asked for beats the transport pacing on both
fragments and surface quality (no paced target, no roughness). Launched (11:25): the 150k
gallery on `--ppc 27 --reattach` (density log loss; prefixes h150q / n150q, four chains,
150k grid 53³) and the ppc-27 40k sweep on the other 16 meshes without re-attachment
(`pp40_<T>`). Pre-registration for the 150k v5 gallery: 0 fragments (the net) with fewer
re-attachments than v3 on every target that had > 30, and silIoU ≥ v3 − 0.005 on all but
C; for the 40k sweep: fragments ≤ 3 on every mesh.

**ppc curve (40k, no re-attach, log density loss):** ppc 8 (cell 2h): dragon 41 / 0.854,
bob 85 / 0.849, armadilo 12 / 0.935; ppc 27 (3h): 0 / 0.955, 2 / 0.958, 0 / 0.955; ppc 64
(4h): dragon 0 / 0.930 (grid 32³, 2.5 min), bob 1 / 0.957 — fracture-free from 3h on, and
27 is the finest fracture-free grid (64 loses silIoU on dragon). Sweep so far at ppc 27:
homer 7 → 0 (0.958), bunny 3 → 0 (0.960), maxplanck 0 → 0, teapot 0 → 0. 150k v5 so far:
teapot 0 fragments, **0 re-attachments** (v3 32), 0.0749 / 0.977; spot 0 / **1** (v3 42),
0.0799 / 0.965; C gate stop as always.

**Why the C++ oracle never ejected (user remark, 12:20; `legacy/configs/
ablation_bunny_ppc6_full_method.yaml`):** grid_dx 1.0 on a shape normalised to an 8 wu
bbox diagonal (the bunny spans ~8 cells; ours spans 38 at 40k and 58 at 150k), with
SHELL-BIASED sampling — 6³ particles per cell in a 2-cell surface shell, 1 per cell inside —
so the cell is six surface spacings wide and the thin features are sub-cell: decoupling
would need a six-spacing gap and there is no far empty cell for a leader to reach. Its
optimiser is also gentler (one GD/Adam iteration per window, alpha 0.01, dLdF clip 0.05;
ours: 8 Adam iterations + line search). Same dt (1/240), same F smoothing (0.955), same
cubic B-spline APIC; drag 0.05 vs our 0.9; no plastic assimilation in the C++ baseline
(eta 0; plasticity lives in its Python driver). So the C++ result is the mesh-size lever at
its extreme, not a different mechanism. Follow-ups: (a) port the shell-biased sampling
(surface resolution without a fine grid — the way to keep detail at a coarse cell, where
ppc 64 lost silIoU); (b) diagnostic `it40_{dragon,bob}` with `--iters 1` (the C++ step
budget) to see whether leader formation depends on the per-window control aggressiveness.

**it40 FALSIFIED (11:50):** one iteration per window makes it far worse — dragon 408
fragments (0.1562 / 0.861, stop at 2 min), bob 363 (0.1868 / 0.799): a single raw Adam step
per window without the line search is a larger, less controlled per-particle move, not a
gentler one. The C++ optimiser is not why the oracle never ejected.

**Shell-biased sampling ported (commit de4f084, `--sample shell --shell_ratio 6
--shell_cells 2`, the C++ numbers):** `sample_volume_shell` samples a 2-cell surface shell
at spacing h_s and the interior at 6 h_s with n solved for h_s; every particle carries its
relative rest volume as mass (density stays uniform; the rest-volume pass, the rollout
spec, the persistent trajectory and the target grid all take per-particle masses). At 40k
on the isosphere: shell 39 737 particles (h_s 0.045 wu), interior 263 (h_i 1.48 wu, masses
up to 89×), shell = 41 % of the volume — i.e. the C++ regime: a hollow-ish body with a dense
skin, and the MPM cell (0.21 wu at ppc 8) is **4.8 shell spacings** wide. Dragon target:
shell 74 % of the body (thin). Trials `sh40_{dragon,bob,armadilo}` (ppc 8, no re-attach).
Pre-registration: fragments ≤ 3 on all three (the decoupling gap is 4.8 spacings); the
interest is whether the surface quality holds with a nearly hollow interior.
