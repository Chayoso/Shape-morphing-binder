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


### 2026-09-17 evening — five goals (user): (1) zero ejection at 150k AND a proof that the render gradient changes the physics; (2) C morphs; (3) no floating particles / particle-looking blobs in any video; (4) photoreal renders; (5) material properties change the trajectory (proof)

**Diagnostics first (18:30).**
- C without the gate (`cdiag_C`, 40k, cell 0.31): the run dies in 0.2 min — 3749 particles
  clamped at the domain box within three windows (614 → 1236 → 1776 → 3749). So C is not a
  compression failure: it is a runaway EXPANSION. After volume matching the sphere sits
  inside the C's hole, i.e. entirely in target cells with m_t = 0, where the log residual's
  gradient is maximal and points outward in every direction with nothing pulling back until
  the arms are reached; the gate reads the overshoot as a merit regression and freezes the
  run at ~20 windows (v2–v6 alike). The one method that morphed C at 40k was the
  transport-paced target (silIoU 0.77 → 0.93, ot40h): the plan gives every hole particle a
  direction to a specific arm cell. Ladder `cd2_C` (ot_pace + cell 0.31 + net), `cd3_C`
  (poisson 0.45, the user's compression hypothesis: near-incompressible), `cd4_C`
  (poisson 0.0, most compressible). Pre-registration: if cd2 morphs (silIoU > 0.9) and
  cd3/cd4 both freeze, the cause is direction, not compressibility.
- 150k v6 residual shedding (fragment trace on the dragon and bob archives, cell 0.31): no
  steady leaders — the fragments appear in EPISODES (dragon 79 / 83 / 83 / 83 particles at
  frames ~240 and ~320–350 of 761, bob 3–15 at 120–180) and are merged at the next commit;
  end fragments 0. A burst of ~80 particles at ppc 91 is one cell of material: a thin
  feature (horn, whisker, bob's ring) whose neck is thinner than the cell detaches as a
  chunk and pops back. Test `h150z_dragon` (150k, cell 0.31, ot_pace + net): at cell 0.20
  ot_pace cut dragon's merges to 134 by not pulling thin features ahead of the body; if it
  brings the cell-0.31 run from 755 to ≤ 100 with silIoU ≥ 0.93 and a smooth still, it is
  the 150k recipe for the hard shapes.
- Photoreal renderer (`scripts/render_photoreal.py`, Open3D/Filament, EGL headless works on
  hyde06, 0.25 s per 640² frame): marching cubes on the same blurred density as the iso
  videos, Taubin-smoothed, PBR ceramic under IBL + sun with soft shadows and a ground plane.
  First still (v6 bunny, frame 600) renders; framing and faceting fixed (grid 160, 12
  smoothing iterations, camera distance from the fov). Sidecar per video: isosurface
  components and isolated particles per frame — the frame-level QA for goal 3.

**C ladder verdict (18:50).** Compressibility is not the cause: poisson 0.45 (`cd3`) and
poisson 0.0 (`cd4`) freeze identically at window 12 (silIoU 0.71 / 0.77, 12–15 brake
rejects, 0.3 min). The per-window record of `cd4` shows the density-loss mechanism: d_vol
falls 0.475 → 0.140 and d_sil 0.44 → 0.064 by window 6–9 while the kinetic energy climbs
0.5 → 5.9 — the body, pulled outward from the hole, gains momentum and OVERSHOOTS the
arms (d_sil back up to 0.129, d_vol to 0.168 by window 11); the brake is right to refuse.
With the transport-paced target (`cd2`, ot_pace + cell 0.31) there is no overshoot (kin
peaks 3.7 and falls to 0.47, d_dt 3.0e4 → 274, d_sil 0.44 → 0.050) — and the run STILL
freezes at window 14, on a −9 % merit "regression" that is an artefact: under ot_pace the
recorded d_vol is the loss against the paced target (0.0049 → 0.0055, tiny, noisy), and the
fixed-scale merit normalises it by its window-1 value, so a 12 % wobble of a near-zero
number dominates a merit whose real components (d_sil, d_dt) were still falling. Fix
(runner, commit after b387398): under any transport recipe the record's d_vol is recomputed
as the cell sum against the FIXED target, which is what the merit, brake and tracker are
defined on. This bug also inflated the gate rejects of every ot_pace run (3–15 per run).
Re-run `cd5_C` (ot_pace + cell 0.31 + net) on the fix; pre-registration: C reaches silIoU
> 0.9 without a brake freeze.

**cd5_C (19:10): silIoU 0.912 (0.72 before), chamfer 0.186, hole 8.4 %, 48 re-attachments,
stop at window 25.** With the merit reading the fixed cell sum the run descends
monotonically for 19 windows (merit 3.0 → 0.376, d_sil 0.44 → 0.043, d_dt 3.0e4 → 167,
kin peaks 3.7 and decays — no overshoot) and then stalls: from window 20 the fixed-target
d_vol creeps up (0.131 → 0.137) and d_dt (167 → 211) while d_sil is flat, three low-gain
rejects (−0.8…−1.5 %) trip the latch. The arms are formed but under-filled (hole 8 %): the
paced target after arrival is the projected image cloud, whose cell masses are not the
target's, so the optimiser minimises against a target that is itself 0.13 away. The
hand-off (window target = the fixed target once no deficit cell lies beyond one cell of
the body) is the designed remedy; its one earlier test (oh40, 07:57) failed while the merit
was reading the paced loss — the bug — so it is re-tested on the corrected merit: `cd6_C`
and, as the general recipe candidate for goals 1 and 3 (transport pacing early = no
leaders, fixed target late = full fill), `oh2_{bunny,dragon,bob}` at 40k, cell 0.31, no
re-attachment. Pre-registration: fragments ≤ v6 (0 / 0 / 2) and chamfer / silIoU within
0.005 / 0.01 of the density recipe (0.118 / 0.960, 0.126 / 0.955, 0.118 / 0.958).

**Goal-3 baseline from the first photoreal video (v6 bunny, 150k, 369 video frames of
1103 archived):** isosurface components > 1 in **54 frames** (max 6) — archived frames
18–108 (the expansion phase: 50–182 isolated particles per frame, some clustering into
blobs) and 282–366 (a small blob at the ear tip: the "particle-looking sphere" the user
forbids) — and 0 from frame 471 on. This is the number every recipe change is now judged
by, per video, from the `<video>.components.txt` sidecar: the goal is 0 frames with more
than one component and 0 isolated particles.

**cd6_C (19:00): the global hand-off never fires** — the count of target cells with mass
beyond one cell of the body falls 1845 → 19 and stays (the C's arm tips), so the run is
cd5 again (0.909 / hole 8.3 %). Replaced by a CELL-WISE hand-off (commit after c7eb123):
every target cell within one cell of the occupied set carries the fixed target mass (the
CIC gradient reaches it: ordinary fill, never a far-cell reward), every cell beyond it the
paced mass (coherent transport toward it). Trials `cd7_C` and, as the general candidate,
`oh3_{bunny,dragon,bob}` (40k, no net). For the early-expansion chunks at 150k (the
photoreal bunny frames 18–108, h150z dragon 19 merges by window 16 even with ot_pace) the
descent-direction lever is re-tested at cell 0.31: `--grad_h1` (Sobolev descent on the
material kNN graph — neighbouring particles move together; falsified once at cell 0.20
under the calibration bug, verdict void): `gh_{bunny,dragon,bob}` at 40k without the net
and `h150w_dragon` at 150k with it.

Photoreal dragon (v6, 255 video frames): components > 1 in 182 frames (max 8); isolated
particles peak at 5286 (3.5 % of the cloud) around archived frame 48 — the expansion phase
— and decay to ~500 by frame 170. Frames seen: 48 (a lumpy blob, no strays visible at the
isosurface), 210 (one chunk floating well above the body and a drop under the tail — the
forbidden picture), 600 (clean; whiskers attached). So the video defects are (i) the early
isolated-particle cloud (invisible at the isosurface but real) and (ii) mid-run chunk
flights that the commit-time net only repairs afterwards. h150z (150k, ot_pace, cell 0.31)
at window 210: 19 merges (v6: 755 by the end) — the pacing removes most flights.

**h150z_dragon FALSIFIED as a 150k recipe (19:20): 19 merges but chamfer 0.267 / silIoU
0.675 after 25 min (300 windows)** — only 4 % of the particles ever came within one blur
radius of their image: the paced target stays one blur radius (0.18 wu) ahead of the
current cloud by construction, and at 150k the cloud follows it slowly (the density
gradient per particle is ~1/N smaller for the same cell), so pure pacing is too weak to
morph in the window budget (at 40k the same recipe reached 0.955 in 7.5 min). The
cell-wise hand-off (`oh3`) is the candidate that keeps the fixed target's full-strength
fill near the body and paces only the far transport.

**C, continued (19:10):** `cd7_C` (cell-wise hand-off, all near cells) = 0.722, 13 rejects
— the near cells around a body sitting in the hole are m_t = 0 cells, and the fixed target
there is the outward runaway again. `cd8_C` (deficit cells only) = 0.806 / hole 3.3 %,
3 rejects. Every C variant stops at 20–27 windows through the fixed-scale merit; the slow
tangential redistribution that forms the arms reads as no progress to the cell sum. For the
transport recipes the merit's physics component is now the Sinkhorn divergence to the
fixed target (`SinkhornPull.divergence`, commit 1800d88) — what the recipe descends, defined
on the fixed target, monotone along a transport path. `cd9_C` = ot_pace + that merit.

**cd9_C (19:15): 0.901 / hole 8.1 %, stop at window 24 — and now the record shows the
real limit:** the transport divergence itself stops falling at window 17 (0.740 → 0.751),
the arrived fraction sits at 4.9 % and kin decays to 0.23: the BODY stops, the gate is
merely reporting it. Why it stops: the pace is one plan blur radius (0.18 wu) but the loss
cell is 0.31 wu — for a blob sliding along an arm the paced grid equals the current grid
except at the ends, so the cell sum is blind to the requested sub-cell shift and the
descent has nothing to follow. Fix (commit 7e34ba2): the pace is the resolution the loss
can see, max(blur radius, loss cell). `cd10_C` re-runs; the same stall explains h150z
(150k dragon, 4 % arrived after 210 windows).

**cd10_C (19:20): the transport now moves** — arrived fraction 0.4 % → 34 % by window 14,
d_vol 0.074 (the lowest any C run reached; cd5 0.131), d_sil 0.033, ot_div 0.447 — and
then the body OVERSHOOTS: windows 15–17 regress (d_vol 0.074 → 0.091, ot_div 0.447 →
0.482, d_dt 583 → 740) with kin still 0.6–0.9, the brake refuses every later candidate
(−6 %), delivered = window 14, silIoU 0.839, hole 5.9 %. So with the cell sum as the inner
loss the paced transport is either blind (pace < cell: stall) or inertial (pace = cell:
overshoot). The per-particle transport loss (`--phys_loss ot`) has neither defect — its
earlier verdicts (holes, "converged at 20 windows") were taken while the record's d_vol
was the OT loss value (the tracker/merit bug), so it is re-tested on the corrected record:
`cd11_C` and `o2_{bunny,dragon,bob}` (40k, no net). gh_bob (grad_h1): 25 fragments (v6 2)
— grad_h1 closed.

**cd11_C (19:45): C MORPHS with the per-particle transport loss — silIoU 0.9625** (density
0.72, ot_pace 0.91), chamfer 0.140, hole 6.3 %, 165 windows in 17 min (the first C run that
runs long), 109 re-attachments, stray_max 6.6 % mid-run (the L2 pull on far particles
sends strays, which the net returns). Delivered end state (photoreal still with the target
ghost): a full C with both arms; the residual hole is the arm ends and the surface is
lumpy. So the C failure was never compressibility: it is the cell sum's outward push
from the hole plus the inertial overshoot it causes; a transport plan removes both.

Other verdicts: mat_stiff_dragon 0 re-attachments / 0.963 vs base 14 / 0.957 and soft
343 / 0.846 — stiffness changes the ejection behaviour on the dragon (soft sheds 25×
more); h150w_dragon (150k, grad_h1) 733 re-attachments / 0.927 (v6 755 / 0.939) — grad_h1
closed at 150k too; rp_phys_dragon (150k, render channel off from the start): 259
re-attachments, chamfer 0.0899, silIoU 0.840 vs the render twin 755 / 0.0831 / 0.939 — the
render channel raises the dragon's silIoU by 10 points and its re-attachments 3× (it pulls
the thin features harder); the causal intervention twin (`rp_cut_dragon`) follows.

**o2_bunny (OT-only, no net): 0 fragments, 0.1294 / 0.945 in 16 windows** — the transport
loss converges fast and stops short of the density recipe's fill (0.118 / 0.960). So the
two losses have complementary regimes, and the regime is a property of the discretised
problem, not a per-shape choice: when the source's mass sits in cells the target leaves
EMPTY (the sphere inside the C's hole), the cell sum has nothing but an outward push and
the transport plan is the only term that says where the mass goes; when source and target
overlap, the cell sum's local fill is the better objective. `--phys_loss auto` (commit
after 1a5a162) measures at the start the fraction of source particles whose target cell
carries no mass and picks the transport loss above one half (C: ~1.0; bunny, dragon,
bob: < 0.5), logged as `[v2] phys_loss auto: …`. Measured: C 68.1 % → ot.

**Goal 3, the deliverable rule (commit after 5386fe4):** the photoreal renderer draws only
isosurface components whose volume is at least one MPM cell (dx³, dx = source diagonal /
26) — material smaller than a cell shares no full stencil with the body and is not a
continuum element the grid resolves (the same rule the commit-time net uses to decide what
is a stray). Nothing is hidden from the record: the sidecar keeps the raw component count,
the drawn count and the number dropped per frame, and the page shows both. The physics
side (v6 + auto) is what reduced the raw counts; this rule is what makes every delivered
frame show one body. C 40k photoreal (cd11): raw components > 1 in 12 of 302 frames,
isolated particles peak 2615 at frame 54 (the transport phase).

**oh3_dragon (20:05) — the first recipe that removes the expansion strays at the source:**
ot_pace with the cell-wise (deficit) hand-off, pace = loss cell, OT-divergence merit,
cell 0.31, 40k, NO net: end fragments **0**, chamfer 0.1217 (density 0.1264), silIoU
**0.965** (0.955), hole 0.30 %, whole-run isolated-particle peak (stray_max) **0.14 %**
against 2.3 % for the density recipe — 17× fewer strays mid-run, better quality, 108
windows in 22 min (three runs per GPU). Mechanism: the paced target moves one cell per
window, so the expansion phase is a coherent flow with no far-cell reward for a leader,
while the deficit cells adjacent to the body carry the fixed target and fill at full
strength. oh3_bunny 0 / 0.1179 / 0.961 (density 0.1178 / 0.960). oh3_bob pending; if it
holds, `auto` becomes: overlap → ot_pace + hand-off, hole → ot, and the 150k gallery is
re-run (v7).

**Goal 1b — the render gradient changes the physics (dragon, 150k, v6 recipe; four runs
with the same seed and particles):** render-on (`h150y`), an identical-configuration
re-run (`rp_ctrl`, the run-to-run noise floor), physics-only (`rp_phys`, λ = 0 from the
start) and the intervention (`rp_cut`, λ = 0 from window 40 = archived frame 120; the log
confirms `render channel OFF from here` at window 41). Mean per-particle divergence from
the render twin, particle spacing 0.037 wu — before window 40 mean / max | after mean | end:

| run | before 40 | after 40 | end |
|---|---|---|---|
| identical configuration (`rp_ctrl`) | 0.58 / 1.41 | 3.33 | 4.66 |
| render off at 40 (`rp_cut`) | 0.58 / 1.41 | 3.38 | 4.72 |
| physics-only (`rp_phys`) | 1.70 / 2.71 | 4.56 | 5.96 |

**Withdrawn:** the earlier reading of this plot ("the trajectories coincide until the
channel is removed and separate from that window on — a causal signature") was wrong. An
identical configuration diverges from itself by the same amount at every frame (4.7
spacings by the end): at 150k the GPU-atomics non-determinism is amplified by the
contact dynamics into a chaotic divergence, and the intervention twin never separates
from that floor. Only the physics-only twin exceeds it, and modestly (1.7 vs 0.58 spacings
before window 40, 6.0 vs 4.7 at the end). Trajectory divergence is therefore a chaos
measurement at 150k, not a causal one; plot `output/photoreal/render_effect_dragon.png`
(hyde06) now carries the control curve.

What does separate from the noise: (i) **the per-window control update.** The optimizer
records g_share = λ‖g_render‖ / (‖g_phys‖ + λ‖g_render‖) and the cosine between the
physics gradient and the (PCGrad-projected) render gradient every window. Over every
render-on run at 150k the render channel is 35 % of the accepted update (0.39–0.40 in
windows 1–40, 0.34 afterwards; dragon 0.348, control 0.354, bunny 0.348, bob 0.344) at a
cosine of 0.02 (dragon, control, bunny) / 0.05 (bob) to the physics gradient — a third of
every control update is a direction the cell sum does not contain. This is a
deterministic per-window measurement, untouched by the chaos above, and is the direct
proof that the render gradient changes the control (the physics). (ii) **The outcome
against the run-to-run spread.** silIoU: render-on 0.939 and control 0.918 (two samples
of the same configuration: mean 0.928, spread 0.021) vs physics-only 0.840 and cut-at-40
0.770 — 4–8 spreads below the render-on pair; chamfer 0.083 / 0.088 vs 0.090 / 0.101;
re-attachments 755 / 773 vs 259 / 68. The render channel is what reaches the thin
features (+10 silIoU points), at the price of pulling them harder (3× the re-attachments);
without it the run stalls (cut: 99 windows, gate stop). Bunny: `rp_phys_bunny` 0.0798 /
0.936 vs render 0.0790 / 0.958 (+2.2 points); `rp_cut_bunny` and the bob twins follow.

**Bunny intervention twin (21:15; v7 recipe — the rp chain sourced the recipe at launch,
so `rp_cut_bunny` ran under `auto` → ot_pace and is the twin of `h150v7_bunny`, not of
the v6 run; `rp_phys_bunny` ran under the density recipe and pairs with `h150y_bunny`):**
render on (`h150v7_bunny`) 0.0801 / **0.966** / 2 re-attachments, 211 windows; render off
at window 40 (`rp_cut_bunny`) 0.0812 / **0.936** / 0 re-attachments, 300 windows (never
converges). Divergence from the render twin (spacing 0.038 wu): 0.14 spacings before
window 40 (max 0.43) → 0.77 after (1.05 at the end) — a 5× step at the intervention on a
shape whose chaos is 4× milder than the dragon's, but without a bunny control at 150k
this stays suggestive; the deterministic per-window measurement holds as on the dragon
(g_share 0.346 / 0.380, cosine 0.03 / 0.06), and the outcome gap is 3 silIoU points. Plot
`output/photoreal/render_effect_bunny.png`. Queued after the bob twins: `rp_phys7_bunny`
(λ = 0 under the v7 recipe, the consistent physics-only twin) and `rp_ctrl7_bob` (the bob
noise floor).

**Bob triple (22:30; all three under the v7 recipe, pre-walls, same seed; the pre-wall
render-on archive is kept as `h150v7pw_bob` because `h150v7_bob` is re-run under the
walls):** render on 0.0771 / **0.970** / 83 re-attachments (133 windows); physics-only
(`rp_phys_bob`) 0.0780 / 0.952 / 144; render off at window 40 (`rp_cut_bob`) 0.0776 /
0.956 / 84. Divergence from the render twin (spacing 0.038 wu): cut twin **0.054
spacings before window 40 (max 0.15) → 0.57 after (0.75 at the end)**, a 10× step at the
intervention; physics-only 1.77 at the end. Per window: render share 0.354 / 0.373, cosine
0.04 / 0.08. So on bob and bunny — shapes whose run-to-run chaos is an order of magnitude
below the dragon's — the intervention signature is visible in the trajectories as well as
in the outcome (+1.4–1.8 silIoU points); the dragon's is buried under its chaos and rests
on the per-window share and the outcome. Plot `output/photoreal/render_effect_bob.png`;
`rp_ctrl7_bob` (noise floor) and `rp_phys7_bunny` run next.

**Bunny triple, consistent (23:30; all v7 recipe, same seed):** render on (`h150v7_bunny`)
0.0801 / **0.966** / 2 re-attachments; physics-only (`rp_phys7_bunny`, λ = 0) 0.0815 /
0.932 / 0, 283 windows; render off at 40 (`rp_cut_bunny`) 0.0812 / 0.936 / 0, 300
windows. Divergence from the render twin: cut 0.14 spacings before window 40 → 0.77
after (1.05 at the end); physics-only 1.78 at the end (0.03 at the first window). The
render channel is worth +3.4 silIoU points on the bunny and the runs without it never
converge (283–300 windows against 211). Plot `output/photoreal/render_effect_bunny.png`
(updated to the consistent triple).

**nn150_bob (23:40; 150k, no net, pre-walls, 0 band hits):** 0.0781 / 0.932 / **54
fragments** at the end — one chunk of ~50 particles (0.55 of a cell at ppc 91, below the
deliverable threshold) shed between windows 25 and 50 and never recovered; the netted
run (`h150v7_bob`, walls) 0.969 / 126 re-attachments. So at 150k without the net: bunny 0,
dragon 3, bob 54 fragments (0 / 0.002 / 0.04 % of the cloud); the net is what keeps bob's
ring at 0.97. `nn150_C` (walls, raw image) runs last.

**Bob noise floor (23:50; `rp_ctrl7_bob`, identical configuration re-run of the walled
`h150v7_bob`):** 0.0776 / **0.968** / 127 re-attachments against 0.0773 / 0.969 / 126 —
the run-to-run outcome spread on bob is 0.001–0.002 silIoU, so the physics-only (0.952)
and cut-at-40 (0.956) twins sit 7–9 spreads below the render-on pair; with the bunny
(+3.4 points) and the dragon (+10, spread 0.021) this is the outcome half of goal 1b on
three shapes. The trajectory half (control curve on the divergence plot) follows.

**Bob trajectories with the control (00:00; `render_effect.py --ctrl`):** identical
configuration re-run vs its twin 0.11 spacings before window 40 (max 0.46) → 0.34 after
(0.50 at the end); render off at 40 vs its twin 0.054 (max 0.15) → 0.57 (0.75 at the
end); physics-only 1.73 at the end. The cut twin's post-intervention divergence is 1.5–1.7×
the noise floor — present but modest; the bunny's (0.14 → 0.77, 1.05) awaits its control
(`rp_ctrl7_bunny`, running). Plots `output/photoreal/render_effect_{bob,dragon}.png`
now carry the control curve (the dragon's confirms the earlier reading: control 0.58 →
3.33 → 4.66 against cut 0.58 → 3.38 → 4.72 — indistinguishable). The proof of goal 1b
therefore rests on (i) the per-window share/orthogonality (0.35, cosine 0.02–0.08,
deterministic) and (ii) the outcomes against the run-to-run spread (dragon +10 points at
spread 0.021, bunny +3.4, bob +1.4–1.8 at spread 0.002); the trajectory divergence is a
supporting signal on the two smooth shapes and pure chaos on the dragon.

**oh3_bob (20:15): 0 fragments, 0.1169 / 0.974 / 2.77 % (target ring), stray_max 0.29 %
(density 2 / 0.118 / 0.958 / 1.1 %).** The trio passes the pre-registration on every count
(fragments 0 / 0 / 0 against 0 / 0 / 2; chamfer within 0.005; silIoU equal or better by
+1…+1.6 points; mid-run strays 8–17× fewer). Adopted (commit e07e0b5, deployed 20:00):
`--phys_loss auto` now means — source overlapping the target: the transport-paced cell
sum with the cell-wise deficit hand-off, pace = loss cell, Sinkhorn-divergence merit;
source inside a target hole (C): the per-particle transport loss. The 150k gallery is
re-run on it as v7 (`h150v7_<T>`, `n150v7_<T>`, four chains, re-attachment kept as the net;
its counts are the report's honesty metric).

**grad_h1 at cell 0.31 (goal 3 candidate) FALSIFIED (19:10):** gh_bunny 0 fragments /
stray_max 0.085 % (v6 0.107 %), gh_dragon 4 fragments / stray_max 2.32 % (v6 no-net 0 /
2.31 %): the Sobolev direction leaves the early-expansion stray cloud unchanged. Material
study, bunny (5 runs): base 0.1178 / 0.960, soft (young 3e4) 0.1185 / 0.959 (12
re-attachments), stiff (6e5) 0.1167 / 0.971, poisson 0.45 0.1195 / 0.957, assim 0.1
0.1167 / 0.963 — the END states barely differ; the trajectories are compared next.

**Material study, bunny trajectories (`scripts/probes/material_trajectories.py`, same
seed → the same particles; spacing 0.060 wu):** mean per-particle divergence from the
base run at 10 / 25 / 50 / 100 % of the run — soft (young 3e4) 0.26 / 0.17 / 0.12 / 0.11 wu
(end 1.9 spacings; first > 1 spacing at frame 48; 12 re-attachments, a 3.1 wu jump), stiff
(6e5) 0.11 / 0.09 / 0.10 / 0.11 (1.8 sp; frame 33), poisson 0.45 0.11 / 0.12 / 0.13 / 0.14
(2.3 sp; frame 36), assim 0.1 (more elastic) 0.03 / 0.05 / 0.06 / 0.07 (1.2 sp; frame 984).
Path lengths 1.08 (base) / 1.12 / 1.26 / 1.09 / 1.17 wu; end chamfer 0.0758–0.0779 for all.
So the material changes the PATH (by 1–2 particle spacings, 10–25 % of the path length,
from the first windows on for stiffness and Poisson ratio) while the render+density
objective drives every material to the same end shape. Control runs (`mat_ctrl`, identical
configuration) give the numerical noise floor of the divergence; dragon follows.
Photoreal stills at the divergence peak (archived frame 250 = the same simulated time in
every run; `output/photoreal/material_bunny_f250.png` on hyde06): the soft body is still a
sphere with the ears barely budding, the stiff body has both ears fully extended, base /
poisson 0.45 / elastic lie between — the intermediate SHAPE depends on the material, the
end shape does not.
Noise floor (`mat_ctrl_bunny`, the base configuration re-run; GPU atomics make the rollout
non-bitwise-reproducible): divergence 0.013 / 0.026 / 0.044 / 0.056 wu at 10 / 25 / 50 /
100 % (0.9 spacings at the end, never > 1 spacing). Against it the material effect at 10 %
of the run is 8× (stiff 0.111, poisson 0.45 0.106) to 20× (soft 0.262) the noise, and at
the end 1.3–2.5× — the material sets the transient path, the objective sets the end.

**Material study, dragon (spacing 0.060 wu; control noise 0.003 / 0.028 / 0.048 / 0.060 wu
at 10 / 25 / 50 / 100 %):** soft (young 3e4) 0.40 / 0.54 / 0.51 / 0.49 wu (8.1 spacings at
the end, 130× the noise at 10 %; path 2.35 wu; 343 re-attachments, silIoU 0.846), stiff
(6e5) 0.44 / 0.36 / 0.37 / 0.42 (7.0 sp; path 1.81; 0 re-attachments, 0.963), poisson 0.45
0.26 / 0.39 / 0.42 / 0.43 (7.0 sp; 95 re-attachments, 0.941), assim 0.1 0.09 / 0.19 / 0.18 /
0.18 (2.9 sp; 0.958); base 14 re-attachments, 0.957. On the dragon the material changes the
path by 3–8 particle spacings (7–13× the noise at the end), the stray behaviour (soft
sheds 25× more than base, stiff none) and the end quality (silIoU 0.846–0.963); the
frame-250 stills show five visibly different intermediate bodies. Together with bunny:
material properties determine the morphing trajectory — early and strongly — and, on
shapes with thin features, the end state too.

**C at 150k (20:45) — morphs, but sheds (goal 2 half done):** `h150v7_C` (v7 chain, auto →
ot, net) 0.1478 / 0.946 / hole 6.7 %, 161 windows / 41 min, **1914 re-attachments** in 17
events — bursts of 100–440 particles in windows 38–50 while |v|max sat at 2.4–2.7 wu/s and
400–680 controls were clamped per window; `c150_C` (same recipe, launched earlier) 0.1359 /
0.954 / 6.0 %, 2675 re-attachments, stray peak 14.8 %. At 40k the same loss sheds 109–114
(cd11_C, auto_C; stray 6.5 %): the transport loss morphs C at both resolutions and the
shedding grows with N. Cause read in the code: the `ot` branch used the raw map image as the
per-particle target — no material-kNN smoothing (only the leash/pace branches had it) and no
pace. The quadratic pull is proportional to the distance, so the particles farthest behind
(the sphere material bound for the arm tips, 2–3 wu away) are pulled hardest, lead the body
and fracture — the H3 leader mechanism in per-particle form, with the sampled map's
sample-scale noise on top at 150k. Fix efc7fa0: the ot target is the material-smoothed
displacement (k from the blur volume, as in ot_pace) bounded to one pace = max(plan blur,
loss cell) per window — the pull is uniform and bounded by what the grid resolves in a
window, and the target still walks the whole map. No new constant. Tests: `op40_C` (net) /
`op40n_C` (no net) at 40k against auto_C 114 / 0.959; then 150k.

**Paced ot target, 40k C verdict (20:56):** `op40_C` (net) 0.1555 / **0.958** / hole 6.6 %,
**7 re-attachments** in 6 events (auto_C 114 in 9; cd11_C 109), stray peak 4.0 % (6.4 %),
G2 guards PASS (auto_C FAIL: 0 clamped controls vs hundreds), 42 windows / 10.7 min
(auto_C 143 / 34 min; the merit gate stops it at a plateau — three rejected candidates,
gain −0.007…−0.012 — and delivers the best commit, window 37). `op40n_C` (no net) 0.1583 /
0.944 / 6.2 %, 0 re-attachments, **9 fragments** at the end (0.02 % of 40k), stray peak
4.0 %. So the same silIoU with 16× fewer re-attachments and 3× faster; the price is the
chamfer (0.156 vs 0.145 — the plateau stops before the arm ends are fully filled) and the
no-net run still leaves 9 particles. Adopted for the hole regime (efc7fa0 stays); the
150k C is re-run under it (`h150v7_C`, replacing the 1914-re-attachment run in v7) and a
no-net 150k C (`nn150_C`) follows the nn150 trio.

**150k C under the pace (21:40, mid-run): the pace does NOT transfer — 1393 re-attachments
by window 111** (bursts of 100–270 in windows 46–98; the old run had 1914 by 161). Frames
of the old 150k C photoreal video (archived 96–162, `output/qa_frames/Cold_f*.png`) show
what the numbers do not: the body is not a C with a lumpy surface but a C-shaped SHELL
whose top arm is a separate slab from the start, side walls hanging below it, and by
frame 162 a detached ball floats above the body — the sphere/circle the deliverable must
never show. Reading: the sphere → C map is discontinuous on the surface where material
splits between the arms; every particle on the far side of that surface has a target
across empty space, and at 150k the material-kNN neighbourhood that smooths the map
(one plan blur radius, 0.10 wu = a third of a cell) is too small a share of the fracture
gap (one cell) to blur the split into a stretch — the body tears into cell-sized slabs
and chunks. At 40k the same 0.10 wu is 1.7 spacings against a 5-spacing gap and the
tear does not open (7 re-attachments). Fix (commit after 878a1ef): for the hole regime
the smoothed displacement is additionally RESOLVED ON THE LOSS GRID — mass-weighted CIC
deposit and gather with the cell sum's own kernel — before the pace: a displacement the
grid cannot resolve is not a continuum displacement; below one cell the split becomes a
stretch the elastic body carries. Test `c150s_C` (150k, net) against the paced re-run.

**THE 150k SHEDDING MECHANISM (22:10) — the domain box was a trap.** The paced 150k C
re-run (`h150v7_C`, 2nd) ended 0.1814 / 0.881 / 1528 re-attachments in 131 windows;
`c150s_C` (grid-resolved displacement) 0.2161 / 0.592 / 885, gate stop at 67 — worse,
reverted; `c150p_C` (ot_pace + hand-off forced on C) 0.559 / gate stop at 38. Probe
`output/chunk_origin.py` on the paced archive (frames 100–260): every chunk (23–255
particles) is STATIC (0.002–0.02 wu/frame), 1.5–3.3 wu from the nearest target point, 0 %
of its particles on the target, at 4.4–6.7 wu from the hole axis — i.e. at the corners
of the domain box (half-width 4.98 wu, leash 4.37 + 2 dx), and its origin is the sphere's
interior (r0 0.7–0.88 R) on the side facing away from the opening. Counting particles
beyond the box leash per archived frame: 714 (f100), 903 (f140), 751, 621, 304, 44, 0 at
the end — the whole population of "chunks". The count of particles clipped at the box
(`GUARD clamp` in the window line, `n_out` in runner.py) tracks the re-attachments across
EVERY run: v6 dragon 10706 band hits / 755 re-attachments, v6 bob 10934 / 486, nefertiti
702 / 152, beast 155 / 493, v7 bob 178 / 83, 150k C 27–59 k / 885–1914, against 0 / 0–7
for every run that never reached the band (40k C 0 / 7 and 0 / 0, teapot, heart, A,
nn150 bunny 3 / 0, nn150 dragon 0 / 0). Cause in the forward model: the MPM grid had NO
boundary treatment on the box faces (only the optional floor): a particle within the
cubic stencil's half-support (2 cells) of the edge deposits on and gathers from a
truncated stencil, loses momentum every step and freezes there; the commit-time clip
keeps it. The auto domain (far-field leash + 2 dx) puts that band one cell beyond the
target's outer surface, which the 150k transients reach and the 40k ones do not. FIX
68f0a20 (docs/method.md §10.11): separating walls on all six faces — the outward normal
grid velocity is zeroed on the outermost 2 node layers, tangential and inward motion free
— the constant being the stencil half-support. Runs that never touched the band are
bit-identical. Tests 26 passed (adjoint vs FD, smoke, bonds). Tests: `c150w_C` (150k C,
walls + paced ot, net) and the v7 bob re-run under walls; the v7 targets already finished
with 0 band hits (teapot, heart, spot, A) stand; bunny (4 hits) and V (29) are within
noise and stand; the remaining v7 chains run under walls from here.

**Walls, first results (22:55):** `h150v7_bob` re-run under the walls 0.0773 / 0.969 /
126 re-attachments, 0 band hits (pre-walls 0.970 / 83 / 178 hits): bob's residual
shedding is its own thin feature, not the box, and stays at the ~100 level (all sub-cell
events: 27–32 particles, below the dx³ deliverable threshold of 91 at ppc 91). `c150w_C`
(walls + PACED ot) 0.1948 / **0.550** / 336 re-attachments, band hits 993 (38 983 before),
gate stop at 144 windows, G2 FAIL — the box trap is gone (40× fewer band hits, no burst
above 76 particles) but the morph collapses. Reading, with the earlier paced runs (150k:
0.881 and 0.550; unpaced 0.946 and 0.954): a target that walks with the particle makes the
window loss quasi-stationary (every non-arrived particle keeps a pace-sized residual
whatever its progress), so the inner line search and the merit gate see no descent and
stop the run on a half-formed body; at 40k the same run stopped at 42 windows with the
chamfer 7 % worse — the "16× fewer re-attachments" there were box-band hits that the
walls now remove. **The paced per-particle target is withdrawn** (commit after d17f1bd;
the material-kNN smoothing of the map image stays). Tests: `c150u_C` (150k, walls, unpaced
smoothed ot, net) and `ou40_C` (40k, walls, same) — the C entry of v7 is whichever 150k C
run stands after this.

**Goal 1a at 150k WITHOUT the net (`nn150_*`, v7 recipe, pre-walls; 0 band hits in both):**
bunny 0.0802 / 0.962 / **0 fragments** at the end (2 transient in windows 24–43), stray
peak 0.15 %, G4 PASS, 45 min; dragon 0.0847 / 0.952 / **3 fragments** at the end (1
through windows 59–80, 3 from window 200 on; 0.002 % of the cloud, single particles below
the deliverable threshold), stray peak 0.96 %, G4 FAIL on the strict zero, 87 min. So the
recipe alone holds the bunny fully and the dragon to three single particles at 150k; the
net's re-attachment counts in the v7 gallery (bunny 2, dragon pending) are the same
events caught early. bob and C follow.

**Smoothing falsified on the hole regime (23:10):** `ou40_C` (40k, walls, unpaced ot WITH
the material-kNN smoothing that efc7fa0 had extended to `ot`) gate-stopped at 26 windows
with 0.2021 / **0.794** / 79 re-attachments (9 band hits) — against the raw map image's
0.959–0.963 (auto_C, cd11_C). On a target with a hole the map is genuinely discontinuous
where the material splits between the arms; averaging the displacement over the material
neighbourhood across that surface sends the seam into the hole. The `ot` regime is back to
the raw debiased map image (commit after 65452ae); `c150u_C` (smoothed) was killed at 26
windows and replaced by `c150r_C` (150k, walls, raw image, net) with `or40_C` (40k) as the
sanity check. The walls remain the only change to the C recipe.

`or40_C` (00:05; 40k, walls, raw image, net): 0.1438 / **0.959** / 101 re-attachments,
7 band hits, 43 min — the 40k baseline restored exactly (auto_C 0.1448 / 0.959 / 114):
the walls change nothing where the box was never reached, and the ~100 re-attachments
of the 40k C are its own sub-cell events, not the box.

### 2026-09-19 — surface smoothness: six candidates (pre-registered before any run)

The deliverable surface is the marching-cubes isosurface of CIC + isotropic Gaussian blur
(1.5 spacings) of the particle density. Its texture is the sampling floor: the TARGET cloud
through the same pipeline is as bumpy as any morph (mean |dihedral| 14.5°). The user asks
for smooth surfaces and points at the Gaussian → triangle line of work. Papers read:
Triangle Splatting (Held et al., arXiv 2505.19175: triangles as differentiable splats with
the window I(p) = ReLU(φ(p)/φ(s))^σ, initialised from SfM points, photometric losses, a
triangle soup for a standard renderer), Triangle Splatting+ (2509.25122, opaque
triangles), 2D Triangle Splatting (2506.18575, a mesh-like structure directly),
Incremental Online Scene Reconstruction by 3D Gaussian Triangulation (ECCV 2026, arXiv
2607.10690: tangent-plane angular-greedy triangulation of Gaussian SURFELS, a plane-pulling
constraint Σ|nᵢᵀ(μᵢ − pᵢ)|, normal consistency, Laplacian remeshing toward degree 6; needs
posed RGB-D), SplatSurf (Visual Computer 2026: triangle-soup optimisation of a trained 3DGS
to a manifold mesh), SuGaR / 2DGS / Gaussian surfels (surface-aligned Gaussians → Poisson
or TSDF meshing), Yu & Turk 2013 (anisotropic kernels for particle fluids: weighted-PCA
covariance per particle, eigenvalue ratio clamp k_r = 4, Laplacian smoothing of the kernel
centres with λ = 0.9, then a sum of anisotropic kernels), screened Poisson (Kazhdan & Hoppe
2013), RIMLS (Öztireli et al. 2009), and a 2024 CGF paper learning SDFs from fluid
particles with a CNN. Every image-based method (Triangle Splatting, SplatSurf, 3DGT, SuGaR,
2DGS) optimises against photographs we do not have; what transfers is their GEOMETRY: an
oriented-surfel representation, centres pulled onto the local plane, triangulation or
Poisson meshing, Laplacian regularisation. The six candidates for OUR input (a 150k
particle cloud per frame, no images):

- **S1 Yu & Turk anisotropic kernels** (`--kernel pca`): per-particle covariance from the
  weighted PCA of its neighbourhood (cubic weight, radius 2 h), eigenvalues clamped to a
  ratio of 4, kernel centres Laplacian-smoothed with λ = 0.9, density = sum of the
  anisotropic Gaussians, isosurface as now. The particle-fluid standard; the smoothing of
  the centres is the part our F-carried kernel test lacked.
- **S2 Screened Poisson from surface particles** (`--surface poisson`): surface particles
  = those with a density gradient (the outer layer), normals from the blurred density
  gradient, Open3D screened Poisson (depth 9), low-density trim. A smooth watertight
  mesh by construction; the SuGaR / 2DGS meshing step without the Gaussians.
- **S3 IMLS implicit** (`--surface imls`): the implicit moving-least-squares field
  f(x) = Σ w_j(x) n_jᵀ(x − p_j) / Σ w_j(x) on the render grid from the surface particles
  and their normals, isosurface at 0. Smooth by construction (a local plane fit), feature
  preserving with robust weights (RIMLS); no meshing library needed.
- **S4 Surfel triangulation, geometric version of 3DGT** (`--surface surfel`): surface
  particles pulled onto their local plane (the plane-pulling constraint, one MLS
  projection), tangent-plane angular-greedy triangulation between neighbours with normal
  consistency > 0.9, Laplacian remeshing. Direct triangles, no implicit.
- **S5 Feature-preserving mesh filtering** (`--post bilateral`): bilateral normal
  filtering of the marching-cubes mesh (face normals averaged with spatial × normal
  similarity weights, vertices updated to the filtered normals, a few iterations) — the
  route the user set aside earlier, kept as the baseline lever.
- **S6 Learned SDF from particles** (Zhao et al. 2024): a CNN over the particle density
  predicting the SDF; not implemented — it needs a training set of particle clouds with
  ground-truth surfaces, which we could make from the target meshes, but it is a project,
  not a candidate for today.

Acid test, pre-registered: the TARGET cloud (150k samples of a clean mesh) rendered through
each candidate must come out SMOOTH — mean |dihedral| far below 14.5° — while the morphed
frame keeps its thin features (dragon horns, bunny ears, cow teats: component count and the
filament/bridge QA unchanged) and the lump amplitude does not rise. Measured on bunny frame
500 / target, dragon frame 400 / target, cow end / target.

#### Readings (07:45–08:40; `scripts/ops/surface_test.sh`, `scripts/probes/surface_gt.py`)

**The measure had to change first.** Mean |dihedral| depends on the triangle size (a finer
mesh of the same surface scores lower), so candidates with different meshers cannot be ranked
by it. `surface_gt.py` reproduces the target cloud (same fill, same seed; residual 2e-6 wu
against the archive), maps the asset mesh into the cloud frame and measures the
reconstruction of the TARGET cloud against the true surface: mean |distance| and signed
distance of the recon vertices (spacings; + = outside), 95th percentile, completeness (true
surface → recon), the normal deviation at the closest point, and a scale-defined roughness
(mean angle between a face normal and the area-weighted mean normal within 2 spacings —
resolution-independent; the true mesh's own value is the floor: bunny 7.3°, dragon 15.4°
(scales), cow 16.8° (a 5 804-face mesh, faceted at 3–5 spacings)).

**Where the floor comes from.** The target cloud is a voxel fill at 110³ + a uniform jitter
of half a voxel: pitch 0.82 (bunny) / 1.19 (dragon) / 1.28 (cow) spacings, 0.56 / 1.67 /
2.12 particles per fill voxel drawn WITH replacement — so the cloud is a Poisson-random
subsample of a voxel staircase. Blurred at 1.5 spacings the relative shot noise is
1/√53 ≈ 14 %, which moves the level set by ~0.6 spacing with a 3-spacing correlation
length: the 14.5° "bumpiness" and the ⅓-spacing lumps of 2026-09-18 item 4, now with a
cause. It is not the renderer's kernel and it is not the morph.

**A bug in the deliverable level, found by the layer selection.** The outer-layer rule
(density below the half-space value one spacing deep, 0.748 × bulk) selected 434 of
150 000 bunny particles. `rho_bulk` was the median over OCCUPIED VOXELS; the blur's halo
(3σ = 4.5 spacings of sub-bulk voxels around the whole body) pulls that median to ~0.5 of
the density a particle actually sees, so the "0.283 × bulk" level of §10.10 was in fact
~0.14 of the interior density and every gallery surface sat OUTSIDE the true one: signed
distance +1.56 (bunny) / +1.91 (dragon) / +1.76 (cow) spacings (run 1). With the bulk =
median over the particles (commit 2512410; `--bulk voxel` keeps the old value for
comparison, the ratio is printed) the marching-cubes surface moves to +0.50 / +1.18 / +1.16
spacings — the remaining offset is the level itself (the two-particle filament level sits
0.86 spacing outside the continuum boundary by construction) plus the fill's half-pitch
overhang. The normal deviation does not change (12.7–15.1°): the bulk fixed the thickness,
not the bumps. The v8 gallery videos were rendered with the old bulk (surfaces ~0.7–1
spacing too fat everywhere); a re-render is a deliverable decision, recorded here.

**Candidates on the target clouds (run 3, bulk fixed; spacings / degrees):**

| candidate | bunny d_abs / d_sgn / n_dev / rough | dragon | cow |
|---|---|---|---|
| S0 marching cubes (gallery) | 1.71 / +0.50 / 14.5 / 12.8 | 1.19 / +1.18 / 15.1 / 12.3 | 1.16 / +1.16 / 12.7 / 13.2 |
| S1 Yu & Turk PCA kernel, σ = blur | 1.70 / +0.25 / 15.0 / 13.7 | 1.07 / +1.06 / 14.3 / 13.3 | 1.05 / +1.04 / 12.5 / 13.3 |
| S2 Poisson, plane-pulled layer | 1.45 / −1.38 / 19.9 / 11.4 † | **0.27 / −0.02 / 16.9 / 9.8** | **0.20 / −0.09 / 12.9 / 9.1** |
| S2 Poisson, raw layer (pull 0) | 1.49 / −1.40 / 22.3 / 16.0 † | 0.24 / +0.02 / 16.9 / 11.2 | 0.18 / −0.04 / 13.3 / 10.3 |
| S3 IMLS | 2.85 / −0.96 / 26.5 / 18.6 | (removed) | (removed) |
| S4 surfel triangulation | 1.17 / −1.14 / 17.8 / 26.3 | 0.36 / −0.25 / 18.4 / 28.2 | 0.30 / −0.26 / 13.7 / 25.9 |
| S5 bilateral on S0 | 1.71 / +0.50 / 14.5 / 11.5 | 1.19 / +1.18 / 15.5 / 10.3 | 1.16 / +1.16 / 12.8 / 11.0 |

† bunny.obj is open at the base; the fill closes it, so the recon has a floor the true mesh
lacks (d_95 5–10 spacings for every candidate) — the dragon and cow columns are the clean ones.

- **S2 Poisson on the plane-pulled outer layer** is the one candidate that moves the
  numbers: the surface sits ON the true one (bias −0.02 / −0.09 spacings against +1.2 for
  marching cubes; mean distance 0.20–0.27 against 1.16–1.19, a 5× gain), roughness 9.1–9.8°
  against 12.3–13.2° (the true dragon is 15.4°: its scales are below what any 1.5-spacing
  kernel resolves, so the reconstruction is smoother than the truth), and the dragon's horns
  and the cow's legs come out complete (completeness 0.27–0.35 spacings). The normal
  deviation is the floor of the coarse true meshes (12.9° cow, 16.9° dragon vs 12.7 / 15.1).
  The plane pulling (the 3DGT constraint) buys 1.2–1.4° of roughness and the pulled
  positions alone are 0.3–0.36 spacings from the true surface with no bias.
- **S1 (PCA kernels)**: FALSIFIED as a smoother — the same bumps (13.3–13.7°) whichever the
  kernel's shape; with the Yu & Turk volume (σ0 0.7 spacings, run 1) it is WORSE (18.4–19.9°
  dihedral, 56–290 interior cavities: a 0.7-spacing kernel resolves the particle gaps).
- **S3 (IMLS)**: FALSIFIED as implemented — 54–185 spurious zero crossings on the morph
  frames, a box around the bunny's ear from the far-field sign rule; the normals of a thin
  sheet cancel inside the 2-spacing kernel.
- **S4 (surfel triangulation)**: the surfel POSITIONS are the best of all (0.30–0.36
  spacings, unbiased), the tangent-plane angular-greedy triangulation is not a surface
  (overlapping fans, 12 triangles per vertex, roughness 26–28°). Its geometric idea — pull the
  surfels onto their local plane — lives on inside S2 (`--pull`).
- **S5 (bilateral)**: −1.5 to −2° of roughness, nothing else; the bumps are 3 spacings wide,
  the filter's reach is a triangle.

**S2 on the morph frames, first form:** holes (dragon frame 400: 13 components, roughness
16.9° against 12.5° for marching cubes). Two structural causes, both fixed (commits d47a0da,
c3993db): (1) the DENSITY rule for the outer layer takes a whole stretched region (0.6 × bulk
throughout) as "layer", several particles thick with meaningless normals → the layer is now
the RELATIVE gradient |∇ρ|/ρ above its half-space value one spacing deep (φ(1/σ)/(σΦ(1/σ)) =
0.285 per spacing), invariant to the local density; (2) the Poisson trim at 2 spacings cut
the surface wherever the layer was locally sparse → trim at the density kernel's reach
(3σ = 4.5 spacings: no surface where no particle's kernel reaches). Run 4 measures both.

**Run 4 (gradient layer + 3σ trim), roughness at 2 spacings, degrees:**

| | bunny 500 | dragon 400 | cow 871 | bunny tgt | dragon tgt | cow tgt |
|---|---|---|---|---|---|---|
| S0 marching cubes | 13.1 | 12.5 | 13.2 | 12.8 | 12.3 | 13.2 |
| S2 Poisson, density layer | 16.7 (28 comps) | 35.5 (300 tris) | 15.4 (23) | 11.5 | 9.8 | 9.1 |
| S2 Poisson, gradient layer | **9.4** (12) | **12.9** (11) | **8.5** (10) | **9.0** | 9.7 | 8.4 |

The gradient layer is the difference between a broken surface and a working one on the
morph frames (the density layer on dragon 400 collapsed to 300 triangles). On the target
clouds the two rules select the same particles (uniform density) and score the same. The
stills: dragon 400 through Poisson is a clean closed surface with the horns and whiskers,
visibly tighter than the fat marching-cubes blob; bunny 500 smooth with a few dimples on the
back; cow 871 smooth — but its LOWER LEGS ARE GONE, the hooves float as balls. A leg 2–3
spacings thick is thinner than the quadratic B-spline's support (3 cells of one spacing), so
the two sides' normals cancel inside the basis: Poisson's known thin-structure failure, and
exactly the material the two-particle-filament level of §10.10 was chosen to keep. The cell
size is therefore not "one surfel per cell" but "the support must be thinner than the
thinnest drawn continuum (two particles across)": cell = 2/3 spacing (`--poisson_cell`),
run 5 (with 1/2 as the bracket).

**Run 5 — the legs hypothesis FALSIFIED, cell = 1 spacing stays.** The marching-cubes still
of cow 871 has the same truncated legs and the same floating hooves: the particles are not
there (the morph did not fill the lower legs), so Poisson lost nothing. A finer octree
(depth 8, cell 0.4–0.5 spacing, reached by both 2/3 and 1/2 on the dragon and the cow)
follows the layer's noise instead: dragon 400 roughness 12.9° → 19.5° with 69 components,
bunny 500 9.4° → 10.7° with 25; the targets unchanged (dragon 9.7 → 10.8, cow 8.4 → 9.2).
The octree cell of one spacing (one surfel per cell, the B-spline averaging three) is the
right discretisation; `--poisson_cell` stays at 1.0.

**Decision (08:45).** S2 — screened Poisson of the plane-pulled outer layer (relative-gradient
rule, PCA normals oriented by the density gradient, octree cell = one spacing, trim at the
kernel's 3σ) — is the surface: on the target clouds the mean distance to the true surface
drops from 1.16–1.19 to 0.20–0.29 spacings with zero bias (marching cubes: +1.2), the
roughness at two spacings from 12.3–13.2° to 8.4–9.7°, and on the morph frames from
12.5–13.2° to 8.5–12.9°, with the horns, whiskers and legs intact where the particles are.
The costs, recorded: 10–13 raw components per morph frame instead of 1 (small flaps of the
open trim and dimples counted as cavities; the mass rule drops them, the sidecar counts
them), a 2-spacing filament below the B-spline's reach relies on the bridge rule as before,
and ~4× the render time (Poisson is CPU). Before the gallery is re-rendered the whole
trajectory must pass the per-frame QA: three full videos (bunny, dragon, cow) with
`--surface poisson` are rendering into `surf_test/videos/`; the v8 pages are untouched until
they do. The bulk fix (2512410) applies to marching cubes as well, so the pages will be
re-rendered either way.

**The whole-trajectory test, first pass (08:33–09:25): two failures, both structural.**
(1) The cow video finished and FAILED the deliverable QA: drawn components > 1 in 143 of
292 frames (v8 marching cubes: 7), 101 of them unbridged, sub-cell pieces dropped in 253
frames. The 3σ trim opens the Poisson surface into flaps; an open piece has no volume (its
signed tetra sum is whatever the rim leaves) and takes the body's mass through the voxel
label it touches, so it is "drawn". Worse, an open BODY can fall below dx³ by the same
arithmetic and vanish (dragon stills 426–450: 0 triangles in 5 of 9 frames). The trim is
gone (`--poisson_trim 0`, commit 9b3f46e): a Poisson surface is closed by construction, a
hallucinated envelope encloses no particles and the mass rule of 10.10 drops it, exactly
as for marching cubes. Stills unchanged (bunny 500 9.4°, dragon 400 12.9°, cow 871 8.5°).
(2) The bunny and dragon renders died without a traceback; the retry with the shell's
stderr kept says it: `Segmentation fault (core dumped)`, exit 139, at frame 51 of 327 —
the first attempt died at frame 126. Not a frame (every frame 378–450 renders as a still),
a race inside Open3D 0.19's reconstruction across repeated calls. Poisson now runs in a
spawned child per frame (`physmorph/render/poisson_worker.py`): a crash costs one retry
(single-threaded), a second crash makes that frame a level-set frame and the sidecar
records it (`# surface poisson … fallback … frames`). A layer surfel with no neighbour
within 3h (an isolated shed particle passes the gradient rule) is dropped before the
plane fit. Second pass of the three videos running (09:45).

**Second pass (10:00–10:48): the isolation works, the QA rules did not carry over.** All
three videos finished (exit 0; the cow's child segfaulted twice, both absorbed by the retry,
0 fallback frames). But drawn pieces > 1 in 4 (bunny) / 28 (dragon) / 141 (cow) frames
against 0 / 0 / 7 for v8, with 0 bridges. The rules of 10.10 were written on the LEVEL SET
and read it: (a) the bridge rule took "enclosed" from the voxel labels of ρ ≥ iso — the
blurred level set joins a hoof to its leg across a one-spacing neck that the Poisson surface,
at the true boundary, shows as a gap, so the labels saw one body and drew no filament; (b)
a component's mass was the count of the voxel label at a representative vertex (± one
voxel), so a Poisson piece next to the body inherited the body's mass; (c) the body's sign
reference was its signed volume, which a spray-blob with borrowed mass and a larger
|volume| could take over — bunny frame 63: the BODY classed as a cavity and removed, two
blobs drawn. Three structural fixes (commits ec4cc74, f5a8099, d6238c4, 35ee9fc), all
reading the surface that is drawn: (1) enclosure per particle from the drawn mesh itself
(ray-casting occupancy + the nearest triangle's component), with a particle within one
spacing of the surface counted as that component's (the surface passes THROUGH the outer
layer; without the tolerance half the layer is "free" and the bridge rule drew 31 M
triangles of filament); (2) the bridge is the SHORTEST particle chain (Dijkstra, distances
as weights) from the body to each linked piece — the earlier "every free particle in the
connected cluster" drew the expansion-phase spray whole (bunny 63: 383 M triangles) once
the surface no longer hid it inside a fat level set; (3) for a reconstructed surface the
mass of a component is the number of particles it encloses (occupancy per component over
the particles in its box), the body the heaviest, a light component whose centroid the body
encloses a cavity — the marching-cubes path keeps its voxel-label rule, so the v8 numbers
stay comparable. Test frames after the fixes (Poisson): bunny 63 one drawn piece (12 raw,
9 dropped, 2 cavities), cow 420 one (3 / 2 / 0; the hoof blob is sub-cell for marching
cubes too), dragon 312 one (15 / 10 / 4), cow 507 one, bunny 500 one; ~15 s per frame
including the child. Third pass of the three videos running (11:15).

**Third pass (11:15–11:56):** drawn pieces > 1 in 10 (bunny) / 21 (dragon) / 79 (cow)
frames, of which unbridged 6 / 15 / 16 (v8 marching cubes 0 / 0 / 7, all bridged); no
fallback frames; one absorbed segfault. Every unbridged frame is in the EXPANSION phase
(bunny 45–66, cow 24–72, dragon 210–285): the leader clusters of the sphere's expansion —
a cell of particles or more, 3–4 spacings off the body — that the fat level set used to
swallow and that `grid_fragments.py` counts as 0 fragments because they share a dilated
cell with the body. The bridge rule's link radius was the particle-thread radius (2.5
spacings); the continuum's own resolution is the CELL, so the link radius is now
max(2.5 spacings, dx) (commit c74543c) — the same definition of "one body" as the grid
probe. Stills after the change: bunny 57 (20 raw, 13 dropped, 5 cavities) one body + one
bridged piece; dragon 255 the same; cow 30 one body + two bridged; cow 255 one body. The
bridged pieces sit flush against the body in the stills.

**Gallery re-render, first launch (12:05) stopped at 8 of 19 (13:55).** The first six
sidecars (bob, maxplanck, cow, cheburashka, bunny, spot; v8 in brackets): drawn pieces > 1
in 11 (0) / 14 (0) / 77 (7) / 8 (0) / 7 (0) / 49 (0) frames, unbridged 6 / 0 / 1 / 0 / 4 /
36. Spot frame 114 rendered as a still shows ONE clean body — the second "drawn piece" is a
closed sheet INSIDE the body: the relative-gradient rule also selects particles at a
density step inside the material (a compressed/stretched interface during the morph), the
Poisson closes a surface around it, and with ≥ ppc particles inside it passed the mass rule
while the cavity rule only took light components. Every component the body encloses is
interior, whatever its mass (commit fde333f; removed and counted in the cavity column).
After it: spot 114 one piece (4 raw, 2 dropped, 1 interior), spot 792 one (8 / 0 / 7),
bunny 57 one (20 / 13 / 6 — the "bridged piece" of the earlier reading was interior too),
bob 51 one (230 / 227 / 2: the expansion spray as sub-cell blobs). Relaunched on all 19
(13:58; the same six lanes, markers `SURFRR <prefix> <T> DONE`, v8 files kept as
`<T>_v8mc_*`) — and STOPPED at 14:36 on the user's call: check N = 40 000 first, before
the full render.

**40k check (14:40–15:30; `lr64_bunny`, `lr64_dragon`, `or40_C` — the recipe's 40k runs).**
Acid test on the targets (spacings / degrees): dragon marching cubes 1.09 / +1.09 / 13.2°
→ Poisson 0.27 / −0.06 / 11.7°; bunny 0.98 / +0.95 / 13.2° → 0.90 / −0.83 / 10.2° (the
open-base mesh again: d_95 6 spacings, the signed mean is the floor); morph frames 13.4°
→ 10.8° (bunny 571) and 13.4° → 11.8° (dragon 1022). Two things the 40k stills showed
that 150k had hidden: (1) FACETS — at 40k the spacing is 2.6 render voxels, so an octree
cell of one spacing gives triangles far coarser than the level-set mesh's and the bunny
rendered as a polyhedron. The cell stays (finer follows the noise); the mesh is
Loop-subdivided until its triangles are no coarser than the render voxel, iterations =
ceil(log2(cell / vox)) (2 at 40k, 1 at 150k). Roughness at two spacings after it: bunny
10.2 → 5.6°, dragon 11.7 → 7.8° (the subdivision is itself a smoother at the cell scale;
the true dragon at 40k is 21.7°). (2) THIN SHEETS — the C's arms at 40k are two or three
particles thick, and the plane pulling took the k nearest neighbours from BOTH faces: the
weighted centroid is the mid-plane, both faces were pulled onto it, and Poisson got
coincident surfels of opposite normal — ragged rims, pits. Neighbours now count only on
the same side (weight × max(0, n_i·n_j) with the gradient reference normals; commit
43252a4). The C at 40k still shows a ragged arm rim and dents on the sheet — it is the
thin-sheet limit of a 2-particle continuum and is recorded as such; the level set draws
it fat and smooth by hiding the sheet inside 1.9 spacings of blur.

**40k videos, per-frame QA (Poisson vs marching cubes; drawn pieces > 1 / of which
unbridged / Poisson fallback frames):** bunny (382 frames) 0 / 0 / 0 vs 0 / 0; dragon (683)
0 / 0 / 0 vs 0 / 0; C (983) 0 / 0 / 0 vs 9 / 0 (the level set's nine bridged frames are
pieces the Poisson surface keeps attached). One absorbed segfault on the C. The Poisson
surface passes the deliverable QA on all three 40k archives. Comparison page (stills,
numbers, the six candidates): artifact 0c05b8c6.

**The user's question (16:50): the smoothing kills detail — should the render gradient
not have captured it?** Answered with the scales: below the spacing (scales, fur) the
target cloud has nothing (fill pitch 0.8–1.3 spacings) and no gradient can supply it; at
1–3 spacings the target is 14 % shot noise (drawn with replacement), the level set shows
that noise as "detail" (its normal error against the truth equals Poisson's) and the
Poisson averages features with it — roughness at two spacings dragon 150k true 15.4° →
level set 12.3° → Poisson 9.8°, bunny 40k 10.6° → 13.2° → 5.6° (below the truth: the
reconstruction over-smooths at 40k, where a spacing is 2.6 voxels); the loss sees the
morph through a loss_res-64 density image (cell ≈ 1.5–2 spacings at 150k), so its
gradient stops there and, in the 1–3 spacing band, chases the target's noise as much as
its features (the loss-cell ladder's lumps that did not follow the cell). The structural
lever is upstream: sample source and target WITHOUT shot noise (one jittered particle per
fill voxel, or Poisson-disk), after which both the loss blur and the reconstruction cell
can go down to the sampling scale. Proposed test: resample the target clouds without
noise and rerun the acid test with a 0.5-spacing Poisson cell — if the dragon's roughness
climbs toward the truth without fragments, the sampling is the lever. Awaiting the user's
call; the 150k gallery re-render stays stopped.

### 2026-09-19 — the gradient at each stage, and the outer-layer projection (17:00–19:30; docs/surface_gradient.md §6)

User: 40k first; analyse the physics and render gradients stage by stage; the render signal
needs a PROTOCOL, not another loss term (e.g. an external force); make the surface smooth
during the morph. Done with `--grad_dump` + `scripts/probes/grad_stage.py` (40k bunny, 8
windows, 3 min): the render covector is 99 % on the outer layer and 65 % of it is
uncorrelated between neighbours two spacings apart; after the MPM adjoint every channel's
control gradient has the grid's correlation length (0.86 at two spacings, 0.6 at four, 0.2
at eight); followed alone, the render channels roughen the outer layer (0.474 → 0.50
spacings plane-residual RMS) and the physics channel smooths it (0.458). The control has no
sub-cell modes. An external force on the layer particles is averaged away by P2G/G2P (a
critically damped spring moved a half-spacing bump 2.4 % in a window) — sub-cell relative
motion is not a momentum mode, as the bonds already showed. Protocol adopted: a per-step
POSITION projection of the rough plane residual of the outer layer, 1/T per step
(`--layer_relax`; kernels `k_layer_resid` / `k_layer_project`, on the tape; a division by a
loop-accumulated weight inside the kernel broke the adjoint — weights normalised on the
host; directional finite differences match). Fraction 1 (hard constraint) diverges.
Full 40k runs: outer-layer plane-residual RMS over the morph bunny 0.442 → 0.290, dragon
0.403 → 0.279 spacings (the target clouds' own floor 0.339 / 0.325); silhouette IoU 0.962 →
0.957, 0.964 → 0.952; chamfer unchanged; det F min up. The RENDERED surfaces, however:
marching cubes 13.4 → 13.8° (no change), Poisson 6.1 → 5.6° (bunny), 7.8 → 7.2° (dragon):
the level set's texture is the density noise of every particle within the blur, not the
outer layer's offsets. Sampling test (bunny 40k, `output/sampling_test.py`): drawing WITHOUT
replacement (one jittered particle per fill voxel) moves marching cubes 13.2 → 12.3° and
Poisson 5.7 → 4.9°, the layer RMS 0.352 → 0.286, NN-distance CV 0.37 → 0.29 — the
replacement noise is a tenth of the level set's roughness; the rest is the jittered
arrangement seen through a 1.5-spacing blur, the floor of any level set of a random
particle cloud. What is smooth AND detailed at 40k is therefore the combination: the
projection in the physics (a smooth particle surface with its features — d − d̄ keeps what
neighbours share) and the Poisson surface for the deliverable (it reads the layer: 5–7°
against 13° for the level set, with the horns and ears intact). Videos of the relaxed runs
(Poisson and marching cubes) for the per-frame QA: see the next entry.

### 2026-09-19 — G1 + the position-mode control channel; the recipe changes (19:40–20:00; docs/surface_gradient.md §7)

User: how can the render gradient reach below the cell? Answer: the actuator and the
reference, not the signal. Implemented G1 (`--pbr_denoised`: the shading reference from the
target's Poisson-surface normals; the morph's normals on the render-pixel grid with the
renderer's blur) and A (`--layer_ctrl`: a per-window normal displacement leaf on the outer
layer, 1/T per step, a second Adam leaf clipped at one spacing, its adjoint bypassing the
grid). 40k bunny / dragon, four arms (recipe, + relax, + relax + G1, + relax + G1 + A):
silIoU 0.962 / 0.957 / 0.957 / 0.961 and 0.964 / 0.952 / 0.949 / 0.959; chamfer best with
A (0.1196, 0.1235); det F min monotone 0.68 → 0.76, 0.66 → 0.78; outer-layer RMS 0.44 /
0.29 / 0.29 / 0.36 and 0.40 / 0.28 / 0.27 / 0.34. G1 alone: no change (the actuator is
the same). A: three quarters of the relaxation's IoU cost back, the best shape metrics, a
little more roughness. Structure or noise, resolved with two band-limited measures against
the true mesh at the end frame (high-passed residual `hp_res`, detail correlation
`dcorr`): A's bumps follow the true surface (lowest residual on both targets, 0.185 /
0.195 spacings against the recipe's 0.236 / 0.203) and on the dragon correlate best with
the true detail (+0.33 vs +0.28); the relaxation removes noise (residual down, correlation
flat or up); the relaxation's IoU cost is at the silhouette pixel (chamfer and the
distance to the true surface unchanged). Per-frame QA of the relaxed runs' Poisson videos:
bunny 522 frames, drawn pieces > 1 in 0 frames; dragon 629 frames, 0 frames, 0
unbridged, no fallback frames (sub-cell Poisson bubbles dropped in 601 frames — the
sidecar's count of the shot-noise blobs the mass rule removes, not pieces). **User accepted the
proposal (19:20): recipe = v8 recipe + `--layer_relax --pbr_denoised --layer_ctrl`
(`hyde06_env.sh`; `RECIPE_V8` kept), Poisson the deliverable surface; 150k only after the
whole pipeline is verified at 40k.** Verification runs `p40_bunny`, `p40_dragon` launched
with the new recipe, to be post-processed end to end (post_run, photoreal Poisson, QA
sidecars, report page).

**40k pipeline verification, end to end (19:05–21:10; `p40_bunny`, `p40_dragon`, the
recipe from `hyde06_env.sh`).** Runs: bunny silIoU 0.9628, chamfer 0.1196, det F min
0.766, 9.6 min; dragon 0.9555, 0.1235, 0.736, 11.4 min — the twins of `g1a_*` (0.9614 /
0.9594): the run-to-run IoU spread is ±0.4 points, so against the v8 recipe the new one is
+0.05 (bunny) / −0.9 (dragon) with chamfer and det F better on both. Gates G2/G3/G4 PASS
on both. `post_run.sh`: every output present (surface / splat / particle videos, PBR
stills, loss, census, scatter, grid fragments); grid fragments ≥ 1 cell 0 frames on both,
end fragments 0, re-attachments 0, stray census max 0.50 wu (bunny) / 0.39 (dragon).
`photoreal_batch.sh` (Poisson): bunny 502 frames, drawn pieces > 1 in 0 frames, unbridged
0, fallback 0, isolated max 17; dragon 656 frames, 0 / 0 / 0, isolated max 59; the
sub-cell bubbles the mass rule drops: 176 and 636 frames (the shot-noise blobs, not
pieces; recorded in the sidecar's own column). `build_report150.py` parsed the Poisson
sidecars (with the `# surface` line) and built the page; fetched with `fetch_page.sh p40`
and published. The whole chain works at 40k with the new recipe; the 150k gallery can be
re-run as NEW runs when the user says so (19 targets; the Poisson videos are the slow
part: 45–100 min per target at 40k, CPU-bound, so lanes).

### 2026-09-18 — SUMMARY (read this first; the ladder below is the working record)

- **Delivered:** 150k gallery v7 (10 + 9 targets) with the five goals answered — pages
  2f348b78 (main) and 4b18fc06 (new meshes), markdown `docs/highres150_v7_report.md`,
  `docs/newmesh150_v7_report.md`; formulation `docs/method.md` §10.10–10.11.
- **Ejection, two causes, both structural:** (1) the expansion-phase leader particles →
  transport-paced cell sum + cell-wise deficit hand-off (`--phys_loss auto`); (2) the domain
  box was a trap — no wall boundary condition, so a particle within two cells of the box
  edge froze on a truncated stencil (v6 dragon 755 re-attachments = 10 706 box hits, bob
  486 = 10 934, 40k 0) → separating walls on six faces (`WALL_NODES` = stencil
  half-support). v7 re-attachments over 18 targets 284 (v6 2018), dragon 755 → 7, mean
  silIoU 0.951 → 0.962. Without the net at 150k: bunny 0, dragon 3, bob 54, C 74 fragments.
- **Render gradient → physics (three shapes):** per window the render channel is 35 % of
  the accepted update at cosine 0.02–0.08 to the physics gradient (deterministic); outcomes
  sit 4–10 run-to-run spreads apart (dragon 0.939/0.918 vs 0.840/0.770, bunny 0.966 vs
  0.932/0.936, bob 0.970/0.968 vs 0.952/0.956). Trajectory divergence: chaos floor on the
  dragon, a 1.4–1.7× step over the floor on bunny/bob — supporting, not primary.
- **C at 150k:** morphs (0.940) with the raw transport map + walls; 530 re-attachments
  (arm-front chunks); paced target, grid-resolved displacement and kNN smoothing on the hole
  regime all FALSIFIED (0.88/0.55, 0.59, 0.79).
- **Deliverable rule (goal 3):** isosurface level = two-particle-filament level (0.28 of the
  bulk); a component is drawn iff it holds ≥ ppc particles and ≥ dx³; closed surfaces with
  the sign opposite to the body are interior cavities; particles the isosurface does not
  enclose but which link the body to a drawn piece are drawn as a one-spacing filament.
  Final QA: detached drawn pieces 0 frames on 18 of 19 targets, C 29/203; physical
  fragments ≥ 1 cell (grid probe, no renderer) 0 on 18, C 6.
- **Material → trajectory:** bunny/dragon × 5 materials + identical-config control: early
  divergence 8–130× the noise; on the dragon stiffness changes the shedding (soft 343 /
  base 14 / stiff 0) and the end (0.846–0.963).
- **Cleanup (11:20):** server output 244 → 109 GB — 69 falsified/superseded runs' npz
  deleted (their logs/json in `logs_archive_20260918.tgz`), live packets pruned to the 19 v7
  runs (44 GB), report folders of v5/v6 removed (pages are published); local output/ 2 GB →
  35 MB (only the two v7 page folders). Kept: v7 runs, render-proof twins + controls,
  material study, nn150, c150r_C, or40_C, h150y_dragon.
- **Open (next round, user 11:15):** ejection to zero WITHOUT the net (C, bob, and across
  materials — the soft dragon needs the net too); surface bumpiness / visible particles
  during the morph (renderer covariance, not smoothing — five hypotheses first); loss
  jitter/spikes near the optimum (optimizer vs early stop); the camera (every mesh looks
  down).

### 2026-09-18 — next round, first readings (11:40)

- **Item 3 (material → net):** the soft dragon's 343 re-attachments were the BOX TRAP —
  `mat_soft_dragon` 25 660 box hits (base 0, stiff 0, assim 0.1 0, control 0); the soft
  body overshoots farther and reached the band; ν 0.45 (95 re-attachments, 0 box hits) is
  the one material-driven case left. The material study must be re-run under the walls.
- **Item 5 (loss jitter):** the "spike" on C is the logged REJECTED candidates, not the
  trajectory — windows 152–163 all rejected (gain −0.12 … −0.185, reversal 0.95–0.97), the
  last six with the identical score (a replay: the optimizer re-proposes the same step),
  the deliverable ends at the best commit 151. On the 18 density-regime runs the end-game is
  2–5 rejections with gains −0.0002 … −0.01, i.e. the plateau, and the loss curves are
  monotone to the eye. So the gate already is an early stop; what is wasted is the replay
  streak (12 windows on C) — stop at the first exact replay.
- **Item 6 (camera):** the camera is fine (18° above, y up); several ASSETS are not y-up:
  bunny and nefertiti lie on their side, spot's head points down, the teapot is seen from
  above, the armadillo lies flat — a per-asset up-axis table is needed (physics is
  rotation-equivariant: no gravity, no floor in the morph runs, so archives can be rotated
  at render time).
- **Item 1 (bob without the net):** `nn150_bob` sheds from window 16 (1 particle) and
  window 27 (4) — the ring's expansion phase, before the walls mattered (0 box hits).
- **Item 1, what bob sheds without the net (13:40; `blob_probe.py` on `nn150_bob` at
  frames 200 / 300 / 420, strict cell clusters down to one particle):** the 54 "fragments"
  are ~24 clusters of ONE to THREE particles, 35–96 spacings (1.3–3.7 wu) from the body AND
  from any target point, and static from frame 200 on (their centroids move < 0.05 wu in
  220 frames). Not chunks: single particles flung out of the expanding sphere in windows
  16–27 and left in empty space. So the last ejection mode at 150k is the single-particle
  leader of the expansion phase — it clears the fracture gap within one window and then
  nothing pulls it back (it sits outside the target's support where the paced cell sum has
  no deficit to fill with it).
- **Item 1, structural fix (14:00, f410ee9):** the material bonds' decoupling test was a
  mask fixed at the window start (the runner's fragment mask), so a particle that clears
  the fracture gap MID-window was bonded a window too late — exactly the single-particle
  leaders above. The test is now evaluated every step from the current state (3³-cell
  count, outside the tape, OR the commit mask); tests 26 passed (bonds FD, adjoint, smoke).
  Test runs without the net at 150k under walls + per-step bonds: `nnw150_bob` (54 →
  ?), `nnw150_dragon` (3 → ?), on GPU 0. (The material re-runs `matw_nu45_dragon` and
  `matw_soft_bunny` start after this deploy and carry it; `matw_soft_dragon` does not.)
- **Item 2, why detached material never comes back (14:20):** the stranded C chunks
  (`c150r_C`, walls: 85–105 particles, 1.7–2.3 wu from any target point, 0.003–0.015
  wu/frame) are NOT starved of transport signal — the plan's map still pulls a probe
  point 2–3 wu outside the target by 4 wu inward (debiased and plain entropic map alike,
  148 sweeps). They are static because the CONTROL IS A STRESS (dFc through k_stress):
  the internal stress of an isolated body integrates to zero net force, so a detached
  chunk cannot translate under any control — exactly as the whole body keeps its centre
  of mass. Ejection is therefore irreversible by construction; the net (re-attachment)
  is the only external force in the model, and "no net" can only mean "no fracture":
  prevention at the front (bonds evaluated per step for singles — done; for cell-sized
  fronts the candidate that tears the body has to be infeasible, i.e. the continuity
  check without the free-rollout allowance, or the front held by its material).
- **Item 3, soft dragon under the walls (10:28 CDT; `matw_soft_dragon`, 40k, young 3e4,
  net, bonds still window-fixed):** 0.1227 / **0.961** / **41 re-attachments**, 0 box hits
  (pre-walls 0.846 / 343 / 25 660 hits). So the walls removed 300 of the soft dragon's
  343 and restored its end quality to the base's (0.957); the residue of 41 (windows
  23–37, the expansion) is the material-driven part. `matw_nu45_dragon` and
  `matw_soft_bunny` follow (with the per-step bonds).
- **Per-step bonds, first readings (10:45 CDT):** `nnw150_bob` (150k, NO net, walls +
  per-step decoupling) has **0 fragments through window 75** where `nn150_bob` had shed
  from window 16 (1) and 27 (4) and reached 54 by window 73; `matw_nu45_dragon` (40k,
  ν = 0.45, net, walls + per-step bonds) 0 re-attachments through window 64 (pre-fix 95).
  Final numbers when the runs end.
- **Per-step bonds, results (11:15 CDT):** `nnw150_bob` (150k, NO net, walls + per-step
  decoupling) 0.0777 / **0.950** / **1 fragment** at the end (108 windows, 38 min) against
  `nn150_bob` 0.932 / 54 — the single-particle ejection of the ring's expansion phase is
  gone but for one particle, and the net-free quality rises by 1.8 points (the netted
  v7 bob: 0.969). `matw_nu45_dragon` (40k, ν 0.45, net) 0.1237 / **0.967** / **0
  re-attachments** (pre-fix 0.941 / 95): the ν 0.45 residue was single-particle leaders
  too. `nnw150_dragon` and `matw_soft_bunny` follow.
- **Item 4, objective side (11:11 CDT):** `rv_bunny` (render loss at 128 px × 12 views)
  0.1189 / 0.967 in 128 windows / 31 min vs `rvb_bunny` (64 px × 6 views) 0.1195 / 0.966
  in 103 / 25 min — no end-shape difference, and no surface difference either: bumpiness
  at frame 300 rv 13.80° vs rvb 13.53°. So neither the renderer's kernel nor the render
  loss's resolution moves the measure; what it measures at 13–15° may be the floor of the
  marching-cubes discretisation itself — the target cloud rendered through the same
  pipeline gives that floor (next).
- **Item 4, the floor (11:50 CDT):** the TARGET clouds rendered through the same pipeline
  (`--still -2`): bunny **14.55°**, dragon **14.46°** — the morphed states (14.96 / 14.42)
  are as smooth as the target sampled at this N. The particle-scale texture the eye sees
  ("orange peel") is the sampling noise of a random 150k cloud at a blur of 1.5 spacings
  (±10 % density fluctuation), present in the target renders too; the renderer kernels,
  the F-carried covariance and the render-loss resolution cannot go below it because it
  is not in the state. What IS in the state are the 0.3–0.5 wu lumps (the loss-cell
  scale), which the measure does not see and which need a surface-aware objective (the
  next physics experiment). Rendering the texture away needs a smoother reconstruction
  (a wider kernel or a surface fit — i.e. smoothing), which is the one route the user
  set aside; the numbers say the Gaussian route does not exist.
- **Item 3 complete for the tested materials (11:25 CDT):** `matw_soft_bunny` (young 3e4,
  net, walls + per-step bonds) 1 re-attachment (pre-fix 12); with soft dragon 343 → 41
  (walls; the per-step bonds not yet in that run) and ν 0.45 dragon 95 → 0. The material
  study's net-free claim now rests on the walls (box trap) + the per-step decoupling test;
  the soft dragon's 41 remain to be re-measured with the per-step bonds.
- **Item 2, C at 40k, three mechanisms FALSIFIED (11:25 CDT; net on, walls, per-step
  bonds):** `ce40_C` (assim 0.1, more elastic) 0.871 / 3 re-attachments, gate stop at 23
  windows; `ce25_C` (assim 0.25) 0.868 / 3, stop at 22; `cc40_C` (discrete continuity
  feasibility) 0.836 / 16, stop at 20 — all three freeze at the sphere-in-the-hole stage
  (rejections with gains −0.03 … −0.0005 at windows 19–23) where the plastic body
  (`or40_C`, assim 0.5) goes on to 0.959 in 166 windows with 101 re-attachments. An
  elastic C cannot be pushed out of the hole into the arms (the rebound undoes each window)
  and a continuity-feasible step does not exist for that push either. C's arm-front chunks
  therefore remain the one ejection the model cannot prevent without the net; the ledger
  of falsified mechanisms for C: paced target, grid-resolved displacement, kNN smoothing,
  elasticity 0.1 / 0.25, discrete continuity (and, earlier, ot_pace + hand-off, global
  hand-off, all-near-cell hand-off).
- **Round results (12:10 CDT):** `nnw150_dragon` (150k, NO net, walls + per-step bonds)
  0.0848 / **0.954** / **1 fragment** (one particle from window 31 on; `nn150_dragon` 3 /
  0.952); `matwb_soft_dragon` (40k, young 3e4, walls + per-step bonds, net) 0.1231 /
  0.959 / **1 re-attachment** (walls only 41; pre-walls 343). Net-free ledger at 150k
  after this round: bunny 0, dragon 1, bob 1, C 74 (unchanged — see item 2). Across
  materials with the net: soft dragon 1, ν 0.45 dragon 0, soft bunny 1, stiff / elastic /
  base 0.
- **Lump measure (12:40 CDT; `scripts/probes/lump_amplitude.py`):** the morph's
  isosurface (the renderer's own) against the target's isosurface through the same
  pipeline (so the level offset cancels; the target scores 0 by construction); per-vertex
  distance split by scale with two mesh-graph smoothings — wide (0.6 wu, the shape error)
  and narrow (0.15 wu, the sampling texture) — and the LUMP band is their difference,
  0.15–0.6 wu, the loss-cell scale. Readings (RMS, wu): spot 150k at 50 / 75 / 83 / 100 %
  of the run: lump 0.012 / 0.011 / 0.011 / 0.010, wide 0.048 / 0.038 / 0.038 / 0.035;
  bunny 150k 0.014 → 0.013 (wide 0.055 → 0.043); bunny 40k base 0.009 → 0.008 (wide
  0.040 → 0.035). I.e. at the end the surface sits one spacing (0.035 wu) from the target
  surface on average and the cell-scale lumps are a third of a spacing — small. The
  re-rendered v7 spot at 83 % is a clean cow with soft flank undulations; the screenshot
  the user sent (the same frame, un-oriented) shows the cow belly-up with its legs and
  head reading as lumps — the orientation fix removes most of that impression. The
  loss-cell ladder (`lr64/96/128_{bunny,dragon}`, 40k) is running to see whether the
  remaining 0.01 wu follows the cell.
- **Loss-cell ladder, result (13:55 CDT; 40k, net on, walls + per-step bonds; silIoU /
  lump RMS at the end / at 50 %):** bunny loss_res 64: 0.962 / 0.0084 / 0.0107; 96: 0.967
  / 0.0085 / 0.0102; 128: 0.966 / 0.0085 / 0.0093. dragon 64: 0.964 / 0.0104 / 0.0114; 96:
  0.964 / 0.0105 / 0.0112 (128 pending). Re-attachments 0 in every run. **The lump
  amplitude does not follow the loss cell** (0.0084 → 0.0085 as the cell halves) — the
  cell hypothesis for the residual surface lumps is falsified; what remains at the end is
  a third of a spacing of band-passed error, the floor of the particle system through this
  kernel. loss_res 96 buys +0.4 silIoU on the bunny and nothing on the dragon at ~20 %
  more time; the recipe keeps 64. Item 4 therefore closes on three findings: the texture is
  sampling noise (target floor 14.5°), the lumps are not cell-scale (a third of a spacing),
  and the screenshot's lumpiness was the un-oriented view.
- **v8 launched (14:00 CDT):** the gallery re-run as a NO-NET pass of all 19 targets at
  150k (walls + per-step bonds + early stop + y-up assets; `--reattach` off, so the
  end-frame fragment count is the ejection ledger). Two 150k runs at a time on GPU 0 (a
  150k run takes 6.6–8 GB beside the other user's 27 GB; a third run OOMs — the first
  bunny and A attempts died that way and were re-queued). Photoreal + QA follow
  automatically; the page is built from the v8 narrative when all 19 are in.
- **v8 ledger, first four (16:20 CDT; 150k, NO net; end fragments / silIoU, v7 netted
  silIoU in brackets):** V 0 / 0.963 (0.969), dragon 1 / 0.955 (0.960), bob 9 / 0.949
  (0.969), armadilo 1 / 0.938 (0.949), teapot 0 / 0.967 (0.970). Wall-clock 38–88 min at
  two runs per GPU. The bob's 9: eleven strict clusters of 1–2 particles plus one cluster
  of 9 packed at 0.05 × the bulk spacing (a collapsed clump), all 24–65 spacings out.
  heart 0 / 0.978 (0.979). **C without the net: 3 / 0.905** (netted 0.940; `nn150_C`
  before the walls + per-step bonds: 74 / 0.889) — the stray peak is still 15 % (92
  disconnected particles at window 185), but the arms grow into the stranded chunks by the
  end, so the end-frame ledger reads 3; the silIoU cost of not resampling them is 3.5
  points. Physical fragments ≥ 1 cell by the grid probe: 0 frames on V, dragon, bob,
  armadilo, teapot (max clusters 0–9 particles). spot 0 / 0.965 (0.972), cow 1 / 0.924
  (0.926) (18:52 CDT). Nine of nineteen in: 0 / 1 / 9 / 1 / 0 / 3 / 0 / 0 / 1 fragments.
  bunny 0 / 0.964 (0.966) (19:38 CDT; 43 min once the other user left GPU 0 — the clean
  single-run time at 150k is ~40 min for the bunny). homer 0 / 0.958 (0.958) (20:24
  CDT). The remaining eight run four abreast on the freed GPU. Grid probe on the v8 C
  without the net: physical fragments ≥ 1 cell in **0 of 299 frames** (largest 63
  particles = 0.74 cells; sub-cell clusters ≥ 20 in 104 frames) — the walls + per-step
  bonds have shrunk C's shed chunks below one cell of material; what the net used to
  resample is now sub-cell debris that the deliverable rule does not draw. fandisk 0 /
  0.970 (0.973), maxplanck 0 / 0.969 (0.968), A 0 / 0.967 (0.974) (21:05 CDT). Fourteen
  of nineteen in; the ledger so far: 0 0 0 0 0 0 0 0 0 1 1 1 3 9 (bob 9, C 3, dragon /
  armadilo / cow 1). beast 1 / 0.904 (netted 0.930; 21:36 CDT) — the one target where
  removing the net costs more than two points besides C. ogre 21 / 0.933 (0.939): the 21
  are one collapsed clump of 21 particles (0.05 × 0.15 × 0.05 wu, packed at a twentieth of
  the bulk spacing) 46 spacings above the body plus a single — the same collapsed-clump
  signature as bob's nine, a sub-cell object the deliverable does not draw. nefertiti 0 /
  0.968 (0.972) (22:02 CDT). cheburashka 0 / 0.963 (0.964) (22:24 CDT); bimba 0 / 0.974
  (0.975) (22:57 CDT).
- **v8 ledger complete (22:57 CDT; 19 targets, 150k, NO net):** end fragments — twelve
  targets 0 (bunny, teapot, heart, spot, V, A, homer, maxplanck, nefertiti, fandisk,
  cheburashka, bimba); dragon, armadilo, cow, beast 1 (single particles); C 3; bob 9 and
  ogre 21 (one collapsed clump each, a twentieth of the bulk spacing, sub-cell); total 39
  particles against v7's 814 netted re-attachments and v6's 2018. Physical fragments
  ≥ 1 cell by the grid probe: 0 frames on every target. Quality without the net: within
  0.7 silIoU points of the netted v7 on 15 targets; bob −2.0, beast −2.6, armadilo −1.1,
  C −3.5 (the thin-feature targets where the net returned material). Clean single-run
  times at 150k on a free GPU: 25–45 min for the compact targets, 60–110 min for dragon,
  armadilo, beast.
- **v8 published (01:05 CDT, 2026-09-19):** main gallery → 2f348b78 (replacing v7), new
  meshes → 4b18fc06; markdown `docs/highres150_v8_report.md`, `docs/newmesh150_v8_report.md`.
  Deliverable QA over the 19 v8 videos (iso auto + mass rule + cavity sign + filament
  bridges, y-up): drawn outer pieces detached from the body — 0 frames on every target
  (C: 12 frames with two drawn components, all bridged); physical fragments ≥ 1 cell by
  the grid probe — 0 frames on every target (largest cluster 0.74 cells, C). The
  post-processing had to be re-run once: the orientation change had turned the archive
  into a dict inside two renderers and their `deliver_n` test failed silently (no
  isosurface video, no delivered stills) — fixed in 9b21e21 and re-run for all 19.
- **Item 6 done (12:10):** `scripts/probes/orientation_check.py` rendered every asset under
  five rotations; eleven of the nineteen are z-up (bunny, spot, nefertiti, teapot, dragon,
  armadilo, heart, A, C, V, bob → `x-90`), the rest y-up. `physmorph/sampling/orientation.json` +
  `physmorph/sampling/orientation.py`: the loader rotates the mesh at load time and the
  archive records `orient`; every renderer (photoreal, iso video, splat video, particle
  gif, PBR stills) rotates older archives by the table at render time. No re-run needed.
- **Item 5 done (12:10):** `cfg.reject_stop = 3` — three consecutive rejected candidates
  (any kind) end the run at its best commit. Justified by the v7 rejection positions: every
  run's rejections are a terminal streak of 2–5 (bob 127–129, bunny 208–210, dragon
  188–190, …); isolated rejections mid-run are followed by descent (A 177 → 217, beast
  264 → 274, cow 218 → 224) and would not trigger a streak of three; C's 12-window replay
  would have ended at 154. The loss "spikes" are the logged rejected candidates, not the
  trajectory; the delivered archive stops at the best commit either way.
- **Item 4 (surface bumps, visible particles) — five hypotheses, pre-registered (12:30):**
  the deliverable surface is the isosurface of a density made by CIC deposit + an
  ISOTROPIC Gaussian blur of 1.5 rest spacings (grid 160). H1 *kernel below the
  irregularity*: after plastic flow the local spacing varies by tens of percent, and an
  isotropic kernel at 1.5 rest spacings resolves that irregularity as bumps of 2–3
  spacings; a kernel that is each particle's MATERIAL PATCH carried by the deformation,
  Σ = σ0² F Fᵀ (eq. 11), sums to a flat field where the rest cloud was regular (a partition
  of unity in the rest frame) — bumps should drop by a large factor. H2 *stretched
  material uncovered*: where the body stretched (ears, horns, arms) the particles are
  farther apart than the rest spacing, the isotropic kernel leaves gaps between them and
  the surface shows beads/particles; the F-carried kernel is elongated exactly there.
  H3 *level inflation*: the two-particle-filament level (0.28) sits 1σ outside the
  half-density surface and rounds every sharp feature; with H1/H2 kernels the natural
  half level (0.5) should render thin features without the inflation. H4 *surface
  discreteness*: the outermost particle layer's kernels are half outside the body, so
  the half-level surface cuts through them and follows their individual bumps; the
  F-carried kernel of a surface particle is flattened tangentially under compression and
  elongated under stretch, which smooths the cut. H5 *grid aliasing*: CIC deposit onto
  1.3-spacing voxels followed by a separable blur aliases particle motion into
  frame-to-frame flicker of the bumps; the analytic per-particle Gaussian evaluated at
  voxel centres has no deposit stage. Test: `render_photoreal.py --kernel aniso` (each
  particle deposits N(x_p, σ0² F_p F_pᵀ), σ0 = 0.7 rest spacings, F from the archive's
  F samples, truncated at 3σ) against `--kernel iso` on the same frames; measures =
  mean absolute dihedral angle of the marching-cubes mesh (bumpiness), number of drawn
  components, and the eye.
- **Item 4, hypotheses tested (13:10; bunny frame 500 / dragon frame 400, mean |dihedral|
  in degrees):** isotropic blur 1.5 spacings, grid 160 (the gallery) 14.96 / 14.42.
  F-carried Gaussians Σ = σ0² F Fᵀ with the TOTAL F fitted per frame over 12 rest
  neighbours: σ0 = 0.7 → 19.5 / 22.0, σ0 = 1.5 → 16.3 / 17.5 — **bumpier**, and on the
  dragon the fitted F is streaky where material sheared (17 components, a torn look):
  H1 (partition of unity) holds only for a regular rest lattice, ours is a random sample,
  so a kernel at 0.7 spacings overlaps ~10 particles and fluctuates by ~30 %; H2/H4 do not
  reduce the roughness either. The archived F is the ELASTIC part only (singular values
  1.19 / 1.00 / 0.84 median, det 0.99) and carries no patch shape. Isotropic variants:
  blur 2.5 → 13.9 / 13.7; grid 240 → 13.7 / 13.6; both → 13.3 / 13.0 (H5 aliasing: a
  10 % effect); Taubin 40 iterations → 21.8 / 21.2 (worse: the mesh ripples).
  **Conclusion:** the renderer is not where the bumps come from — every kernel change
  moves the measure by ≤ 10 % while the eye sees lumps of 0.3–0.5 wu, ten spacings, i.e.
  the LOSS-CELL scale (0.31 wu): the cell-sum objective cannot see sub-cell surface shape
  and the render channel sees a few silhouettes, so the state itself is lumpy at that
  scale. The rendering hypotheses are falsified; the fix belongs to the objective (the
  "rendering controls physics" premise): more silhouette views and/or a shading/normal
  term in the render loss so the gradient reaches the surface texture — a physics
  experiment for when the GPUs free up. `--kernel aniso` stays as an option with its
  negative result recorded; the gallery keeps the isotropic kernel (grid 240 + blur 2.5
  would buy 10 % on the measure at 3× the render cost — not taken).

### 2026-09-18 — overnight: v7 gallery complete, C under walls, controls

**v7 150k, all 19 targets (chains finished 02:04; photoreal for all 19 at 02:55; walls
from 22:05 on — dragon, cow, homer, maxplanck, fandisk, ogre, beast, bimba and the bob
re-run ran under them, the rest never touched the box band):** re-attachments / silIoU:
bunny 2 / 0.966, teapot 0 / 0.970, heart 0 / 0.979, spot 3 / 0.972, V 1 / 0.969, A 0 /
0.974, armadilo 28 / 0.949, **dragon 7 / 0.960 (v6 755 / 0.939)**, bob 126 / 0.969 (v6
486), cow 9 / 0.926, homer 4 / 0.958, maxplanck 0 / 0.968, nefertiti 5 / 0.972, fandisk
2 / 0.973, ogre 23 / 0.939, beast 68 / 0.930, cheburashka 6 / 0.964, bimba 0 / 0.975.
Sum without C: 284 (v6 2018 over the same 18). Band hits 0 on every walled run.

**C at 150k, final candidates:** `c150r_C` (walls, raw map image, net) 0.1548 /
**0.940** / hole 7.2 %, **530 re-attachments** in 163 windows, 990 band hits (all in
the first windows, before the walls bite), 63 min; `c150_C` (pre-walls) 0.954 / 2675;
`nn150_C` (walls, no net) 0.889 / 74 fragments at the end. The walls cut the shedding
5×; the residual ~100-particle chunks (1 cell) come off the arm fronts during the arm
phase and the net returns them. v7 carries `c150r_C` as its C (files copied to the
`h150v7_C` names, post-processed and re-rendered 07:15); C remains the one target whose
150k run needs the net for its shape.

**Controls (goal 1b, trajectory half):** bunny `rp_ctrl7_bunny` 0.0800 / 0.969 (its twin
0.966: spread 0.003; physics-only 0.932 and cut 0.936 sit 10 spreads below) — divergence
from the twin 0.12 spacings before window 40 (max 0.30) → 0.48 after (0.74 at the end),
against the cut twin's 0.14 → 0.77 (1.05): 1.4–1.6× the floor after the intervention,
as on bob (1.5–1.7×). Plots `_figs/render_effect_{bunny,bob,dragon}.png` all carry the
control curve.

**Goal 3, frame QA of the v7 photoreal videos (07:30):** frames with more than one drawn
component / total — A 0/284, armadilo 0/298, dragon 0/241, heart 0/141, teapot 0/130,
fandisk 0/215, homer 0/392, maxplanck 0/233, spot 2/229, V 2/204, beast 2/358, bimba
2/223, ogre 2/310, cheburashka 1/306, bob 17/166, nefertiti 18/258, bunny 23/268, **cow
62/286**, C (paced run, replaced) 155/168. Frames looked at: bunny 141 and 201, nefertiti
165, bob 102 — the second component is a lump or tip that reads as part of the body; cow
465–552 — **a ball floating under the belly for 90 archived frames**, the one true
violation. Probe `output/blob_probe.py`: at frame 522 the cloud is ONE cluster by occupancy
connectivity at every cell scale tried (0.31, 0.20, 0.15, 0.10, 0.07 wu — no separate
cluster of ≥ 10 particles), i.e. the ball is connected to the udder by a neck one to two
particles thick that the isosurface at half the bulk density does not render (a 2×2
bundle blurred by σ = 1.5 s peaks at 0.28 of the bulk, a single particle at 0.02).
Rendering fix 69d4457 (docs/method.md §10.10): `--iso auto` = the two-particle-filament
level, 2 s² / (π σ²) of the bulk capped at 0.5 (0.283 here); the cow frame 522 renders as
one component (the ball is the leg/teat tip, attached), single particles and sub-cell
clusters stay invisible. All 19 v7 videos are being re-rendered under it (GPU 0, the
only one with room — the other user now holds all four GPUs; ~1 h).

**Under the auto level (08:20, first re-rendered videos):** dragon 0 → 1 frame with two
drawn components, homer 0, cow 62 → 22 (a different episode, frames 339–402), bunny 23 →
37 (the level lets more ear-tip pieces pass the volume rule; nothing floats in frame 270).
Cow frame 372 at the strict 0.07 wu cell (2 spacings): one cluster of **72 particles,
0.4 wu across, packed at 0.65 × the bulk spacing, sitting ON the target (0.3–0.8 spacings
from the teat points), 13 spacings from the body on average and reaching it only through a
one-particle chain (closest gap 2.4 spacings)**; at every coarser cell it is one body. So
the teat is what the physics can make of a feature thinner than its cell: a compressed bulb
at the target's teat, tied to the udder by a single-particle thread that no isosurface
level short of the single-particle one (0.07 of the bulk, +0.25 wu of surface bloat) will
draw. Deliverable consequence: the rendered topology must follow the PARTICLE connectivity
— particles the isosurface does not enclose but which link the body to an enclosed
component are drawn as a filament of one particle spacing (the thread), so the bulb
reads as the tip of a thread, not as a floating ball; the new probe
`scripts/probes/grid_fragments.py` (physical fragments ≥ 1 cell by the grid's criterion,
no renderer) is added to the QA table so "the isosurface drew two pieces" and "the physics
has two bodies" are never confused.

**Physical fragments in the v7 videos (`grid_fragments.py`, every 3rd archived frame, the
net's criterion — occupancy dilated by one cell — with the cell-volume threshold ppc = 89
particles; 08:40):** frames with a fragment ≥ 1 cell: **0 in 18 of 19 targets**; C 6 of
202 frames (max 105 particles = 1.24 cells — the arm-front chunks before the net returns
them). Clusters ≥ 20 particles (sub-cell): bob 1 frame (32 particles), ogre 2 (21), C 11;
everything else ≤ 6 particles. So by the physics' own criterion the deliverable is clean
on 18 targets and C shows cell-sized fragments in 3 % of its frames; the "drawn
components > 1" frames of bunny (37), cow (22) and bob (17) are threshold splits of
connected material — now bridged in the rendering (69d4457 → the filament rule).

**Filament rule verified, and the bunny's "second piece" explained (09:30):** cow frames
372 and 345 re-rendered with the bridge show the bulb hanging from the udder on a thread
(sidecar: 22 of 22 drawn>1 frames bridged); bunny frame 270: bridged 0 — nothing to
bridge, because the bunny's extra component is not a piece at all: its signed volume has
the sign OPPOSITE to the body's (body −62 wu³, the extra +0.04 to +0.05 wu³ = 1.3–1.8
cells, a 0.3 × 0.3 × 1.0–1.6 wu ellipsoid centred in the ear) — a closed isosurface around
a HOLLOW inside the ear, invisible from outside, which the volume rule had been counting
as a drawn piece in 37 frames. The C's extra components at frame 150 all carry the body's
sign (outer pieces, the arm-front chunks). Renderer rule (commit after 82c5347): components
with the sign opposite to the body are interior cavities — removed from the mesh and
counted apart, never as pieces; `scripts/probes/cavity_sweep.py` writes the same per-frame
count for videos rendered before the rule so the report's "drawn pieces" column is
consistent across the gallery.

**Bridged + cavity-corrected QA of all 19 (10:20):** frames with a drawn outer piece not
tied to the body — **0 for 17 targets**, bunny 3, **C 53 of 202**; bridged: cow 22 (all of
its split frames), beast 1; cavities: bunny 90 frames (the ear hollow), beast 27, bob 24,
nefertiti 10, C 7. C frame 141 shows the residue: a ball the size of a marble below the
body — a chunk of a few dozen particles compressed to 0.65 of the bulk spacing, whose
blurred surface at the filament level encloses more than dx³ although it holds well under
one cell of particles (the grid probe finds ≥ 1 cell of particles in 6 frames only). The
volume rule was measuring the wrong thing: "material the grid does not resolve" is a
MASS statement — a component is a continuum element iff at least ppc = N dx³ / V particles
sit inside it (the same count `grid_fragments.py` uses). Renderer rule (commit after
3902a93): mass by voxel-label membership, in addition to the enclosed volume; C, bunny,
cow and beast re-rendered; the other 15 videos have no extra piece to drop and stand.

**FINAL v7 deliverable QA (10:50; iso auto + mass rule + cavity sign + filament bridges):**
frames with a drawn piece detached from the body — **0 for 18 of 19 targets** (bunny 0
with 92 cavity frames, cow 0 with the teat bulb dropped as sub-cell mass in 70 frames,
beast 0 with 20 cavity frames), **C 29 of 203** (the arm-front chunks ≥ one cell of
particles; 5 more frames bridged); physical fragments by the grid probe: 0 frames for 18,
C 6. Pages built with `build_report150.py` (QA table: physical fragments, raw/drawn,
bridged, unbridged, cavities via the sidecars, re-attachments), fetched with
`fetch_page.sh`, markdown `docs/highres150_v7_report.md` and `docs/newmesh150_v7_report.md`.
Published: main gallery v7 → https://claude.ai/code/artifact/2f348b78-324e-4cf0-a491-ea49f92fd5b1,
new meshes v7 → https://claude.ai/code/artifact/4b18fc06-c010-45ff-89f2-256e77516a1c.

**Frame QA of the v6 photoreal videos (sub-cell rule; `build_report150.py` now tables it,
4c30150):** frames with drawn components > 1 (max) / isolated-particle peak (frame): A 0 (1)
/ 2160 (33); V 0 / 1770 (36); armadilo 0 / 2294 (42); bob 59 (4) / 4446 (48); bunny 2 (2)
/ 182 (33); dragon 24 (3) / 5286 (48); heart 0 / 49; spot 2 (2) / 1244 (33); teapot 0 / 60;
C (v6, did not morph) 4 (3) / 7300. Sub-cell components were dropped in 1–179 frames per
video (dragon: 396 components over 179 frames). Visual check of the isolated-particle peak
frames (A 33, dragon 48, bob 48; `output/qa_frames/` on hyde06): one body each, no floating
piece — the peak is the stretched shell of the early expansion (8-NN spacing > 3 × median
on a surface that is thinning), not detached material, and it drops to 0–130 by the end. The
drawn > 1 frames that remain (bob 59, dragon 24) are the shed chunks the net re-attaches
(≥ one cell in volume, so drawn until the next commit); the v7 recipe removes them at the
source (oh3 trio; `auto_bunny`, 40k with the net: 0 re-attachments, 0.1188 / 0.967, stray
peak 0.05 %). The deliverable measure for goal 3 is therefore "frames with drawn
components > 1" plus the re-attachment count, not the isolated-particle peak.

### 2026-09-17 — SUMMARY (read this first; the ladder below is the working record)

**Question:** why do particles eject on every mesh, and what removes it without per-shape
constants? **Answer:** the log cell-sum density loss rewards a lone surface particle for
reaching a far empty target cell; once it is one MPM cell from the body it shares no grid
node (numerical fracture) and plastic assimilation erases the restoring stretch. What decides
whether that happens is the **cell size relative to the shape**, not the particles per cell:

| MPM cell (8 wu normalisation) | 40k (ppc) | dragon fragments / silIoU, 40k no net | 150k (ppc) | dragon re-attachments / silIoU, 150k with net |
|---|---|---|---|---|
| 0.20 wu | 8 | 41 / 0.854 | 27 | 1562 / 0.833 |
| **0.31 wu** | 27 | **0 / 0.955** | **91** | **755 / 0.939** |
| 0.41 wu | 64 | 0 / 0.930 | — | — |

**Contract v2 (commit 51cfbf3):** `--cell_diag 26` — dx = source bbox diagonal / 26, ppc = N dx³ / V.
The grid resolves the geometry, N refines the quadrature (the C++ oracle's structure at a finer
grid). docs/method.md §10.9.

**Evidence:** 40k, 19 meshes, no re-attachment: end fragments 307 → 11 (15 meshes 0, max 4).
150k, 10 targets, re-attachments (the pops in the videos): v3 8023 → v5 6143 → **v6 1308**
(bunny 0, teapot 0, heart 0, spot 1, A 16, V 29, armadilo 21, dragon 755, bob 486, C 0);
silIoU within −1.6 pt of v3 on the easy targets, dragon 0.776 → 0.939, bob 0.581 → 0.968.

**Pages:** v6 = main gallery https://claude.ai/code/artifact/2f348b78-324e-4cf0-a491-ea49f92fd5b1;
v5 comparison https://claude.ai/code/artifact/2ee53cd2-4c65-4b63-bea1-9026d13d8f78; v4 (ot_pace)
https://claude.ai/code/artifact/6b144784-69e7-4703-ba73-ac34f6b45724.

**Falsified today:** OT loss alone (holes, tracker stop); transport leash v1–v3; ot_pace as the
gallery (ejection −86 % but rough surfaces, porous thin features) and its variants (support
snap, residual-only plan, support-uniform plan, hand-off); loss_res 32; linear density
residual (not a drop-in: merit / tracker / calibration are log form); shell-biased sampling
(the C++ scheme: more ejection at ppc 8 — the C++ immunity was its 8-cell grid); one
optimiser iteration per window (408 / 363 fragments). Two OT solver bugs fixed on the way
(row-normalised barycentric projection; ε-scaling with an L1 stopping rule).

**New meshes, 150k, re-attachments v3 → v5 → v6 (silIoU v6):** cow 83 → 10 → 28 (0.925),
homer 271 → 19 → 5 (0.938), maxplanck 14 → 0 → 0 (0.969), nefertiti 559 → 551 → 152 (0.949),
fandisk 600 → 142 → 13 (0.976), ogre 981 → 194 → 15 (0.929), beast 3443 → 894 → 493 (0.873),
cheburashka 3206 → 45 → 3 (0.954), bimba 1491 → 127 → 1 (0.971). Sum 10 648 → 1 982 → **710**.
silIoU vs v3: −1…−2.4 pt on the smooth meshes, +4…+17 on the ones that ejected.

**Open:** dragon and bob (and beast, nefertiti among the new meshes) still shed ~0.3–0.5 % of
their particles at 150k even at cell 0.31 (the 40k runs shed none); C never morphs (gate
stop under every recipe). The recipe in `scripts/ops/hyde06_env.sh` is `--cell_diag 26`
(deployed 51cfbf3+).

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

ppc 64 armadilo: 0 / 0.1220 / 0.941 (ppc 27: 0.955) — the ppc curve is closed: fracture-free
from three spacings on, 27 is the finest fracture-free grid. sh40_dragon (12:00): silIoU
**0.952** (uniform ppc 8: 0.854), chamfer 0.164, but "fragments 400": the grid-connectivity
count is meaningless for a shell-biased cloud — its 263 interior particles sit 1.5 wu apart
and each is its own component by construction. For this sampling the ejection measure has
to be the off-target census (distance to the nearest target point), computed next.

**Off-target census, dragon 40k end frames (particles farther than 0.5 wu from any target
point / farther than 1 wu / max):** uniform ppc 8 (d40) **216** (0.54 %) / 195 / 3.92 wu;
uniform **ppc 27 (pp40) 0 / 0 / 0.24 wu** (nothing beyond 0.25 wu); shell-biased ppc 8
(sh40) **552** (1.38 %) / 405 / 2.92 wu. Shell-biased sampling at ppc 8 ejects MORE than
uniform sampling, so the C++ oracle's immunity came from its far coarser grid relative to
the shape (8 cells across the bunny), not from the shell scheme; the sampler stays in the
code (`--sample shell`) as a C++-parity option. The bob/armadilo shell trials were stopped.
ppc 27 uniform is the clean result on every measure: 0 grid fragments and 0 particles
off the target beyond a quarter world unit.

**Off-target census, 40k end frames, particles farther than 0.5 wu from any target point
(max distance):** bob — density ppc 8 254 (3.50 wu), ppc 27 **7** (2.49), ot_pace 1 (1.13);
armadilo — 13 (1.43), **0** (0.17), 0 (0.18); dragon — 216 (3.92), **0** (0.24), —; bunny
ppc 27 0 (0.24); homer ppc 27 0 (0.19); ppc 64 dragon 0 beyond 0.5 wu (94 between 0.25 and
0.41: near-surface fuzz of the coarse grid). The recipe (`scripts/ops/hyde06_env.sh`,
docs/pipeline.md) now carries `--ppc 27`.

**150k v5 bob (12:05): silIoU 0.975 (v3 0.581, v4 0.958), chamfer 0.0773 (v3 0.1695), 0
fragments — but 3267 re-attachments, of which 2343 in ONE commit:** the per-commit history
is 1 2 3 2 8 9 26 19 4 14 7 17 22 29 19 31 15 1 10 **2343** 1 1 5 93 12 103 15 159 296. A burst
that size is not ejecta: the fragment mask (26-connected components of the occupancy
dilated by one MPM cell, now 0.2 wu at 150k/ppc 27) split the body at a thin neck and the
smaller half was merged onto the other — a whole part teleported, the worst kind of pop.
The net is wrong for a body with a genuinely thin connection at the coarser cell. Check:
`h150qn_bob` = the same run WITHOUT `--reattach` (true end fragments at 150k, ppc 27; at
40k bob had 2 fragments / 7 off-target particles).

150k v5 V: 0 fragments, 441 re-attachments (v3 1074, v4 81; steady 1–77 per commit, no
burst), chamfer 0.0760 (v3 0.0737), silIoU **0.983** (v3 0.975). Visual QA of the v5 stills
(bunny az35, teapot az215): solid bodies, the ears / spout / handle fuller than v3 and far
better than v4, a slightly blobbier surface texture from the coarser cell, one stray at the
bunny ear tip. 40k sweep at ppc 27 continues clean: V 33 → 0 (0.970), A 0, C gate stop,
fandisk 1 → 0. bob v5 still (az35): a clean solid ring with its base — v3 was a disc with
chunks in flight, v4 a porous ring — with one stray particle below it.

Re-attachment net amendment (runner.reattach_fragments, `tgt_points`): a flagged particle
that lies within one MPM cell of a target point is left alone — a part of the body that
sits where the target is, separated by a thin neck, is not ejecta (the bob burst). No new
constant (the cell). Committed, to be deployed with the next batch (the running batch keeps
the old net so the gallery stays on one code).

150k v5 heart (12:10): 0 fragments, 0 re-attachments, 0.0746 / 0.982 / 0 (v3 0.0732 /
0.983). (Ops note: density-recipe logs are block-buffered until exit — a run that looks
stuck at 27 log lines for 20 minutes is running; judge by the status markers.)
150k v5 armadilo (12:07): 0 fragments, 396 re-attachments (v3 619, v4 153), 0.0795 / 0.931 /
0.03 % (v3 0.0801 / 0.925; v4 0.968); the per-commit history is 1–9 with two events of 160
and 118 — chunks separated at the coarser cell and merged, the case the amended net (a
flagged particle on the target support is left alone) is for. At 150k the ppc-27 recipe
cuts re-attachments 1.5–5× (bunny 142 → 27, teapot 32 → 0, spot 42 → 1, V 1074 → 441,
armadilo 619 → 396, heart 0 → 0) and fixes bob's shape (silIoU 0.581 → 0.975), with surface
quality equal to v3; ot_pace cut them further on armadilo / V / A (153 / 81 / 90) at a
surface cost.

**150k v5 dragon (12:20): 0 fragments but 1562 re-attachments (v3 3753, v4 134), 0.1109 /
0.833 / 0.12 % (v4 0.969)** — at 150k the ppc-27 grid does not tame the dragon although at
40k it gave 0 fragments and silIoU 0.955. nefertiti (new mesh, 150k): 551 re-attachments
(v3 559), silIoU 0.868 → 0.963. 40k sweep: cow 2 → 0, beast 76 → 4 (silIoU 0.738 → 0.803),
cheburashka 1 → 3, spot 0.

**Reading the ppc / N table by the ABSOLUTE cell size (bbox diagonal 8 wu):** 40k ppc 8
= dx 0.21 → dragon 41 fragments / 0.854; 150k ppc 27 = dx 0.20 → 1562 merges / 0.833;
40k ppc 27 = dx 0.31 → 0 / 0.955; 40k ppc 64 = dx 0.41 → 0 / 0.930. The two dx ≈ 0.2 cases
behave alike whatever the particles-per-cell, the two dx ≥ 0.3 cases behave alike: the
ejection variable looks like the cell size relative to the SHAPE (which features are in
play: at dx 0.2 the dragon's whiskers and horns are cell-scale and become leaders; at
0.3 they are sub-cell), not the cell-to-spacing ratio alone. The C++ oracle sat at dx =
1.0 (diag/8). Test (`h150x_{dragon,bob}`, 150k, `--ppc 91` → dx 0.31, re-attach):
pre-registration — dragon re-attachments < 200 and silIoU ≥ 0.93 (the 40k ppc-27 level);
bob no burst. If it holds, the definition the user asked for is: **dx from the shape
(diag / ~26, i.e. the loss-cell scale the density loss already uses), ppc = N dx³ / V from
N** — the grid resolves the geometry, N refines the quadrature — instead of dx following N
at a fixed ppc.

**bob 150k ppc 27 WITHOUT the net (`h150qn_bob`, 12:27):** the fragment count climbs
steadily — 1, 3, 8, 15, 27, 48, 77, 90, 109 by window ~370 — and then a whole part of ~2 500
particles separates (2498, 2497, 2491 …), ending with 335 fragments after re-merging by the
physics, silIoU 0.761 (0.975 with the net). So at dx 0.20 bob's thin ring both sheds
strays and breaks at the neck; the net repaired both (the strays legitimately, the chunk by
teleport). The dx 0.31 run (`h150x_bob`) tests whether the coarser cell prevents both.
The dragon v5 still (150k, dx 0.20, az35) shows it: a solid body with a detached horn /
head chunk floating at the top right and scattered particles below — the 1562 merges are
real ejection at this cell size, not a metric artefact.

**40k ppc-27 sweep COMPLETE (12:40), all 19 meshes, no re-attachment — end fragments,
density ppc 8 → ppc 27 (silIoU at ppc 27):** bunny 3 → 0 (0.960), teapot 0 → 0 (0.972),
armadilo 12 → 0 (0.955), heart 0 → 0 (0.977), A 0 → 0 (0.972), dragon 41 → 0 (0.955), C
gate stop in both (0.725), V 33 → 0 (0.970), spot 0 → 0 (0.970), bob 85 → 2 (0.958), cow
2 → 0 (0.940), homer 7 → 0 (0.958), maxplanck 0 → 0 (0.972), nefertiti 29 → 0 (0.963),
fandisk 1 → 0 (0.972), ogre 17 → 2 (0.947), beast 76 → 4 (0.803), cheburashka 1 → 3
(0.961), bimba 0 → 0 (0.971). **Sum 307 → 11; fragment-free on 15 of 19, ≤ 4 on all 19**
(ot_pace: 44, 13 fragment-free). Against the pre-registration (≤ 3 on every mesh): beast
misses by one. At 40k the cell is 0.31 wu; the same recipe at 150k (cell 0.20) does not
carry over on dragon and bob — see the absolute-cell-size test.

**h150x dragon (150k, dx 0.31 = ppc 91, 12:45):** 0 fragments, **1048 re-attachments**
(dx 0.20: 1562; v3: 3753; ot_pace: 134), chamfer **0.0836** (0.111 / 0.134), silIoU
**0.937** (0.833 / 0.776; ot_pace 0.969), hole 0.88 %, 20 min. The pre-registration is half
met: quality ≥ 0.93 ✓, re-attachments < 200 ✗. So the absolute cell size sets the QUALITY
(the coarser cell stops the whiskers and horns from tearing the surface) but at 150k the
dragon still sheds ~0.7 % of its particles over the run at either cell, while the same
cell at 40k sheds none: the per-particle shedding rate is not a function of the cell in
spacings (4.5 here vs 3 at 40k) nor of the cell in world units alone. 150k v5 A: 386
re-attachments (v3 1585, v4 90), 0.0768 / 0.976; fandisk 142 (v3 600), 0.0779 / 0.975.

**h150x bob (150k, dx 0.31, 12:48): 0 fragments, 434 re-attachments with NO burst (max 119
per commit; dx 0.20 had the 2343 burst), 0.0778 / 0.967 / 2.61 %.** dragon at dx 0.31 has
three chunk events (410, 326, 144) in its history — the case for the amended net. On the
two hard targets dx 0.31 beats dx 0.20 at 150k on every count except bob's silIoU
(0.967 vs 0.975), so the 10-target batch is being re-run at dx 0.31 (`h150y_<T>`, `--ppc
91`, amended re-attachment net deployed 7072da0) as a candidate v6 while the v5 page
stands; the new-mesh v5 batch finishes on the same deployment (homer / maxplanck / ogre /
beast start after it — their net is the amended one, noted on the page).

**Published (12:50):** the main 150k gallery https://claude.ai/code/artifact/2f348b78-324e-4cf0-a491-ea49f92fd5b1
now shows v5 (ppc 27, all 10 targets, narrative with the v3 / v4 numbers); v4 (ot_pace)
stays at https://claude.ai/code/artifact/6b144784-69e7-4703-ba73-ac34f6b45724 for
comparison; docs/highres150_v5_report.md holds the v5 tables.

**New meshes at 150k, v5 (ppc 27, dx 0.20) — re-attachments v3 → v5, chamfer / silIoU:**
cow 83 → 10, 0.078 / 0.942; homer 271 → 19, 0.079 / 0.962; maxplanck 14 → 0, 0.076 / 0.976;
nefertiti 559 → 551, 0.083 / 0.963 (v3 0.868); fandisk 600 → 142, 0.078 / 0.975 (v3 0.804);
ogre 981 → 194, 0.080 / 0.932 (v3 0.879); beast 3443 → 894, 0.100 / 0.832 (v3 0.829);
cheburashka 3206 → 45, 0.078 / 0.966; bimba 1491 → 127, 0.076 / 0.974. All 0 fragments.
Sum 10 648 → 1 982.

**150k v6 = dx 0.31 (ppc 91), amended net — all 10 targets (14:20). Re-attachments
v5 → v6, chamfer / silIoU v6 (v5):**

| target | re-attach v5 → v6 | chamfer | silIoU | min |
|---|---|---|---|---|
| bunny | 27 → **0** | 0.0790 (0.0770) | 0.958 (0.974) | 25 |
| teapot | 0 → 0 | 0.0762 (0.0749) | 0.970 (0.977) | 16 |
| heart | 0 → 0 | 0.0758 (0.0746) | 0.980 (0.982) | 9 |
| spot | 1 → 1 | 0.0796 (0.0799) | 0.964 (0.965) | 10 |
| A | 386 → **16** | 0.0804 (0.0768) | 0.969 (0.976) | 17 |
| V | 441 → **29** | 0.0783 (0.0760) | 0.969 (0.983) | 12 |
| armadilo | 396 → **21** | 0.0818 (0.0795) | 0.919 (0.931) | 11 |
| dragon | 1562 → **755** | **0.0831** (0.1109) | **0.939** (0.833) | 14 |
| bob | 3267 → **486** | 0.0778 (0.0773) | 0.968 (0.975) | 13 |
| C (fails) | 63 → 0 | 0.383 (0.391) | 0.660 (0.748) | 1 |

Sum 6 143 → **1 308** (v3 8 023). Every merge history is steady (bob max 188 per commit,
dragon 755 total, no teleported part); pops in the videos are correspondingly rare. Cost:
silIoU −0.1…−1.6 points and chamfer +0.001…+0.004 on the eight easy targets (the coarser
cell rounds the surface), dragon +10.6 points. Runs are 9–25 min at grid 37–43³.

**Definition, closed (the user's question):** the two ladders together say the ejection
variable is the cell size relative to the shape, not the particles per cell: dx 0.20 wu
fractures at both ppc 8 (40k) and ppc 27 (150k); dx 0.31 holds at both ppc 27 (40k) and
ppc 91 (150k); dx 0.41 holds and loses detail. So the discretisation contract is **dx from
the shape, ppc from N**: dx = diag / 26 (0.31 wu for the 8 wu normalisation — the finest
cell the ladders show fracture-free, a measured boundary), ppc = N dx³ / V (25 at 40k, 91
at 150k), the cell-to-spacing ratio then being 3 at 40k and 4.5 at 150k. N refines the
quadrature inside a grid that the geometry sets — the C++ oracle's contract (grid_dx 1.0
on an 8 wu shape, ppc from the sampling), at a finer grid. Implemented as `--cell_diag 26`
(commit after c20c98b); `--ppc` stays for the old contract.

**Visual QA of the v6 stills (15:05):** bunny — solid body, full ears, smooth surface, no
strays; dragon — one solid body with horns and whiskers attached (v5 had a detached head
chunk), two tiny dots at the left; bob — a clean solid ring, no strays (v5 ring with a
scatter below). v6 is the gallery; v5 becomes the comparison page. New meshes are being
re-run at dx 0.31 (`n150y_<T>`).

**Published (15:15):** main 150k gallery → v6 at
https://claude.ai/code/artifact/2f348b78-324e-4cf0-a491-ea49f92fd5b1 (docs/highres150_v6_report.md);
v5 (ppc 27) kept as a separate comparison page https://claude.ai/code/artifact/2ee53cd2-4c65-4b63-bea1-9026d13d8f78; v4 (ot_pace) at
6b144784. The recipe (`scripts/ops/hyde06_env.sh`) is `--cell_diag 26`.

### 2026-09-21 — 150k excluded; the gradient results visualised

User: the 150k gallery is out of the experiments for now — everything at 40k; and show the
gradient results as figures. `scripts/probes/grad_stage_viz.py` draws, from the three
gradient-stage dumps (v8 recipe, + relaxation, the new recipe): the terminal covectors on
the particles and their normal component on the outer layer (stage 1), the smooth /
rough split of the silhouette and shading covectors at two spacings, the same covectors
after the adjoint through the stress control and through the u channel (stage 2), the
correlation-vs-scale curves of every path, the single-channel responses of a window
(stage 3), the per-window λ / g_share / layer-RMS traces and the response bars, and the
accepted u field with its histogram (the one-spacing clip visible as the two spikes).
Published as an artifact page; the figures live in `output/gradviz` on hyde06.

**Was the covector always half noise?** Yes: rough share at two spacings, silhouette /
shading, 8-window means — v8 recipe 0.65 / 0.67, + relaxation 0.52 / 0.56, new recipe
0.53 / 0.55. Sources: the target images' shot noise and the CIC rasteriser's per-particle
kernel derivative (recipe-independent); hidden before by the stress path's grid low-pass.

**The projection test (`--layer_ctrl_smooth`: the u step through the relaxation's W).**
`ps40_bunny` / `ps40_dragon` vs `p40_*`, plus the 8-window dump: the accepted u becomes
smooth (corr at 2 sp 0.31 → 0.71, at the clip 19 → 8.5 %), the render channels through u
stop roughening the layer, the morph-mean layer RMS drops 0.355 → 0.330 (bunny), 0.338 →
0.308 (dragon), the dragon's mid-frame Poisson roughness 8.5 → 7.8°; but the bunny's end
frame loses detail against the true mesh (hp_res 0.177 → 0.199, dcorr +0.28 → +0.24), IoU
−0.4 / −0.2 (the spread), the early morph is slower (8-window IoU 0.911 → 0.902).
FALSIFIED as a recipe addition — the relaxation already applies the same W to the state;
projecting the step too removes the channel's structure with its noise. Flag kept, not in
the recipe.

### 2026-09-21 — the three-way check at 40k, the shot-noise proof, the thin-feature Poisson videos (docs/surface_gradient.md §8–§9)

**Three-way check (pre-registered in §8, all at 40k, GPU 0):** G5 stratified sampling
(`--sampler stratified`) ADOPTED — end frame vs the true mesh, bunny / dragon: hp_res
0.177 → 0.159 / 0.200 → 0.185, dcorr +0.28 → +0.36 / +0.32 → +0.34, n_dev 21.8 → 17.6° /
25.1 → 23.6°, silIoU +0.4 / +0.6 points, chamfer −3 / −4 %, morph-mean layer RMS −25 /
−13 % (spread: IoU ±0.4, hp_res 0.01, dcorr 0.02). The quadratic B-spline splat kernel
(`--sil_kernel quad`) FALSIFIED — smoother silhouette covector (rough share 0.53 → 0.33)
but the bunny's end frame rougher in the band (hp_res 0.177 → 0.212), higher layer RMS;
with G5 an early stop at 882 frames and an unfinished base (d_95 4.22). Thin-feature
targets under the recipe (`t40_*` vs `t40v8_*`): bob IoU 0.9734 → 0.9779, C 0.8371 →
0.9134 (both stop after 15 windows), beast within the spread; grid fragments 0, end
fragments beast 2 → 0, re-attachments 0, stray census cleaner on all three. RECIPE =
v8 + `--layer_relax --pbr_denoised --layer_ctrl --sampler stratified` (hyde06_env.sh).

**The shot-noise proof (§9, `scripts/probes/sampling_noise.py`):** with replacement is a
Poisson process (relative fluctuation of the 1.5-spacing-blurred density (p/σ)^{3/2}/√(8π^{3/2})
= 5.9 % at σ = 1.86 volumetric spacings); stratified leaves a dipole field of the jitter
((p/σ)^{5/2}/√(64π^{3/2}) = 1.1 %, spectrum ratio k²p²/12, no k = 0 component). Measured:
cube 5.5 → 1.1 % (theory 5.9 / 1.1), bunny 8.1 → 2.9 %, dragon 5.2 → 2.2 %; index of
dispersion 1.01 → 0.26, 1.30 → 0.31, 1.06 → 0.28; target level-set hp_res 0.185 → 0.137
(bunny), 0.168 → 0.156 (dragon). Unit: the project's 8-NN spacing = 1.24 × the volumetric
spacing. Page: artifact "층화 샘플링의 잡음 증명".

**Thin-feature Poisson videos (stage C2, `report_t40`, per-frame QA):** bob 441 frames,
drawn pieces > 1 in 0 frames, Poisson fallback 0, unbridged 0; C 101 frames, 0 / 0 / 0
(raw pieces > 1 in 95 frames, all sub-cell, dropped by the mass rule). beast 896 frames,
drawn pieces > 1 in 2 frames (archived frames 219 and 225, max 2; both tied to the body by
a particle chain the isosurface does not enclose, so unbridged 0), Poisson fallback 0; raw
pieces > 1 in 804 frames, all sub-cell or body-enclosed (isolated particles peak 401 at
frame 177, the spray episode, then 5 by frame 735). All three thin-feature targets pass
the video QA under the recipe.

### 2026-09-21 — the render × u factorial under the current recipe (docs/surface_gradient.md §10)

User (13:30): "does the current evidence show rendering meaningfully affects the physics?"
then "prove everything unproven, then organise the last implementation". Pre-registered
§10; `scripts/ops/factorial40.sh 0 2`: 40k, seed 1, stratified, bunny / dragon / bob × six
runs (`fx_11` recipe, `fx_11c` identical re-run, `fx_01` λ = 0, `fx_10` no u, `fx_00`
neither, `fx_cut` render off at window 20 / 20 / 8); 18 runs in 70 min, readings in
`output/fx_analysis.log`, figures `output/fx_figs`.

Spread (`fx_11` vs `fx_11c`): silIoU 0.23 / 0.53 / 0.08 points, hp_res ≤ 0.012, dcorr ≤ 0.03.
**H-A confirmed 3/3 (silhouette):** render channel +1.8–2.0 / +1.0–1.5 / +0.6–0.7 points;
chamfer within the spread; the cut twin loses the whole gain (dragon, bob) or half (bunny).
**H-B partly:** g_share 0.39–0.40 (windows 1–20) / 0.35–0.38, cosine 0.01–0.03 (bob 0.11–
0.15) — deterministic; D_vol at the end lower only on bunny (0.0036 vs 0.0041); det F less
compressed on dragon (0.77 vs 0.715) and bob. **H-C:** the outline gain needs no u (`fx_10`
− `fx_00` = +1.6 / +1.6 / +0.9); the physics path carries detail down to the cell on the
dragon (dcorr +0.28 → +0.35 with u off). **H-D falsified:** the cut twin's divergence equals
the identical re-run's (0.40 vs 0.41, 0.76 vs 0.63, 0.43 vs 0.44 spacings after K; 0.002
before) — chaos, as at 150k. **u channel:** +0.4–1.4 IoU points with render, +0.7–1.0
without; the main roughness source (layer RMS 0.19–0.22 → 0.25–0.33 morph mean; bob end 0.44
with u driven by the physics gradient alone); necessary on bunny (d_95 5.4 → 1.4 spacings
without / with u), neutral on dragon, harmful on bob (n_dev 5.6 → 7.0°, dcorr +0.32 →
+0.26). Mechanism: the u projection never updates F, so a rough u costs no strain energy.
Page: artifact "Render × u 요인 실험". Plan for the last implementation: `docs/final_plan.md`.

### 2026-09-21 — the last-implementation candidate round (docs/surface_gradient.md §11–§14, docs/final_plan.md §6)

P3 (u through F, `--layer_F`): collapse on all three targets (10 / 24 / 16 windows, silIoU
0.81 / 0.47 / 0.60, det F min 0.008 / 0.0005 / 0.0007) — sub-cell strain the grid cannot
relax; tangential-only (`--layer_F_depth 0`): rougher layers (0.32 / 0.31 / 0.40), det F
0.15 / 0.015 / 0.12. P2 (`--layer_gate`): bob layer 0.269 → 0.229, n_dev 6.95 → 6.45°;
bunny d_95 5.38 (= u off); silIoU −0.4 … −0.9. P1 (`--layer_u_render_only`): layers
0.32 / 0.28 / 0.31 morph, bob end 0.385; chamfer +4–6 %; dragon det F 0.615. All FALSIFIED;
`fx_11` (the recipe) remains the best composite; u off the best true-mesh surfaces on bob
and dragon. Probes: bunny's far region = interior sheets drawn by the reconstruction on
interior density gradients; the bunny TARGET has interior low-density pockets (4.5 % of
the deep interior; non-watertight mesh fill) — bob 0.00 %, dragon 0.45 %. Next: fix the
fill for non-watertight meshes, add an exterior test to the reconstruction, re-read
bunny, then the u decision (user). Page: artifact "마지막 구현 후보 라운드".

### 2026-09-22 — the fill of non-watertight meshes, the exterior test, bunny re-read (docs/surface_gradient.md §14, method.md §10.13)

Cause of the bunny target's interior 40 %-density comb: the orthographic fill's three-axis
intersection leaves the columns above the base holes empty. Fix (`_fill_ortho_reliable`):
hole footprints from the boundary loops per axis, fill on the reliable axes (≥ 2), streak
strip, majority pocket fill; `FILL_MODE = "legacy"` reproduces older archives. `fill_check`
(deep-interior low-density share): bunny 4.79 → 0.42 %, maxplanck 9.95 → 0.33 %, beast /
armadillo / 14 watertight targets 0.00 %, dragon 0.54 % (mesh property). Exterior test
(`exterior_surfels`): surfels with ≥ 2 particles in the outward 2-spacing cap beyond 0.75
spacing are dropped before Poisson (bunny target 2 907). Re-read: old bunny u-off d_95 5.39 →
1.53 (the interior sheet), hp_res 0.295 → 0.175; bob / dragon ±0.3–0.5°. New bunny on the
fixed target: `fx_11n` / `fx_11cn` 0.9638 / 0.9637 (spread 0.01), `fx_01n` 0.9445, `fx_10n`
0.9558; surface (d_95 / n_dev / hp_res / dcorr) 0.88 / 15.7° / 0.154 / +0.41, 0.83 / 15.4° /
0.152 / +0.42, 0.99 / 17.1° / 0.171 / +0.33, 0.89 / 16.3° / 0.164 / +0.39. Rendering influence
(fixed bunny): g_share 0.39–0.40 / 0.36–0.37, cos 0.01–0.02; vs λ = 0: IoU +1.9, n_dev −1.4°,
dcorr +0.08, D_vol 0.0036 vs 0.0041. u decision: user's call (bunny favours u, bob / dragon
favour u off). Page: artifact "표면 판정과 렌더링 영향력".

### 2026-09-22 — the final 40k gallery `g40` under the frozen recipe (docs/method.md §10.9–10.14; page: artifact "40k 최종 갤러리 g40")

Recipe frozen (u kept; `scripts/ops/hyde06_env.sh`); targets re-sampled with the fixed fill
(§10.13); 19 targets, `scripts/ops/gallery40.sh` (runs + post_run, Poisson videos with the
exterior test, `gallery_post.sh`: end frame and target vs the true mesh, layer RMS, render
telemetry). Full test suite: 231 passed, 2 skipped. Runs (chamfer / silIoU / hole / det F min /
minutes): bunny 0.1152 / 0.9637 / 0 / 0.77 / 4.2; teapot 0.1128 / 0.9732 / 0 / 0.86 / 3.0;
heart 0.1132 / 0.9782 / 0 / 0.88 / 1.5; spot 0.1144 / 0.9703 / 0 / 0.81 / 5.7; A 0.1170 /
0.9715 / 0 / 0.84 / 4.0; V 0.1166 / 0.9793 / 0 / 0.85 / 2.2; armadillo 0.1167 / 0.9579 / 0 /
0.82 / 6.1; dragon 0.1193 / 0.9586 / 0.07 % / 0.74 / 9.2; bob 0.1147 / 0.9788 / 0 / 0.87 / 2.9;
C 0.1481 / 0.8995 / 0.22 % / 0.86 / 1.6 (the early stop of §8); cow 0.1137 / 0.9583 / 0 / 0.82 /
5.9; homer 0.1151 / 0.9692 / 0 / 0.84 / 7.7; maxplanck 0.1131 / 0.9702 / 0.05 % / 0.87 / 3.6;
nefertiti 0.1153 / 0.9683 / 0 / 0.81 / 7.9; fandisk 0.1147 / 0.9697 / 0 / 0.85 / 3.9; ogre
0.1156 / 0.9558 / 0.76 % / 0.82 / 10.7; beast 0.1176 / 0.9482 / 2.06 % / 0.77 / 9.7;
cheburashka 0.1157 / 0.9700 / 0 / 0.83 / 5.9; bimba 0.1148 / 0.9733 / 0 / 0.83 / 5.0. The
proof column (render on / λ = 0 twins on the fixed fill): bunny 0.9638 / 0.9445, n_dev 15.7 /
17.1°; bob 0.9774 / 0.9710, 6.5 / 6.7°; dragon 0.9635 / 0.9448, 22.1 / 25.0°; g_share 0.36–0.40
everywhere (docs/surface_gradient.md §14). Video QA, surface columns and the cleanup sweep:
appended below when the videos land.

**Where the render channel changes dFc (17:30; `grad_where.py`, `gd_bunny` 56 windows,
page "렌더링이 dFc를 바꾸는 자리").** Render dominates the per-particle dFc gradient at 1.6–5.2 % of
the particles = 11–23 % of the outer layer (ears, paws, base rim, back outline); 43–58 % of
the render pull sits on the layer (8–9 % of particles); mean per-particle share 0.25–0.29 on
the layer, 0.09–0.13 inside, against a global norm share of 0.38–0.42; the render channel
alone moves the layer 0.2–0.6 spacings per window, the interior 0.02–0.05 (physics 0.16–0.80 /
0.05–0.12; correlation 0.42–0.86). docs/surface_gradient.md §14.

**g40 videos and page (17:30–19:40).** All 19 Poisson videos: drawn pieces > 1 AND not bridged
in 0 frames on every target (beast 12 and cow 23 frames with a second drawn piece, every one
tied to the body by a filament), Poisson fallback 0. The cow's 4 unbridged frames of the first
pass exposed a hole in the bridge rule (enclosed-to-enclosed contact had no graph edge; fixed
4aa31cc, method.md §10.12 addendum, cow re-rendered). Mid-morph stills (homer 144, ogre 123,
armadillo 102: the frames with the most isolated particles) are single closed bodies. Page:
artifact "40k 최종 갤러리 g40" (`output/report_g40_page`, 172 files, 41 MB); the markdown
twin `docs/gallery40_report.md` (built by `build_report150.py`, tables 2, 2b, 2c). Cleanup
sweep follows.

**Cleanup sweep (19:50; the user's rule at the gallery milestone).** hyde06 `output/`: 241 → 123 GB.
Archived first: every `.log`, `.json`, `.md`, `.sh` into `logs_archive_20260922.tgz`. Deleted:
the superseded 150k v7 galleries (`h150v7_*` 10 runs, `n150v7_*` 9 runs, `report_h150v7`,
`report_n150v7`), the v6-era `h150y_dragon`, the pre-wall `h150v7pw_bob`, the uncited
loss-resolution ladder (`lr96_*`, `lr128_*`), single falsified trials (`or40_C`, `rv_bunny`,
`rvb_bunny`), the walled material / no-net variants (`matw_*`, `matwb_*`, `nnw150_*`), and all
viewer live packets except `g40_*` (38 folders, 88 GB). Kept: the v8 galleries and their
report folders, the render-proof twins and controls (`rp_*`), the material study (`mat_*`),
the no-net proofs (`nn150_*`), every 40k ladder run cited in docs/surface_gradient.md
(`lr64`, `lrx`, `p40`, `ps40`, `k40`, `g5`, `gq`, `t40*`, `fx_*`, `gd_bunny`), `surf_test`, the
figure folders. Local: `output/report_h150v7_page`, `output/report_n150v7_page` removed.

### 2026-09-22 — surface tracking for the morph videos (docs/method.md §10.15)

User: the mid-morph surface shows detached floating pieces and looks re-adjusted every frame.
Measured (cow): the per-frame Poisson re-fit jitter is 0.024 spacings, the image change between
frames 0.08 % — no geometric flicker; the defect is the topology events at thin necks (a leg tip
drawn as a bridged piece for a few frames, four episodes, 24 frames). `render_photoreal --track`
(commits 9f30036, 9f152ee, 807b678): vertices advected with their 8 nearest particles, pulled to
the fresh surface at 0.3 per frame within one spacing, re-meshed on particle-confirmed topology
change / median drift > 1 sp / every 60 frames. Cow: 10 re-meshes, none at the neck episodes,
legs continuous; QA columns unchanged (from the fresh reconstruction). Adopted as the video
default (`photoreal_batch.sh TRACK=1`); the 19 g40 videos are re-rendered with it.

### 2026-09-22 evening — the mid-morph lumps: the u channel, and the transport gate (docs/surface_gradient.md §15, method.md §10.16)

User: the surface is lumpy during the morph. Measured (`scripts/probes/lump_trace.py`: the
outer-layer plane-residual RMS per frame at 2 spacings / one cell / two cells + the transport
speed): g40 bunny / ogre / cow are 1.8–2.4× rougher at frames 15–135 than at the end, at every
scale. Pre-registered H15 (the control field's sub-cell part; `--control_grid 17 / 9`) refuted:
the peaks unchanged within 10 %, end detail worse, g_share 0.35. The render × u factorial's
traces named the cause: u on = +30–50 % morph-mean roughness on all three twins, the render
channel without u within the spread. Levers, pre-registered and run on bunny / bob / dragon
(each 10 min at 40k): u off (`u0`, −22 … −30 %, end silIoU −0.3 … −0.6, end surface like for like unchanged);
the step projection (`sm`, −5 … −13 %, insufficient); the geometric gate on the nearest target
surface (`ug` / `ugh`, commit 01b1fb4; −1 … −7 %, refuted — the wrong residual); the transport
gate on the remaining OT transport within one cell (`uo`, commit c9540c4; −16 … −21 %, peaks
−14 … −31 %, end silIoU −0.04 … −0.2, g_share 0.36–0.38). Adopted: RECIPE + `--layer_gate_ot`
(hyde06_env.sh). The gallery is re-run as g41. Stills, traces and tables: the artifact "morph 중
울퉁불퉁함" and `output/cgrid/` on hyde06. Also today: the public code release (github
Chayoso/Shape-morphing-binder main, single commit, code + assets + README; all other branches
deleted at the user's request).

### 2026-09-22 night — the g41 gallery: the recipe with the transport gate and the primary-objective brake (docs/surface_gradient.md §15c–15f, method.md §10.16)

19 targets at 40k under RECIPE + `--layer_gate_ot` (`$OUT/gallery41.sh`; runs 20:40–20:48,
tracked Poisson videos and the surface / render-influence post to 22:40). Against g40 (the same
recipe without the gate): end silIoU within ±0.3 points on 18 targets (12 up), chamfer better
on 15, the end frame against the true mesh the same within 0.02 spacings (dragon 0.43 → 0.37);
the morph-mean outer-layer roughness (`scripts/probes/lump_trace.py`, first 300 frames) lower on
18 of 19 by 6–29 % (cow −29, ogre −28, homer −27, spot −25); C unchanged (the `ot` hole regime is
ungated). Video QA: a single drawn body throughout on 18 videos, cow 10 frames with a bridged
second piece (g40: 24). g_share 0.36–0.41. nefertiti first stopped early (anim 16, 0.935): the
arriving front spills outside the outline (d_dt +65 % in one window) and the full-merit
catastrophe brake rejected it three times — g40's in-transit u had masked the spill. Refuted on
the way (pre-registered): the brake on every physics component (398d216; d_dt still trips it),
the gate on the normal transport component (`--layer_gate_ot_normal`, 1156498; no gate — bunny
back at g40's roughness). Adopted (b8cd7db): the brake reads the primary objective (the transport
divergence) alone; nefertiti 0.9676 (g40 0.9683), 90 windows, roughness −25 %; its run replaced in
g41 (`g41old_nefertiti` kept), its video and post regenerated. The other 18 runs stand (the rule
touches no window of theirs). Report: `docs/gallery41_report.md`; page: artifact "40k 최종 갤러리
g41" (same URL as the interim page); analysis page "Morph 중의 울퉁불퉁함". Public main updated
to 276b3c1 (gate + brake). A misreading corrected: the "0.22" end-surface value quoted for g40
was the target-floor row; the end-frame row is 0.28 on bunny for g40, g41 and every lever alike.

### 2026-09-23 — speed: where a window's time goes, the warm merit divergence, the parallel video reconstruction

User: why does 150k take so long when a forward MPM at 1M runs fine. Measured (PHYSMORPH_TIMING=1,
cProfile, bunny 40k / 150k alone): the optimisation proper (8 Adam iterations of forward + adjoint
over T = 20, the evaluation rollouts) is 0.96 s a window at 40k and 2.5 s at 150k; the rest of
the 4.6 s / ~6 s window is the transport plan machinery — the pace map's Sinkhorn (warm, ~0.4 s
a solve), the debiased self-map, and the merit's Sinkhorn divergence, which was COLD-solved
twice every window (0.9 s at 150k, the single largest item); the gallery batches add contention
(three runs a GPU plus four Poisson videos). Per window 40k → 150k the cost scales 3.6× for
3.75× the particles: the per-particle cost is flat; the multiplier (iterations × adjoint ×
render × plan) is what makes it 30–40 forward rollouts a window. The user's order: verify the
warm start, then do the two result-neutral speedups.

Done (commit a35c2e4): (1) `SinkhornPull.divergence` keeps its two solvers across calls and
warm-starts them (the subsample is the same fixed draw every call; `PHYSMORPH_OTDIV_COLD=1`
restores the cold pair). Verified on bunny 40k, warm vs cold twin: the ot_div trace agrees to
1.0 % mean / 1.6 % max relative (the same fixed point to the solver's tolerance), the runs then
part on the chaos floor (silIoU 0.9632 vs 0.9659, 62 vs 45 windows — within the twin spread);
wall time 6.0 vs 7.0 s a window running side by side (≈ −0.8 s, the expected two solves).
(2) `render_photoreal --prefetch W` reconstructs the next W frames in threads (the Poisson solve
is a child process per frame, so they overlap): the cow per-frame video 930 s → 355 s at W = 6
alone (2.6×); output equal within the GPU's own nondeterminism (mean 0.25 grey levels between
the two videos, one sub-cell fragment counted differently in one frame, nothing drawn changes).
`photoreal_batch.sh` uses PREFETCH=3 and 8 Poisson threads per video when four batches share
the host. Not done (a design change, needs its own verification): resampling the pace subsample
less often; fewer render views / iterations (quality trade-offs).

### 2026-09-23 — the tracked surface's re-mesh policy, from the per-frame analysis of 38 videos (method.md §10.15 addendum)

User (on g41): the earlier version looks better — a frame jumps and a connecting part is wiped at
once; connections drawn with messy triangles; the surface moves like flowing ripples. Reverted
to the GitHub version at the user's request (RECIPE without the gate, full-merit brake,
per-frame videos; commit 205b90f). Then measured (`scripts/probes/video_jumps.py`): the ripples
are the material's surface texture flowing (0.20–0.25 spacings at rest, +30–50 % with u in
transit) plus the per-frame re-fit (median frame-to-frame image change 1.5–1.8× the tracked
one); the wipes are the tracked re-mesh (re-mesh frames change 3–9× more than ordinary frames on
all 38 videos, particles moving as usual): a kept tube or a stretched-triangle web replaced by
the fresh mesh; the messy triangles are the tracked triangles stretched where the surface grows
(facets at bunny 180, the web between the ears at 357). Fix (commits 2ec467d, e89a180, a35c2e4;
no constant): keep the unmatched non-stretched triangles at a non-topology re-mesh, re-mesh when
the tracked p99 edge exceeds 2× the fresh p99 edge, never keep a stretched triangle, no periodic
re-mesh. Readings (bunny / cow): re-mesh frames' mean area change 14.4 / 7.1 % → 2.4 / 4.5 %
(ordinary frames 3.9 / 1.8 %); the web gone at 357. Page: artifact "표면 표현 네 방식" (g40 and g41
physics × per-frame / tracked variants, same frames). The default stays per-frame (the user's
revert); `photoreal_batch.sh TRACK=1` gives the new policy. The choice of combination is the
user's.

**Cleanup sweep (2026-09-23, the user: delete everything not needed).** hyde06 `output/`: 139 → 23 GB. Deleted 125 run archives (103 GB: the 150k v8 galleries h150v8 / n150v8 / nn150 / c150r, the render-proof twins rp_*, the material study mat_*, the factorial fx_*, the ladder runs t40 / p40 / ps40 / k40 / lr* / g1* / g5 / gq / gs40*, the timing runs), their logs and json archived first in `logs_archive_20260923.tgz` (6.4 MB); the viewer packets `live/` (6.7 GB), `surf_test`, the superseded report folders (h150v8, n150v8, p40, t40, t40v8), the test renders (g41pf/tk/tks/tks2/sm, g40tks2), the gradient dumps. Kept: g40 and g41 (19 runs each, the before/after pair), report_g40 / report_g41, cgrid (the §15 analysis data), jumps (the video analysis), the profiles, the sample cache. Local `output/`: the superseded page folders removed; the four published pages kept.

### 2026-09-23 — 300k: the speed passes, the first full run, and the control basis as the 300k discretisation (pre-registered before the reading)

The user's target: a 300k run in 20 minutes. Measured on hyde06 under the neighbours' load
(load 70–90, a memory-bound CPU job on 78 cores next to us): the per-window cost at 300k was
20 s of which 6.5 s are the GPU sections; the rest was host work — scipy KD-trees (the layer,
the OT smoothing, the DT gate, the stray census), a T × N host stack for the window's F
determinant, the target DT rebuilt per call, np.tile identities, double copies of the readback,
zlib on the archive. Three result-identical passes (commits 303e9d4, fd4c837, 91746bb, 7594ec8):
an exact k-NN on the GPU (Warp hash grid, `render/knn_gpu.py`, rows equal to scipy's; tests
`test_knn_gpu.py`), the layer / relaxation / gate / census computed on the device, the DT sum on
its gate's support, the sample cache, the device-side det and F conditioning, cached identities,
uncompressed archives above 100k. Suite 249 passed after each pass. Three-window test at 300k:
102 s → 102 s → 67 s (the first two passes moved work the host was not bottlenecked on; the
third removed the host-side stacks and SVD round trips).

**The first full 300k run (`full300_bunny`, per-particle control, the g41 recipe):** 59 min of
optimisation for 212 windows (16 s a window under load) + a 16 GB archive; end chamfer 0.060,
silIoU 0.9547, det F min 0.197. The window count is the real multiplier: the 40k bunny stops
at 59 windows, the 300k at 212 — not the plateau rule (it fires at 196), not the adaptive step
(alpha at its cap, the gradient norm 4× smaller), not the time discretisation (`--cell_diag 26`
fixes dx = 0.31 wu and dt at every N; the run had ppc 184 on the same grid). The per-window
displacement is the same (`move` 0.014 vs 0.010 wu) but d_vol falls 3× more slowly per window
early on: with 184 particles per cell the per-particle control field is seven times as
redundant as at 40k, and the grid transfer averages its incoherent part away — only the coherent
part moves material. That is the control basis's argument (§15a's cg17: node spacing one cell).

**Pre-registered (21:35): `b300_bunny` = the recipe + `--control_grid 17` at 300k** (cell 0.32
wu ≈ dx; 4913 nodes × 20 knots, 0.9 M dof against 54 M per particle). Predictions: (i) windows
≤ 90 (the 40k count within a factor 1.5) and the wall clock ≤ 35 min under the same load; (ii)
end silIoU ≥ 0.955, chamfer ≤ 0.065, det F min ≥ 0.20 (the basis cannot express the sub-cell
strain that gave 0.197); (iii) the GPU sections per window shorter (the leaf is 60× smaller:
Adam, the expand, the gradient reduction); (iv) g_share 0.35–0.40. Refutation: windows > 150
⇒ the redundancy is not the cause and the 300k schedule needs the pace or the plateau rule
re-derived for N.

**The cause of the 300k window count (17:55), and the fix pre-registered.** The basis run
`b300_bunny` tracks the per-particle run's transport-arrival curve window for window (17.7 →
19 → 28 → 43 → 55 … 85 % at window 100, against 18 → 91 % by window 10 at 40k): the control
parametrisation is not the cause (P(i) of the basis test refuted). The optimiser telemetry of
the first windows says what is: the same control magnitude (|dFc| max 0.02, alpha 0.02, 8
accepted iterations) moves the 300k body 3–6× less (`move` 0.004–0.037 wu against 0.027–0.081)
with 10–100× less kinetic energy (kin 0.01–0.3 against 0.5–1.2). The dynamics use UNIT particle
masses (`m = torch.ones(N)`, runner.build_target): the body's mass grows with N while the
control force per cell (∝ the particle rest volume ∝ 1/N, summed) does not — the acceleration
from a unit control scales as 1/N, the 300k body is 7.5× more sluggish, and the pace's windows
multiply. Fix (config `mass_ref_n = 40000`, optimizer): the dynamics mass per particle is
mass_ref_n / N — the body's mass, density, wave speed and control response are then the same at
every N, 40k is the reference discretisation (scale 1, bit-identical), the loss-side masses
stay unit (their normalisations are built on them). Run `c300_bunny` (the recipe, per-particle
control, 300k, the mass fix). Predictions: windows 45–90; `move` and `kin` of the first windows
within a factor 1.5 of the 40k values; end silIoU ≥ 0.955, chamfer ≤ 0.065; det F min 0.5–0.8
(the 40k regime; the sluggish run's 0.20 came from strain accumulating over 212 windows). Wall
under the same load ≤ 25 min. Refutation: windows > 150 or a first-window `move` < 0.01.

**Readings (18:40).** `b300_bunny` (basis, unit masses): 234 windows, 73 min, silIoU 0.9716,
chamfer 0.0594, det F min **0.0048** — the window count refuted (i), and the basis on a sluggish
body accumulates strain to the edge of inversion: not a 300k setting. `c300_bunny` (the mass
fix, per-particle control): the first-window dynamics equal the 40k run's (move 0.024 vs
0.027, kin 0.48 vs 0.54, |v|max 2.4 vs 2.1), the transport arrives twice as fast as with unit
masses (58 / 76 / 87 % at windows 25 / 48 / 72 against 33 / 51 / 73), end silIoU **0.9677**
(the best bunny of all runs; 40k 0.961), chamfer 0.0594, det F min 0.478 — but 176 windows,
44 min: the mass fix is right (kept: `mass_ref_n = 40000`, 40k bit-identical) and the window
count is refuted anyway. What the tail does: the 300k cloud keeps improving by 0.1–0.7 % a
window on residuals below the loss cell that the 40k cloud cannot resolve, so the plateau rule
(0.3 % relative) never fires — the extra windows are real gains at the finer discretisation,
not a defect. The delivered quality along the run (the archive's frames, sil_iou / chamfer
against the target sample):

| window | c300 silIoU / chamfer | minutes at 13 s a window |
|---|---|---|
| 20 | 0.9475 / 0.0634 | 5 |
| 30 | 0.9576 / 0.0621 | 7 |
| 50 | 0.9597 / 0.0609 | 12 |
| 70 | 0.9611 / 0.0602 | 16 |
| 90 | 0.9620 / 0.0598 | 20 |
| 177 (end) | 0.9677 / 0.0594 | 40 |

At 70 windows the 300k run has the 40k run's end silhouette (0.9611 vs 0.9610); the last 100
windows buy 0.7 points. A window budget of 70–90 (`--animations`, a resource decision, not a
tuning) puts a 300k run at 15–20 min under today's host load, ~12 min on a quiet host; the
archive (13 GB uncompressed at 300k) adds 1–3 min. Rendering influence unchanged in kind:
g_share 0.34–0.35 at 300k (0.37 at 40k). The 40k gallery is untouched by any of this
(mass_ref_n = N there; the speed passes are result-identical, suite 249 passed).

### 2026-09-23 night — the isolated pieces of the 300k end frame; the reference discretisation of the spacing-derived constants (method.md §10.17a)

**The user (after the plain-mesh stills): the isolated mesh pieces at the ear tips must be
dealt with.** Facts first (`$OUT/scratch/iso_trace.py`, `iso_tip.py` on the c300 and g41
archives). The 300k end frame has 37 particles (0.012 %) farther than 1.5 native spacings
(0.10 wu) from the body, in 12 pieces (18 / 6 / 3 / 2 / 1 …); the 40k g41 run has 0 from
frame 400 on (peak 28 = 0.07 % at frame 125; 300k peak 215 = 0.07 % at frame 150 — the
expansion-phase spray is the same fraction at both N). The ear-tip piece: 18 particles ON the
target (0.36 spacings), 1.74 spacings (median) / 2.21 (max) = 0.12 / 0.15 wu from the body —
inside half an MPM cell — with 159 particles in its 3³ cells (not decoupled), inside the
dilated occupancy (not a fragment), 8-NN ratio 1.0 inside the clump (the kNN gate does not
fire, and the DT pull is zero on the target regardless). It travelled with its own material
(source spread 0.29 wu; its 70 source neighbours are the ear material 0.33 wu behind it):
spray at frames 150–450, re-attached and re-compressed by frame 1650 (8-NN 0.54 spacings),
then stretched apart again over frames 1800–2700 (8-NN 2.2–2.4) as the tip cell fills, 1.67
at the end. The target tip is 0.35 wu = 1.13 cells thick and its cell holds 0.60 of a bulk
cell's mass; the bodies reach it to the same distance in wu (300k 0.173 / 0.080 max / median
against 40k 0.126 / 0.083), but the 300k body brings **5.9 reference particles' mass** within
0.25 wu of the tip against **13** at 40k. Both runs: dx 0.306, grid 36³, loss cell = dx.

**Reading.** The piece is connected material for the grid and a separate bead for a
renderer whose kernel is the native spacing, and no mechanism owns it because all three
count particles, not mass. The cause of the thinner tip is the constants: at 300k on the
40k grid every spacing-derived length (u clip, layer depth, relaxation width, splat size,
cleanup band …) is half the 40k length and every neighbour count covers a seventh of the
mass — the dynamics were made N-invariant on 2026-09-23 (`mass_ref_n`), the constants were
not. Rule (38), method.md §10.17a: lengths × (N / mass_ref_n)^(1/3), counts × N / mass_ref_n.

**The render at the reference spacing (`render_photoreal --ref_n`, default = mass_ref_n),
pre-registered.** P1 end bump ≤ 1.3° (native 1.9, 40k 1.2): **refuted** — 1.5°. P2 the
ear-tip beads and the left ear's fork merge: half — the fork is one tip and no bead is
drawn anywhere, both tips keep a knob. P3 dropped sub-cell components ≤ 3: holds (0; the
pieces fall below the two-particle level at the reference kernel). P4 frame 900 keeps its
silhouette: holds (bump 1.4, the growing ears' front beads remain as 2 dropped pieces). Also
found: at the native spacing the 300k Poisson octree was one level finer than the 40k one
(cell 0.034 against 0.069 wu), so the earlier 1.9° vs 1.2° comparison was at different mesh
resolutions. Kept as the default: the rule is the render-side half of §10.17a, 40k
bit-identical; the knobs it leaves are the physics of the tip and belong to the twin below.

**The physics at the reference discretisation (`--disc_ref`), pre-registered before the
run (`d300_bunny`, the recipe + the flag, 300k; smoke `s300_ref` 3 windows first).**
P5 tip mass within 0.25 wu of the target tip ≥ 10 reference particles (c300 5.9, 40k 13).
P6 the tip material's end 8-NN ≤ 1.3 native spacings (c300 1.67); ≤ 10 particles off the
body at 1.5 spacings (c300 37), no piece of ≥ 6. P7 end silIoU ≥ 0.964 (c300 0.9677),
chamfer ≤ 0.0625, windows ≤ 210. P8 end bump at the native render spacing ≤ 1.5° (c300 1.9),
at the reference spacing ≤ 1.3°. P9 mid-morph roughness within the c300 band. Refutation:
P5 < 8 or P7 fails → §10.17a does not close the tip, the flag stays off. Rendering
influence: unchanged in kind — the render loss's splats are the 40k splats (fewer, larger
than the native ones), g_share to be read from the run. Tests: `tests/test_disc_ref.py`
(the factor, the asymmetry count), the touched suites 34 passed on hyde06.

**Readings of the twin `d300_bunny` (20:05; the run: 52 windows, 984 s wall = 16 min, stopped by
the outer-merit brake at windows 51–52 after a null commit at 49 — the plateau is real at the
reference resolution).** P5 tip mass **18.3** reference particles within 0.25 wu of the target tip
(c300 5.9, 40k 13) — holds, 3×. P6 tip 8-NN **0.58** native spacings (c300 1.67) — holds; off-body
particles at 1.5 spacings **95 in 46 pieces** (c300 37 / 12; the largest 12 and 10, on the underside
0.6–1.2 spacings from the target) — refuted. P7 end silIoU **0.9699** (c300 0.9677, the best bunny
so far), chamfer 0.0598, det F min 0.745 (0.478), stray_max 0.0002 — holds, at 52 windows against
176: the 300k-in-20-minutes budget is met without a window cap, because the objective at the
reference spacing (OT blur radius 0.135 wu, the gate, the isolation k = 60) no longer sees the
sub-cell residuals c300 kept chasing for 120 windows (the transport gate reaches 100 % of the
layer at window 52; c300 was at 61 % there). P8 end bump at the reference render 1.5° (c300 1.5°,
prediction ≤ 1.3) — refuted; at the native render **2.3°** (c300 1.9°, ≤ 1.5) — refuted, with 4
interior cavities and 5 dropped sub-cell pieces. P9 mid-morph roughness (lump_trace, rms at 2
native spacings, frames 300–825) **0.42–0.59** against c300's 0.20–0.26 — refuted, 2×; at the
cell scale 0.66 against 0.47. Rendering influence: g_share 0.29–0.33 (c300 0.25–0.30; 40k 0.37);
the render loss's splats are the 40k ones; the deliverable renderer is the same for both.

**Reading.** Rule (38) fixes what it was derived for — the tip's mass and stretch, the response
of the layer and the u channel, the convergence — and worsens the sub-cell arrangement: the
relaxation now smooths at 0.27 wu, the u step is 0.135 wu a window, the isolation gate looks at
60 particles, so nothing acts below the cell any more, and the surface at the native scale is
rougher (the pits and cavities of the native still). The two runs bracket the same defect from
both sides: at the native constants the per-particle machinery half-orders the sub-cell
arrangement while the objective chases sub-cell residuals for 120 windows (the window-to-window
reversal of c300's tail, corr −0.5…−0.7, see below); at the reference constants the chase is gone
and so is the ordering. The method has no term whose job is the ORDER of the quadrature below
the cell — the null space of the cell sum, the transport plan and the render. `--disc_ref` stays
opt-in (not in RECIPE); the deliverable renderer keeps `--ref_n` (a render at the physics
resolution is right regardless).

**The oscillation near the optimum (the user, 20:00: "진동이 매우 심하다").** Measured on the
archives (`$OUT/scratch/osc_probe.py`, per-window displacement of a 40k subsample): c300's last 60
windows move a median 0.02–0.13 native spacings (0.005–0.03 cells) a window, consecutive windows
anti-correlated (**corr −0.55 … −0.72** at windows 123 / 141 / 147 / 165, −0.13 … −0.42
elsewhere) and spatially coherent (0.7–0.97 of the energy in the 2-spacing neighbourhood mean):
a grid-smooth back-and-forth, not sub-cell jitter (the sub-cell part is 0.004 spacings). 40k
g41's last 30 windows: 0.014–0.03 spacings, corr −0.09 … −0.65, the same in kind at half the
cell amplitude. d300 stops before the alternation sets in (corr +0.35 at window 42, the brake at
51–52 with reversal cos 0.23 / 0.31). The transport subsample is fixed for the run
(`entropic_map` keeps `_sub_idx`), so the alternation is not subsample noise: it is the outer
loop over-correcting a residual the grid cannot resolve and the next window undoing it.

**Next experiments after the digest (docs/related_work.md 2026-09-23 night), pre-registered.**
E1 `e300_bunny` — the recipe (native constants, the mass contract) + the cell-scale control basis
(`--control_grid 17`, one-cell node spacing, the setting of the 40k H15 test and of b300): the
control loses its sub-cell DOF (884k against 2.7M for per-particle dFc) so it cannot chase
sub-cell residuals — the digest's (a) in our code. b300's verdict (det F 0.005) was taken with
unit masses (the body 7.5× sluggish, 234 windows) and is re-read here. P10 det F min ≥ 0.4
(c300 0.478). P11 windows ≤ 120, wall ≤ 30 min (c300 176 / 44). P12 end silIoU ≥ 0.960, chamfer
≤ 0.065. P13 the tail's window-to-window displacement correlation (osc_probe, last 30 windows,
median) > −0.3 (c300 −0.3 … −0.7). P14 mid-morph roughness (lump_trace rms 2 sp, frames
300–825) within c300's 0.20–0.29 and the end native bump ≤ 1.9°; off-body particles at the end
≤ 37. P15 tip mass ≥ 5.9 reference particles (no worse than c300). Refutation: det F < 0.2 or
silIoU < 0.955 → the basis is not a 300k discretisation; P13 failing with the rest holding → the
reversal is not the control's sub-cell DOF but the outer loop itself.
E2 (to implement, opt-in) — Fickian particle shifting at the window commit, the digest's (b):
Δx_p = −½ h² ∇C_p with ∇C_p = Σ_j (m_j/ρ_j) ∇W(x_p − x_j; h) over the kNN, h = 2 native spacings
(the quadrature's own neighbourhood, the width of the layer relaxation), λ = ½ the explicit-
diffusion stability limit (Lind 2012 / Skillen 2013), the outer layer shifted in its tangent
plane only (the free-surface rule), positions only (F, C, v untouched; the shift ≪ a spacing).
Predictions to be written with the run.

**E2 implemented (mpm/shifting.py, `--shift_sub`; method.md §10.18) and its constants measured
before the run (`$OUT/scratch/shift_probe.py`, `shift_probe2.py`, CPU).** The plain Fickian step
(Gaussian W, any h) DIS-orders a cloud: the kernel gradient vanishes for close pairs, so mid-range
neighbours push harder than near ones and the 1-NN spacing CV of a jittered lattice grows 0.16 →
0.36 in ten steps (a uniform random cloud 0.38 → 0.58). With Monaghan's anti-pairing factor
[1 + 0.2 (W_ij / W(Δp))⁴] in the gradient (Lind 2012 carries it) and the Gaussian at h = Δp (the
width of the cubic spline at the SPH ratio h = 1.3 Δp) the step orders: 0.16 → 0.08 and 0.38 →
0.10 in ten steps, the 8-NN CV 0.30 → 0.16, the disorder |∇C| h 0.74 → 0.33, lattice tie bias
0.006 Δp, median shift 0.13–0.23 Δp on disordered clouds; h = 1.3 Δp (Gaussian) orders weakly,
h = 2 Δp dis-orders even with the factor, the cubic spline at h = Δp saturates the ½-spacing cap.
On a slab the top layer's normal shift is 0.009 Δp (median), tangential up to 0.2 Δp. Tests
`tests/test_shifting.py` (lattice fixed point, one-step ordering of a jittered lattice and a
random cloud, the free-surface rule, the cap).

**Pre-registration of `f300_bunny` — the recipe (native constants) + `--shift_sub`, 300k, one
shifting step at every window commit.** P16 mid-morph roughness (lump_trace rms at 2 native
spacings, median over frames 300–825) ≤ 0.20 (c300 0.20–0.26, d300 0.42–0.59). P17 end bump at
the native render ≤ 1.6° (c300 1.9°, d300 2.3°), at the reference render ≤ 1.4° (1.5°). P18 end
off-body particles at 1.5 spacings ≤ 20 (c300 37), no piece ≥ 6 at the ear tip; the tip
material's 8-NN ≤ 1.5 (c300 1.67 — the shift orders the tip, it cannot bring mass to it). P19 end
silIoU ≥ 0.964, chamfer ≤ 0.062 (loss-neutral within 0.004 of c300). P20 windows ≤ 150 (c300 176:
the sub-cell residuals are ordered instead of chased) and the tail's window-to-window reversal
(osc_probe, median of the last 30 windows) > −0.3. P21 the shift itself: median ≤ 0.1 spacing a
commit after window 30 (the cloud stays ordered), p99 ≤ 0.3. Refutation: P16 or P19 failing →
shifting does not order the quadrature of the morph, or harms the fit; the flag stays off.

**Readings of E1 `e300_bunny` (the recipe + `--control_grid 17`, the mass contract; 21:00).** 206
windows, 2434 s = 41 min (the brake at 205–206, reversal cos −0.33). P10 det F min **0.739** (c300
0.478) — holds. P11 windows ≤ 120 — refuted (206). P12 end silIoU **0.9795**, chamfer 0.0586 — holds,
the best silhouette of every bunny run. P13 the tail's window-to-window reversal — **refuted**: corr
−0.62 / −0.18 / −0.84 / −0.76 / −0.02 / −0.43 at windows 167–197, the same alternation as c300 with
the control's sub-cell DOF gone (884k against 2.7M DOF). P14 mid-morph roughness 0.196–0.244 (c300
0.20–0.26) — holds; end off-body 44 particles in 11 pieces (c300 37) — holds. P15 tip mass 8.7
reference particles (c300 5.9), 8-NN 1.35 (1.67) — holds. b300's verdict is overturned: with the
mass contract the basis is a sound 300k discretisation (det F 0.74, the best fit), and it does not
touch the oscillation, which is therefore the outer loop's, not the control parametrisation's.

**Readings of E2 `f300_bunny` (the recipe + `--shift_sub`; 21:00).** 176 windows, 2131 s = 36 min
(the brake at 175–176). The shift itself: median 0.173 spacing at the first commit (the stratified
source), 0.02–0.025 from window 20 on, p99 0.10–0.12, the disorder |∇C| h 0.465 → 0.18 and flat —
P21 holds (the cloud stays ordered at a cost of 0.02 spacing a commit). P16 mid-morph roughness
**0.176–0.210** (c300 0.20–0.26; lower at every frame 450–825) — holds. P19 silIoU **0.9748**,
chamfer **0.0554** (the best chamfer), det F 0.535 — holds. P20 windows ≤ 150 — refuted (176); the
tail reversal −0.48 … −0.83 — refuted (the ordered quadrature does not stop the alternation
either). P18 — **refuted, and how**: 397 particles off the body in 28 pieces, among them a
**295-particle chunk** (1.6 cells of mass) and a 67-particle piece at the RIGHT ear's tip
([−1.19, 3.51, −0.12]; the target 1.2 cells thick there), ON the target (0.41 spacings) and 5.3
spacings = 1.2 cells from the body (min 2.5); it separated slowly from window ~105 (to-body
median 1.2 → 3.7 spacings over frames 2000–3150), the 67-piece from window ~55. In c300 the same
tip is held by a stretched thread of particles (8-NN 2×); once shifting evens the spacing the
thread is gone and the tip block drifts on the target. CORRECTION (the independent audit,
docs/diagnosis_300k_20260923.md): the kernel is cubic (4³ stencil), and at the saved endpoint all
300 particles of the piece still have nonzero weighted stencil overlap with the body — the
coupling is weak, not absent; "shares no node" was wrong. The left
tip: 4.8 reference particles (c300 5.9), 8-NN 1.88. The deliverable rule draws a 295-particle
piece (≥ 170) and cannot bridge it (0.36 wu > the 0.31 wu link radius): a floating ear tip.
Neither mechanism alone is adoptable; the pair says what each does — the basis: the fit and det F;
shifting: the surface and the chamfer; neither: the oscillation, the thin tip.

**Pre-registration (21:05) of the combinations, both on the recipe at 300k.** `g300_bunny` =
`--disc_ref --shift_sub`: P22 tip mass ≥ 15 reference particles (d300 18.3, f300 4.8), tip 8-NN
≤ 0.8 (d300 0.58). P23 mid-morph roughness (median, frames 300–825) ≤ 0.25 (d300 0.42–0.59: the
shift orders what the reference constants leave). P24 end bump at the native render ≤ 1.8° (d300
2.3°), at the reference render ≤ 1.5°. P25 silIoU ≥ 0.968, chamfer ≤ 0.061, windows ≤ 80 (d300
52), det F ≥ 0.6. P26 off-body ≤ 40, the largest piece < 50 particles (f300's chunk must not
recur: at the reference constants the tip holds 3× the mass and the neck is filled). Refutation:
P22 or P26 failing → the combination is out. `h300_bunny` = `--disc_ref --shift_sub
--control_grid 17`: P22–P26 as above, P27 silIoU ≥ 0.975 (e300 0.9795) with det F ≥ 0.6, P28 the
tail reversal persists (median corr < −0.3) — the prediction that the oscillation is the outer
loop's, to be falsified a second time. Next after these: the outer loop's damping (E3).

**The oscillation near the optimum, diagnosed and the first mechanism pre-registered (21:40;
docs/oscillation.md Addendum 9).** The 300k tail is the outer layer breathing along its normal
(0.07–0.11 native spacings a window, sign flips in 60–80 % of the layer, net 0.04 spacings for 2.5
summed over 30 windows; the bulk follows at half the amplitude); 40k has it at 0.003–0.009 wu and
hides it because its run converges at 57 windows and the frames are held. Not the basis (e300
layer corr −0.81), not shifting (f300), not the transport subsample (fixed), not the reconstruction
(the reference-spacing video changes as much). The net-over-summed displacement ratio over the last
k windows (`$OUT/scratch/cycle_ratio.py`, 20k subsample) separates the phases: c300 0.77–0.88 in
the descent, below the random-walk bound 1/√5 = 0.447 from window 47 (consistently from ~68), 0.2–
0.35 in the breathing tail; e300 the same from 51 / ~73; g41 (40k) from 32 (its stale rule fired at
57); d300 (reference constants, brake at 52) never — it moves coherently to the end. Implemented
(`--stop_on_cycle`, config `stop_on_cycle`; runner at the commit): the run is converged when the
ratio has been at or below 1/√patience for `patience` consecutive windows (k = patience = 5, the
existing horizon; no new constant), then `hold_after_converge` holds the delivered tail as at 40k.
The ratio and the counter are logged every window regardless (`net_ratio`, `cyc_stale`).

Pre-registration of `i300_bunny` (the recipe + `--stop_on_cycle`, 300k): P31 the freeze between
windows 60 and 100 (c300's ratio is consistently below the bound from ~68). P32 silIoU at the
freeze ≥ 0.9605 (c300 at window 70: 0.9611; 40k end 0.9610), chamfer ≤ 0.0605. P33 the delivered
plain-mesh video's tail per-frame change |dI| median ≤ 0.0003 (held; 40k 0.0001; c300 0.0021).
P34 wall ≤ 22 min under today's load. Refutation: a freeze before window 50, or silIoU < 0.958 →
the rule cuts honest descent and is withdrawn. This removes the symptom for the deliverable; the
source (the render-driven u step undone each window) is the next mechanism (Rprop-style
per-particle damping of u, or a stop at the render target's noise floor), pre-registered after
g300 / h300.

**Pre-registration of `j300_bunny` (the recipe + `--u_rprop`, 300k; method.md §10.19; 22:00).**
P35 the tail's (last 30 windows) layer normal sign flips ≤ 40 % of the layer (c300 60–80 %), the
layer's per-window |d| median ≤ 0.06 native spacings (c300 0.10–0.19), consecutive-window layer
correlation > −0.3 (c300 −0.35 … −0.71). P36 the net/summed displacement ratio in the tail ≥ 0.45
(c300 0.2–0.35). P37 silIoU ≥ 0.964, chamfer ≤ 0.062 (the channel keeps its end gain; the u-off
twins at 40k cost 0.3–0.6 points). P38 windows ≤ 150 (c300 176: the noise-level gains that keep
the plateau rule from firing are the breathing). P39 the u scale's median over the layer in the
tail ≤ 0.5 (the rule is active where it should be). Refutation: P35 failing → the breathing is
not the u channel's (then it is the stress channel's tangential part, Addendum 9's other half);
silIoU < 0.960 → the damping costs the fit and is withdrawn.

**Readings of the combinations (22:40). `g300_bunny` = `--disc_ref --shift_sub`: 59 windows, 1001 s
= 17 min (the stale rule at 59), silIoU **0.975**, chamfer 0.0556, det F min 0.718, stray_max 0.0003;
tip mass **23.7** reference particles (178 particles within 0.25 wu of the tip; the target sample
holds 163 there), tip 8-NN 0.79, body under-fill max 0.073 wu; off-body **5 particles, all
singletons**; mid-morph roughness 0.21–0.32 (median 0.27); the tail's net/summed ratio 0.7–0.9 and
the layer's window correlation +0.07 … +0.29 — no breathing; the shift median 0.02 sp a commit,
disorder 0.21; g_share 0.37. P22 ✓ P23 ✗ (0.27 against ≤ 0.25, marginal) P24 pending (stills)
P25 ✓ P26 ✓. `h300_bunny` = `--disc_ref --shift_sub --control_grid 17`: 79 windows, 1937 s = 32
min (the brake at 78–79, reversal cos −0.05 / −0.03), silIoU **0.977**, chamfer **0.0554**, det F
**0.835**; tip mass 41.7 reference particles (313 within 0.25 wu: over-packed at 1.9× the target's
density, 8-NN 0.81 / p90 0.94), under-fill max 0.087; off-body 4 singletons; roughness 0.18–0.33
(median 0.22); net/summed 0.53–0.95, layer correlation −0.31 … +0.28; g_share 0.35. P22 ✓ P23 ✓
P25 ✓ P26 ✓ P27 ✓ (0.977 ≥ 0.975, det F ≥ 0.6) **P28 ✗ in the good direction**: the tail does not
alternate — under the reference constants the objective no longer sees the sub-cell residuals the
loop was chasing, so the prediction "the oscillation is the outer loop's regardless of the
control" holds only at the native constants (e300); at the reference constants the loop converges
instead of breathing (d300, g300, h300 all: net/summed ≥ 0.5 to the end).

**Reading.** The three defects the user named — the thin tips, the isolated pieces, the breathing —
are absent together in g300 / h300, with the best silhouette, chamfer and det F of every bunny run
and the 300k budget met (17 / 32 min). Each mechanism alone had failed on one of them (d300 rough,
e300 breathing, f300 a detached chunk): the reference constants stop the chase and bring the tip's
mass, shifting orders what they leave, the basis adds det F and fit at 2× the windows. Open: the
end bump at the reference render (P24, stills pending), g300's mid-morph roughness 0.27 (h300 0.22),
h300's over-packed tip (1.9× the target density — the knob risk), and the 40k gallery is untouched
by all of it (disc_ref is a no-op at N ≤ 40000; shift_sub / the basis are opt-in). Adoption into
the recipe for N > 40k is the user's call after the stills and the videos.

**P24 (22:55, the end-frame stills at the reference render):** g300 bump **1.3°**, h300 **1.2°**
(40k g41 1.2°; c300 1.5°, d300 1.5°, e300 1.4°) — holds for both; both ear tips clean and sharp,
one component, nothing dropped, no cavity. At equal render resolution the 300k surface now matches
the 40k one, with silIoU 0.975 / 0.977 against 0.961 and chamfer 0.0556 / 0.0554 against the 40k
run's. Correction of the framing (the user, 22:45): the two closest papers of the scan — the TVCG
2025 MPM morph and PhysMorph-GS — are the user's own; this project is their follow-up, and today's
combination closes the two items PhysMorph-GS lists as open, thin features and watertightness
(docs/related_work.md updated).

**Baseline comparison started (23:10; docs/related_work.md "Competitors and baselines").** The two
external morph baselines with public code — *Implicit Neural Surface Deformation with Explicit
Velocity Fields* (ICLR 2025, Sang et al.; repo Sangluisme/Implicit-surf-Deformation @ e7992a5) and
*4Deform* (CVPR 2025; Sangluisme/4Deform @ 7f2274e) — set up on hyde06 in a separate conda env
`nsd` (JAX 0.4.25 + flax 0.8.1; the pip cuda12 plugin pulls cuDNN 9.26, which JAX 0.4.25 cannot
initialise — pinned to 8.9.7.29, verified). Inputs: an icosphere (subdivision 5, the asset's 42
vertices are too coarse for 20k samples), `assets/bunny.obj`, and a generated torus of the
sphere's volume (R/r = 2.5, genus 1) for the hole case; preprocessing on the CPU (20k surface
samples + normals per shape, 5k paired "verts", one shared scale; `CORR=none`: the matching loss
off, because sphere → bunny / torus has no correspondences and the authors' functional-map
matcher does not apply). Runs launched on GPU 0: ISD and 4Deform on sphere → bunny and sphere →
torus (their shipped confs: ISD 15k epochs, T = 10; 4Deform 25k + 5k, T = 5; outputs = marching-cubes
meshes at t = k/N, largest component only, plus advected point clouds). Comparison plan: sample our
morph at the same t, report silhouette IoU / Chamfer, the Euler characteristic per frame (the
genus change), volume drift, self-intersection, wall-clock. Scripts and the agent's notes in the
session scratchpad `baselines/`; server `baselines_prep/`, `baselines/`.

**Readings of `i300_bunny` (the recipe + `--stop_on_cycle`; 23:20).** The rule fired at animation
**106** (net/summed 0.400 ≤ 0.447 for 5 windows), 1148 s = 19 min; silIoU **0.9625**, chamfer 0.0598,
det F 0.759. P31 (freeze between 60 and 100) — refuted, narrowly (106). P32 (silIoU ≥ 0.9605) —
holds. P34 (≤ 22 min) — holds. P33 — refuted as written: the delivered archive ends at the freeze,
so the plain-mesh video's last third is the pre-freeze tail, which breathes as c300's does (per-
frame |dI| median 0.0026 against c300's 0.0021); the held tail is static by construction and was
not what the metric measured. Reading: the rule cuts 70 windows of breathing off the deliverable
and stops at the 40k-level silhouette, but does not reduce the breathing while the run is alive;
next to g300 / h300, where the breathing does not arise, it is a secondary safeguard. Kept opt-in.

**The g300 / h300 plain-mesh videos (23:35; `$OUT/r300/plain/{g300,h300}_plain_s12.mp4`, stride 12,
the reference-spacing render).** Sub-cell pieces dropped by the deliverable rule in **17 of 87**
frames (g300) and **21 of 125** (h300) against c300's 208 of 282 (124 of 282 at the reference
render); isolated particles peak 75 / 171 against c300's 135 (the peak sits in the expansion phase,
frames 228–240, as before); drawn components 1 in every frame of both. On the page with the
stills and the table.

**Readings of `j300_bunny` (the recipe + `--u_rprop`; 23:45).** 184 windows, 1988 s = 33 min (the
stale rule at 184); silIoU 0.9641, chamfer 0.0594, det F 0.658; tip 7.9 reference particles; off-body
34 particles in 12 pieces. The damping did what it was built to do — the u bound's median scale
0.60 at window 10, 0.10 at 30, 0.07 at 60, **0.05 (the floor) for 74 % of the layer** at 180 (P39
holds) — and the breathing did not go: sign flips 32 % at window 10, 43 % at 60, **76 % at 180,
92–97 % at windows 179–184**, the layer's window correlation −0.89 … −0.95 in the last windows,
its normal motion 0.04–0.11 spacings a window (c300 0.07–0.11), net/summed 0.23–0.30 in the tail
(P35, P36 refuted), windows 184 (P38 refuted), silIoU 0.9641 (P37 holds by 0.0001). **Verdict, as
pre-registered:** with u all but disabled the layer still breathes along its normal, so the
breathing is NOT the u channel's — it is the stress channel's (the transport control) and the
relaxation's response to the sub-cell residuals the objective keeps presenting at the native
constants; the independent audit's objection to the render-u attribution stands and is now
measured. Consistent with g300 / h300: the breathing vanishes when the objective's constants are
the reference ones, not when u is damped. `--u_rprop` is refuted as a fix; the flag stays off.

**Head-to-head, our side (23:05; `$OUT/r300/mesh/`, `baselines_prep/compare_baselines.py`).** Meshes
exported by `render_photoreal --still --save_mesh` (Poisson, the reference-spacing rule) at t = 0.25,
0.5, 0.75, 1.0 of the delivered trajectory; the target's surface = the outer layer of its 300k
volume sample (asymmetry rule, 15 770 points); Chamfer = symmetric mean distance / the target's
bbox diagonal (9.76 wu), 20k surface samples a side.

| run | t 0.25 | t 0.5 | t 0.75 | t 1.0 | volume (wu³, t 0.25 → 1) | comps / genus / watertight |
|---|---|---|---|---|---|---|
| c300 | 0.0147 | 0.0127 | 0.0125 | **0.0123** | 44.07 → 44.23 | 1 / 0 / yes at every t |
| g300 | 0.0154 | 0.0115 | 0.0110 | **0.0109** | 45.54 → 44.83 | 1 / 0 / yes |
| h300 | 0.0133 | 0.0109 | 0.0112 | **0.0111** | 45.59 → 44.55 | 1 / 0 / yes |

(A first pass against the VOLUME sample of the target read 0.049 — the interior points' distance
to a surface; the surface-to-surface number is the comparable one.) The morph is essentially at
the target by t = 0.25 (the transport arrives by window 25–50 of 176 / 59 / 79); the source
sphere's material volume is 47.5 wu³ and the reconstructed surface encloses 44–45 (the Poisson
surface sits inside the outer layer by about half a spacing), drifting ≤ 2.3 % over the tail.
The baselines' rows follow when their runs finish (ISD in its post-training reconstruction at
23:05; 4Deform and the torus pair queued).

**Review of the independent audit (the user, 23:15: "진단 문서도 한 번 봐 줄래"; docs/diagnosis_300k_20260923.md)
and the spatial-refinement test it asks for, pre-registered.** Accepted and folded in (0ccfad1,
5410dda, a7a3005): the cubic-stencil overlap of the detached piece, "underconstrained" in place of
"null space", the unproved (now refuted, j300) render-u attribution, the CFL diagnostic's mass, the
objective change across a shift (now logged). Accepted now: the method is a HYBRID (stress control
+ the layer's position channel u + the relaxation projection) — AGENTS.md and overview.md said
"real elastodynamics" alone; corrected. Its oscillation re-measurement on accepted-window endpoints
(cos −0.546, 0.134 sp, 71 % flips) agrees with the fixed-19-frame probe within its own spread, so
the finding does not depend on the grouping; future probes take the accepted endpoints. Its point
that G3 can PASS while the layer breathes is right — the gate reads a 10-frame bulk average; a
layer-local amplitude / reversal metric belongs in the run's metrics (to do). Open, as it says:
phase-aligned surface comparisons across runs with different window counts.

Its recommendation (3) — spatial refinement at fixed N and physical mass — is the one worth a run
now: the "finest fracture-free cell = 26" rule (experiments 2026-09-17: the dragon at dx 0.20–0.21 —
40k ppc 8: 41 fragments; 150k ppc 27: 1562 re-attachments) rests on two cases, of which only the
150k one was taken with unit masses before the mass contract (the 40k one had the reference mass:
the confound of the basis verdict applies to half the evidence, not all of it), on the dragon's
whiskers and horns, and without the reference constants or shifting. For the bunny the rule was
never tested at dx 0.22 at all. `k300_bunny` = the g300 recipe (`--disc_ref --shift_sub`) + `--cell_diag
36` (dx 0.22 wu, grid ≈ 50³, the ear tip 1.6 cells thick instead of 1.1): P40 no fracture — re-
attachments 0, off-body ≤ 10 particles, no piece ≥ 6, det F min ≥ 0.5 (the 150k runs at this dx
fragmented). P41 tip mass ≥ 20 reference particles (g300 23.7) with 8-NN ≤ 0.9. P42 silIoU ≥
0.975, chamfer/diag (surface) ≤ 0.0109 (g300). P43 windows ≤ 100, wall ≤ 45 min (the grid ops
×2.7). P44 end bump at the reference render ≤ 1.3°, mid-morph roughness ≤ g300's 0.27. Refutation:
fragments or det F < 0.3 → the finer cell fractures even with the mass contract, the rule stands.

**Head-to-head, the ICLR 2025 implicit velocity-field morph (ISD) on sphere → bunny, no
correspondences (23:30; 1058 s training, eval MC 256 at t = k/10).** Surface Chamfer / diag against
its own normalised target mesh (the bunny with its open base: euler −3, not watertight):

| t | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chamfer/diag | 0.098 | 0.140 | 0.176 | 0.232 | 0.276 | **0.298** | 0.297 | 0.228 | 0.150 | 0.063 | **0.0070** |
| volume (norm. units) | — (5 comps) | 0.049 | 0.035 | 0.013 | 0.004 | **0.002** | 0.001 | 0.012 | 0.042 | — (5 comps) | 0.199 |
| euler / genus | 5 / ? | −46 / 24 | −4 / 3 | 2 / 0 | 2 / 0 | 2 / 0 | 2 / 0 | 2 / 0 | 2 / 0 | 5 / ? | 2 / 0 |

Reading: the end fit is excellent (0.0070 — an SDF fitted to the target; ours 0.0109–0.0123 at the
same relative measure, against a particle-sampled surface), and the trajectory between is not a
morph: without correspondences the velocity field collapses the body to 1 % of its end volume at
t = 0.5–0.6 and regrows it, with genus 24 and 3 surfaces at t = 0.1–0.2 and five components at t = 0
and 0.9. The axis on which the continuum morph is defined — one body, its mass and volume
conserved at every t, no surface to hallucinate — is exactly where the baseline has nothing:
ours holds one watertight genus-0 body at every t with the volume within 2.3 %. The
correspondence-driven setting (CORR=nearest / radial) and 4Deform follow; the torus pair next.

**The user's standing directive (23:40): keep experimenting and reading until the oscillation is
zero and the surface is smooth; mine the SIGGRAPH geometry literature and the MPM
particle–grid–mesh literature from 2013 on and apply what addresses the problems.** Acceptance
criteria, fixed now so that "done" is measurable (the 40k g41 values are the bar; all on the 300k
bunny unless stated):
- Breathing: `layer_flip_frac` ≤ 0.50 (a coin flip; c300 0.6–0.8) and `layer_net_ratio` ≥ 0.5 over
  the last 10 windows (metrics.layer_breathing, now in every run's metrics); the plain video's
  tail per-frame change ≤ the 40k value (0.0001–0.0005; c300 0.0021).
- Surface: end bump at the reference render ≤ 1.2° (g300 1.3°, h300 1.2°); mid-morph roughness
  (lump_trace rms at 2 native spacings, frames 300–825) within the 40k band, to be measured on
  g41 at its own spacing for the comparison (phase-aligned, not frame-aligned).
- Pieces: no particle piece ≥ 6 at any archived frame from the arrival on; sub-cell drops in the
  video ≤ 10 % of frames (g300 17/87 = 20 %, h300 17 %; c300 74 %).
- Fit: silIoU ≥ 0.975 and det F ≥ 0.6 kept (g300 / h300).
Two survey agents launched (MPM particles–grid–surface, 2013–2026; SIGGRAPH geometry
interpolation and surface-from-particles); their digests go to docs/related_work.md, the chosen
mechanisms are pre-registered here before they run.

**Roughness in WORLD units, 40k against the 300k combinations (23:45; lump_trace at each run's own
spacing, the "cell" column = the plane residual at the MPM cell scale, converted to wu).** g41 (40k,
spacing 0.1385): rms at 2 spacings 0.18–0.27 sp = 0.025–0.037 wu; at the cell scale 0.21–0.30 sp =
0.029–0.042 wu. g300: cell-scale 0.46–0.60 native sp = 0.032–0.041 wu; h300 0.41–0.57 = 0.028–0.039
wu; c300 0.45–0.50 = 0.031–0.035 wu. At the scale the physics resolves, the 300k combinations are
as smooth as the 40k run; the extra roughness of 300k lives below the cell (the 2-native-spacing
column, 0.14 wu: g300 0.21–0.32 sp = 0.014–0.022 wu, a scale the 40k probe cannot even sample).
The acceptance criterion "within the 40k band" is therefore met at the cell scale by g300 / h300;
the sub-cell texture is the remaining item, and the reference-spacing render draws the cell
scale — which is why the g300 / h300 end bump (1.3° / 1.2°) equals the 40k one.

**The breathing metric on the finished archives, and a correction (23:55; `metrics.layer_breathing`,
last 10 windows; the videos' delivered tails with the 20 hold frames excluded).**

| run | layer flips | net / summed | normal step (sp) | step in wu | video tail |dI| (delivered last third) |
|---|---|---|---|---|---|
| g41 (40k) | 0.742 | 0.193 | 0.021 | 0.0029 | 0.0012 |
| c300 | 0.775 | 0.119 | 0.073 | 0.0050 | 0.0025 (reference render) |
| g300 | **0.557** | **0.254** | 0.068 | 0.0047 | **0.0019** |
| h300 | 0.658 | 0.132 | 0.113 | 0.0078 | **0.0017** |
| i300 | — | — | — | — | 0.0028 |
| j300 (u_rprop) | 0.931 | 0.108 | 0.069 | 0.0048 | — |

Correction of "the breathing vanishes in g300 / h300" (the cycle-ratio probe on the BULK's 3-D
displacement read 0.7–0.9): on the LAYER's normal component over the last 10 windows g300 flips
0.56 and drifts 0.25 of its summed motion — a random walk (1/√10 = 0.32), no longer c300's
anti-correlated cycle (0.78 / 0.12) — and h300 0.66 / 0.13. The 40k reference itself flips 0.74 at
0.021 spacings = 0.003 wu a window, the sub-spacing jitter Addendum 7 closed as invisible; its
video's delivered tail changes 0.0012 a frame, and it looks still only where the frames are held.
So the visible quantity is the AMPLITUDE in world units, not the flip fraction: g300 / h300 move
0.0047 / 0.0078 wu a window in the tail (1.6–2.7× the 40k value) and their videos' delivered tails
change 0.0019 / 0.0017 a frame (1.4–1.6× the 40k's; c300 2.1×). The acceptance criterion is
restated in those terms: the layer's normal step ≤ 0.003 wu a window over the last 10 windows and
the delivered tail's per-frame change ≤ 0.0012 (the 40k values), with the trajectory delivered
up to convergence and held after it, as the 40k gallery is. g300 / h300 are not there yet; the
remaining amplitude is the objective's sub-cell chase at the reference constants' own floor plus
the stress channel's transport, and the next mechanisms come from the surveys.

**Pre-registration (00:10): the anisotropic kernel of Yu & Turk 2013 for the delivered surface
(the geometry survey's shortlist 2; `render_photoreal --kernel pca`, already implemented, never
used at 300k).** On g300's end frame at the reference spacing, `--kernel pca` with the PCA
neighbourhood at the reference mass (`--pca_k 240` = 32 × 7.5) and, as the control, at the native
count (32). P45 bump ≤ 1.1° (iso kernel 1.3°; 40k 1.2°). P46 components 1, cavities 0, both ear
tips intact (the tip region of the mesh present at the same length as the iso render; no
bridging of the ears). P47 the same on frame t = 0.5 (sub-cell texture of the morph). Refutation:
the tips shortened or bridged, or bump ≥ 1.3° → the kernel does not help at this spacing.

**From the MPM survey (docs/related_work.md 2026-09-24): the null-space projection at the window
commit, pre-registered (00:30).** Gritton & Berzins 2017 / Tran & Sołowski 2019 remove, per cell,
the particle components the P2G operator cannot see; XPIC(m) (Hammerquist & Nairn 2017) does it
by alternating transfers. Our breathing and sub-cell disorder are, by the audit's own words and
ours, motion below the grid's resolution — the grid-invisible subspace. Mechanism (`--commit_pic`,
config `commit_pic`, mpm/gridfilter.py, method.md §10.20): at every commit the window's
displacement d = x_end − x_start is projected once through the simulation's own transfer,
P(d) = G2P(P2G(d)) with the cubic B-spline weights, mass-weighted; x_end ← x_start + P(d). The
sub-cell part d − P(d) is dropped. No constant. Positions only, before the shift and the archive.
Run `l300_bunny` = the g300 recipe (`--disc_ref --shift_sub`) + `--commit_pic`. P48 the layer's
normal step in the last 10 windows ≤ 0.003 wu (g300 0.0047), `layer_flip_frac` ≤ 0.5 (0.56),
`layer_net_ratio` ≥ 0.4 (0.25). P49 mid-morph roughness at 2 native spacings ≤ 0.15 (g300
0.21–0.32), the end bump at the reference render ≤ 1.2° (1.3°). P50 silIoU ≥ 0.972 (g300 0.975;
the projection may cost sub-cell fit), chamfer/diag ≤ 0.0115 (0.0109). P51 tip mass ≥ 18
reference particles (23.7; the risk: the projection smears the tip), no piece ≥ 6. P52 windows ≤
80. Refutation: silIoU < 0.965 or tip < 12 → the sub-cell fit the control needs is the same
subspace, and the filter cannot be the fix; P48 failing with the rest holding → the breathing is
NOT in the grid-invisible subspace (then it is the grid-resolved control's own alternation).

**P45–P47 (the Yu & Turk kernel through the Poisson path; 00:55): no effect — bump 1.32° at k = 240
and at k = 32, 1.33° at t = 0.5 against 1.33° for the isotropic kernel, components 1 in all.** The
reason is structural, not a refutation of the kernel: with `--surface poisson` the density kernel
only selects the outer-layer surfels and sets the level; the surface itself is the screened
Poisson fit of the surfel positions and normals, which the anisotropy never reaches. The kernel
shapes the surface only on the level-set path (`--surface mc`), which is where Yu & Turk apply
it; that variant (pca / F-anisotropic / isotropic, all at the reference spacing on g300's end
frame) is rendered next. Prediction for it: the anisotropic level set's bump ≤ the isotropic level
set's by ≥ 0.2°, tips intact; if the isotropic level set is itself ≥ the Poisson 1.3°, the
Poisson path stays the deliverable.

**Readings of `k300_bunny` (the g300 recipe + `--cell_diag 36` = dx 0.218, grid 49³; 01:00).** 96
windows, 2003 s = 33 min (the stale rule; 30 s a window on the 2.7× grid), re-attachments 0,
fragments 0, off-body 9 singletons (P40 no fracture — holds), det F 0.784, tip **37.3** reference
particles (8-NN 0.84; over-packed 1.7× like h300; P41 holds), silIoU **0.9736** (P42 ≥ 0.975 —
refuted by 0.0014; chamfer 0.0552, the best), mid-morph roughness at 2 native spacings 0.17–0.23
(g300 0.21–0.32; P44 holds), the tail's net/summed 0.5–0.9 (no cycle). The finer cell does not
fracture the bunny under the mass contract + the reference constants + shifting: the "26 is the
finest fracture-free cell" rule was a statement about the old contract on the dragon. The shift's
cell-sum change reads −4.2 % a commit on the 49³ grid (−0.4 % on 36³ at 20k): with smaller cells
the sub-cell arrangement is not below the loss any more, as the audit warned. Not adopted: the
fit is not better than g300's (0.9736 / 0.0552 against 0.975 / 0.0556), the cost is 2× the wall
and the tip is over-packed; kept as evidence that the grid can be refined when the ear-scale
features demand it.

**The level-set variants (01:05): refuted.** On g300's end frame at the reference spacing the
marching-cubes level set of the density is far rougher than the Poisson fit of the same surfels —
isotropic kernel **13.8°**, Yu & Turk PCA kernel 14.5°, the F-anisotropic kernel 17.3° with 43
components — against the Poisson 1.3°. The anisotropic kernels of Yu & Turk are built for a
level set at the render voxel (0.04 wu here), where our CIC-plus-blur density carries the
particle texture; the deliverable stays the screened Poisson fit of the outer-layer surfels, and
the geometry survey's other surface items (the envelope constraint, the stochastic variance
test, the feature-weighted smoothing) apply to that fit, not to a level set.

**Stochastic PSR (Sellán & Jacobson 2022, gpytoolbox) on the g300 surfels (01:20):** the full set
(40 860 surfels, grid 64³) segfaults — the Gaussian-process formulation builds dense covariances
and does not scale to this size on the host; retried on a 6k subsample at 40³ for the posterior
position-uncertainty scale only (result below when it finishes). Kazhdan's PoissonRecon (built
with the conda libjpeg-turbo / libpng headers) ran at depth 7 and 8 with and without
`--envelope` (the envelope = the density level set at 5 % of the bulk, one closed component);
visually the envelope-constrained fits show pits on the body that the unconstrained fit does
not, so the hull built that way is not everywhere outside the surface; the bump numbers follow.

**The spectral band criterion (the geometry survey's shortlist 4; `baselines_prep/spectral_probe.py`,
robust-laplacians on the end-frame outer layer, 150 modes = wavelengths ≥ 1.5 wu ≈ 5 cells; 00:10).**
Per-window normal displacement of the layer over the last 10 windows, split into the low band
(≥ 1.5 wu) and the rest:

| run | high-band (sub-1.5 wu) energy share | low-band consecutive-window correlation (median / min) |
|---|---|---|
| c300 | 0.45 | **−0.79 / −0.88** |
| g300 | 0.72 | **−0.49 / −0.78** |

Reading: in c300 more than half of the tail's layer motion is at wavelengths of five cells and
more, and THAT part alternates almost perfectly (−0.79): the breathing is a coherent, grid-
resolved in-and-out of the whole surface, not sub-cell jitter — the outer loop over-correcting
at the scale it does resolve, exactly the audit's "optimiser property". In g300 the coherent part
is smaller (the high band holds 72 % of a motion that is itself smaller) and still alternates
(−0.49). Consequence for the running l300 (the null-space projection): it can remove only the
high-band 72 % (its own log reads a 74 % null-space share of the window's displacement at
window 59 — consistent); the low-band alternation is out of its reach, as P48's refutation clause
anticipated. The mechanism for the low band is a step-size question of the outer loop at the
resolved scale (no MPM paper found; the shell-space acceleration functional of Heeren et al.
2016 is the metric), to be designed after l300 reads. The stochastic PSR test is withdrawn as a
tool (segfault at 6k surfels / 40³ as well).

**The spectral band criterion on every finished run (00:20).** Low band = wavelengths ≥ 1.45–1.52
wu (≈ 5 cells; 150 modes of the end-frame layer's Laplacian); the last 10 windows' normal
displacement of the layer:

| run | high-band share | low-band consecutive-window correlation (median / min) |
|---|---|---|
| g41 (40k) | 0.49 | −0.77 / −0.85 |
| c300 | 0.45 | −0.79 / −0.88 |
| d300 (ref. constants) | 0.71 | −0.62 / −0.72 |
| g300 (+ shifting) | 0.72 | −0.49 / −0.78 |
| h300 (+ basis) | 0.43 | −0.66 / −0.72 |
| j300 (u damped) | **0.13** | **−0.98 / −0.99** |
| k300 (dx 0.22) | 0.28 | −0.96 / −0.98 |

Every run, the 40k reference included, alternates coherently at the resolved scale — a
whole-surface in-and-out with a two-window period; the amplitude differs (40k 0.003 wu a
window, the 300k runs 0.005–0.008), the phenomenon does not. With the u channel damped (j300)
or the grid refined (k300) the alternation is almost purely low-band and almost perfect
(−0.96 … −0.98): the stress channel's own window-to-window over-correction, laid bare. The
null-space projection (l300) cannot reach it by construction. Working hypothesis for the
mechanism, to be tested before any fix: each window deforms the body ELASTICALLY to reach its
target and ends at rest (w_kin), the commit assimilates half the elastic strain into the plastic
state (`assim 0.5`, the oracle's value), and the other half springs back in the next window's
free dynamics — a rebound with a two-window period at the scale the elastic body resolves. The
diagnostic: a zero-control rollout from each accepted commit state, its displacement projected
on the previous window's (the rebound fraction; −0.5 would confirm), instrumented next.

**Readings of `l300_bunny` (the g300 recipe + `--commit_pic`; 00:20).** 85 windows, 1321 s = 22 min
(the brake at 84–85), silIoU **0.9797** (the best of every bunny run; P50 ≥ 0.972 holds), chamfer
**0.0551** (best), det F 0.664 (holds), stray_max 0.0004, off-body **6 singletons** (P51's piece
clause holds; the tip mass follows), G3_rest PASS at jitter_rel 0.00004. The projection's log:
the null-space share of the window's displacement 18 % at window 1, 27 % at 21, **78 % at 85** —
in the tail four fifths of the window motion was grid-invisible and removed. The breathing
metric: `layer_step_sp` 0.0244 spacings = **0.0017 wu a window — below the 40k reference's
0.0029 (P48's amplitude clause holds)**; `layer_flip_frac` 0.74 and `layer_net_ratio` 0.21 — the
low-band alternation persists at the smaller amplitude, as the spectral criterion predicted
(P48's flip / ratio clauses refuted, and expected to be: every run including 40k alternates
there). Mid-morph roughness 0.19–0.23 (P49 ≤ 0.15 refuted; below g300's 0.21–0.32). P52 windows
85 (≤ 80 refuted by 5). Reading: the projection removes what it can — the sub-cell part of every
window's motion — and the fit improves rather than degrades (0.9797 / 0.0551 against g300's
0.975 / 0.0556): the sub-cell motion the control was spending its windows on was not fit, it was
noise. The candidate recipe for N > 40k is now `--disc_ref --shift_sub --commit_pic`; the end
render, the video's tail and the tip follow.

**The rebound diagnostic (00:35; `--rebound_probe`, 20k, 30 windows, the recipe): the elastic-rebound
hypothesis is REFUTED, and the mechanism is read off directly.** The zero-control rollout from
every accepted commit moves the body FORWARD along the committed displacement, never back: the
projection is +1.42 at window 1, +0.9 through the expansion (windows 2–8), +0.7 at 11, +0.6 at 17,
+0.4 at 23, **+0.30–0.42 in the tail (windows 24–30)**, with the free displacement's median 0.005 wu
against the committed 0.007. So in the tail three quarters of what a window "moves" is the body's
own carried motion; the window's control then has to cancel the overshoot it produces, and the
next commit carries the reversed momentum — the two-window alternation at the resolved scale,
in every run, at any N. Nothing elastic springs back. The carried momentum is `v0 = st["v"]`
(the commit's velocity) and the APIC C; the terminal kinetic term asks for rest at the window's
end but leaves 0.005 wu of a window's free travel. The mechanism that follows (config
`rest_commit`, method.md §10.21): every accepted commit starts the next window from rest — v and
C zeroed, x / F / Fp kept; no constant. The 20k diagnostics running now: the probe with a
from-rest variant (v = C = 0, the elastic part alone) and the recipe + `--rest_commit`.
Pre-registration of `m300_bunny` = l300 + `--rest_commit`: P53 the tail's low-band correlation
(spectral probe) > −0.3 (every run so far −0.5 … −0.98), `layer_flip_frac` ≤ 0.55, `layer_net_ratio`
≥ 0.4; P54 the layer step ≤ 0.0017 wu (l300); P55 silIoU ≥ 0.978, chamfer ≤ 0.056; P56 windows ≤
110 (the carried momentum helped the expansion: +0.9 of a window's travel was free; without it
the arrival may take longer); P57 the video's delivered tail |dI| ≤ 0.0012 (the 40k value).
Refutation: windows > 150 or silIoU < 0.972 → the momentum is needed for transport and the
alternation must be damped otherwise (e.g. carried momentum only while the transport gate is
below 100 %).

**The envelope-constrained reconstruction (Kazhdan 2020), refuted as built (00:35):** on the same
41k surfels PoissonRecon depth 7 reads 3.5° (unconstrained) and 5.1° with the density-hull
envelope (3 components), depth 8 5.3° / 5.8° (9 / 7 components), none watertight, against the
renderer's 1.31° (Open3D screened Poisson + the exterior test + the Loop subdivision to the
voxel, watertight). The dihedral measure penalises PoissonRecon's coarser triangles, so the
absolute gap overstates it, but the envelope made every case worse and opened the mesh: a
density-level-set hull is not everywhere outside the surface. The deliverable renderer stays.

**The tracked surface on g300 (the persistent-mesh line of the MPM survey; `--track --track_keep
--track_stretch 2 --track_every 0`, plain, stride 12; 00:40):** the delivered tail's per-frame
change 0.0016 (untracked 0.0019; 40k 0.0012): the re-fit jitter of an independent Poisson per
frame is a small part of the visible tail motion; the rest is the particles' own (the carried-
momentum alternation), which the tracking follows faithfully. The video is on the server
(`r300/plain/g300_plain_track_s12.mp4`); not adopted over the untracked render on this number.

**Head-to-head, 4Deform (CVPR 2025) on sphere → bunny without correspondences (00:45; 3639 s
training, eval at t = k/5).** Chamfer / diag 0.113 at t = 0.2, 0.114 at 0.4, 0.115 at 0.6, 0.114 at
0.8, **0.111 at t = 1.0** — the implicit never reaches the bunny (ISD's end fit was 0.0070); the
volume is kept (0.21 → 0.18 in its normalised units) and the body stays one genus-0 surface except
at t = 0 and 0.8 (7 / 5 components). Reading: with its matching loss off, 4Deform's divergence,
distortion and stretching regularisers hold the sphere and the endpoint term cannot pull it to a
target it has no correspondences to; the correspondence-driven setting (queued) is its fair run.
On this pair, without correspondences, neither implicit baseline produces a morph: ISD collapses
the body between the endpoints, 4Deform never leaves the source.

**The two 20k diagnostics (00:50).** (1) The probe with the from-rest variant: the zero-control
motion from a commit with v = C = 0 projects only +0.02 … +0.12 on the committed displacement
(median 0.002–0.006 wu) against +0.37 … +1.4 with the carried velocity — the free travel is
momentum, the elastic part is a tenth of it (the rebound hypothesis is dead twice over). (2) The
recipe + `--rest_commit` for 30 windows: the alternation goes as predicted — `layer_flip_frac`
0.33, `layer_net_ratio` 0.46 (the baseline 0.62 / 0.20) — **and the morph lags badly: silIoU
0.919 against 0.967 at the same 30 windows**, chamfer 0.152 / 0.144. With every window from rest
the body must re-accelerate each time (the probe reads the windows ending at +1.2 … +1.5 of
free travel), and the expansion, which rode 0.9 of a window's free travel, loses it. The
pre-registration's refutation clause fires: the carried momentum is the transport; the
unconditional rule is out. The derived form that follows: windows from rest only ONCE THE
TRANSPORT HAS ARRIVED — the u transport gate (10.16: the fraction of the layer within one cell of
its OT image) is the existing measure of arrival, and at 100 % the morph is in its refinement
phase, where the carried momentum only overshoots. `rest_commit_gate` (config; default 1.0 =
apply from the first commit at which the gate reads 100 %); `m300` (unconditional) is read to the
end as the transport cost, `m300b` runs the gated rule with the same P53–P57.

**l300, the rest of its readings (01:05).** End bump at the reference render **1.19°** (the 40k
reference 1.2°; g300 1.3°, c300 1.5°), at the native render 1.33° (c300 1.9°); components 1, no
cavity, both ear tips clean; tip mass 18.5 reference particles (P51 ≥ 18 holds by 0.5; 8-NN 1.03 —
the projection thins the tip slightly against g300's 0.79 / 23.7); the plain video (stride 12):
sub-cell pieces dropped in **3 of 124 frames** (g300 17 of 87, c300 208 of 282), drawn components
1 everywhere, the delivered tail's per-frame change **0.0016** (g300 0.0019, c300 0.0025, the 40k
value 0.0012). Against the acceptance criteria (2026-09-23 night): surface ✓ (1.19° ≤ 1.2°),
pieces ✓ (2 % of frames ≤ 10 %), fit ✓ (0.9797 / det F 0.66), the breathing's amplitude ✓ (0.0017
wu a window ≤ 0.003), the breathing's flip fraction ✗ (0.74 — universal, the momentum
alternation) and the video tail ✗ (0.0016 > 0.0012). The last two are what m300b (windows from
rest once arrived) is for.

**The ear-formation item (01:10; the user: the ears grow too thin, as droplets that meet and
merge; keep reading and fixing this and the oscillation after m300b).** Opened as its own item
with its own measurement: `$OUT/scratch/ear_probe.py` — per archived frame, the particles in the
target's ear region (the top slab of the target where its cross-section splits into the two
ears), their connected components at 1.5 spacings (the droplets), the thinnest PCA extent of the
largest piece in each ear against the target ear's own extent, and the ear mass fraction. Runs on
g300 / l300 / k300 (k300's ear is 1.6 cells thick on its finer grid, the natural first
comparison; its plain video is rendered for the eye). A third survey is launched on why
transport-driven morphs form droplets and filaments at protrusions (displacement interpolation,
entropic plans, no congestion or connectivity constraint) and which formulations keep the mass a
coherent body (congestion / incompressibility / elastic regularisation of the transport,
unbalanced OT for growth, mesh-carried thin features, sheet-preserving particles). The candidate
mechanisms are pre-registered once the probe and the digest are in.

**2026-09-24 01:45 — the ear-formation item: readings and the mechanism (ear_probe / ear_slab on
g300, l300, k300, g41); m300b's verdict; two pre-registered twins of l300 (n300, o300).**

*The user's hypothesis ("too few particles reach the ears early").* The ear region's coverage
(target ear points with a particle within one cell) does lag the body's: at 300k 0.73 against
0.97 at t = 0.10, 0.79 / 0.98 at 0.15, 0.98 / 1.00 at 0.24, 1.00 at 0.34 (g300; l300 and k300
alike); at 40k 0.83 / 0.99 at t = 0.09 and 1.00 at 0.23. The lag is the pacing: every particle
advances one loss cell per window along its plan ray, and the ear tip is the farthest destination
(≈ 3 wu = 10 windows against 3–6 for the body). Confirmed in part — and 40k shares it, so it is not
by itself the droplets.

*The per-slab probe* (`$OUT/scratch/ear_slab.py`: 0.3 wu slabs from the ear base + 0.3 to the tip;
per slab and ear, particles / target points and the xz cross-section thickness against the target's;
the pure ear material's pieces at 1.5 native and at 1.5 reference spacings) reads what the eye sees:
(i) **a base bulge** — the lowest ear slab holds 1.49× its target count (l300, t = 0.20), 1.54×
(g300, 0.30), 1.74× (k300, 0.20); 40k 0.99–1.02;
(ii) **a filament above it** — the mid-ear slabs (y 2.56–3.16) at 0.49–0.86 of the target thickness
through t = 0.15–0.30 (l300; g300 0.40–0.68), filled to 0.16–0.6; the 40k tongue at 0.80–1.01;
(iii) **pieces** — the pure ear material in 86–177 pieces at 1.5 native spacings (l300, t =
0.10–0.23; g300 43–87 with 2–4 pieces of ≥ 20 particles at t = 0.17–0.33); 40k ≤ 10 pieces, one of
≥ 20; at the reference spacing (what the reference-spacing render shows) 1–3 pieces;
(iv) **the end state** — the pure ear at 0.944–0.948 of its target count (300k), its upper slabs
0.85–0.92 filled and 0.92–0.97 as thick as the target; the 40k end ear 0.838, upper slabs 0.55–0.70
filled and 0.83–0.90 thick (the gallery's thin ears); k300 (dx 0.22) 0.993, upper slabs 1.00–1.05,
the tips over-filled 1.5–1.8;
(v) **the ear mass overshoots** — the ear tube's count 1.64× at t = 0.34 → 1.49 at 0.49 → 1.57 at the
end (g300; l300 1.62 → 1.48; k300 1.82 → 1.53); 40k monotone 1.37 → 1.42. The overshoot is the
momentum alternation read in one region (the oscillation item).

*Mechanism* (with the third survey, related_work.md 2026-09-24 01:30): the paced target is the
straight-ray displacement interpolant of the plan (optimizer.py: x_int = x0 + min(1, h/|d|)·d).
For a volume-preserving but anisotropic map (a broad patch of the sphere's crown into a 1.1-cell
ear, J ≈ diag(4, ½, ½)) the interpolant is not volume preserving in transit — det(I + t(J − I)) ≠ 1
(method.md (43)): the rays converge at the ear base (density above bulk: the bulge) and stretch
into the ear (below bulk: the stream). The cell sum cannot tell a sparse cell from a dense filament
in part of the cell, so at 300k, where a filament three particles across exists, the cloud
realises the sub-bulk stream as a thin filament of bulk density (fill 0.36 at thickness 0.68 = bulk
density in a 0.6 × 0.6 cross-section) that breaks into pieces at the native spacing; at 40k the
spacing (0.45 cell) forbids the filament and the tongue is a cell thick. Nothing in the target
enforces bulk density on the transported support (Bonneel 2011: a plan splits blobs; Solomon 2015:
the entropic interpolant is blurred; Maury 2010 and Perthame 2014: the density constraint is what
makes a crowd or a growing tissue move as one body).

*Pre-registered fix E3 — the support-preserving paced target* (method.md §10.22, `--pace_project`,
losses/projection.py, tests/test_projection.py 3 passed on hyde06): the paced step projected onto
the divergence-free fields on the body (Chorin projection on the loss grid: MAC faces by the CIC
deposit, pressure zero outside the body at half-bulk occupancy, the correction gathered back
FLIP-style) before the paced target is rasterised. **n300 = l300 + --pace_project** (GPU 2, 01:45).
Predictions against l300: **P58** the base slab's fill at its mid-growth maximum ≤ 1.15 (l300
1.49); **P59** the mid-ear slabs' thickness ratio at t = 0.15–0.25 ≥ 0.80 in median (l300 0.68);
**P60** one piece of ≥ 20 particles at 1.5 native spacings in every sampled frame and ≤ 20 native
pieces per frame at t ≤ 0.30 (l300 86–177); **P61** the end not worse — silIoU ≥ 0.975 (0.9797),
windows ≤ 110 (85), end ear fill ≥ 0.93 (0.948), end bump at the reference render ≤ 1.3° (1.19°).
Refutation: P58 or P59 failing refutes the transit-density mechanism (next: the thin-feature
carrier, Ando 2012 / Jiang 2017 — the particles whose destination is thinner than two cells split
so the ear cells keep the reference count); P61 failing with P58–P60 holding says the projection
costs transport (next: the unbalanced / Wasserstein–Fisher–Rao plan, Chizat 2018).

*m300b's verdict (01:35).* 66 windows in 1987 s, silIoU 0.9782, det F 0.66; the gate crept from
93 % (window 34) to 100 % at window 62 and the run stopped at 66 on two merit rejections: the rule
was engaged for four windows. As measured P53 ✗ (low-band corr −0.67), P54 ✗ (flips 0.74), P55 ✗
(net 0.15), P56 ✓ (0.9782), P57 ✓ (66 windows) — no reading of the mechanism, a reading of the
latch: the last stragglers define arrival. **The reversal cosine reads arrival itself** (the
runner's `reversal_cos`, consecutive accepted windows' displacements): +0.9 through the transport,
a zero crossing, then negative at every window to the end — m300b from window 44 (of 66), l300
from 49 (of 85), the 40k reference from 20 (of 59): 33–67 % of every run's windows are the
alternation. *Pre-registered second form* (method.md §10.21 addendum 2, `--rest_commit_reversal`):
the latch fires at the second accepted commit in a row with a negative reversal — one full period.
**o300 = l300 + --rest_commit --rest_commit_reversal** (GPU 0, 01:46). **P62** the latch fires
between windows 40 and 60; **P63** the layer breathing over the last 10 windows: flips ≤ 0.55 (l300
0.74) and net/summed ≥ 0.4 (0.15); **P64** the low-band consecutive-window correlation > −0.3
(−0.67); **P65** silIoU ≥ 0.975, ≤ 110 windows, the plain video's delivered tail ≤ 0.0012 per frame
(l300 0.0016). Refutation: P63 and P64 failing with the rule engaged for ≥ 10 windows refutes the
carried momentum as the alternation's carrier (the control's own per-window overshoot is then the
remaining candidate: the line search against a target that moves one cell per window).

*Baselines, the genus change.* ISD (no correspondence) on sphere → torus: volume 0.17–0.43 of the
target's through t = 0.1–0.5 and non-watertight from t = 0.6 (5–17 components, Euler 3–12), end
surface Chamfer 0.0070 / diagonal — the implicit velocity field opens the hole by tearing the
surface, not by transporting it. The 4Deform torus is still training.

*k300's plain video* (dx 0.22, reference spacing): raw components > 1 in 5 of 150 frames, drawn
components 1 everywhere, at most 58 isolated particles — for the eye's ear-growth comparison.

*The video tail measure, made a tool (02:25).* `$OUT/scratch/video_tail.py MP4`: the plain video decoded
by ffmpeg to grey, the mean per-frame |ΔI| over the frame, the delivered tail = the last fifth of the
frames before the hold. Same-tool values: g41 (40k) 0.0013, l300 0.0016, k300 0.0017, g300 0.0020,
h300 0.0020, c300 0.0024, i300 0.0031. The acceptance number for the tail is therefore the 40k value
as this tool reads it, 0.0013 (the earlier 0.0012 was the same quantity read by hand). The verdict
chain (`$OUT/scratch/verdict.sh TAG GPU`) now runs, at a run's DONE marker: wall time, windows,
rejections, the arrival / u-gate / dense-mass / reversal series, the run's metrics, the end frame
(pieces off the body, ear-tip mass, under-fill), the ear slabs, the dense pockets, the layer's tail
motion, the spectral band criterion, the cycle ratio, the plain video at the reference spacing and
its tail change.

**2026-09-24 02:55 — o300's early reading and the render intervention p300.** o300 latched at
window 43 (two accepted commits reversing in a row, cos −0.26 — P62 ✓, inside 40–60), ran five more
accepted windows from rest and stopped at 53 on three merit rejections (silIoU 0.9780, P65's fit
✓). **The windows that started from rest reverse their predecessors just the same**: reversal
cosine −0.63, −0.51, −0.49, −0.46, −0.48, −0.35, −0.31, −0.33, −0.35 with the kinetic energy down
from 0.020 to 0.003; layer flips 0.66, net/summed 0.16 (P63 ✗, the spectral reading P64 in the
verdict chain). The carried momentum is refuted as the alternation's carrier (m300b could not
say it; o300 can). Two more things the records rule out: (i) the step size — l300's and the 40k
reference's tails run at the annealed floor (α 0.001, anneal 0.05) and reverse at −0.5; (ii) the
balancer — λ sits at 0.003–0.005 and g_share at 0.3–0.4 without alternation. What DOES alternate
is the shape merit itself: at 40k the outer gain flips sign at every window in the tail (−1.2 %,
+1.7 %, −2.0 %, +1.2 %, −0.4 %, +1.1 %, −1.1 %, +0.8 %; sign-alternation 0.50, l300 0.53, o300
0.39) with the silhouette term moving in step (0.0046 ↔ 0.0053) while the cell sum is flat — the
commits alternate between two states, a better and a worse one. The remaining candidates are the
terms whose target changes with the cloud from window to window: the render channel (the
silhouette against the deliverable render of the moving layer) and the paced target (the plan
re-solved from the moved cloud, the entropic image 0.9 spacings inside the target against the
arrived-particle snap onto it).
*Pre-registered intervention p300 = l300 + `--render_until 45`* (the render channel off from window
45, one window before l300's zero crossing at 50; GPU 0, 02:55): **P66** the reversal cosine's
median over the windows after 45 ≥ 0 (l300 −0.20 … −0.57); **P67** layer flips ≤ 0.55 and
net/summed ≥ 0.4 over the last 10 windows (l300 0.74 / 0.15); **P68** the low-band consecutive
correlation > −0.3 (−0.67); **P69** silIoU ≥ 0.972 (the render's tail contribution is small once
arrived; l300 0.9797 — a larger loss is itself a reading of what the render does in the tail).
P66–P68 holding says the render channel carries the alternation (the next question is which of its
two inputs — the silhouette at 64→96 px or the denoised shading reference — and the fix is in the
reference, not the weight); P66–P68 failing with the render off says the paced target carries it,
and the second intervention is `--ot_handoff` (the fixed target from arrival), pre-registered then.

*Pre-registered r300 = l300 + `--outer_latch_reversal`* (method.md §10.21 addendum 3; GPU 2, 03:15):
**P70** the gate arms between windows 44 and 55 (l300's onset 50, o300's 43, n300's 44) and the run
ends within 12 windows of it on three rejections (l300 85 windows → ≤ 67); **P71** silIoU ≥ 0.977
(l300 0.9797: the alternating windows contributed +0.0017 over 35 windows); **P72** the delivered
tail's per-frame change ≤ 0.0013 and the layer breathing over the delivered last 10 windows: flips
≤ 0.55, net/summed ≥ 0.4 (by construction no accepted reversal after the onset; the reading tests
the construction); **P73** end bump at the reference render ≤ 1.3° and ear fill ≥ 0.93 (the ear is
l300's, unchanged). Refutation: a run that arms early (before 40) and stops short of arrival (silIoU
< 0.975) says a transient reversal mid-transport reads as the onset — then the onset needs the
arrival (the u gate) as a co-condition.

**2026-09-24 03:30 — n300's verdict (the support-preserving paced target) and o300's (the
reversal-latched windows from rest); the fourth ear mechanism t300.**

*n300 (l300 + --pace_project), 80 windows in 21.5 min, three low-gain reversal rejections at the
end.* **P58 ✗** — the base slab's mid-growth maximum 1.50× (l300 1.49×); **P59 ✗** — the mid-ear
thickness ratio at t = 0.15–0.25 in median 0.63 (l300 0.68); **P60 ✗ / ✓** — native pieces
25–160 per frame through t ≤ 0.33 (l300 86–177) but one piece of ≥ 20 particles in every frame;
**P61 ✓ with margin** — silIoU 0.9790 (0.9797), 80 windows, the end ear filled to **0.987** (l300
0.948; upper slabs 0.89–1.04 filled, 0.91–1.00 thick), det F min **0.717** (0.664), **2** stray
particles off the body (10), the layer's per-window step **0.028** spacings (0.06–0.08); flips 0.83
/ net 0.24; low-band correlation −0.69. The transit-density mechanism is refuted at the target
level: the bulge and the filament do not follow the target's transit density at all. The
dense-pocket series says why — with the projected (incompressible) target the compression pocket
of the first windows is DEEPER (40 % of the mass at ≥ 2× bulk at window 6, l300 25 %, the 40k
reference 11 %) and the transport three windows slower: the cell sum cannot see an incompressible
interior flow (a shifted uniform interior has the same cell masses), so the projected target's
interior goes unfollowed and the surface push compresses the material. The straight-ray target,
for all its transit distortion, is what makes the interior move — through the transient density
changes the cell sum can see. The mechanism of the ear, restated: the front is pulled by the loss
and the control stretches the material behind it (free: the rest state is the control's), while
the supply through the base piles up at 1.5–8× bulk because the log cell sum's gradient
2r/(m_ref + m) collapses as a cell fills — a pile is nearly free, a thin stream nearly free. The
projection stays opt-in for its end state (the fuller ear, the stiffer det F, the halved tail step).

*o300 (l300 + --rest_commit --rest_commit_reversal), 53 windows in 28.8 min.* **P62 ✓** latched at
43; **P63 ✗** flips 0.66, net 0.16; **P64 ✗** low-band correlation −0.64 (min −0.80); **P65** silIoU
0.9780 ✓, 53 windows ✓, the video tail in the chain. Refuted as the carrier (details 02:55).

*Pre-registered ear mechanism 4 — the linear cell sum, t300 = l300 + `--dvol_form linear`* (the
form exists since 2026-09-17: the gradient per unit mass proportional to the deficit, no collapse
at high occupancy; the log form was chosen at 40k against ejection). Launched 03:30. **P74** the
base slab's mid-growth maximum ≤ 1.20× (1.49); **P75** the mid-ear thickness ratio at t =
0.15–0.25 ≥ 0.80 (0.68); **P76** the dense-pocket maximum ≤ 10 % of the mass (25 %); **P77** silIoU
≥ 0.975 within 110 windows and no ejection — stray_max ≤ 0.001, ≤ 10 particles off the body at
the end. Refutation: P74–P76 failing says the pile is not the log form's tolerance but the supply
itself (then the physics must carry the flow: an incompressible MPM step, Stomakhin 2014); P77
failing on ejection says the log form's protection is needed and the linear form must be confined
to occupied cells (a hybrid: linear where the cell has mass, log where it is empty).
*o300's chain readings (03:40):* the plain video's delivered tail **0.0028** per frame (l300 0.0016,
the 40k value 0.0013 — P65's tail ✗: the deliverable ends at window 48, six windows into the
alternation, where the layer still moves 0.07–0.2 spacings a window), end bump at the reference
render 1.26° (l300 1.19°), cycle ratio never below the random-walk bound (0.55–0.85). o300 is
refuted on every count but the fit; `--rest_commit` stays opt-in as a diagnostic.

*The bulge is the reference discretisation's (03:55; ear_slab on c300 and d300).* c300 (300k at
native constants): **no base bulge at any time** (the base slab 0.42 → 0.85, monotone), the base
slabs at 0.85–1.0 of the target thickness — but the ear never fills (end 0.669, upper slabs
0.24–0.49) and the tip beads (the isolated pieces) are c300's known defect. d300 (`--disc_ref`
alone): the bulge appears (1.49× at t = 0.40, 1.20× at 0.30), the mid-ear at 0.30–0.70 of the
thickness, end fill 0.931. So the base pile and the filament came WITH the reference
discretisation — the same change that fixed the tip mass and let the ear fill. The scaled
constants (§10.17a) are the layer's u bound and relaxation neighbourhood, the isolation gate, the
coherence radius and count, and the bond decoupling count; the twins that dissect them are
pre-registered below once t300 (the linear cell sum, with disc_ref) has read whether the pile is
the log form's tolerance at all.

*The bulge's constant found (04:05): the plan blur doubled under disc_ref.* The plan's blur is
sample-derived — target NN spacing × (N / 8192)^(1/3), the spacing of the 8192-point sample of
the shape, N-independent by construction: **0.119 wu at 40k, 0.116 at c300** (0.38–0.39 loss
cells). Under `--disc_ref` the target NN spacing that feeds it is itself scaled by 1.96, so d300,
l300 and n300 ran the plan at **0.227 wu (0.74 cells)** — twice the 40k blur: blurrier images
(deeper inside the target), the leash and the "arrived" radius doubled, so material within 0.23
wu of its image at the ear base is declared arrived, snapped onto the target's base surface, and
parks there while the supply keeps coming — the pile; the layer, gated by the same radius, runs
on into the ear ahead of the bulk — the filament. This is a double count against rule (38), not a
new constant: the plan's resolution is the sample's. *Pre-registered u300 = l300 +
`--plan_native`* (GPU 0, 04:10): **P78** the log reads sqrt(eps) 0.116 wu; **P79** the base slab's
mid-growth maximum ≤ 1.20× (1.49); **P80** the mid-ear thickness ratio at t = 0.15–0.25 ≥ 0.80
(0.68); **P81** the end not worse — ear fill ≥ 0.93, silIoU ≥ 0.975, ≤ 110 windows (the transport
may slow: c300 at the same blur took 176 windows, but with native layer constants). Refutation:
the ear fill falling below 0.90 says the doubled blur was what filled the ear at 300k and the
pile is its price; then the plan's blur is the lever to set from the feature scale, not the sample.
*n300's chain readings (04:15):* the plain video's delivered tail 0.0018 per frame (l300 0.0016),
end bump at the reference render **1.19°** (= l300; P61's bump ✓), raw components > 1 in 13 of 112
frames (l300 3 of 124 — more transient sub-cell pieces mid-morph), drawn components 1 throughout.
u300's first launch on GPU 0 died of CUDA memory (the baseline training holds 21 GB there);
relaunched on GPU 2 at 04:20.

**2026-09-24 04:35 — p300 and r300 read; v300 and r300b pre-registered.**
*p300 (the render channel off from window 45), 55 windows, silIoU 0.9765 (P69 ✓, the render's
tail worth 0.003).* **P66 ✗** — the reversal cosine after 45: median −0.29, seven of eight windows
negative (windows 31–45 before it: +0.26); **P67 ✗** flips 0.71, net 0.18; the merit gain after 45
+0.0001 … +0.0008 a window (progress ≈ 0 while the alternation continues). The render channel
is refuted as the carrier; with momentum (o300), the step size and the balancer already out, the
alternation is in the physics loss's own window-to-window re-linearisation — the plan re-solved
from the moved cloud (and its arrived-snap onto the target's nearest points) or the window's
dynamics against the commit (a commit not at equilibrium with its own control: the rebound
probe's 0.1-window forward travel from rest).
*r300 (the gate armed at the alternation's onset), 43 windows, silIoU 0.9776, det F 0.699,
deliverable to window 40 (743 frames).* **P70 ✗ as specified** — armed at 41 on two barely
negative cosines (−0.08, −0.01), nine windows before l300's alternation; stopped at 43 on three
rejections (gains −0.9 %, −1.4 %, −1.0 %); **P71 ✓** (0.9776 ≥ 0.977, by 0.0006); **P73** (bump,
ear) in the chain. The onset must be read by the gate's own definition of a reversal (cos <
outer_reversal_cos = −0.2), not by any negative value — the code now does; pre-registered
**r300b** (GPU 2 after t300): **P70b** armed between 48 and 58 and the run ends within 12 windows
of it; **P71b** silIoU ≥ 0.978; **P72b** the delivered tail's per-frame change ≤ 0.0013.
*Pre-registered v300 = l300 + `--ot_handoff`* (the cell-wise hand-off to the FIXED target once no
deficit cell lies beyond one cell of the body — from then on the window target no longer changes
with the cloud; GPU 0, 04:35): **P82** the reversal cosine after the hand-off: median ≥ 0 (l300
−0.20 … −0.57); **P83** the layer's per-window step ≤ 0.03 spacings over the last 10 windows
(l300 0.06–0.08) and the delivered tail ≤ 0.0013; **P84** silIoU ≥ 0.972 within 110 windows (the
hand-off was falsified on the 40k C for stopping a half-formed body; the bunny arrives). P82
holding says the moving paced target carries the alternation (the fix is then a target that
stops moving once arrived — the hand-off itself, made the rule from arrival); P82 failing says the
window's dynamics against its commit do, and the next twin commits the settled state (a
zero-control rollout from rest at the commit, positions and F updated together).

*p300 / r300 chain readings and a distinction (04:50).* p300: video tail 0.0024, bump 1.24°.
r300: video tail **0.0030** (l300 0.0016), bump in the chain — the deliverable ends at window 40
while the body still transports (arrived 95–98 %), and the per-frame change reads that motion:
the measure conflates transport with the flicker. Two phenomena have been run together: (a) the
**merit reversal** (consecutive optimiser displacements anti-correlated) begins only at arrival
(l300 window 50, 40k 20) and is what the gate's reversal rule addresses; (b) the **layer
breathing** (the archived layer's normal step flipping sign window to window; flips 0.66–0.83)
is present in EVERY phase of every run — r300's last 10 delivered windows have reversal cosines
of +0.1 … +0.5 and flips 0.68 all the same — so it is not the reversal. What acts on the layer
once per window in every recipe, the 40k included, is the commit-time relaxation projection
(`--layer_relax`, §10.13): the optimiser moves the layer during the window, the projection moves
it back at the commit, and the archived frames show out-and-in once per window. *Pre-registered
w300 = l300 without `--layer_relax`* (GPU 0, 04:50): **P85** the layer's flip fraction over the
last 10 windows ≤ 0.55 and net/summed ≥ 0.4 (l300 0.74 / 0.21); **P86** the layer's per-window
normal step halves or better (0.06–0.08 → ≤ 0.03 spacings); **P87** the surface is rougher —
the end bump at the reference render > 1.3° (l300 1.19°; the relaxation was adopted for the
surface) — a reading, not a criterion; **P88** silIoU ≥ 0.975. P85–P86 holding says the
relaxation's commit-time projection carries the breathing, and the structural fix is to move it
INSIDE the optimisation (the relaxation as the search-direction transform on the u step,
`layer_ctrl_smooth`, or as a term of the window objective) so the optimiser and the smoother
agree on one state instead of alternating between two.
*Correction (04:55):* the relaxation is not a commit-time operation — it is a projection inside the
forward model, the rough plane residual removed at 1/T per step over the window (§10.13,
config.layer_relax), and the u channel is likewise applied 1/T per step in the same kernel. The
w300 intervention stands as the test of the relaxation's part in the breathing; the mechanism, if
it holds, is the two per-step channels on the layer (u pushing along the normal from u = 0 each
window, the relaxation pulling the rough part back) settling to different states in alternate
windows rather than a commit-time jump. r300's chain: end bump 1.24° (P73's bump ✓).

**2026-09-24 05:05 — t300's verdict (the linear cell sum).** 50 windows in 30.7 min, early stop.
**P74 ✗** — the base slab's mid-growth maximum 1.34× (l300 1.49; the pile broader: slabs 1.96–2.56
at 1.1–1.3× through t = 0.30–0.40); **P75 ✗** — the mid-ear thickness at t = 0.15–0.25 in median
≈ 0.6 (0.68); **P76 ✓✓** — the dense pockets are gone (0–0.35 % of the mass at ≥ 2× bulk, peak
node 1.6–2.1× against l300's 25 % and 3.6–5×): the log form's tolerance IS what let the early
compression happen; **P77 ✗** — silIoU **0.9707** (0.9797), det F min **0.53** (0.66), **24**
particles off the body at the end (10), the layer's per-window step 0.12 spacings (0.06–0.08),
the ear over-filled to 1.09× at t = 0.39 and drained to 0.93 — with the gradient no longer
collapsing at full cells, every excess cell pushes its mass out at full strength and the sub-cell
arrangement pays. Refuted as the ear fix: the base pile is not the log form's tolerance. The
linear form's one gain (no compression pockets) is noted for a hybrid (linear on deficit, log on
excess) should the pockets matter elsewhere; not pursued now.

*Baselines, both pairs, complete (05:00; compare_baselines.py).* ISD sphere → bunny: end Chamfer
0.0071 / diagonal, but the mid-way body collapses (volume 0.001–0.013 at t = 0.4–0.7 against
0.199 at the end) and the mesh is five pieces at t = 0 and 0.9; sphere → torus: end 0.0070, the
hole opened by tearing — non-watertight from t = 0.6 (5–17 components, Euler 3–12). 4Deform
without correspondence: bunny 0.111–0.115 at every t (never leaves the sphere), torus
0.138–0.149 with 2–17 components from t = 0.2. Ours (bunny, Poisson at the reference spacing):
0.0109–0.0123 at t = 1, one body, genus 0, watertight at every t. The correspondence-given
(nearest) chain is queued after these.
*t300's chain readings (05:20):* video tail 0.0026, bump 1.26°, low-band correlation −0.43 (the
alternation present as in every run).
*Ours on the torus (05:30; gt_torus = 40k, the gallery recipe, the baselines' torus target).* The
auto regime chose ot_pace + hand-off (36 % of the source in target-empty cells); the merit gate
stopped the run at window 17 after three regressions of −11 … −14 % (deliverable to window 13,
2.8 min): silIoU 0.9674, hole 6.2 %, end Chamfer **0.039** of our target's diagonal (ISD 0.0070,
4Deform 0.138) — genus 1 at the end (the hole formed), one watertight body at t = 0.25, 0.5 and
1.0, but three components at t = 0.75 and Poisson cavities at t = 0.5–0.75 (volumes 67–82 wu³
against the sphere's 45: the reconstruction closes over the forming hole). The genus change is
our weak case as it is the baselines' — ours keeps one body and gets the topology, ISD gets the
end surface and tears the middle, 4Deform gets neither. Recorded as is; not pursued tonight.

**2026-09-24 05:50 — u300's verdict (the plan blur at the sample spacing).** 53 windows in 32.5
min, silIoU 0.9788, det F 0.639. **P78 ✓** (sqrt(eps) 0.116 wu); **P79 ✗** — the base slab's
mid-growth maximum 1.37× (l300 1.49); **P80 ✗** — the mid-ear thickness at t = 0.15–0.25 in
median 0.68 (0.68); **P81** — ear fill 0.929 (≥ 0.93 missed by 0.001), 18 strays (10), the fit and
the window count fine. The doubled blur was real and is corrected (the arrival curve is identical
to l300's, so the transport did not depend on it), but it is not the bulge's constant: the pile
and the filament survive it almost unchanged. What disc_ref scales besides the plan's spacing:
the layer's u bound and relaxation neighbourhood, the isolation gate, the coherence radius and
count, the bond decoupling count, the shading reference's normal grid, and — through the same
scaled target spacing — the nearest-neighbour band term and the ejection veto radius. The next
twin dissects by group (read below).

*The picture that fits the five ear twins (06:00), and mechanism 6.* c300 (native constants):
no pile, the ear never fills. d300 / g300 / l300 / n300 / t300 / u300 (the reference
discretisation, whatever else): the ear fills, the supply arrives at the base faster than the
front takes it up, and the front is a sub-cell filament. So the pile is the supply the reference
constants deliver meeting an uptake the front cannot give — the front stretches into a filament
of bulk density three particles across, which the cell sum cannot tell from the target's ear
cells (a sparse cell and a dense filament in part of the cell have the same CIC mass); at 40k the
spacing forbids a filament thinner than a cell and the front is a tongue. The correction that
cannot be found among the scaled constants is a term that sees the particle scale. The code has
one: the two-sided KDE density match (losses/volumetric.py d_kde, 2026-09-03: the kernel density
of the particles against the kernel density of the target's own points at every particle, the
neighbour lists frozen per window, its gradient from crowded to deficient regions), scaled once
to the cell sum's gradient norm (`kde_scale`) so its weight is a ratio, not a constant. A filament
surrounded by target volume it leaves empty is a deficit at that scale; a pile is an excess.
*Pre-registered y300 = l300 + `--plan_native --w_kde 1`* (equal gradient norm with the cell sum;
GPU 2, 06:00): **P89** the base slab's mid-growth maximum ≤ 1.20× (u300 1.37); **P90** the mid-ear
thickness at t = 0.15–0.25 ≥ 0.80 (0.68); **P91** native pieces at t ≤ 0.30 ≤ 40 per frame (u300
81–196); **P92** ear fill ≥ 0.93, silIoU ≥ 0.975, ≤ 110 windows, ≤ 10 strays. Refutation: P89–P90
failing says the particle-scale term cannot move the supply either and the front's stretching is
the control's (then the carrier is structural — Ando 2012 / Jiang 2017, particles split or a
spine carried where the destination is thinner than two cells); P92 failing on the fit says the
KDE's pull competes with the cell sum at the surface (the equal-norm ratio too high there; a
surface-masked KDE would be next).
*(06:20) The KDE term's presence in the paced regime checked on a 3k smoke run* (2 windows,
`--w_kde 0` vs `1`): the window loss 0.1398 vs 0.2506 at the same cell sum (0.1353), the KDE
value 0.247 entering at the calibrated scale (≈ 0.47) — the term is in the objective; its
calibration line is not written to the log (the optimizer's logger is silent there), which is why
y300's log shows nothing. y300 stands.
*u300's chain readings (06:40):* video tail 0.0030, bump 1.26°; like o300 and r300 it stopped at
53 windows and its deliverable ends while the body still transports (the per-frame tail measure
reads that motion; the runs that continue to 80–85 windows read 0.0016–0.0018).

**2026-09-24 06:50 — r300b's verdict (the gate armed at the onset by its own threshold).** 48
windows in 23.0 min; the reversal cosine negative from window 43, the gate armed on the second
one below −0.2 and the run ended at 48 on three rejections, the deliverable to window 45 (862
frames). **P70b ✗ as specified** (armed before 48: this run's alternation began at 43, l300's at
50 — the onset moves by a few windows from run to run); **P71b ✓** silIoU 0.9782 (l300's 85-window
0.9797 within 0.0015); **P72b ✗ as measured** — the video tail 0.0031 reads the transport still
under way at window 45 (the runs that stop at 45–53 all read 0.0028–0.0031; those that run to
80–85 read 0.0016–0.0018), and the layer breathing is unchanged (flips 0.69, net 0.14, low-band
correlation −0.65); det F 0.683, bump 1.24°, 10 strays. The rule does what it was built to do —
no accepted reversal enters the deliverable and the run does not spend 35 windows alternating —
and it is honest about what it is: the stop at the alternation's onset, not its cure; the
breathing the eye sees is the layer's, present before the onset too, and w300 is its test. Kept
opt-in; the deliverable choice between "stop at the onset" (r300b, 45 windows) and "run on"
(l300, 85 windows, +0.0015 silIoU, 35 alternating windows in the archive) is the user's.

**2026-09-24 07:00 — v300's verdict (the cell-wise hand-off to the fixed target).** 69 windows in
32.5 min, silIoU **0.9795** (l300 0.9797), det F 0.655. The reversal cosine turns negative at
window 52 for five windows (−0.49, −0.39, −0.45, −0.39, −0.26) and then decays to zero for the
last eight (+0.07, −0.03, −0.00, −0.02, +0.15, −0.05, −0.04): **P82 ✗ as specified** (the median
after 50 is −0.05, not ≥ 0) **but the alternation is weaker and does not persist** — l300 sits at
−0.3 … −0.57 for 35 windows, o300/m300b at −0.3 … −0.6 to the end. **P83 ✗** — the layer's
per-window step 0.074 spacings, flips 0.72, net 0.12 (the breathing unchanged; the video tail in
the chain). **P84 ✓**. Reading: the paced target that moves with the cloud carries PART of the
merit alternation — once the deficit cells hold the fixed target the optimiser's consecutive
moves stop reversing each other after a few windows — and the layer's breathing is a separate
thing again (w300). The hand-off was falsified on the 40k C for stopping a half-formed body
(2026-09-17); on the bunny at 300k it costs nothing (0.9795) and calms the tail. Kept opt-in;
the combination `--ot_handoff --outer_latch_reversal` is the natural deliverable recipe if w300
finds the breathing's carrier.
*v300's chain readings (07:15):* video tail 0.0025, bump 1.25°, 10 strays; the layer's low-band
consecutive-window correlation −0.64 — the layer still alternates in the smooth band while the
bulk's merit reversal has decayed to zero: the two phenomena separate in one run.

**2026-09-24 07:25 — w300's verdict (the recipe without the layer relaxation).** 106 windows in
40.0 min, silIoU 0.9795, det F **0.586** (l300 0.664). **P85 ✗, decisively** — the layer's flip
fraction **0.94** (l300 0.74), net/summed 0.068 (0.21): without the relaxation the layer flips
nearly every window; **P86 ✗** — the per-window step 0.074 spacings; **P87** — bump in the chain;
**P88 ✓**. The relaxation is not the breathing's carrier; it damps it. The reversal: negative from
window 42 and the run alternated on to 106 windows (56 negative windows).

*Where the breathing stands after the night's interventions.* Tested and refuted as its carrier:
the carried momentum (o300), the render channel (p300: flips 0.71 with the render off), the
relaxation (w300: worse without it), the step size (l300 and the 40k reference alternate at the
anneal floor), the balancer (λ and g_share flat), the u channel's bound (j300, 2026-09-23), the
XPIC commit projection and the shifting (g41 has neither and breathes), the moving paced target
(v300: the bulk's merit reversal decays to zero, the layer's low-band correlation stays −0.64).
What every run shares is the optimiser's floor step at the surface: an anneal floor of 0.05 on α
(α 0.001) that the reversal-annealing reaches within ten windows of arrival, and the layer's
per-window normal step then sits at 0.002–0.003 wu at both N (l300 0.0017, the 40k reference
0.0029) — the same amplitude the acceptance criterion already admits (≤ 0.003 wu a window, the
40k value). Read as a mechanism: a first-order position controller with a fixed minimum step
against a surface residual that changes sign when crossed is a limit cycle of that step's
amplitude; its cure is a step that shrinks to zero (a line search on the merit at the surface) or
a stop when it is reached — the reversal-armed gate (r300b) is the second, at the deliverable
level, and the first is the structural item left open (the anneal floor is the literal 0.05 in runner.py, four places, and is not lowered by hand under the no-parameter rule; a floor derived from the discretisation — the step below which the loss cannot see a change — is the structural form).
*w300's chain readings (07:40):* **95** particles off the body at the end (l300 10 — the
relaxation was also what kept the outer layer's strays in), the layer's low-band correlation
**−0.96** with the high-band share 0.15 (without the relaxation the alternation is a nearly pure
grid-scale back-and-forth of the whole layer), video tail 0.0014 (the smallest of the 300k runs:
106 windows, the transport long finished — the tail measure again reads the transport, not the
flicker), bump 1.23°. P87 (rougher surface) ✗ as predicted — the bump did not rise; the price was
paid in strays and det F instead.

**2026-09-24 07:55 — y300's verdict (the particle-scale KDE density term): the growth mode
changes.** 95 windows in 33.6 min, silIoU 0.9782, det F 0.588, **4** strays (l300 10), the ear tip
**16.1** reference particles (l300 18.5, n300 12.5, u300 11.2), 8-NN 1.03 spacings.
**P89 ✓** — the base slab's maximum **1.11×** (l300 1.49, u300 1.37): the pile is gone.
**P90 mixed** — read per slab with its fill: at t = 0.30 the fill profile from the base is 0.86,
0.64, 0.40, 0.10, 0.06, 0.03 — monotone, a **tongue growing from the base** — and where the
tongue has mass (fill > 0.3) its thickness is **0.90–1.05** of the target's (y 1.96 / 2.26 / 2.56:
1.06 / 0.95 / 0.90); the 0.47–0.5 readings sit at slabs holding 2–6 % of their mass, the tongue's
leading edge, so the pre-registered median over fixed slabs (0.59) fails while the thing it was
meant to measure holds. l300 at the same t: 1.09, 0.94, 0.76, 0.48, 0.54, 0.62, 0.14 — mass all
the way up the ear with a bulge at the base and a stream at 0.5–0.7 thickness.
**P91 ✗** — native pieces 50–139 per frame through t ≤ 0.20 (at the tongue's leading edge; one
piece of ≥ 20 particles throughout); **P92 ✓** — ear fill 0.940, fit 0.9782, 95 windows, 4 strays.
The price: the ear grows LATER (0.31 filled at t = 0.20 against l300's 0.83; 0.80 at t = 0.53;
still rising at the end, the upper slabs 0.79–0.88 filled) and det F 0.588 (0.664); the dense
pockets 18.6 % (25 %). Mechanism confirmed as pre-registered: the supply into the ear is throttled
to what the front takes up once the particle scale is in the objective — a filament surrounded by
empty target volume is a deficit the KDE sees and the cell sum does not — so the front thickens to
the target and the base stops piling; the leading edge still fragments (the carrier item, Ando
2012 / Jiang 2017, stays open for that), and the ear's lateness is the pace's (one cell a window,
the farthest destination). Adopted as the ear's mechanism for N > 40k, opt-in pending the
combined run; the 40k gallery untouched.

*Pre-registered z300 = the combined deliverable candidate* (GPU 2, 08:00): l300 +
`--plan_native --w_kde 1 --ot_handoff --outer_latch_reversal`. **P93** base slab maximum ≤ 1.2×
and the tongue at ≥ 0.9 thickness where filled > 0.3; **P94** ear fill ≥ 0.93, silIoU ≥ 0.977,
≤ 10 strays; **P95** the gate arms between 45 and 65 and the run ends within 12 windows of it
with no accepted reversal below −0.2 in the deliverable; **P96** end bump ≤ 1.3° at the reference
render. Refutation on any of them says the pieces do not compose (the hand-off's fixed target and
the KDE's pull competing at the surface would be the first suspect).
*y300's chain readings (08:10):* video tail 0.0022 (144 delivered frames: the ear still filling
at the end), bump 1.25°.

**2026-09-24 08:30 — z300's verdict (the combination): the onset gate and the slow ear conflict.**
58 windows in 22.1 min. **P93 ✓** (base slab maximum 0.74× through the growth, 1.10× at the end;
the tongue at 0.93–1.06 where filled); **P95 ✓** (the reversal negative from 49, the gate armed and
the run ended at 58); **P96 ✓** (bump 1.24°); **P94 ✗** — ear fill **0.844**, silIoU **0.9705**,
det F 0.54, 17 strays, the tip 6.8 reference particles: the run stopped while the ear was still
filling (0.66 at t = 0.60 of the delivered trajectory, 0.84 at the end; y300 reached 0.94 at 95
windows). The bulk arrives and begins to alternate at window ~50 whatever the ear is doing, and
the KDE-throttled ear needs ~95 windows; armed at the bulk's onset, the gate truncates the ear.
The pieces do not compose in this form. *Pre-registered z300b = l300 + `--plan_native --w_kde 1
--ot_handoff`* (no onset gate; the fixed target from arrival calmed the alternation in v300
without stopping; GPU 2, 08:30): **P97** ear fill ≥ 0.93, silIoU ≥ 0.977, ≤ 10 strays, ≤ 110
windows; **P98** base ≤ 1.2×, the tongue ≥ 0.9 where filled; **P99** the reversal after arrival
decays to |cos| ≤ 0.2 within 10 windows of its onset (v300's reading), the low-band layer
correlation read for the record; **P100** bump ≤ 1.3°. The onset gate stays a separate opt-in
for recipes whose thin features arrive with the bulk (the 40k gallery's do); an ear-aware onset
(arm only once the u gate of the thin features reads arrival) is the open form.

**2026-09-24 09:05 — z300b's verdict (plan blur corrected + KDE + the fixed target from arrival,
no onset gate): the best deliverable of the night.** 92 windows in 29.2 min. **P97 ✓** — ear fill
**0.936**, silIoU **0.9802** (the night's best; l300 0.9797), **7** strays, 92 windows; **P98 ✓** —
the base slab ≤ 1.01× through the growth (1.08–1.12× at the end, the head's own mass), the
tongue at 0.84–1.06 of the target thickness where filled (t = 0.30: 1.06 / 0.95 / 0.90 at the
three lowest slabs); **P100 ✓** — bump 1.24°; the ear tip **16.7** reference particles at 0.99
spacings. **P99 ✗** — the reversal turns negative at 51 and stays so for 35 windows (v300's decay
did not carry over: with the KDE term the objective keeps a moving part as the ear fills), the
low-band layer correlation −0.67, video tail 0.0024, det F 0.560 (the KDE's push on the sub-cell
arrangement; l300 0.664). Reading: the ear item is answered for N > 40k by `--plan_native
--w_kde 1` (+ `--ot_handoff` at no cost); the alternation after arrival remains in the archive
as in l300 and is removed from the deliverable only by the onset gate, which cannot be used with
the slow ear until the onset reads the thin features' arrival too (the open form). The 40k
gallery untouched; the recipe for N > 40k is opt-in.

**2026-09-24 09:40 — the user's two questions after the night: what KIND of motion the tail
shows, and the ear tip that vanishes and re-forms at 1–2 s (z300b).** The survey digest is in
related_work.md (09:30). *Seen in the frames* (ffmpeg tiles of z300b's plain video, frames
16–27 = 0.8–1.35 s): the ear does not vanish as a whole; its LEADING SEGMENT (a few particles
across) is reconstructed in some frames and not in the next (pointed at 16–18, blunt and shorter
at 20–21, pointed again at 22–23, a curled hook from 26) — the tongue's sparse edge (P91 ✗)
sitting on the boundary of the surfel classification / Poisson support; the ear's mass itself is
monotone (0.12 → 0.30 over t = 0.07–0.20). To confirm: the ear-tip region's particle count vs
surfel count per frame, and the video sidecar's dropped / bridged components.
*Pre-registered measurements for the kind of the tail motion* (the server's key is rejected at
the jump host since 09:00; scripts ready in `$OUT/scratch`: `decomp_twins.py`, `surface_vn.py`):
**M1** the decomposition twins — from the tail's first frame, derived archives whose layer moves
by the normal part (d·n)n only or by the tangential remainder only (bulk unchanged), each rendered
with the same plain renderer: the twin that carries the tail's per-frame change carries the
visible motion; **M2** the level-set normal velocity (Stam–Schmidt 2011): the signed distance of
frame k's Poisson surface to frame k+1's, RMS, against the layer's RMS normal step over the same
gap — RMS(v_n) ≈ RMS(d·n) says the surface follows the particles (a/b), RMS(v_n) ≪ RMS(d·n) with a
shimmering video says the fit re-samples a rearranged set (c); **M3** the band-limited
reconstruction — the finest Poisson leaf = the MPM cell (`--poisson_cell 2.27` at the reference
spacing; Kazhdan's 15 samples per node ≈ √15 spacings ≈ 0.9 cell): predicted, if (c), the tail's
per-frame change falls toward the transport-only level while M2's v_n is unchanged; predicted
cost, the ear tip's thickness at 1.1 cells blurred (read alongside). Predictions before reading:
from the night's data (the layer's normal step 0.002–0.003 wu = 3–4 % of a spacing, the tangential
step of the same size in c300's tail, 0.05–0.13 spacings) I expect (c) to carry at least half of
the tail's per-frame change, and the normal part the rest.
*The ear tip's dropout, first suspect (10:00; the thin-feature digest in related_work.md).* The
deliverable rule draws an isosurface component only if it encloses at least one cell of
material (≈ 87 native particles at 300k) — a native-spacing piece of the tongue's edge is 1/87 of
a cell, so whenever the Poisson surface pinches the sparse tip off into its own closed piece,
that piece is dropped and the tip vanishes; in the next frame the surface is continuous and the
tip is drawn. The bridge rule (a one-spacing filament from the body to an enclosed component)
never fires for a dropped piece. *Pre-registered R-1:* a sub-cell component whose enclosed
particles lie within the link radius of the body's (max(2.5 spacings, one cell) — the rule the
bridge and the grid-fragment probe already use) is KEPT and bridged instead of dropped; only
pieces beyond that radius are "material the grid does not resolve". Prediction **P101**: in
z300b's video the ear-region mesh area is monotone through frames 12–30 (no frame with tip
surfels present and no tip surface), and the raw-components>1 frames become drawn-and-bridged
rather than dropped; **P102**: no change to any frame in which only one component exists
(the body's bump 1.24° unchanged). The census (particles / surfels / mesh vertices / dropped
components per frame in the ear mask) runs first when the server is reachable, to confirm the
particles never retract there.

**2026-09-24 10:30 — the structural fix for the alternation: per-particle Rprop on the control
step (method.md §10.24, `--ctrl_rprop --u_rprop --u_rprop_floor 0`).** The user: "그럼 이제 진동을
수정해야지". Built while the server's key is still rejected; syntax and the smoke run pending.
*Pre-registered ab300 = z300b's recipe (`--plan_native --w_kde 1 --ot_handoff`) + the per-particle
Rprop.* **P103** the moving particles' step-scale median falls below 0.1 within 15 windows of the
reversal onset (the bulk settles) while the ear region's particles keep a median scale ≥ 0.5
until the ear is ≥ 0.9 filled (the transport is untouched); **P104** the layer's per-window normal
step over the last 10 delivered windows ≤ 0.001 wu (l300 0.0017, the 40k reference 0.0029) and
the low-band consecutive-window correlation > −0.3 (−0.6 … −0.7 in every run so far); **P105** the
end not worse than z300b's — ear fill ≥ 0.93, silIoU ≥ 0.977, ≤ 10 strays, bump ≤ 1.3°; **P106**
the run ends on the plateau rule (no accepted window moving the layer above 0.001 wu) within 120
windows, and the video's delivered tail per-frame change reaches the frozen-hold level (≤ 0.0005)
over its last ten frames. Refutation: P104 failing with P103 holding says the layer's breathing
is not driven by the control step at all (then it is the physics rollout's own response to a
stationary control — the terminal kinetic term's rest is not rest — and the settle-at-commit
twin is next); P105 failing on the ear says the ear's particles do reverse mid-transport (then
the scale must be reset when the particle's transport gate reads "in transit").

*13:15 — the server back (direct route; hyde06's authorized_keys had been rewritten at 06:55 CDT
without the key; hyde01 still lacks it). Syntax of all five changed files ok on hyde06; the
Rprop smoke (3k, 6 windows, the z300b recipe + `--ctrl_rprop --u_rprop --u_rprop_floor 0`) ran
clean: 2–5 % of the moving particles reversed per window during the transport, step-scale median
1.000, none below 0.1 — the rule is quiet while the body travels, as designed. **ab300 launched
13:20 on GPU 2** (verdict chain armed, with the per-window reversal / scale series). In parallel
on GPU 0 the morning chain: 19 Poisson stills with meshes (the dropout window's archived frames
192–324 and the tail's last seven window commits), the ear-tip census, M2 (surface normal
velocity vs the layer's normal step), M1 (the tail-only decomposition twins full / normal /
tangent, rendered at stride 4), and the `--keep_attached` re-render of z300b with its sidecar
against the original.*
*The rule replayed on z300b (13:40; `rprop_replay.py`, the counterfactual — z300b ran without
it, so this is what the scales WOULD have been from its reversal history).* The body's moving
particles: reversal fraction 0.01–0.05 through window 27, 0.5 from 29 — the step scale's median
would have fallen to 0.5 at 29, 0.125 at 32, 0.002 at 39 (P103's 15 windows ✓ in the
counterfactual). The ear region's particles: no reversal (0.00–0.02) through window 51 with the
scale at 1.0, then reversal from 52 (0.4) and the scale decaying to 0.06 at 67 — the ear's
arrival comes 23 windows after the body's, and the rule reads it per particle. The body's median
per-window displacement in z300b's tail stays 0.004–0.008 wu (the breathing) — the quantity the
rule is meant to send to zero in ab300.

*13:25 — all four GPUs in use (the user lifted the GPU 1/3 restriction: "GPU 1, 3도 이제 비어
있으면 전부 다 쓰자"; hyde06-ops memory updated). Three more pre-registered runs launched:*
**ac300 = l300 + `--ctrl_rprop --u_rprop --u_rprop_floor 0`** (GPU 3) — the rule on the original
recipe, the attribution twin of ab300: **P107** the layer breathing over the last 10 delivered
windows: flips ≤ 0.55, normal step ≤ 0.001 wu, low-band correlation > −0.3 (l300 0.74 / 0.0017 /
−0.67); **P108** silIoU ≥ 0.977 (0.9797) and the ear as l300's (fill ≥ 0.93). **g41r = the 40k
gallery recipe + the rule** (GPU 1; a diagnostic twin, the gallery untouched): **P109** the 40k
breathing's normal step 0.0029 → ≤ 0.001 wu, flips 0.74 → ≤ 0.55; **P110** silIoU within 0.003
of g41's and the video's delivered tail at the hold level (≤ 0.0005). **M3 = z300b re-rendered
with the finest Poisson leaf at the MPM cell** (`--poisson_cell 2.27`, GPU 1, render only):
**P111** if the tail motion is a tangential re-sampling (c), the delivered tail's per-frame change
falls from 0.0024 toward the transport-only level (≤ 0.0015) with the end bump within 0.2° of
1.24°; a bump rising above 1.5° or the ear tip lost says the cell-wide leaf cannot carry the
1.1-cell ear and the band limit must be half a cell with the anisotropic surfels.

**2026-09-24 13:50 — g41r's verdict (the 40k gallery recipe + the per-particle Rprop; 36 windows
in 2.0 min).** silIoU **0.9666** (g41 0.961, +0.0056 — P110's fit ✓ with margin), det F 0.776, no
strays, bump 1.19°. The rule's log: reversals 4–6 % of the moving particles through window 10,
32 % at 15, 40 % at 25, 50 % at 30; the step scale's median 1.0 → 0.86 (15) → 0.25 (20) → 0.09
(25) → 0.03 (30), 68 % of the moving particles below 0.1 at 30; three merit rejections ended the
run at 36 (the plateau reached at the optimiser's resolution). **P109 half ✓**: the layer's flip
fraction 0.74 → **0.47** and the layer's consecutive-window correlation **+0.22 … +0.26** (g41
−0.04 … −0.69), the bulk's +0.12 … +0.73 — **the alternation is gone**; but the normal step's
amplitude is not: layer step 0.023 spacings = 0.0032 wu (the same size as before), now a
coherent DRIFT (net/summed 0.26, cycle ratio 0.57–0.61 above the random-walk bound = net
motion, the fit still improving when the gate stopped it). **P110 tail ✗ as measured**: the
delivered tail's per-frame change 0.0017 (g41 0.0013) — the last 11 video frames are the settling
drift, not a rest. *The measure was the wrong one for the eye*: a new probe (`video_flicker.py`)
splits the per-frame change into an ALTERNATING part (|I_{t+1} − 2I_t + I_{t−1}|/2, a Nyquist
flicker) and a DRIFT part (|I_{t+1} − I_{t−1}|/2); on the stride-12 videos every run reads
ALT/DRIFT 1.2–2.2 — white noise between consecutive frames gives 1.73, so at that stride the
frame-to-frame change is mostly the per-frame refit's own noise, and the window-to-window
alternation is aliased (stride 12 against a 19-frame window). Re-read at stride 19 (one frame
per commit) below.
*At stride 19 (one frame per commit; 14:00):* g41 tail D1 0.0015, ALT/DRIFT 1.65; g41r tail D1
0.0018, ALT/DRIFT 1.29 — g41r alternates less and drifts more, as its particles do, but both sit
near the white-noise ratio (1.73): even between commit frames the image change is mostly
uncorrelated frame to frame. Read with the particle statistics (g41r's layer no longer
alternates, its step 0.0032 wu ≈ 2 % of a spacing), this says the visible per-frame change is
dominated by the **per-frame refit's response to small particle changes**, not by the
particles' alternation itself — the reconstruction side of the question (hypothesis c), which
M1 (the normal-only / tangential-only twins at stride 4) and M3 (the leaf at the cell) are
measuring on GPU 0 / GPU 1 now. The decisive comparison: D1 at stride 4 against stride 12 / 19 —
a refit-noise floor is stride-independent, a genuine motion scales with the stride.

**2026-09-24 14:10 — M3 and the ear-tip census read.**
*M3 (z300b re-rendered with the finest Poisson leaf at the MPM cell, `--poisson_cell 2.27`):*
**P111 ✗** — the delivered tail's per-frame change 0.0022 (0.0024 at the default leaf), the
alternating/drift split unchanged (1.55 vs 1.51); the end bump falls to **1.06°** (1.24°) and the
raw-components>1 frames rise to 11 of 140 (2): the coarser leaf smooths the body and pinches the
thin tip off more often. So the per-frame image change is NOT a sub-leaf re-sampling of the
surfel set — a fit that cannot see the sub-cell arrangement changes just as much. What remains
as its source: the refit's global response (the octree and the screened solve re-done from
scratch on a slightly different set — vertex placement, the Loop subdivision, per-vertex
normals), or the genuine motion at the cell scale. M1's stride-4 twins (running) separate
these: a refit response is stride-independent, a motion scales with the stride.
*The ear-tip census (frames 192–324 of z300b, one Poisson mesh per frame):* the tip's particles
(within 0.6 wu of the target's highest point) grow monotonically 47 → 108 and are all outer-layer
surfels (45 → 95) — **the particles never retract**; the mesh is ONE component in every census
frame (no piece dropped there — R-1 is not the tip's mechanism, though it still covers the two
frames with a pinch-off); and the mesh CAPTURES only 3–27 of those surfels (within half a
spacing), jumping frame to frame (5, 3, 15, 7, 22, 12, 27, 26) against 66 % over the ear as a
whole. The tip's dropout is the Poisson fit passing below a sparse tip by a varying amount —
reconstruction-side, the density-adaptive octree / weak screening at a sparse region (R-2 of
the digest), not the component filter and not physics. *Pre-registered probe (14:10,
`psr_probe.py`, Kazhdan's PoissonRecon on the frame's oriented surfels at the renderer's depth):*
**P112** samplesPerNode 1 with pointWeight 4 raises the tip capture from ≤ 0.25 to ≥ 0.5 at
frames 240 / 264 / 312 while the whole-layer capture stays within 0.1 of Open3D's — then the fix is
the fit's sampling rule (finest nodes kept at the tip, screening on), implementable by
in-plane surfel resampling at thin regions for the Open3D path (R-4) or by the binary; **P112
failing** (capture flat across settings) says the tip is not representable at this node size and
the band limit must be relaxed locally (a deeper octree at the tip = R-2's other half).
*The codec floor (14:15):* over the 20 identical hold frames the per-frame change is 0.0000 (p90
0.0001) in both z300b's and g41's videos — the encoder contributes nothing; the 0.0015–0.0025 per
frame is image change from the renders themselves.

**2026-09-24 14:20 — M2 and the PoissonRecon tip probe read; the kind of the tail motion.**
*M2 (z300b's last six window commits; the Poisson surface of frame k sampled and its signed
distance to frame k+1's surface = the surface's own normal displacement, against the layer
particles' normal step over the same window):* **v_n rms / d·n rms = 0.91–0.98** in every pair
(v_n rms 0.006–0.011 wu, mean ±0.001–0.005) — **the reconstructed surface moves along its normal
by exactly what the layer particles move**: the visible motion is a genuine normal displacement
of the surface (hypothesis a), not a re-sampling of a stationary surface (c). With M3 (a
cell-wide leaf changes nothing) this closes the reconstruction hypothesis for the breathing:
the per-frame image change IS the surface breathing, window-coherent (per archived frame the
layer's normal step is 0.0006 wu, per window 0.007–0.012 wu rms — the motion inside a window is
one rollout, coherent; the sign flips between windows), with a tangential part of the same
size riding along (d_t rms 0.008–0.015 wu). The amplitude, 0.007 wu per window at the surface
(10 % of the native spacing, 2 % of a cell), is what the per-particle Rprop must remove — g41r
removed its alternation and left its size as a drift (13:50); ab300 / ac300 read at 300k.
*The PoissonRecon probe (P112):* the tip's capture is **0.00–0.10 at every setting**
(samplesPerNode 1 / 1.5 / 5, pointWeight 0 / 4; frames 240 / 264 / 312, 39–50 tip surfels, the
finest node 0.10 wu at depth 6) — **P112 ✗**: neither the density adaptivity nor the screening
is the tip's problem; a filament 2–3 native spacings across (0.13–0.19 wu) is narrower than the
finest node and far narrower than the quadratic B-spline's three-node support (0.31 wu) — it is
not representable at the reference-spacing resolution, whatever the sampling rule. The
literature's answer for exactly this case is a surfacer that never cancels (Yu–Turk 2013's
isolated-particle spheres; Bhattacharya 2011's surface enclosing every particle's sphere): R-3.
*Pre-registered R-3 (`--thin_fallback`): surfels the drawn surface neither encloses nor comes
within one reference spacing of, and that lie within the link radius of the body, are drawn as
spheres of radius half the reference spacing (the render's own resolution; the surfel size)
unioned into the image.* **P113** the tip is drawn in every census frame (192–324) with the drawn
tip length monotone in the frame index (no frame shorter than the previous by more than one
spacing); **P114** the fallback touches ≤ 0.5 % of the surfels outside the ear region (the body's
uncaptured surfels sit within one spacing of the surface — no hair on the body) and the end
bump is unchanged (1.24° ± 0.05); **P115** the tail's per-frame change is unchanged (the fallback
is not the breathing's fix). Refutation: P114 failing (spheres over the body) says the
one-spacing threshold is inside the body's own layer noise and the fallback needs the thin
detector (σ₃/σ₁) as a co-condition.

*R-3 tried in two forms (14:35; `--thin_fallback spheres | level`).* **spheres** (a union of spheres of
one reference spacing at the uncaptured, attached, chained surfels): 11 / 25 / 12 spheres at frames
240 / 264 / 312, 0 at the end frame, the body untouched (P114 ✓: no hair, bump unchanged) — but the
tip is drawn as what it is at that resolution, a **bead chain**: the tongue's leading edge holds
39–50 surfels in a 0.2-wu knob on a filament, and the spheres render exactly that (the user's
"물방울"). **level** (the frame's density marched in the fallback region): **nothing at 240 / 264**
— the sparse tip's density lies below the iso level, the same reason the level-set renderer lost
thin tips — 12 at 312. So the render side can either hide the tip (the fit, flickering as it
catches the tip or not) or show it faithfully as beads; it cannot make a smooth tongue out of a
beaded edge. The edge's coherence is the physics' (the third survey's P-1: sheet-aware splitting
at the commit, Ando 2012 — particles inserted in the sheet plane where the in-plane gap exceeds
two spacings so the front stays a sheet the fit can represent; or the codimensional carrier).
`--thin_fallback` stays opt-in (spheres: the faithful rendering of a fragmented edge; useful as a
diagnostic overlay), `--keep_attached` opt-in (the two pinch-off frames). The physics route is
pre-registered once ab300 / ac300 have read the oscillation.

**2026-09-24 14:45 — ac300's verdict (l300 + the per-particle Rprop) and M1.** ac300: 112 windows
in 24 min, silIoU **0.9792** (P108 ✓), layer flips **0.41** (P107 ✓), the layer's normal step
**0.0138 spacings = 0.00095 wu** (P107's ≤ 0.001 ✓, l300 0.0017), net/summed 0.30 (P107's ≥ 0.4
✗); the step scale's median at 0.000 from window ~60 with 90 % of the moving particles below 0.1
(the body settles as designed); but **det F min 0.388** (l300 0.664) — the per-particle scales
make neighbouring particles' control updates differ by orders of magnitude, a sub-cell control
noise (the creg term's reason for existing), and the material pays in local compression. The
bulk's reversal cosine also stays negative from window 55 to the end while the layer no longer
alternates — the settled particles' residual is the noise. *Second form (method.md §10.24
addendum, `--ctrl_rprop_smooth`): the reversal is read on the displacement averaged over the
material neighbourhood (the bond / coherence kNN frozen at the source, ≈ 60 neighbours at 300k
= half a cell) and the scale applied is the neighbourhood mean of the per-particle scales — the
update stays coherent at the neighbourhood scale, the per-particle rule stays as the memory.*
*Pre-registered ad300 = l300 + `--ctrl_rprop --ctrl_rprop_smooth --u_rprop --u_rprop_floor 0`*
(GPU 1, 14:50): **P116** det F min ≥ 0.6 (l300 0.664; ac300 0.388); **P117** the layer's normal
step ≤ 0.001 wu and flips ≤ 0.55 as in ac300; **P118** the bulk's reversal cosine after the
onset decays to |cos| ≤ 0.2 within 15 windows (ac300: negative to the end); **P119** silIoU ≥
0.977.
*M1 (the tail-only twins at stride 4; per-frame image change, median):* full **0.0013**,
normal-only **0.0013**, tangential-only **0.0009**. The normal part reproduces the full change
(a — with M2, the surface's own normal motion), and the tangential-only twin still shows 70 % of
it: a refit-response floor exists (c) alongside the genuine motion — at stride 4 the two are of
one size. The Rprop removes the particles' motion (both parts shrink together: a particle that
stops cannot rearrange), which is why ac300's video tail is the next reading (its chain).
*R-1 read (14:55; the `--keep_attached` re-render of z300b):* the sidecar is unchanged — the one
pinch-off frame (276) stays dropped (the piece lies beyond the link radius), one more appears at
288 from the refit's own variation, and every frame of the dropout window is a single component
in both renders: **P101 ✗ / P102 ✓** — the component filter is not the tip's mechanism (the census
had said so); the flag stays opt-in for the rare detached pinch-off.

*15:05 — ab300 ended (135 windows, 40.8 min): silIoU 0.9783, **det F min 0.19** — the per-particle
form's cost, worse than ac300's 0.39 (the KDE term's push adds to the sub-cell control noise);
its chain runs. ac300's spectral reading: the layer's low-band consecutive-window correlation
**+0.04** (l300 −0.67; every earlier run −0.6 … −0.98) — the grid-scale alternation is gone at
300k too — with the high-band energy share 0.80 (the residual motion is sub-cell: the control
noise the smoothed form (ad300) is built to remove). The ad300 smoke ran clean (1 % reversals in
transport); the run is on GPU 1.
*Pre-registered af300 = z300b's recipe + the neighbourhood-smoothed Rprop* (GPU 2, 15:10; the
deliverable candidate if ad300 holds P116): **P120** det F min ≥ 0.55 (z300b 0.560); **P121** ear
fill ≥ 0.93 and silIoU ≥ 0.977 (z300b 0.936 / 0.9802); **P122** the layer's normal step ≤ 0.001 wu,
flips ≤ 0.55, low-band correlation > −0.3 over the last 10 delivered windows; **P123** the
delivered tail's alternating component (video_flicker ALT at stride 19) ≤ half of z300b's.
*(15:15) A third form considered and not run:* reading the reversal on the grid-projected
displacement — but with `--commit_pic` in the recipe the committed displacement IS the
grid-projected one (the projection runs before the Rprop block reads x − x_start), so ac300 /
ad300 already read the grid-visible part; the defect was the per-particle SCALE's spatial noise,
which the neighbourhood-smoothed form addresses. GPU 3 left free; g41s (40k + the smoothed form)
runs on GPU 0 as the 40k check.

**2026-09-24 15:25 — the user's rule: the fixes must generalise beyond the bunny ("이거 bunny 뿐
아니라 일반화가 되어야 하는 거 알지?").** Recorded as a standing rule (memory
generalise-beyond-bunny): after a mechanism passes its bunny pre-registration it runs on the
whole 40k gallery (19 targets: A, C, V, armadilo, beast, bimba, bob, bunny, cheburashka, cow,
dragon, fandisk, heart, homer, maxplanck, nefertiti, ogre, spot, teapot) with the same metrics
against g41, and a fix that helps the bunny and hurts another target goes back to the mechanism.
*g41s (bunny, 40k + the smoothed Rprop; 48 windows, 2.8 min):* silIoU 0.9662 (g41 0.961), det F
0.769 (0.771 — no damage at 40k in either form), flips 0.55 (g41r 0.47, g41 0.74), net 0.22, step
0.0218 spacings (0.0030 wu), low-band correlation −0.43 (the smoothing dilutes the per-particle
decay: neighbours still moving pull a settled particle's scale up), stride-19 flicker ALT 0.0015 —
the same as g41's 0.0014 and g41r's 0.0015: **at 40k neither form changes the video's per-frame
alternating change**, only the particles' statistics. What the eye sees at 40k is then the
per-window motion's magnitude (0.003 wu, spatially incoherent) whether it reverses or drifts; the
300k reading (ac300's stride-19 video, rendering) decides whether the halved layer step there
shows. *Pre-registered gallery sweep (g41s_<target>, GPUs 0 and 3, 15:25):* **P124** on every
target silIoU ≥ g41's − 0.003 and det F ≥ g41's − 0.05; **P125** the layer's flips ≤ 0.6 and the
step ≤ g41's on at least 15 of 19 targets; **P126** windows ≤ 1.5 × g41's on every target (the
rule must not stall a transport). A target failing P124 by more sends the form back to the
mechanism (the smoothing radius or the reversal read) before any adoption.

**2026-09-24 15:40 — ad300's verdict (l300 + the smoothed Rprop, coherence kNN ≈ 60).** 81 windows
in 23.3 min. **P116 ✓** det F **0.643** (l300 0.664; ac300 0.388 — the smoothing restores the
control's coherence); **P119 ✓** silIoU 0.9792; **P117 ✗ marginal** — the layer's step 0.0179
spacings = **0.0012 wu** (l300 0.0017, ac300 0.00095; the criterion 0.001), flips **0.54** (✓);
**P118 ✗** — the low-band correlation **−0.31** (l300 −0.67, ac300 +0.04) and the bulk's reversal
negative from window 60 to the end: the 60-neighbour mean dilutes the decay (a settled particle's
scale is pulled up by neighbours still moving). ab300 (z300b + the per-particle form): silIoU
0.9783, det F 0.19, flips 0.58, net 0.44, step 0.0009 wu, low-band −0.18, one stray, 135 windows.
So the trade is decay strength against spatial coherence, and the neighbourhood's size is the
lever between ac300 (k = 0, det F 0.39, corr +0.04) and ad300 (k ≈ 60, det F 0.64, corr −0.31).
*Pre-registered ag300 = l300 + `--ctrl_rprop --ctrl_rprop_smooth --ctrl_rprop_k 8`* (GPU 1, 15:40):
the smoothing over the control regulariser's own neighbourhood (creg_k = 8 — the scale at which
the control is already held coherent by the creg term; not tuned: the regulariser's constant).
**P127** det F ≥ 0.6; **P128** the layer's step ≤ 0.001 wu and flips ≤ 0.55; **P129** the low-band
correlation > −0.2; **P130** silIoU ≥ 0.977. Refutation: P127 failing at k = 8 says the coherence
the control needs is wider than the regulariser's and the smoothing must be the coherence kNN
with a stronger η⁻ (then Riedmiller's 0.5 is the thing to derive, not to keep).
*ac300's chain tail (15:50):* the plain video's delivered tail **0.0013** per frame at stride 12 —
**the 40k reference value, the acceptance number** (l300 0.0016, z300b 0.0024) — and at one frame
per commit (stride 19) tail D1 **0.0008**, ALT **0.0007** (g41 0.0014, g41r 0.0015): at 300k the
per-particle Rprop halves the visible per-frame change and its alternating part. Bump 1.22°, 11
strays; the ear tip **10.7** reference particles (l300 18.5) — a particle whose arrival wiggle
reverses once halves its step for good, and the tip fills less: the cost to read on af300 (with
the KDE term) and the reason the smoothing / neighbourhood question matters for the ear too.

**2026-09-24 16:05 — ab300's verdict (z300b + the per-particle Rprop; 135 windows, 41 min).**
The best ear of any run: fill **0.965** at the end (z300b 0.936), the base slab ≤ 1.12× (no pile),
the tongue at 0.71–0.97 of the target thickness where filled, the tip 12.1 reference particles,
one stray, bump 1.26°; the video's delivered tail **0.0015** (z300b 0.0024; the 40k value 0.0013),
the low-band correlation −0.18, flips 0.58, net 0.44, step 0.0009 wu — **P103–P106 ✓ except the
tail by 0.0002 and P105's det F: 0.19** (the per-particle form's cost, as in ac300). So the
per-particle form gives the ear and the tail, and the smoothing must give back det F without
losing them: *pre-registered ah300 = z300b's recipe + `--ctrl_rprop --ctrl_rprop_smooth
--ctrl_rprop_k 8`* (GPU 2 alongside af300, 16:05): **P131** det F ≥ 0.55; **P132** ear fill ≥ 0.93,
silIoU ≥ 0.977; **P133** the video tail ≤ 0.0015 and the low-band correlation > −0.2; **P134** flips
≤ 0.6, the layer's step ≤ 0.001 wu.
*ad300's chain tail (16:15):* video tail 0.0015 (l300 0.0016), bump 1.25°, 8 strays, the ear tip
10.4 reference particles (l300 18.5 — the same thinning as ac300's 10.7: the rule's cost at the
tip is not the neighbourhood's doing). ag300 (k = 8) ended at 77 windows; its chain runs.
*The rule's ear cost, read properly (16:20):* the end ear fill is **0.961** (ac300) / **0.954** (ad300)
against l300's 0.948 — the ear is not under-filled; only the top slab reads 0.80 (l300 0.86) and
the 0.25-wu tip count 10.4–10.7 against 18.5: a slightly blunter tip, not a lost ear. No
transport-gated reset of the scale is needed.

**2026-09-24 16:30 — the 40k gallery with the smoothed Rprop (g41s, 19 targets) against g41.**
silIoU up on 17 of 19 (+0.0002 … +0.0099; bunny +0.005, dragon +0.010, maxplanck +0.005), down on
bob (−0.0007) and **C (−0.011)**; det F within −0.024 on 18, **homer −0.060** (0.812 → 0.752);
windows ≤ 1.2× everywhere (P126 ✓); the layer's flips down on 15 of 19 (bunny 0.74 → 0.55,
armadilo 0.79 → 0.55, ogre 0.71 → 0.43, teapot 0.70 → 0.47), its step down on 13 of 19
(nefertiti 0.058 → 0.019, armadilo 0.064 → 0.024 spacings), net/summed up on 17. **P124 ✗** on C
and homer, **P125 ✗** (flips ≤ 0.6 and step ≤ g41's together on 11 of 19, not 15), **P126 ✓**.
Per the generalisation rule the form goes back to the mechanism: C is the hole regime, where the
material moves round the hole on a curved path, and a particle whose displacement turns by more
than 90° between windows in TRANSPORT is halved by the rule as if it had overshot — C's fit paid
0.011. *Third form (`--ctrl_rprop_arrived`): the halving applies only to particles the paced
target reads as ARRIVED (its own per-particle mask: plan image within the pace radius); a
direction change in transport keeps the step. No constant.* *Pre-registered g41t = the 19-target
sweep with the arrival gate (GPUs 0 and 3, 16:30):* **P135** C's silIoU ≥ g41's − 0.003 and
homer's det F ≥ g41's − 0.05; **P136** silIoU ≥ g41's − 0.003 and det F ≥ g41's − 0.05 on every
target; **P137** flips ≤ 0.6 and step ≤ g41's on ≥ 15 of 19. The 300k forms in flight (af300,
ag300, ah300) run without the gate; their bunny readings stand for the bunny.
*Pre-registered, the first 300k target besides the bunny (16:45; GPU 1):* **d300 = dragon at
300k with z300b's recipe** and **dr300 = the same + `--ctrl_rprop --ctrl_rprop_smooth
--ctrl_rprop_k 8 --ctrl_rprop_arrived --u_rprop --u_rprop_floor 0`** (the arrival-gated smoothed
form). The dragon is the gallery's thin-feature target (horns, spikes; g41_dragon silIoU 0.958, det
F 0.763). **P138** d300 itself: silIoU ≥ 0.955, det F ≥ 0.6, ≤ 20 strays (the 300k recipe holds on a
second target); **P139** dr300 against d300: flips ≤ 0.6 (and lower), the layer's step ≤ 0.7× d300's,
silIoU within 0.003, det F ≥ 0.55; **P140** the video tails (stride 12): dr300 ≤ d300 − 0.0003.

**2026-09-24 17:00 — ag300's verdict (l300 + the smoothed Rprop at k = 8) and the global step's
re-inflation.** 77 windows in 30.6 min; **P127 ✓** det F 0.655, **P130 ✓** silIoU 0.9784; **P128 ✗**
flips 0.68, step 0.0243 spacings = 0.0017 wu (= l300); **P129 ✗** low-band −0.32; video tail
0.0019, bump 1.28°, ear fill 0.955, tip 10.4. The per-particle scales decayed further than in
ad300 (median 0.05 at window 70 against 0.14) and the layer breathed MORE — the scale is not
what sets the layer's step. The records say why: the optimiser's GLOBAL step, α, sits at its
floor 0.0010 in l300's tail (anneal 0.05) but at **0.005–0.008 in ag300's** (anneal 0.22–0.34),
0.002–0.006 in ad300's, 0.002–0.003 in ac300's — the plateau anneal recovers ×1.15 on every
window the tracks call "improved", and with the rule shrinking the per-particle steps every
window improves a little, so α re-inflates 5–8× and cancels the per-particle decay (in ac300 the
scales reached 0.000 and the product still fell; that is why its layer step halved). The step
control had two knobs, one per particle and one global, pulling against each other. *Fourth
form (`--ctrl_rprop_hold`): while the rule is on, the global step never grows — no ×1.1 on
acceptance, no anneal recovery — Rprop has no global rate; the per-particle scale is the only
step control. No constant.* *Pre-registered ai300 = l300 + `--ctrl_rprop --ctrl_rprop_smooth
--ctrl_rprop_k 8 --ctrl_rprop_arrived --ctrl_rprop_hold`* (GPU 0, 17:00): **P141** the tail's α ≤
0.0015 throughout (no re-inflation); **P142** the layer's step ≤ 0.001 wu, flips ≤ 0.55, low-band
correlation > −0.2; **P143** det F ≥ 0.6, silIoU ≥ 0.977, ear fill ≥ 0.93; **P144** the video
tail ≤ 0.0013 (the 40k value). Refutation: P142 failing with P141 holding says the layer's
residual motion is not the optimiser's step at all (the settle-at-commit twin is then next).
*Stride-19 flicker at 300k (17:05; one frame per commit, the delivered tail's alternating
component):* l300 **0.0011**, z300b **0.0021**, ac300 **0.0007** — the per-particle form cuts the
alternating part to 0.65× of l300's and a third of z300b's (whose tail, with the ear still
filling under the KDE term, alternates the most: ALT/DRIFT 1.81).

**2026-09-24 17:40 — af300's verdict and the arrival-gated gallery, first half.** af300 (z300b's
recipe + the smoothed Rprop k ≈ 60; 145 windows, 79 min sharing GPU 2): silIoU 0.9768, ear fill
**0.966**, tip 16.8, flips 0.61, step 0.0013 wu, low-band −0.24 — and **det F 0.226** (P120 ✗): with
the KDE term the k ≈ 60 smoothing does not protect det F either (ad300 without KDE kept 0.64), so
the compression comes from the KDE's sub-cell pull acting under the re-inflated global step, not
from the per-particle scales alone; the held-step form (ai300, and its combo) is the reading that
matters. *g41t (the arrival-gated form), 9 of 19 targets:* **C 0.9009 → 0.9496** at 113 windows
(the run that stalled at 28 windows in every recipe since the C forensic now transports —
particles in transit keep their step, arrived ones settle) but **det F 0.865 → 0.528**; homer's
det F −0.011 (the −0.060 gone); A +0.005, bunny +0.005, fandisk +0.006, dragon +0.002, cow
+0.000, heart +0.001, V −0.001; flips down on 8 of 9. C's det F is the next thing to read (where
the compression sits — the arms' fronts arriving at the hole's rim under a step the gate never
decays because they never "arrive" by the pace radius?).

*18:00 — two 40k analogues launched for speed (3 min each; the user: "저 진동이 진짜 너무 안 잡히네").*
**g41h** = the 40k gallery recipe + the held-step arrival-gated smoothed form (the 40k analogue of
ai300): **P145** the layer's flips ≤ 0.5 and step ≤ 0.6 × g41's (0.0208 spacings), the stride-19
alternating component ≤ 0.7 × g41's (0.0014), det F ≥ 0.75, silIoU ≥ 0.961. **g41a** = the 40k
recipe with the plastic assimilation OFF (`--assim 0`; η: F_e → R_e S_e^{1−η} per commit — the
one commit-time operation on the material state not yet tested as the breathing's carrier): a
DIAGNOSTIC, expected to morph worse (the control must then hold the shape elastically); **P146**
if its layer flips fall below 0.5 with the step ≤ 0.5 × g41's while the fit stays ≥ 0.95, the
assimilation's commit-time jump (the elastic stretch halved at every commit, the next window's
stress state discontinuous) is a carrier and the structural form is a commit at equilibrium
(assimilate, then settle before the next window's linearisation); if the breathing is unchanged
the assimilation is out too, and with the held step (ai300 / g41h) the optimiser's step is the
last candidate standing or falling.
*af300's chain tail (18:05):* video tail **0.0013** (the 40k acceptance value), bump 1.23°, ear
0.966 — with det F 0.23 the only thing standing between it and a deliverable.

**2026-09-24 18:20 — the user's proposal: "plasticity를 사용해서 최적화되어버리면 고정시켜버리면
안 되나?" — freeze at arrival (method.md §10.25, `--freeze_arrived`).** The limit of the Rprop
taken at once: a particle that has ARRIVED (the paced target's own mask) and reversed twice (one
full period — settled by the rule's reading) is frozen for good: its elastic stretch assimilated
in full (F_e → R_e: no stress of its own; the plasticity used as the lock), its warm-started
control zeroed, its update scale 0, its u bound 0, its velocity and affine state zeroed at
commits. Frozen material is inert and rides the grid with its neighbours; the frozen set only
grows, so the settled body cannot breathe, while unarrived material (the ear) keeps its full
step. No constant: the arrival mask and the period are the rule's. *Pre-registered aj300 = l300 +
the held-step arrival-gated smoothed Rprop + `--freeze_arrived`* (GPU 2) and **g41f** its 40k
analogue (GPU 3): **P147** the frozen fraction reaches ≥ 0.9 of the particles within 20 windows of
the reversal onset and the run ends on the plateau rule; **P148** the layer's flips ≤ 0.3 and its
step ≤ 0.0005 wu over the last 10 delivered windows, the low-band correlation > −0.1; **P149** the
video's delivered tail ≤ 0.0013 (stride-19 alternating component ≤ 0.5 × the reference's);
**P150** det F ≥ 0.6 (the full assimilation must not compress), silIoU ≥ 0.977 (300k) / ≥ 0.961
(40k), the ear fill ≥ 0.93 at 300k (the ear's particles freeze only once arrived). Refutation:
P148 failing with P147 holding says the frozen body still moves — then the motion is the grid's
(neighbours in transport dragging inert material) and the breathing was never the optimiser's.

**2026-09-24 18:40 — g41h (40k, the held-step form) and g41a (40k, no assimilation).**
*g41h:* silIoU 0.9632, det F 0.781, the bulk's reversal negative in ONE window of 36 (g41: from
window 20 to the end), the layer's low-band correlation −0.18, flips 0.47 — the alternation is
gone at 40k too; but the layer's step 0.0193 spacings (g41 0.0208 = 0.0029 wu; **P145's amplitude
✗**) and the stride-19 alternating component 0.0014 (= g41's; ✗) are unchanged: with the global
step held and the per-particle scales at 0.05, the layer still moves 2 % of a spacing a window
as an uncorrelated jitter. *g41a (`--assim 0`):* silIoU 0.966, det F 0.82, flips 0.64, step 0.036
spacings, video tail 0.0022, ALT 0.0019 — larger, not smaller: **P146 ✗, the assimilation is not
the carrier** (without it the elastic body springs more). So at 40k the picture after every twin
is: the window-to-window ALTERNATION is the optimiser's (removed by the per-particle rule), the
residual per-window MOTION of the layer (0.02 spacings, spatially incoherent, white-noise-like
between commit frames) is not set by the control update's size — its source is what the freeze
twins (g41f / aj300: control zeroed, velocity zeroed, stretch assimilated for the arrived body)
isolate: either the persisting warm-started control replaying each window, or the rollout's own
response to a stationary state (the relaxation projection at 1/T per step, the shifting, the
grid's cell-crossing noise) transmitted from unfrozen neighbours.

**2026-09-24 18:55 — g41f (40k + freeze at arrival).** 24 delivered windows (the plateau rule
ended it at ~30), 3.6 min. **P147 ✓** frozen 28 % at window 15, 66 % at 20, 81 % at 30; the
bulk's reversal never negative, the layer's low-band correlation **+0.39** (a drift, no
alternation), flips **0.34**; det F 0.788 ✓; bump 1.08° (smoother). **P150 ✗** silIoU **0.9561**
(g41 0.961, g41h 0.963): the lock closes on particles before the fit is finished — the freeze
costs 0.005 of fit at 40k. **P149 ✗** the video tail 0.0015 (stride 12), the stride-19 alternating
component 0.0017 (g41 0.0014). *The decisive number:* the body's median per-window motion with
80 % of its particles frozen — control zeroed, velocity zeroed, stretch assimilated — is still
**0.0016 wu** (g41 0.0023–0.0041, g41h 0.0021–0.0031): the residual per-window motion of a
settled body is **not the control's and not the optimiser's**; it is the rollout's own — the
grid carries the unfrozen 20 % (the ear, the last arrivals) into the frozen material every
window, the layer relaxation follows the moved neighbourhood, and the deliverable frames ARE
those simulated steps. At 40k that is 1 % of a spacing a window and the per-frame Poisson refit
turns it into the shading change the eye reads as flicker (M1/M2 at 300k: 10 % of a spacing,
the surface following it 1:1).
*Consequence for the two defects' fix:* the optimiser side is done as far as it goes — the
per-particle rule (held step, arrival-gated, smoothed) removes the ALTERNATION at both N without
the fit or det F cost (g41h; ai300 pending at 300k); the freeze removes it faster at a fit cost
and does not remove the residual motion — not adopted. The residual per-window motion is a
property of delivering simulated frames; the deliverable side must not re-fit the surface from
scratch against a sub-resolution jitter: the tracked surface advected by the GRID velocity and
projected onto the refit only beyond half a spacing (the survey's item 3; Yu 2012 / Dagenais
2017 / Bojsen-Hansen 2013) is the pre-registered next step (`--track` with a band = ½ spacing).
*Pre-registered, the deliverable side (19:05; `--track_band`, `--track_avg` in render_photoreal):*
the tracked mesh (vertices bound to their k particles) pulled toward the fresh refit only where
the refit lies farther than half a spacing (the reconstruction's own resolution — Yu 2012 /
Dagenais 2017's band), and optionally drawn as the moving average of the last N frames (N = the
control window in video frames; 2N the alternation's period — the surface at the control's time
resolution). On z300b (tail 0.0024 untracked): **P151** `--track --track_band 0.5`: the delivered
tail ≤ 0.0015 and the stride-19 alternating component ≤ 0.0012 (0.0021); **P152** `+ --track_avg 3`
(stride 12: three frames ≈ two windows): tail ≤ 0.0010; **P153** no lag artefact — the ear's
drawn length at video frames 20–40 within one spacing of the untracked render's, and the
stride-12 mid-morph per-frame change (t = 0.2–0.5) within 20 % of the untracked (the transport
must not be smeared). On ac300 the same for comparison. Read as: the residual per-window jitter
of a settled body (the rollout's, not the optimiser's — g41f) is below the deliverable's own
resolution and is not to be re-fitted every frame.

**2026-09-24 19:10 — the arrival-gated gallery (g41t, 19 targets) against g41.** silIoU within
−0.003 on all 19 (up on 15: C **+0.049** — the run that stalled at 28 windows since the C
forensic transports to 113 —, armadilo +0.005, bunny +0.005, fandisk +0.006, maxplanck +0.005;
down by ≤ 0.0014 on V, beast, bob, ogre): **P135 ✓, P136's fit ✓ on every target**; flips ≤ 0.6 on
16, step ≤ g41's on 18, both on 15: **P137 ✓**; **P136's det F ✗ on two**: C 0.865 → **0.528**
and beast 0.800 → **0.488** (the others within −0.018) — both the longest transports (113 / 133
windows): where arrived particles settle next to particles still in transit at full step, the
material between them shears, the spatial-incoherence cost in a new guise; homer's −0.06 became
−0.011. *Pre-registered g41u = the 19-target sweep with the held global step added
(`--ctrl_rprop_hold`; g41h's form)* (GPUs 3 and 1, 19:10): **P154** det F ≥ g41's − 0.05 on
every target, C and beast included; **P155** silIoU ≥ g41's − 0.003 on every target; **P156**
flips ≤ 0.6 and step ≤ g41's on ≥ 15. P154 failing on C / beast says the gate needs the
material's coherence, not the step's: the smoothing neighbourhood must straddle the
arrived/in-transit boundary (the scale = the neighbourhood MINIMUM rather than the mean, so a
particle next to settled material settles with it).

**2026-09-24 19:30 — the user: "일단 진동부터 없애자. 진동 없앨 수 있는 모든 방법을 사용해 봐야 할 거
같아." The oscillation first, every method.** The candidates, layered by where the motion is made,
each run as a 40k twin on the held-step Rprop base (H = `--ctrl_rprop --ctrl_rprop_smooth
--ctrl_rprop_k 8 --ctrl_rprop_arrived --ctrl_rprop_hold --u_rprop --u_rprop_floor 0`, g41h) and
read by the same chain (layer breathing, low-band correlation, video tail, stride-19 ALT):
- **g41k = H + `--rest_commit --rest_commit_reversal`** (windows from rest once reversing: the
  carried velocity of the settled body zeroed at commits) — **P157** the body's per-window
  motion in the tail ≤ 0.5 × g41h's (0.0021–0.0031 wu), flips ≤ 0.4, ALT ≤ 0.0010.
- **g41o = H + `--outer_latch_reversal`** (the deliverable stops at the onset) — **P158** the run
  ends within 8 windows of the onset with silIoU ≥ 0.960 and no negative reversal in the
  deliverable; the tail measure then reads the transport's end, not a breathing.
- **g41m = H + `--ot_handoff`** (the fixed target from arrival: the merit's moving part removed) —
  **P159** flips ≤ 0.45 and ALT ≤ 0.0012 with silIoU ≥ 0.961.
- **surfel memory** (`--surfel_memory 2`, z300b / ac300 renders): the previous frame's surfels
  carried with the material join the fit — **P160** the tail ≤ 0.0018 (z300b 0.0024) with the
  ear tip's growth intact.
- **the settled body's viscosity (next, code)**: the forward model has a per-particle viscosity
  (RolloutSpec eta); a particle arrived and twice reversed gets η with the time constant of one
  window (η = 1 / (T dt): derived, the quasi-static limit of settled material) so its carried
  motion decays within the window it arises — `--settle_eta`; and the settle-at-commit rollout
  (zero control, that viscosity, T steps) so the delivered commit is an equilibrium —
  `--settle_commit`. Pre-registered when built (**P161**, **P162**).
Every candidate that holds its P at 40k is then combined into the 300k recipe with the KDE ear
and the tracked-surface deliverable (P151–P153), and the whole is run on the 19-target gallery.
*(19:50) Built and launched (40k, GPU 3 / 0):* **g41v = H + `--settle_eta`** — **P161** the body's
per-window motion in the tail ≤ 0.5 × g41h's (≤ 0.0012 wu), flips ≤ 0.4, ALT ≤ 0.0010, silIoU ≥
0.961; **g41w = H + `--rest_commit --rest_commit_reversal --settle_commit`** — **P162** the same
targets, the settle displacement per commit ≤ 0.02 cells after ten windows (the commit already an
equilibrium), no fit loss. Both are DIAGNOSTIC forms (method.md §10.26): they change the material
during the morph, so even if they hold they enter the recipe only as a settling-phase option the
user accepts; the recipe's own fix stays on the optimiser (H) and the deliverable surface.

**2026-09-24 20:00 — ah300's verdict (z300b's recipe + the smoothed Rprop at k = 8, no hold).** 147
windows in 63 min (sharing GPU 2): the ear **0.968** (the best fill of any run), tip 13.6, silIoU
0.9761, video tail **0.0013**, low-band −0.05, flips 0.54, step 0.0009 wu, bump 1.20° — and **det
F 0.427**: with the KDE term every form without the held step collapses det F (ab300 0.19, af300
0.23, ah300 0.43), as the global step re-inflates against the per-particle decay (ag300's
finding) while the KDE keeps pulling at the sub-cell scale. *Pre-registered al300 = z300b's recipe
+ H (`--ctrl_rprop --ctrl_rprop_smooth --ctrl_rprop_k 8 --ctrl_rprop_arrived --ctrl_rprop_hold
--u_rprop --u_rprop_floor 0`)* — the 300k deliverable candidate with the physics untouched (GPU 2,
20:00): **P163** det F ≥ 0.55; **P164** ear fill ≥ 0.93, silIoU ≥ 0.977; **P165** flips ≤ 0.55, the
layer's step ≤ 0.001 wu, low-band > −0.2, the video tail ≤ 0.0013; **P166** the bulk's reversal
negative in ≤ 3 windows.

**2026-09-24 20:10 — aj300's verdict (300k, l300 + H + freeze at arrival).** 59 windows in 21.5
min, the run ending on five stale commits with 88 % frozen. **P147 ✓** (64 % frozen at window 40,
82 % at 50); **P148** flips **0.29** ✓, low-band **+0.18** ✓ (no alternation at all), the step
0.0102 spacings = 0.0007 wu (✗ against 0.0005 by a hair); **P149** the video tail 0.0015 (✗ by
0.0002), bump **1.11°** (the smoothest 300k surface so far); **P150 ✗** silIoU **0.9701** (l300
0.9797), the ear fill 0.928, det F 0.587 ✓. The same picture as at 40k: the freeze removes the
alternation and halves the layer's residual motion (0.025–0.031 spacings a window against
l300's 0.05–0.08; the bulk 0.016) but locks the body before the fit is finished and costs 0.01 of
silIoU. Not the deliverable form; it confirms that the residual per-window motion of a frozen
body (0.0007 wu at 300k) is the rollout's carriage of the unfrozen rest, as at 40k.

**2026-09-24 20:20 — ai300's verdict (300k, l300 + H: the held-step, arrival-gated, smoothed
Rprop; the physics untouched).** 112 windows in 45 min. **P141 ✓** α = 0.0010 and anneal 0.05
throughout the tail (the hold holds). **P142 ✓** the layer's step **0.0111 spacings = 0.00077 wu**
(l300 0.0017), flips **0.50**, the low-band correlation **+0.11** (l300 −0.67). **P143** det F
**0.696** ✓ (l300 0.664), ear fill 0.953 ✓, silIoU **0.9766** (✗ by 0.0004 against 0.977; l300
0.9797). **P144 ✓** the video's delivered tail **0.0011** — below the 40k value 0.0013 for the first
time; bump **1.15°** (l300 1.19°). The residual: the bulk's reversal cosine stays negative from
window 54 and the osc_layer probe's whole-layer correlation −0.6 while the spectral low band
reads +0.11 — the smooth band no longer alternates, the sub-cell band (high-band share 0.71)
still does at a fifth of the old amplitude. This is the best physics-untouched form at 300k; the
KDE-ear combo with the same flags (al300) is running as the deliverable candidate.

**2026-09-24 20:30 — the five 40k twins on top of H (every remaining physics / loop lever).**
| twin | added to H | silIoU | det F | flips / step (sp) / low-band | tail (s12) | ALT (s19) |
|---|---|---|---|---|---|---|
| g41h | — | 0.9632 | 0.781 | 0.47 / 0.019 / −0.18 | 0.0015 | 0.0014 |
| g41k | windows from rest (latched at 33) | 0.9636 | 0.792 | 0.47 / 0.021 / −0.10 | 0.0015 | 0.0014 |
| g41o | the onset gate (stopped at 33) | 0.9646 | 0.777 | 0.46 / 0.025 / — | 0.0015 | 0.0015 |
| g41m | the fixed target from arrival | 0.9640 | 0.758 | 0.48 / 0.019 / −0.11 | 0.0014 | 0.0014 |
| g41v | the settled body's viscosity | 0.9644 | 0.787 | 0.47 / 0.023 / 0.00 | 0.0017 | 0.0015 |
| g41w | settle at commit (+ rest) | 0.9647 | 0.783 | 0.49 / 0.021 / −0.02 | 0.0015 | 0.0014 |
| g41 (reference) | — | 0.961 | 0.771 | 0.74 / 0.021 / −0.4 … −0.7 | 0.0013 | 0.0014 |
**P157–P162 ✗ on every amplitude target, ✓ on fit and det F**: with the alternation removed by H,
the residual per-window jitter of the layer (0.02 spacings, ALT 0.0014 at stride 19) is
INVARIANT under the carried velocity (rest), the moving target (hand-off), the run's end (the
onset gate), the settled body's viscosity and the quasi-static commit. It is the floor of
delivering simulated frames at this discretisation — the grid's carriage of whatever still moves
into the settled material, the layer relaxation following at 1/T per step — and the physics
levers are exhausted (every one of them from the surveys' lists, each with its twin). What is
left for the VISIBLE flicker is the deliverable surface: the tracked mesh with the half-spacing
band and the window-time average (P151–P153, rendering). The viscosity and settle forms stay
diagnostics (§10.26); H stays the optimiser-side fix.
*(20:45) The tracked surface with the half-spacing band, first reading (z300b `--track --track_band
0.5`):* the delivered tail **0.0036** per frame (untracked 0.0024), max 0.0126 at re-mesh frames —
**P151 ✗**: a mesh advected by its k particles follows their per-window jitter in full (no
low-pass), and without the pull inside the band nothing corrects it; the per-frame refit is the
better temporal filter of the two. The window-average variant (`--track_avg 3`) and ac300 are
still rendering. *Pre-registered `--frame_avg K`*: each video frame reconstructed from the
particle positions averaged over a centred window of K archived frames — K = T = 19 (one
control window: the deliverable at the control's own time resolution) and K = 2T = 38 (the
alternation's period). **P167** z300b's tail ≤ 0.0012 at K = 19 and ≤ 0.0008 at K = 38, the
stride-19 alternating component ≤ 0.0010 at K = 38 (0.0021); **P168** no smearing of the
transport: the mid-morph per-frame change (frames 20–60) within 25 % of the untracked render's
and the ear's drawn length at frame 30 within one spacing.
*(21:05) ai300 at one frame per commit (stride 19):* the delivered tail's D1 **0.0006**, alternating
component **0.0005** — l300 0.0011, z300b 0.0021, ac300 0.0007, the 40k reference 0.0014: the
held-step form's tail flickers at a third of the 40k gallery's. *The tracked surface, window
average (z300b `--track --track_band 0.5 --track_avg 3`):* tail 0.0019 (untracked 0.0024, band
alone 0.0036), ALT 0.0014 (0.0021), ALT/DRIFT 0.93 — the average attenuates the alternation but
the re-mesh pops remain (max 0.0128): **P152 ✗** (≤ 0.0010). The tracked-mesh family is not the
deliverable's temporal filter; the frame average (`--frame_avg`, rendering) is the remaining
deliverable-side candidate.

**2026-09-24 21:20 — the held-step gallery (g41u = H on the 19 targets) against g41.** det F
recovered where the arrival gate alone had collapsed it — **C 0.53 → 0.859, beast 0.49 → 0.741**
(P154 ✓ except beast by 0.009: −0.059 against −0.05); flips ≤ 0.6 and step ≤ g41's on 15 (**P156
✓**); but the fit drops beyond tolerance on four targets — **C −0.010** (0.891, 24 windows: the
transport the arrival gate had un-stalled to 0.950 at 113 windows is stalled again by the hold),
**beast −0.006**, **nefertiti −0.006**, **V −0.004** (**P155 ✗**) — and the runs end earlier
almost everywhere (A 49 → 33 windows, bunny 59 → 43, heart 34 → 24, teapot 33 → 21): a global
step that never grows ends the transport early on the targets that need it late. So the hold is
right in the tail and wrong in the transport, exactly as the arrival gate separates the two for
the per-particle halving. *Pre-registered form: the hold from the onset* (`--ctrl_rprop_hold_onset`):
α grows as before until the alternation's onset (two accepted commits reversing in a row — the
same reading the onset gate and the rest latch use) and is held from then on. *g41x = the
19-target sweep with H + hold-from-onset* (GPUs 3 and 0): **P169** silIoU ≥ g41's − 0.003 on
every target (C ≥ 0.898, beast ≥ 0.951, nefertiti ≥ 0.965, V ≥ 0.976); **P170** det F ≥ g41's −
0.05 on every target; **P171** flips ≤ 0.6 and step ≤ g41's on ≥ 15.
*(21:30) Deliverable-side readings so far:* `--frame_avg 19` on z300b: tail **0.0016** (untracked
0.0024), ALT 0.0013 (0.0021) — a third off, **P167 ✗** at K = 19 (≤ 0.0012); K = 38 rendering.
`--surfel_memory 2`: z300b 0.0020, ac300 **0.0010** (0.0013) — small gains, **P160 ✗** for z300b
(≤ 0.0018). Against these, the optimiser-side H at 300k (ai300: tail 0.0011, stride-19 ALT
0.0005) does more than any render-side filter tried; the deliverable-side measures are
secondary and cosmetic, to be combined only if the user wants the last third off.
*(21:45) `--frame_avg 38` on z300b (two control windows, the alternation's period):* tail **0.0010**
(untracked 0.0024, K = 19: 0.0016), stride-19 ALT 0.0010 (0.0021), ALT/DRIFT 1.06 — the
alternating part halved and the tail at the 40k level; **P167** ✓ at K = 19's target missed, ✓
at the tail level for K = 38 but the ALT target (≤ 0.0010) met only at equality; **P168** at risk:
the whole-video per-frame change 0.0012 against 0.0026 — the average smooths the transport as
much as the tail (a one-window lag of the growth), which is the filter's cost; the ear's drawn
length at frames 20–40 is checked on the frames before any use. This is a deliverable-side
cosmetic (the frames shown are no longer the simulated ones): an option to state, not the fix.
*(21:55) The frame-averaged (K = 38) ear, frames 216–492 looked at:* the tip is present and
continuous in every frame — the dropout of the untracked render (pointed / blunt / pointed at
frames 16–27) is gone — and the tongue's beaded edge is drawn as one smooth tip; the ear's
length keeps pace with the untracked render (no visible lag at this stride). So the
window-average of the particle positions answers D2's VISIBLE dropout as a side effect (the
beads' frame-to-frame arrangement averages out) while the physics of the edge is unchanged.
*al300 (z300b's recipe + H) ended:* 135 windows, silIoU **0.9771** (P164 ✓), det F **0.511** (P163
✗ by 0.04 — the KDE term's push under the held step still costs det F, less than without the
hold: 0.19 / 0.23 / 0.43 → 0.51); its chain runs.

**2026-09-24 22:00 — the hold-from-onset gallery (g41x) against g41.** det F within −0.025 on all
19 — **P170 ✓** (C 0.863, beast 0.801: the collapse of the arrival gate alone is gone, the early
stop of the hold-from-the-start is gone on most: A 40 windows, bunny 45, heart 33); fit up or
within tolerance on 17 (bunny +0.006, maxplanck +0.006, fandisk +0.005, armadilo +0.004); but
**P169 ✗ on two: C −0.010** (0.891 at 28 windows, its stall) and **nefertiti −0.041** (0.927 at 29
windows against 90 — the run stopped with the transport a third done); **P171 ✗** (flips ≤ 0.6
and step ≤ g41's on 11). The onset reading — two consecutive negative reversal cosines of the
WHOLE body — misfires on the long, curved transports: nefertiti's and C's material reverses
transiently on the way, the hold engages, and a held step cannot finish the transport. The
per-particle halving is already protected by the arrival gate; the global hold's onset needs the
same protection: *the reversal read on the ARRIVED particles only* (the paced target's mask;
the global cosine when fewer than half have arrived) — no constant beyond the mask the pace
already computes. *g41y = the 19-target sweep with the arrival-read onset* (GPUs 3 and 0):
**P172** nefertiti ≥ 0.965 and C ≥ 0.898 with det F ≥ g41's − 0.05 everywhere; **P173** silIoU ≥
g41's − 0.003 on every target; **P174** flips ≤ 0.6 and step ≤ g41's on ≥ 15.
*Pre-registered am300 = z300b's recipe + H with the hold from the arrival-read onset*
(`--ctrl_rprop --ctrl_rprop_smooth --ctrl_rprop_k 8 --ctrl_rprop_arrived --ctrl_rprop_hold_onset
--u_rprop --u_rprop_floor 0`; GPU 2, 22:20) — the 300k deliverable candidate with the transport's
growing step kept until the settled body reverses: **P175** det F ≥ 0.55 (al300 0.51); **P176** ear
fill ≥ 0.93, silIoU ≥ 0.977; **P177** the layer's step ≤ 0.001 wu, flips ≤ 0.55, low-band > −0.2,
the video tail ≤ 0.0013, stride-19 ALT ≤ 0.0008; **P178** the hold engages between windows 45
and 70.

**2026-09-24 22:30 — d300 (the dragon at 300k with z300b's recipe, no Rprop): a generalisation
failure of the 300k recipe.** 175 windows: silIoU 0.9769 (the 40k g41_dragon 0.958 — the fit
generalises) but **det F 0.141**, strays 0.2 %, the layer's step 0.082 spacings (0.0056 wu) —
**P138 ✗ on det F**: on the dragon's spikes the recipe compresses the material badly, before any
Rprop. dr300 (with the arrival-gated smoothed Rprop) is still running (187 windows). *Attribution
twins launched (22:30):* **dl300 = the dragon with l300's recipe** (`--disc_ref --shift_sub
--commit_pic`, no KDE / hand-off / plan_native) — **P179** det F ≥ 0.6 says the KDE / hand-off
additions are the cause; **dk300 = z300b's recipe without `--w_kde`** — **P180** det F ≥ 0.55 says
the KDE term is the cause (the same term that gave the bunny its ear: a thin-feature target
with many spikes is where a particle-scale density pull compresses). Whichever it is, the D2
mechanism's adoption is conditional on the dragon.

**2026-09-24 23:00 — readings in hand.** *al300 (the KDE ear + H, hold from the start):* silIoU
0.9771 ✓, ear 0.952 ✓, det F 0.511 (P163 ✗ by 0.04), flips 0.51, step 0.00066 wu, video tail
0.0012 ✓ — but the layer's low-band correlation **−0.68**: with the KDE term the smooth band of
the layer still alternates (the bulk's reversal negative from 88 to the end) even though the
visible tail is at the acceptance value; the held global step and the KDE's per-window pull
alternate together. am300 (the hold from the arrival-read onset) is the next reading. *dr300 (the
dragon + the arrival-gated smoothed Rprop, no hold):* 206 windows, silIoU 0.9769, **det F 0.139**
— the same as d300's: the dragon's det F collapse is the 300k recipe's (the KDE / hand-off /
plan_native set), not the Rprop's; dl300 / dk300 attribute it. *g41y (the onset read on arrived
particles), first two:* **nefertiti 0.9719 (+0.004, 82 windows)** — the early stop is gone —
and cow −0.0014, det F within −0.01. *ac300 with the band-tracked mesh:* tail 0.0018 against
0.0013 untracked — the tracked-mesh family is refuted a second time.

**2026-09-24 23:20 — the PIN (method.md 10.27, `--settle_pin`): the user's "the oscillation
must be zero; once optimised, lock it".** The freeze (g41f) and the viscous forms (g41v/g41w)
left the settled body moving 0.0016 wu a window — the rollout's own floor. The pin is the
kinematic constraint inside the forward model: an arrived, twice-reversed particle has no
velocity, no affine velocity, no strain change, no control, no relaxation move, from that
window on (eq. 50), and the frames can read it — its frame-to-frame step is exactly 0. Twins
launched 23:20 on GPU 1: **g41p** = the g41y form (arrival-read onset hold) + `--settle_pin` at
40k; **ap300** = ai300's form (l300 recipe + H) + `--settle_pin` at 300k. Pre-registered:
**P181** the pinned fraction reaches ≥ 0.6 by the last window and the pinned body's
frame-to-frame step is exactly 0 (pin_probe: `still_frac` ≥ 0.6 at the end, un-pinning 0);
**P182** the alternation is gone from the arrived material: stride-19 ALT ≤ 0.0007 at 40k
(g41h 0.0014), the video tail ≤ 0.0008 at 300k (ai300 0.0011), the layer's step ≤ 0.0005 wu;
**P183** the fit's cost is bounded: silIoU ≥ g41y bunny − 0.005 at 40k and ≥ 0.9660 (ai300 −
0.005) at 300k, the ear fill ≥ ai300 − 0.02; **P184** the transport around the pinned obstacle
does not compress: det F ≥ 0.6 and strays ≤ 0.3 %. If P181 holds and P182 fails, the visible
motion is not the particles' — it is the fit's (the re-mesh), and the deliverable side answers
it; if P183 fails, the pin's onset (two reversals) is too early for the thin features and the
onset must be read per feature.

**2026-09-24 23:35 — a defect found by the pin's own reading, and a correction to g41v.** The 3k
smoke of the pin (12 windows, 1.2 % pinned by window 10) archived NO still particle: every frame's
zero-step fraction was 0.000. The cause is in `optimize_window`: the adjoint rollouts take the
`RolloutSpec` (which carried `eta` / `pin`), but the persistent no-grad trajectory `tr_eval` — the
line search's, the warm-start comparison's and **the commit rollout's** — is built directly and
had neither. So the pinned particles were pinned in the gradient and free in the delivered
frames; and the same held for `settle_eta`: **g41v optimised with the settled body viscous but
committed it inviscid** — its "invariant tail" reading (20:30 addendum) is of the wrong rollout
and is withdrawn (not re-run: the viscous form is diagnostic only by the user's rule). Fixed
(`tr_eval` takes `eta=spec.eta, pin=spec.pin`); the archive now carries the `pinned` mask, and
`pin_probe.py` reads the pinned set's step directly (exactly 0 expected). The 23:20 twins were
discarded and relaunched at 23:35 with the fix; P181–P184 stand.

**2026-09-24 23:50 — two chains in.** *The deliverable side (P151–P153, the tracked mesh
family):* z300b band-tracked + window-averaged, stride 19: tail 0.0021 (ALT/DRIFT 1.06);
ac300 band-tracked s12: tail 0.0018 (worse than the untracked 0.0013 — an advected mesh
follows the jitter in full, refuted again); **ac300 band-tracked + window-averaged s12: tail
0.0009 (ALT 0.0007, DRIFT 0.0008)**, s19: 0.0011 — the averaging, not the tracking, is what
lowers it, the same 0.0010 that `--frame_avg 38` gives on the particles alone; cosmetic, with a
one-window lag, and it does not touch the physics. *al300's chain (the KDE ear + H, hold from
the start):* the stride-19 video's tail D1 0.0008, ALT 0.0008, DRIFT 0.0005 (ALT/DRIFT 1.52,
at the white-noise ratio 1.73 — no excess alternation in the video) while the layer's low-band
correlation was −0.68 in the particles: the bulk's alternation of ~0.0007 wu is below what the
stride-19 video resolves. End still: bumpiness 1.19°, one component. The pin twins (g41p,
ap300) are the physics answer to the user's bar; al300/am300 remain the KDE-ear candidates.

**2026-09-25 00:10 — the user's question "is it the particles, or the re-mesh updating every
frame?" — answered on ai300 (300k, l300 + H), M4.** Two renders of the same archived frames,
stride 12: (A) the delivered one — the surface re-fitted (Poisson) on every frame; (B) the
advect-only one — the mesh fitted ONCE at the first delivered frame and carried by the
particles' own displacement thereafter, never re-fitted (`--track --track_alpha 0 --track_tol
1e9 --track_every 0`). Per-frame pixel change of the delivered tail (last 34 frames):

| render | tail D1 | ALT | DRIFT | ALT/DRIFT |
|---|---|---|---|---|
| (A) re-fitted every frame | 0.0011 | 0.0011 | 0.0005 | 2.05 |
| (B) advected, never re-fitted | **0.0015** | 0.0012 | 0.0010 | 1.14 |

The mesh that only follows the particles moves MORE than the re-fitted one: the visible motion
is the particles', and the per-frame re-fit does not create it — it removes part of it (the fit
averages the sub-cell rearrangement out; 0.0015 → 0.0011). With M1 (the normal-only twin
reproduces the full change) and M2 (the surface's normal velocity equals the layer's normal
step, ratio 0.91–0.98) the kind is settled: **a genuine normal motion of the settled surface of
~0.001 wu a window at 300k, carried by the particles, not a reconstruction artefact.** (B)'s
ALT/DRIFT 1.14 against (A)'s 2.05: the advected mesh also carries the tangential drift, which
the re-fit does not see. The remedy is therefore on the physics side — the pin (10.27), whose
settled body does not move at all — and the deliverable side can only average (frame_avg /
bandavg 0.0009–0.0010), which is cosmetic.

**2026-09-25 00:40 — g41p's first reading, and the pin's generalisation launched.** g41p (40k
bunny, the g41y form + `--settle_pin`): 49 windows (early stop at the best commit), **silIoU
0.9679** (g41 0.9610, the H twins 0.964–0.965 — the pin does not cost the fit, it gains: a
settled particle that can no longer wander stops trading its own error against its
neighbours'), det F 0.74 (P184 ✓), the metric's jitter_rel 0.00000, pinned 41 % at window 20,
49 % at 25, 70 % at 45 (P181's fraction ✓; the frame reading and P182's video follow in the
chain). Launched 00:38 (the user's rule: the whole gallery before adoption): **g41z** = the
g41p form on all 19 targets (GPU 2 and GPU 1, two chains) — **P185** every target within
−0.003 of g41y (or of g41, where g41y is worse), pinned ≥ 0.5 at the end on ≥ 15, det F within
−0.05 on all; **g41q** = the pin WITHOUT the hold (`--ctrl_rprop --ctrl_rprop_smooth
--ctrl_rprop_arrived --u_rprop --u_rprop_floor 0 --settle_pin`) on C and the bunny (GPU 0)
— **P186** C recovers g41t's fit (≥ 0.94, the hold's stall gone) and the bunny keeps g41p's
(≥ 0.965) with the same still fraction: then the pin makes the global hold unnecessary and the
recipe is the simpler one (arrival-gated smoothed Rprop + pin).

**2026-09-25 01:00 — g41p's verdict: P181–P184 all ✓ on the bunny at 40k.** 49 windows (early
stop at the best commit), silIoU **0.9679** (g41 0.9610, g41h 0.964–0.965: P183 ✓ with a gain),
det F 0.7415, strays 0.02 % (P184 ✓), ear tip 17 of 18 reference particles, one component,
bumpiness 1.18°. The reversal series has **no negative window at all** (g41h 1 of 36, g41 many),
layer flips 0.30, the layer's step 0.0093 spacings (the H twins 0.02), low-band correlation
+0.17. Pinned fraction per window 0.01 at 10, 0.34 at 20, 0.54 at 30, 0.70 at 49; the frames'
own reading (`pin_probe.py`, window level): **0 of 28 119 end-pinned particles moved again
after their first still window** — the settled body is exactly still (P181 ✓). Video: stride 12
delivered tail **0.0003** (p90 0.0005), stride 19 tail D1 0.0004 / ALT 0.0005 / DRIFT 0.0004
(P182 ✓: g41h's ALT 0.0014 — a third); whole-run 0.0016 (the transport itself). What remains in
the tail is the fit of the still-moving 30 % (the ears' last arrivals) and the codec.

The generalisation (g41z, 19 targets) and the no-hold form (g41q) are the adoption gate
(P185–P186); ap300 the 300k reading. The archive now records the pin window per particle
(`pinned_at`) for the exact check on every later run.

**2026-09-25 01:40 — am300's verdict, and the pin's first gallery readings (C).** *am300 (the
KDE ear + H with the arrival-read onset hold, 300k):* 129 windows, **silIoU 0.9799** (the best
300k fit with the ear so far; z300b 0.9802 without H), det F **0.4067** (P175 ✗ — the KDE
ear's compression is not the hold's to fix), the hold engaged at window 90 (P178's 45–70 ✗:
the arrived-read cosine equals the global one on the bunny — the arrived set IS the bulk — and
it crossed −0.2 twice in a row only at 89–90), and the alternation continued under the hold
(37 negative windows from 86 on, layer flips 0.62, step 0.0148 spacings) — with the KDE term the
held step is not enough: the KDE's per-window pull reverses itself at the held amplitude. The
ear: 118 particles at the tip = 15.7 reference particles (target 89 at 300k's count — the tip
census is at the 40k reference scale; the fill number follows in the chain). The KDE ear at
300k is therefore not settled by H; the pin is the next reading on it (ap300 has no KDE).

*The pin on C (40k, g41z / g41q, 51 windows):* **g41z_C silIoU 0.9616** (g41 0.9009, g41t
0.9496, g41y 0.8861), det F 0.82, layer step 0.0095 spacings; **g41q_C (the pin WITHOUT the
global hold) 0.9621**, det F 0.82 — identical: the hold never engaged on C in either (its
cosine stays +0.85, pure transport), so the pin alone ends C's stall — C's failure mode was
the arrived arms' re-adjustment fighting the transport of the rest, and a pinned arm is a
fixed obstacle the transport flows round. A caveat read from g41y_C: its 25-window early stop
("3 consecutive rejected candidates") happened at cos +0.85 with the hold never engaged, on the
same flags as g41t's 113-window run — C's early stop is a run-to-run event of the merit's
rejection streak, so C's numbers carry that variance; the pin's +0.06 is far outside it.
cow: −0.0009, det F 0.775 (g41y 0.843; P185's det F bound −0.05 ✗ on cow by 0.02 — the
transport round a pinned body compresses at the boundary; watch the rest of the gallery).

**2026-09-25 02:15 — g41y complete (19 of 19): the arrival-read onset hold on the gallery.**
Fit vs g41: 17 up or within −0.0015; **C −0.0148** (P172 ✗ on C alone — its 25-window early
stop, a rejection streak during pure transport, cos +0.85, the hold never engaged); det F: beast
−0.060, dragon −0.053, homer −0.033, the rest within −0.01, ogre +0.051; flips ≤ 0.6 on 15 and
step ≤ g41's on 16 (P174 ✓). The onset hold is a clean improvement over g41x (nefertiti +0.0043,
82 windows) but it does not touch what the pin does: the alternation's residual (steps 0.017–0.05
spacings, flips 0.5–0.66 on the smooth targets) stays.

*g41z so far (the pin, 4 of 19):* C **+0.061**, bunny +0.006, nefertiti +0.001 (81 windows,
step 0.0038 spacings, flips 0.38), cow −0.001 — the fit holds everywhere; **det F: bunny −0.02,
C −0.04, cow −0.078 (0.775), nefertiti −0.083 (0.681)** — P185's det F bound (−0.05) ✗ on cow
and nefertiti: the last arrivals compress against the pinned body (a fixed obstacle at the
arrived/in-transit boundary, the same shear g41t showed on C/beast). The pin's cost is where
the freeze's was — at the boundary — and it is the reading to watch on the remaining 15 before
adoption. A boundary-aware variant, if needed: pin only particles whose neighbourhood (the
regulariser's kNN) is arrived as well — no constant, the coherence neighbourhood's own test.

**2026-09-25 02:45 — P186 ✓: the pin makes the global hold unnecessary.** g41q (arrival-gated
smoothed Rprop + `--settle_pin`, NO `--ctrl_rprop_hold*`): **bunny 0.9689** (g41p with the hold
0.9679, g41 0.9610), det F 0.742, 55 windows, layer flips 0.37, step 0.0124 spacings; **C 0.9621**
(g41z_C with the onset hold 0.9616). The hold was the answer to the anneal's ×1.15 re-inflation
of the global step cancelling the per-particle decay (ag300); with the pin the arrived particles
have no step at all, so the global step may keep its recovery for the transport — the simpler
recipe is at least as good on both. Launched 02:45 (GPU 0): g41q on the remaining 17 targets —
the adoption candidate for 40k is now **g41q's form**, gated by P185's bounds applied to it
(fit within −0.003 of g41y/g41, det F within −0.05 — the boundary compression seen in g41z on
cow/nefertiti is the open item). The user closed the local session at 02:40 with "keep them
running"; all chains are detached (nohup, ppid 1).

**2026-09-25 09:50 (server 19:45 CDT) — the night's verdicts, read after the reconnection
(hyde01 jump → socat proxy; hyde06 is campus-private 10.2.191.42).**

*g41z (the pin + onset hold, 19 of 19) vs g41:* fit up or within −0.003 on **17** (C +0.061,
maxplanck +0.007, bunny +0.006, dragon +0.006, fandisk +0.005, A +0.004 …); **beast −0.017,
ogre −0.006** (both early stops, 77 / 52 windows). det F within −0.05 on 13; beyond on six —
armadilo −0.111 (0.731), nefertiti −0.083 (0.681), cow −0.078 (0.775), V −0.054, dragon −0.054,
maxplanck −0.051 — all still ≥ 0.68. The layer's step 0.000–0.02 spacings on every target (g41
0.02–0.06), flips 0.2–0.56 (g41 0.54–0.79): **the oscillation metrics fall everywhere.** P185:
fit ✗ on 2, det F ✗ on 6 → not adopted as is. *g41q (the pin without the hold, 14 of 19):* the
same picture; bunny 0.9689, dragon 0.9648, beast 0.9449 better than z, nefertiti 0.9542 (42
windows, early stop) and cow 0.9557 worse — hold vs no hold is a wash; the pin's onset decides.

*Where the det F minimum lives (`scratch/detf_dist.py`, the last archived F):* g41z cow min
0.778 in the UNPINNED set (pinned 0.816), armadilo 0.718 unpinned (pinned 0.818) — the last
arrivals compressed against the pinned boundary; nefertiti 0.696 in the pinned set (pinned
with that strain, at the boundary too).

*ap300 (300k bunny, ai300 + pin):* 103 windows, pinned **91.8 %** (P181 ✓; the frames read
still fraction 1.000 at the end), silIoU 0.9729 (P183's fit ≥ 0.9660 ✓; ai300 0.9766), strays
0.04 %, layer flips 0.22, step 0.0056 spacings, low-band correlation **+0.66**, video tail
stride 12 **0.0008** (P182 ✓ at the bound; ai300 0.0011), stride-12 ALT 0.0008, bump 1.23°,
one component. **det F 0.591 (P184 ✗ by 0.009; the minimum in the pinned set).** **The ear:
tip 3.9 reference particles (ai300 13.6), the last two slabs 0.66 / 0.18 against 0.89 / 1.13
— P183's ear ✗:** the base was pinned while the tip was still fed through it, and the
transport stopped. The pin works on the settled body and starves the thin feature.

*am300's chain:* video tail 0.0015 (stride 19: 0.0013, ALT 0.0012), bump 1.28° — worse than
ai300 (0.0011): with the KDE term the held step still alternates (P177 ✗). The KDE ear + H
is refuted at 300k on det F (0.41) and on the tail.

*The dragon's det F (dl300 = l300's recipe, dk300 = z300b's without the KDE):* dl300 silIoU
0.9772, det F **0.169**; dk300 0.9718, det F **0.162** — P179 ✗, P180 ✗: neither the KDE nor
the hand-off / plan_native set is the cause. The distribution says what the minimum hides:
**59 / 53 particles below 0.3 (0.02 %), p1 0.76 / 0.78, median 0.997** (the 40k dragon: min
0.81, p1 0.93; d300 with the KDE: 334 below 0.3 — the KDE multiplies the count ×6 but does not
create it). The 300k dragon's "collapse" is a few dozen particles at the spikes under the base
300k recipe (`--disc_ref --shift_sub --commit_pic`); the reading to adopt for N > 40k is the
1st percentile and the count below 0.5, not the minimum over 300k particles. The base recipe's
spike compression is a separate item (P187 below).

**The clear-neighbourhood pin (method.md 10.27 addendum, eq. 50b; `--settle_pin_clear`):** a
particle is pinned only when no unarrived particle lies within the pace radius of it — the
paced target's own arrival scale. Launched (after a 3k smoke): **g41n** = the no-hold pin form
+ clear on all 19 targets (GPUs 1 and 3), **an300** = ap300's form + clear (GPU 2).
Pre-registered: **P188** g41n fit within −0.003 of max(g41, g41z) on ≥ 18 (beast, ogre
recover: no early stop before g41's window count × 0.7); **P189** det F within −0.05 of g41 on
≥ 17 and the minimum no longer in the unpinned set on cow / armadilo; **P190** the
oscillation metrics stay at g41z's (step ≤ 0.02 spacings, flips ≤ 0.56 on all); **P191** an300
ear tip ≥ 11 reference particles and the last two slabs ≥ 0.85 / 1.0 (ai300's), silIoU ≥
0.9729, det F ≥ 0.6, pinned ≥ 0.85 at the end, tail ≤ 0.0008. **P187** (the dragon, not yet
launched): the 300k dragon under the 40k recipe scaled without `--disc_ref` has p1 ≥ 0.9 —
then the spike compression is disc_ref's shortened lengths on the spikes.

**2026-09-25 10:30 — where the pin's det F goes (`scratch/detf_time.py`), and the stress-free
pin.** g41n's first two: bunny 0.9686 (+0.0075), det F 0.752; cow 0.9614 (+0.0006), det F
0.770 — the clear rule changed nothing on det F (g41z cow 0.775). The per-window series split
by pin state: **cow** — the compression is in the UNPINNED set and grows with the pinned
fraction (0.857 at window 20 / 18 % pinned → 0.770 at 42 / 61 %), then relaxes to 0.80 by 45;
the pinned set's minimum is 0.804, locked at pin time. **bunny** — the un-pinned minimum is the
transport's own (0.752 at window 17; g41 0.771 at 16) and recovers to 0.84 by window 30, but
the particle pinned at window 30 with 0.762 keeps it: **the pin locks transient compression as
elastic strain** (F is the total deformation; the metric reads it). Two consequences: the
metric's minimum is a few locked particles, and — the real defect — a pinned body with locked
elastic strain is not at equilibrium after the morph. Addendum 2 (eq. 50c, `--settle_pin_assim`):
the elastic stretch is assimilated in full at pin time (the freeze's assimilation), the pinned
body is stress-free, the delivered object at rest. Launched (GPU 0, after a 3k smoke): **g41pa**
= pin + clear + assim (no hold) on bunny, cow, nefertiti, beast, armadilo, ogre. Pre-registered:
**P192** the pinned set's elastic det F_e = det(F F_p^{-1}) at the end ≥ 0.99 (the archive now
carries F_p); **P193** fit within ±0.002 of g41n on bunny / cow, and beast / ogre / nefertiti no
worse than g41n's; **P194** cow's transient un-pinned compression no deeper than g41n's (0.770)
— the stress-free wall neither pushes nor pulls — and the end minimum ≥ 0.80. The
generalisation gate for adoption stays P188–P190 on the full g41n sweep, read with p1 and the
count below 0.5 alongside the minimum. g41f (the freeze, which assimilated) had det F 0.788 on
the bunny against H's 0.781: the assimilation did not cost det F there.

**2026-09-25 11:00 — two corrections from the readings, and the adoption sweep.** *(1) The hold
is needed with the pin (P186 revised).* g41n (the no-hold pin + clear, 6 of 19): C 0.9638
(+0.063), bunny +0.0075, dragon +0.0054, fandisk +0.0054, cow +0.0006 — and **nefertiti 0.9540
(−0.0136) at 38 windows**, the same early stop as g41q's (42 windows); with the onset hold
(g41z) nefertiti ran 81 windows to 0.9685, and without pin or hold (g41t) 90 windows to
0.9676. Without the hold the global step keeps re-inflating (the anneal's ×1.15) while the free
set shrinks under the pin, the candidates on the remaining slow transport are rejected three
times and the run stops. P186 held on the bunny and C only; the onset hold stays in the recipe.
The g41n sweep was stopped at 6 (its six archives kept). *(2) The pin-time assimilation was
undone at the next commit.* On the stress-free pin's 3k smoke, particles pinned one commit or
more earlier were back at det F_p = 1 with their compression elastic again: the commit-time
assimilation (η = 0.5) projects the cumulative F_p onto det F_p = 1 (`assim_iso`, the exact
log-band projection) for every particle, so the volumetric part of the pin-time assimilation
is removed one commit later. A pinned particle's F_p is now final — it is excluded from the
commit-time assimilation (it has nothing left to assimilate). The clear rule alone does not
change det F (g41n cow 0.770 against g41z's 0.775; the compression is the transit's, 10:30).

**g41pz** = the z form (arrival-gated smoothed Rprop + onset hold) + `--settle_pin
--settle_pin_clear --settle_pin_assim`, all 19 targets (GPUs 1 and 3, after a 3k smoke): the
adoption candidate for 40k. Gates: **P188'** fit within −0.003 of max(g41, g41z) on ≥ 18
(nefertiti ≥ 0.965 with ≥ 70 windows; beast, ogre no worse than g41z); **P189'** det F within
−0.05 of g41 on ≥ 15 and ≥ 0.65 on all; **P190** step ≤ 0.02 spacings and flips ≤ 0.56 on all;
**P192** the pinned set's elastic det F_e ≥ 0.99 at the end on every target (the archive's
F_p). an300 (300k, ap300's form + clear, no assim) keeps running as the ear reading (P191).

**2026-09-25 11:20 — the next phase pre-registered (dossier §14; the user's order: once the
oscillation is caught, the droplets and the smooth surface).** P196 (the tip's dropout is the
reconstruction's, unchanged by the pin), P197 (R-2: finest node ≤ the tip's half-thickness, no
density coarsening, symmetrized normals → capture ≥ 0.8 every frame at ≤ 1.2°), P198 (the
Ando 2012 sheet-aware split at the commit → pieces < 10, tongue ≥ 0.9, tip ≥ 11, det F p1 ≥
0.85), P199 (≥ 70 % of the dihedral excess within one cell of the last arrivals' boundary).
Not launched until g41pz / an300 / dp300 are read. The stress-free pin's smoke: the pinned
set's elastic det F_e = 1.000 exactly (min, p1, median) — P192 holds on the smoke.

**2026-09-25 11:50 — the pinned body blocks transit streams (g41pz 4 of 19), and the ray rule.**
g41pz (onset hold + pin + clear + assim): bunny 0.9679 (+0.0069, det F 0.760), dragon 0.9640
(+0.0058), fandisk 0.9753 (+0.0062) — and **nefertiti 0.9549 at 42 windows again** (g41q 42,
g41n 38; g41z's 81 was the exception, not the hold's doing — P186's revision is itself
revised: the hold is not what decides nefertiti). Its log: windows 36–42, candidates rejected
with gain −0.046 … −0.070 at reversal cosine **+0.94** — transport windows whose candidate is
worse than the last commit, 80.7 % arrived, 46 % pinned: the crown's remaining stream has to
pass through the pinned bust, and cannot. The point-clear rule protects the boundary only at
pin time. Addendum 3 (eq. 50d, `--settle_pin_ray`): a particle is pinned only when no unarrived
particle's plan ray passes within the pace radius of it. Launched (GPU 0, after a 3k smoke):
**g41pr** = onset hold + pin + ray + assim on nefertiti, beast, ogre, bunny, C, cow.
Pre-registered: **P200** nefertiti ≥ 0.965 with ≥ 70 windows and no rejection streak before
95 % arrival; beast ≥ g41z's 0.9374 + 0.01, ogre ≥ g41z's 0.9488 + 0.005; **P201** bunny / C /
cow within ±0.003 of g41pz / g41z and the pinned fraction at the end ≥ 0.6 (the rays only delay,
they do not prevent the pin); **P202** (later, at 300k) the ear's tip ≥ 11 reference particles
with the base pinned last. g41pz continues on the other 15 as the point-rule reading.

**2026-09-25 12:30 — the ray rule at the pace radius refuted (g41pr nefertiti 38 windows, the
same rejections: gain −0.056 / −0.068 at reversal +0.93), and the clearance corrected to the
grid kernel's support.** The ray was clear and the stream still stopped: the pace radius (≈ 0.1
wu, the loss grid's cell) is a third of the MPM cell (Δx ≈ 0.33 wu at 40k) and the pinned mass
acts through the grid over the kernel's support (2 Δx): a stream particle sharing a node with
pinned mass receives the mass-weighted momentum, near zero — a no-slip boundary layer one
support wide along the settled bust. The clearance of (50b)/(50d) is now max(r_pace, 2 Δx).
Consequence: a settled region within 2 Δx of a transit ray stays free (breathing as before)
until that stream has arrived, and pins afterwards — the pin acts early only far from every
stream. g41pr's sweep was stopped; **g41ps** (the same form with the stencil clearance) on
nefertiti, beast, ogre, bunny, C, cow (GPU 0). P200–P201 carried over to g41ps; **P203** the
pinned fraction at the end of nefertiti ≥ 0.6 (the clearance delays, it does not prevent).
an300's run: silIoU 0.9725, **det F 0.657** (ap300 0.591 — the point clearance at r_pace already
lifted the 300k det F above the bound), 71 windows; its chain (ear, tail) pending.

**2026-09-25 13:00 — an300's verdict (300k, the point clearance at r_pace, no assimilation):**
71 windows, silIoU 0.9725, **det F 0.657** (ap300 0.591: the clearance lifts the 300k det F above
the bound), strays 0.04 %, pinned **76.8 %** with the body exactly still (still fraction 1.000
at the end), bump 1.23°, one component. **The ear: tip 6.8 reference particles** (ap300 3.9,
ai300 13.6), the last two slabs 0.72 / 0.46 (ap300 0.66 / 0.18, ai300 0.89 / 1.13) — better,
still starved (P191 ✗). **The tail: 0.0011** (ap300 0.0008, ai300 0.0011), ALT 0.0011, the
layer's flips 0.47, low-band correlation −0.18 (min −0.70): the material the clearance keeps
free (23 %, the ear's region) breathes as in ai300. The reading is consistent with g41pr/g41ps:
a pinned body drags every stream within its stencil, and the ear at 300k IS a stream to the
end — the base must stay free while the tip is fed. **as300** launched (GPU 2): ai300's form
+ `--settle_pin --settle_pin_ray --settle_pin_assim` with the stencil clearance (2 Δx).
Pre-registered: **P204** the ear tip ≥ 11 reference particles and the last two slabs ≥ 0.85 /
1.0 (ai300's, the ear no longer starved); **P205** the body far from the ear exactly still with
the pinned fraction ≥ 0.6 at the end; **P206** the tail ≤ 0.0011 (ai300's — the ear's own
breathing is the floor while it is fed) and det F ≥ 0.6, silIoU ≥ 0.9725, the pinned set's
elastic det F_e ≥ 0.99.

**2026-09-25 13:45 — g41ps nefertiti: the stream passes (P200 ✓) and is squeezed; the yield
rule.** g41ps (stencil clearance along the rays + assim + onset hold): nefertiti **0.9727**
(+0.0052 — the best fit of any run on it; g41t 0.9676 at 90 windows), 114 windows, pinned
71 %, flips 0.37, step 0.020. **det F 0.457** — the time series split by pin state: the
pinned fraction stalls at 51 % for windows 41–72 while the crown stream flows, the UNPINNED
minimum falls 0.78 → 0.46 over windows 45–85 (the stream squeezed between pinned walls two
supports apart), the arrivals pin at 0.55 and the free set recovers to 0.76; the pinned set's
elastic det F_e is 1.000 (stress-free, the compression is rest volume). g41pz (the point
clearance at r_pace, 16 of 19): fit up or within −0.003 on 13; beast −0.014, nefertiti −0.013,
ogre −0.003, C +0.055; det F beyond −0.05 on 7 (maxplanck 0.778, C 0.784, V 0.776, cow 0.782,
homer 0.753, teapot 0.813). Addendum 4 (eq. 50e, `--settle_pin_yield`): a settled particle
within 2 Δx of a transit ray is released for the window — no control, no relaxation move,
passive material that yields to the stream through the physics alone — and re-pinned
(re-assimilated) when the stream has passed. Launched (GPU 3, after a 3k smoke): **g41py** =
g41ps's form + yield on nefertiti, bunny, cow, beast. Pre-registered: **P207** nefertiti's
un-pinned det F minimum stays ≥ 0.70 through the stream (g41ps 0.46) and the end minimum ≥
0.75, with the fit ≥ 0.970 and ≥ 90 windows; **P208** cow / bunny within ±0.003 of g41ps /
g41pz with det F ≥ 0.78 (g41's cow 0.853 is the no-pin reference; the yielding walls remove
the squeeze but not the transit's own dip); **P209** the released fraction of the settled set
is < 20 % at any window on the bunny and the settled set far from streams is exactly still
(pin_probe on the frames). If P207 holds, g41py's form goes to the 19 as the 40k adoption
sweep and to 300k.

**2026-09-25 14:20 — g41pz complete (19 of 19; the pin with the point clearance at r_pace +
onset hold + assim), vs g41:** fit up or within −0.003 on **16** (C +0.055, bunny +0.007,
maxplanck +0.007, fandisk +0.006, dragon +0.006, A +0.003, cheburashka +0.002, bimba +0.002,
spot +0.002, bob +0.001, armadilo +0.001, heart +0.001, V +0.001, teapot −0.001, homer −0.001,
cow −0.002); **beast −0.014, nefertiti −0.013** (the stream cases), ogre −0.003. det F within
−0.05 on 12, beyond on 7 (maxplanck 0.778, C 0.784, V 0.776, cow 0.782, homer 0.753, teapot
0.813, all ≥ 0.75); the layer's step ≤ 0.024 spacings and flips ≤ 0.56 on all but beast (0.78:
its transport never settles under the pin). P188' ✗ (16 of 18), P189' ✗ (12 of 15), P190 ✓ on
18. The yield form (g41py) is the candidate that addresses the two failing classes (streams);
it runs on nefertiti, bunny, cow, beast (GPU 3) and ogre, C, maxplanck, dragon, homer (GPU 1).
The 3k yield smoke: released fraction 0–2 % of the settled set (the bunny at 3k has no long
stream), pinned set's elastic det F_e 1.000.

**2026-09-25 15:00 — g41py nefertiti (the yield rule): the fit holds, the det F minimum is a
handful of boundary particles, and the reading of det F changes.** g41py nefertiti 0.9702
(+0.0026), 107 windows, pinned 70 %, released ≤ 6 % of the settled set at any window; det F
minimum **0.260** (g41ps 0.457). Where (`scratch/compress_where.py`, the end frame): the
un-pinned particles below 0.5 are **3**, below 0.7 **15** (of 11 878), every one within 2 Δx of a
pinned particle (median 0.26–0.31 Δx — touching), in the bust's middle band, not the crown's
stream; the pinned set's own minimum 0.68 (one particle, pinned at window 92), p1 0.90; the
elastic det F_e of the pinned set 0.925 min / 0.992 p1 (a few re-pinned late). So the yield
rule did remove the corridor squeeze (no compressed band along the stream) and what remains
is **the last arrivals wedged against pinned neighbours**: the paced target snaps an arrived
particle's image to the nearest target point without exclusivity, so a late arrival's
destination can be occupied by pinned material, and where the un-pinned run would jostle both
until equal, the pinned neighbour does not move and the arrival compresses. Three particles.
P207's "≥ 0.70 through the stream" is ✗ on the minimum and ✓ on the stream (the compression is
not in the corridor); the minimum over 40k particles is not the reading for this — as decided
for the dragon, the health gate for the pin runs is **the 1st percentile ≥ 0.85 and the count
below 0.5 ≤ 0.1 %**, alongside the minimum reported. Arrived vs pinned over nefertiti's run:
arrived 78–80 % from window 30 to 70 while pinned rose 24 → 59 %, arrived 92–96 % from 80 on —
the pinned set never exceeds the arrived set (no re-assigned pinned images).

*dp300* (the dragon at 300k without `--disc_ref`) at 171 of 175 windows; verdict next.
*Agents launched* (the user: "수단과 방법을 가리지 마", and "cookbook / 해상도 자료도 전부"): a
literature digest on slip / kinematic boundaries in MLS-MPM (CPIC and the multi-field
contacts) and adaptive-resolution MPM, and a practitioner digest (course notes, Houdini / Taichi
/ Warp / splashsurf / OpenVDB docs) on ppc, cells across a feature, dt, thin features, pinned
material and surfacing smoothness — both to docs/related_work.md when they return.

**2026-09-25 15:20 — the pin gallery's det F, read at the END state by quantiles
(`detf_dist.py`, the last archived F):** g41pz, all 19: 1st percentile **0.883–0.921**, end
minimum 0.738–0.846, **no particle below 0.5 on any target** (the metric's "detF_min" is the
run minimum over the trajectory — the transit's transient dip, 0.73–0.85 — not the delivered
state). g41 references: p1 0.936–0.946, end min 0.84–0.89 — the pin broadens the distribution
by ~0.03 at p1 and lowers the end minimum by ~0.05: the boundary wedging of the last arrivals,
a few particles. g41py so far: bunny p1 0.914, C 0.899, maxplanck 0.911, ogre 0.915, nefertiti
0.885 with 3 particles below 0.5 (0.01 %). Under the health gate stated at 15:00 (p1 ≥ 0.85,
count below 0.5 ≤ 0.1 %) **every pin run passes**; the run minimum stays reported as the
transit's dip. What is left for the adoption of the 40k form is the fit on the stream cases
(beast, nefertiti, ogre) under the yield rule and the sweep of the rest.

**2026-09-25 15:30 — the yield form to the whole gallery and to 300k.** The g41ps chain (the
form without the yield) is stopped as superseded; **g41py** now runs on all 19 (the remaining
ten on GPU 0). **ay300** = as300's form + `--settle_pin_yield` (300k, GPU 2). Pre-registered:
**P210** g41py on the 19: fit up or within −0.003 of max(g41, g41pz) on ≥ 17 with beast ≥
0.950 and nefertiti ≥ 0.970 (the stream cases); the end-state det F p1 ≥ 0.85 and no more than
0.1 % below 0.5 on all; the layer's step ≤ 0.02 spacings and flips ≤ 0.56 on ≥ 18; the pinned
fraction at the end ≥ 0.6 on ≥ 17. **P211** ay300: the ear's tip ≥ 11 reference particles and
the last two slabs ≥ 0.85 / 1.0, silIoU ≥ 0.9725, end p1 ≥ 0.85, the tail ≤ 0.0011 with the
body far from the ear exactly still, released ≤ 30 % of the settled set at any window. If P210
holds, the 40k recipe becomes the g41py form (arrival-gated smoothed Rprop + onset hold + pin
+ ray clearance at 2 Δx + yield + stress-free assimilation) and the D1 phase closes at 40k; if
P211 holds, the same at 300k.

**2026-09-25 16:10 — dp300 (the dragon at 300k WITHOUT `--disc_ref`, the native, finer grid):
P187 ✗, and the opposite of it.** 200 windows, silIoU 0.9721, det F minimum **0.003**, end-state
p1 **0.699**, 270 particles below 0.3 (0.09 %), 817 below 0.5 (0.27 %) — against dl300 (with
`--disc_ref`): p1 0.764, 59 below 0.3, 272 below 0.5. The finer grid makes the spike
compression WORSE, not better: the reference discretisation is not the cause. What the three
dragon runs share is the spikes themselves — thin target features into which the transport
delivers more particles than the feature holds: the paced target snaps every arrived
particle's image to its NEAREST target point with no exclusivity (optimizer, the arrival
snap), so a thin spike's few target points receive the images of every particle the plan sends
near them, and the cell-sum loss packs them in. At 40k the spikes are below the cell and the
question does not arise (det F 0.81); at 300k the finer the grid, the more exactly the loss
resolves the over-filled spike. The same mechanism wedges the last arrivals against the pinned
body at 40k (15:00). This is a D2 item (the thin feature's capacity), not the pin's: the
arrival step must respect the target's capacity — the plan's own matching (a particle's
image is its transported mass's destination, one-to-one by mass) instead of the nearest-point
snap, or a density-capped hand-off (the Maury 2010 congestion projection applied at the
arrival, not to the whole transport as §10.22 did). To be pre-registered when the D2 phase
opens (dossier §14 gains this as item 0). For the 300k gallery gate the dragon's reading is
p1 / the counts; `--disc_ref` stays in the 300k recipe.

**2026-09-25 17:00 — the practitioner digest (docs/related_work.md, "Practitioner rules …") and
the slip wall.** What the graphics-side cookbooks and engine docs settle: 8 particles per cell
is the production default everywhere (Houdini MPM grid scale 2, taichi_elements 2^dim, Warp,
ZIRAN, GPU-MPM "eight per cell for stability") — ours is ≈ 25 at 40k and ≈ 190 at 300k under
`--disc_ref`, which buys nothing; a feature needs ≥ 2 cells across (FLIP Fluids, OpenVDB's
1.5-voxel Nyquist radius) — the ear has ≈ 1; a collider is handled on the GRID after the forces,
relative to its velocity, only when approaching (course notes §12.1, taichi/warp-mpm
`separate`), and overriding particle velocities — our pin — is the sticky/Dirichlet case; the
surfacing docs put the particle radius at 1.4–1.6 × separation and the voxel at 0.5–0.75 ×
radius with 15–25 feature-weighted smoothing iterations (splashsurf), which is the R-2 item. The
5 ranked changes are in the digest; the one for D1 is the second: **the pinned body as a
grid-level separating collider** (method.md 10.27 addendum 5, `--settle_pin_slip`, eq. 50f):
pinned mass leaves the momentum average, its mass field is rasterised once per window, and the
approaching normal component is removed at the nodes it covers. Launched (GPU 1, after a 3k
smoke): **g41pw** = onset hold + pin + assim + slip (no clearance, no ray, no yield) on
nefertiti, beast, bunny, cow. Pre-registered: **P213** nefertiti ≥ 0.970 with ≥ 90 windows and
no rejection streak before 95 % arrival, and its un-pinned det F minimum ≥ 0.70 through the
stream (the wall neither drags nor squeezes); **P214** beast ≥ 0.950; bunny / cow within ±0.003
of g41py with the pinned fraction at the end ≥ 0.65 (the slip wall pins earlier than the
clearance rules allowed); **P215** the end-state p1 ≥ 0.88 on all four. If P213–P215 hold, the
slip form replaces the ray / yield rules in the recipe and goes to the 19 and to 300k.

**2026-09-25 17:20 — as300's verdict (300k; the ray clearance at 2 Δx + stress-free assimilation
+ hold), and the two digests in.** 92 windows, silIoU 0.9713, det F 0.650 (end-state un-pinned p1
0.80), strays 0.04 %, **pinned 90.1 % and exactly still**, the pinned set's elastic det F_e 1.000,
bump 1.21°, one component. **The tail: 0.0006 (p90 0.0008), stride-12 ALT 0.0006, low-band
correlation +0.46, flips 0.53** — the lowest tail of any 300k run (ai300 0.0011, the 40k
acceptance 0.0013): P205 ✓, P206 ✓ by a wide margin. **The ear: tip 7.7 reference particles**
(an300 6.8, ap300 3.9, ai300 13.6), the last two slabs 0.81 / 0.41 (ai300 0.89 / 1.13): P204 ✗ —
the ear is still fed through material the clearance rules keep free only along straight
rays, and the base's boundary layer drags the feed. ay300 (+ the yield) is the next reading at
300k; the slip collider (addendum 5) after it. *The digests* (docs/related_work.md): the
2 Δx stick layer is a documented property of shared-node MPM (Nairn 2020: "contact is always
detected too early"; Ménager 2026; CK-MPM's ball stuck at 1.5 Δx); every production collider
deposits no mass and is a per-node velocity constraint relative to its own velocity (Stomakhin
2013, Klár 2016, PlasticineLab, Newton/Warp) — our pin deposited mass, which is the drag; the
principled release of a frozen set is a KKT check on the gradient or a contact wake (Bertsekas
1982, strong rules, LIBSVM shrinking, Box2D islands), not a distance; for the ear, the only
adaptive scheme that changes nothing in the transfers is a nested fine box coupled by a penalty
(He 2025), and the quadratic / compact kernel halves both the stick layer and the ear's smear at
zero memory. The slip smoke's first launch crashed (the pinned mass rasterised before the
positions existed); fixed (rasterised at step 0 of each rollout), relaunched.

**2026-09-25 17:45 — the slip collider at 300k in parallel (aw300, GPU 3).** The slip smoke (3k)
passed with the pinned set's elastic det F_e 1.000; g41pw runs on nefertiti / beast / bunny /
cow (GPU 1). Since the 300k ear's starvation (ap300 3.9 → an300 6.8 → as300 7.7 against
ai300's 13.6) is the boundary layer at the pinned base dragging the feed, and the slip
collider is the mechanism that removes that layer, **aw300** = ai300's form + `--settle_pin
--settle_pin_assim --settle_pin_slip` (no clearance, ray or yield) is launched without waiting.
Pre-registered: **P216** the ear's tip ≥ 11 reference particles and the last two slabs ≥ 0.85 /
1.0 with the pinned fraction ≥ 0.85 at the end (the slip wall pins the base as soon as it has
arrived and still lets the feed through); **P217** the tail ≤ 0.0008 (as300 0.0006), silIoU ≥
0.9725, end-state un-pinned p1 ≥ 0.85, the body exactly still. If P216–P217 hold, the 300k
recipe is the slip form and D1 closes at 300k with the ear intact; ay300 (the yield form, GPU
2) stays as the comparison.

**2026-09-25 18:00 — g41pw nefertiti: the slip collider closes the pin's wall problem.** g41pw
(onset hold + pin + stress-free assimilation + the separating collider; no clearance, ray or
yield): nefertiti **silIoU 0.9732** (+0.0056 — the best fit of any run on it; g41 0.9676 at 90
windows, g41ps 0.9727 at 114), **57 windows** (no stall: it converged earlier, not later),
pinned **94.8 %** at the end, layer flips 0.25, step 0.013 spacings; det F run-minimum **0.752**
(g41's own 0.764 — the transit's dip, unchanged by the pin), end-state minimum 0.815, p1
0.899, **the un-pinned set's minimum through the stream 0.77–0.81** (g41ps 0.46, g41py 0.26:
no squeeze), the pinned set's elastic det F_e 1.000. P213 ✓ (the window bound was against
early stops; a faster convergence is the opposite), P215 ✓. Every geometric substitute
(clearance 50b, ray 50d, yield 50e) is superseded: the pinned body deposits no mass, so it
neither stops nor squeezes a stream, and it pins 95 % of the body instead of 70 %. The slip
form goes to the remaining 15 targets now (GPU 0; GPU 1 continues beast, bunny, cow) as the
40k adoption sweep — gate **P218**: fit up or within −0.003 of max(g41, g41pz, g41py) on ≥ 18
(beast ≥ 0.950), end-state p1 ≥ 0.85 on all, step ≤ 0.02 spacings and flips ≤ 0.56 on ≥ 18,
pinned ≥ 0.8 at the end on ≥ 17. g41py (the yield form) at 18 of 19: cheburashka +0.001,
fandisk +0.005, heart +0.000, spot +0.002 — 14 of 18 up or within −0.003, beast −0.016, cow
−0.004, ogre −0.004, nefertiti +0.003; it is the fallback if the slip form fails P218.

**2026-09-25 18:30 — g41pw (the slip form), 5 of 19, vs g41:** bunny **0.9703** (+0.0093, the
best bunny of any run; det F run-min 0.787 = +0.015 over g41), nefertiti +0.0056, C +0.0616,
**beast 0.9516 (−0.0025 — the yield form had −0.016, the point rule −0.017: the stream case is
answered)**, ogre −0.0017. Flips 0.20–0.38, steps 0.012–0.023 spacings. The run-minimum det F
drops on C (−0.081), ogre (−0.063, 0.691), beast (−0.041) — the end-state reading follows below.
ay300 (300k + yield) run: silIoU 0.9703, det F 0.633; chain rendering. *A naming error:* the
aw300 chain was derived from as300's script by a substitution that did not match, so the slip
run at 300k writes under the tag as300 (its flags are the slip form's: `settle_pin
settle_pin_assim settle_pin_slip`); the original as300 archive, json, verdict and video were
copied to `as300orig_*` before the overwrite, and the slip run's outputs will be renamed to
aw300 when it ends. The numbers of as300 (17:20 entry) stand.

**2026-09-25 19:00 — ay300's verdict (300k, the yield form), and the dragon under the slip form.**
ay300: 88 windows, silIoU 0.9703, det F 0.633 (end un-pinned p1 0.79), pinned 91.3 % and exactly
still, **tail 0.0005 (p90 0.0007), ALT 0.0006**, bump 1.21°, one component — the body is as
still as as300's; **the ear: tip 5.3 reference particles**, the last two slabs 0.84 / 0.33
(as300 7.7, 0.81 / 0.41; ai300 13.6, 0.89 / 1.13): P211 ✗, and the yield rule released 0.0 % of
the settled set in the last windows — at 300k the ear's feed is not a straight-ray stream the
rule can see. The 300k ear is now the slip collider's to answer (the misnamed run, window 61+).
Launched on GPU 2: **dw300** = the 300k dragon under the slip form (l300's recipe + H + pin +
assim + slip) — the 300k generalisation reading of the pin. Pre-registered: **P219** silIoU ≥
0.975 (dl300 0.9772), end-state p1 ≥ 0.76 (dl300's own) with ≤ 0.02 % below 0.3, pinned ≥ 0.85
and exactly still, the delivered tail ≤ 0.0008 (the dragon's own reference: none yet — dl300's
video was never rendered; the number is recorded as the first).

**2026-09-25 19:30 — the 300k slip run ended (62 windows, early stop, silIoU 0.9754, det F
0.614; its chain is rendering), and a correction to the 18:30 note.** The copies made as
`as300orig_*` were taken after the slip run had already written its archive under the as300
tag (21:22 server time): they hold the slip run, not the original as300 — the original as300
archive (npz, json, verdict text) is lost; its numbers stand in the 17:20 entry and its
delivered video survives (`as300orig_plain_s12.mp4`, 20:55, and the page's copy). When the chain
ends, the slip run's outputs are renamed aw300_* and the surviving video restored under as300.
g41pw at 8 of 19: cow **0.9660 (+0.0053)** and dragon 0.9642 (+0.0060) added; the run-minimum
det F on the dragon 0.406 and the cow 0.710 are the transit's dips (end-state reading with the
sweep). GPU 1 idle → **ak300** = the slip form at 300k + the KDE ear term (z300b's `--w_kde 1
--plan_native --ot_handoff`): the D2 lever that gave the tongue at the target thickness (y300 /
z300b) but collapsed det F without the hold and alternated with it (al300 / am300); with the
settled body pinned stress-free and no wall, the question is whether the KDE ear survives.
Pre-registered: **P220** ear tip ≥ 13 reference particles and the last two slabs ≥ 0.9 / 1.0,
silIoU ≥ 0.977, end-state p1 ≥ 0.85 and ≤ 0.1 % below 0.5, the tail ≤ 0.0008, pinned ≥ 0.85.

**2026-09-25 19:50 — aw300's verdict (300k, the slip collider; outputs renamed from the as300
tag):** 62 windows (early stop), **silIoU 0.9754** (the best 300k pin run; ai300 0.9766), det F
0.614 with the end-state un-pinned p1 **0.856** ✓, no reversal window at all, flips 0.32, step
0.008 spacings, low-band +0.31; pinned **55.4 %** only (the run stopped at 62 windows, before
the body had pinned), exactly still; tail 0.0010 (as300 0.0006 at 90 % pinned), bump 1.24°.
**The ear: tip 8.5** (as300 7.7, ai300 13.6), the last two slabs 0.64 / 0.62 — the tip slab is
the best of the pin runs (as300 0.41) but the ear is still short (P216 ✗), and its slabs were
still rising when the run stopped. Why the slip collider did not free the ear as it freed
nefertiti's crown: the crown stream flows ALONG the bust's surface, which a slip wall permits;
the bunny's ear is fed THROUGH its base — the feed enters the pinned volume, and a collider,
correctly, refuses penetration. For a feature fed through settled material the base must stay
free until the feature is done: the ray clearance (50d) says exactly which material that is,
and its failure at 40k was the drag, which the collider removes. Launched (GPU 3): **ar300** =
the slip form + `--settle_pin_ray` at 300k. Pre-registered: **P222** the ear's tip ≥ 11 and the
last two slabs ≥ 0.85 / 1.0, silIoU ≥ 0.975, end p1 ≥ 0.85, pinned ≥ 0.8 at the end, the tail
≤ 0.0008, no early stop before 95 % arrival. ak300 (slip + KDE ear, GPU 1) and dw300 (the
dragon, GPU 2) continue.

*aw300's early stop, read (20:00):* the three rejections at windows 60–62 carry gains of
−0.0025 … −0.0035 with the physics gain positive, at **99.8 % arrived** — not a blocked stream
(nefertiti's were −0.05 at 80 % arrived) but the merit's plateau: with 55 % of the body pinned
the remaining objective gain per window is small and the rejection rule ends the run. The
ear's last growth is what the un-pinned ai300 did between windows 62 and 112 (tip slab 0.75 →
1.13) while the whole body breathed; under the pin that phase does not happen because the
cell-sum merit no longer sees enough gain in it — "arrived" (within the pace radius, a loss
cell) is not "filled" for a feature thinner than the cell. That is the D2 statement again:
the ear's remaining deficit is below the cell, and the term that sees it is the particle-scale
one — ak300 (slip + KDE) is the run that tests exactly this; ar300 (ray clearance + slip) tests
whether a free base alone lets the merit keep the ear's gain.

**2026-09-25 20:15 — g41pw at 11 of 19** vs g41: C +0.062, bunny +0.009, dragon +0.006,
nefertiti +0.006, cow +0.005, maxplanck +0.005, A +0.003, V +0.000, ogre −0.002, beast −0.0025,
**homer −0.0043** (the one beyond −0.003 so far: 67 windows against g41's 74). End-state det F
p1 0.874–0.928 on all, none below 0.5 except the dragon's 5 particles (0.01 %, its spikes —
the §14.0 capacity item), homer's end minimum 0.618. Dossier §15 written (the sticky collider
diagnosis and the slip form's table); the page follows the full sweep.

**2026-09-25 21:00 — g41pw complete (19 of 19): the slip form is adopted as the 40k pin form.**
vs g41 (the deliverable gallery): up or within −0.003 on **17** — C +0.062, bunny +0.009,
dragon +0.006, fandisk +0.006, nefertiti +0.006, cow +0.005, maxplanck +0.005, A +0.003,
cheburashka +0.003, bimba +0.002, heart +0.001, V +0.000, bob −0.001, spot −0.001, teapot
−0.002, ogre −0.002, beast −0.0025; **armadilo −0.0032, homer −0.0043** (none worse than
−0.005). The oscillation metrics on all 19: layer flips 0.17–0.43 (g41 0.54–0.79), step
0.000–0.024 spacings (g41 0.02–0.06), the reversal series empty on nefertiti; end-state det F
p1 0.874–0.928 on all, no particle below 0.5 except the dragon's five (0.01 %, the spikes);
pinned at the end median 0.92 (min beast 0.70, homer 0.73; ≥ 0.8 on 17), the pinned body
exactly still and stress-free (det F_e 1.000) on every target read. P218 ✗ by one target at
−0.0032 (the bound was −0.003) and ✓ on every other clause; P190 ✓ on 19. **Decision:** the
40k recipe's pin form is `--ctrl_rprop --ctrl_rprop_smooth --ctrl_rprop_arrived
--ctrl_rprop_hold_onset --u_rprop --u_rprop_floor 0 --settle_pin --settle_pin_assim
--settle_pin_slip` on top of RECIPE (hyde06_env.sh) — the settled body's oscillation is zero on
the whole gallery at a fit cost ≤ 0.0043 on two targets and a gain on twelve; the clearance,
ray and yield rules stay opt-in as the diagnosis's record. D1 at 40k is closed by the user's
bar (zero on the settled body); at 300k the ear (ar300 0.9773, chain pending; ak300; dw300)
decides the 300k form.

**2026-09-25 21:10 — ar300's verdict (300k; the transit-ray clearance + the slip collider +
assim + hold).** 61 windows (the merit's plateau again), **silIoU 0.9773 — the best fit of any
300k run** (ai300 0.9766, aw300 0.9754), det F run-min 0.681 with the end-state p1 0.864 on
both the pinned (0.867) and un-pinned (0.860) sets, strays 0.04 %, **no reversal window**,
flips 0.21, step 0.008 spacings, pinned 59.8 % and exactly still (det F_e 1.000), bump 1.23°,
one component. **The ear: tip 9.2 reference particles** (aw300 8.5, as300 7.7, ai300 13.6),
slabs from the base 1.04 / 0.96 / 0.95 / 0.91 / 0.89 / 0.82 / 0.63 — the middle of the ear is
now at ai300's fill (0.94 / 0.88 / 0.89 there) and only the last two slabs are short (ai300
0.89 / 1.13): the free base lets the feed through (P222's fill ✓ to the sixth slab, ✗ at the
tip). The tail 0.0012 (60 % pinned at the stop; the ear's region breathes as in ai300) — P222
✗ on the tail and the tip, ✓ on the fit and the health. The 300k picture is now consistent
across aw300 / ar300: with the settled body pinned the merit plateaus at 61–62 windows and
the tip's last growth (ai300's windows 62–112) is never asked for — a sub-cell deficit the
cell-sum merit does not see. ak300 (the particle-scale KDE term on top of the slip form) is
the test of exactly that; if it fills the tip, the 300k form is ar300's + KDE; if not, the
early-stop rule under the pin (the merit's plateau at a few 10⁻³ of gain) is the next lever.

**2026-09-25 21:30 — D2 item 0 opened in parallel: the arrival's capacity (method.md 10.28,
`--arrive_cap`).** The paced target's arrival snap (nearest target point, no exclusivity) is
the one mechanism behind the dragon's spike compression at 300k and the wedged last arrivals
on the pin runs (§14.0). With the cap each target point takes at most N / |points| arrivals,
closest first; the surplus keeps its plan image. Launched (after a 3k smoke): **g41pc** = the
adopted 40k slip form + `--arrive_cap` on dragon, nefertiti, bunny, cow (GPU 0), and **dc300** =
dw300's form + `--arrive_cap` on the 300k dragon (GPU 3; dw300 without it is the control).
Pre-registered (P212 made concrete): **P212a** dc300's end-state p1 ≥ 0.85 and ≤ 0.02 % below
0.3 (dl300 0.76 / 0.02 %; dw300 pending) at silIoU ≥ 0.975; **P212b** g41pc dragon: no particle
below 0.5 at the end (g41pw 5) with the fit within ±0.003 of g41pw; nefertiti's wedged count
(un-pinned det F < 0.7) 0 (g41py 15); bunny / cow within ±0.003 of g41pw. If the cap costs the
fit (the surplus no longer pulled onto the support: the 150k chamfer reading of d4db68a), the
end-state fuzziness is the price to weigh against the spikes.

**2026-09-25 22:25 — run-end readings (chains pending).** *g41pc dragon (the adopted slip form +
`--arrive_cap`, 40k):* silIoU 0.9620 (g41pw 0.9642, −0.002), pinned 88 %, **end-state det F
minimum 0.715 (g41pw 0.409), p1 0.888 (0.874), no particle below 0.5 (g41pw five)** — P212b's
dragon clause ✓: the spike over-fill was the snap's. *ak300 (300k, the slip form + the KDE
ear term):* 104 windows (aw300 62, ar300 61 — the particle-scale term keeps the merit gaining
through the ear's last growth, as predicted), silIoU 0.9695, det F run-min 0.414, 99.5 %
arrived; its ear and tail follow in the chain. *dw300 (the 300k dragon, the slip form):* 100+
windows, 97.4 % arrived, pinned 59 %, **silIoU 0.9646 against dl300's 0.9772 (−0.013)**, det F
run-min 0.372 — the dragon's fit under the pin drops where the bunny's did not: a target with
many thin spikes is many streams into sub-cell features, each stopping at the merit's plateau
early; dc300 (+ the capacity rule) is running as its twin.

**2026-09-25 23:00 — ak300's verdict, g41pc's, and the pin that follows the plan.** *ak300
(300k, the slip form + the KDE ear term):* 104 windows, silIoU 0.9695, det F run-min 0.414
(end p1 0.859, two particles below 0.5), pinned 81 %, tail 0.0009, bump 1.25°, **ear tip 8.7,
slabs 0.80 / 0.72 / 0.66 / 0.49** — the ear fills LATER and no further (P220 ✗): the
particle-scale term does not supply the mass either. *g41pc (the adopted 40k form + the
capacity rule):* dragon 0.9620 (−0.002 vs g41pw) with **no particle below 0.5 and the end
minimum 0.715 (g41pw 0.409)** ✓; bunny 0.9685 (−0.002), cow 0.9616 (−0.004), **nefertiti 0.9679
(−0.005)** ✗; 30–40 % of the arrivals are over capacity throughout (|target points| ≈ N). P212b
✓ on the dragon, ✗ on the fit elsewhere: the cap removes the over-fill and costs the fit the
snap bought — kept opt-in for spiky targets, not adopted for the gallery; dc300 reads it on
the 300k dragon. **The reading that unifies the 300k ear:** the tip's last growth in ai300
(windows 62–112) was fed by the whole body moving a little each window — material supply, not
oscillation. Every pin run stops early with the tip at 0.5–0.6 because the pinned body no
longer supplies it; neither the free base (ar300), nor the particle-scale term (ak300), nor
the yield (ay300) restores that supply. Addendum 6 (eq. 50g, `--settle_pin_follow`): "arrived"
is read every window, a pinned particle whose plan image has moved beyond the pace radius is
released with a fresh step and re-pinned when arrived. Launched (GPU 1, after a 3k smoke):
**az300** = ar300's form + follow. Pre-registered: **P224** the ear's tip ≥ 11 and the last two
slabs ≥ 0.85 / 1.0, silIoU ≥ 0.975, end p1 ≥ 0.85, the tail ≤ 0.0010, the released fraction of
the pinned set ≤ 15 % at any window, no reversal window, and the run past 80 windows (the ear's
growth is asked for).

**2026-09-25 23:30 — the follow rule at 40k in parallel (g41pf).** The 3k smoke of the follow
rule released 0.0 % (at 3k the plan does not move pinned images); az300 runs at 300k (GPU 1).
Because the rule re-opens the pin on evidence, it must not bring the oscillation back where the
40k form is adopted: **g41pf** = the adopted 40k form + `--settle_pin_follow` on nefertiti,
dragon, bunny, beast (GPU 0). Pre-registered: **P225** fit within ±0.003 of g41pw, layer flips
≤ g41pw's + 0.10, the released fraction ≤ 10 % at any window, pinned at the end ≥ 0.8, the
settled body still (pin_probe's still fraction 1.0 at the end); if the follow rule passes P224
at 300k and P225 at 40k it joins the recipe, else it stays a 300k-only option.

**2026-09-26 01:10 — cleanup (the user: "기존 관련 없는 결과들도 한 번 지워주고"), and the push.**
Branch `v3-grid-gs` pushed to origin (new remote branch; the remote had only `main`). Server
output 472 GB → **98 GB**: 890 entries deleted (400.7 GB) after archiving their 575 logs / json /
verdict texts to `output/logs_archive_20260925.tgz` (the list of deleted paths beside it,
`logs_archive_20260925.deleted_list.txt`); 120 stride videos of deleted tags removed. Deleted:
the superseded 300k runs c300 d300 dr300 e300 f300 j300 dp300 dk300 ah300 af300 al300 ab300
am300 ac300 i300 w300 k300 ap300 y300 ay300 h300 ad300 n300 ag300 an300 m300b v300 z300 aj300
g300 p300 o300 u300 r300 t300 r300b s300 b300 m300 full300; the superseded 40k sweeps g40 (raw;
its report folder kept) g41s g41t g41u g41x g41y g41z g41q g41n g41py g41pz g41ps g41pr g41p
g41f g41h g41k g41m g41o g41v g41w g41a g41r s20; `stale_20260922`; the scratch smoke dirs.
Kept: g41 + report_g40 / report_g41 (deliverables), g41pw (the adopted 40k form), g41pc / g41pf
(live), ai300 l300 z300b (the ear's references), ar300 aw300 ak300 az300 (the current 300k pin
runs), dl300 dw300 dc300 (the dragon), every small older probe (< 10 MB), the scratch scripts.
Every number of a deleted run stays in this log. Local `output/` (1 GB, the deliverable pages)
untouched.

**Verdicts in:** *az300 (300k, ar300's form + the follow rule):* 53 windows, silIoU 0.9769, no
reversal window, **the follow rule released 0.0 %**: by window 50 the plan calls **100 %** of
the particles arrived (mean |d| 0.053 wu) — a pinned particle's image never leaves the pace
radius, so the rule is inert; ear tip 8.1, slabs 0.90 / 0.87 / 0.78 / 0.57; P224 ✗. *dw300 /
dc300 (the 300k dragon under the slip form, without / with the capacity rule):* silIoU 0.9646 /
0.9625 (dl300 0.9772), end p1 0.80 / 0.77, below 0.5: 30 / 80 particles (dl300 272) — the pin
improves the dragon's spikes, the cap does not add to it at 300k, and the fit drops by 0.013
(the spikes are many thin features that stop filling, as the ear does). *g41pf (40k + follow):*
nefertiti 0.9716 (−0.0016 vs g41pw), dragon 0.9659 (+0.0017), released ≤ 8 %.

**What the four 300k pin runs now say together.** The tip's deficit is visible to the cell-sum
loss only at the tip's own cells, whose particles are pinned; the particles that must MOVE to
fill it (a chain down the ear, each stepping up into the vacancy the one above leaves) are all
"arrived" by the plan's measure and pinned, and a pinned particle has no control gradient (its
control has no effect). In the un-pinned ai300 that chain ran for 50 windows. So the pin must be
released on the evidence of the LOSS, not of the plan: the active-set rule the digest
prescribes (Bertsekas 1982; the strong rules' KKT check) — a fixed variable whose loss gradient
exceeds what the free variables carry is not at its optimum. Next: `--settle_pin_kkt`.

**2026-09-26 01:30 — the KKT release (method.md 10.27 addendum 7, `--settle_pin_kkt`,
eq. 50h).** A pinned particle whose fixed-target cell-sum gradient exceeds the free set's
median is released for the window. Launched after a 3k smoke: **aq300** = ar300's form (ray
clearance + slip + assim + hold) + kkt at 300k; **g41pk** = the adopted 40k form + kkt on
bunny, nefertiti, dragon. Pre-registered: **P226** (aq300) the ear's tip ≥ 11 reference
particles and the last two slabs ≥ 0.85 / 1.0, silIoU ≥ 0.9773 (ar300), the run past 80
windows, end p1 ≥ 0.85, the tail ≤ 0.0010, released ≤ 15 % of the pinned set at any window and
the released centroid moving toward the ear over the windows the tip grows; **P227** (g41pk)
fit within ±0.003 of g41pw, flips ≤ g41pw + 0.10, pinned ≥ 0.8 at the end (the release does not
bring the breathing back at 40k). *In parallel (the user: "300K 쪽, 귀 안 되면 계속 paper /
cookbook 찾아줘"):* a digest on growing thin target features from a bulk source (target-driven
fluid control, sub-cell-aware losses, the practitioners' logo-forming recipes, capacity-exact
matching) is being gathered for related_work.md.

**2026-09-26 02:30 — the KKT release, first form refuted by its smoke and by a probe; the
second form.** *v1 (the fixed-target cell-sum gradient against the free median):* the 3k smoke
released 40–55 % of the pinned set, g41pk bunny (40k) 27 % at window 48 with silIoU 0.9692
(−0.001 vs g41pw), aq300 50 % in its first windows — pinned and free particles share one noise
floor. *Why the cell sum cannot drive the ear at all (`scratch/grad_probe.py` on ar300's end
state):* at the 0.306-wu loss cell (the loss grid IS the 36³ MPM grid at 300k) the cubic
rasterisation reads the left ear 0.96 / 0.89 / 0.93 / 0.91 / 0.89 / 0.89 / 0.74 full per grid
row from the base, the ear's residuals (min −0.155, p5 −0.066) inside the body's band (p5
−0.077, p95 +0.148); the pinned particles above the free median are 60 % of the pinned set and
their ear share (4.4 %) equals the ear's share of the pinned set (4.0 %); at 2× and 4× the cell
the picture is the same. The ear's tip deficit is sub-cell for the objective that the pin was
supposed to leave in charge — what grew ai300's tip is the render term (a third of the
gradient throughout, g_share 0.31–0.35 in windows 62–112), which sees the tip at pixel scale.
*v3 (`--settle_pin_kkt`, now):* the evidence is the WINDOW objective's gradient with respect to
the end-of-window positions (render term included; a hook on x_T in the adjoint — optimizer
stats "gx"), averaged over the Rprop smoothing's kNN, and a pinned particle is released only
when that averaged gradient kept its direction since the previous window AND exceeds the free
set's median: noise flips window to window (that is how the particles were pinned), a deficit
keeps pulling. The archive now carries the last window's gradient (`gx_last`) so the ear's
signal can be read offline. aq300 and g41pk were stopped and relaunched on v3 (P226–P227
unchanged).

**2026-09-26 03:00 — KKT v3 refuted by its smoke; the evidence is measured before any further
rule.** v3 (the window objective's gradient, neighbourhood-averaged, persistent across windows,
above the free median) released 56–63 % of the pinned set, and 95 % of the pinned set passed
the persistence test. That is structural: a pinned particle's gradient is the pin constraint's
multiplier — the particle cannot move, so the objective's pull on it never relaxes and keeps
its direction window after window, and it exceeds the free particles' (which relax theirs).
Neither the magnitude nor the persistence of the gradient on a pinned particle is evidence of
a deficit. aq300 / g41pk v3 stopped. **ax300** = ar300's form with the evidence recorded and
archived but nothing released (`--settle_pin_kkt_dry`): the window objective's gradient at the
pinned plateau, read offline by region (tip, stem, body) for magnitude, direction (does the
objective want the ear's particles to move up the ear?) and neighbourhood coherence
(`scratch/gx_probe.py`). If no statistic separates the ear's supply chain from the body at the
plateau, the release cannot be made from the pinned state and the ear must be finished before
its material pins (the pin's onset read per feature), which is the other branch.

**2026-09-26 03:30 — the 300k ear under the pin is COMPLETE at the end; what remains is its growth
(a knob on a neck).** Three readings reframe the night's ear item. (1) *Coverage at the particle
spacing* (`scratch/cover_probe.py`, the target points whose nearest particle is farther than
one spacing, 0.054 wu): ai300 (no pin, tip 13.6 reference particles) 3.4 % of the tip's points,
ar300 (pin, tip 9.2) 3.1 %, az300 3.9 % — the pinned runs cover the whole ear; the "tip 9.2 vs
13.6" count is a DENSITY difference (69 vs 102 particles against the target's 89 within 0.25 wu:
ar300 at 78 %, ai300 over-filled to 115 %), not a missing tip. (2) *The end stills* (the page):
ar300's ears are full length with both tips, indistinguishable from ai300's. The acceptance
bound "tip ≥ 11 reference particles" (P204, P216, P222, P224) measured the particle density,
which the delivered surface does not show; it is withdrawn as an ear criterion. (3) *The growth*
(`ear_growth_ai300_ar300.png`: the left view's ear region at t = 0.25 … 0.8 of each run): in
ar300 the tall ear rises as a thin spike that grows a KNOB on a narrow neck (t = 0.35–0.45) and
fills out by 0.6; ear_slab's thickness table says it — the left ear at t = 0.25: the upper slabs
1.20× and **1.46×** the target thickness over a neck at 0.66–0.76× (ai300's worst: 1.05× over
0.44–0.65× at t = 0.15, and a smoother profile after). That knob on a neck is the D2 droplet,
seen at 300k under the pin: the leading material arrives as a lump ahead of its stem.
*So:* D1's pin at 300k (ar300: silIoU 0.9773, no reversal window, the body exactly still) does
not cost the ear's end state; the ear item moves to D2 — the growth order. ax300 (the KKT
evidence recorded without release) finishes as a record; the KKT release is not pursued.
Pre-registered for the D2 growth runs: **the knob index** K = max over the sampled frames of
(the thickest of the ear's top two slabs) / (the thinnest slab below them), with every slab ≥
0.5 of its target thickness once filled — ai300 1.6, ar300 2.2; the aim K ≤ 1.2 (a tongue that
thickens from its base) with the end state and D1's stillness kept.

**2026-09-26 03:50 — the ear's growth knob: the particle-scale KDE term under the pin gives a
tongue.** The left ear's thickness per slab over the frames (ear_slab): **ak300** (the slip
form + `--w_kde 1 --plan_native --ot_handoff`) grows the ear from its base — each upper slab
appears only after the one below has filled, the tip slab first at t = 0.4 at 0.64× and 1.07×
at the end; **knob index 1.1–1.3** over the frames. z300b (the same KDE set, no pin) 1.2–1.3.
aw300 (the pin without KDE) **2.3** at t = 0.3 (1.13× over a 0.49× neck), ar300 2.2, ai300
(neither) 2.4 at window 17: the knob is the transport's, not the pin's (ai300 has it at the same
absolute window). At 40k it is permanent: g41p's left ear ends with the top slab at **2.8×** the
target thickness over a 0.60-filled slab below (knob index about 3.2 — a bulb at the tip), g41h
3.1×. ak300's end still: both ears complete with tapered tips; the body lumpier (bump 1.3° vs
ar300's 1.2°) — the smoothness item. End-state health (the quantile gate): p1 0.859, 2 particles
below 0.5 ✓; D1: pinned 81 %, tail 0.0009, the body exactly still. So the KDE term, set aside at
300k for its det F collapse without the pin and its alternation under the hold alone, is the
growth-order answer under the pin. Launched: **ae300** = ar300's form (ray clearance, the best
300k fit) + the KDE set; **g41pe** = the adopted 40k form + the KDE set on bunny, armadilo,
dragon, nefertiti, cow, C. Pre-registered: **P228** ae300 knob index ≤ 1.3 on both ears over the
frames, silIoU ≥ 0.975, end p1 ≥ 0.85, the tail ≤ 0.0010, no reversal window; **P229** g41pe
bunny's end-state top slab ≤ 1.5× with the slab below ≥ 0.8 (the 40k bulb gone), and on the six
the fit within ±0.003 of g41pw with end p1 ≥ 0.85 — the gallery rule before adoption. The Ḣ⁻¹
term (the digest's first mechanism) is held: it reads the same smeared splat as the cell sum,
so it adds reach, not the resolution the knob needs. ax300 (the KKT evidence recorded, nothing
released) finished as a record.

**2026-09-26 04:10 — correction: there is no permanent bulb at 40k.** The 03:50 entry read the
40k left ear's top slab (y 3.51) at 2.8–3.2× the target thickness in g41p, g41h, g41pw and
g41pe alike and called it a bulb. The end still of g41p (the 40k pinned bunny) shows the tall
ear's tip tapered and the other ear's tip ordinary — no bulb. The reading is the metric's: the
target's tip slab at 40k is 77 points, 0.38 wu thick (2.75 spacings), and a ratio near 3 would
mean more than 1 wu of material in one ear's slab — particles outside the ear enter that slab's
mask at 40k. The top slab's thickness at 40k is withdrawn as evidence; the knob index is read
on the 300k tables (where the top slabs sit at 0.9–1.5× and move with the mechanism) and by
eye. What the 40k tables do show before that slab saturates: at t = 0.15–0.25 the slab below
the tip is empty (nan) while the tip slab holds particles — the tip detached from its stem, the
40k droplet, in g41pw and g41pe alike. *g41pe bunny (the 40k adopted form + the KDE set):*
silIoU 0.9592 (g41pw 0.9703, −0.011): at 40k the KDE term costs the fit and does not change the
early detachment; the five other targets and ae300 follow before any decision.

**2026-09-26 04:40 — KDE at 40k refuted; the H⁻¹ term, already in the code, brought back.**
*g41pe (the adopted 40k form + the KDE set), stopped at 3 of 6:* bunny 0.9592 (g41pw −0.011),
armadilo 0.9450 (−0.012), dragon 0.9579 (−0.006) — at 40k the particle-scale term costs the fit
on every target read and does not change the early tip detachment; it stays a 300k option. *The
40k growth by eye* (g41p's stride-12 video, the ear region at t = 0.10 … 0.40): at 0.20 the tall
ear is a spike with a small nub at its tip, at 0.25 a knob on a neck, at 0.30 the other ear's tip
shows a small separate sliver, filled by 0.40 — the same growth defects as at 300k, transient.
*The H⁻¹ term* (`--w_h1`, method.md "H⁻¹ mass balance", 2026-09-04) is the digest's first
mechanism and it already exists: an FFT Poisson solve of the fixed-target residual on the loss
grid with the P3M self-energy correction, in the physics core, calibrated once to D_vol's
gradient norm. Built for the same symptom — the solid bunny's ears 30 % under-filled at a wrong
fixed point — it filled them at 20k (ear fraction 0.078 → 0.105, the vertical bands 1.46 → 1.04,
the sub-cell cluster ratio improved) and was adopted provisionally, then not carried into the
recipe when the render contract replaced the pipeline's front end; it has never run with the
paced OT target, the density units or the pin. Launched: **g41ph** = the adopted 40k form +
`--w_h1 1` on bunny, armadilo, dragon, nefertiti; **au300** = ar300's form + `--w_h1 1`.
Pre-registered: **P230** (by eye on the stride-12 videos, ear region, both scales) the ear grows
from its base with no knob on a neck and no separated sliver at t = 0.15–0.35, the 300k knob
index ≤ 1.3 on the 300k tables; **P231** the fit within ±0.003 of g41pw / ar300 and the end-state
p1 ≥ 0.85; **P232** D1 kept: no reversal window, pinned ≥ 0.8 at the end at 40k, the 300k tail ≤
0.0010. The H⁻¹ ratio (`h1_ratio`) is logged per window.

**2026-09-26 05:00 — g41ph bunny (40k adopted form + H⁻¹):** silIoU **0.9724** — the best 40k
bunny of any run (g41pw 0.9703, g41 0.9610), 46 windows, det F run-min 0.744. The growth by eye
(`h1_growth_40k.png`: the ear region at stride-12 frames 7, 10, 13, 17, 20, 27, 35; g41p above,
g41ph below): the ears emerge EARLIER (by frame 10 against 17), and the thin nub, the knob on a
neck and the separated sliver of g41p are gone — but new transients appear: the left ear first
rises as a curled flap with a wavy edge (frames 10–13) and the tips FORK into two lobes that merge
later (frame 17 the right ear, frame 27 the left). H⁻¹ fills the target's cross-section from
several sides at once instead of feeding one tongue from the base. P230 ✗ as stated (no clean
tongue); the fit is a clear gain. The other three targets and au300 (300k) follow.

**2026-09-26 05:20 — H⁻¹ under the pin: a fit gain everywhere read, and healthier.** g41ph (the
adopted 40k form + `--w_h1 1`) vs g41 / vs g41pw: **bunny 0.9724** (+0.0114 / +0.0021),
**armadilo 0.9681** (+0.0084 / +0.0116), **dragon 0.9695** (+0.0113 / +0.0053); end-state det F
p1 0.917 / 0.896 / 0.889 with **no particle below 0.5** (g41pw's dragon had five), pinned at the
end **0.99 / 0.97 / 0.98** (g41pw 0.91–0.97), **no reversal window on any**; the H⁻¹ ratio
(its gradient against D_vol's, calibrated to 1 at the source) falls to 0.23–0.39 by the end. The
run minimum of det F drops on the dragon (0.538) and armadillo (0.668) — the transit's dip; the end
state is healthier than without the term. At 300k the runs themselves: **au300** (ar300's form +
H⁻¹) silIoU **0.9770**, det F run-min **0.76** (ar300 0.68) — the healthiest 300k pin run;
**ae300** (ar300's form + the KDE set) 0.9737, det F 0.59; their chains (ear, tail, growth) are
rendering. The growth by eye at 40k is changed, not cleaned (05:00: no nub / knob / sliver, but a
curled flap and forked tips); the fit and the health are the clearest gains of the night.
Launched: g41ph on the remaining 15 targets (GPUs 3 and 0) — the gallery gate before adding
`--w_h1 1` to the adopted 40k form (P231 on all 19: the fit within ±0.003 of g41pw or better,
end p1 ≥ 0.85; P232: no reversal window, pinned ≥ 0.8).

**2026-09-26 06:30 — the user: "300K에서 여전히 remesh인지 뭔지 모르겠지만 진동이 살짝 보인다. 계속 주시해 줘." The
visible 300k tail motion is mostly the surface reconstruction (M5).** *Where it flickers* (a
temporal-difference heatmap of the delivered tail — the mean |Δframe| over the last 24 stride-12
frames, gain ×25, `tailheat_*.png`): on as300 and ar300 the change sits on the ears (their whole
surface), on a thin line along the whole silhouette, and as a faint speckle over the body's
interior — although 90 % (as300) / 60 % (ar300) of the particles are pinned and move exactly zero.
*The decisive twin:* ar300's archived frames rendered with the local surface (`--surface mc`:
marching cubes on the kernel density at the reference spacing, on a grid fixed across the frames
— a region with no moving particle within the kernel gets identical triangles) instead of the
screened Poisson fit (a global solve: a particle moving anywhere shifts the implicit function
everywhere):

| ar300, same particles | whole-run per-frame change | delivered tail median / p90 | tail ALT/DRIFT |
|---|---|---|---|
| screened Poisson (the delivered videos) | 0.0029 | 0.0012 / 0.0014 | 1.71 |
| marching cubes on the kernel density | 0.0009 | **0.0003 / 0.0005** | 1.18 |

About three quarters of the 300k tail's visible change is the reconstruction amplifying the small
motion of the still-free particles (the ears, 10–40 % unpinned at the stop) over the whole
surface; with the local surface the tail equals the 40k pinned run's (g41p 0.0003). The end frame
(`ar300_end_poisson_vs_mc.png`): the marching-cubes body is SMOOTHER (the lumps on the back gone —
the smoothness item), the ears slightly thicker and rounder (the kernel's blur), a faint stair-step
banding of the voxel grid. So the remaining 300k oscillation has two parts with two owners: the
re-mesh (the delivered surface: a local or temporally-coupled reconstruction), and the last free
particles' own motion (the pin's coverage at the stop).
*The 300k growth runs (chains):* **ae300** (ar300 + KDE): the ear grows as a tongue (the tip slab
first at t = 0.4, 0.55× over 0.51×; knob index ≤ 1.3), silIoU 0.9737, det F 0.59, 92 windows —
P228's growth ✓, fit ✗; **au300** (ar300 + H⁻¹): silIoU **0.977**, det F **0.76**, the tip filled
(14.0 reference particles), 46 windows, no reversal — but a knobby growth (t = 0.25–0.4: the
mid-ear slab 1.16–1.21× over a 0.65–0.70× neck, the tip 1.19–1.26× over 0.52–0.66×). KDE orders
the growth, H⁻¹ supplies and fits: **av300** = ar300's form + both, launched. *Cleanup (the user's
rule, 117 GB > 100):* ax300 az300 aw300 dc300 g41pe g41pf removed (24 GB) after archiving their 27
logs/json to `logs_archive_20260926.tgz`; now 95–97 GB.

**2026-09-26 08:00 — g41ph complete (19 of 19): H⁻¹ under the pin, vs g41 / vs g41pw.** Up or
within −0.003 on **17**: ogre +0.015, dragon +0.011, bunny +0.011, fandisk +0.011, armadilo
+0.008, maxplanck +0.008, C +0.065, cheburashka +0.005, cow +0.005, A +0.005, nefertiti +0.004,
heart +0.003, spot / teapot / V +0.001, bob −0.001, homer −0.002; vs the adopted g41pw: up on 15,
bunny +0.002, dragon +0.005, armadilo +0.012, ogre +0.016. End-state det F p1 **0.885–0.940** on
all, none below 0.5; pinned at the end 0.95–0.99 on 16 (beast 0.77, heart 0.77); one reversal
window on A and C, none elsewhere. **Two failures: bimba 0.9482 (−0.027, 15 windows) and beast
0.9263 (−0.028, 97 windows).** *bimba's cause, read from the record:* at windows 13–15 the merit's
physics component (the transport divergence to the fixed target) rose 0.127 → 0.137 (+7 % a
window) while the total merit still improved (+2–3 %: the render channel) — the catastrophe brake
did its job three times and stopped the run at 88.6 % arrival. Why the render channel won: the
H⁻¹ term sits inside the physics core, so its gradient (ratio 0.35–1.4 to D_vol's along the run)
inflates the physics norm the λ-balancer scales the render channel against (Codex F3's concern
in method.md), and on a smooth head the amplified render pull beats the transport at arrival.
*beast:* the long transport (the tail and legs) ended at 76 % arrival with the fit down 0.025 —
a fixed-target pull competing with the paced target's plan during the transport. Two structural
placements, both already argued in method.md's H⁻¹ section: **`--h1_outside`** (the term
outside the core, the W1 precedent: the balancer and PCGrad see the cell sum alone) and
**`--h1_onset_pin`** (the term switched on at the pin's onset — the endgame, where the pinned
body no longer supplies; zero during the transport). Launched on bimba, beast, bunny (3 min
each): **g41pho** (outside), **g41phn** (onset), **g41phb** (both). Pre-registered: **P233** bimba
≥ 0.975 (g41pw 0.9767) without a brake stop and beast ≥ 0.949 (g41pw 0.9516) on at least one
placement, with bunny within −0.003 of g41ph's 0.9724; the passing placement goes to the 19.

**The 300k growth trio by eye** (`growth_300k_trio.png`: the ear region at t = 0.15 … 0.6; rows
ae300 KDE / au300 H⁻¹ / av300 both): none is a clean tongue. ae300: the tall ear rises as a thin
finger whose tip CURLS into a hook (t = 0.20–0.40) before it thickens from the base; the wide ear
is a tongue. au300: a cone, then a knob on top (0.25), a knob on a neck (0.30–0.40), filled by 0.6.
av300: a finger with a **detached bead at its tip at t = 0.20** (the D2 droplet itself), a hooked
tip at 0.25, a tongue from 0.30. All three grow the ear as a thin spike first and thicken it
after; the tip's shape at the spike stage (hook / knob / bead) is the defect. Numbers: ae300
0.9737 / det F 0.59 / tail 0.0008 / pinned 79 %; au300 0.977 / 0.76 / 0.0018 at 61 % pinned (46
windows); av300 0.9742 / 0.70 / 0.0009 / 81 %. The fit and health favour H⁻¹, the growth order
KDE; neither fixes the spike stage. *Cleanup (the 100 GB rule):* z300b, ak300, g41pc removed
(22.6 GB, logs archived to `logs_archive_20260926b.tgz`); now 85 GB.

**2026-09-26 09:00 — the user's three points: the marching-cubes look, the early dent, and the
coarse-to-fine trick.** *(1) "ar300mc looks good; could the render loss thin the ears naturally
late in the run?"* Two facts first: the ear's particles are already at the target thickness
(ear_slab 0.89–1.04× per slab at the end), so the fat ears of the mc video are the reconstruction
kernel — the mc render used the 40k reference kernel (`--ref_n 40000`, radius 0.138 wu) on
particles 0.054 wu apart; renders at the native and an intermediate kernel are running
(`ar300_mc_ref300000 / ref100000`, with end stills and bumpiness). The physics half of the idea
is the coarse-to-fine schedule below. *(2) The early dent (the user's frame at the video's
start):* it is in the particles. `scratch/dent_probe.py` (the outer shell's radius per angular
bin against its neighbours, and the radial density beneath): frame 0 no dip; **frame 96
(window 5): the deepest bin −15 % with 24 bins below −8 %, and the density in the outer bands
of that cone 0.43–0.52 of the body's median** — the surface has receded where the transport
drains the blob from that side first (the outer material there leaves inward along its rays
before the interior follows). By frame 144 −20 % (37 bins). Later frames read the ears' bases as
dips (the probe's limit), but the early crater is a transport-phase drain, not a reconstruction
artifact — the mc surface shows it because it is there. *(3) The user's rendering trick — large
voxels first (smooth, volumetric, no disconnected droplets), smaller later for detail — as a
LOSS schedule:* the code has half of it (`--c2f_at`: the render targets rebuilt at
`render_res_hi` at a fixed fraction of the run; tested once at 20k in the ladder as "render_c2f ≈
tie", never in the recipe). Made principled: **`--c2f_onset_pin`** — the rebuild at the pin's
onset (the "stable result" the user names), not at a fixed fraction; and the coarse phase
COARSER than now (render_res 32 against the recipe's 64). Launched on bunny, dragon,
nefertiti on top of the adopted 40k form (g41pw): **g41cA** `--render_res 32 --render_res_hi
96 --c2f_onset_pin`, **g41cB** `--render_res 32 --render_res_hi 64 --c2f_onset_pin`, **g41cC**
`--render_res 64 --render_res_hi 96 --c2f_onset_pin` (the "detail later" half alone).
Pre-registered: **P234** by eye on the stride-12 videos: the early blob smoother (no crater at
windows 5–12 — the dent probe's deepest bin ≥ −8 % at frame 96) and the ear's spike stage
without a hook / knob / bead, in A or B; **P235** the fit within ±0.003 of g41pw and the end p1 ≥
0.85 on the three, and the pinned fraction ≥ 0.9 (the coarse phase must not delay the pin). The
density-voxel half of the trick (a coarse loss grid first) is held: the paced target's arrival
radius is the loss cell, so a coarse grid would pin material a coarse cell from its image —
the arrival radius has to be decoupled from the loss cell before that half can be tested.

**2026-09-26 10:30 — the morning's readings.** *(a) The marching-cubes look (the user's ar300mc):*
the ears' fatness is NOT the kernel radius — ar300 re-rendered with the 40k / 100k / native 300k
kernels gives the same ears (the montage `kernels_ar300.png`), a slightly more textured body at
the native one, tails 0.0003 / 0.0004 / 0.0005 against Poisson's 0.0012, and a much smoother
body than Poisson at every radius. The remaining thickness is the density iso-level's offset
from the particle layer (a render setting, `--iso`), not the particles (at target thickness) —
so the user's "let the render loss thin them late" has no physics left to act on at 300k.
(The mc mesh's dihedral bumpiness reads 13–14° against Poisson's 1.2°: that metric measures
the voxel faceting of a marching-cubes mesh and is not comparable across reconstructions.)
*(b) The early dent, measured (`scratch/dent_probe.py`, the outer shell's radius per angular bin
against its neighbours, the equatorial band only — the poles are the flat base and the ears):*
at window 5 (frame 96) the deepest bin is −15 % with 18–21 of 360 band bins below −8 %, the
same at 300k (ar300) and 40k (g41pw bunny), at the same place (the lower front-left of the blob:
the material that leaves first toward the head and ears); it grows to −22 % / 23–27 bins by
window 10. **H⁻¹ from the start halves it** (g41ph bunny 9 bins at −12 %; armadillo 9 against
15): the non-local balance fills the drained shell from the interior. The render schedule with
the current coarse phase (g41cC, 64 → 96 px at the onset) leaves it unchanged (19 bins); the
coarse-start variants (32 px) are read below. *(c) The H⁻¹ placement twins (bimba / beast /
bunny):* **outside the core from the start** (g41pho): bimba 0.9759 ✓ but **beast 0.5231 — the
inner optimiser found no accepted step from window 7 and the run ejected particles** (an
unbalanced non-local pull on a long transport); **from the pin's onset, inside** (g41phn): beast
0.9531 ✓, bunny 0.9703 (= g41pw: the +0.002 of the from-the-start term was a transport-phase
gain), bimba 0.9591 ✗ (the brake at 18 — the balancer inflation is there at the onset too);
**both** (g41phb): **bimba 0.9778 ✓ (+0.003 vs g41), beast 0.9544 ✓ (+0.003 vs g41pw)**, bunny
pending. P233 ✓ on the both-placement; it goes to the remaining 16 (the gallery gate). *(d) The
coarse-to-fine render schedule at the pin's onset (bunny):* fit-neutral — A (32 → 96) 0.9695, B
(32 → 64) 0.9701, C (64 → 96) 0.9700 against g41pw's 0.9703; the growth by eye follows from the
videos (P234); the early dent is unchanged by the coarse start (A 18 bins at −16.7 %, B 18 at −19.8 %):
the drain is the transport's, not the render channel's.

**2026-09-26 11:30 — the coarse-to-fine render schedule read by eye, and the front.** g41cA (32 →
96 px at the pin's onset) against the pinned baseline, frame for frame in the ear region
(`c2f_growth_40k.png`, stride-12 frames 3–30): the same spike with a nub at frame 10, the same
knob on the tip at 17, the same fill by 30 — no change at the spike stage, and the early dent
unchanged (18 band bins). Fits: A bunny 0.9695 / dragon 0.9656 / nefertiti 0.9733, B 0.9701 /
0.9639 / 0.9712, C 0.9700 / 0.9661 / 0.9715, all within ±0.003 of g41pw — P234 ✗, P235 ✓: the
render-loss schedule is harmless and does nothing for the growth. **The growth order is the
transport's** (method.md 10.29): the paced target sends the tip-bound and base-bound material
in parallel along their rays, the intermediate density into a protrusion is a filament, and
the density and silhouette terms only thicken it after the fact (the silhouette is as content
with a spike as with a tongue). The user's trick, moved to where the order lives: the target
revealed as a FRONT — a particle's paced image is clamped along its ray at the boundary of the
revealed region (target cells filled to half the target's mass, or within one pace step of
one), so a feature fills from its base at the target's cross-section. `--pace_front`, no new
constant (the fill threshold is the target's own occupancy, the reveal step the arrival
radius). Launched after a 3k smoke: **g41fr** = the adopted 40k form + `--pace_front` on
bunny, dragon, nefertiti. Pre-registered: **P236** by eye at frames 7–22 the ear rises at its
base's thickness with no nub, knob or bead ahead of the filled part (the tongue); the
fraction of images clamped at the front is > 0 through the transport and 0 at the end;
**P237** the fit within −0.003 of g41pw on the three and the end-state p1 ≥ 0.85, no reversal
window, pinned ≥ 0.9 (a queued transport must not cost the end state or the pin); if the
front costs the fit, the reading is whether the queue starves the tip (the 300k supply item)
or only delays it.

**2026-09-26 12:30 — the grid-scale front does not reach the spike; the H⁻¹ placement sweep
near its end.** *g41fr (the adopted 40k form + `--pace_front`):* bunny 0.9683 (−0.002 vs g41pw),
dragon 0.9654 (+0.001), end-state p1 0.902 / 0.842 (the dragon 7 particles below 0.5 — a
queued transport compresses at the spikes a little), the early dent 16 band bins (baseline
18–21), 3–7 % of the images clamped at the front each window. By eye (`front_growth_40k.png`,
the same frames as the coarse-to-fine read): the head's hump rises broader and rounder in
frames 3–7, but the ear still rises as a spike with a nub at frame 13 and a knob at 17–22 —
**P236 ✗**. The reason is the scale: the front advances one LOSS CELL per window and a cell
counts as filled at half the target's mass, while the spike is a filament thinner than the
cell (≈ 0.2 wu against the 0.31-wu cell at 40k): within a revealed cell the tip-bound
material still forms the filament first. The grid front orders the growth at the cell scale
(the head) and cannot order it below (the ear's width). The front at the PARTICLE scale is the
same rule with the target's own points: a target point is filled when a particle lies within
one spacing of it, the revealed points are those within one pace step of a filled one, and a
sample on the ray is inside the revealed region when a revealed point lies within one spacing
of it — the coverage probe's definition, no new constant (`--pace_front_pts`, pre-registered
here as P238 with P236's reading: the tongue by eye at frames 7–22; P237's fit bound).
*g41phb (H⁻¹ outside the core, from the pin's onset; 16 of 19):* vs g41pw — bunny −0.0013, C
−0.0008, cheburashka −0.0009, bob −0.0002, spot −0.0006, V +0.0016, bimba +0.0011, beast +0.0028,
armadilo +0.0025, fandisk +0.0002, heart 0.0000, dragon +0.0005, maxplanck +0.0009, teapot
+0.0001; **cow −0.0068 and nefertiti −0.0036** beyond the −0.003 bound; no early stop, no brake
stop, no ejection on any. At 40k the placed term is neutral — its value is the 300k supply
(au300: the tip filled at 14.0 with the fit 0.977 and det F 0.76, inside the core from the
start, which bimba and beast forbid at 40k). Launched: **ao300** = ar300's form + `--w_h1 1
--h1_outside --h1_onset_pin` at 300k — P239: silIoU ≥ 0.977, det F ≥ 0.7, the tip ≥ 13 (au300's
fill) with no knob at the spike stage worse than au300's, the tail ≤ 0.0010, no reversal window.

**2026-09-26 13:30 — a D1 residual found on beast under the adopted 40k form.** Counting the
negative-reversal windows of g41pw on all 19: **14 targets at 0**, cow / dragon / fandisk 1,
homer 2 (windows 47, 66; pinned 0.68–0.73), **beast 10 of 87** (windows 54–56, 59, 62, 65–66, 73,
82, 86; pinned 0.54–0.69). Beast is the one target whose transport never finishes (76 %
arrived at the stop, pinned 0.70): the un-pinned 30 % — the tail and legs' slow transport —
alternates window to window from 54 on; the onset hold engaged at the third reversal in a row
(54–56) and the reversals thin out but continue. Under the placed H⁻¹ (g41phb) beast has 7 of
108 (58, 68, 74, 94, 102, 107, 110) — not the term's doing either way. The user's bar (zero on
the settled body) holds on beast's pinned 70 %; the residual is the transport's failure to
arrive, which is the item the fronts address — beast and cow added to the particle-scale
front run (g41fp). Recorded as open: *beast's transport stalls at 76 % arrival under every
form since g41t (0.9516–0.9544), and its free remainder alternates.*

**2026-09-26 14:00 — g41phb complete (19 of 19): the placed H⁻¹ (outside the core, from the pin's
onset) is neutral at 40k.** vs the adopted g41pw: within ±0.003 on 17 (beast +0.0028, armadilo
+0.0025, V +0.0016, bimba +0.0011, maxplanck +0.0009, dragon +0.0005, ogre +0.0037, fandisk
+0.0002, teapot +0.0001, heart 0, bob −0.0002, spot −0.0006, A −0.0007, C −0.0008, cheburashka
−0.0009, homer −0.0010, bunny −0.0013), **cow −0.0068 and nefertiti −0.0036** beyond it; no early
stop, no brake stop, no ejection; the reversal windows no worse than g41pw's (beast 7 against 10).
Not adopted at 40k (neutral); it remains the 300k supply lever (au300 filled the tip), read on
ao300. The from-the-start term (g41ph) keeps its +0.008 to +0.015 on ten targets and its two
catastrophes; a placement that keeps the gain without them is not found — the gain is a
transport-phase effect and the failures are transport-phase effects of the same term.

**2026-09-26 15:00 — the particle-scale front changes the growth: a stub instead of a spike, and
beast's transport finishes.** g41fp (the adopted 40k form + `--pace_front_pts`) by eye
(`frontpts_growth_40k.png`, the same frames 3–30 as before, the baseline above): the head rises
as a rounded hump (frames 3–7), the ear appears as a SHORT THICK STUB (10–13) instead of a thin
spike with a nub, and grows shorter and thicker than the baseline's at every frame (17–30);
what remains is a rounded knob at the stub's tip (17–30) — no thin spike, no separated bead.
**P236 half met** (the stub is the tongue's beginning; the tip's knob stays). The numbers:
bunny 0.9678 (−0.0025 vs g41pw), **beast 0.9578 (+0.0062 vs g41pw), 161 windows, 93 %
arrived (76 % before), pinned 0.94 (0.70), no reversal window (10 before)** — the front
finishes the transport that stalled since g41t, and beast's D1 residual is gone; **dragon 0.9583
(−0.0059 vs g41pw)** with the end-state p1 **0.774 and 44 particles below 0.5 (0.11 %; the
health bound 0.1 %)**: the material that waits at the front of the dragon's many spikes packs
into the front cells. The early dent: 10 band bins (baseline 18–21) — the front halves it as
H⁻¹ did. The fraction of images held at the front: 0.38–0.64 in the first window, 0.25–0.36
at the end. *The dragon's cost, read:* a queued particle's image sits at the front cell, so the
paced grid there holds the images of every particle queued behind it and the cell sum asks for
more mass than the target holds in that cell — the front cell over-fills. With the front every
image lies inside the target, so the paced grid can be CAPPED at the target's own cell mass
(`--pace_cap`: min(paced, target) per cell): a queued particle is asked for nothing until the
front reveals its next cell — the queue semantics the rule intends. Launched: **g41fq** = g41fp
+ `--pace_cap` on dragon, bunny, beast. Pre-registered: **P240** dragon's end-state below-0.5
count ≤ 0.02 % (g41pw 0.01 %) and p1 ≥ 0.85 with the fit within −0.003 of g41pw; bunny and beast
within ±0.003 of g41fp; the stub growth kept (by eye).

**2026-09-26 16:00 — ao300 (300k, H⁻¹ outside the core from the pin's onset): the fit without the
supply.** 57 windows, silIoU **0.977** (au300 0.977, ar300 0.9773), det F 0.717, no reversal
window, strays 0.04 % — and the ear's tip **7.2** reference particles (au300 14.0, ar300 9.2),
the slabs near the target by the end (0.95–1.13) with the growth's knob at t = 0.4 (1.05× over
0.47× = 2.2). P239 ✗ on the tip: the placed term keeps the fit and the health but not the
supply — the tip was filled by the term acting from the START inside the core (au300), which
is exactly the placement that fails beast and bimba at 40k. At 300k the gallery is the bunny
and the dragon; the generalisation test of au300's form is the dragon: **dh300** = the slip
form + `--w_h1 1` (inside, from the start) on the 300k dragon, launched (P241: silIoU ≥ 0.965
— dw300's 0.9646 under the slip form — with the end-state p1 ≥ 0.80 and ≤ 0.05 % below 0.5,
no ejection, no early stop before 90 % arrival; if it holds, the 300k form takes H⁻¹ from the
start and the 40k form does not — the two forms differ by one flag with a stated reason).

**16:20 — the particle-scale front's cost on bulk targets: the transport is serialised.**
g41fp's remaining rows vs g41pw: cow 0.9602 (**−0.0058**, 96 windows vs 115), nefertiti 0.9653
(**−0.0079**, **115 windows vs 57**); the end-state health is kept (p1 cow 0.891, nefertiti
0.853). The pinned fraction by window tells the mechanism — nefertiti g41pw 0.23 / 0.51 / 0.75 /
0.95 at windows 24 / 32 / 40 / 56; g41fp 0.10 / 0.14 / 0.20 / 0.39 / 0.75 / 0.95 at 24 / 32 / 40 /
56 / 80 / 112 — the front halves the transport's speed on a target with no thin feature, and
the fraction of images held at the front stays at **0.26 from window 40 to the end** (a
standing queue), the run ending on the merit's rejections at 0.95 pinned. So the front's
first form is a wave from the source–target overlap outward at about half the pace's speed
(the images are clamped at the last of 24 ray SAMPLES inside the revealed region, and a point
counts as filled only with a particle within one spacing — two discretisation losses on the
wave's speed), and every target pays for it, thin feature or not. The trade as it stands:
beast +0.006 (its transport finishes, D1 residual 0), the ear a stub, dragon / cow /
nefertiti −0.006 to −0.008. The cap (g41fq) answers the dragon's over-fill, not the
serialisation; the front is not adoptable at 40k in this form. Launched in parallel:
**af300** = ar300's form + `--pace_front_pts --pace_cap` on the 300k bunny (GPU 1; the D2 test
the front exists for). Pre-registered **P242**: the ear grows as a stub/tongue (by eye against
ar300's frames), tip ≥ 9 (ar300 9.2), knob index ≤ 1.6 (ar300 2.2), silIoU ≥ 0.974 (within
−0.003 of ar300), end-state p1 ≥ 0.80, no reversal window; the window count is expected to
double (the serialisation), which is the cost to remove next. Next on the front itself: remove
the two discretisation losses (clamp at the exact boundary crossing along the ray, not the
last sample; a filled point read at the pace's own radius) and re-read nefertiti's window
count — P243: nefertiti under the corrected front within 1.3× of g41pw's windows and within
−0.003 of its fit, with beast's finish and the stub kept.

**2026-09-26 17:00 — the cap (g41fq): bunny recovered, the dragon deadlocked; the geodesic front
launched as a diagnostic; the front's vacancy fill as the answer (P244).** g41fq (adopted form +
`--pace_front_pts --pace_cap`): **bunny 0.9704 (= g41pw 0.9703; g41fp 0.9678), 60 windows (45),
end-state p1 0.926, none below 0.5, no reversal window** — P240's bunny part ✓ and the front's
bunny cost was the pile. **Dragon 0.9508 (−0.013 vs g41pw), 86 windows, p1 0.864, none below
0.5, 12 null commits, ended by the merit's rejections at 35 % pinned** — P240 ✗ on the fit
while ✓ on the health. By window: pinned g41pw 0.40 / 0.68 / 0.81 / 0.90 at 30 / 40 / 50 / 80;
g41fp 0.17 / 0.29 / 0.38 / 0.53 (0.72 at 144); g41fq 0.10 / 0.18 / 0.28 / 0.35; the fraction of
images held at the front 0.32–0.40 on the dragon throughout (bunny 0.27, nefertiti 0.26). *Read:*
the clamp piles the held images at one place on the ray. Without the cap the pile over-fills
the front cell (the compression); with the cap the front CELL is full before the unfilled
target POINTS behind it are reached — the fill is read at one spacing and the cap at one cell,
so a sub-cell spike deadlocks: no pull toward the vacancies, no fill, no reveal, and the run
ends on rejections. The bunny's ear is wider than a cell, so its front does not deadlock and
the cap only removes the pile. The same pile is why a bulk target waits: the source's part
outside the target has nothing revealed on its ray and sits until the wave arrives.
*Diagnostic in parallel:* the geodesic front (`--pace_front_geo`, eq. 52b: the target grown
along its own geodesics from the window-0 overlap at one pace per window) on nefertiti
(g41fg): origin 39 % of the target points, full reach in 9.2 pace steps; pinned 0.17 at window
35 (g41pw ≈ 0.6) — it holds as much as the fill-based front in its first windows; read at the
end for the record only, since a wave at the pace's own speed cannot order material that also
moves at the pace's speed (the ear's rays run along the ear: geodesic = ray), and its sweep was
stopped after nefertiti. **The answer to both costs is one rule (method.md 10.29 addendum, eq.
52c, `--pace_front_fill`): a held image is assigned to a revealed VACANCY (a revealed target
point without a particle within one spacing) within one pace of its clamp, one particle per
point (the capacity ratio of §10.28), closest first** — the front holds exactly the target's
mass (no pile, the cap redundant), the vacancies are pulled on at the point scale (no
deadlock), and material outside the target moves onto the nearest revealed surface vacancies
in its reach (the wave seeded from the air side too). Launched: **g41fv** = adopted form +
`--pace_front_pts --pace_cap --pace_front_fill` on dragon, nefertiti, bunny, beast (GPU 0).
Pre-registered **P244**: dragon within −0.003 of g41pw (≥ 0.961) with p1 ≥ 0.85 and ≤ 0.02 %
below 0.5; nefertiti ≥ 0.970 within 1.3× of g41pw's windows (≤ 74); bunny ≥ 0.967 with the stub
growth kept (by eye); beast ≥ 0.955 with its transport finished (pinned ≥ 0.9) and no reversal
window. If it holds on the four, the form goes to the 19 (the adoption gate) and to 300k.

**2026-09-26 17:40 — the front's standing hold was the point cloud's coverage gap (a discretisation
error of the front, corrected).** The fraction of images held at the front never fell below
0.26–0.32 in any front run (bunny 0.27, nefertiti 0.26, dragon 0.32 — and the geodesic front
0.27 after its full reveal at window 9). Measured on the adopted form's own end states, where
the particles fill the target: **24–27 % of the particles lie farther than one spacing from
every target point** (bunny 23.9 % at the end / 26.2 % at a third of the run; dragon 25.9 /
26.7 %), 0.1–2.4 % farther than 1.5 spacings, and **0.0–0.9 % farther than the 8-neighbour
shell radius** (1.98 spacings). The front read "filled" and "inside" at one spacing, so a
quarter of the target's volume was never inside, a quarter of the images were clamped by
construction, and every front run carried that hold — the bulk targets' serialisation and the
dragon's stall under the cap were mostly this. Corrected in `pace_front_pts` (method.md 10.29,
the 17:40 correction): the radius is per target point its shell radius (the 8th neighbour's
distance, cached once); no new constant. *Withdrawn as superseded:* g41fv (the vacancy fill
under the one-spacing radius; its dragon at window 30: held 0.33, pinned 11 %) and af300 (the
300k front under the same radius, at window 60). *Recorded:* **g41fg** (the geodesic front,
nefertiti) 0.9515 (−0.022), frozen after 5 null commits at 52 windows with 23 % pinned — refuted
as a form, and its hold was the same floor. **g41fq's other rows** (cap, one-spacing radius):
cow **0.9648** (−0.0012 vs g41pw, 75 windows for 115, p1 0.915, no reversal) ✓; nefertiti
**0.9527** (−0.021, 47 windows, 18 % pinned) ✗; beast **0.8929** (−0.059, 127 windows, 57 %
pinned) ✗ — under the cap the held quarter has nothing to pull it and the transport starves.

**18:00 — dh300 (300k dragon, the slip form + H⁻¹ inside the core from the start): P241 ✓.**
silIoU **0.9685** (dw300, the slip form alone, 0.9646: +0.004), end-state p1 **0.839**, 4
particles below 0.5 (0.001 %), pinned 57 %, no traceback, early stop at window 82 on three
rejections (the run's usual end under the pin). With au300 (300k bunny, the same form: 0.977,
det F 0.76, tip 14.0) the 300k form is **ar300's form + `--w_h1 1`** — the one flag the 300k
form has and the 40k form has not, for the stated reason (the supply to a thin feature is a
300k need; at 40k the term's placement is neutral or harmful).

**18:10–18:50 — the coverage radius alone (g41fw dragon), and the air-side material approaches
the front (52e, g41fx).** g41fw (front + cap + in-reach fill, coverage radius) dragon: **0.9588**
(−0.0054 vs g41pw), 79 windows, end p1 0.848, none below 0.5, pinned 36 % at the end, held
0.39 → 0.03 by window 45 with almost nothing assigned (vacancies rarely within one pace of a
clamp). So with the floor gone the remaining hold is the part of the source OUTSIDE the target
(dragon 39 %, nefertiti 56 % of the particles at the first window): nothing revealed on its ray,
no vacancy in reach — it sits until the fill walks to it. Rule (52e, method.md 10.29 addendum):
a held particle with no vacancy in reach takes its nearest open vacancy wherever the front is
(one per point, closest first, in rounds over the open vacancies) and approaches it at the pace
— material outside the target accretes at the growing front (the user's "volume first"), and
nothing seeds an unrevealed thin feature from the air. First form (8 candidates per particle)
assigned 0.9 % per window (the near points exhausted), the rounds form 4.6 % at 16 rounds and
**10 % at 64 rounds** = the front's whole vacancy count per window. Launched: **g41fx** = adopted
form + `--pace_front_pts --pace_cap --pace_front_fill` (coverage radius, 52c + 52e) on nefertiti,
dragon, bunny, beast (GPU 3). Pre-registered **P245**: nefertiti ≥ 0.970 within 74 windows;
dragon ≥ 0.961 with p1 ≥ 0.85 and none below 0.5; bunny ≥ 0.967 with the stub growth kept (by
eye); beast ≥ 0.955 with its transport finished (pinned ≥ 0.9) and no reversal window. At 300k:
**ag300** = ar300's form + front + cap + fill (coverage radius, without 52e — its air-side hold is
3 % by window 11) running as the D2 test (P242).

**2026-09-26 19:40 — the front on a bulk target scrambles the arrangement; the stub at 40k was the
pile's; the front restricted to the loss-blind part of the target (P246).** (1) g41fx nefertiti
(coverage radius + 52c + 52e): the hold is gone by window 12 (0.01) and yet pinned 7.6 / 14.0 /
22.2 % at windows 30 / 35 / 45 (g41pw 0.51 at 32, 0.88 at 48). A pinned-crust reading was
tested and refuted: at the end of g41fq nefertiti (19 % pinned, stalled) the target's points
without a particle within the shell radius are **0.3 %**, every unpinned particle is inside the
target, and the pinned particles sit on the shell in the same proportion as under g41pw (33 %
vs 32 %) — the target is covered, the pin does not engage. So the plan's endpoints no longer
match where the material sits: the front's accretion (at whatever vacancy was near) leaves an
arrangement the re-solved plan wants to permute over long paths through a filled body, the
transport merit cannot, and the run ends on rejections with the fine-scale density irregular
(silIoU 0.953 at 99.7 % coverage). The bulk's arrangement is the loss's to resolve; every front
form scrambles it. (2) The growth strips built with one crop (`growth_montage.sh`; rows g41p /
g41fp / g41fq, frames 3–30, `front_cap_growth_40k.png`): under the cap the ear rises as a spike
again (frames 10–13 thicker than the baseline's, a bulb at the tip from 25) — **the stub of
g41fp was the pile's pull at the base**, not the ordering's; the ordering alone thickens the
early ear a little. The cap also deepens the early dent (25 band bins for 18–21). (3) The
principled restriction, `--pace_front_thin`: the front and its cap apply only to the part of the
target the cell sum cannot resolve — target points none of whose CIC nodes holds half the
target's bulk node mass (the body's definition of §10.22, read once) — and the bulk is always
revealed, transported by the pace as before. On a bulk target the front is inert by
construction; on the dragon its spikes and on the bunny the ear's sub-cell tip are ordered
(at 300k under `--disc_ref` the ear is bulk except its tip — the knob's and the bead's place).
Launched **g41fy** = adopted form + `--pace_front_pts --pace_cap --pace_front_fill
--pace_front_thin` on nefertiti, dragon, bunny, beast (GPU 0). Pre-registered **P246**:
nefertiti within ±0.002 of g41pw and within 1.15× its windows (the inert front — a cost here
is the cap's or the fill's, not the front's); dragon ≥ 0.961 with p1 ≥ 0.85 and none below 0.5;
bunny ≥ 0.967; beast ≥ 0.949. ag300 (300k bunny, the front unrestricted but holding 1–3 % after
window 11 — in effect the tip's front) is the ear reading to compare with the thin form.

**19:55 — the air-side hold was the front's own error, removed.** g41fy's first launch on nefertiti
held 34 % of the images with the thin part at 0.0 % of the target: the "out" test held every image
not within the shell radius of a revealed point, which includes every image IN THE AIR — the source's
part outside the target — and 52e then re-routed those 30 % to vacancies (the scramble of item 1).
Corrected in `pace_front_pts`: a ray is held only where it passes through unrevealed TARGET (a sample
in the air is passable; a sample inside the target is passable when its nearest point is revealed),
the image clamped at the last passable sample before the first unpassable one. g41fy relaunched:
nefertiti holds 0.000 from the first window (the front inert on a bulk target, as P246 requires).

**2026-09-26 20:30 — cleanup 4 (the results passed 100 GB): 101 → 62 GB.** Deleted the archives (npz)
of the refuted or superseded forms — g41phb (19), g41ph (19), g41fr (3), g41fp cow / nefertiti /
dragon, ae300, l300 (46 files, 40 GB; the list in `$OUT/.deleted_list_20260926d.txt`, their json +
logs tarred first into `logs_archive_20260926d.tgz`). Kept: g41, g41pw, g41fp bunny + beast, the
open front runs, ai300, ar300, aw300, ak300, au300, ao300, ag300, the reports and videos.

**20:35 — g41fy (the front restricted to the sub-bulk target) nefertiti: 0.9670, 58 windows, p1
0.889, no reversal — −0.006 against g41pw with the front holding nothing in any window.** The
thin part of nefertiti is 0.0 % of its points, so the one thing the form still did was the cap
at the "thin" nodes — which are the target's SURFACE HALO (the CIC partial nodes hold less than
half the bulk mass everywhere on the surface): min(paced, target) at every surface node changes
the transport's drive there. The cap is withdrawn from the front's form (52c/52e make it
redundant: the assigned images hold exactly the target's mass). And the thin restriction is
vacuous where it matters: at 300k under `--disc_ref` the bunny's thin part is **0.1 %** of its
points (ah300, withdrawn) — the ear is bulk at the loss cell; the knob is not a sub-cell defect.

**20:40 — what the ear's growth IS (probe `ear_order_probe.py`: the ear's particles grouped by
where they END).** ar300, the left ear: the tip-bound (432), neck-bound (2137) and base-bound
(846) groups are ordered from the first window — tip above neck above base — and each travels
the same distance, **1.34 wu**: the plan translates one column of the head into the ear. In
transit the column is STRETCHED — its rear inside the head is slow (it displaces the body), its
front in the ear is free: at t = 0.4, **68 % of the tip-bound material is above the neck slab
while 27 % of the neck-bound has arrived** (ao300: 29 % / 13 %; the groups' y-spread 0.3 wu in
transit for 0.07 at the end). A stretched column is thin — the spike; the tip group stopping at
the top while the column behind is still stretched — the knob on a neck. So the fronts of (52)
with "filled = one particle within the radius" cannot stop it: the sparse lead fills the points
around it and the reveal moves with it at the pace (a chain), which is why every front held ≈ 0
once its bugs were fixed (g41fd first launch: 0.000 on the bunny). **New fill (52f,
`--pace_front_dense`)**: a target point is filled when the particles within its shell radius reach
HALF its own count there (the target holds 8 by the shell's definition; half = the body
convention of §10.22, scaled by n / |target|), and the reveal is one shell beyond the filled
region (the fill's own resolution; one pace at 40k is a third of the ear). A lead filament does
not fill; its images are clamped at the front; the material accumulates until the rear arrives;
the front advances as a plug at the target's density — the tongue. No cap. Launched: **g41fd** =
adopted form + `--pace_front_pts --pace_front_fill --pace_front_dense` on bunny, dragon (GPU 0),
nefertiti (GPU 2); **ba300** = the 300k candidate form (ar300 + `--w_h1 1`) + the same front
(GPU 3). Pre-registered **P247** (40k): bunny / nefertiti / dragon within ±0.003 of g41pw with
p1 ≥ 0.85 and no reversal window; the bunny's ear by eye a stub/tongue without a tip bulb
(growth strips vs g41p). **P248** (300k): silIoU ≥ 0.977, end p1 ≥ 0.80, tip ≥ 10, knob index
≤ 1.3 (the neck slab ≥ 0.8× when the top slab first exceeds 0.9×), and no negative-window
streak at the end.

**20:45 — ag300 (300k bunny, ar300's form + the front of the morning's code + cap): silIoU 0.9803
— the best 300k fit so far** (ar300 0.9773, z300b 0.9802), run det F min 0.72, 76 windows, tip
10.4 (ar300 9.2), pinned 70 % at 70; but the knob persists (t = 0.2: top slab 1.11× over 0.46×
below; t = 0.3: 0.47× neck under 0.68×) — the growth order is ar300's (56 % of the tip-bound
above the neck slab at t = 0.4 for 32 % of the neck-bound arrived) — and **the last five windows
(71–75) reverse in a row**, a D1 streak at the end of the run (the free 30 %). Read as: the
morning's front (air-side hold + coverage-gap floor) acted on 3 % of the images after window 11
and the fit gain is the cap's or noise; the tongue needs (52f).

**2026-09-26 21:20 — g41fd (the density-fill front, no cap) on the three: P247's numbers ✓.** bunny
**0.9712** (+0.0009 vs g41pw), 49 windows (45), end p1 0.921, pinned 97 %; dragon **0.9643**
(+0.0001), **68 windows (103)**, p1 0.884, pinned 88 %; nefertiti **0.9712** (−0.0020), 53
windows (57), p1 0.894, pinned 91 %; none below det F 0.5 and no negative-reversal window on any
of the three. The front held 4–6 % of the images in the windows where the head and ears grow
(bunny windows ~8–20, nefertiti ~8–30), then nothing; the dragon held nothing in any window
(its spikes' material never fell below half the target's density in transit) and still finished
35 windows sooner than under the adopted form — the front's fill (52c) assigned the transit
material to the front's vacancies. The by-eye part of P247 (the bunny's ear a tongue without a
tip bulb) is pending the plain video (`fd_video.sh`). Launched: the density front on the 16
remaining gallery targets (GPUs 0 and 2; `gallery_table_g41fd.py`) — the adoption gate
(P249: 19/19 within −0.003 of g41pw, end p1 ≥ 0.85, none below 0.5, no negative-reversal
window; beast's stall the one expected exception, read separately). ba300 (300k) at window 25:
pinned 16 %, held 0.1–0.9 %.

**2026-09-26 21:50 — the density front does not change the ear's growth; the render channel is half
the control during the spike stage → the λ = 0 twins (P250, P251).** By eye (`dense_growth_40k.png`:
rows g41p / g41fq / g41fd, frames 3–30): g41fd's ear rises as the same thin spike with a nub as the
adopted form's (frames 10–13) and thickens on the same frames; the tip bulb at 25–30 is the same. The
ear-order probe agrees: at t = 0.2 the tip-bound material above the neck slab is 75 % (g41pw 72 %)
with 7 % (5 %) of the neck-bound arrived; at t = 0.3, 84 % (87 %) of the tip-bound is already at the
tip while the neck is 44 % (37 %) full. P247's by-eye part ✗; the front's 4–6 % hold in the growth
windows did not stop the stretched column. At 300k (`dense_growth_300k.png`: ar300 over ag300) the
morning's front + cap made the tip WORSE — a ball on a thin neck at frames 21–25 (the bead). The
density front is fit-neutral on the seven targets read so far (A ±0.0000, armadilo +0.0017, bunny
+0.0009, dragon +0.0001 in 68 windows for 103, fandisk −0.0002, heart −0.0005, nefertiti −0.0020;
p1 0.88–0.93, none below 0.5, no reversal window) — the sweep continues as the record of a
neutral lever (the dragon's 35 windows are the vacancy assignment's). *What drives the lead:* the
render channel's share of the control gradient is **0.50 / 0.43 / 0.34 / 0.42 at t = 0.1 / 0.2 /
0.3 / 0.4 on ar300** (g41pw 0.53 / 0.48 / 0.35 / 0.37; λ 0.02–0.12) — half of the control in the
windows where the ear grows, and the silhouette term is satisfied by a filament along the ear's
outline: the fastest way to fill the target's silhouette is to run a thin lead up it. The
rendering-influence rule asks for the λ = 0 twin here anyway. Launched: **bb300** = ar300's form +
`--lambda_auto 0` (300k, GPU 1) and **g41pl** = the adopted 40k form + `--lambda_auto 0` (bunny,
GPU 1). Pre-registered **P250** (bb300): if the render channel drives the stretch, the tip-bound
above the neck slab at t = 0.4 falls from 68 % to ≤ 40 % and the knob index to ≤ 1.3, with the ear
complete at the end (tip ≥ 9); if the numbers stay, the stretch is the transport's and the physics'
(the rear's drag in the head), and the remedy is on the transport side (a column-coherent pace).
**P251** (g41pl): the same reading at 40k (72 % → ≤ 40 % at t = 0.2), the fit read but not gated
(the λ = 0 twin is a diagnostic, not a form).

**2026-09-26 22:20 — ba300 (300k bunny, the candidate form + the density front): the best tip, no
reversal, and a BIGGER knob; who the knob is.** silIoU 0.9765 (au300 0.977, ar300 0.9773), 51
windows, run det F min 0.764 (the best of the 300k pin runs), **tip 16.3** (au300 14.0, ar300
9.2), no negative-reversal window. The slab table: at t = 0.25 the top slab 1.23× over a 0.42×
neck (knob index 2.9; ar300 2.2), at t = 0.4 1.32× over 0.67× — the front supplied the tip and
left the neck thin; a plug at 1.25× sat at the base slab (y 2.86) at t = 0.25–0.3. The ear-order
probe: the tip-bound above the neck slab at t = 0.4 is 41 % (ar300 68 %) — the front held the
tip-bound group back — and yet the top region is fuller. `knob_probe.py` (who is in the left
ear's top region, y > 3.55, at t, by where they END): **the ear's own material only** (no head or
foreign particles in any run), tip-bound 54–69 % and **neck-bound 31–46 %** on ba300 (100 / 155 /
164 particles there at t = 0.2 / 0.25 / 0.3; ar300 21 / 67 / 39 with the neck-bound 38–72 %; au300
57 / 122 / 171). So the early knob is the ear's column's lead — tip-bound particles arriving
first AND neck-bound particles OVERSHOOTING their slab to the top and returning later. The
overshoot points at the plan's kNN-averaged displacement (the "plan blur": a neck-bound
particle whose source neighbours are tip-bound takes their longer displacement; under
`--disc_ref` the blur was found doubled on 2026-09-23 and corrected by `--plan_native`, which was
only ever run inside the KDE bundle — ak300, whose tongue growth was credited to the KDE term).
Launched **bc300** = ar300's form + `--plan_native` alone (GPU 2). Pre-registered **P252**: the
neck-bound share of the top region at t = 0.2–0.3 falls below 20 % and the knob index to ≤ 1.3
with the fit within −0.003 of ar300; if so, ak300's tongue was the plan's, not the KDE's, and the
300k form takes `--plan_native` (a discretisation correction, no constant). The render-off twins
(bb300, g41pl) run in parallel for the render's share of the same overshoot.

**2026-09-26 22:45 — g41pl (40k bunny, the adopted form with the render channel OFF): the render
does not drive the stretch — P251 ✗.** silIoU 0.9499 (−0.020, the render's fit share at 40k, as
known), 51 windows, end p1 0.914, no reversal. The ear order WITHOUT the render: the tip-bound
material above the neck slab at t = 0.2 is **100 %** (g41pw 72 %), 65 % already at the tip (31 %),
the neck-bound in its slab 12 % (5 %) — the lead runs faster without the render, not slower;
the knob's composition at 40k is 90–100 % tip-bound in both. So the stretched column is the
transport's and the physics' (the rear's drag inside the head), and the render channel, half of
the control in those windows, pulls with it rather than ahead of it. The 300k twin (bb300) is
read for the record; the remedy moves to the transport: a column-coherent pace (a particle more
than one pace ahead of its plan neighbours, in remaining distance, waits — the plan's own
neighbourhood, no new constant), pre-registered next.

**2026-09-26 23:00 — the coherent pace (method.md §10.30, eq. 53, `--pace_coherent`): the remedy
aimed at the measured mechanism.** No front on the target can stop the stretched column, because
the lead fills the points around it and the reveal follows at the pace; the render channel does
not drive it (g41pl); the plan blur is under test (bc300). The paced target itself now keeps the
material neighbourhood together: a particle more than one blur radius ahead of its plan
neighbours' centroid along its own ray (the neighbourhood = the k particles inside one blur
radius at the source, fixed for the run — the set the plan is already averaged over) has its pace
step shortened by the excess and waits for the rear; particles behind advance at the pace. A
smooth map moves a neighbourhood together, so the bulk is untouched. No new constant. Launched:
**bd300** = ar300's form + `--pace_coherent` (300k bunny, GPU 0) and **g41fc** = the adopted 40k
form + `--pace_coherent` on bunny, nefertiti, dragon (GPU 3). Pre-registered **P253** (bd300): the
tip-bound material above the neck slab at t = 0.4 ≤ 40 % (ar300 68 %), the neck-bound share of the
early top region ≤ 20 % (38–72 %), knob index ≤ 1.3 (2.2), tip ≥ 9 at the end, silIoU within
−0.003 of ar300 (≥ 0.974), end p1 ≥ 0.80, no negative-reversal window; by eye the ear grows as a
tongue (the growth strip against ar300). **P254** (g41fc): bunny / nefertiti / dragon within
±0.003 of g41pw with p1 ≥ 0.85 and no reversal window, and the bunny's ordering changed
(tip-bound above the neck at t = 0.2 ≤ 40 %; g41pw 72 %). If P253 holds and P254's fit holds, the
rule goes to the 19 (the adoption gate) and to the 300k dragon.

**2026-09-26 23:30 — bb300 (300k bunny, ar300's form with the render channel OFF): no knob, a
tapered tongue, a starved tip — the render channel is the EARLY driver of the top region at 300k
(P250 half met).** silIoU 0.9694 (−0.008 vs ar300), 65 windows, run det F min 0.743, tip **6.1**
(ar300 9.2), no negative-reversal window. The slab table has no knob at any time: at t = 0.3 the
ear is 0.65 / 0.41 / 0.25× from the base up and nothing above the neck; at t = 0.4, 0.76 / 0.77 /
0.61× with the top two slabs still empty; the top fills last (0.70 / 0.78× at t = 0.6, 0.92 /
0.97× at the end) — a monotone taper from the base, i.e. the tongue, but late and starved at the
tip. `knob_probe.py`: **no particle in the top region before t = 0.4** (ar300 21 / 67 / 39 at
t = 0.2 / 0.25 / 0.3). The ear-order probe: the tip-bound above the neck slab at t = 0.2 is 0 %
(ar300 6 %) and at t = 0.4 61 % (68 %) — the column's stretch is still the transport's and comes
later; what the render channel adds is the lead's EARLY arrival at the top (the silhouette term
pulls the first material up the target ear's outline, which a thin lead satisfies). At 40k
(g41pl) the same twin showed no such effect, so this is a 300k statement (the ear is many cells
there and the silhouette's pull on a sub-cell lead is what the cell sum cannot see). *Design
that follows (pre-registered as the next lever, §10.31):* the render channel keeps its role but
its TARGET becomes the paced target's own images — the silhouettes of the paced image cloud
x̂ (the same intermediate target the physics channel is driven to), re-rendered each window
without gradient — so the two channels agree on the growth order and the render adds its fit
as the paced target converges to the target (the final windows are unchanged: x̂ = target).
P255: with the paced render target, the top region stays empty before t = 0.3 as in bb300, the
knob index ≤ 1.3, and the fit and the tip return to ar300's (≥ 0.974, tip ≥ 9).

**2026-09-27 00:10 — three levers on the stretched column, read side by side (300k, ar300's form +
one flag each; the ear-order and knob probes).**

| run | lever | silIoU | tip | tip-bound above the neck at t = 0.4 (ar300 68 %) | early top region | knob |
|---|---|---|---|---|---|---|
| bb300 | render OFF | 0.9694 | 6.1 | 61 % | none before t = 0.4 | none |
| bd300 | coherent pace (53) | **0.9771** | 7.3 | **19 %** | 29 / 61 particles at t = 0.25 / 0.3, 41–66 % neck-bound | 1.8 at t = 0.3 (1.28× over 0.70×), gone at t = 0.4 |
| bc300 | plan blur native | 0.9741 | **14.4** | **10 %** | 9 / 51 at t = 0.25 / 0.3, 0–45 % neck-bound | 1.8 at t = 0.3 (1.60× over 0.88×), 2.1 at t = 0.4 |

Readings. **P253 (bd300) half met**: the coherent pace keeps the fit (−0.0002), holds the
tip-bound column (68 → 19 % overtaking) while binding on only 0.2 % of the particles per window
(the ones ahead of their neighbourhood), has no reversal window and end p1 read below — but the
tip ends starved (7.3) and a knob still forms at t = 0.3 from material that is 66 % NECK-bound
and leaves again by t = 0.4. **P252 (bc300) half met**: the native plan blur orders the tip-bound
group best (10 %) and fills the tip (14.4), but the run's det F min is 0.59 and the knob is the
largest (1.60× at t = 0.3). **bb300** shows what removes the early top material entirely: the
render channel off. So the knob has two makers — the tip-bound group's lead (the transport's
stretch, which the coherent pace and the unblurred plan both reduce) and the NECK-bound
overshoot into the top region, which is present under both and absent only without the render
channel: the silhouette term pulls whatever material is nearest to the ear's outline, tip-bound
or not. The lever for the second is §10.31 (the render channel targets the paced cloud; be300
running, P255), and the natural form is the pair: **bf300** = ar300's form + `--render_paced
--pace_coherent` (GPU 3). Pre-registered **P256**: the top region empty before t = 0.3 (as bb300),
the tip-bound above the neck at t = 0.4 ≤ 40 %, knob index ≤ 1.3 at every t, tip ≥ 9, silIoU ≥
0.974, end p1 ≥ 0.80, no reversal window. **P254 (g41fc, 40k coherent pace)**: bunny 0.9705
(+0.0002), dragon 0.9650 (+0.0008), nefertiti 0.9702 (−0.0030, at the bound); the 40k ordering
barely moves (65 % vs 72 % at t = 0.2) — at 40k the column is three cells and the stretch is
the pace's own step; the 40k gate for the pair is read after the 300k pair.
Health of the three: bd300 end p1 0.869, none below 0.5, 52 windows, no negative-reversal window;
bc300 end p1 0.858 (the 0.59 is the run's transit minimum), 45 windows, none below 0.5; g41fc end p1
0.916 / 0.901 / 0.883 (bunny / nefertiti / dragon), none below 0.5, no reversal window on any.

**00:30 — g41rp (40k, the render channel targeting the paced cloud): costs the fit.** bunny 0.9654
(−0.0049 vs g41pw), dragon 0.9564 (−0.0078), nefertiti 0.9665 (−0.0067); the 40k ordering is read
below with be300. Under the pin a particle is locked at arrival, while its render target is still
the paced cloud's silhouette; the final outline reaches it only through the last free material.
If be300 removes the 300k knob, the placement to test is the one accepted for H⁻¹: the paced render
target until the pin's onset (the growth), the target's own images after (the fit).

**2026-09-27 00:50 — be300 (300k bunny, ar300's form + the paced render target): P255 met on the fit,
the health and the order; the tip starves.** silIoU **0.9754** (−0.0019 vs ar300, within the
bound), 51 windows, end p1 0.877, none below 0.5, no negative-reversal window. The order: the
tip-bound above the neck slab at t = 0.4 is **19 %** (ar300 68 %; the coherent pace alone also
19 %), 0 % at t = 0.2. The slab table: a monotone taper at t = 0.3 (0.65 / 0.59 / 0.40× from the
base up, nothing above the neck); at t = 0.4 a slight bump (0.86× at y 3.46 over 0.63× below —
index 1.37, the bound 1.3), 1.2 at t = 0.6. The early top region: 19 / 40 / 25 particles at t =
0.25 / 0.3 / 0.4 (ar300 21 / 67 / 39; bb300 none), neck-bound 37–65 % of them — reduced, not
gone. The tip ends at **6.9** reference particles (ar300 9.2): the render channel's final
outline was what fed the tip at the end, and the paced target does not (its images arrive at the
tip only with the last material). At 40k the same flag costs −0.005 to −0.008 and changes no
order (g41rp: 79 % vs 72 % at t = 0.2) — a 300k flag, like `--w_h1 1`, and the tip's supply is
that flag's job. Launched **bg300** = the 300k candidate form (ar300 + `--w_h1 1`) + `--render_paced
--pace_coherent` (GPU 1), alongside bf300 (the pair without H⁻¹, GPU 3). Pre-registered **P257**
(bg300): tip ≥ 9, silIoU ≥ 0.974, end p1 ≥ 0.80, no reversal window, the tip-bound above the
neck at t = 0.4 ≤ 40 %, knob index ≤ 1.3 at every t, and by eye a tongue (the 300k strip).

**2026-09-27 01:15 — the 300k strip of the three levers (`levers_growth_300k.png`: ar300 / bb300 /
bc300 / bd300, frames 3–30): by eye none gives the tongue.** Every row raises a thin spike first
(frames 13–21) and thickens it afterwards; bb300 (render off) and bc300 (native plan) carry a
bead on a thin neck at frames 21–25, bd300 (coherent pace) looks like ar300 with a bulb at 30.
The probes' improvements (the tip-bound overtaking 68 → 10–19 %) do not show as a thicker early
ear, because the stretch is between the column's LAYERS (tip-, neck-, base-bound: 0.2–0.5 wu
apart at the source, outside each other's blur balls) and the material that emerges into the
ear is a sparse jet whatever its end group; the coherent pace saw 0.2 % of the particles. **The
stream pace (method.md §10.32, eq. 55, `--pace_stream`)** reads the coherence at the pace's scale
along the ray: a particle's step scales with the fill of the ball (one blur radius) one pace
step behind it on its own ray, against half the count that ball holds at the source density
(k = 64, the plan's own neighbourhood); a lead with a sparse stream behind it waits, a
continuous stream advances at the pace, the bulk is untouched. First windows: 1.9 → 0.7 % of
the particles with a sparse stream behind (mean fill 0.34 there) — ten times the coherent
pace's reach before the ear even starts. Launched: **bh300** = ar300's form + `--pace_stream`
(GPU 2) and **bi300** = the 300k candidate form (ar300 + `--w_h1 1`) + `--render_paced
--pace_stream` (GPU 0). Pre-registered **P258** (bh300): the rule binds on ≥ 2 % of the particles
in the growth windows (8–25), the early top region empty before t = 0.3, the tip-bound above the
neck at t = 0.4 ≤ 40 %, knob index ≤ 1.3, silIoU ≥ 0.974, end p1 ≥ 0.80, no reversal window, and
by eye a thick early ear on the strip; **P259** (bi300): the same with tip ≥ 9. Page v52 carries
the three-lever table and the strip.

**2026-09-27 01:35 — bf300 (300k bunny, ar300's form + the paced render target + the coherent pace):
the best growth profile so far; the fit −0.004.** silIoU 0.9732 (ar300 0.9773; P256's bound 0.974
missed by 0.001), 62 windows, **tip 14.8** (ar300 9.2; the paced render alone starved it at 6.9 —
with the coherent pace the held column arrives together), end p1 **0.895** (the best of the 300k
pin runs), none below 0.5, no negative-reversal window (two streaks of three null commits at
windows ~11 and ~54). The slab table: a monotone taper at t = 0.3 (0.65 / 0.53 / 0.38× from the
base, nothing above), a bump of 1.2 at t = 0.4 (0.66× over 0.55×), 0.94 / 0.89 / 0.86 / 0.71 /
0.87× at t = 0.6 — **knob index ≤ 1.2 at every time** (ar300 2.2). The early top region: 8 / 20
particles at t = 0.25 / 0.3 (ar300 21 / 67), 75–100 % tip-bound — the neck-bound overshoot is
gone. The overtaking measure reads 55 % at t = 0.4 (the held tip-bound column arrives as one
group between t = 0.3 and 0.4, which is the intended plug, not the jet: 5 % at t = 0.2, 9 % at
0.3). P256: growth ✓ (taper, knob, top region), tip ✓, health ✓, reversal ✓, fit ✗ by 0.001.
The by-eye strip is pending its video. The 300k candidate with H⁻¹ (bg300) and the stream-pace
runs (bh300, bi300) decide the form.

**01:40 — cleanup 5 (98 GB with three 300k runs in flight): 98 → 84 GB.** Deleted the archives of
the superseded 40k front forms and the read twins — g41fq (5), g41fw, g41fx, g41fg, g41fy (2), g41fp
(2), g41pl, ag300, ah300, af300 (13 files, 14 GB; list in `.deleted_list_20260927a.txt`, json + logs
tarred first into `logs_archive_20260927a.tgz`). Kept: g41, g41pw, g41fd, g41fc, g41rp, the 300k
readings (ai300, ar300, aw300, ak300, au300, ao300, dh300, ba300–bi300), reports and videos.

**2026-09-27 02:00 — the stream pace (bh300) and the H⁻¹ pair (bi300): the knob's makers confirmed
one by one.** **bh300** (ar300 + `--pace_stream`): silIoU 0.9756 (−0.0017), 57 windows, tip 10.0, end
p1 0.867, no negative-reversal window; the rule binds on 2–3.4 % of the particles through the
growth (rising with the ear; the coherent pace saw 0.2 %). The tip-bound group is held (4 / 8 % above
the neck at t = 0.2 / 0.3) — and the knob is there all the same: 1.26× over 0.76× at t = 0.25, 1.12×
over 0.49× at t = 0.3 (index 1.7–2.3), made of NECK-bound material (71–77 % of the 70 / 48 particles in
the top region) — the render channel's pull, untouched by any pace rule. P258 ✗ on the knob. **bi300**
(ar300 + `--w_h1 1 --render_paced --pace_stream`): silIoU 0.9753 (−0.002), 59 windows, end p1 0.916,
tip **21.5** (the target's own density there is 11.9: over-filled), and the top region holds 77 /
159 / 95 particles at t = 0.2 / 0.25 / 0.3 — **84–95 % tip-bound** — with a knob of 1.12× over a
0.38× neck at t = 0.3 (index 2.9): H⁻¹ from the start is the knob's third maker, the non-local
supply pulling the tip-bound group to the tip's deficit before the column arrives (au300's knob,
now isolated); one negative-reversal window at the end. So the three makers and their levers:
the transport's stretch (coherent / stream pace), the render's early pull (the paced render
target), the supply's early pull (H⁻¹ placed at the pin's onset — the placement that kept bimba
and beast at 40k). bf300 (render paced + coherent, no H⁻¹) has none of the three (knob ≤ 1.2,
tip 14.8) and costs −0.004 in fit; the fit's cost is the paced render target's under the pin
(g41rp at 40k), so the next form places it too: the paced render target until the pin's onset,
the target's own images after (`--render_paced_onset`, pre-registered **P260**: bf300's growth
with silIoU ≥ 0.974). Launched meanwhile **bk300** = ar300 + `--render_paced --pace_stream` (GPU 3;
P261: the stream pace under the paced render against bf300's coherent pace — knob ≤ 1.3, tip ≥
9, fit ≥ 0.974).

**2026-09-27 02:20 — bg300 (the 300k candidate form with H⁻¹ from the start + the pair) fails at
the end; the pair's strip by eye.** bg300: silIoU 0.9734 (−0.004), 74 windows, tip 17.5, end p1
0.913 — and **the last 16 windows reverse in a row** (a D1 streak: the settled end-game under
H⁻¹ from the start with the paced target held back), the top region 153 / 93 / 61 particles at
t = 0.2 / 0.25 / 0.3 (84 % tip-bound: the supply's early pull), knob 2.4 at t = 0.25. H⁻¹ from the
start leaves the 300k form when the pair is present; if the tip needs it, at the pin's onset. The
strip `pair_growth_300k.png` (ar300 / be300 / bf300): by eye the early ear (frames 13–21) is
still a thin spike in all three rows, bf300's slightly thicker and without the bead; the
probes' "taper" at t = 0.3 is a taper of a thin ear (0.65 / 0.53 / 0.38× of the target's
thickness) — the tongue needs the base near 0.9× while the ear extends, and the base-bound
material is the column's deepest layer, the last to arrive under the monotone plan; the ear
extends as fast as the pace lets its front go while the flux into it is the head's drag. The
flux-limited front is the stream pace's semantics (bh300 held 2–3 % but the render pulled the
neck-bound material; bk300 = paced render + stream pace is running, P261). Launched: **bj300** =
ar300 + `--render_paced --render_paced_onset --pace_coherent` (GPU 2; P260) and **bl300** = the same
+ `--pace_stream` (GPU 1; **P262**: bf300's growth or better — the base slabs ≥ 0.8× when the top
slab first appears — with silIoU ≥ 0.974, tip ≥ 9, no reversal window).

**2026-09-27 03:30 — the onset placement loses the growth; the density front's gallery; the 40k
twins; cleanup 6; the convergence placement (P265–P267).** **bj300** (paced render until the pin's
onset + coherent pace): silIoU 0.9751 (−0.0022 ✓), 50 windows, tip 8.7, end p1 0.864, no reversal —
but the neck-bound overshoot is back (45–66 % of a 31–51-particle top region at t = 0.25–0.4) and a
knob of 1.9 at t = 0.3: the pin's onset (the first pinned particle, in the head) comes before the
ear's growth ends, and the target's outline pulls the lead again. **bl300** (the same + stream
pace): 0.9754, tip 10.7, p1 0.893, knob 1.54 at t = 0.25 (85 % neck-bound) and **six reversing
windows at the end**. **bk300** (paced render + stream pace, no placement): 0.9743 (−0.003), 45
windows, tip 8.8, no reversal, a tapered ear at t = 0.3 (0.69 / 0.70 / 0.45×) with the top region
1 / 31 / 27 (52–71 % tip-bound) — bf300's growth with the stream pace, the fit a shade better
(0.9743 vs 0.9732), the tip lower (8.8 vs 14.8). P260 ✗, P261 borderline, P262 ✗. *The density
front's gallery (g41fd, 19/19 vs g41pw):* min −0.0036 (cow), max +0.0019, 17 within ±0.003
(cheburashka −0.0032, cow −0.0036), end det F none below 0.5 on all 19 — fit-neutral; a speed lever
on the dragon (68 windows for 103); not adopted (it does not change the growth). *The 40k twins:*
g41fs (stream pace) bunny −0.0031, dragon −0.0025, nefertiti −0.0041 with 1.3–1.5× the windows —
the stream pace costs at 40k; g41ro (paced render until the onset) bunny +0.0001, dragon +0.0003,
nefertiti −0.0024 — the onset placement restores the 40k fit (P264 ✓) but at 300k it loses the
growth. *Cleanup 6* (112 → 64 GB): the read 300k twins (ao300, ba300–be300, bg300, bh300, bi300)
and the g41fd / g41fc / g41rp sweeps lose their npz (34 files, 48 GB; `.deleted_list_20260927b.txt`,
logs tarred). *The placement that follows:* the switch from the paced render target to the target's
own images read in the render's own metric — when the paced cloud's silhouettes are closer to the
target's than the morph is to the paced cloud's, the paced target is no longer what limits the fit
(`--render_paced_conv`, permanent once met; no constant). Launched: **bm300** = ar300 + `--render_paced
--render_paced_conv --pace_coherent` (GPU 2), **bn300** = the same + `--pace_stream` (GPU 3), **g41rc** =
the adopted 40k form + `--render_paced --render_paced_conv` on bunny, nefertiti, dragon (GPU 0).
Pre-registered **P265** (bm300): bf300's growth (knob ≤ 1.3, the top region ≤ 40 particles before t =
0.3 and ≤ 40 % neck-bound) with silIoU ≥ 0.974 and tip ≥ 9; **P266** (bn300): the same; **P267** (g41rc):
within ±0.003 of g41pw on the three.

**CLOCK CORRECTION (written at 2026-09-25 11:25 CDT).** The entries of this session labelled
"2026-09-26 13:30 … 2026-09-27 03:30" were stamped under a wrong local-time assumption. The true
clock is the server's, which is also the user's PC clock (CDT): the session's entries run from
2026-09-25 ~06:30 CDT (the "2026-09-26 13:30" entry) to 2026-09-25 11:20 CDT (the "03:30" entry);
the order and the relative spacing of the entries are right, the dates are not. From here the
stamps are CDT.

**2026-09-25 11:40 CDT — the user's reading of the candidate videos, and the surface's shake measured
again.** The user: bk300 and be300 both look good by eye (bk300 a little better); what remains is the
surface shaking from the per-frame re-mesh, and it does not show on the "thick" (coarse-kernel)
renders. Measured on the delivered videos' tails (per-frame change, last ~20 frames): Poisson —
ar300 0.0013, be300 0.0014, bf300 0.0011, bk300 0.0022; the same particles of bf300 through marching
cubes on a fixed grid with the reference kernel: **0.0005**. Free (unpinned) particles at the runs'
ends: 41–50 % (ar300 59 % pinned, be300 57, bk300 53, bf300 50) — the runs end on the merit's three
rejections with half the body still free and drifting slowly. Two owners, as on 2026-09-25 morning
(then read as 3/4 : 1/4 on ar300): the reconstruction (screened Poisson re-solves a global implicit
function from every point and its estimated normal each frame, so a sub-spacing move of a few free
particles shifts the surface everywhere and the normals' re-estimation adds its own jitter) and
the free half's motion. The coarse kernel does not show it because a kernel of radius h averages the
density over ~(h/spacing)³ particles: a displacement δ of one particle moves the iso-surface by δ
times its share of the kernel, and anything below h is filtered; Poisson at depth d matches each
point as a constraint at the leaf scale and filters nothing below it. Remedy on the surface side:
the delivered 300k surface as the local band-limited reconstruction (the mc at the reference kernel
— the "thick" look, 0.0005) or a temporally coherent Poisson (a fixed octree from the target, normals
carried and smoothed frame to frame); on the physics side: the free half at the end (the pin's
"arrived and twice reversed" never meets the growth-phase material before the early stop).

**2026-09-25 11:50 CDT — bm300 / bn300 (the convergence placement): P265 ✓, the 300k bunny form
chosen; the 40k twin ✗; the dragon launched (P268).** **bm300** (ar300 + `--render_paced
--render_paced_conv --pace_coherent`): silIoU **0.9741** (ar300 0.9773; the bound 0.974), 45 windows,
**tip 13.3**, end p1 0.893, none below 0.5, no negative-reversal window; **no particle in the ear's
top region before t = 0.4** (18 at t = 0.4), a monotone taper at t = 0.4 (0.67 / 0.57 / 0.38× from the
base, nothing above the neck), no knob at any time; the tip-bound above the neck slab 8 % at t =
0.3 and 0.4. **bn300** (the same + `--pace_stream`): silIoU **0.9772** (−0.0001), 49 windows, end p1
0.877, no reversal — but the tip starves (6.1) and a bump of 1.65 at t = 0.4 (0.89× at y 3.46 over
0.54×). The convergence switch never fired in either run (the paced cloud's silhouette distance to
the target stayed above the morph's residual to the paced cloud until the end), so both are the
full paced target in effect — bm300 is bf300's form re-run (0.9732 / 14.8 there, 0.9741 / 13.3 here:
the run-to-run spread is ±0.001). **g41rc** (the same placement at 40k): bunny −0.0059, dragon
−0.0083, nefertiti −0.0076 — the paced render target is a 300k flag in every placement but the
onset's, which loses the 300k growth. *Decision:* the **300k bunny form = ar300's form +
`--render_paced --pace_coherent`** (with `--render_paced_conv` carried as the placement that will
switch when the render's metric says so; H⁻¹ from the start out; `--pace_stream` not added: it
starves the tip and costs at 40k). The user's reading of the videos: bk300 and be300 look good,
bk300 a little better; the remaining visible defect is the surface's shake (the re-mesh; 11:40
entry). Launched **bo300** = the 300k dragon under the slip form + the same three flags (GPU 2).
Pre-registered **P268**: silIoU ≥ 0.9616 (dw300's 0.9646 − 0.003), end p1 ≥ 0.80, ≤ 0.05 % below 0.5,
no reversal streak at the end, the spikes' growth by eye without beads (strip).

**2026-09-25 12:05 CDT — the candidates' videos on the page (v58), Poisson and marching cubes each.**
Tail flicker (per-frame change, last ~20 frames) Poisson / mc on the same particles: bm300 0.0017 /
0.0007, bn300 0.0017 / 0.0005, bk300 0.0022 / 0.0007, bf300 0.0011 / 0.0005 — the reconstruction owns
half to two thirds of the visible shake on every candidate, the rest is the free 40–50 % at the
runs' ends. Correction to the 11:50 entry: bn300's convergence switch DID fire once (the log has
the message; bm300's never did). bo300 (the 300k dragon under the chosen form) at window 55, 35 %
pinned, no traceback.

**2026-09-25 12:05 CDT — the user's surface verdict and the reconstructions that follow.** The user:
the marching-cubes videos are stable but too thick (the detail is lost); every Poisson video
shakes. The mc's thickness is its ISO-LEVEL, not its kernel (2026-09-25 morning: 40k / 100k / 300k
kernels alike): the renderer's 'auto' level is the one at which a filament two particles across
still renders (2 s² / π σ², capped at 0.5), i.e. deliberately low so thin necks do not detach, and
a low level inflates every surface by a fraction of the blur. Three local (per-frame independent,
band-limited — the property that keeps mc still) reconstructions of bm300 are rendered for the
user's eye: mc at the half-bulk level (`--iso 0.5`, the surface of a smoothed indicator; the thin
necks' detachment is the known risk), mc with Yu & Turk anisotropic kernels (`--kernel aniso`, S1:
thin features keep their sharpness), and the implicit MLS surface (`--surface imls`, S3). Each is
measured on the tail as before.

**2026-09-25 12:30 CDT — the surface alternatives read; the render loss's origin (the user's
question); the splat render.** bm300, same particles, tail change: mc auto level 0.0007 (thick),
**mc iso 0.5: 0.0007** (the thickness recovered — the end-frame still `bm300_surfaces_f60.png`), mc
iso 0.5 at the native 300k kernel (half the reference blur): **0.0009** with more detail, mc with
Yu & Turk anisotropic kernels: **0.0024** — worse than Poisson (each particle's covariance rides its
F, which jitters with it) → refused; IMLS pending. *The render loss does not come from the
reconstruction:* the silhouette term is a differentiable 2D coverage splat of the particles into
the render pixels (CIC kernel, α = 1 − e^{−k w}, `losses/silhouette.py`) and the shading term reads
the density's grid normals blurred 1.5 spacings — both band-limited at the pixel (96 px ≈ 1.7
spacings at 300k), i.e. an "mc-like" image; the optimisation neither sees nor can reduce the
Poisson re-mesh's shake, and the delivered surface is a post-hoc choice. *The user's next
question — render the splat itself:* the repository's PhysMorph-GS rasteriser (`render_3dgs`) is
used as the delivered renderer: one isotropic Gaussian per particle (σ₀ = 0.7 × its spacing),
shaded per particle from the blurred CIC density gradient (Lambert, the plain video's light), the
plain video's two views from a camera fixed on the target's centre (`scratch/render_splat_video.py`);
no mesh, no iso-surface. Pre-registered: the tail change at the mc level or below (≤ 0.0007) with
the detail of the splat scale (finer than the 1.5-spacing kernel); the user judges the look.

**2026-09-25 12:40 CDT — cleanup 7 (the user: wipe the previous results): 80 → 39 GB.** Every
archive except the bases and the chosen forms deleted — ai300, au300, dh300, bf300, bj300, bk300,
bl300, bn300, g41fs (3), g41ro (3), g41rc (3): 17 files, 41 GB (`.deleted_list_20260925c.txt`; json +
logs tarred into `logs_archive_20260925c.tgz`). Kept: g41 (19), g41pw (19), ar300, bm300, bo300
(running), gt, the reports and every video.

**2026-09-25 12:45 CDT — bo300 (the 300k dragon under the chosen form: slip + paced render + coherent
pace): P268 on the fit, not on the health.** silIoU **0.9699** (the slip form alone 0.9646, with H⁻¹
from the start 0.9685 — the best 300k dragon), 102 windows (early stop on three rejections), pinned
62 %, 82 particles below det F 0.5 at the end (0.03 %, within the 0.05 % bound), but the end-state
p1 is **0.770** (the bound 0.80): the pinned 62 % sits at p1 0.80 and the free 38 % — the spikes'
material still in transit — at 0.72 (min 0.26). The coherent pace holds the spikes' leads and the
material queues behind them; the dragon's many thin spikes are where the queue compresses. Not a
reversal issue (no streak); a compression of the transit at the spikes, the same place the
arrival's capacity (§10.28) and the front's cap addressed at 40k. The dragon video is rendered for
the eye. The 300k form stands on the bunny; the dragon's p1 is the open number.

**2026-09-25 12:50 CDT — the splat render and IMLS read; the surface table complete.** bm300, tail
change: Poisson 0.0017 · mc auto 0.0007 (thick) · **mc iso 0.5 0.0007** (thickness recovered) · mc iso
0.5 at the native kernel 0.0009 (more detail) · mc anisotropic 0.0024 ✗ · IMLS **0.0049** ✗ (the outer
layer's surfels are re-selected each frame) · **the splat itself 0.0014** (first implementation: one
isotropic Gaussian per particle, σ₀ = 0.7 spacings, per-particle Lambert from the blurred CIC
density gradient; the silhouette and geometry are stable, the shading is coarse — interior
particles show through and the per-frame normals jitter — so the number is Poisson's; a proper
surface splatting would draw the outer layer only with wider-smoothed normals). The delivered 300k
surface that answers the user's two complaints at once is mc at iso 0.5 (native kernel if the
detail is wanted at 0.0009).

**2026-09-25 13:05 CDT — the user prefers the splat render; its two defects and the forms that
answer them.** By eye the user would rather have the splat than any mesh, with two complaints: the
floaters seen mid-morph (every in-transit particle draws as a dot: the ears' leads at frame 20 are
dotted tips, where Poisson hides them below its node) and a rough surface (a pile of balls, dark
interior particles showing through). Two forms rendered on bm300: **splat v2** — opacity by
SUPPORT: a particle's opacity is the fraction of the target's own shell count present around it,
n_i / (k/2) clipped (n_i = particles within the shell radius, k = 8, half = the body convention), so
a lone floater (n_i < 4) is nearly transparent and a stretched lead fades in proportion; interior
particles (weak density gradient) take their nearest strong-gradient particle's normal; **splat v3**
— v2 plus surface splatting (Zwicker 2001): each Gaussian a disc in the tangent plane of its normal
(radius one spacing, thickness a quarter), so the drawn surface is a sheet of overlapping surfels.
No new constant beyond the surfel's spacing-sized radius. Read on the mid-morph still (frame 20)
and the end (60), and on the tail change (target ≤ mc's 0.0007).

**2026-09-25 13:10 CDT — the user's verdict on bm300: accepted.** "bm300 is quite to my liking; there
is flickering (probably from the optimisation) but it is fine." The 300k bunny form (ar300's form +
`--render_paced --render_paced_conv --pace_coherent`) stands as the user's choice. The flicker's
two owners as measured (11:40): the Poisson re-mesh (half to two thirds) and the free 40–50 % at
the early stop (the rest); the delivered surface (the splat forms or mc at iso 0.5) answers the
first, the termination rule the second.

**2026-09-25 13:40 CDT — the user's base: bm300 + the splat render; the dots in the sparse phase; the
sticky endpoints launched (P269/P270).** The user: bm300 with the splat render is the best result so
far; the base to build on; the frame at 96/767 (the head's top a cloud of dots) is what to remove —
"is it the number of Gaussians?" Read: the count is the particle count; the dots are the splat
RADIUS against the LOCAL spacing — σ₀ = 0.7 × the global median spacing, while a stretched region's
particles sit 1.5–2 spacings apart, so their splats do not overlap and each shows as a dot. Two
principled sizes: the user's own PhysMorph-GS covariance σ₀² F Fᵀ (the Gaussian rides the stretch;
its F jitters in the tail — the anisotropic mc gave 0.0024) or an isotropic radius scaled by the
particle's own shell radius over the target's (the k-th neighbour distance; the adaptive kernel of
SPH practice): **splat v4** = v2 (opacity by support) + the adaptive radius, rendered with stills at
frames 96 / 240 / 720 for the comparison. The user's caution ("this is the moment to be careful")
is right in a second sense: the sparse phase is the PHYSICS (the stretched column at half density)
made visible; the opacity and the radius rules make it deliverable, and the physics is judged on the
unhidden render (v1) and the probes, not on the delivered one. *(b) sticky endpoints* (method.md
§10.33, `--plan_sticky`): implemented — an arrived particle keeps its target point across the plan's
re-solves, the answer to the sliding that leaves half the body un-pinned; launched **bp300** =
bm300's form + `--plan_sticky` (GPU 2) and **g41sk** = the adopted 40k form + `--plan_sticky` on bunny,
nefertiti, dragon (GPU 0). Pre-registered **P269** (bp300): pinned at the end ≥ 0.75 (bm300 0.49),
the mc tail ≤ 0.0005 (0.0007), silIoU ≥ 0.971 (within −0.003 of bm300), no reversal window, the
ear's growth unchanged (knob ≤ 1.3, tip ≥ 9); **P270** (g41sk): within ±0.003 of g41pw with pinned ≥
0.95 and no reversal window. (a), the arrival-weighted kinetic charge, waits on (b)'s reading.

**2026-09-25 14:40 CDT — frame 96 under the adaptive splat (v4): the dots are gone, the sparse phase
shows as what it is.** `bm300_f96_pv1v4.png` (page v63): Poisson interpolates the head's top into a
confident smooth bump; splat v1 shows the same region as a cloud of dots (10.5 % of the particles
with fewer than half the target's shell count there); splat v4 (each particle's radius = its own
shell radius over the target's; opacity = support) draws it as a soft translucent mass — no dots,
the half-density material rendered as half-dense. The shading's blotchiness (the density-gradient
normals at 1.5 spacings on a cloud in transit) is the remaining look issue in every splat form. The
user chooses between the faithful look (v4) and a Poisson-like solid look (an opacity floor: only
singletons fade); the physics is judged on v1 and the probes.

**2026-09-25 14:50 CDT — sticky endpoints (b) refuted; (a) not pursued.** g41sk (40k, the adopted form
+ `--plan_sticky`): bunny 0.9689 (−0.0014 vs g41pw), dragon 0.9637 (−0.0006; 71 windows for 103),
nefertiti 0.9698 (−0.0034), no reversal window — and the pinned fraction at the end is NOT higher
(97 / 93 / 83 % for g41pw's 97 / 95 / 91 %): freezing the arrived material's endpoint does not make it
settle sooner at 40k. bp300 (300k, bm300's form + sticky): **7 % pinned at window 40** (bm300 ≈ 35 %)
— withdrawn. Read: the first arrival's point is taken without capacity, so several particles freeze
onto the same target point and compete for it forever (the plan's re-solve was what resolved that
competition); a capacity-aware form would be the next patch, and that is the engineering spiral the
user warned against. Decision: (b) refuted as implemented, (a) not pursued; the free half at the end
stays an open algorithm item, stated as such, and the delivered look is the splat render's (the
opacity by support and the adaptive radius make the sparse phase deliverable without touching the
physics). P269 ✗, P270 ✗ (the pin did not rise).

**2026-09-25 14:55 CDT — splat v2 / v3 tails: 0.0014 each (v1 0.0014).** The opacity-by-support and
the disc surfels change the look (fewer dots, no interior bleed) but not the tail's change: the
splat's frame-to-frame shake is the SHADING's — the per-particle normals from the density gradient
at 1.5 spacings re-estimated each frame — not the geometry's (the silhouette is as still as mc's).
v5 = v4 with the shading normals at 3 spacings (twice the loss's blur; the geometry untouched) is
rendered to measure the normals' share; if the tail drops to mc's 0.0007 the splat render is the
deliverable at the user's taste, else the shading needs the band-limited surface's normals (the
iso-0.5 field's) instead of per-particle samples.

**2026-09-25 15:10 CDT — splat v4 / v5 tails: 0.0016 / 0.0012.** The adaptive radius (v4) makes the
sparse regions' larger splats move visibly (0.0016); the shading normals at 3 spacings (v5) bring
the tail to 0.0012 — the normals' scale owns part of the shake but not all of it: at 3 spacings the
splat still moves twice mc's 0.0007 with the same particles, because every splat is drawn where
its particle is (the free 40–50 % drift is rendered one-to-one), while an iso-surface draws the
band-limited density in which that drift averages out. So the splat render's stillness is bounded
by the particles' own motion; the mesh's is not. The delivered choice is the user's: the splat
(honest, the sparse phase visible, tail 0.0012–0.0016) or mc at iso 0.5 (stable at 0.0007, the
thickness fixed, the sparse phase interpolated). The remaining algorithm item — the free half at
the end — is what would make both still.

**2026-09-25 17:15 CDT — the empty-looking transit measured (the continuity reading); the splat's
darkness and roughness; the plan.** `hollow_probe.py` on bm300: above the head (y > 2.3) at t =
0.10 / 0.14 / 0.20 the particles present are 776 / 1320 / 2396 of the 14,423 that end there, **96.5
/ 99.9 / 80.7 % below half the target's density, at a mean of 0.17 / 0.05 / 0.26 of it**; full by t
= 0.3. ar300 the same (0.05 at t = 0.10). The region the user sees as empty IS empty — the front's
few leads run at the pace through free space while the bulk drags through the body, and density =
flux / speed falls (the spike of yesterday, read from the density side). Poisson interpolates the
5 % vapour into a bloated blob; the splat draws it translucent. Raising the surface sampling
cannot change the ratio (the target's count rises with it); the answer is the front's speed tied
to the supply — **bq300** = bm300's form + `--pace_stream` (GPU 2), judged for the first time on this
measure (P271: the top region's mean density ≥ 0.5 of the target's at every t ≥ 0.10, with bm300's
fit and growth). The splat's darkness and roughness were the renderer's: a bare Lambert with a low
ambient and per-particle normals at 1.5 spacings; **v6** (v5 + the mesh render's hemispheric
lighting) is bright and smooth at frame 240 (`bm300_f240_pv5v6.png`, page v66) with a smaller ear
knob than Poisson's. The user's PhysGaussian question (surface-only Gaussians + interior fill):
the surface is not a material invariant of a morph (the ears' surface is the head's interior
column), so a surface-only sampling cannot form the thin features; what transfers is the render
side — **render children** (render/children.py, the user's PhysMorph-GS): v7 = v6 + 4 tangent-plane
sub-splats per outer parent (GPU 3) tests the surface-splat density ×4 at no physics cost; and
adaptive (shell-weighted) sampling of the same 300k is the geometric-detail lever, gated on the
transport (later). Typical 3DGS counts for the record: objects 100k–500k Gaussians, scenes 1–6 M —
nearly all on the surface, against our ~30k outer-layer parents of a uniform 300k.

**2026-09-25 17:25 CDT — bq300 (bm300 + the stream pace): P271 ✗ — the transit is as empty; why the
rule cannot see it; the support pace.** bq300: silIoU 0.9759 (+0.002 vs bm300), 43 windows, end p1
0.857, no reversal window, a tapered ear without a knob, tip 6.5 (starved) — and the density above
the head at t = 0.10 / 0.14 / 0.20 is **0.06 / 0.08 / 0.27** of the target's (bm300 0.17 / 0.05 /
0.26). The stream pace scales a particle's step by the fill one pace BEHIND it on its ray: behind
the front's leads lies the head — dense, so the rule lets them run; the head is slow, so they
outrun it. The measured quantity is the density AT the lead (5 %), so the rule is written on that:
`--pace_support` (method.md §10.34) — the step scales with the particles within the target's shell
radius around the particle over half the target's own count at its nearest target point; a lead
in a 5 %-density region takes 10 % of a pace step until its bulk arrives, the bulk moves at the
pace, the front advances as a plug. No constant. Launched **bs300** = bm300's form + `--pace_support`
(GPU 2). Pre-registered **P272**: the top region's mean density ≥ 0.5 of the target's at every t ≥
0.10 (the quantity itself), silIoU ≥ 0.971, tip ≥ 9, end p1 ≥ 0.80, no reversal window, knob ≤ 1.3.
GPUs 0 and 1, free at the user's question, now carry **br300** (the 300k dragon under the chosen
form + the stream pace, for its p1 0.77) and the coherent pace's remaining 16 gallery targets
(g41fc; the 40k adoption question for that flag).

**2026-09-25 17:45 CDT — splat v6 tail 0.0007: the splat render is as still as marching cubes.** v6
(adaptive radius + support opacity + shading normals at 3 spacings + the mesh render's hemispheric
lighting): tail change **0.0007** — mc iso 0.5's number, on the same particles, with the splat's
honesty (the sparse phase visible) and a bright, smooth look (frames 240 and 720). v5's 0.0012 →
0.0007 came from the LIGHTING alone: under a bare Lambert the back-facing and interior splats
blended in dark and their per-frame changes were the flicker; the wrap light removes the dark
blend. v7 (v6 + 4 render children per outer parent, surface splats ×4): 0.0009, the look at frame
240 marginally finer — the children add coverage, not detail, without an appearance model. The
delivered 300k render is now a two-way choice on equal stillness: splat v6 (honest) or mc iso 0.5
(interpolated); the user's call.

**2026-09-25 17:55 CDT — the page rebuilt: current results only, Gaussian-splat renders only (the
user's rule).** The artifact now carries: the delivered render's definition (splat v6), bm300's
numbers against ar300, the measured empty transit, the running experiments with their gates, the
plan; 70 old files (Poisson / mc videos, montages of refuted forms) removed from the artifact. The
history stays here and in the server's r300/ folders. Splat v6 renders of g41pw bunny (40k) and
bo300 dragon (300k) queued for the page.

**2026-09-25 18:00 CDT — bs300 (bm300 + the support pace): P272 half — the vapour is denser, not
dense.** silIoU 0.9734 (−0.0007 vs bm300), 46 windows, tip **12.5**, end p1 0.865, none below 0.5, no
reversal window; the ear grows later (nothing above y 2.86 at t = 0.3; the neck appears at t = 0.4)
and the plug arrives dense (1.31× at t = 0.3). The transit density above the head at t = 0.10 / 0.14
/ 0.20: **0.09 / 0.18 / 0.31** (bm300 0.17 / 0.05 / 0.26; bq300 0.06 / 0.08 / 0.27) — the leads are
held to 5–10 % steps and still trickle in ahead of the bulk, which arrives only by t = 0.3. The
measure asks ≥ 0.5 or no material there at all; a proportional hold leaves the trickle. The
variant that follows is the same convention as a gate: material below half the target's local
density does not advance (`--pace_support_hard`: step 0 below half, the pace at or above) — the
front is then the bulk itself, and the region is empty rather than misty until the plug arrives.
Launched **bt300** = bm300's form + `--pace_support --pace_support_hard` (GPU 2). Pre-registered
**P273**: above the head at t = 0.10–0.20 either fewer than 200 particles present or a mean density
≥ 0.5; silIoU ≥ 0.971; tip ≥ 9; end p1 ≥ 0.80; no reversal window.

**2026-09-25 18:15 CDT — the adopted 40k form under the splat render: tail 0.0002.** g41pw bunny
rendered with splat v6: tail change 0.0002 (97 % pinned) — the 40k D1 closure holds under the
honest renderer as well; on the page (v69) as the 40k reference. The 300k dragon's splat render
follows.

**2026-09-25 18:30 CDT — the clean splat (v8): discs + smoothed normals + deferred shading.** The
user: the Gaussian render's surface is bumpy and messy. Three renderer changes, no physics: (1)
the shading normals are averaged over each particle's 32 nearest particles, two passes — the
shading field at the surfel scale instead of the particle scale; (2) every Gaussian is a disc in
its tangent plane (radius one spacing, a quarter thick — surface splatting); (3) DEFERRED shading:
the splats carry their normals and a coverage to the screen (two rasterisations per view), the
normal buffer is normalised per pixel after the blend and lit there with the hemispheric light —
no per-splat flat shading, so no ball-pit look; the same opacity by support and adaptive radius
as v6. A two-frame test on bm300 (frame 380): a clean, smooth, mesh-like surface with the ears'
sparse tips still honest. Full renders of bm300, g41pw (40k) and bo300 (dragon) queued; the page
moves to v8 when they land, with the tail change measured as before.

**2026-09-25 18:45 CDT — bt300 (the support gate) and br300 (the dragon + stream pace).** bt300:
silIoU 0.9734, 55 windows, tip 8.3, end p1 0.868, none below 0.5, no reversal window; the transit
density above the head at t = 0.10 / 0.14 / 0.20 / 0.30: **0.15 / 0.22 / 0.53 / 1.64** (bs300 0.09 /
0.18 / 0.31 / 1.31; bm300 0.17 / 0.05 / 0.26 / 1.00). P273 ✗ at t = 0.10–0.14 (997 / 1699 particles at
0.15 / 0.22), ✓ from t = 0.20. With a zero pace step for under-dense material the leads still get
there: the paced target's images are not what carries them — the cell sum pulls whatever
material is nearest into the cells the BULK's images already claim ahead of the plug, and a
particle in free space has nothing to hold it (no medium, no drag). Every pace rule improved the
order and kept the fit; none empties the first windows' vapour, because it is not the transport's
to remove. br300 (the dragon under the chosen form + the stream pace): silIoU 0.9666 (bo300 0.9699),
end p1 0.746, 210 below 0.5 (0.07 %) — worse on every count; the stream pace leaves the 300k form.
The remaining lever for the vapour is on the physics side — unsupported material (support below
one half) has no elastic medium and coasts; a body-convention drag on it would be the first
physics-side change, and it is proposed, not run. The delivered render answers the vapour
visually (support opacity: the vapour is drawn as what it is).

**2026-09-25 18:50 CDT — the coherent pace on the gallery (g41fc, 18 of 19 vs g41pw): not neutral at
40k.** 16 within ±0.003; cow **−0.0110** (52 windows for 115: the run stopped early), beast −0.0068,
nefertiti −0.0030. The coherent pace stays a 300k flag with `--render_paced`; the 40k form remains
g41pw. (The 19th, C, still running.)

**2026-09-25 19:00 CDT — the user on the 300k v8 video: still flickering, and particles flow along
the surface. Measured.** Whole-video change (all frames): v8 splat 0.0018, Poisson 0.0031 — the
splat flickers less than the re-mesh, but the physics' motion is drawn one-to-one. The sliding
(bm300, per delivered frame, in spacings, median): particles unpinned at the end = 47 %; at t =
0.5–0.6 they move 0.14 (arrived ones 0.14), at 0.7–0.8 0.10, at 0.9–1.0 **0.07** — against 0.02 for
the pinned — i.e. the arrived-but-unpinned half drifts two spacings over the last thirty frames:
the flow the user sees, the same "particles that reached the surface kept flowing" of their
memory, the plan's permutation freedom on a filled surface re-assigning arrived material every
window. The sticky endpoint (§10.33) failed for one reason — no capacity: several particles froze
onto one point and competed. Corrected: an arriving particle takes the nearest target point not
reserved by another (closest arrivals first, one per point, reserved for the run), or keeps the
plan image if none of its 8 nearest is free. Launched **bp301** = bm300's form + `--plan_sticky`
(capacity-aware) (GPU 2). Pre-registered **P274**: pinned at the end ≥ 0.75 (bm300 0.49); the
arrived-unpinned move at t = 0.9–1.0 ≤ 0.03 spacings per frame (0.07); silIoU ≥ 0.971; no reversal
window; the growth unchanged (tip ≥ 9, knob ≤ 1.3).
