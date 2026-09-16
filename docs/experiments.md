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
