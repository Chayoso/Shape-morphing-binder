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
| b8_gate_v3 | gate v3 (shape-only merit) + best-commit truncation | 450 | pending | | | hyde01 GPU1 |
| e4_nnfar | nn_far_k 4.5 → 1000 (own far clumps) | 300 | pending | | | hyde01 GPU1 |

Probes (docs/probes/): transfer_function (adjoint gain 0.89 vs 1.03 — §2 refuted),
sobolev_precond (no-op; s≤0.8 falsified; render descends at parity), material_carrier
(Tier M NO-GO: dFc reproduces material motion at 0.9% residual / 1/1700 cost),
observability (97% of floaters see the silhouette; median cos to target 0.3; surface-parent
gauss blind to the interior half). Ops: hyde06 key rejected from ~12:10; b8/e4 moved to
hyde01 GPU1 (warp-lang 1.16 installed into miniconda3/envs/diffmpm_v2.3.0).
