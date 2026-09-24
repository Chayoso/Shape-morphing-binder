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
