# Gradient analysis — how the render signal drives the physics (v2 vs PhysMorph-GS v1)

Measured 2026-09-01 on hyde06 (RTX 6000 Ada, GPU 1) with `scripts/grad_analysis.py`
(probe scale: N=5000, T=10, dx=0.5, dt=1/240, isosphere→bunny, probes at commits 0/3/6/9;
raw JSON: `output/grad_analysis.json`), plus forensics of archived v1 runs
(`output/step0_sweep_lam0.npz`, `output/k120_lam0.npz`) and the full-scale v2 A/B
(`output/v2_full.json`, N=20000, T=20, 30 commits — §9 of pipeline_v2.md).

---

## 1. The gradient chain (what actually flows)

```
D_render(x_T)                     asymmetric multi-view silhouette (render_loss.py)
   │  ∂D_render/∂x_T              CIC splat adjoint: per-particle 2-D pull, per view
   ▼
x_T  ─── wp.Tape adjoint ───────  T reverse MPM steps (function.py::_WarpMPM.backward)
   │        k_update† → k_g2p† → k_grid_op† → k_p2g† → k_stress†   per step
   ▼
∂L/∂dFc[t]  (T,N,3,3)             control sequence  → line-searched Adam (channel 1)
∂L/∂λ_i, ∂L/∂μ_i (N,)             material leaves   → same tape backward (channel 2)
```

Two more channels carry render information into physical state *between* windows:
`Fp ← S_e^η Fp` assimilates the elastic stretch of the render-optimised motion
(channel 3, exact — pipeline_v2.md §3.5), and the terminal kinetic loss `w_kin·mean‖v_T‖²`
forces the optimised controls to arrive at rest (channel 4), which is what makes the
*promoted* velocity quiet (full scale: |v|max 28.7 → 0.12 across 30 commits).

Key structural fact: the stress kernel uses `F_e = (F + dFc)·Fp⁻¹` (kernels.py:40), so a
control gradient is literally a stress gradient — the render signal changes the *forces*,
not the positions. Everything downstream (APIC transfer, grid momentum, advection) stays
physical; there is no displacement injection anywhere in the blessed path.

## 2. Was the v1 render term really inert? (measured, not asserted)

Per-term gradient norms **on dFc, through the MPM adjoint**, at the window-start decision
point:

| commit | ‖∇D_vol‖ | ‖∇D_render‖ | raw ratio | v1 contribution (fixed λ=0.5) | v2 λ_R | v2 contribution |
|---|---|---|---|---|---|---|
| 0 | 10.5 | 0.00874 | 1200× | **4.2e-04** | 602 | 0.5 (by construction) |
| 3 | 2.98 | 0.0178 | 167× | 3.0e-03 | 149 | 0.5 |
| 6 | 1.55 | 0.00775 | 200× | 2.5e-03 | 142 | 0.5 |
| 9 | 0.93 | 0.00616 | 151× | 3.3e-03 | 69 | 0.5 |

v1's fixed λ=0.5 gave the render term **0.04–0.3% of the update** — the "~45000x gap"
folklore, now measured at 150–1200× in gradient norm on this problem: the render channel
mathematically existed and practically did nothing. v2's norm-balanced λ_R pins the render
contribution at α_λ = 0.5 of the physics gradient *regardless* of the raw scale gap, and
the balanced λ_R anneals itself (full scale: 1130 → 61) as D_render converges — the render
term hands control back to the physics as its own objective saturates.

## 3. Is the render gradient redundant with D_vol, or new information?

Cosine alignment of ∇D_render vs ∇D_vol on dFc:

| commit | cos(∇ren, ∇vol) |
|---|---|
| 0 | **+0.68** |
| 3 | −0.07 |
| 6 | −0.52 |
| 9 | **−0.74** |

Early on the two agree (both say "move sphere mass toward the bunny bulk"). As the bulk
converges the render gradient rotates to *oppose* the mass gradient: it encodes rim/
silhouette corrections that the coarse (32³) mass grid cannot see and partially trades
against. This is the quantitative version of the thesis: **the render signal contains
shape information that 3-D mass supervision does not**, and it only becomes usable when λ
is balanced (v1's inert weighting silenced it precisely in the regime where it disagrees).
Consistent with the outcome: full-scale sil_iou 0.8885 (phys) → 0.9536 (render) and
final-frame holes 0.05% → 0.00% at identical budget.

## 4. Surface dominance (why "surface-only feedback" is already true)

Mean |∂D_render/∂x_T| in the sparsest-decile band (surface) vs densest decile (interior):

| commit | render pull, surf/interior | D_vol pull, surf/interior |
|---|---|---|
| 0 | **15.0×** | 3.4× |
| 3 | **15.5×** | 1.5× |
| 6 | 7.7× | 1.1× |
| 9 | **12.9×** | 1.2× |

The CIC alpha saturates in the interior (`1−exp(−k·w)→1`), so image-space pull concentrates
8–16× on the surface band while D_vol pulls near-uniformly. The render channel is
effectively a surface force field riding on a volumetric backbone — exactly the division of
labour a Gaussian-surface deliverable needs, with no explicit surface extraction.

## 5. The old failure modes — mechanism → fix → evidence

| v1 symptom | mechanism (v1) | v2 fix | evidence |
|---|---|---|---|
| render had no effect | fixed λ vs 150–1200× gradient gap → contribution ≤0.3% | per-window norm-balanced λ_R (EMA), single objective per window | §2 table; G5 PASS at both scales |
| noisy/unstable F | greedy per-frame dFc reset + silent SV clamp [0.5,2] *rewriting* F every frame; reflection-preserving conditioning | sequence optimisation + line search; conditioning repairs only non-finite/reflections (counted, gate G2); elastic-stretch assimilation keeps F_e moderate | v1 k120: aniso max **2.7**, detF min **0.32** (clamp active!); v2 full scale, **no clamp**: aniso max 1.70, p95 1.40, detF min 0.51; guards all 0 |
| visible ellipsoids in renders | Σ=σ₀²FFᵀ with noisy F → splat aspect up to ~2.7:1+, exposed through holes | F health above + hole-free surfaces | splat aspect bounded by F aniso ≤1.7; holes 0.00% |
| mass ejection | large accepted steps (no line search), no objective term saw escapees (render/D_vol gradients vanish outside viewport/grid) | line search rejects blow-ups; `w_spray` penalises in-view excess; box leash `relu(|x|−r)²` has gradient everywhere; velocity clamp REMOVED | outside_frac 0.000% both scales; `no_box` ablation currently identical (leash is a pure safety net) |
| holes / tearing during morph | render arm tore the body (v1 diagnosis via hole_frac); symmetric MSE seeing deficit and excess equally | asymmetric `w_hole=2` (deficit inside target weighted 2×) + full-state promotion (no energy re-injection) + assimilation (no spring-back) | full scale: hole 0.00% (render) vs 0.05% (phys); per-frame QA strip 0.0% at every sampled frame |
| in-place oscillation ("제자리 진동") | greedy loop with partial-state promotion re-released stored energy every frame; no rest objective | full (x,F,v,C) promotion + w_kin terminal rest + elastic assimilation + λ-free plateau freeze | v1 step0 tail jitter_rel **0.00485** (fails G3); v2 full scale **0.00003**, drift_rel 0.0004; kin 8.97→0.002 monotone |
| ellipsoid/hole flicker between frames | oscillation + per-frame geometric post-ops (pull_outliers, max_move, repel/taubin) teleporting particles | post-ops deleted; motion is MPM-integrated only; freeze holds converged shape | jitter above; G2 guards 0 (no clamp ever fired) |

Ablation notes (grad_analysis.json): `no_assim` doubles residual kin (0.099 vs 0.047 at
commit 10) — assimilation is the settling mechanism, not just anti-spring-back; `no_box`
is currently a no-op because nothing escapes — the leash costs nothing inside the box and
exists for the regimes (higher λ, softer material) where the adversarial round showed
escape is unrecoverable by pixels alone.

## 6. Engineering differences (v1 → v2)

| axis | PhysMorph-GS v1 | v2 |
|---|---|---|
| code paths | 4 divergent loops (greedy, trajectory, displacement-space, style), Type-A and Type-B coupling mixed | 1 blessed path (`pipeline/`); baseline = same path, λ=0 |
| stabilisation | 7+ non-physical operators mutating state after the physics | none; objective terms only (kin, ctrl, box, asym render); guards must read 0 |
| stepping | fixed-step Adam + clipping | backtracking line search + adaptive α; acceptance requires finite (x,F,v) state |
| λ policy | hand-tuned constant | per-window norm balancing + EMA, λ-free freeze tracking |
| measurement | metrics shared operators with the loss; per-frame autoscaled extents; jitter blind to held frames | loss-independent binary-splat metrics at one fixed target extent; held-aware jitter; provenance embeds full discretisation |
| verification | none automated | 34 CPU tests incl. e2e smoke; FD gradient gates (G1b) at every run start; adversarial review protocol (Codex+Opus, 26 findings closed) |
| memory/perf | tape built even for no-grad evals; extra logging rollout per iter | no-tape eval path for line search; history reuses accepted evaluation |
| GPU workflow | ad hoc | gates G1–G5 machine-checked per run; G6 scripted (quicklook/gif) |

## 7. Open items the analysis surfaces

1. **cos → −0.74 late**: render and mass objectives increasingly trade off; the composite
   optimum depends on α_λ. A small α_λ sweep (0.25/0.5/1.0) at full scale would show
   whether more render weight buys ear/paw sharpness without D_vol regression.
2. **λ_R self-annealing** is a nice emergent behaviour but currently unbounded above;
   consider a cap if a future target's render gradient is pathologically small at t=0.
3. `no_box` no-op at current settings — keep the leash; revisit if material softens.
4. Extremity sharpness at full scale is good (ears/paw form) but tips are still rounder
   than target; candidates: more azimuths near the ear plane, higher render_res late in
   the run (coarse-to-fine), or the material channel (`render_mat` arm, not yet run at
   full scale).
