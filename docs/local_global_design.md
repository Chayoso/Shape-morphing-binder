# Local(render) + Global(physics) split solver — design v2 (post-REFUTE)

Status: **DESIGN v2 — answers both adversarial rounds; not implemented**. Date 2026-09-02,
branch `v3-grid-gs`.

## §0 Changelog

- **v1 (2026-09-02, this file's git history)**: two-tier local channel — Tier D
  (observation-layer dressing) + Tier R (rest-state demand through assimilation).
  Adversarial verdicts: **REDESIGN** from both reviewers
  (`docs/reviews/refute_lg_codex_20260902.md`, 16 findings, 10 BLOCKER;
  `docs/reviews/refute_lg_opus_20260902.md`, B1–B5, M6–M19, m20–m26).
- **v2 (this document)**:
  - **Tier R is WITHDRAWN.** The two reports independently proved the same three
    kills: the Fp demand lies in the exact reachable range of `dFc` and is cheap to
    annul while `w_kin` prices its realization (Codex 8 / Opus B2); the non-coaxial
    demand component is absorbed into a rotation that a corotational energy turns
    into zero stress and zero motion (Codex 1 / Opus M6); and the band solver's
    correctness gate cannot fire before sweep 2, so the common case is full cost,
    zero donation (Opus M10). §5 records the withdrawal; **Appendix A** sketches the
    strongest replacement carrier (per-particle material remodeling — a demand
    outside span(dFc)) with its own mini-falsification plan. The appendix is
    **deferred** and does not gate Tier D's ladder.
  - **Tier D proceeds as a separately-gated arm** after mandatory repairs:
    d_sil/d_gauss telemetry split with all gates and freeze tracks reading d_sil
    only (Opus B1 / Codex 9); world-space offset cap (Opus M14 / Codex 15); the
    per-child log-scale DOF is **dropped** (Opus M15); exact joint feasibility for
    centroid + cap (Codex 15); per-frame dressing archive (Codex 16); c2f dressing
    persistence (Codex 12); envelope claim demoted to an empirical bet
    (Codex 11 / Opus M13); energy-based stop rule (Opus M11).
  - Settled in production 2026-09-02 (cited, not redesigned): exact KKT projection
    for the SV-band ∩ det constraint in assimilation (Codex 2 / Opus m20); c2f
    `configure_source` re-call (Opus B5 / Codex 12-part); work telemetry hoisted out
    of the `on_iter` guard (Opus M18).
  - New **§13**: disposition line for every finding of both reports.
  - Cost honesty: stage-0 GPU microbenchmark is now a hard gate before any GPU
    ladder (Codex 14); stage matching is on **commits with `cfg.pace` held fixed**,
    wall-clock reported (Opus M19).

**Mechanism in one paragraph (v2).** The global solver is untouched: MPM windows,
volumetric + hybrid render losses, full adjoint to `dFc`, line search, outer
fixed-merit gate. One LOCAL channel remains — **Tier D dressing**, a pure observation
change: the existing massless render children gain bounded tangent-plane position
DOFs `(a,b)` (no scale, no opacity), capped in **world space** at half the target
surface sample spacing, and optimized once per *accepted* commit directly against the
Gaussian image loss with no MPM adjoint in the loop. During the next window the
dressing is frozen, and — decisively, after REFUTE — the outer merit gate, the freeze
plateau tracks, and every commit-acceptance input read the **pure silhouette scalar
`d_sil` only**, so dressing can neither launder a commit nor block the freeze; it can
only spend the high-frequency render signal where it lives (sub-Nyquist surface
arrangement) and report its effect through separate `d_gauss` telemetry and visual QA.
The rest-state demand channel of v1 is withdrawn: any force-like demand is inside the
control's exact reachable range and gets cancelled, not realized. The one in-contract
carrier the control cannot cancel — per-particle material remodeling, which changes
the *response operator* rather than injecting a force — is sketched in Appendix A as
a deferred, separately-reviewed candidate.

---

## §1 Problem restated (unchanged diagnosis, corrected guard reading)

`docs/root_analysis.md` (all measured): the visually important modes — surface-local,
fine-scale, thin features — are

- **R1 unobservable**: CIC/silhouette are low-pass; floaters live in the loss null
  space (nn-band forensic: making one null-space mode observable killed its class);
- **R2 uncontrollable**: the render covector's unique content is high-frequency; the
  cubic-B-spline P2G/G2P adjoint plus F-smoothing 0.955 attenuate exactly that
  (render work share ~0.02%, unchanged after the sub-pixel splat fix; dx=0.5 vs
  splat σ≈0.04 — a 12× scale gap crossed twice by the gradient);
- **R3 flat**: near-optimum curvature in those modes is ~0; the iterate wanders
  the valley (oscillators ON the surface at 1.2× spacing — the dossier's direct
  forensic, not an inference from `move`; cf. Opus m25).

Why the retired `surface_local.py` failed — **corrected after Opus M16**: its
documented failure precondition was twofold, (i) *state overwrite* (writing `x/F`
after the window, leaving `v`/`C`/interior inconsistent) AND (ii) *objective
exclusion + assimilation ratcheting* (its energy excluded the one-signed W1/fill
terms, so it could undo an accepted cleanup step and assimilation ratcheted the
regression — the `lg_sweeps+(w_dt|w_fill)` `ValueError` guard in
`runner.py:run_pipeline`). v1 wrongly claimed removing (i) dissolved the guard;
Codex 7 showed (ii) alone reproduces an alternating accept/reject deadlock through
any *state-persistent* carrier. **v2 retains the guard untouched.** Tier D triggers
neither precondition: it edits no state and no rest state.

A third v1-era failure is now first-class: (iii) *same-nullspace objective* — the
retired pass descended the silhouette the global window had already exhausted
(h15: tie at 3.3× wall-clock). Tier D's objective is the Gaussian image loss, whose
unique content is exactly what the silhouette cannot see.

---

## §2 Literature grounding (read/verified 2026-09-02; unchanged from v1)

| # | paper (venue, link) | contribution to THIS design (v2 reading) |
|---|---|---|
| 1 | **Projective Dynamics** — Bouaziz, Martin, Liu, Kavan, Pauly, ACM TOG 33(4), SIGGRAPH 2014. https://users.cs.utah.edu/~ladislav/bouaziz14projective/bouaziz14projective.pdf | The local/global alternation template — and, post-REFUTE, the cautionary half: PD's guarantees exist because both phases descend ONE energy through a coupling variable the global solve cannot simply undo. v1's Tier R violated exactly that (the global solve *could* undo it, exactly and cheaply). |
| 2 | **DiffPD: Differentiable Projective Dynamics** — Du, Wu, Ma, Wah, Spielberg, Rus, Matusik, ACM TOG 41(2), 2021. https://arxiv.org/abs/2101.05917 | Solve-convergence as a gradient-correctness gate. v2 keeps the lesson for Tier D's stop rule and notes Opus M10/M11's sharpening: a convergence *flag* is only meaningful if its reference point and noise class are specified. |
| 3 | **Vertex Block Descent** — A. H. Chen, Liu, Yang, Yuksel, ACM TOG 43(4), SIGGRAPH 2024. https://graphics.cs.utah.edu/research/projects/vbd/ | Block-descent machinery for a complete variational subproblem in its own DOFs. Used by v1's band solve; retained here only as the solver of record if any band solve returns (with the M10 `g0`-at-`u=0` fix). Partial dynamic-state updates stay banned. |
| 4 | **As-Rigid-As-Possible Surface Modeling** — Sorkine, Alexa, SGP 2007. https://dl.acm.org/doi/10.5555/1281991.1282006 | The split's grandfather and its warning (one shared energy). v2 heeds it the hard way: the only surviving local phase (Tier D) shares *no* state with the global solve at all. |
| 5 | **Sobolev Active Contours** — Sundaramoorthi, Yezzi, Mennucci, IJCV 73(3):345–366, 2007. https://link.springer.com/article/10.1007/s11263-006-0635-2 | L2 shape gradients are pathologically high-frequency; smooth metrics give coherent flows. Context for R3 and for Appendix A's compliance shaping. |
| 6 | **Repulsive Curves** — Yu, Schumacher, Crane, ACM TOG 40(2), 2021. https://www.cs.cmu.edu/~kmcrane/Projects/RepulsiveCurves/ | Resolution-independent preconditioned steps for geometric energies; lineage of the repo's grid-GS smoother (method.md §6). |
| 7 | **2D Gaussian Splatting** — Huang, Yu, Chen, Geiger, Gao, SIGGRAPH 2024. https://surfsplatting.github.io/ | Oriented planar disks: the surfel's legitimate freedom is in-plane position/orientation — the basis for Tier D's tangent-only DOFs (and for dropping the scale DOF: 2DGS disks do not shrink to dodge coverage). |
| 8 | **High-quality Surface Reconstruction using Gaussian Surfels** — Dai, Xu, Xie, Liu, Wang, Xu, SIGGRAPH 2024. https://dl.acm.org/doi/10.1145/3641519.3657441 | z-scale→0 improves optimization stability and alignment — evidence that *removing* DOFs (normal, and now scale) helps the fit rather than hurting it. |
| 9 | **SuGaR** — Guédon, Lepetit, CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Guedon_SuGaR_Surface-Aligned_Gaussian_Splatting_for_Efficient_3D_Mesh_Reconstruction_and_CVPR_2024_paper.html | Surface-bound Gaussians cannot float; Tier D's primitives stay bound to material parents with capped offsets — an unsupported dressed splat is impossible by construction. |
| 10 | **Mip-Splatting** — Yu, Chen, Huang, Sattler, Geiger, CVPR 2024. https://niujinshuchong.github.io/mip-splatting/ | Primitive size constrained by sampling frequency. Grounds the Nyquist res floor in `gauss_loss.py` — which stays valid in v2 precisely because the dressed sigma equals the undressed sigma (scale DOF dropped, Opus M15). |
| 11 | **PhysGaussian** — Xie et al., CVPR 2024. https://xpandora.github.io/PhysGaussian/ | F-driven splat kinematics; the "what is seen is what is simulated" contract — which is why dressing must ship per-frame to the viewer (Codex 16) and why its world-space footprint must be capped where `F` is large (Opus M14). |
| 12 | **Stress-dependent finite growth in soft elastic tissues** — Rodriguez, Hoger, McCulloch, J. Biomech. 27(4):455–467, 1994. https://pubmed.ncbi.nlm.nih.gov/8188726/ | Multiplicative rest-state remodeling realized by elasticity. In v2 this grounds only the existing `w_grow` channel and Appendix A's *response-side* sibling; the v1 force-side use is withdrawn with Tier R. |
| 13 | **Variational multirate integrators** — Ober-Blöbaum et al., arXiv:2406.12991. https://arxiv.org/abs/2406.12991 | Two-clock legitimacy plus the resonance caveat; retained for Appendix A's per-commit carrier cadence. |

---

## §3 Design overview (v2)

```
wavelength:   body scale ───────────── dx=0.5 ──────── NN spacing≈0.04 ───────── 0
              │        GLOBAL (unchanged)              │        TIER D (dressing)
carrier:      │        dFc via MPM adjoint             │   render parametrization
signal path:  │        loss → adjoint → control        │   local solve → observation
              │        → rollout                       │   (no physics, no gates)
```

- **Global (unchanged)**: `optimize_window` + `run_pipeline` byte-for-byte, except
  the B1 telemetry split (§6.2), which makes existing gate inputs *cleaner*, not
  different in kind. The surface restriction stays on the terminal observation
  covector.
- **Tier D** owns wavelengths below what grid forces can shape (sub-dx surface
  texture, pinholes, splat-scale arrangement). These are unphysical for this
  discretization by definition; the only honest place for them is the observation
  layer, with capacity band-limited (in world space) so it cannot impersonate
  transport, and with **zero coupling into any accept/freeze decision**.
- **Tier R (withdrawn)** — see §5. The transport-and-curvature gap it aimed at
  (R2, R3's physical side) is explicitly **not addressed by v2's main path**; the
  candidate replacement carrier is Appendix A, deferred.

Symptom ownership after the withdrawal (honest map):

| symptom | v2 owner | status |
|---|---|---|
| floaters (out_nn>2sp, far counts) | existing nn-band W1 + target-free support check; Tier D adds *observability telemetry* (`d_gauss`) and G6 fidelity, claims no count reduction | regression-guarded only (K-D1) |
| dead render channel (work share) | **unowned in v2 main** — Appendix A candidate | deferred, stated |
| late-run oscillation (rev-cos) | dossier closed by h16/h17 at the optimizer level; Tier D carries a residual *empirical bet* on sub-Nyquist wander (§8 P-osc, demoted per M13) | empirical bet with falsifier |

---

## §4 Tier D — surface dressing (v2, repaired)

### 4.1 DOFs

Per child `c` of surface parent `p` (children per `render/children.py`, count 1–4):

- tangent coefficients `(a_pc, b_pc)` in the **frozen** source-material PCA basis
  `(t1_p, t2_p)`; effective material offset `off_pc = baseline_pc + a_pc·t1_p + b_pc·t2_p`;
- rendered center `x_p + F_p·off_pc`; covariance exactly as today
  (`Σ = σ_child² F_p F_pᵀ`).

**Dropped from v1** (Opus M15 / Codex 15, accepted): the per-child log-scale `s_pc`.
It was a partial opacity DOF ("tearing rewarded by a disappearing render loss" via
coverage area), it invalidated the Nyquist floor computed at `__init__`, and it
multiplied the parent `sv(F)` band into an unmonitored total scale. With it gone:
the dressed sigma ≡ the undressed sigma, the existing Nyquist floor and
`primitive_sigma`/`gauss_scale_*` diagnostics stay truthful, and no σ-band constants
are imported (dissolves Opus m23). 2DGS/Surfels ground the choice: a surfel's
freedom is in-plane.

**No opacity DOF** (floaters.md contract, unchanged).

### 4.2 Caps and exact feasibility (Opus M14, Codex 15)

- **World-space cap**: `‖F_p·off_pc‖ ≤ 0.5·h_src` (`h_src` = frozen source-surface
  median NN spacing). v1's material-space cap acted in the wrong space: at the ears
  `s_max(F) > 2` is *required*, so a material cap of `0.5·h_src` rendered to
  `≥ h_src` exactly in the contested region. The cap now bounds what the image
  actually sees, everywhere, by construction.
- **Exact joint feasibility** (cap ∩ zero-centroid): after each optimizer step,
  (1) subtract the per-parent child mean (projection onto the centroid plane), then
  (2) uniformly rescale the parent's child offset set by
  `min(1, 0.5·h_src / max_c‖F_p·off_pc‖)`. Uniform per-parent rescaling **commutes
  with the zero-centroid constraint** (a scaled zero-mean set is zero-mean), so both
  constraints hold *exactly* after step (2) — unlike v1's clamp-then-center, whose
  violation Codex 15 exhibited (`[r,r,−r] → [2r/3,2r/3,−4r/3]`). This is a feasible
  map, not the Euclidean projection; for a box-constrained descent that is
  sufficient, deterministic, and order-independent.

### 4.3 Solve (cadence: strictly post-gate)

Runs once per commit, **after** the outer merit gate accepts (Opus B4 resolved for
Tier D by ordering: dressing feeds no gate input, so nothing forces it before the
gate; a rejected or null commit runs no dressing solve and mutates nothing). On the
promoted terminal `(x, F)`, with everything else frozen: minimize
`L_gauss(x, F; a, b)` (the existing `GaussViews.loss`, all views — no stochastic
subsampling) by line-searched Adam with the feasibility map of §4.2 applied after
each accepted step.

**Stop rule** (Opus M11, accepted): fixed max iterations (from the stage-0
benchmark, §7) with early exit on relative *energy* decrease
`ΔL < ls_noise_rel·max(|L|,1)` on two consecutive iterations — the same noise
tolerance class as the replay check, and no discrete branch on a
rasterizer-noise-bearing gradient norm. Dressing is not permanent physics; a
noise-marginal extra iteration is bounded by the caps.

### 4.4 What Tier D may and may not claim (Codex 11 / Opus M13, accepted)

The v1 "envelope theorem" framing is **removed**. Dressing is optimal only at the
commit-time state where it was fitted; the next window's gradient is evaluated at a
different terminal state (nonzero `v`/`C`, T=20 of evolution), so the frozen
dressing carries a stale-fit bias — plausibly *retarding* (a pull toward the fitted
state) rather than cleanly frequency-separating. The oscillation prediction P-osc is
therefore a **pure empirical bet** with an unchanged falsifier (§8). No repair is
attempted (per-candidate dressing profiling would destroy the fixed-objective window
contract and the cost model).

### 4.5 Per-frame dressing archive (Codex 16, accepted)

`dressing_frames` is archived one-to-one with `frames`: rollout intermediates carry
the window's frozen (pre-window) dressing; the terminal frame carries the newly
accepted dressing; held commits copy; outer-rejection truncation mirrors the
existing `del frames[rollback["frames"]:]`. Dressing is passed through `on_commit`
and the result dict, so the viewer/export renders each frame with the dressing that
was actually in force — no pre-echo, no retroactive re-observation (the PhysGaussian
"what is seen is what is simulated" contract, made temporal).

### 4.6 c2f behavior (Codex 12, accepted; B5 cited)

Dressing is **not** reset at c2f. The c2f rebuild changes `cfg.render_res`
(silhouette targets); the Gaussian objective's resolution (`gauss_res` + Nyquist
floor) does not change, so v1's reset would have manufactured a residual
discontinuity and phantom improvement. Dressing state lives in the runner (not in
`GaussViews`), so it survives the `TargetPack` rebuild; the `configure_source`
re-call after the c2f `build_target` is **already fixed in production 2026-09-02**
(Opus B5) and is simply consumed here.

### 4.7 Invariants

Touches nothing physical: `x/v/C/F/Fp`, particle count, mass, momentum bit-identical;
children massless observation-only. Coefficients live in compact sets (§4.2), the
solve is monotone in `L_gauss` by line search, and — after the §6.2 split — **no
gate, track, latch, or merit component consumes any dressed quantity**. The remaining
sanctioned couplings, stated openly: dressing changes the *window objective's* gauss
component (that is its job — the optimization channel), hence λ_R's balance and the
`g_*`/`render_work*` telemetry; all of these are logged, none of them gate.

---

## §5 Tier R — WITHDRAWN (the reachability verdict)

The v1 mechanism (virtual band displacement → `A = exp(clamp(sym∇u))` →
`Fp ← assimilate(A·F)`) is withdrawn in full. The killing findings, in order of
depth:

1. **In-span cancellation (Codex 8 / Opus B2).** The control is an additive offset
   on the same tensor the demand divides: `Fe = (F+dFc)·Fp⁻¹`
   (`mpm/kernels.py:41-46`). For any demanded `Fp_new = Sa·Fp` the control
   `dFc = F·(Fp⁻¹SaFp − I)` reproduces the pre-demand elastic state bit-for-bit at
   every step — identical stress, identical trajectory. The annulling direction is
   available at iteration 0, smooth (CIC-band-limited), and priced at
   `w_ctrl·‖dFc‖²/(TN) ≈ 6e-7`, while the demand's *realization* is priced by
   `w_kin` (its elastic kick is terminal velocity) and its benefit rides the
   measured ~0.02% render share. The gradient therefore *favors cancellation*;
   expected steady state: zero realized motion, monotone `Fp` mutation.
2. **Rotation leak (Codex 1 / Opus M6).** `F_e,new = F_e S'^{-η}` with `S_e, S'`
   non-commuting for any non-coaxial demand; the non-coaxial component is absorbed
   into a polar rotation that a corotational energy converts to zero stress and
   zero motion. "Rotation untouched / per-particle exact" was false (Codex's
   2.17° counterexample).
3. **Solve/emission mismatch (Codex 10) + gate pathologies (Opus M10, M11).** The
   band solve descends in `(x+u, (I+∇u)F)` but emits only `exp(clamp(sym∇u))`
   — translation demands emit `A=I`, shear is distorted, `ψ_SNH` accepts
   reflections the covariance cannot see; and the convergence gate could not fire
   before sweep 2 (`g0` set after the first sweep), so the common case was full
   cost, zero donation.
4. **Containment unimplementable as written (Codex 5, 6 / Opus B3, B4).** The fused
   single assimilation call has no recoverable `Fp_plain`; pending demand survives
   null/replay-failure/grad-converged/freeze/budget-end paths; "accepted commits
   only" contradicts the pre-gate call site.
5. **Ratchets real, falsifier vacuous (Codex 3, 4 / Opus M7, M8, M9).** Volume
   ratchet at default `assim_iso=False`; areal ratchet under isochoric projection;
   the global-median throttle licenses full demand exactly in the low-strain
   (= uncontrollable) regions; `‖ΔFp‖` during held commits is identically zero by
   control flow, so K-R4 passed vacuously.

**The classification this leaves behind** (the useful residue): `dFc` is a complete
per-particle, per-step stress-argument actuator, so *every force-like demand* —
prestress, rest-shape, growth-shaped pushes — lies in its exact reachable range and
can be annulled at ~zero cost. Carriers **outside** span(dFc) are exactly three:
(i) the *response operator* (per-particle material — changes how ALL forces act;
annulment would require solving `P_new(Fe')=P_old(Fe)` pointwise in time while `Fe'`
simultaneously feeds the kinematic update (9) — overdetermined, no exact null
direction); (ii) the *discretization/topology* (resampling, splitting — touches the
mass distribution, outside this project's state contract); (iii) the *observation*
(Tier D; produces no transport by construction). Hence the only in-contract,
motion-relevant replacement carrier is (i) — sketched in Appendix A, deferred.

---

## §6 Global solver: unchanged, plus the B1 split

### 6.1 Unchanged

`optimize_window` losses, λ_R once-per-window, adjoint, line search, `_state_ok`,
replay validation; `run_pipeline` promotion, guards, outer gate, rollback + cold
restart; the `lg_sweeps+(w_dt|w_fill)` guard **retained untouched** (Codex 7 /
Opus M16 accepted — its precondition (ii) survives any state-persistent local
carrier, and Tier D never trips it because it persists no state).

### 6.2 The d_sil/d_gauss split (Opus B1 / Codex 9 — mandatory before any run)

Today, with `use_gauss_loss` and `gauss_mix>0`, the scalar
`lr = lsil + gauss_mix·gauss_scale·L_gauss` is logged as `d_render`
(`optimizer.py:losses_of`, history at `optimizer.py:574-580`) and consumed by
`components["render"]` (outer merit, `runner.py:391`), `rend_track` (freeze
plateau, `runner.py:409`), `improved`, `best_rend`, `stale`, `anneal`, and the
latch. Opus B1 quantified the laundering: at `gauss_mix=0.25`, a K-D2-sized
dressing improvement is a free ≈2.5%/commit `d_render` drop — `improved=True`
unconditionally, `stale=0` forever, the latch never arms, the merit gate saturates.

Repair (one change, two files):

- `optimizer.py:losses_of` returns `lsil` and `lg_` separately; the window history
  and `stats` log **`d_sil`** (pure silhouette) and **`d_gauss`** (raw gauss L1)
  alongside the objective scalar `d_render_obj`;
- `runner.py`: `components["render"] = rec["d_sil"]`, `rend_track = rec["d_sil"]`,
  `best_rend` tracks `d_sil`; `d_gauss` is telemetry only. The v1 plan to delete the
  old post-pass metric recompute ships only together with this split (Opus m26):
  with no state edits and d_sil-only tracks, the archived state *is* the window
  state and the recompute is dead code by construction.

λ_R balancing and the window objective still consume the hybrid — that is the
sanctioned optimization channel; it is logged (`d_gauss`, `g_*`, `render_work*`)
and gates nothing.

### 6.3 Transaction hygiene (Codex 13)

Tier D's transaction is trivial by cadence: dressing mutates only post-gate on
accepted commits, so outer rejection of commit k+1 finds the post-k dressing
already in force and correct — nothing to restore; `dressing_frames` truncation
follows frame truncation (§4.5); the frozen/c2f/termination paths carry the last
accepted dressing forward. Codex 13's remaining observations — empty/replay-invalid
windows retain mutated `s`/`dfc`/λ; `mom_carry` without `dfc_init` skips the warm
safeguard — are **pre-existing production issues independent of this design**;
accepted and flagged for a standalone fix (they bite any arm, not just this one),
with `lg_balancer` rollback now moot (no λ_loc exists in v2).

---

## §7 Constants and budgets (all rule-bearing; none tuned)

| constant | value | rule |
|---|---|---|
| dressing amplitude cap | `‖F·off‖ ≤ 0.5·h_src` (world space) | Nyquist: sub-sample-spacing in the *image-facing* metric (Opus M14); A/B `{0.5, 1.0}` in the ladder with a pick-by-rule (raw-chamfer tie + max `d_gauss` drop). |
| scale / opacity DOFs | none | dropped (Opus M15; floaters.md contract). |
| dressing iteration budget | largest of `{5, 10, 20}` whose stage-0 benchmark p50 overhead ≤ 50% of the global commit p50 | pre-registered selection, measured not estimated (Codex 14). If even 5 exceeds the bound, the arm is void before any GPU ladder. |
| stop rule | `ΔL < ls_noise_rel·max(|L|,1)` twice consecutively | existing noise-floor constant; energy-based (Opus M11). |
| views | all `render_views × render_elevs` | determinism; no stochastic subsampling. |
| cadence | post-gate, accepted commits only | resolves Opus B4 for Tier D; rejected/null commits mutate nothing. |
| stage matching | **commits matched, `cfg.pace` held fixed**, wall-clock reported | Opus M19: wall-clock matching silently rewrites the glidepath; pace is part of the mechanism under test and must not move. |

---

## §8 Predictions (v2 — narrowed and honest)

Baseline `b0` = the production gauss arm (`render_stable_gauss`:
`use_gauss_loss`, `gauss_mix=0.25`, `gauss_children=4`, `outer_merit=True`) **with
the §6.2 split applied to both arms** (the split changes gate inputs, so the
baseline must carry it too — otherwise the A/B confounds the split with the
dressing). Matched seed, commits, pace, discretisation
(`dx=0.5, dt=1/240, 64³, smoothing 0.955, loss_res 64, T=20`).

**P-dress (explanatory power).** Dressing drops the *undressed-state* gauss
residual it is allowed to explain: `d_gauss` (post-dressing, logged per commit)
≥10% below the matched baseline's from mid-run on, at a raw-state chamfer tie
(±2%). **Kill K-D2**: <10% at tie — no explanatory power, drop the arm.

**P-osc (pure empirical bet — M13 accepted).** With sub-Nyquist residual explained
at fit states, the render channel's flat-valley gradient churn shrinks. Predict:
over the last 40 pre-freeze commits, min `reversal_cos` ≥ the matched baseline's,
and tail mean `move` ≤ baseline's; freeze commit index ≤ baseline's +10%. **Kill**:
any of the three inverted at stage-1 scale. (No mechanism claim survives M13; this
is a bet, and it dies quietly if wrong.)

**P-QA (G6).** Dressed frames (rendered with the per-frame archive, §4.5) show
strictly fewer pinholes/blebs than undressed frames of the same run at 512px QA,
with zero temporal pops across commit boundaries (the archive contract makes this
testable frame-by-frame). **Kill**: any retroactive-observation artifact (pre-echo,
pop) — archive bug or design flaw, either way the arm blocks.

**Guard predictions (regressions).** `out_nn_frac`, `out_nn_far_frac`, hole_frac,
G2 counters: no regression beyond noise vs baseline. **Kill K-D1**: raw chamfer
regression >2% or any guard ≠ 0.

**Not predicted by v2 main** (explicitly): render work share (R2) and floater-count
reductions. R2's carrier was withdrawn; see Appendix A for the deferred candidate
and its own falsifiers.

---

## §9 Integration plan (v2)

| where | change |
|---|---|
| `physmorph/pipeline/optimizer.py:losses_of` + history/stats | return/log `d_sil`, `d_gauss`, `d_render_obj` separately (§6.2). |
| `physmorph/pipeline/runner.py:run_pipeline` | gate/track/merit inputs → `d_sil`; delete the dead post-pass recompute (paired, m26); `dressing_frames` archive + truncation + `on_commit`/result plumbing; dressing state owned here, post-gate update only. |
| `physmorph/render/children.py` | `DressingState` (per-child `(a,b)` in the frozen basis; feasibility map of §4.2; numpy twin for export). |
| `physmorph/pipeline/gauss_loss.py:GaussViews._render/loss` | accept a live offsets override (positions only — no scale path). |
| new `physmorph/pipeline/dressing.py` | the post-gate solve (§4.3): line-searched Adam, energy stop rule, feasibility projection, telemetry (`dress_iters_used`, `dress_dL`, `d_gauss_post`). |
| `physmorph/pipeline/config.py` | `local_dress_iters` (0=off; benchmark-selected), `dress_cap_frac=0.5`. No other knobs. |
| `scripts/pipeline_run.py` | arm `d1`; benchmark subcommand for stage 0. |
| `physmorph/viewer/server.py`, `scripts/render_sequence.py` | consume `dressing_frames` per frame. |
| `tests/` | stage-0 suite (§10). |

Cost (Codex 14's arithmetic, now owned): worst case at 20 iters with 1–10
backtracks ≈ 720–3,960 view-forwards + 360 backwards per commit — the stage-0
benchmark measures the real number at production res/children/N and *selects the
iteration budget* (§7) rather than trusting any estimate. The v1 "3.3×" citation is
struck: it measured a CIC-silhouette inner loop, not all-view 3DGS.

---

## §10 Staged ladder (v2)

| stage | arm | budget | metrics | kill criterion |
|---|---|---|---|---|
| 0a | CPU tests | pytest | feasibility map exactness (centroid ∩ world-cap after every step, incl. Codex 15's `[r,r,−r]` case); zero-DOF ⇒ bit-identical render; archive alignment incl. rollback truncation; determinism of the solve given fixed state | any failure blocks |
| 0b | GPU microbenchmark | one commit-shaped run, production res/children/N | dressing-solve p50/p95 (accepted and rejected commit shapes) vs global commit p50 | overhead >50% at `iters=5` ⇒ **arm void** (Codex 14 gate) |
| 1 | `d1` vs `b0` (both with §6.2 split) | N=20k, 120 commits, pace fixed | §8 P-dress, P-osc, P-QA, guards | K-D1 (chamfer >2% or guard ≠ 0), K-D2 (<10% `d_gauss` drop at tie), P-QA pop |
| 1b | cap A/B `{0.5, 1.0}·h_src` | N=20k, 120c | raw-chamfer tie + `d_gauss` | pick-by-rule; runs only if stage 1 passes |
| 2 | `d1_hero` | N=40k, 4 pairs, **commits matched, pace fixed**, wall-clock reported | full G2–G6 battery + §8 | G5 rule per pair; adoption only with all gates and no regression |

Verdict discipline unchanged: pre-registered thresholds, no post-hoc metric swaps,
every number with its discretisation, REFUTE round on the diff before hero runs.

---

## §11 Adversarial self-review (v2 residuals)

**11.1 "The split changes the baseline too."** Yes — §6.2 alters gate inputs for
every gauss arm, so it is applied to both A/B arms and, if adopted, to production
regardless of Tier D's fate. It is a harness-integrity fix that B1 exposed, not a
Tier D feature; its own effect gets measured by re-running `b0` pre/post split once.

**11.2 "Dressing still shapes the window objective."** True and sanctioned: the
hybrid gauss term is the optimization channel. The exposure is bounded: λ_R is
capped, `d_gauss` is logged, and no gate consumes it. Residual risk — λ_R drift
caused by dressing-induced `‖∇L_gauss‖` changes — is visible in the per-commit
`lambda` telemetry and pre-registered as a stage-1 watch item, not a gate.

**11.3 "Sub-Nyquist capacity, image-space." ** The world-space cap closes M14's ear
hole (image amplitude ≥ h_src exactly where `sv(F)>2`). What remains unrepresentable
by dressing — silhouette-scale error — stays visible to the global channel via
`d_sil`, which is now also the only gate currency.

**11.4 "Determinism."** The solve is deterministic given the state up to rasterizer
atomic noise; the stop rule is energy-based with the replay-check tolerance class
(M11); post-gate cadence means rejected lineages never see a dressing mutation.

**11.5 "What does v2 actually buy?"** Narrow, honest scope: (a) the B1 harness fix;
(b) a capacity-bounded observation layer that can only be adopted if it explains
≥10% of the gauss residual at a raw-fidelity tie; (c) the P-osc bet; (d) the
per-frame observation contract for deliverables. The R2 transport gap is *not*
bought — it is deferred to Appendix A with the reachability classification as the
design constraint any future carrier must satisfy.

---

## §12 Non-goals (v2)

- No rest-state, growth-shaped, or prestress demand from any render-local channel
  (withdrawn, §5); the `w_grow` channel remains the only commit-time rest-volume
  actor, unchanged.
- No implicit-MPM integrator replacement; no control-space H1/kNN preconditioner.
- No opacity/scale/deletion DOFs in any differentiable objective.
- No change to mass, momentum, P2G/G2P, the adjoint, `_state_ok`, guard semantics,
  the metric battery, or the `lg_sweeps+(w_dt|w_fill)` guard.
- No constant without a §7 rule.

---

## §13 REFUTE dispositions — every finding, both reports

Legend: **A** = accept (with repair or withdrawal), **A/S** = accept, settled by a
production fix (cited per coordinator; not redesigned here), **A/moot** = accepted
and mooted by the Tier R withdrawal (recorded as a constraint on any future carrier).

| finding | disposition | repair / evidence |
|---|---|---|
| Codex 1 / Opus M6 (rotation leak, non-commuting polar) | **A/moot** | Withdrawal evidence (§5.2). Appendix A's carrier has no polar algebra to leak through. |
| Codex 2 / Opus m20 (SV-band violated by alternating projection; growth governor undoes clamp) | **A/S** | Fixed in production 2026-09-02: exact KKT projection onto log-box ∩ det constraint. v2 additionally adopts the reviewer's residual ask: an `Fp_bad` guard counter (finite/det/SV audit) at the assimilation call site, G2 class. |
| Codex 3 / Opus M7+M8 (volume ratchet at default `assim_iso`; areal ratchet; nonlocal τ; no self-termination) | **A/moot** | Withdrawal evidence (§5.5). Appendix A uses per-particle hysteresis increments with a cumulative band; no strain-derived throttle exists. |
| Codex 4 / Opus M9 (K-R4 vacuous — held commits never run the pass) | **A/moot** | K-R4 deleted with Tier R. The replacement falsifier class — descent-phase p99 monotonicity + demand-sign autocorrelation — is adopted verbatim in Appendix A §A.5. |
| Codex 5 / Opus B3 (fused assimilation ⇒ no recoverable `Fp_plain`; restoring pre-demand Fp creates an ungated third state) | **A/moot** | Withdrawal evidence. Hard requirement **A-T1** on any future carrier: two-variant transaction `{carrier_plain, carrier_demand, origin_commit}`, finalized only after the next window passes the outer gate. |
| Codex 6 (pending demand survives null / replay-fail / grad-converged / freeze / budget-end) | **A/moot** | Same as above; A-T1 enumerates the restore paths verbatim (empty, invalid, converged, frozen, rejected, exception, c2f, budget-termination). |
| Codex 7 / Opus M16 (retiring the W1/fill guard recreates the accept/reject deadlock; half the precondition survives) | **A** | Guard **retained untouched**; §1 corrected: precondition = objective exclusion + assimilation ratcheting, not merely state overwrite. Tier D trips neither. |
| Codex 8 / Opus B2 (dFc exactly cancels the prestress; cancellation is the gradient-favored direction) | **A** | The central kill; generalized into the span(dFc) classification (§5, end). The four-way telemetry split (passive / cancellation / net realization via counterfactual rollout) is requirement **A-T2** for any motion-claiming carrier. |
| Codex 9 / Opus B1 (Tier D launders merit/freeze through `d_render`) | **A** | Repaired (§6.2): `d_sil`/`d_gauss` split; all gate, track, latch, and merit inputs read `d_sil` only; `d_gauss` telemetry-only. Applied to baseline arms too (§11.1). |
| Codex 10 (band-solve convergence says nothing about the emitted demand; reflections invisible to Σ) | **A/moot** | Withdrawal evidence (§5.3). If any band solve returns, it must optimize in the emitted variables directly — recorded as a constraint. |
| Codex 11 / Opus M13 (envelope theorem at the wrong state / only at the first iterate) | **A** | Claim removed (§4.4); P-osc demoted to a pure empirical bet with an unchanged falsifier. No profiling repair attempted — cost model preserved. |
| Codex 12 / Opus B5 (c2f: missing `configure_source` ⇒ RuntimeError; dressing reset manufactures discontinuity; cleared `outer_prev` ungates a pre-c2f demand) | **A/S + A** | `configure_source` re-call: fixed in production 2026-09-02 (cited). Dressing reset: withdrawn — dressing persists across c2f (§4.6; its objective's resolution never changed). Pre-c2f pending-demand hole: moot (no demand exists in v2). |
| Codex 13 (nontransactional rollback/warm start; lg_balancer not rolled back; empty windows retain s/dfc/λ; mom_carry loads stale moments without safeguard) | **A** | Tier D's transaction is trivial by post-gate cadence (§6.3); `lg_balancer` moot (no λ_loc in v2). The pre-existing production residues (empty-window s/dfc/λ retention; mom_carry-without-dfc_init safeguard bypass) are accepted as real, independent bugs and flagged for a standalone fix — v2 does not build on them. |
| Codex 14 (cost undercounted ~10×; 3.3× proxy invalid; τ-shrink does not cause early exit; wall-clock math) | **A** | 3.3× citation struck; Tier-D worst-case counts owned (§9); **stage-0 GPU microbenchmark is a hard gate** with a pre-registered iteration-budget selection rule (§7, §10 0b); τ claim gone with Tier R. |
| Codex 15 / Opus M14+M15 (caps not jointly enforced; material-space cap acts in image space; log-scale is a partial opacity DOF breaking the Nyquist floor) | **A** | `s_pc` **dropped**; world-space cap `‖F·off‖ ≤ 0.5·h_src`; exact joint feasibility by centroid-projection-then-uniform-per-parent-rescale, which commutes with the centroid constraint (§4.2); Nyquist floor and `primitive_sigma` diagnostics stay truthful with dressed σ ≡ undressed σ. |
| Codex 16 (no per-frame dressing ⇒ QA contract violated) | **A** | `dressing_frames` 1:1 archive with holds/truncation semantics; `on_commit` + result plumbing; viewer/export consume it (§4.5). P-QA gates on zero temporal pops. |
| Opus B1 | **A** | = Codex 9 row. |
| Opus B2 | **A** | = Codex 8 row. |
| Opus B3 | **A/moot** | = Codex 5 row (A-T1). |
| Opus B4 ("accepted commits only" vs fused pre-gate call — ordering unnamed) | **A** | For Tier D: resolved by explicit post-gate cadence (§4.3, §7) — dressing feeds no gate input, so it legitimately runs only on accepted commits with no second assimilation call. For Tier R: withdrawal evidence. |
| Opus B5 | **A/S** | = Codex 12 row, production fix cited. |
| Opus M6 | **A/moot** | = Codex 1 row. |
| Opus M7 (volume demand possible at default; growth branch skips det-1) | **A/moot** | Withdrawal evidence; the `assimilate_growth` branch-precedence observation is recorded for the growth channel's own docs (behavioral by design for commanded growth, but worth the docstring). |
| Opus M8 (areal ratchet; global-median τ licenses low-strain regions) | **A/moot** | = Codex 3 row. |
| Opus M9 (K-R4 vacuous) | **A/moot** | = Codex 4 row; falsifier class inherited in §A.5. |
| Opus M10 (convergence gate cannot fire before sweep 2; `g0` after first sweep) | **A/moot** | Withdrawal evidence; recorded repair (`g0` at `u=0`, already computed for λ_loc) binds any revived band solve. Tier D's stop rule is energy-based and unaffected. |
| Opus M11 (binary gate on rasterizer-noise gradient ≠ replay-check class) | **A** | Tier D stop rule switched to relative energy decrease with the `ls_noise_rel` tolerance (§4.3); dressing is non-permanent, transactional. |
| Opus M12 (λ_loc recipe pinned at cap ⇒ raw L2 shape gradient made permanent) | **A/moot** | No λ_loc exists in v2 (the dressing solve is unweighted `L_gauss` + box constraints). Recorded as a mandatory saturation-kill (`lam at cap >20% of commits ⇒ arm void`) for any future balancer-bearing carrier. |
| Opus M13 | **A** | = Codex 11 row. |
| Opus M14 (cap acts in image space; ears need `sv(F)>2`) | **A** | = Codex 15 row (world-space cap). |
| Opus M15 (log-scale = partial opacity DOF; Nyquist floor invalidated; 384 cap silent) | **A** | = Codex 15 row (`s_pc` dropped — the stricter of the two offered repairs). |
| Opus M16 | **A** | = Codex 7 row (guard retained). |
| Opus M17 (`lg2_realized_cos` measures transport alignment, not realization) | **A/moot** | Withdrawn with Tier R; the counterfactual-rollout realization metric (same accepted `dFc`, carrier_plain vs carrier_demand, one extra forward rollout) is adopted as **A-T2** in Appendix A. |
| Opus M18 (headless runs record `render_work=None`) | **A/S** | Fixed in production 2026-09-02 (work telemetry hoisted out of the `on_iter` guard); the ladder consumes the fixed fields. |
| Opus M19 (wall-clock matching rewrites the pace glidepath) | **A** | Stages match on **commits with `cfg.pace` held fixed**, wall-clock reported (§7, §10). |
| Opus m20 | **A/S** | = Codex 2 row. |
| Opus m21 (τ measured pre-assimilation ⇒ 2× budget) | **A/moot** | τ withdrawn; Appendix A budgets are defined post-assimilation by construction (increment-based, not strain-derived). |
| Opus m22 (`ok = det>1e-6` on `A·F_e` silently drops demand AND global assimilation) | **A** | Moot for demand; the reviewer's `n_assim_skipped` counter is adopted for the production assimilation site regardless (cheap telemetry; a skipped row is invisible today). |
| Opus m23 (reusing `cov_smin/cov_smax` imports unvalidated constants) | **A** | Dissolved by dropping `s_pc`; no σ band is imported anywhere in v2. |
| Opus m24 (K-R2's "window-start kinetic kick" has no recorded field) | **A/moot** | K-R2 withdrawn; Appendix A pre-registers `rec["kin_start"]` (mean&#124;v&#124;² after the window's first step) as a first-class field before any carrier revival. |
| Opus m25 (§11.7 inferred amplitude from `move`, a step size) | **A** | §1 rewritten to cite the dossier's direct forensic (oscillators at 1.2× spacing) and the `move`-based inference is dropped. |
| Opus m26 (recompute deletion must pair with B1 repair) | **A** | Paired explicitly (§6.2): the deletion ships only with the d_sil split, in the same change. |

---

## Appendix A — deferred carrier sketch: per-particle material remodeling ("Tier M")

**Status: DEFERRED.** Not part of the v2 ladder; runs only after Tier D's stage-2
verdict AND its own REFUTE round. Recorded here because §5's classification reduces
the search space to exactly one in-contract, motion-relevant carrier, and the
coordinator's settled ground rules ask for the strongest candidate.

### A.1 Why material, and not the other two out-of-span carriers

- *Target-side warping* is an observation change on the target: it can de-noise the
  objective (largely redundant with Tier D) but produces no transport — it cannot
  touch R2.
- *Rest-topology/sampling* (splitting/merging particles) changes the mass
  distribution — outside this project's state contract (mass/momentum untouched).
- *Material* (per-particle Lamé) changes the **response operator**: it is already a
  physical field with an existing optimized twin (`s`, `Lamé = base·e^s`,
  `mat_clamp`, G1b-gated gradients), and no cheap exact annulling control exists —
  to reproduce a trajectory under modified stiffness, `dFc` would have to solve
  `P_new(Fe') = P_old(Fe)` pointwise in time while `Fe'` simultaneously drives the
  kinematic update (9); overdetermined, no null direction (contrast §5.1).

### A.2 Mechanism

A commit-time channel writes the per-particle **base** log-multipliers
`b = (b_λ, b_μ)` (the optimizer's leaf `s` continues to fine-tune around the base;
effective Lamé `= base·e^{b+s}`):

- **stiffen** (`b_μ += δ`) where the per-particle gauss residual has sat below its
  noise floor for K consecutive commits — raising the curvature of the energy in
  the converged surface modes: the flat valley (R3) gets walls from physics, not
  from state damping;
- **soften** (`b_μ, b_λ −= δ`) where the residual has sat in its top decile for K
  consecutive commits (ears) — compliance amplification: the same attenuated render
  gradient produces more motion where motion is demanded (R2 addressed as *plant
  conditioning*, the Sobolev idea executed through a physical, guarded field);
- hysteresis `K=3` (the Schmitt pattern already queued in config notes); increment
  `δ = mat_clamp / animations` (glidepath: the band edge is reachable only by
  unanimous demand across the whole run — the `pace_budget` derivation pattern, no
  tuned scalar); cumulative `|b| ≤ mat_clamp` (the band the `s` channel already
  validates for exactly this quantity).

### A.3 Hard requirements inherited from the REFUTE rounds

- **A-T1 (Codex 5, 6 / Opus B3)**: two-variant transaction
  `{b_plain, b_demand, origin_commit}`; `b_demand` finalized only after the next
  window passes the outer gate; `b_plain` restored on every empty, invalid,
  converged, frozen, rejected, exception, c2f, and budget-termination path.
- **A-T2 (Codex 8 / Opus M17)**: realization is measured counterfactually — re-roll
  the accepted `dFc` once under `b_plain`; `realized = Δx_T(b_demand) − Δx_T(b_plain)`,
  reported as passive/cancellation/net fractions. One extra forward rollout per
  commit, priced into the stage budget.
- **Falsifiers (Codex 4 / Opus M9 pattern)**: descent-phase — per-particle p99
  `|b|` must not grow monotonically over any 20-commit window; demand-sign
  autocorrelation must not stay positive >10 consecutive commits. Plus
  `rec["kin_start"]` (Opus m24) recorded before any run.
- **Saturation kill (Opus M12 pattern)**: any balancer or normalizer introduced
  here that pins at a cap >20% of commits voids the arm.

### A.4 What could kill it at the whiteboard (pre-registered doubts)

- The optimizer leaf `s` can partially counteract `b` within its own clamp
  (`|s| ≤ 1`): the net band is `e^{b+s}`, so the channel's authority is only the
  part of `b` the window objective does not want undone — which is the point
  (arbitration by physics), but also a cheap half-cancellation the counterfactual
  telemetry must separate.
- Stiffening converged regions raises the global energy scale and could slow
  legitimate late corrections (a lock-in ratchet in disguise); the hysteresis
  release path (residual returns ⇒ soften) must be shown live, not assumed.
- Softened ears deform more under *all* forces, including the W1/nn cleanup pulls —
  interaction pre-registered as a stage A/B, not assumed benign (the M16 lesson).

### A.5 Mini-falsification plan

| stage | arm | budget | metrics | kill |
|---|---|---|---|---|
| M-0 | CPU tests + bench | pytest + stage-0b pattern | A-T1 restore paths enumerated in a test; determinism; counterfactual-rollout cost | any failure blocks |
| M-1 | `m1` (Tier M only) vs `b0` | N=20k, 120c, commits matched, pace fixed | A-T2 net realization; `kin_start`; p99 &#124;b&#124; trace; rev-cos tail; ear cov<0.3; chamfer/silIoU; guards | net realization median <10% of counterfactual delta by c40; chamfer >2%; any falsifier of §A.3; `kin_start` >2× baseline median |
| M-2 | `m1+d1` | N=20k, 120c | interaction with dressing + W1/nn (the §A.4 doubts) | pre-registered per-doubt thresholds set at M-1 verdict time |

---

*(end of v2)*

## §14 Stage-1 result (2026-09-02) — Tier D fails K-D2

Arm `render_flag_dress` (flagship window byte-identical; gauss built for dressing +
telemetry only; surface parents, 4 children, res 384 Nyquist floor, cap 0.5·h_src,
20 iters selected by stage 0b). t2_b0 vs t2_d1, N=20k, 120 paced commits:

- explanatory power: dressed `d_gauss_post` second-half median 0.04273 vs undressed
  0.04432 (−3.6%); last commit 0.02889 vs 0.03045 (−5.1%). **K-D2 (≥10%) fails.**
  Every solve hit the 20-iteration ceiling (never energy-converged).
- raw state: d1 chamfer 0.1307 vs b0 0.1548, out_nn 36.1% vs 48.9% — but the window
  is byte-identical, so this spread is CUDA-nondeterminism path divergence (null
  commits 8 vs 4), a warning about single-seed 120c A/Bs, not a dressing effect.
- observability probe (docs/probes/observability.md): the surface-parent gauss loss
  is blind to the interior half of the viewer-visible floaters by construction, so
  dressing could never touch them.

Disposition: Tier D is dropped as a floater/oscillation mechanism (pre-registered
rule). The dressing code stays as an optional QA/export dressing (no gate reads it).
