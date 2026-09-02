# REFUTE round (Opus) — local-global design, 2026-09-02

# BLOCKERS

## B1 — Tier D feeds the outer merit gate and the freeze tracks. §4.3 and §11.3(b) are false for the production arm.

**Anchor.** `physmorph/pipeline/optimizer.py:218-223` — under `gauss_mix>0` the render scalar is `lr = lsil + cfg.gauss_mix * tgt.gauss_scale * lg_`. `optimizer.py:575-580` — the history field is `"d_render": float(lr_n)` (w_pbr=0 in every gauss arm, so `lpbr_n is None`), i.e. **d_render includes L_gauss**. `runner.py:391` `components["render"] = rec["d_render"]` → outer merit. `runner.py:409` `rend_track = rec["d_render"]` → freeze plateau track. `scripts/pipeline_run.py:275-314` — the production arm `render_stable_gauss` sets `gauss_mix=0.25`, `outer_merit=True`, `use_gauss_loss=True`, `gauss_children=4`.

The design's armor (§11.3b) cites `metrics.py`, which is indeed renderer-free — but `metrics.py` is the *endpoint battery*, not the gate that decides commits. The commit gate and the freeze both read the optimizer's render scalar.

**Failure sequence (quantified).** `gauss_scale` is calibrated once so `scale·L_gauss ≈ lsil`; with `gauss_mix=0.25`, L_gauss carries ≈20% of `d_render`. K-D2 pre-registers a ≥10% L_gauss drop from dressing → a ≈2.5% free drop in `d_render` at every commit, with no state change.
- Freeze track: `runner.py:411-412` requires `rend_track < best_rend - 0.003·|best_rend|`. 2.5% ≫ 0.3% ⇒ `improved=True` unconditionally ⇒ `stale=0` forever ⇒ `frozen` never set ⇒ `anneal` never shrinks.
- Latch: `near_stationary = (not improved) and stale >= 2` can never become true ⇒ `outer_gate_latched` stays 0 ⇒ the reversal gate and the low-gain reject **never arm**. Tier D silently disables oscillation-dossier fix #4 — the very gate §11.1 nominates as Tier R's arbiter.
- Merit: one component of ~4 normalized components drops 2.5%/commit ⇒ `outer_gain ≈ +6e-3`, versus `outer_merit_tol=1e-4`. The gate is not biased, it is saturated: every commit passes on dressing alone.

`rec[]` fields dressing can touch: `loss`, `d_render`, `lambda` (λ_R is balanced off `_norm(gr)` which includes the gauss gradient), `g_cos`/`g_raw_cos`/`g_share`/`g_rend_norm`, `render_work*`, `grad_norm`/`step_norm`/`predicted_decrease`/`accepted`/`rejected`/`iters`, `outer_merit`/`outer_gain`/`outer_accepted`/`outer_gate_latched`, and — through the changed control — `move`, `kin`, `d_vol`, `Jmin`, `reversal_cos`.

**Minimal repair.** Split the render telemetry: log `d_sil` (pure silhouette) and `d_gauss` separately, and make `components["render"]` and `rend_track` consume **`d_sil` only**.

## B2 — The Fp demand lies exactly in the reachable range of `dFc`. §5.4's "one crossing of the low-pass, not two" is false.

**Anchor.** `physmorph/mpm/kernels.py:41-46` — `Fe = (F[p] + dFc[p]) @ inverse(Fp[p])`. The control is an **additive offset on the same tensor the demand divides**.

**Failure sequence.** Let the demand set `Fp_new = Sa·Fp`. The control `dFc[t] = F[t]·(Fp⁻¹ Sa Fp − I)` reproduces `Fe` bit-for-bit at every step, hence identical stress, hence identical `x, v, F` for the whole window. This direction is available at iteration 0, costs `w_ctrl·‖dFc‖²/(TN) ≈ 6e-7`, and is spatially smooth (the demand is CIC-band-limited) so `w_creg` barely charges for it. Meanwhile the demand's *benefit* to the window objective is the gauss share of `lr` scaled by λ (render work share ~0.02%), and its *cost* is the terminal-kinetic term at `w_kin = max(args.w_kin, 20.0)` — the demand's elastic kick is a velocity the window is heavily paid to null. The gradient favours cancellation, and the null direction is exact and cheap. Expected steady state: **zero realized motion, monotone Fp mutation**.

**Minimal repair (diagnostic).** Log the *realized* elastic-state change `‖(F_T+dFc_T)Fp_new⁻¹ − F_T Fp_old⁻¹‖` per commit. No cheap repair for the mechanism itself; a demand the control can exactly annul needs a control-space constraint or a different carrier.

## B3 — `Fp_predemand` is not recoverable, because §5.2 fuses global assimilation and demand into one call.

`rollback["Fp"]` is Fp **already containing commit k−1's demand**. Restoring it yields `(x_k, F_k, Fp_{k−1})` — elastic strain never relaxed: the commit-boundary stress spike the module docstring records as fatal. The containment claim "a bad demand survives at most one window" is false along three live paths: null commit (`continue`s before the gate), `grad_converged` (freezes immediately), `frozen`/held.

**Minimal repair.** Compute both variants: `Fp_nodemand = assimilate_elastic(Fc, Fp, …)`, `Fp = assimilate_elastic(A@Fc, Fp, …)`; carry `Fp_nodemand` one commit deep and restore *it* on an outer rejection; on any null/grad_converged/frozen exit, roll `Fp` back to `Fp_nodemand`.

## B4 — §7 "accepted commits only" cannot coexist with §5.2's single fused assimilation call.

The outer merit gate runs **after** assimilation. Before-the-gate placement runs the pass on candidates subsequently rejected (violating §7, multiplying cost by 1+reject rate); after-the-gate placement needs a second assimilation call (killing §5.2). The design specifies both properties simultaneously and does not name the ordering.

**Minimal repair.** Pick "before the gate", delete the §7 claim, re-price §9.3 with the observed reject rate; B3's repair supplies containment.

## B5 — c2f rebuild + `gauss_children>1` raises today.

`runner.py:171` is the **only** `configure_source` call. The c2f rebuild constructs a fresh `GaussViews` whose `source_offsets = None`; `gauss_loss.py:216-217` raises `RuntimeError`. Any `render_stable_gauss` run reaches `int(0.5·animations)` and dies.

**Minimal repair.** Re-call `tgt.gauss.configure_source(src, surface_w > 0.5)` after the c2f `build_target`; factor the pair into one helper.

# MAJOR

## M6 — The A-premultiply leaks into rotation; §5.2's "rotation demand discarded" and §5.3's exactness row are wrong.

With `F → A·F` the SVD is of `A F_e`, but the division `F_e,new = F Fp_new⁻¹ = F_e Sa⁻¹` divides the *original* `F_e`. So `F_e,new = R_e S_e S'^{-η}` with `S_e`, `S'` both SPD but **non-commuting** for any demand not co-axial with the current elastic stretch. The realized stretch is `sqrt(S'^{-η} S_e² S'^{-η})`; the non-coaxial component of the demand is absorbed into a rotation `Q` and **produces no stress and therefore no motion** (corotational energy). A first-class predictor of low `lg2_realized_cos`, independent of B2.

**Minimal repair.** Drop "rotation untouched" from §5.3; add per-commit telemetry `max angle(polar_R(S_e S'^{-η}))`; stage-0 test that the angle is bounded.

## M7 — Volume demand is *not* structurally impossible; the precondition is `assim_iso=True`, which is not the default.

With `assim_iso=False`, `det A ∈ [0.56, 1.78]` per commit ⇒ `det Fp` gains up to 1.33/commit, compounding — verbatim the falsified ratchet. The growth branch **skips** the det-1 projection entirely (`if grow is not None` takes the governor branch; `elif isochoric` unreachable), and §9.1 routes `F_virt` into `assimilate_growth` too.

**Minimal repair.** Hard-code `isochoric=True` for any demand-carrying call (or raise); stage-0 test asserting `det Fp` unchanged under random capped `A`, including the growth branch.

## M8 — The isochoric projection bounds volume, not **area**; τ is a band median, licensing demand exactly where strain is lowest.

`det Fp = 1` permits `(5, 1, 0.2)` — an areal ratchet of 5:1 with zero volume change. Nothing bounds `Fp` anisotropy; `w_cov` bounds `F`, not `Fp`, and is retired. Rate: at `η=0.5`, `τ≈0.05` → `log 5` reached in ~64 commits. The throttle is a *global* median used as a *per-particle* cap: the ears are the low-strain region *because they are uncontrollable*, so the body's median licenses full-budget demand there every commit. Self-termination fails: at steady descent `‖log S_e‖ ≈ Δ/η`, not 0.

**Minimal repair.** Per-particle τ (`r_p = min(1/3, ‖log S_e,p‖)`) + cumulative anisotropy governor on `Fp` (`max_sv/min_sv ≤ band`), mirroring the growth governor.

## M9 — K-R4, the pre-registered ratchet falsifier, is vacuous.

Held commits reach neither the local pass nor assimilation; `‖ΔFp‖` per held commit is identically 0 by control flow.

**Minimal repair.** Descent-phase falsifier: `‖log Fp‖` per-particle p99 must not grow monotonically over any 20-commit window; demand-direction autocorrelation must not stay positive >10 consecutive commits.

## M10 — The band solver's convergence gate can never fire before sweep 2, and requires a 20× drop from an already-descended point.

`g0` is set **after the first full sweep** from that same `gn`; at `_s=0` the test is `gn ≤ 0.05·gn` — always false. Combined with §7's correctness gate (`lg_converged=False` ⇒ no demand), the common case is **pay the full 10-sweep cost, donate nothing**, deterministically.

**Minimal repair.** Set `g0` from the gradient at `u=0` before the sweep loop (already computed for the λ_loc estimate).

## M11 — A binary gate on a rasterizer-noise-dependent gradient norm is not "the same equivalence class as the replay check".

`converged` is a discrete branch on `gn` (containing gauss rasterizer atomic-order noise) deciding a permanent `Fp` mutation. A coin-flip near the threshold makes the demand non-reproducible in kind.

**Minimal repair.** Hysteretic two-of-three-sweep criterion, or gate on relative *energy* decrease.

## M12 — §7 reuses a λ_loc recipe the design's own cited evidence records as pinned at the cap.

At the cap, `E(u)` is a pure render fit with negligible elastic regulariser — deleting the Sobolev/coherence grounding and making `u` the raw pathologically-high-frequency L2 shape gradient, then made permanent plastic state.

**Minimal repair.** Pre-register a λ_loc saturation check as a stage-2 kill: `lg_lam` at cap >20% of commits ⇒ arm void.

## M13 — §4.2's envelope argument holds only at the window's first iterate.

The gradient the window uses is `∂L/∂x(x, d*(x_k))`, equal to the reduced gradient only at `x = x_k`; elsewhere it carries the stale dressing's mis-model — a *retarding* bias toward the fitted state. P-osc does not follow.

**Minimal repair.** Re-solve dressing each iteration (breaks fixed objective) or state P-osc as a pure empirical bet.

## M14 — Tier D's capacity cap is stated in material space but acts in image space.

`center = x + F·off`: the rendered displacement is `‖F_p·off‖ ≤ s_max(F_p)·0.5·h_src`, and `sval > 2` is *required* at ears — so image-space dressing amplitude ≥ h_src exactly in the contested region: enough to close a genuine geometric hole visually.

**Minimal repair.** Cap in world space: clamp `‖F_p·off‖ ≤ 0.5·h_src`, re-projecting after each dressing step.

## M15 — The per-child log-scale **is** a partial opacity DOF, and it invalidates the Nyquist res floor.

`need` is computed at `__init__` from the undressed smallest splat; `s = log 0.5` projects to 0.75 px — below the 1.5 px floor. Shrinking a lone bad splat is a cheap way to lower the loss: "tearing rewarded by a disappearing render loss" via coverage area instead of α. `res` capped at 384, so doubling `need` may silently fail. Also `primitive_sigma` diagnostics report the undressed value.

**Minimal repair.** Drop `s_pc` (tangent (a,b) only — what 2DGS/Surfels actually support), or bound `s_pc ≥ 0` and recompute `need` from the minimum dressed sigma, erroring past the 384 cap.

## M16 — Retiring the `lg_sweeps` ValueError misreads the guard: half its precondition survives.

The guard's stated precondition is *objective exclusion plus assimilation ratcheting*, not state overwrite. lg2 removes the overwrite and **keeps the ratchet**. Concrete path: nn pulls a fringe particle inward; the gauss band objective reads the sparsity as a hole and demands outward stretch; baked into Fp before the gate; window k+1's nn term fights a permanent rest state (and per B3 the demand may never be gated).

**Minimal repair.** Keep a retargeted guard: refuse `local_demand=True` unless the band energy includes the same one-signed terms, or pre-register the `w_dt`/`w_nn` interaction as a stage-2 A/B.

## M17 — `lg2_realized_cos` cannot measure what P-render needs.

Band Δx over a window is dominated by paced bulk transport; high cos proves alignment with transport, low cos proves orthogonality to it. **Minimal repair.** Counterfactual: re-roll the same accepted `dFc` once with `Fp_nodemand`; `realized = cos(u, x_T^demand − x_T^nodemand)` — one extra forward rollout per commit.

## M18 — P-render's headline metric is not recorded in the ladder's own run configuration.

`render_work`/`phys_work` are assigned only inside `if on_iter is not None:`; headless runs record None. **Minimal repair.** Hoist `_linearized_work` out of the on_iter guard (or gate on cfg.work_telemetry that stages 2-4 set).

## M19 — Stage 4's wall-clock matching silently changes the pacing schedule.

Wall-clock-matching lg2 at 3× per-commit cost means ~100 commits vs 300 — raising `pace` 0.0153 → 0.0455 (3× step cap), confounding the mechanism with the glidepath. **Minimal repair.** Match on commits with wall-clock reported, or hold `cfg.pace` fixed directly.

# MINOR

- **m20** — SV band violated ~2% at saturation (alternating projection ends on det plane). [FIXED in production 2026-09-02: exact KKT projection.]
- **m21** — τ measured pre-assimilation ⇒ ≈2× the intended per-window budget.
- **m22** — `ok = det(Fe) > 1e-6` on `A·F_e`: a compressive demand can silently drop both demand AND global assimilation for marginal particles; add `n_assim_skipped`.
- **m23** — reusing `cov_smin/cov_smax` as a per-splat σ band imports two unvalidated constants under a "no new constant" banner.
- **m24** — K-R2's "window-start kinetic kick" has no recorded field (rec["kin"] is terminal).
- **m25** — §11.7 infers oscillation amplitude from `move` (a step size); a limit cycle of 0.05 wu traversed in 15 small steps has the same `move`.
- **m26** — the lg post-pass metric recompute (`runner.py:325-334`) overwrites d_render with pure silhouette; its planned deletion must pair with B1's repair.

# Verdict

**REDESIGN.**

- **Tier R: redesign.** The demand sits in the exact reachable range of dFc and is cheap to annul while w_kin prices it as a cost (B2); the non-coaxial part is absorbed into rotation that a corotational energy cannot turn into stress (M6); the convergence gate cannot fire early → full cost / zero donation (M10); λ_loc pinned at cap (M12); containment not implementable as written (B3, B4); ratchet falsifier vacuous (M9); ratchet real in area and, at default assim_iso, in volume (M7, M8); the retired guard still has half its precondition live (M16).
- **Tier D: implement-with-repairs, as a separately gated arm** — only after B1 (d_sil/d_gauss split; gates and freeze read d_sil only), B5 (configure_source on c2f rebuild), M14 (world-space cap), M15 (drop or bound s_pc + Nyquist recompute). Its theoretical justification (M13) does not hold: run as a pure empirical bet with K-D1/K-D2 on a channel no gate consumes.
