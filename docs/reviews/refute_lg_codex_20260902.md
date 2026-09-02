 a bad `Fp` permanently while §11.2’s claimed rollback never executes.

   Minimal repair: maintain a persistent transaction `{Fp_plain, Fp_demand, origin_commit}`. Finalize it only after the following window passes outer arbitration; restore `Fp_plain` on every empty, invalid, converged, frozen, rejected, exception, c2f, and budget-termination path.

7. **BLOCKER — retiring the W1/fill guard recreates an alternating accept/reject deadlock.**  
   Evidence: `docs/local_global_design.md:57-74,377-387`; `physmorph/pipeline/runner.py:134-140,439-486,498-508`; `physmorph/pipeline/optimizer.py:302-332,450-457,509-543`.

   Concrete sequence near a thin feature:

   1. DT/NN/fill accepts cleanup displacement `+q`.
   2. The Gaussian-only local energy asks for opposing correction `−q` and bakes it into `Fp`.
   3. The next passive rollout realizes `−q`, while global `dFc` pushes `+q` to satisfy DT/NN/fill.
   4. If demand wins, the latched fixed merit rejects the reversed candidate.
   5. Rollback/cold retry accepts cleanup again, and Tier R emits the same demand again.

   Every accepted cleanup resets `stale`; every intervening rejection consumes an animation index. A 300-attempt paced run can deliver roughly 150 useful global commits without converging. If anti-correction eliminates Armijo-sized descent, repeated nulls reproduce the s4 absorbing freeze. Thus “can cost budget, never correctness” is false under fixed attempt and wall-clock budgets.

   Minimal repair: retain the current `lg_sweeps + (w_dt | w_fill)` guard until the local phase includes the same cleanup objectives or an existing-weight directional veto.

8. **BLOCKER — `dFc` can exactly cancel the demanded prestress before P2G.**  
   Evidence: `docs/local_global_design.md:239-246,303-309`; `physmorph/mpm/kernels.py:37-46,184`; `physmorph/pipeline/config.py:31,49`; `physmorph/pipeline/optimizer.py:350-370`.

   Let `F=Fp=I`, `e=diag(ε,-ε,0)`, `A=exp(e)`, `η=0.5`. Demand produces `Fp_new=A^0.5`, so zero control gives the intended elastic state `A^-0.5`. But choosing

   \[
   dFc_0=A^{0.5}-I
   \]

   makes `(F+dFc_0)Fp_new^-1=I` exactly, eliminating the stress before grid transfer. At `ε=1/3`, the largest component is only about `0.18`; `dfc_clip` defaults to zero and `w_ctrl=10^-3`, so cancellation is cheap. G2P then writes `F+dFc`, potentially changing Gaussian covariance without producing demanded transport.

   The warm-start safeguard explicitly retains any such anti-control if it beats the zero start under the changed `Fp`.

   `lg2_realized_cos/frac` sees only the net displacement and cannot distinguish cancellation from damping or ordinary aligned motion. Required telemetry is:

   - zero-control terminal difference under `Fp_demand` versus `Fp_plain`;
   - optimized demanded terminal minus zero-control demanded terminal;
   - projection of `dFc[0:T]` onto the analytic stress-neutralizing control;
   - passive, cancellation, and net realization fractions separately.

   Preventing cancellation requires reserving or penalizing the demand-neutralizing control subspace, which breaks the unchanged-global-solver premise.

9. **BLOCKER — Tier D directly launders the outer merit and freeze accounting.**  
   Evidence: `docs/local_global_design.md:169-174,250-258,402-410`; `physmorph/pipeline/optimizer.py:209-229,573-580`; `physmorph/pipeline/runner.py:350-416,425-508`.

   With Gaussian loss enabled, `d_render` is not raw silhouette. It is either Gaussian loss or `silhouette + scaled Gaussian`; runner then uses it as `rend_track`, `components["render"]`, `improved`, `best_rend`, outer merit, annealing, and freeze state.

   Tier D lowers the Gaussian residual after commit `k`, but the planned removal of post-pass recomputation leaves `rec[k]` unchanged. At `k+1`, the frozen lower residual appears as “render improvement” even if the raw state is stationary or worse. That resets `stale`, increases `anneal`, and can authorize an otherwise marginal commit.

   Dressing can touch these exact `rec[]` fields through the global objective: `loss`, `d_render`, `lambda`, `grad_norm`, `g_cos`, `g_raw_cos`, `g_share`, `g_rend_norm`, `render_work`, `render_work_x`, `render_work_F`, `step_norm`, `predicted_decrease`, `dfc_absmax`, `s_absmax`, `accepted`, `rejected`, and `iters`. It consequently changes `outer_merit`, `outer_gain`, `outer_gate_latched`, `outer_accepted`, `outer_rejected`, and `null_commit`. Raw-state fields are indirectly changed through the different optimized control.

   Conversely, `gauss_scale_*` diagnostics use parent `F` only (`gauss_loss.py:44-76`) and will not report child log-scale dressing.

   Minimal repair: log `d_render_obj` and `d_gauss_dressed` separately, recompute a raw `d_render_gate` from undressed `x` and target silhouettes, and use only `d_render_gate` for outer merit, bests, stale, annealing, and freeze.

10. **BLOCKER — convergence of the revived band solve says nothing about the emitted Tier-R demand.**  
    Evidence: `docs/local_global_design.md:181-205`; `physmorph/pipeline/surface_local.py:117-173,203-235`.

    The solver descends an objective in `x+u` and `(I+∇u)F`; Tier R emits only `exp(clamp(sym∇u))`. These are different variables and different maps.

    A locally constant translation can lower Gaussian loss while emitting `A=I`. A simple shear is converted into a different SPD stretch. Worse, `_psi_snh` remains finite for `det(I+∇u)≤0`; a reflected virtual map can converge, while Gaussian covariance is insensitive to the reflection sign and the emitted exponential is merely a bounded contraction. `lg_converged=True` therefore does not establish descent—or even directional agreement—for the actual demand.

    Minimal repair: optimize directly in the emitted SPD, preferably traceless, demand variables. At minimum, re-evaluate the mapped `A` objective and donate demand only if it decreases relative to `A=I`.

11. **MAJOR — the envelope-theorem claim is applied at the wrong state.**  
    Evidence: `docs/local_global_design.md:153-165`; `physmorph/pipeline/runner.py:222-226`; `physmorph/pipeline/optimizer.py:236-259,596-619`.

    Dressing is optimized at promoted `(x_k,F_k)`. The global derivative is evaluated at the terminal state of the next `T=20` rollout, starting with nonzero `v/C` and a newly modified `Fp`. Even at zero `dFc`, that terminal state generally differs from `(x_k,F_k)`. Therefore the frozen dressing is not locally optimal at the point where the gradient is taken, and high-frequency residual leaks straight back into `dFc`.

    Refitting dressing on a zero-control predicted terminal is only approximate. Exact envelope semantics require profiling dressing for every candidate state, which destroys the fixed-objective cost model. The exact claim should be removed unless this profiling is implemented.

12. **MAJOR — c2f both bypasses demand arbitration and mishandles Gaussian children.**  
    Evidence: `docs/local_global_design.md:429-435`; `physmorph/pipeline/runner.py:105-111,168-171,204-213,427-451`; `physmorph/pipeline/gauss_loss.py:124-158,216-217`.

    c2f changes `cfg.render_res`, while Gaussian loss uses independent `cfg.gauss_res` plus its own Nyquist floor. Tier D’s Gaussian objective normally has not changed resolution, so resetting dressing manufactures a residual discontinuity and later phantom improvement.

    Rebuilding `TargetPack` constructs a new `GaussViews` with `source_offsets=None`; `configure_source` is called only before the outer loop. With children enabled, the next loss raises `RuntimeError`.

    Finally, c2f clears `outer_prev` and `outer_scales`. The first post-c2f candidate has no `outer_gain`, so a demand emitted immediately before c2f cannot be rejected by the fixed-merit comparison.

    Minimal repair: retain dressing/source bases across silhouette-only c2f, explicitly reconfigure children when rebuilding Gaussian state, and suppress or separately validate the pending pre-c2f demand with a carried comparator.

13. **MAJOR — rollback and warm-start state remain nontransactional.**  
    Evidence: `physmorph/pipeline/runner.py:146-152,216-251,456-473`; `physmorph/pipeline/surface_local.py:217-227`; `physmorph/pipeline/optimizer.py:350-371,436-448,606-647`.

    Outer rollback stores the global balancer but not `lg_balancer`. A rejected local candidate therefore changes the next commit’s `λ_loc`, defeating the claimed deterministic retry semantics.

    Empty or replay-invalid windows can retain changed global lambda, material `s`, and `dfc`: runner assigns them before checking `whist`, while the null path resets only moments. After pending-demand rollback, those controls were optimized under the wrong `Fp`.

    The warm safeguard checks only whether a control improves the total objective under changed `Fp`; it does not check whether that improvement comes from canceling the demand. With `mom_carry` and no `dfc_init`, stale moments can be loaded without the safeguard running at all.

    Minimal repair: make `s`, `dfc`, moments, both balancers, dressing, and pending `Fp` one deep-cloned transaction; restore all of them on every discarded path. A new demand should cold-reset controls/moments unless a cancellation-aware safeguard passes.

14. **MAJOR — the cost model undercounts thousands of all-view rasterizations per commit.**  
    Evidence: `docs/local_global_design.md:341-350`; `physmorph/pipeline/surface_local.py:125-173`; `physmorph/pipeline/gauss_loss.py:212-222`; `physmorph/pipeline/config.py:15,21,59-60`.

    With eight active colors and ten sweeps, the retained solver performs `171–251` full energy forwards and `90` backwards per Tier-R commit. At 18 views, that is approximately:

    - Tier R: `3,078–4,518` renderer forwards plus `1,620` backwards.
    - Tier D with 20 iterations and 1–10 backtracks: another `720–3,960` forwards plus `360` backwards.
    - Combined: `3,798–8,478` view forwards plus `1,980` backwards per accepted local pass.

    The old `3.3×` measurement used the cheap CIC silhouette inside this loop, so it cannot predict the all-view 3DGS version. Also, `τ` is applied after solving and convergence is relative to `g0`; shrinking `τ` does not cause the claimed late 1–2-sweep exit.

    At the measured `N=20k,T=20` baseline of about `1 s/commit` on the RTX 6000 Ada (`docs/experiments.md:46-50`), a crude linear projection gives about `2 s` for the requested `N=40k,T=20,dt=1/240,dx=0.5,smoothing=0.955` global commit. Even the obsolete optimistic `3.3×` proxy gives roughly `6.6 s/commit`, `33 minutes` per 300-commit pair, and `2.2 hours` for four pairs. A wall-clock-matched baseline budget would buy only about 91 lg2 attempts—and likely far fewer with 3DGS—so the 300-commit-derived pace schedule does not survive.

    Minimal repair: require a stage-0 GPU microbenchmark at actual resolution, child count, and active-node count; report accepted/rejected p50 and p95 times, then recompute animation count and pace for wall-clock matching.

15. **MAJOR — Tier D’s stated capacity bounds are not jointly enforced.**  
    Evidence: `docs/local_global_design.md:136-149,167-174`; `physmorph/pipeline/config.py:183-185`; `physmorph/pipeline/gauss_loss.py:26-41,44-76`.

    The `cov_smin/cov_smax` band currently bounds singular values of parent `F`. Reusing the same range for child `exp(s)` multiplies the bounds: a legal parent singular value `2` and legal child multiplier `2` produce rendered scale `4`, while diagnostics still report `2`. Thus the claimed viewer-legal band and anti-gauge armor do not hold.

    The offset constraints also require projection onto the intersection of per-child balls and the zero-centroid plane. “Clamp each child, then subtract the centroid” can violate the cap: `[r,r,-r]` becomes `[2r/3,2r/3,-4r/3]`.

    Minimal repair: constrain total rendered scale `sv(F)·exp(s)`, log child scales explicitly, and use an exact joint projection for centroid plus offset caps.

16. **MAJOR — there is no per-frame dressing state, so rendered deliverables cannot satisfy the repository’s QA contract.**  
    Evidence: `docs/local_global_design.md:319-339`; `physmorph/pipeline/runner.py:124-129,317-320,478-493,522-524`.

    Runner archives every intermediate `x/F` frame, but the design adds no aligned dressing history and does not pass dressing through `on_commit`. Rendering earlier frames with the final dressing retroactively changes their observation, creating pre-echo or temporal pops; rendering every frame with baseline dressing does not show what was optimized.

    Minimal repair: archive `dressing_frames` one-to-one with `frames`: pre-window frozen dressing for rollout intermediates, newly accepted dressing for the terminal frame, copied values during holds, and matching truncation on rollback.

**Verdict: REDESIGN.**
tokens used
139,662
1. **BLOCKER — Tier R does not preserve rotation; the polar step does not commute with `A`-premultiplication.**  
   Evidence: `docs/local_global_design.md:200-225`; `physmorph/plasticity/assimilation.py:15-23,59-73`.

   Ignoring projections, the proposed call computes

   \[
   S'=\sqrt{F_e^T A^2F_e},\qquad F_p^+=S'^\eta F_p,
   \qquad F_{e,\mathrm{actual}}^+=F_eS'^{-\eta}.
   \]

   The old polar rotation is preserved only when `S'` commutes with the old right stretch. That is not generally true. A legal counterexample is

   \[
   F_e=\mathrm{diag}(2,1,0.5),\quad
   e_{01}=e_{10}=1/(3\sqrt2),\quad A=\exp(e),\quad \eta=0.5.
   \]

   Here `||e||F=1/3`, `det(A)=det(F_e)=1`, and the initial rotation is identity, but the resulting actual elastic state has about `2.17°` polar rotation. Isochoric normalization is a no-op in this example; an isotropic growth factor does not restore commutation. Therefore “rotation demand discarded” and “per-particle exact” are false for both isochoric branches.

   Minimal repair: formulate a dedicated demand assimilation map whose post-demand elastic polar state is specified explicitly. Restricting demand to the eigenbasis of `S_e` would preserve rotation but discards non-coaxial corrections.

2. **BLOCKER — the advertised SV-band invariants are already false in two assimilation branches.**  
   Evidence: `docs/local_global_design.md:219-237,360-365`; `physmorph/plasticity/assimilation.py:72-90`.

   In the isochoric/no-growth branch, four alternating projections finish with an unconstrained mean subtraction. Starting from singular values `[0.2, 5, 0.2]` produces approximately `[0.445735, 5.033226, 0.445735]`: determinant one, but outside `smax=5`.

   In the growth branch, the determinant governor runs after the SV clamp and undoes it:

   - `[0.2, 0.2, 5] → [0.29876, 0.29876, 7.46901]`
   - `[0.2, 5, 5] → [0.133887, 3.34716, 3.34716]`

   Runner guards inspect `x/F/v/C`, not `Fp` (`runner.py:255-273`), and `_state_ok` does not validate `Fp` (`optimizer.py:110-125`). A malformed demand can therefore pass the current commit and be discovered only by the next rollout—or survive via a null path.

   Minimal repair: project log singular values exactly onto the intersection of the box and target log-determinant plane/interval using water-filling or scalar bisection, then add an immediate finite/determinant/SV `Fp` guard.

3. **BLOCKER — `τ` permits both the old volume ratchet and an isochoric surface-area ratchet.**  
   Evidence: `docs/local_global_design.md:233-237,266-277,389-400`; `physmorph/pipeline/config.py:215-223`.

   `e` is not made traceless. With the repository default `assim_iso=False`, take every accepted window to carry `F_e=exp(0.1)I` and let the persistent residual request `e=0.1I`. This is within `τ=||log S_e||F`. At `η=0.5`, every commit multiplies `det(Fp)` by `exp(0.3)≈1.35`, monotonically approaching the unconstrained determinant ceiling `5³=125`. With growth enabled, this accidental volume change is merely mixed into the commanded-growth governor.

   Isochoric mode does not solve the ratchet generally. Let

   \[
   B=\mathrm{diag}(e^{0.1},e^{-0.1},1),\quad F_e=B,\quad A=B.
   \]

   Then `τ` remains constant and `η=0.5` multiplies `Fp` by `B` each accepted commit. `det(Fp)=1`, but the singular values ratchet toward `[5,0.2,1]`; the surface area of a unit rest cube rises from `6` to `12.4`.

   The throttle is also nonlocal: a torso-dominated median authorizes the same strain cap on a locally stress-free ear particle. Static equilibrium need not have `S_e→I`, so accepted motion going to zero does not imply `τ→0`.

   Minimal repair: require Tier R to use `dev(e)` with `assim_iso=True`, and throttle on an incremental realized-demand quantity rather than stored elastic strain. A cumulative signed demand/rest-metric budget is also required.

4. **BLOCKER — K-R4 cannot detect the ratchet it claims to falsify.**  
   Evidence: `docs/local_global_design.md:365`; `physmorph/pipeline/runner.py:196-203`.

   Once frozen, runner only duplicates held frames. It never invokes Tier R or assimilation. Consequently `||ΔFp||` is identically zero during every held commit, even if evaluating the local pass would produce the same nonzero demand forever. K-R4 passes vacuously.

   Minimal repair: during the held phase, evaluate but do not apply the predicted `A/Fp` increment, and require that predicted increment to vanish.

5. **BLOCKER — the proposed single assimilation call cannot produce a demand-only rollback state.**  
   Evidence: `docs/local_global_design.md:207-214,319-323,389-397`; `physmorph/plasticity/assimilation.py:59-91`.

   The baseline and demanded states are different nonlinear branches:

   \[
   Fp_{\rm plain}=P(F,Fp_{\rm old}),\qquad
   Fp_{\rm demand}=P(AF,Fp_{\rm old}).
   \]

   Polar decomposition, clamping, isochoric projection, and growth governing make their difference non-invertible. Saving `Fp_old` as `Fp_predemand` removes ordinary accepted assimilation and commanded growth too. Restoring it while holding post-commit `x/F/v/C` creates a third constitutive state that was never gated.

   Minimal repair: compute and store both `Fp_plain` and `Fp_demand`; rollback must select `Fp_plain`. That requires two evaluations or an equivalent explicit refactor, contradicting the “one existing assimilation call” design.

6. **BLOCKER — pending demand survives null, replay-failure, immediate-convergence, freeze, and budget-end paths.**  
   Evidence: `docs/local_global_design.md:389-419`; `physmorph/pipeline/runner.py:215-251,495-524`; `physmorph/pipeline/optimizer.py:596-627`.

   Demand from commit `k` is already present when rollback for `k+1` is constructed. If `k+1` returns empty history because of line-search exhaustion or replay failure, runner exits before the outer gate. If `grad_converged` is set, it freezes immediately. A demand emitted on the last animation or on a commit that triggers patience is returned without ever crossing a validating rollout.

   Five null attempts can therefore freeze a bad `Fp` permanently while §11.2’s claimed rollback never executes.

   Minimal repair: maintain a persistent transaction `{Fp_plain, Fp_demand, origin_commit}`. Finalize it only after the following window passes outer arbitration; restore `Fp_plain` on every empty, invalid, converged, frozen, rejected, exception, c2f, and budget-termination path.

7. **BLOCKER — retiring the W1/fill guard recreates an alternating accept/reject deadlock.**  
   Evidence: `docs/local_global_design.md:57-74,377-387`; `physmorph/pipeline/runner.py:134-140,439-486,498-508`; `physmorph/pipeline/optimizer.py:302-332,450-457,509-543`.

   Concrete sequence near a thin feature:

   1. DT/NN/fill accepts cleanup displacement `+q`.
   2. The Gaussian-only local energy asks for opposing correction `−q` and bakes it into `Fp`.
   3. The next passive rollout realizes `−q`, while global `dFc` pushes `+q` to satisfy DT/NN/fill.
   4. If demand wins, the latched fixed merit rejects the reversed candidate.
   5. Rollback/cold retry accepts cleanup again, and Tier R emits the same demand again.

   Every accepted cleanup resets `stale`; every intervening rejection consumes an animation index. A 300-attempt paced run can deliver roughly 150 useful global commits without converging. If anti-correction eliminates Armijo-sized descent, repeated nulls reproduce the s4 absorbing freeze. Thus “can cost budget, never correctness” is false under fixed attempt and wall-clock budgets.

   Minimal repair: retain the current `lg_sweeps + (w_dt | w_fill)` guard until the local phase includes the same cleanup objectives or an existing-weight directional veto.

8. **BLOCKER — `dFc` can exactly cancel the demanded prestress before P2G.**  
   Evidence: `docs/local_global_design.md:239-246,303-309`; `physmorph/mpm/kernels.py:37-46,184`; `physmorph/pipeline/config.py:31,49`; `physmorph/pipeline/optimizer.py:350-370`.

   Let `F=Fp=I`, `e=diag(ε,-ε,0)`, `A=exp(e)`, `η=0.5`. Demand produces `Fp_new=A^0.5`, so zero control gives the intended elastic state `A^-0.5`. But choosing

   \[
   dFc_0=A^{0.5}-I
   \]

   makes `(F+dFc_0)Fp_new^-1=I` exactly, eliminating the stress before grid transfer. At `ε=1/3`, the largest component is only about `0.18`; `dfc_clip` defaults to zero and `w_ctrl=10^-3`, so cancellation is cheap. G2P then writes `F+dFc`, potentially changing Gaussian covariance without producing demanded transport.

   The warm-start safeguard explicitly retains any such anti-control if it beats the zero start under the changed `Fp`.

   `lg2_realized_cos/frac` sees only the net displacement and cannot distinguish cancellation from damping or ordinary aligned motion. Required telemetry is:

   - zero-control terminal difference under `Fp_demand` versus `Fp_plain`;
   - optimized demanded terminal minus zero-control demanded terminal;
   - projection of `dFc[0:T]` onto the analytic stress-neutralizing control;
   - passive, cancellation, and net realization fractions separately.

   Preventing cancellation requires reserving or penalizing the demand-neutralizing control subspace, which breaks the unchanged-global-solver premise.

9. **BLOCKER — Tier D directly launders the outer merit and freeze accounting.**  
   Evidence: `docs/local_global_design.md:169-174,250-258,402-410`; `physmorph/pipeline/optimizer.py:209-229,573-580`; `physmorph/pipeline/runner.py:350-416,425-508`.

   With Gaussian loss enabled, `d_render` is not raw silhouette. It is either Gaussian loss or `silhouette + scaled Gaussian`; runner then uses it as `rend_track`, `components["render"]`, `improved`, `best_rend`, outer merit, annealing, and freeze state.

   Tier D lowers the Gaussian residual after commit `k`, but the planned removal of post-pass recomputation leaves `rec[k]` unchanged. At `k+1`, the frozen lower residual appears as “render improvement” even if the raw state is stationary or worse. That resets `stale`, increases `anneal`, and can authorize an otherwise marginal commit.

   Dressing can touch these exact `rec[]` fields through the global objective: `loss`, `d_render`, `lambda`, `grad_norm`, `g_cos`, `g_raw_cos`, `g_share`, `g_rend_norm`, `render_work`, `render_work_x`, `render_work_F`, `step_norm`, `predicted_decrease`, `dfc_absmax`, `s_absmax`, `accepted`, `rejected`, and `iters`. It consequently changes `outer_merit`, `outer_gain`, `outer_gate_latched`, `outer_accepted`, `outer_rejected`, and `null_commit`. Raw-state fields are indirectly changed through the different optimized control.

   Conversely, `gauss_scale_*` diagnostics use parent `F` only (`gauss_loss.py:44-76`) and will not report child log-scale dressing.

   Minimal repair: log `d_render_obj` and `d_gauss_dressed` separately, recompute a raw `d_render_gate` from undressed `x` and target silhouettes, and use only `d_render_gate` for outer merit, bests, stale, annealing, and freeze.

10. **BLOCKER — convergence of the revived band solve says nothing about the emitted Tier-R demand.**  
    Evidence: `docs/local_global_design.md:181-205`; `physmorph/pipeline/surface_local.py:117-173,203-235`.

    The solver descends an objective in `x+u` and `(I+∇u)F`; Tier R emits only `exp(clamp(sym∇u))`. These are different variables and different maps.

    A locally constant translation can lower Gaussian loss while emitting `A=I`. A simple shear is converted into a different SPD stretch. Worse, `_psi_snh` remains finite for `det(I+∇u)≤0`; a reflected virtual map can converge, while Gaussian covariance is insensitive to the reflection sign and the emitted exponential is merely a bounded contraction. `lg_converged=True` therefore does not establish descent—or even directional agreement—for the actual demand.

    Minimal repair: optimize directly in the emitted SPD, preferably traceless, demand variables. At minimum, re-evaluate the mapped `A` objective and donate demand only if it decreases relative to `A=I`.

11. **MAJOR — the envelope-theorem claim is applied at the wrong state.**  
    Evidence: `docs/local_global_design.md:153-165`; `physmorph/pipeline/runner.py:222-226`; `physmorph/pipeline/optimizer.py:236-259,596-619`.

    Dressing is optimized at promoted `(x_k,F_k)`. The global derivative is evaluated at the terminal state of the next `T=20` rollout, starting with nonzero `v/C` and a newly modified `Fp`. Even at zero `dFc`, that terminal state generally differs from `(x_k,F_k)`. Therefore the frozen dressing is not locally optimal at the point where the gradient is taken, and high-frequency residual leaks straight back into `dFc`.

    Refitting dressing on a zero-control predicted terminal is only approximate. Exact envelope semantics require profiling dressing for every candidate state, which destroys the fixed-objective cost model. The exact claim should be removed unless this profiling is implemented.

12. **MAJOR — c2f both bypasses demand arbitration and mishandles Gaussian children.**  
    Evidence: `docs/local_global_design.md:429-435`; `physmorph/pipeline/runner.py:105-111,168-171,204-213,427-451`; `physmorph/pipeline/gauss_loss.py:124-158,216-217`.

    c2f changes `cfg.render_res`, while Gaussian loss uses independent `cfg.gauss_res` plus its own Nyquist floor. Tier D’s Gaussian objective normally has not changed resolution, so resetting dressing manufactures a residual discontinuity and later phantom improvement.

    Rebuilding `TargetPack` constructs a new `GaussViews` with `source_offsets=None`; `configure_source` is called only before the outer loop. With children enabled, the next loss raises `RuntimeError`.

    Finally, c2f clears `outer_prev` and `outer_scales`. The first post-c2f candidate has no `outer_gain`, so a demand emitted immediately before c2f cannot be rejected by the fixed-merit comparison.

    Minimal repair: retain dressing/source bases across silhouette-only c2f, explicitly reconfigure children when rebuilding Gaussian state, and suppress or separately validate the pending pre-c2f demand with a carried comparator.

13. **MAJOR — rollback and warm-start state remain nontransactional.**  
    Evidence: `physmorph/pipeline/runner.py:146-152,216-251,456-473`; `physmorph/pipeline/surface_local.py:217-227`; `physmorph/pipeline/optimizer.py:350-371,436-448,606-647`.

    Outer rollback stores the global balancer but not `lg_balancer`. A rejected local candidate therefore changes the next commit’s `λ_loc`, defeating the claimed deterministic retry semantics.

    Empty or replay-invalid windows can retain changed global lambda, material `s`, and `dfc`: runner assigns them before checking `whist`, while the null path resets only moments. After pending-demand rollback, those controls were optimized under the wrong `Fp`.

    The warm safeguard checks only whether a control improves the total objective under changed `Fp`; it does not check whether that improvement comes from canceling the demand. With `mom_carry` and no `dfc_init`, stale moments can be loaded without the safeguard running at all.

    Minimal repair: make `s`, `dfc`, moments, both balancers, dressing, and pending `Fp` one deep-cloned transaction; restore all of them on every discarded path. A new demand should cold-reset controls/moments unless a cancellation-aware safeguard passes.

14. **MAJOR — the cost model undercounts thousands of all-view rasterizations per commit.**  
    Evidence: `docs/local_global_design.md:341-350`; `physmorph/pipeline/surface_local.py:125-173`; `physmorph/pipeline/gauss_loss.py:212-222`; `physmorph/pipeline/config.py:15,21,59-60`.

    With eight active colors and ten sweeps, the retained solver performs `171–251` full energy forwards and `90` backwards per Tier-R commit. At 18 views, that is approximately:

    - Tier R: `3,078–4,518` renderer forwards plus `1,620` backwards.
    - Tier D with 20 iterations and 1–10 backtracks: another `720–3,960` forwards plus `360` backwards.
    - Combined: `3,798–8,478` view forwards plus `1,980` backwards per accepted local pass.

    The old `3.3×` measurement used the cheap CIC silhouette inside this loop, so it cannot predict the all-view 3DGS version. Also, `τ` is applied after solving and convergence is relative to `g0`; shrinking `τ` does not cause the claimed late 1–2-sweep exit.

    At the measured `N=20k,T=20` baseline of about `1 s/commit` on the RTX 6000 Ada (`docs/experiments.md:46-50`), a crude linear projection gives about `2 s` for the requested `N=40k,T=20,dt=1/240,dx=0.5,smoothing=0.955` global commit. Even the obsolete optimistic `3.3×` proxy gives roughly `6.6 s/commit`, `33 minutes` per 300-commit pair, and `2.2 hours` for four pairs. A wall-clock-matched baseline budget would buy only about 91 lg2 attempts—and likely far fewer with 3DGS—so the 300-commit-derived pace schedule does not survive.

    Minimal repair: require a stage-0 GPU microbenchmark at actual resolution, child count, and active-node count; report accepted/rejected p50 and p95 times, then recompute animation count and pace for wall-clock matching.

15. **MAJOR — Tier D’s stated capacity bounds are not jointly enforced.**  
    Evidence: `docs/local_global_design.md:136-149,167-174`; `physmorph/pipeline/config.py:183-185`; `physmorph/pipeline/gauss_loss.py:26-41,44-76`.

    The `cov_smin/cov_smax` band currently bounds singular values of parent `F`. Reusing the same range for child `exp(s)` multiplies the bounds: a legal parent singular value `2` and legal child multiplier `2` produce rendered scale `4`, while diagnostics still report `2`. Thus the claimed viewer-legal band and anti-gauge armor do not hold.

    The offset constraints also require projection onto the intersection of per-child balls and the zero-centroid plane. “Clamp each child, then subtract the centroid” can violate the cap: `[r,r,-r]` becomes `[2r/3,2r/3,-4r/3]`.

    Minimal repair: constrain total rendered scale `sv(F)·exp(s)`, log child scales explicitly, and use an exact joint projection for centroid plus offset caps.

16. **MAJOR — there is no per-frame dressing state, so rendered deliverables cannot satisfy the repository’s QA contract.**  
    Evidence: `docs/local_global_design.md:319-339`; `physmorph/pipeline/runner.py:124-129,317-320,478-493,522-524`.

    Runner archives every intermediate `x/F` frame, but the design adds no aligned dressing history and does not pass dressing through `on_commit`. Rendering earlier frames with the final dressing retroactively changes their observation, creating pre-echo or temporal pops; rendering every frame with baseline dressing does not show what was optimized.

    Minimal repair: archive `dressing_frames` one-to-one with `frames`: pre-window frozen dressing for rollout intermediates, newly accepted dressing for the terminal frame, copied values during holds, and matching truncation on rollback.

**Verdict: REDESIGN.**

[exited with code 0]
