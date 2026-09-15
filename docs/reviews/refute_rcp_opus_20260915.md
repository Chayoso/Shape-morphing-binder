# REFUTE — render-controls-physics contract (Claude Opus, 2026-09-15)

Reviewer: Claude Opus subagent in REFUTE mode on the uncommitted working tree of
`v3-grid-gs` (21 tracked + 19 new files). Every finding below is answered; the
implementer's disposition follows each. Numbers in the findings are the reviewer's
CPU measurements. The reviewer also listed 20 things it tried to refute and could
not (bottom).

| # | sev | finding (reviewer) | disposition |
|---|---|---|---|
| F1 | MAJOR | `unit_ratio = n·2m/(1+m)` is 4–45× below the true loss ratio and 5–44× below the gradient ratio; one scalar cannot serve weights and eps. | **FIXED.** `runner.calibrate_units` measures both ratios at the source (`unit_ratio` = loss ratio → weights; `unit_grad_ratio` → eps/target_norm). Logged per run. Test `test_density_units_are_measured_at_the_source`. Doc §2 rewritten. |
| F2 | MAJOR | Basis grids 2–3 degenerate (one node); `project` is a lumped smoother losing 38–61 %; the test passed vacuously. | **FIXED.** Node box = bbox + 5 %/side (every G ≥ 2 valid); `project` = least-squares via Jacobi-PCG on the normal equations (<2 % reproduction at G 2/4/12, no contraction). Tests parametrised over grids + a non-degeneracy test. |
| F3 | MAJOR | "blend" with β = λ‖g_r‖/‖g_p‖ is direction-identical to the sum. | **FIXED.** β is fixed (= α_λ); documented as the un-EMA'd, uncapped, magnitude-anchored composite. Test exercises `combine` and asserts a distinct direction. |
| F4 | MAJOR | `--ppc` rescales legacy D_vol ×535 by moving loss_res; literals for E/ν; prm rebuilt from defaults. | **FIXED.** loss_res follows dx only under `--loss_units density`; material from `PipelineConfig()`; `dataclasses.replace`. Doc §7 + ladder row corrected. `sigma0` documented as reported, not enforced. |
| F5 | MAJOR | `gauss_scale`/`kde_scale` re-calibrated at the c2f rebuild (hidden weight step in both gauss arms). | **FIXED.** Both (and the unit ratios) join the `keep` tuple. Test `test_c2f_rebuild_keeps_every_one_shot_calibration` (kde path, CPU). |
| F6 | MINOR | Chebyshev worse below ~8 sweeps; ω recursion not restarted at ω₁=1; ρ estimate is the periodic bound. | **FIXED** (guard: plain sweeps below 8; recursion restarted at ω₁=1). ρ note added; the estimate is kept (conservative for convergence). Test added. |
| F7 | MINOR | `w_cov` and the Gaussian shape diagnostics read the physics F while F_g is rendered. | **FIXED.** Both read the rendered F (F_g under `render_F_geom`). |
| F8 | MINOR | `on_iter` streamed the physics F, `on_commit` F_g. | **FIXED.** `on_iter` streams F_g when it is what the viewer renders. |
| F9 | MINOR | kNN gather built and multiplied by zero when only `control_h1` needed the list. | **FIXED.** Penalty gated on `w_creg > 0`. |
| F10 | MINOR | `Fg_commits` archived whole and past the delivered slice. | **FIXED.** Filtered to `frame ≤ deliver_n`; its cadence equals `F_samples` at the default stride (both one per commit), so no extra subsampling. |
| F11 | MINOR | `dfc_clip` on a node leaf is a conservative per-particle bound → basis-vs-flagship at a fixed clip confounds budget. | **ACCEPTED, documented** (§4); the ladder reports clipped and unclipped runs. |
| F12 | MINOR | `lambda_capped` read at window end reports a stale flag. | **FIXED.** Captured at the balancer update (`it == 0`). |

Could-not-refute list (kept as evidence, reviewer's numbers): extended-bridge adjoint
FD 0.004–0.037 % with all five outputs seeded, additivity 5e-6; `v_T` seed not
double-counted (0.0 difference); F_g uncontaminated by damped C (η=0 in the bridge)
and is the discrete flow-map Jacobian; Fg0 promotion/rollback/null/hold/`Fg_commits`
alignment correct; control_h1 × node leaf fails fast; `w_creg` indexing correct on the
expanded field; partition of unity 1.2e-7 incl. outside the box; per-node clip ⇒
per-particle bound (convex); CAGrad dual/convexity/sign correct; λ estimated from the
vector it multiplies in every mode; per-channel rescale is not a bias; autograd reaches
the leaf through F_g in the non-special branch; Chebyshev fixed point = plain; dx/ρ/c/CFL
formulas and the §7 numbers reproduce; `--ppc` loss-grid geometry right (the defect was
the units, F4); density conversion covers gd_tol/pace/outer_merit/best_truncate/
phys_track/fill/noise floor/replay_tol; density D_vol shares the minimiser; DOF numbers
in §4 right; `cap_rel` logic sound; Charbonnier covers both render paths.

After the fixes: `python -m pytest tests/ -q` → 186 passed (2026-09-15).
