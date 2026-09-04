# REFUTE review — H⁻¹ mass-balance tranche (Codex gpt-5.6-sol xhigh, 2026-09-04)

Reviewed: commits 7ed1337, 8c2d523 (H⁻¹ term `d_h1`, gate reject-streak fix, oscillation
Addendum 7, v4/v5/v5b/v6 falsifications). Reviewer verdict: do not adopt before the v7
census; FFT half-spectrum weighting verified correct; pad-2 vs pad-4 gradient difference
7.2e-6 (wrap is not a blocker). Implementer responses below; fixes are in commit
"REFUTE fixes" of the same day.

| # | Finding | Response | Status |
|---|---|---|---|
| 1 | Critical: c2f rebuilds `TargetPack` → `h1_scale` None → mid-run recalibration; calibration in x-space, not control space | Real. Rebuild now carries `h1_scale`/`jd_scale`/`jd_rho0` over (runner c2f block). x-space calibration kept as the pre-registered rule (same as kde/jdens); the control-space mismatch is documented as a caveat in method.md; the census attribution uses raw D_vol, which the calibration cannot fabricate. | FIXED (rebuild); caveat documented |
| 2 | High: any `w_h1>0` gives H⁻¹ a full normalized gate/plateau/delivery vote | By design and consistent with every active track (dt/kde/jdens): a shape term measuring the same residual votes like the others; `w_h1` weights the objective, not the vote. The concern that delivery ≠ min-D_vol state is checked directly: `solid_census.py` now prints the delivered commit vs the min-d_vol commit. | ANSWERED; measured per run |
| 3 | High: H⁻¹ inside `phys_core` changes the λ-balancer numerator and the PCGrad reference (cos(g_vol,g_h1)≈0.14) | Real, and the codebase's own precedent (W1 term moved out, finding 9). H⁻¹ moved into `dt_term` (fixed weight, not λ-scaled, outside the PCGrad reference). v7a/v7b ran the in-core version; an isolation arm v7c (out-of-core) is pre-registered to attribute the gain. | FIXED; v7c pending |
| 4 | High: `reject_streak>=2` charges patience for DISTINCT over-threshold attempts (b4/b6 class); silent change for w_h1=0 runs | Real. Replaced by a replay signature: a brake reject counts stale only when its merit equals the previous rejected candidate's within max(outer_merit_tol, 10×replay_rel). Distinct attempts stay free. | FIXED |
| 5 | Medium: "L2 blind inside a uniform surplus" overstated for the log-mass form (+0.05/+0.15, not 0) | Accepted; wording changed to "weak". The probe shows contrast (H⁻¹ +0.46 vs L2 +0.05), not causation of the floor — the v7 census is the causal test. Grid-translation robustness not yet run. | WORDING FIXED; robustness open |
| 6 | Medium: lateral decompression unmeasured; report out_nn trajectory | Measured: band lateral radial component +0.50 for H⁻¹ (66% lateral-dominant), −0.20 for L2; outward-normal on off-support particles ~0 for both. Census now prints floater_frac first/max/last. Arbiter = v7 census. | MEASURED; verdict pending |
| 7 | Medium: v5b "undo" reading wrong (reversal_cos +0.93 = aligned); class-level falsification unsupported | Accepted: the candidate continued the previous step and overshot after a near-zero-gradient calibration hit the 1e3 cap. Docs corrected: parity-at-zero-gradient calibration is falsified, the prior class is parked, not disproved. | DOCS FIXED |
| 8 | Medium: oscillation closure uses increments, not amplitude | Amplitude measured (drift-removed peak-to-peak over the 40-commit tail): median 0.16 sp, p99 0.42, max 1.03; ears p99 0.44. Closure holds; reopening rule restated on amplitude. | MEASURED; closure stands |
| 9 | Low: sign of ∂D/∂x, grid units, 401-vs-400 mass test | Sign and units corrected in method.md; test rewritten with equal total mass. | FIXED |

Most informative next measurement (agreed): matched v1 / v7a / v7c traces vs cumulative
inner iterations (not commit index) with raw D_vol, band/pole ratios, out_nn, and the
c2f boundary marked. `solid_census.py` covers the endpoint; the trace comparison is queued
for the v7 verdict.
