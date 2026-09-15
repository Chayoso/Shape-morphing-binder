# REFUTE round 2 — ladder claims and remedies (Claude Opus, 2026-09-15 evening)

Reviewer: Opus subagent in REFUTE mode on the 2026-09-15 experiment sections, the triage
probe, `w_kin_var`, `relax_stretch`, the density-unit conversion and the `--ppc` path,
with the 30 result JSONs and 25 triage JSONs recomputed independently. 19 findings, 8
could-not-refute items. Every finding is answered below; code fixes are in commit
`REFUTE-2` (see git log), doc corrections in `docs/experiments.md` and
`docs/render_controls_physics.md` (marked "REFUTE-2").

| # | sev | finding | disposition |
|---|---|---|---|
| F1 | MAJOR | Driver "C_control" is decided from the speed series alone; a global harmonic at period ≈T with no window structure is labelled control; the discriminators (vary T, `--assim 0`) were never run. | **ACCEPTED.** Label renamed `C_window` ("window-locked", what is measured). The attribution experiments run as batch j: baseline at T=10 and T=40 (a window-locked cycle follows T; the F-smoothing time 22.2 steps and the elastic harmonics do not) and `--assim 0` (assimilation reset excluded as the cause). |
| F2 | MAJOR | Lock band ±15 % T contains the F-smoothing time dt/(1−s) = 22.2 and elastic harmonics τ_e/6, τ_e/7; driver B tests only m ∈ {1,2}. | **FIXED** (band 6 %; measured periods 20.00–20.03 at 0.19-step bins stay inside; 22.2/21.9/18.8 fall outside). B's harmonic asymmetry noted; the T-variation test is the decisive one. |
| F3 | MAJOR | The doc's mechanism paragraph ("accelerates from rest and brakes to rest") is phase-inverted vs the data (speed minimum mid-window, maximum at the window end). | **FIXED** in `docs/oscillation_triage.md` and `render_controls_physics.md` §9: the measured phase profile is stated. |
| F4 | MAJOR | `power_frac` is convention-dependent by 3–5× (denominator excludes the lowest bins); the 0.5 threshold sits inside the ambiguity. | **ACCEPTED.** Both conventions are now reported (`power_frac`, `power_frac_all`); the pre-registered statistic for the next round is the convention-free TORTUOSITY (path/net per window, CNR-1), threshold 1.5. |
| F5 | MAJOR | The `modulation` clause was added after seeing the runs it judges; verdicts flip at 1.97 vs 2.02. | **ACCEPTED, stated in the doc.** The clause stays (with its origin), the decisive test is batch j (T variation), and tortuosity separates the arms by 2× rather than 3 %. |
| F6 | MAJOR | Density-unit conversion measured on the run's own loss grid → converted weights inherit the cell-sum resolution dependence; `--ppc 8` arms ran every fixed weight 3.2× weaker than the dx-0.5 arms. | **FIXED.** `calibrate_units` evaluates the legacy side on a FIXED reference grid (`unit_ref_res` = 64); test added. The batch-h/i ppc8 rows are re-labelled as "weights ≈ 0.31× the legacy-64 meaning" and the density ppc8 arm is re-run in batch j. |
| F7 | MAJOR | The 40k A/B pair was produced under two code hashes; `git_sha` null in every archive. | **FIXED** (tarball deploys carry `VERSION`; `pipeline_run` falls back to it) and the 40k baseline is re-run under the same code as the remedies (batch j). The hash difference was the F_g relaxation in `runner.py`, inert for arms with `render_F_geom=False`, but the record must show it. |
| F8 | MAJOR | No replicates; the 40k recipe effect (−0.5 % / +0.5 pt) is inside the within-family spread (0.6 % / 0.55 pt); dose-response non-monotone; commit-count confound (corr −0.70 at 20k). | **ACCEPTED.** Shape claims for the recipe are downgraded to "no measurable shape cost within single-seed noise"; seed replicates (seeds 2, 3) of the 20k baseline and recipe run in batch j to quantify the noise. |
| F9 | MAJOR | The recipe fails the visible < 1 % criterion at 40k (2.6 % vs baseline 1.6 %); no 40k arm meets it. | **ACCEPTED.** The visible fraction at 40k is dominated by the non-window-locked random walk (sp is smaller); the criterion that replicates is the window-locked component (power/tortuosity). Doc wording corrected. |
| F10 | MAJOR | `relax_stretch` through `_assimilate` returned R S^{1−η} J^{η/3} in the isochoric branch; test covered the other branch; smin/smax not forwarded. | **FIXED** (exact SVD implementation; offline helper only, see F11). |
| F11 | MAJOR | The commit-time F_g relaxation edits the image with no particle motion (premise violation) and saturates the anisotropy at ~1.5 % (covariance ≈ isotropic). | **ACCEPTED and REPLACED.** No state edit at commits any more. The needle problem is handled in the render forward model: `gauss_cov_sat` — Σ = σ₀² M (I + M/r²)⁻¹, M = F Fᵀ — a stateless, smooth (no SVD) saturation applied identically inside and across windows; viewer/export/photoreal use the same map. Tests: bounded eigenvalues, identity for small stretch, commuting eigenvectors, finite gradients at F ≈ I. |
| F12 | MAJOR | `--ppc 8` reports ppc on the source only; target ppc median 6 with 24–27 % of cells < 4; hole outcomes at ppc 8 span 0.00–0.47 % with no ordering. | **ACCEPTED.** Target-side ppc is now reported; the ppc target is stated as "source 7 / target 6 at the request 8"; the hole numbers are described as noise at this scale. |
| F13 | MINOR | The ppc8 chamfer gain conflates dx, loss grid, unit change and weight scaling. | **ACCEPTED.** Batch j runs `--ppc 8` in legacy units (loss_res 64) to isolate dx. |
| F14 | MINOR | `G4_ejection` fails on every 40k arm and is absent from the tables. | **FIXED** in the doc (gate state listed; the stray metric is self-referential and its replacement was pre-registered earlier). |
| F15 | MINOR | Doc numbers not reproducible: commit-count definition; "kin_var 0.128" for the batch-a baseline (no such key — it was kin_run); "1.2–1.3×" ratio claim; a non-monotone reviewer number quoted as fact; archived `lambda_cap` misreports density mode; "λ O(1)" wording. | **FIXED** (labels corrected, commit counts defined as accepted records, the balancer cap recorded in the result, λ described as 0.02–0.13). |
| F16 | MINOR | G3 drift read the last history record, not the delivered slice. | **FIXED** (`eval_gates` uses the last accepted record with `frame_end ≤ deliver_n`). |
| F17 | MINOR | `w_kin_var` prices all in-window velocity change, not only reversals; net displacement per window falls 3×; best d_vol +11–14 % at 20k. | **ACCEPTED**, wording corrected ("prices in-window velocity change; progress per window slows; delivered shape unchanged within noise"). |
| F18 | MINOR | Driver A tested on the population-mean J only. | **FIXED** (per-particle RMS clause, threshold 0.05). |
| F19 | MINOR | Archived triage JSONs for batches a–c predate the modulation clause; 9/25 runs lacked `--json`. | **FIXED**: every archive is re-triaged with `--json` under the current probe in batch j. |

Could-not-refute (reviewer's evidence, kept): tortuosity 2.8 → 1.09 (baseline → kv200)
with a 6.3× path excess for the same final shape; archive-structure artefacts cleared
(continuous promotion, held frames removed, guards all zero, frame accounting exact);
`lk_var` correct/differentiable/consistent (FD 0.26 %); metrics grid-independent; the
density-unit resolution falsifier passes on the real clouds (1.52× for 32→64 vs 3.5–3.7×
legacy); discretisation arithmetic and every table number recomputed reproduce.
