# Oscillation triage probe

`scripts/probes/oscillation_triage.py` — a MEASUREMENT with pre-registered decision rules,
not a fix. Given a finished run archive it decides whether an observed vibration is
(A) a VOLUME driver, (B) a STIFFNESS driver, (C) a GRADIENT/CONTROL driver, or (D) none
(sub-spacing noise), and prints every number with the discretisation it was measured at
(AGENTS.md rule 4). Tests: `tests/test_oscillation_triage.py` (synthetic archives, numpy).

```
python scripts/probes/oscillation_triage.py --npz RUN_<arm>.npz [--json RUN.json --arm ARM]
    [--T 20] [--dt 1/240] [--dx 0.5] [--young 1.4e5 --poisson 0.2] [--out triage.json] [--png triage.png]
```

## Inputs (scripts/pipeline_run.py archive)

`RUN_<arm>.npz`: `frames (M,N,3)` per-substep positions (frames[0] = source; each accepted
commit appends its T rollout states, held/null commits ONE duplicated frame; outer-rejected
candidates roll back and append nothing), `deliver_n`, `src`, `tgt`, `F_samples (K,N,3,3)` at
`F_sample_idx`. `RUN.json`: `provenance` (T, `mpm`: dt, dx, smoothing, drag, grid_min) and
`arms[arm].history` (`frame_end`, `d_vol`, `kin`, `lambda`, `move`, `dfc_absmax`,
`reversal_cos`, `held`, `null_commit`, …) plus `arms[arm].config` (young, poisson). Only
`frames` and T are required; missing inputs are reported as "not measured". Parameter
precedence: CLI > json > defaults (dt 1/240, dx 0.5, E 1.4e5, ν 0.2), source tagged.

## What is measured

1. **Speed / windows.** `s_t = mean_p ||x_{t+1} − x_t|| / dt` over the delivered frames.
   Exact-duplicate frames (held/null) are removed as non-time. Windows are the T steps
   ending at each accepted record's `frame_end − 1`; without history, every T steps.
   Per window: stop-and-go `sag_w = (max_t s_t − s_end)/max_t s_t`, boundary jump
   `r_w = s_first(w+1)/s_last(w)`, and `(s_end/s_max)² < 0.05` ("kin at end below 5 % of the
   window's peak speed²"). Spectrum of the linearly detrended, Hann-windowed series: dominant
   period `P_s` (substeps) and its power fraction. `window_locked` = |P_s − T| ≤ 0.15 T, or
   P_s within 10 % of T/m for an integer m ≥ 2. Note that speed is |velocity|: a harmonic
   mode of period P shows in `s_t` at P/2.
2. **Volume.** `J = det F` per particle per sample; series of mean J and of RMS(J − per-
   particle running mean); the detrended mean-J peak-to-peak; `corr_J_speed` = the larger in
   magnitude of corr(J̃, s) and corr(|J̃|, s) at the sample instants (J̃ = detrended mean J,
   s = mean of the two adjacent kept steps). The |J̃| form is the energy-exchange signature:
   a volumetric spring is fastest when J crosses its trend, so |J̃| and speed anticorrelate.
   F-free cross-check: median 8-NN spacing³ on a fixed random subset (≤ 4000 particles).
3. **Stiffness.** `c = sqrt((λ+2μ)/ρ)`, `ρ = N·m/V`, `V` = occupied CIC nodes × dx³ on the
   MPM grid from the source cloud (over-counts by one surface layer; `V_bbox` reported);
   `CFL = c·dt/dx`; `L` = source bbox diagonal; `τ_e = 2L/c` (also in substeps).
   `stiffness_ringing` = P_s within 25 % of τ_e/dt or τ_e/(2dt) AND not window_locked.
   `window_over_tau = T·dt/τ_e`; a fractional part near ½ means the window ends
   mid-oscillation.
4. **Gradient / control (history).** Fraction of accepted commits with `reversal_cos < −0.2`;
   corr(Δλ, Δd_vol) (antiphase ⇒ negative); `dfc_absmax` median/max/last and the fraction
   of zero-control windows; sign-flip rate of Δ`move`; the kin-at-end fraction from (1).
5. **Visibility (docs/oscillation.md Addendum 7).** Over the last 40 accepted commit end
   states: per-particle peak-to-peak excursion about the linear drift, in target median
   NN spacings (sp), and per-commit increments; fractions above 0.5 sp.

## Decision rules (pre-registered; thresholds in `RULES`, never tuned on the run)

- VISIBLE if > 1 % of particles have a drift-removed tail excursion > 0.5 sp, or > 1 % of
  particle-commits move > 0.5 sp per accepted commit; else INVISIBLE (drivers are still
  reported, tagged sub-spacing — the Addendum 7 reopening rule).
- Driver C (control/window) if `window_locked` AND (median stop-and-go > 0.5, or median
  boundary jump ratio > 2 or < 0.5, **or median intra-window speed modulation
  max_t s_t / min_t s_t > 2**). The modulation clause was added on 2026-09-15 after the
  first hyde06 archives: every window showed speed 0.47 → 0.10 → 0.45 with a turning
  point mid-window and continuity across the boundary (sag 0, jump 0.93, modulation 4.3,
  95 % of the speed power at period T) — a control limit cycle that the sprint-then-brake
  form cannot see.
- Driver B (stiffness) if `stiffness_ringing` AND CFL > 0.3. CFL > 0.5 alone flags
  "CFL violation" even without ringing.
- Driver A (volume) if detrended mean-J peak-to-peak > 0.02 AND |corr_J_speed| > 0.5.
- Multiple drivers can be true; all are reported with their evidence numbers. None ⇒
  "no identifiable driver".

Output: markdown summary on stdout (every section header carries dt, dx, T, smoothing,
drag, E, ν and their sources) and, with `--out`, a JSON report holding every number, the
speed series, the window ranges, the J series and the rule thresholds. `--png` draws the
speed series with window boundaries, the spectrum with T / τ_e / τ_e/2 marked, the J and
NN-proxy series, and the excursion histogram.

## What each driver implies for the pipeline

**C — control/window.** The window objective (docs/method.md §5) ends every window with
the TERMINAL kinetic term `w_kin·mean|v_T|²`, and each window's control starts from a warm
start or, after a rejection, from zero (runner.py cold restart). The rollout therefore
accelerates from rest and brakes to rest T substeps later — stop-and-go at period T,
sharpened when consecutive commits reverse (`reversal_cos < −0.2`, drivers #5/#6 in
docs/oscillation.md). Remedies to TEST, not adopt: a RUNNING kinetic term
`Σ_t mean|v_t|²` over the whole window in place of the terminal one, and temporal
continuity of `dFc` across windows (penalise `dFc[0] − dFc_prev[T−1]` the way the
first-difference cost already couples steps inside a window).

**B — stiffness.** The explicit MLS-MPM step needs `CFL = c·dt/dx` well below one; both
Lamé constants scale with E, so reduce E or dt, or increase dx by the same relation. If the
elastic period `τ_e = 2L/c` is comparable to the window length `T·dt`, every window ends
mid-oscillation (ratio near k + ½) and the terminal-rest penalty fights the elastic mode
instead of the shape; choose `T·dt` an integer multiple of τ_e or damp the mode with the
objective viscosity `eta_sym` (docs/method.md eq. (7)) — measured at the stated dt, dx.

**A — volume.** The fixed-corotated energy `ψ = μ Σ(σ_i−1)² + λ/2 (J−1)²` (docs/method.md
§2, eq. (2)) contains a volumetric spring. Isochoric assimilation (`assim_iso`, §8) keeps
the plastic J at 1, so every commanded dilation stays elastic and permanently loaded (the
`w_jvol` comment in pipeline/config.py: the permanent volumetric spring of isochoric
assimilation arms inversions), and the `w_jvol` prior `mean (J−1)·log J` pulls the terminal
J back to 1 each window. A mean-J breathing at commit cadence whose speed anticorrelates
with |J̃| is that spring exchanging energy with the kinetic term; the levers are `w_jvol`,
`assim_iso` (let dilation be assimilated) and ν (the λ/μ ratio).

**D — none.** Addendum 7 of docs/oscillation.md closed the tail zigzag as sub-spacing
(median 0.16 sp, p99 0.42 sp on v3 40k). The reopening rule is the VISIBLE test above;
below it, drivers are informational and no mechanism is warranted — v4/v6 (reversal
brakes) were spent on an invisible signal and both cost real descent.

## Caveats

- Everything is first-order in dt; compare numbers only at the same dt, dx, smoothing.
- `V_cic` over-counts thin bodies (one node layer), so ρ, c and CFL are lower bounds on
  ρ and upper bounds on c; `V_bbox` brackets them from the other side.
- `s_t` is a mean of |v|, so `kin` (mean|v|²) ≥ `s_end²`; the 5 % rule is approximate.
- The spectrum needs ≥ 2 cycles in the record; periods longer than half the delivered
  trajectory are unresolved and reported as "n/a".


## Measured on hyde06 (2026-09-15) — what the rules found and what they missed

Six archives (20k, T=20, dt=1/240, dx=0.5): every arm VISIBLE with driver C after the
modulation clause was added (intra-window speed max/min 2.7–4.1, power at period T
0.81–0.95); the sprint-then-brake form alone had returned "no identifiable driver" (sag
0, jump 0.93) because the turning point sits mid-window and the boundary is continuous.
Remedy ladder (docs/render_controls_physics.md §9): warm start (control continuity) does
not change the cycle (power 0.90) — the cause is the per-window terminal-only objective,
not the cold start; the velocity-VARIANCE term `w_kin_var` is the mechanism-matched lever
(50: power 0.95 → 0.58, visible 9.6 → 2.5 %, shape unchanged). Paragraph C above should
be read with this: "running kinetic" attenuates, "variance of v over the window"
targets the reversal itself.


## REFUTE-2 corrections (2026-09-15, evening)

- **Label.** The window-locked driver is now `C_window`: the rule reads the speed series
  only, so it can say the cycle is locked to the optimisation window, not what inside the
  window causes it. Attribution comes from the discriminators (batch j): a window-locked
  cycle follows T when T is changed (10 / 40); the F-smoothing time dt/(1−s) = 22.2 steps
  and the elastic harmonics τ_e/m do not; `--assim 0` excludes the commit-time assimilation
  reset. The lock band is now ±6 % of T (22.2, 21.9 and 18.8 fall outside; the measured
  periods 20.00–20.03 stay inside).
- **Phase.** The measured intra-window profile is the OPPOSITE of "accelerate from rest,
  brake to rest": the speed is highest at the window boundary (step 19 → step 0
  continuous), falls to its minimum near mid-window (step 9–11, 0.07–0.10 wu/s), and rises
  again — a velocity reversal inside the window with continuity across the boundary.
- **Statistics.** `power_frac` depends on its denominator convention by 3–5× (both are now
  reported: `power_frac` excludes the two lowest bins, `power_frac_all` keeps every non-DC
  bin); the `modulation` clause was added after the first archives (post hoc, kept with
  that caveat). The pre-registered statistic for the next round is the convention-free
  **tortuosity** = path / net displacement per window (baseline 2.8, `w_kin_var 200` 1.09;
  threshold 1.5, gated on visibility). Driver A also fires on the per-particle RMS of J
  (threshold 0.05), not only the population mean.
