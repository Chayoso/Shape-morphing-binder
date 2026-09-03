# Probe: Sobolev preconditioning vs F-smoothing — render work share in real windows (2026-09-02)

Script: `scripts/probes/sobolev_precond.py` (new; nothing else edited). Machine: hyde06 GPU 0,
`/tmp/pm31` = branch `v3-grid-gs` @ `087a8c8` (md5-verified for every module the probe imports;
`traj.py` differs by line endings only). Raw results: `/tmp/pm31/probe_out/sobolev_precond/*.json`
(26 runs, ~400 s GPU wall total, 1.53 GB peak process footprint).

## Question

`docs/root_analysis.md` §2 claims the render covector's unique (high-frequency) content is
attenuated by the MPM adjoint (cubic B-spline transfers + F-smoothing s=0.955) before it reaches
dFc, and cites the work telemetry: render-attributable linearized work in accepted steps is
~0.02% of the physics work. Two candidate remedies were measured in REAL optimisation windows:

1. **Sobolev / screened-Poisson preconditioning of the render covector before the adjoint
   pullback** — the existing `cfg.render_gs_iters` / `cfg.render_gs_kappa` path
   (`optimizer.py`: `smooth_particle_field` on the x-covector, then
   `autograd.grad(state[0], leaves, grad_outputs=gxs)`).
2. **Reducing the dynamics' own low-pass** — `MPMParams.smoothing`.

Headline criterion (pre-registered): does either raise the render work share by >= 10x from
the ~0.02% baseline WITHOUT hurting d_vol descent or stability?

**Answer: no, neither. Preconditioning leaves the share at 1.0x (0.0073–0.0122% vs 0.0073–0.011%
baseline band, 12 vs 4 runs) and halves the λ-weighted share. F-smoothing cannot be reduced:
`s` is the retention weight on the OLD F, so the requested `s=1.0` freezes F (no elastic memory)
and the actual "less smoothing" direction (`s<0.955`) collapses J to the 1e-4 inversion floor
within 1–2 commits. The share metric itself turns out to be a loss-scale ratio, not an
attenuation signature (see §5).**

## 1. Setup

Config = `scripts/dress_bench.py` flagship minus the gauss channel: `PipelineConfig(T=20, iters=8,
loss_res=64)`, `lambda_auto=0.5, w_kin=5, w_dt=0.2, w_nn=0.2, w_jvol=50, assim_iso=True,
work_telemetry=True`, `render_surface_only=True, surface_grad_frac=0.5` (binarised surface mask
passed as `surface_w`, 51.2% of particles), `use_gauss_loss=False`. N=20000, isosphere (seed 1)
-> bunny (seed 2), 18 silhouette views at res 64, `vol0` computed once at the source as the
runner does. All conditions share one `TargetPack`.

Per condition: 6 consecutive commits, each = one `optimize_window` + runner-style promotion
(grid clip -> `condition_F(clamp=False)` -> `assimilate_elastic(eta=0.5, isochoric=True)`);
a window with no surviving accepted step is a null commit (state held, no assimilation),
exactly as `runner.py`. Every condition was run twice (`[r2]` = second pass, same seeds; the
CUDA-atomic rollouts are not bit-identical, so the two passes are a run-to-run noise estimate),
and the baseline four times.

Metrics (per commit, from the returned `stats`/`hist`):

* `share_last` = render_work / (render_work + phys_work) from `stats` — the LAST accepted
  iteration, i.e. exactly the number `runner.py` logs as the headline P-render metric.
  `render_work = -<∂D_render/∂x_T, Δx_T>` (raw, surface-masked covector; λ NOT applied),
  `phys_work = -<∂L_phys/∂(x,F,v), Δ(x,F,v)>`.
* `share_sum` = the same ratio with both works summed over every accepted iteration of the window.
* `lam-weighted share` = λ·render_work / (λ·render_work + phys_work) — the render term's actual
  fraction of the composite objective's first-order decrease.
* `g_cos` (= `g_raw_cos`, PCGrad off), `g_share`, `|g_rend|`, `|g_phys|` from iteration 0.
* d_vol / d_render evaluated on the PROMOTED x (`losses.volumetric.d_vol`,
  `render_loss.d_render`), Jmin of the promoted F and over the whole committed trajectory,
  and every guard counter the runner keeps (clamped, nan_x, nan_state, F_reset, F_flip,
  F_invert_steps).
* Added diagnostics: `|gx|` = norm of the raw covector at the window-start state;
  `pullback/raw` = `|g_rend|` (pulled back to dFc space) / `|gx|`; `precond cos` = cosine
  between the raw and the Sobolev-smoothed covector; `realization` = (first accepted candidate's
  d_render − last accepted candidate's) / Σ render_work over the same steps.

Conditions:

| name | render_gs_iters / kappa | smoothing s | note |
|---|---|---|---|
| base, base_rep | 0 / – | 0.955 | production; 4 runs total |
| gs2_k1, gs2_k4, gs2_k16, gs6_k4, gs6_k16 | 2 or 6 / 1, 4, 16 | 0.955 | screened diffusion on the 64³ loss grid (cell 0.5 wu = MPM dx): `(I + κ(I − avg6)) u = ĝ`, screening length √(κ/6) cells = 0.41 / 0.82 / 1.63, red-black GS sweeps each propagate ~1 cell |
| gs20_k4 | 20 / 4 | 0.955 | `scripts/pipeline_run.py --render_gs_iters` default |
| s0.98, s1.0 | 0 | 0.98, 1.0 | the requested sweep (MORE smoothing — see §2) |
| s0.8, s0.5, s0.0 | 0 | 0.8, 0.5, 0.0 | the actual "less low-pass" direction |

## 2. What `s` is (premise correction)

`physmorph/mpm/kernels.py::k_update` (eq (9)):

```
F_out[p] = (1.0 - s) * F_new[p] + s * F_in[p]      # blend new with OLD F
```

with `F_new = (I + dt·C)(F + dFc)` from G2P. `s` is the per-step retention weight on the OLD F —
a temporal EMA with time constant 1/(1−s): 22 steps at 0.955 (longer than the T=20 window),
50 at 0.98, infinite at 1.0. So:

* `s = 1.0` is NOT "no smoothing": F never integrates. With `F0=None` the total F stays the
  identity forever, `F_e = (I + dFc) Fp⁻¹`, stress has no memory of deformation, `det F ≡ 1`
  (the `w_jvol` prior and the isochoric assimilation become no-ops). Numerically stable, but
  the material has no elasticity beyond the control — a degenerate regime, not a remedy.
* `s = 0.0` is "no smoothing": each step commits the full `(I + dt·C)(F + dFc)`.
* The stress path sees `F + dFc` directly at every step regardless of `s`; what `s` limits is
  how much of `dFc` (and of the velocity-gradient increment) persists in F: 4.5% per step at
  0.955, 100% at 0.0.

Both the requested values and the meaningful direction were run.

## 3. Headline table (pooled over replicates)

`share_last` pooled = median over all live commits of the group; per-run = range of the six-commit
medians. Baseline band is the reference for "unchanged".

| group (runs) | share_last per-run med | pooled share_last med (p10–p90) | λ-weighted share med | g_cos med | d_vol min / end over 6 commits | d_render min / end | Jmin_traj min | non-finite | Finv |
|---|---|---|---|---|---|---|---|---|---|
| baseline (4) | 0.0073–0.0110% | **0.0076%** (0.0056–0.0170%) | **23.0%** | 0.68–0.75 | 199–206 / 350–465 | 0.0289–0.0309 / 0.051–0.069 | 0.40–0.44 | 0 | 0 |
| precond, all 6 points (12) | 0.0073–0.0122% | **0.0082%** (0.0049–0.0167%) | **12.2%** | 0.66–0.78 | 191–220 / 207–471 | 0.0284–0.0325 / 0.032–0.074 | 0.40–0.49 | 0 | 0 |
| s=0.98 (2) | 0.0118–0.0128% | 0.0128% (0.0064–0.0235%) | 35.3% | 0.72–0.74 | 200–202 / 241–263 | 0.0289–0.0291 / 0.036–0.043 | 0.64–0.74 | 0 | 0 |
| s=1.0, F frozen (2) | 0.0079–0.0082% | 0.0082% (0.0044–0.0200%) | 21.7% | 0.72 | 155–157 / 155–157 | 0.0238–0.0239 / 0.024–0.026 | 1 (by construction) | 0 | 0 |
| s=0.8 (2) | 0.0073–0.0074% | 0.0074% (0.0058–0.0093%) | 22.4% | 0.71–0.74 | 418–433 / 455–566 | 0.0342–0.0361 / 0.059–0.063 | **0.0044–0.0099** | 0 | 0 |
| s=0.5 (2) | 0.0073–0.0084% | 0.0074% (0.0064–0.0114%) | 21.7% | 0.70–0.72 | 549–550 / 904–1515 | 0.0475–0.0479 / **0.133–0.207** (> start 0.131) | **6e-4–8e-4** | 0 | 0 |
| s=0.0 (2) | 0.0083–0.0089% | 0.0083% (0.0079–0.0103%) | 24.2% | 0.67 | 773–777 / 1196–1246 | 0.0672–0.0676 / **0.153–0.158** (> start) | **1.0e-4 (= `_state_ok` floor)** | 0 | 0 |

Source state: d_vol 1651, d_render 0.1313. Required for a positive verdict: >= 0.076% pooled
(10x). Largest single-run median anywhere: 0.0128% (s=0.98), 1.7x the pooled baseline and inside
its p90.

## 4. Full per-run tables (script `--report` output)

| condition | gs/kappa | s | share_last med | rw/pw med | share_sum med | lam-weighted share med | g_cos med | g_share med | d_render drop (rel) | d_vol drop (rel) | d_vol min | acc/rej | null (replay) | Jmin_traj min | Finv | nonfinite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0/4 | 0.955 | 0.00981% | 9.81e-05 | 0.00581% | 30.9% | 0.749 | 0.31 | 0.06184 (47.1%) | 1284 (77.8%) | 199.9 | 40/0 | 1 (1) | 0.442 | 0 | 0 |
| base [r2] | 0/4 | 0.955 | 0.00763% | 7.63e-05 | 0.00524% | 22.9% | 0.699 | 0.317 | 0.0802 (61.1%) | 1259 (76.3%) | 199.1 | 48/0 | 0 (0) | 0.4 | 0 | 0 |
| base_rep | 0/4 | 0.955 | 0.011% | 0.00011 | 0.00532% | 29.7% | 0.68 | 0.317 | 0.0713 (54.3%) | 1186 (71.8%) | 206.2 | 48/0 | 0 (0) | 0.42 | 0 | 0 |
| base_rep [r2] | 0/4 | 0.955 | 0.00734% | 7.34e-05 | 0.00587% | 21.1% | 0.748 | 0.314 | 0.06776 (51.6%) | 1300 (78.8%) | 200.3 | 40/0 | 1 (1) | 0.434 | 0 | 0 |
| gs2_k1 | 2/1 | 0.955 | 0.00862% | 8.62e-05 | 0.00548% | 14.3% | 0.712 | 0.34 | 0.07096 (54.1%) | 1292 (78.3%) | 220 | 40/0 | 1 (1) | 0.461 | 0 | 0 |
| gs2_k1 [r2] | 2/1 | 0.955 | 0.0106% | 0.000106 | 0.00573% | 17% | 0.769 | 0.308 | 0.06577 (50.1%) | 1267 (76.7%) | 218.9 | 40/0 | 1 (1) | 0.465 | 0 | 0 |
| gs2_k4 | 2/4 | 0.955 | 0.00757% | 7.57e-05 | 0.00599% | 9.7% | 0.76 | 0.341 | 0.08223 (62.6%) | 1271 (77%) | 211.7 | 48/0 | 0 (0) | 0.43 | 0 | 0 |
| gs2_k4 [r2] | 2/4 | 0.955 | 0.00802% | 8.02e-05 | 0.00528% | 11.5% | 0.746 | 0.31 | 0.06457 (49.2%) | 1264 (76.6%) | 220.1 | 40/0 | 1 (1) | 0.436 | 0 | 0 |
| gs2_k16 | 2/16 | 0.955 | 0.00971% | 9.71e-05 | 0.00698% | 13.4% | 0.663 | 0.336 | 0.09603 (73.2%) | 1434 (86.9%) | 216.6 | 32/0 | 2 (2) | 0.442 | 0 | 0 |
| gs2_k16 [r2] | 2/16 | 0.955 | 0.00984% | 9.84e-05 | 0.00581% | 16.3% | 0.779 | 0.316 | 0.06908 (52.6%) | 1307 (79.2%) | 199.2 | 40/0 | 1 (1) | 0.448 | 0 | 0 |
| gs6_k4 | 6/4 | 0.955 | 0.00731% | 7.31e-05 | 0.00577% | 9.68% | 0.766 | 0.339 | 0.08277 (63%) | 1279 (77.4%) | 207.4 | 48/0 | 0 (0) | 0.446 | 0 | 0 |
| gs6_k4 [r2] | 6/4 | 0.955 | 0.0076% | 7.6e-05 | 0.00483% | 11.2% | 0.705 | 0.333 | 0.0731 (55.7%) | 1180 (71.4%) | 215.3 | 48/0 | 0 (0) | 0.399 | 0 | 0 |
| gs6_k16 | 6/16 | 0.955 | 0.00893% | 8.94e-05 | 0.00537% | 13% | 0.712 | 0.34 | 0.07864 (59.9%) | 1232 (74.6%) | 213.6 | 48/0 | 0 (0) | 0.456 | 0 | 0 |
| gs6_k16 [r2] | 6/16 | 0.955 | 0.00999% | 9.99e-05 | 0.00606% | 13.9% | 0.697 | 0.344 | 0.05771 (44%) | 1220 (73.9%) | 191.2 | 40/0 | 1 (1) | 0.489 | 0 | 0 |
| gs20_k4 | 20/4 | 0.955 | 0.0122% | 0.000123 | 0.00727% | 16.2% | 0.766 | 0.34 | 0.09928 (75.6%) | 1444 (87.5%) | 207 | 32/0 | 2 (2) | 0.481 | 0 | 0 |
| gs20_k4 [r2] | 20/4 | 0.955 | 0.00831% | 8.31e-05 | 0.00584% | 11.5% | 0.759 | 0.343 | 0.08138 (62%) | 1306 (79.1%) | 211.9 | 48/0 | 0 (0) | 0.443 | 0 | 0 |
| s0.98 | 0/4 | 0.98 | 0.0128% | 0.000128 | 0.00608% | 35.3% | 0.739 | 0.313 | 0.08829 (67.3%) | 1388 (84.1%) | 200.4 | 40/0 | 1 (1) | 0.741 | 0 | 0 |
| s0.98 [r2] | 0/4 | 0.98 | 0.0118% | 0.000118 | 0.00567% | 32.9% | 0.715 | 0.315 | 0.09564 (72.9%) | 1410 (85.4%) | 201.7 | 48/0 | 0 (0) | 0.642 | 0 | 0 |
| s1.0 | 0/4 | 1 | 0.00815% | 8.15e-05 | 0.0064% | 21.7% | 0.722 | 0.318 | 0.1074 (81.8%) | 1496 (90.6%) | 155.2 | 48/0 | 0 (0) | 1 | 0 | 0 |
| s1.0 [r2] | 0/4 | 1 | 0.00792% | 7.92e-05 | 0.00655% | 21.7% | 0.721 | 0.32 | 0.1056 (80.5%) | 1494 (90.5%) | 157.4 | 48/0 | 0 (0) | 1 | 0 | 0 |
| s0.8 | 0/4 | 0.8 | 0.00733% | 7.33e-05 | 0.00749% | 22.6% | 0.744 | 0.327 | 0.07269 (55.4%) | 1085 (65.7%) | 418.2 | 48/0 | 0 (0) | 0.00991 | 0 | 0 |
| s0.8 [r2] | 0/4 | 0.8 | 0.00739% | 7.39e-05 | 0.00668% | 22.1% | 0.708 | 0.326 | 0.06817 (51.9%) | 1196 (72.4%) | 432.5 | 48/0 | 0 (0) | 0.00438 | 0 | 0 |
| s0.5 | 0/4 | 0.5 | 0.0073% | 7.3e-05 | 0.00721% | 21.6% | 0.699 | 0.327 | -0.07567 (-57.6%) | 135.7 (8.22%) | 548.8 | 29/2 | 2 (2) | 0.000604 | 0 | 0 |
| s0.5 [r2] | 0/4 | 0.5 | 0.00836% | 8.36e-05 | 0.00792% | 23.5% | 0.715 | 0.329 | -0.001228 (-0.936%) | 746.7 (45.2%) | 550.4 | 32/1 | 2 (2) | 0.00076 | 0 | 0 |
| s0.0 | 0/4 | 0 | 0.00832% | 8.32e-05 | 0.00971% | 24.2% | 0.669 | 0.357 | -0.02198 (-16.7%) | 455.1 (27.6%) | 777 | 38/1 | 1 (1) | 0.0001 | 0 | 0 |
| s0.0 [r2] | 0/4 | 0 | 0.00892% | 8.92e-05 | 0.00962% | 24.2% | 0.669 | 0.36 | -0.02624 (-20%) | 404.8 (24.5%) | 773 | 44/3 | 0 (0) | 0.0001 | 0 | 0 |

"drop" = start − value after commit 6 (negative = worse than the source). `null (replay)`:
null commits, of which how many had accepted steps that the commit-rollout replay check
discarded (§7).

| condition | rw med | pw med | rw_x med | rw_F med | \|gx\| raw med | \|g_rend\| pullback med | pullback/raw | precond cos | realization med | \|g_phys\| med | lambda end | d_vol end | d_render end | Jmin end | move/commit | s/commit | GPU MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.00469 | 61.1 | 0.00469 | 0 | - | 0.00301 | - | - | - | 27.2 | 4.57e+03 | 366.5 | 0.06944 | 0.443 | 0.17 | 1.78 | 1530 |
| base [r2] | 0.00411 | 63.8 | 0.00411 | 0 | 0.0118 | 0.00286 | 0.248 | - | 0.774 | 26.1 | 4.47e+03 | 391.6 | 0.05108 | 0.4 | 0.197 | 1.89 | 1530 |
| base_rep | 0.00474 | 64.3 | 0.00474 | 0 | - | 0.00298 | - | - | - | 26 | 4.14e+03 | 465.1 | 0.05997 | 0.42 | 0.196 | 1.66 | 1530 |
| base_rep [r2] | 0.0037 | 59.7 | 0.0037 | 0 | 0.0121 | 0.00316 | 0.231 | - | 0.707 | 26.7 | 4.33e+03 | 350.5 | 0.06351 | 0.435 | 0.17 | 2 | 1530 |
| gs2_k1 | 0.00512 | 68.5 | 0.00512 | 0 | - | 0.00699 | - | - | - | 24.3 | 1.88e+03 | 358.6 | 0.06032 | - | 0.178 | 1.85 | 1508 |
| gs2_k1 [r2] | 0.00524 | 69.2 | 0.00524 | 0 | 0.0115 | 0.0073 | 0.63 | 0.368 | 0.738 | 27.1 | 2.02e+03 | 384.4 | 0.06551 | 0.465 | 0.179 | 2.35 | 1514 |
| gs2_k4 | 0.00302 | 50.9 | 0.00302 | 0 | - | 0.00817 | - | - | - | 26.3 | 1.76e+03 | 380.1 | 0.04905 | 0.43 | 0.207 | 1.75 | 1528 |
| gs2_k4 [r2] | 0.00397 | 66.2 | 0.00397 | 0 | 0.0114 | 0.00758 | 0.617 | 0.307 | 0.724 | 27.4 | 2.22e+03 | 387.1 | 0.0667 | 0.522 | 0.184 | 2.41 | 1534 |
| gs2_k16 | 0.00553 | 66.5 | 0.00553 | 0 | - | 0.00697 | - | - | - | 24.1 | 1.91e+03 | 216.6 | 0.03525 | - | 0.148 | 1.63 | 1528 |
| gs2_k16 [r2] | 0.00585 | 71.5 | 0.00585 | 0 | 0.0117 | 0.00654 | 0.577 | 0.284 | 0.732 | 26.3 | 1.98e+03 | 343.8 | 0.06219 | 0.494 | 0.178 | 2.16 | 1514 |
| gs6_k4 | 0.00338 | 53.5 | 0.00338 | 0 | - | 0.00802 | - | - | - | 26.4 | 1.86e+03 | 372.5 | 0.04851 | 0.446 | 0.204 | 1.79 | 1528 |
| gs6_k4 [r2] | 0.00522 | 71.1 | 0.00522 | 0 | 0.0114 | 0.00707 | 0.606 | 0.319 | 0.802 | 24.1 | 1.93e+03 | 471.5 | 0.05818 | 0.399 | 0.211 | 2.08 | 1514 |
| gs6_k16 | 0.00424 | 59.1 | 0.00424 | 0 | - | 0.00699 | - | - | - | 23.9 | 1.96e+03 | 419.4 | 0.05263 | 0.456 | 0.218 | 2.02 | 1528 |
| gs6_k16 [r2] | 0.00433 | 43.3 | 0.00433 | 0 | 0.0117 | 0.00745 | 0.622 | 0.229 | 0.731 | 25.3 | 2.04e+03 | 431.2 | 0.07357 | 0.54 | 0.18 | 2.08 | 1534 |
| gs20_k4 | 0.00447 | 55.9 | 0.00447 | 0 | - | 0.00785 | - | - | - | 24.6 | 1.93e+03 | 207 | 0.032 | - | 0.14 | 2.8 | 1528 |
| gs20_k4 [r2] | 0.00453 | 53 | 0.00453 | 0 | 0.0119 | 0.00835 | 0.658 | 0.32 | 0.803 | 25.6 | 1.69e+03 | 344.9 | 0.0499 | 0.443 | 0.206 | 1.86 | 1534 |
| s0.98 | 0.00521 | 38.1 | 0.00521 | 0 | - | 0.0027 | - | - | - | 24.9 | 4.47e+03 | 262.6 | 0.04299 | 0.749 | 0.155 | 2.16 | 1530 |
| s0.98 [r2] | 0.00434 | 34.9 | 0.00434 | 0 | 0.0119 | 0.00269 | 0.217 | - | 0.766 | 23 | 4.31e+03 | 240.8 | 0.03563 | 0.642 | 0.176 | 1.82 | 1530 |
| s1.0 | 0.00166 | 32.9 | 0.00166 | 0 | - | 0.00251 | - | - | - | 20.9 | 4.36e+03 | 155.2 | 0.02391 | 1 | 0.155 | 2.32 | 1530 |
| s1.0 [r2] | 0.00181 | 33.8 | 0.00181 | 0 | 0.0108 | 0.0026 | 0.24 | - | 0.704 | 21 | 4.37e+03 | 157.4 | 0.02563 | 1 | 0.155 | 2 | 1530 |
| s0.8 | 4.71e-06 | 0.0752 | 4.71e-06 | 0 | - | 0.00509 | - | - | - | 40.2 | 3.74e+03 | 565.7 | 0.05858 | 0.148 | 0.254 | 3.37 | 1530 |
| s0.8 [r2] | 9.85e-06 | 0.119 | 9.85e-06 | 0 | 0.0117 | 0.0045 | 0.427 | - | 1.11 | 36 | 3.94e+03 | 455.1 | 0.0631 | 0.151 | 0.218 | 2.56 | 1530 |
| s0.5 | 6.85e-05 | 0.997 | 6.85e-05 | 0 | - | 0.00817 | - | - | - | 65.4 | 3.92e+03 | 1515 | 0.2069 | 0.0233 | 0.171 | 2.57 | 1530 |
| s0.5 [r2] | 0.00021 | 2.84 | 0.00021 | 0 | 0.00977 | 0.00855 | 0.875 | - | 1.09 | 68.3 | 3.95e+03 | 904.3 | 0.1325 | 0.36 | 0.179 | 2.41 | 1530 |
| s0.0 | 2.18e-08 | 0.000249 | 2.18e-08 | 0 | - | 0.0123 | - | - | - | 78.2 | 3.19e+03 | 1196 | 0.1533 | 0.00574 | 0.171 | 3.19 | 1530 |
| s0.0 [r2] | 5.47e-07 | 0.00631 | 5.47e-07 | 0 | 0.0103 | 0.0124 | 1.22 | - | 1.06 | 77.1 | 3.16e+03 | 1246 | 0.1575 | 0.00673 | 0.186 | 2.91 | 1530 |

`rw_F = 0` everywhere: the silhouette loss has no F-covector, so the `render_gs` path's known
omission of the F component (method.md §6) costs nothing in this config. `|gx|`, `precond cos`,
`realization` exist only for the second pass (added after the first).

Per-commit trajectories (second pass), `d_vol / d_render / Jmin_traj / acc:rej`, source
1651 / 0.131:

| run | c0 | c1 | c2 | c3 | c4 | c5 |
|---|---|---|---|---|---|---|
| base [r2] | 429/0.035/0.695/8:0 | 248/0.033/0.564/8:0 | 210/0.036/0.553/8:0 | 199/0.030/0.464/8:0 | 360/0.064/0.43/8:0 | 392/0.051/0.40/8:0 |
| base_rep [r2] | 429/0.035/0.695/8:0 | 249/0.033/0.564/8:0 | 249/0.033/–/NULL | 209/0.035/0.553/8:0 | 200/0.029/0.465/8:0 | 350/0.064/0.434/8:0 |
| gs2_k16 [r2] | 447/0.036/0.702/8:0 | 260/0.031/0.564/8:0 | 260/0.031/–/NULL | 224/0.041/0.57/8:0 | 199/0.031/0.448/8:0 | 344/0.062/0.46/8:0 |
| gs20_k4 [r2] | 445/0.035/0.703/8:0 | 262/0.032/0.584/8:0 | 218/0.039/0.591/8:0 | 212/0.032/0.475/8:0 | 392/0.071/0.485/8:0 | 345/0.050/0.443/8:0 |
| s0.98 [r2] | 459/0.037/0.856/8:0 | 267/0.034/0.776/8:0 | 202/0.034/0.759/8:0 | 204/0.029/0.741/8:0 | 264/0.043/0.746/8:0 | 241/0.036/0.642/8:0 |
| s1.0 [r2] | 491/0.040/1/8:0 | 302/0.036/1/8:0 | 192/0.029/1/8:0 | 198/0.024/1/8:0 | 173/0.027/1/8:0 | 157/0.026/1/8:0 |
| s0.8 [r2] | 433/0.036/0.177/8:0 | 517/0.087/0.036/8:0 | 630/0.093/0.0070/8:0 | 896/0.111/0.0044/8:0 | 1036/0.116/0.019/8:0 | 455/0.063/0.151/8:0 |
| s0.5 [r2] | 550/0.048/0.045/8:0 | 983/0.152/0.0057/8:0 | 983/0.152/–/0:1 NULL | 983/0.152/–/NULL | 1336/0.195/7.6e-4/8:0 | 904/0.133/0.054/8:0 |
| s0.0 [r2] | 773/0.068/0.0061/8:0 | 1157/0.151/1.0e-4/8:0 | 1452/0.198/1.0e-4/4:3 | 1408/0.189/1.1e-4/8:0 | 1225/0.159/1.0e-4/8:0 | 1246/0.158/1.0e-4/8:0 |

## 5. What the preconditioner did, and why the share did not move

* It acted. `precond cos` = 0.23–0.37: the smoothed covector is a substantially different
  direction from the raw one (κ and iters matter little in this range — every point lands
  between 0.23 and 0.37). Its pullback to dFc space survives the adjoint 2.5x better in norm:
  `pullback/raw` 0.58–0.66 vs 0.23–0.25 for the raw covector. That is a direct measurement of
  the low-pass in §2 of the root analysis: ~75% of the raw covector's norm dies in the adjoint,
  ~40% of the smoothed one's.
* The balancer saw the bigger `|g_rend|` and halved λ (4.1–4.6e3 -> 1.7–2.2e3); `g_share`
  (nominal, norm-based) rose from 0.31 to 0.34.
* The REALIZED render work per accepted step did not change: rw med 0.0030–0.0059 (precond)
  vs 0.0037–0.0047 (baseline). The extra norm that survives the pullback does no extra render
  work — it is the redundant low-frequency component, which is exactly the root analysis'
  prediction seen from the other side. Net effect on the objective: the λ-weighted share of the
  first-order decrease FELL from 21–31% to 9.7–17%, because λ halved with no gain in rw.
* d_vol / d_render descent, Jmin, null-commit rate: all inside the baseline band (12 runs).
* The linearized render model is realized inside a window: `realization` 0.70–0.80 in every
  s>=0.955 condition (e.g. base [r2] c0: d_render 0.084 -> 0.035 over the 8 accepted
  iterations; from the first accepted candidate on, 0.049 realized against Σ rw = 0.080
  predicted, 0.61). The covector is not "unrealizable"; it is small in absolute units.

**The share metric is a loss-scale ratio, not an attenuation signature.** Per accepted step,
relative progress of the two channels is the same order:

| run | render_work / d_render (med, range) | phys_work / d_vol (med, range) |
|---|---|---|
| base | 0.111 (0.040–0.292) | 0.161 (0.027–0.329) |
| base [r2] | 0.097 (0.071–0.299) | 0.196 (0.083–0.318) |
| base_rep | 0.109 (0.037–0.318) | 0.180 (0.032–0.328) |
| base_rep [r2] | 0.111 (0.055–0.257) | 0.162 (0.075–0.328) |
| gs20_k4 [r2] | 0.117 (0.019–0.285) | 0.161 (0.019–0.285) |
| s0.98 [r2] | 0.122 (0.043–0.235) | 0.139 (0.045–0.201) |

Each accepted step takes ~10% off d_render and ~16% off d_vol to first order. The raw ratio
`render_work / phys_work ~ 1e-4` is (0.10 x 0.035) / (0.16 x 250): the d_vol/d_render VALUE
ratio (~1e4, an artefact of d_vol being a summed log-mass-ratio over a 64³ grid and d_render a
per-pixel mean), not a 1e4 attenuation. The same scale ratio is what λ ~ 4e3 compensates. The
render term's actual fraction of the objective decrease is the λ-weighted share, 21–31% at
baseline. A preconditioner that could "raise the raw share 10x" would have to make the render
term decrease 10x faster per step in RELATIVE terms — no reshaping of the direction can do that
against a line search that calibrates the step on the composite.

## 6. F-smoothing sweep

* `s = 0.98` (more smoothing): share 0.012–0.013% (1.2–1.7x pooled baseline, inside its p90),
  d_vol descent slightly better (84–85% vs 72–79%), Jmin_traj healthier (0.64–0.74 vs 0.40–0.44),
  d_render end 0.036–0.043 vs 0.051–0.069. A mild positive on stability, not on the share.
* `s = 1.0` (F frozen): best d_vol (90.5%) and d_render (81%) descent of the whole probe, and
  the only condition whose per-commit trajectory is monotone (no spring-back at c4–c5). But
  `det F ≡ 1`, `w_jvol` and the isochoric assimilation are no-ops, and the material has no
  elastic memory of deformation — it is a drag-only medium moved by the control. That is why it
  descends: nothing resists. It is not a valid remedy; it is a measurement of how much of the
  d_vol residual the elasticity itself is holding.
* `s < 0.955` (the actual low-pass reduction): share unchanged (0.0073–0.0089%), and the
  dynamics collapse toward inversion within 1–2 commits. Jmin_traj: 0.004–0.010 (s=0.8),
  6e-4–8e-4 (s=0.5), 1.0e-4 (s=0.0) — the last is precisely the `_state_ok` floor `jt > 1e-4`,
  i.e. every accepted candidate sits on the acceptance guard. phys_work collapses — median
  60 at baseline vs 0.08–0.12 (s=0.8), 1.0–2.8 (s=0.5), 2.5e-4–6.3e-3 (s=0.0), i.e. 20x to
  2e5x smaller (the line search shrinks the step to keep det above the floor), d_vol regresses
  (s=0.5: 8–45% net; s=0.0: 25–28%), and d_render ENDS ABOVE the source value for s<=0.5.
  Zero non-finite events, zero `F_invert_steps` (det<=0 never committed): the guards hold, but
  the trajectory is pinned at the inversion floor. Mechanism: with s=0.955 each `dFc_t` persists
  in F at 4.5% per step, and the optimizer's step scale (adaptive α on `target_norm`, `dfc_clip`
  off) is calibrated to that; at s=0 the same controls accumulate 22x more deformation per step.
  `s` is therefore load-bearing for the control parametrisation, not a free low-pass knob —
  reducing it requires re-scaling the control (or a different control) first.

## 7. Side findings

* **Replay-discard null commits at ~10%.** 13 of 26 runs contain a null commit whose window
  HAD accepted steps (`accepted = 0` in the returned stats yet the work telemetry is present and
  `grad_converged` is false): `optimizer.py`'s commit-rollout check (`replay_bad`: E_final >
  E_accept + `ls_noise_rel`·|E|, tol 1e-7) discarded the window. The det criterion is ruled out
  for the s>=0.955 runs (adjacent commits' Jmin_traj ~ 0.55). 12 of the 120 commits in the
  s>=0.955 runs (17 of 156 overall) went this way, the BASELINE included (2 of 4 baseline
  runs, commit 2 both times). 1e-7 relative is at or below CUDA-atomic replay noise for a
  rollout of this size; a production run pays a wasted window every ~10 commits for it.
* **Between-commit regression of the render term.** In every s>=0.955 run d_render descends
  inside the window (0.084 -> 0.035 at c0) and then jumps back at c4–c5 (0.030 -> 0.064) with
  d_vol following (199 -> 360). The window's optimum is not a fixed point of the promotion
  (momentum carry v0/C0 + assimilation + free rollout). Same in all preconditioned runs; absent
  only for s=1.0 (no elasticity). This is root_analysis §3's oscillation seen at commit
  resolution, and it dominates the 6-commit "cumulative drop" numbers — the min-over-commits
  columns are the stable comparison.

## 8. Verdict

| remedy | share vs baseline | d_vol descent | stability | verdict |
|---|---|---|---|---|
| Sobolev precond (6 points, 12 runs) | 1.0x (0.0082% vs 0.0076% pooled; 0.0073–0.0122% vs 0.0073–0.011% per run); λ-weighted share 12% vs 23% | unchanged (min 191–220 vs 199–206) | unchanged (Jmin_traj 0.40–0.49; 0 events) | **no effect on the headline; slightly negative on the weighted share** |
| s = 0.98 | 1.7x (inside baseline p90) | slightly better (84–85%) | better (Jmin_traj 0.64–0.74) | **not a remedy for the share; harmless** |
| s = 1.0 | 1.1x | best (90.5%) | trivially (det ≡ 1) | **invalid regime: F frozen, no elasticity** |
| s = 0.8 / 0.5 / 0.0 | 1.0x | regresses (66–72% / 8–45% / 25–28%), d_render above source for s<=0.5 | collapses to the 1e-4 det floor in 1–2 commits (no NaN) | **falsified: s is load-bearing for control authority** |

Neither remedy raises the render work share by >= 10x; nothing reached 2x. The reason is
structural: the 0.01% number is the d_vol/d_render scale ratio, and both channels already
progress at ~10–16% per accepted step in relative terms. Preconditioning changes which
component of the covector survives the adjoint (2.5x more norm) but that component does no
additional render work — consistent with root_analysis §2 (the surviving part is the
redundant one) — while the norm-based balancer reacts to the surviving norm and de-weights the
channel. The measurement supports the root analysis' second prescription (route fine
corrections around the low-pass) over any in-channel preconditioning, and adds that the
F-smoothing is not removable in the current control parametrisation.

Independent cross-check: `docs/probes/transfer_function.md` (committed in parallel, `a8ead52`,
synthetic-covector transfer gains) reaches the same three conclusions from the other side —
the 1000x is loss scale, F-smoothing is temporal not spatial, and `render_gs` pre-smoothing has
a x1.6–1.9 ceiling. This probe's real-window measurement puts the realized ceiling at 1.0x on
the raw share and 0.5x on the λ-weighted share.

## 9. Reproduction

```
cd /tmp/pm31 && OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 \
  /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python scripts/probes/sobolev_precond.py [--tag r3]
# tables only, from the saved JSONs:
  ... scripts/probes/sobolev_precond.py --report
```

~15 s per condition (6 commits x 1.7–3 s), 1.5 GB GPU. Results dir `probe_out/sobolev_precond/`
(deliberately off `/data`).
