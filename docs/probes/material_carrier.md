# Probe — is the material leaf a carrier the render channel can use? (Tier M)

**Date** 2026-09-02 · **Branch** `v3-grid-gs` · **Script** `scripts/probes/material_carrier.py`
· **Host** hyde06 GPU 2, N=20000, T=20 · **GPU peak** 2410 MiB, ~9 min total

**Question.** `docs/local_global_design.md` Appendix A defers a carrier ("Tier M") that writes
per-particle base log-Lamé multipliers so the render channel steers the *response operator*
instead of the control. Its load-bearing claim is A.1:

> `Material` … changes the **response operator** … and no cheap exact annulling control
> exists — to reproduce a trajectory under modified stiffness, `dFc` would have to solve
> `P_new(Fe') = P_old(Fe)` pointwise in time while `Fe'` simultaneously drives the kinematic
> update (9); overdetermined, no null direction.

This probe measures that claim, plus the premise it inherits from `docs/root_analysis.md` §2
(that the material leaf would carry render content the adjoint does not attenuate).

**Verdict: NO-GO for Tier M.** Every measurement points the wrong way. Detail below.

---

## Setup

Flagship `dress_bench.py` configuration with the gauss channel OFF (probe directive), so the
render covector measured here is the pure silhouette `d_render` — the same scalar every gate
consumes after the B1 split.

| | |
|---|---|
| source → target | `assets/isosphere.obj` → `assets/bunny.obj`, 20000 particles each |
| MPM | `MPMParams()` defaults: dx 0.5, dt 1/240, drag 0.9, F-smoothing 0.955, grid 64³ from −16 |
| cfg | `T=20, iters=8, animations=8, loss_res=64`; `lambda_auto=0.5, w_kin=5, w_dt=0.2, w_nn=0.2, w_jvol=50`; `render_surface_only=True, surface_grad_frac=0.50`; `assim_iso=True`; `use_gauss_loss=False` |
| material base | `young=1.4e5, poisson=0.2` → `lam0 = 38888.9`, `mu0 = 58333.3` |
| leaves | `s` = (2, N) log-Lamé, `λ = lam0·e^{s₀}`, `μ = mu0·e^{s₁}`; `dFc` = (T, N, 3, 3) |
| surface set | `_surface_weights(src, 24, 0.5, 0.05) > 0.5` → 51.17 % of particles |
| replicates | seed 0 / seed 1 (perturbation amplitude `max|δs| = 0.3`), seed 2 (`max|δs| = 1.0` = the whole `mat_clamp` band) |

## A — warm state (4 accepted commits, full promotion + isochoric assimilation)

`optimize_window` exactly as `dress_bench.py` calls it, then the runner's promotion
(`condition_F` → x/F/v/C) and `assimilate_elastic(..., isochoric=True)` on Fp. All four
windows accepted with 8/8 inner steps; 8.8 s total.

| commit | d_vol | d_sil | kin | `dfc_absmax` | control cost `w_ctrl·‖dFc‖²/(T·N)` |
|---|---|---|---|---|---|
| 1 | 428.78 | 0.03511 | 31.97 | 0.1505 | 4.14e−5 |
| 2 | 246.33 | 0.03337 | 21.13 | 0.1593 | 3.39e−5 |
| 3 | 210.32 | 0.03550 | 25.68 | 0.1247 | 3.33e−5 |
| 4 | 200.01 | 0.02918 | 40.39 | 0.1580 | 6.28e−5 |

**Reference scales for §C: an accepted window spends control cost ≈ 3.8e−5 (median) at
`dfc_absmax` ≈ 0.15.**

## B — gradients on the two leaves

At the warm state, `s = 0`, `dFc = 0`, both requiring grad; one `warp_mpm_full` forward,
two backward passes. `d_render` = 0.2057, `d_phys` = `d_vol + 5·mean|v_T|²` = 2828.3.

| leaf | dim | ‖dL_render/·‖ | ‖dL_phys/·‖ | cos | **render / phys** |
|---|---|---|---|---|---|
| `s` | 40 000 | 8.119e−4 | 6.9928 | 0.777 | **1.161e−4** |
| `dFc` | 3 600 000 | 3.575e−3 | 27.301 | 0.667 | **1.309e−4** |

Seed 1 replicate: 1.102e−4 (`s`) vs 1.251e−4 (`dFc`).

> **The material leaf gets no relief from the attenuation.** Its render-to-physics ratio is
> 11–12 % *lower* than the control leaf's, in both replicates. The 1000×-small render channel
> is 1000×-small on `s` too.

### Band decomposition of the render covector

Per-particle magnitude field, split by successive kNN-mean smoothing (k = 16 neighbours plus
self — a self-excluding neighbour mean is not a low-pass), bands = `S_r − S_{r+1}` for
r ∈ {0,1,2,4,8,16} plus the coarse residual. Bands are non-orthogonal, so fractions are of
the band-energy sum.

| field | b0 (<1) | b1 (1–2) | b2 (2–4) | b3 (4–8) | b4 (8–16) | b5 (≥16) | **HF = b0+b1** |
|---|---|---|---|---|---|---|---|
| render → `s` | 0.0571 | 0.0033 | 0.0032 | 0.0036 | 0.0045 | 0.9283 | **0.0604** |
| render → `dFc` | 0.0762 | 0.0047 | 0.0037 | 0.0033 | 0.0032 | 0.9089 | **0.0809** |
| *phys* → `s` (ref) | 0.0639 | 0.0038 | 0.0028 | 0.0021 | 0.0023 | 0.9252 | 0.0677 |
| *phys* → `dFc` (ref) | 0.0718 | 0.0034 | 0.0021 | 0.0015 | 0.0012 | 0.9200 | 0.0752 |

Seed 1: HF = 0.0691 (`s`), 0.0842 (`dFc`), 0.0709 / 0.0744 for the physics references.

> **Answer to the probe question "does the material leaf retain more high-frequency render
> content than the control leaf?" — No, it retains ~25 % less.** Worse for the premise: on `s`
> the render covector is *smoother* than the physics covector (0.0604 vs 0.0677), whereas on
> `dFc` it is marginally *rougher* (0.0809 vs 0.0752). The small render-HF surplus that exists
> at all lives on the control leaf, not the material leaf.

## C — non-cancellability (the decisive test)

Band-limited random perturbation (4 kNN-smoothing rounds), no-grad rollouts for the target
displacement, then Adam over the other leaf against `‖x_T(fit) − x_T(0) − δx‖²`. Because a
single step size cannot decide a reachability question (Adam moves every element by ≈ lr per
step: at lr = 1e−3 the dFc residual was *above* 1.0 after three steps), each fit is a ladder
of 5 (dFc) / 4 (`s`) step sizes × 200 iterations, and the **best iterate found anywhere** is
kept and re-verified on an independent no-grad rollout.

### C1 — material perturbation δs, best reproducing control dFc

| run | max&#124;δs&#124; | ‖δx‖ | δx / replay noise | **residual frac** | cos | amplitude ratio | `dfc_absmax` | control cost | vs. accepted window |
|---|---|---|---|---|---|---|---|---|---|
| seed 0 | 0.3 | 0.5350 | 11 754× | **0.0088** | 0.99996 | 0.9999 | 0.0342 | 2.22e−8 | **1 / 1700** |
| seed 1 | 0.3 | 0.5376 | 13 000× | **0.0083** | 0.99997 | 0.9999 | 0.0235 | 1.77e−8 | **1 / 2100** |
| seed 2 | 1.0 | 2.3121 | 57 911× | **0.0094** | 0.99996 | 0.9999 | 0.1630 | 3.52e−7 | **1 / 107** |

Step-size ladder (seed 0): 0.0109 / **0.0088** / 0.0140 / 0.0293 / 0.0762 at
lr = 3e−3 … 3e−5 — a broad, well-resolved minimum, not a lucky step size. The residual is
~100× above the CUDA-atomic replay-noise floor, so it is real, just tiny.

> **A control that annuls the material change to 0.9 % exists and is cheap.** At the *full*
> `mat_clamp` band (Lamé × e^{±1} = 0.37…2.72×, everything Tier M could ever command) the
> annulling control is `dfc_absmax` = 0.163 — inside the amplitude an accepted window already
> spends (0.125–0.159) — and costs 1/107 of that window's control budget. At the realistic
> per-commit increment (`δ = mat_clamp/animations`, A.2) it is ~1/2000.

### C2 — the reverse: dFc perturbation, best reproducing material `s`

Perturbation rescaled so `‖δx‖` matches C1's, making the two residual fractions directly
comparable. `s` fitted under the production clamp `|s| ≤ mat_clamp = 1`.

| run | `dfc_absmax` | ‖δx‖ | **residual frac** | cos | amplitude ratio | fitted `s_absmax` |
|---|---|---|---|---|---|---|
| seed 0 | 0.0549 | 0.5376 | **0.2470** | 0.9692 | 0.948 | **1.000 (clamp saturated)** |
| seed 1 | 0.0531 | 0.5361 | **0.2380** | 0.9716 | 0.946 | **1.000 (clamp saturated)** |
| seed 2 | 0.2286 | 2.9877 | **0.2815** | 0.9601 | 0.929 | **1.000 (clamp saturated)** |

> **The asymmetry runs against Tier M by 28×.** dFc reproduces material-driven terminal motion
> to 0.9 %; the material leaf reproduces control-driven motion only to 24–28 %, and only by
> pinning its entire clamp — the "saturation kill" pattern (Opus M12) at the very first ask.
> The material carrier is a strict, cheap subset of `span(dFc)`; `dFc` is not a subset of the
> material's reach.

## D — where the material render signal lives

Energy fraction of the per-particle render-covector magnitude on the surface set (51.17 % of
particles).

| leaf | surface energy fraction | enrichment vs. population share |
|---|---|---|
| render → `s` | 0.6202 | **1.212×** |
| render → `dFc` | 0.7733 | **1.511×** |

The material leaf's render signal is *less* surface-concentrated than the control leaf's — a
larger share sits in the interior, where Appendix A's stiffen/soften semantics have no
surface residual to read.

---

## Verdict — **NO-GO for Tier M**

Appendix A's A.1 claim is refuted on its own terms, and its two supporting premises fail too.

1. **"No cheap exact annulling control exists."** One exists: residual 0.83–0.94 %,
   cos ≥ 0.99996, at 1/107 (full-clamp) to 1/2100 (realistic increment) of an accepted
   window's control cost, within the control amplitude a window already spends. A.1's
   argument requires `dFc` to match stress *pointwise in time*; nothing in this pipeline ever
   asks for that — every loss term (`d_vol`, `d_sil`, kinetic, `w_jvol`, W1, nn) reads only
   the terminal `(x_T, F_T, v_T)`, and the terminal state is what turns out to be reachable.
2. **"The material leaf carries render content `dFc` cannot."** Its render-to-physics gradient
   ratio is 11–12 % *worse* than `dFc`'s, and its high-frequency render share is ~25 % *lower*.
   The adjoint low-pass of `docs/root_analysis.md` §2 attenuates the material path at least as
   hard as the control path — as it must, since both covectors traverse the same P2G/G2P
   B-spline and the same s = 0.955 F-smoothing.
3. **The asymmetry Appendix A needs points backwards.** Tier M wants a carrier outside
   `span(dFc)`. Measured: material ⊂ span(dFc) to 0.9 %, while dFc ⊄ span(material) at 24–28 %
   with the clamp pinned.

The A.4 pre-registered doubt ("the optimizer leaf `s` can partially counteract `b` … a cheap
half-cancellation") understates the problem: it is not the material leaf that half-cancels a
material command, it is the **control leaf** that fully cancels it, essentially for free, and
the window objective will find that control because it is the cheaper way to keep the loss
where it was.

### Caveats — stated so the verdict can be attacked

- **Dimension.** `dFc` has 3.6 M parameters against the material leaf's 40 k (90×). That
  asymmetry is real but it is not a confound: the pipeline's control *is* 3.6 M-dimensional,
  and cancellation only needs to exist and be affordable — both measured directly.
- **Objective.** The fits use a bare terminal-position L2, which is *harder* than the
  optimizer's task: the optimizer never has to reproduce a displacement exactly, only to undo
  the part of it the window objective dislikes. The measured residual is therefore an upper
  bound on what a real window would leave standing.
- **Horizon.** This is single-window reachability from one warm state (4 commits into the
  flagship), not a 120-commit accumulation of a persistent base `b`. Tier M could still argue
  a slow cumulative effect — but A.1's own claim, and A-T2's counterfactual realization
  metric, are both defined per-commit against exactly the quantity measured here.
- **Fit convergence.** The C2 (`s`) fit was still descending at 200 iterations, ~0.001 per 20
  iterations; its residual is an upper bound. That trend does not close a 28× gap, and the
  clamp is already saturated.
- **Channel.** Measured with the gauss channel off, per the probe directive. If Tier M is
  ever revisited, the same three numbers (`render/phys` per leaf, HF band share per leaf, C1
  residual + control cost) should be re-measured with `use_gauss_loss=True`; there is no
  mechanism by which a different render scalar would change the adjoint's low-pass, but the
  covector's band content would differ.

### If Tier M is revived anyway, these are its pre-registered kills

Both are already measurable with this script: **C1 residual ≥ 0.25 at a control cost ≥ the
accepted-window median** (i.e. the carrier is genuinely outside `span(dFc)` and expensive to
annul), and **HF band share on `s` > HF band share on `dFc`** (i.e. the material leaf actually
carries the render channel's unique content). Measured today: 0.0088 at 1/1700, and
0.0604 vs 0.0809. Both fail by a wide margin, in three independent replicates.

### Reproduce

```
cd /tmp/pm31 && OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  CUDA_VISIBLE_DEVICES=2 <python> scripts/probes/material_carrier.py --fit_iters 200
```
`--seed`, `--ds_max`, `--lr_dfc`, `--lr_s` select the replicates above; results land in
`output/material_carrier*.json` on hyde06.
