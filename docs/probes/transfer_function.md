# Probe: spatial transfer function of the MPM adjoint (2026-09-02)

Measurement probe for `docs/root_analysis.md` §2, which asserts (unmeasured) that the render
covector's unique high-frequency content is attenuated by the MPM adjoint (cubic B-spline
P2G/G2P + F-smoothing s=0.955) before it reaches the control `dFc`, and that this is the
mechanism behind the "render gradient 1000x small" symptom. Script:
`scripts/probes/transfer_function.py`. Raw numbers: `/tmp/pm31_tf_results.json`
(primary) and `/tmp/pm31_tf_results_assim.json` (variant) on hyde06.

## Verdict

The adjoint `x_T -> dFc` is a **band-pass filter with a ~7x floor, not a 1000x wall**: on the
production window (T=20, s=0.955) a per-particle white covector reaches `dFc` at 0.14 of the
gain of a 2-dx-smooth covector, a 0.1-wu covector (3 NN spacings) at 0.31, a 0.25-wu (dx/2)
covector at 0.69, and everything at or above dx (0.47 wu) passes unattenuated (0.99-1.08);
DC (rigid translation) is an exact null direction (momentum conservation of internal stress),
so the transfer is zero at infinite wavelength, peaks at ~0.6-0.8 wu and rolls off 7x toward
per-particle scale. The **real silhouette covector is spatially rough** (62-65% of its energy
below 0.1 wu, 90% of it on ~1500 rim particles) **but is transmitted almost as well as the
physics covector**: whole-covector gain 0.89 vs 1.03 of the smooth reference (0.86x; 1.05x on
the assimilated variant, 0.81x at T=1, 0.61x at T=5 - the worst case is 1.6x), because its
sub-0.1-wu band is rim-coherent (0.30 of reference, 2x white noise) and its 3-4% coarse
residual carries 30-40% of what gets through. The norm gap between the channels in `dFc`
space (7600x) is the norm gap in `x_T` space (6500x) times 1.2 - it is the loss scale
(d_render 0.03 vs d_vol 240), not the adjoint, and it is what lambda ~ 5e3 compensates. The
adjoint does increase redundancy (cos(g_r, g_p) 0.17 in x-space -> 0.63 in dFc space) but
78% of the transmitted render norm stays orthogonal to the transmitted physics direction.
**F-smoothing is a temporal blend, not a spatial low-pass** (`F_out = (1-s)F_new + sF_in`,
kernels.py:196): at s=0.955 it costs the x-path 4.8x and the F-path 23-34x of gain uniformly
across all scales, and s=0 makes the *relative* fine-scale attenuation worse (0.074 vs 0.141)
because the F-accumulation path adds coarse response; s=1.0 does not disable it, it freezes
F (F-path gain exactly 0). The **F_T -> dFc path is spatially flat** (+-20% from per-particle
to 1.1 wu), so the gauss/covariance channel is not spectrally attenuated at all, only
temporally damped by (1-s). Pre-smoothing the render covector 1-16 kNN rounds (what
`render_gs_iters` approximates) raises its adjoint gain by 1.6-1.9x while keeping cos 0.63-0.40
with the raw direction - that is the ceiling of the "route around the low-pass" idea for the
x-path. Conclusion: the root-analysis §2 mechanism is real but small (<= 7x on white noise,
<= 1.6x on the actual render covector); the three-orders-of-magnitude work gap must be
explained by covector magnitude / lambda / step composition, not by the adjoint.

## Setup

- Warm state: `assets/isosphere.obj -> assets/bunny.obj`, N=20000 (`load_normalized`, seeds
  1/2), `MPMParams()` defaults (dx=0.5, dt=1/240, 64^3 grid, drag 0.9, s=0.955),
  `PipelineConfig(T=20, iters=8, animations=8, loss_res=64)` with lambda_auto=0.5, w_kin=5,
  w_dt=0.2, w_nn=0.2, w_jvol=50, assim_iso=True, no gauss, no surface mask, no PCGrad;
  vol0 = `compute_rest_volumes(src)`; 4 accepted commits of `optimize_window` (8/8 accepted
  steps each), promotion of x/F/v/C exactly as `scripts/dress_bench.py` (Fp = I, **no
  assimilation**). Commit trace (primary): d_vol 428 -> 254 -> 225 -> 242, d_render
  0.035-0.040, lambda 3.4e3-3.9e3, cos(g_p, g_r) in dFc space 0.81 -> 0.62.
- **Variant `--assim 1`**: promotion as `runner.py` (clip, `condition_F`, `assimilate_elastic`
  eta=0.5 isochoric). Motivation: without assimilation the promoted state is a loaded spring
  (F_e singular values 0.56-1.32, |v|max 54 wu/s) and the free dFc=0 rollout the adjoint is
  linearised around flies apart over 20 steps (d_vol 244 -> 2281, kin 44 -> 118; the variant
  reaches 2334/95 - the velocity, not the stretch, dominates, so both states are early-morph
  violent). All relative quantities agree between the two within 10-20%.
- Warm-state geometry: median NN spacing 0.030 wu (0.034 assim), median 16-NN radius 0.112
  wu = 0.22 dx. Both are far below dx: a B-spline stencil (2 dx = 1 wu radius) covers
  thousands of particles.
- Rollout for the measurement: `warp_mpm_full(dfc, spec)` with `dfc = zeros(T,N,3,3)`,
  `RolloutSpec(..., F0=F, Fp, v0=v, C0=C, vol0)` built as in `optimize_window`, `lame(1.4e5,
  0.2)`. Sweep T in {1, 5, 20} x s in {0.955, 0.0, 1.0}. Every gain is one `wp.Tape`
  backward with the covector as `grad_outputs` (68 backward passes per combo).
- Bands: white N(0,1) field per particle-component (seed 0), low-passed by k rounds of the
  (self + 16-NN) mean on the warm-state positions, unit-normalised. The rms radius of S^k
  applied to a one-hot particle field (64 probes) gives the honest scale of each band:

| k | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|
| r_rms (wu) | 0 | 0.095 | 0.130 | 0.176 | 0.244 | 0.349 | 0.473 | 0.627 | 0.826 | 1.063 |
| r/dx | 0 | 0.19 | 0.26 | 0.35 | 0.49 | 0.70 | 0.95 | 1.25 | 1.65 | 2.13 |

  The task's k<=32 list stops at ~1 dx; k=64..256 were added so the sweep crosses the
  B-spline support (2 dx).
- Normaliser: a pure translation covector is a **null direction** of the dFc adjoint
  (internal stress conserves total momentum: partition of unity gives sum_g grad w = 0 and
  drag scales every node alike), measured 1e-8..1e-7 absolute at T<=5 (1e-6 of the reference)
  and a 0.3-1% leak at T=20 on the violently moving trajectory. It therefore cannot serve as
  the "1.0 = no attenuation" reference; the tables use the k=256 band (r = 1.06 wu = 2.1 dx)
  instead. The response peaks at k=32-64, so peak-normalised numbers are ~0.8x the k=256 ones.
- Real covectors, evaluated at each combo's own x_T: g_r = d D_render/d x_T (18 views, 64 px,
  sil_k 1.5, w_hole 2, w_spray 1); g_p = d(D_vol + w_kin kin)/d x_T (= dD_vol/dx_T, kin has no
  x_T dependence); "phys (x_T, v_T)" also seeds w_kin d kin/d v_T.
- Cost: 60 s wall (primary, includes 4 commits), 35 s (variant); peak GPU memory 1.76 GB;
  backward 1.5 ms (T=1) to 16 ms (T=20). GPU 0 only, thread caps set.

## B. x_T -> dFc transfer function

Gain ||J^T g|| for unit g, relative to the k=256 band (1.0 = no attenuation vs a 2-dx-smooth
covector). Primary run; last column = assimilated variant.

| band k | r_rms (wu) | r/dx | T=1,s=0.955 | T=1,s=0 | T=1,s=1 | T=5,s=0.955 | T=5,s=0 | T=5,s=1 | T=20,s=0.955 | T=20,s=0 | T=20,s=1 | T=20,s=0.955 (assim) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.00 | 0.150 | 0.150 | 0.150 | 0.143 | 0.135 | 0.144 | 0.141 | 0.074 | 0.151 | 0.111 |
| 1 | 0.095 | 0.19 | 0.455 | 0.455 | 0.455 | 0.368 | 0.387 | 0.367 | 0.307 | 0.240 | 0.305 | 0.314 |
| 2 | 0.130 | 0.26 | 0.630 | 0.630 | 0.630 | 0.513 | 0.539 | 0.511 | 0.428 | 0.337 | 0.425 | 0.443 |
| 4 | 0.176 | 0.35 | 0.801 | 0.801 | 0.801 | 0.658 | 0.689 | 0.656 | 0.549 | 0.438 | 0.544 | 0.560 |
| 8 | 0.244 | 0.49 | 0.989 | 0.989 | 0.989 | 0.826 | 0.857 | 0.824 | 0.685 | 0.563 | 0.678 | 0.684 |
| 16 | 0.349 | 0.70 | 1.168 | 1.168 | 1.168 | 1.006 | 1.029 | 1.004 | 0.837 | 0.716 | 0.828 | 0.832 |
| 32 | 0.473 | 0.95 | 1.284 | 1.284 | 1.284 | 1.154 | 1.161 | 1.153 | 0.990 | 0.884 | 0.981 | 0.983 |
| 64 | 0.627 | 1.25 | 1.275 | 1.275 | 1.275 | 1.197 | 1.188 | 1.198 | 1.081 | 1.007 | 1.077 | 1.071 |
| 128 | 0.826 | 1.65 | 1.147 | 1.147 | 1.147 | 1.115 | 1.107 | 1.116 | 1.066 | 1.032 | 1.067 | 1.059 |
| 256 | 1.063 | 2.13 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| translation | - | - | 1.9e-06 | 1.8e-06 | 1.8e-06 | 2.5e-06 | 1.9e-06 | 2.5e-06 | 9.0e-03 | 7.8e-03 | 1.0e-02 | 5.9e-03 |
| **render g_r** | - | - | 0.932 | 0.932 | 0.932 | 0.715 | 0.732 | 0.710 | 0.885 | 0.497 | 0.800 | 0.999 |
| **phys g_p** | - | - | 1.147 | 1.147 | 1.147 | 1.177 | 1.218 | 1.176 | 1.032 | 1.137 | 1.018 | 0.950 |

Reading: (i) the single-step filter (T=1) is s-independent (s only touches F accumulation)
and attenuates white 6.7x, 0.1 wu 2.2x, dx/2 1.0x; (ii) longer windows sharpen the roll-off
mildly (k=1: 0.455 -> 0.368 -> 0.307) and shift the peak from ~0.5 wu to ~0.6-0.8 wu; (iii)
removing F-smoothing (s=0) makes the *relative* fine-scale attenuation worse (0.074 at T=20)
because the added F-accumulation response is coarse; (iv) the real render covector sits at
0.72-0.93 of reference on the primary state and at 1.00 on the assimilated one - it is not
attenuated like white noise.

Absolute gains ||J^T g|| (unit g), s=0.955 columns plus the s=0 window for scale:

| covector | T=1 | T=5 | T=20 | T=20, s=0 | T=20 (assim) |
|---|---|---|---|---|---|
| x k=0 (white) | 0.000788 | 0.00595 | 0.0496 | 0.140 | 0.0411 |
| x k=1 (0.095 wu) | 0.00239 | 0.0153 | 0.108 | 0.453 | 0.116 |
| x k=8 (0.24 wu) | 0.00519 | 0.0344 | 0.241 | 1.063 | 0.253 |
| x k=32 (0.47 wu) | 0.00674 | 0.0480 | 0.348 | 1.668 | 0.363 |
| x k=64 (peak, 0.63 wu) | 0.00669 | 0.0498 | 0.380 | 1.901 | 0.396 |
| x k=256 (1.06 wu) | 0.00525 | 0.0416 | 0.352 | 1.887 | 0.370 |
| x translation | 1.0e-08 | 1.0e-07 | 3.2e-03 | 1.5e-02 | 2.2e-03 |
| render g_r | 0.00489 | 0.0298 | 0.312 | 0.937 | 0.369 |
| phys g_p | 0.00602 | 0.0490 | 0.363 | 2.145 | 0.351 |
| phys (x_T, v_T) | 0.0379 | 0.1015 | 0.477 | 2.38 | 0.463 |

s=1.0 (F frozen) gives 0.78-0.83x the s=0.955 x-path gains at T=20 (e.g. k=32: 0.269 vs
0.348): at s=0.955 the x-channel already behaves ~80% like a frozen-F channel.

## C. Where the real covectors live, and what gets through

Band = successive smoothing difference S^{k_i} g - S^{k_{i+1}} g (last row = residual
S^256 g). E = band energy / ||g||^2; gain = ||J^T b||/||b||; gain/ref = relative to the
k=256 white-derived band; share = ||J^T b||^2 / sum over bands. Bands are not orthogonal
(sum E 0.53-0.74) and their images are strongly coherent (sum of per-band transmitted
energies is only 0.24-0.39 of the whole covector's), so read E and share as indicators.

### Primary, T=20, s=0.955 (production window)

| band | r (wu) | render E | render gain | render gain/ref | render share | phys E | phys gain | phys gain/ref | phys share |
|---|---|---|---|---|---|---|---|---|---|
| k0-1 | <0.095 | 0.619 | 0.1042 | 0.30 | 0.276 | 0.309 | 0.1061 | 0.30 | 0.068 |
| k1-2 | 0.095 | 0.021 | 0.1834 | 0.52 | 0.028 | 0.013 | 0.1245 | 0.35 | 0.004 |
| k2-4 | 0.130 | 0.013 | 0.2837 | 0.81 | 0.045 | 0.009 | 0.1937 | 0.55 | 0.006 |
| k4-8 | 0.176 | 0.011 | 0.3319 | 0.94 | 0.051 | 0.009 | 0.2194 | 0.62 | 0.008 |
| k8-16 | 0.244 | 0.009 | 0.3337 | 0.95 | 0.042 | 0.009 | 0.2825 | 0.80 | 0.013 |
| k16-32 | 0.349 | 0.006 | 0.4155 | 1.18 | 0.043 | 0.009 | 0.3449 | 0.98 | 0.020 |
| k32-64 | 0.473 | 0.004 | 0.4869 | 1.38 | 0.043 | 0.009 | 0.4287 | 1.22 | 0.031 |
| k64-128 | 0.627 | 0.003 | 0.5403 | 1.54 | 0.038 | 0.009 | 0.5069 | 1.44 | 0.048 |
| k128-256 | 0.826 | 0.003 | 0.568 | 1.61 | 0.037 | 0.011 | 0.5433 | 1.54 | 0.066 |
| k256+ | >1.063 | 0.036 | 0.5193 | 1.48 | 0.398 | 0.151 | 0.4971 | 1.41 | 0.735 |
| sum | | 0.725 | | | | 0.536 | | | |
| **whole covector** | | 1 | **0.3116** | **0.89** | | 1 | **0.363** | **1.03** | |

### Assimilated variant, T=20, s=0.955

| band | r (wu) | render E | render gain | render gain/ref | render share | phys E | phys gain | phys gain/ref | phys share |
|---|---|---|---|---|---|---|---|---|---|
| k0-1 | <0.100 | 0.652 | 0.132 | 0.36 | 0.393 | 0.382 | 0.1004 | 0.27 | 0.094 |
| k1-2 | 0.100 | 0.020 | 0.2305 | 0.62 | 0.036 | 0.013 | 0.1668 | 0.45 | 0.009 |
| k2-4 | 0.141 | 0.010 | 0.3194 | 0.86 | 0.035 | 0.007 | 0.2408 | 0.65 | 0.010 |
| k4-8 | 0.191 | 0.007 | 0.3664 | 0.99 | 0.032 | 0.006 | 0.2841 | 0.77 | 0.012 |
| k8-16 | 0.262 | 0.006 | 0.4119 | 1.11 | 0.033 | 0.006 | 0.3212 | 0.87 | 0.016 |
| k16-32 | 0.359 | 0.005 | 0.4765 | 1.29 | 0.037 | 0.007 | 0.3824 | 1.03 | 0.026 |
| k32-64 | 0.482 | 0.004 | 0.5451 | 1.48 | 0.044 | 0.008 | 0.4552 | 1.23 | 0.042 |
| k64-128 | 0.649 | 0.004 | 0.5783 | 1.56 | 0.048 | 0.010 | 0.5077 | 1.37 | 0.063 |
| k128-256 | 0.872 | 0.004 | 0.5706 | 1.54 | 0.045 | 0.011 | 0.5236 | 1.42 | 0.072 |
| k256+ | >1.139 | 0.029 | 0.5431 | 1.47 | 0.297 | 0.122 | 0.4676 | 1.27 | 0.655 |
| sum | | 0.740 | | | | 0.574 | | | |
| **whole covector** | | 1 | **0.3693** | **1.00** | | 1 | **0.3512** | **0.95** | |

### Single step, primary, T=1 (the raw P2G/G2P filter)

| band | render E | render gain/ref | render share | phys E | phys gain/ref | phys share |
|---|---|---|---|---|---|---|
| k0-1 | 0.655 | 0.62 | 0.741 | 0.250 | 0.68 | 0.234 |
| k1-2 | 0.024 | 0.90 | 0.056 | 0.020 | 0.89 | 0.031 |
| k2-8 | 0.023 | 0.97-1.16 | 0.076 | 0.041 | 1.04 | 0.089 |
| k8-64 | 0.013 | 1.37-1.47 | 0.081 | 0.052 | 1.09-1.46 | 0.173 |
| k64-256 | 0.003 | 1.46-1.54 | 0.016 | 0.019 | 1.49-1.50 | 0.086 |
| k256+ | 0.011 | 0.96 | 0.031 | 0.187 | 1.01 | 0.386 |
| **whole covector** | 1 | **0.93** | | 1 | **1.15** | |

Reading: the render covector's dominant sub-0.1-wu band is transmitted at 0.30 of reference
(T=20) / 0.62 (T=1) - 2-4x better than white noise at the same scale, because rim particles
share coherent directions - and it supplies 28-39% of what the render channel gets through;
the coarse residual (3-4% of the energy) supplies another 30-40%. The physics covector is
coarse-dominated in transmission (66-74% from k256+) even though 31-38% of its energy is
also per-particle. Sparsity: 90% of ||g_r||^2 sits on 1528 particles (assim 1563), of
||g_p||^2 on 8602 (7853).

### Low-pass curve: pre-smoothing the real covectors (primary, T=20, s=0.955)

S^k g: E_lp = energy kept, gain_lp = ||J^T unit(S^k g)||, cos = alignment with the raw
covector. This is what a grid-GS / kNN preconditioner on the render pull buys.

| k | r (wu) | render E_lp | render gain_lp | gain_lp/gain_0 | render cos | phys E_lp | phys gain_lp | gain_lp/gain_0 | phys cos |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 1.0000 | 0.3116 | 1.00 | 1.000 | 1.0000 | 0.363 | 1.00 | 1.000 |
| 1 | 0.095 | 0.2798 | 0.5037 | 1.62 | 0.625 | 0.5701 | 0.4764 | 1.31 | 0.835 |
| 2 | 0.130 | 0.2187 | 0.5358 | 1.72 | 0.578 | 0.5155 | 0.4961 | 1.37 | 0.800 |
| 4 | 0.176 | 0.1725 | 0.5549 | 1.78 | 0.509 | 0.4698 | 0.5102 | 1.41 | 0.767 |
| 8 | 0.244 | 0.1334 | 0.5735 | 1.84 | 0.454 | 0.4249 | 0.5215 | 1.44 | 0.738 |
| 16 | 0.349 | 0.1029 | 0.5876 | 1.89 | 0.401 | 0.3787 | 0.5304 | 1.46 | 0.709 |
| 32 | 0.473 | 0.0816 | 0.5813 | 1.87 | 0.347 | 0.3309 | 0.5344 | 1.47 | 0.676 |
| 64 | 0.627 | 0.0652 | 0.5615 | 1.80 | 0.299 | 0.2768 | 0.5309 | 1.46 | 0.640 |
| 128 | 0.826 | 0.0506 | 0.5386 | 1.73 | 0.260 | 0.2144 | 0.5192 | 1.43 | 0.602 |
| 256 | 1.063 | 0.0359 | 0.5193 | 1.67 | 0.230 | 0.1509 | 0.4971 | 1.37 | 0.560 |

(Assim variant: render 1.55-1.67x, cos 0.60-0.25; s=0: 1.9-3.0x, cos 0.59-0.21.) One
smoothing round already keeps only 28% of the render covector's energy and 62% of its
direction: the render signal *is* per-particle rough, and smoothing it trades direction for
gain at a ceiling of ~1.9x.

## Alignment and norm ratios

| combo | cos_x(g_r,g_p) | cos_dFc(J^T g_r, J^T g_p) | cos_dFc(J^T g_r, J^T g_p(x,v)) | lambda_est (alpha 0.5, no EMA) | \|\|g_r\|\| | \|\|g_p\|\| | \|\|g_p\|\|/\|\|g_r\|\| in x | in dFc |
|---|---|---|---|---|---|---|---|---|
| T=1 (any s) | 0.205 | 0.435 | 0.179 | 9.6e3 | 0.0113 | 27.9 | 2474 | 3044 |
| T=5, s=0.955 | 0.194 | 0.462 | 0.381 | 6.1e3 | 0.0104 | 36.9 | 3546 | 5835 |
| T=5, s=0 | 0.198 | 0.496 | 0.383 | 6.8e3 | 0.0103 | 35.7 | 3456 | 5756 |
| T=20, s=0.955 | 0.171 | 0.630 | 0.681 | 5.0e3 | 0.0095 | 61.7 | 6492 | 7564 |
| T=20, s=0 | 0.231 | 0.791 | 0.784 | 6.8e3 | 0.0105 | 56.2 | 5373 | 12299 |
| T=20, s=1 | 0.157 | 0.622 | 0.680 | 4.9e3 | 0.0105 | 60.6 | 5798 | 7382 |
| T=20, s=0.955 (assim) | 0.172 | 0.574 | 0.669 | 3.8e3 | 0.0098 | 59.0 | 6036 | 5740 |

The adjoint multiplies the physics/render norm ratio by 0.95-1.65 (2.3 at s=0) - the
"1000x" lives entirely in the covector magnitudes. It raises the render/physics cosine from
~0.2 to 0.43-0.63 (0.79 at s=0): more redundant, but sqrt(1-0.63^2) = 78% of the transmitted
render norm remains orthogonal to the transmitted physics direction. Note lambda_est is at or
above `lambda_cap = 5e3` on this state (the warm commits logged 3.4e3-3.9e3 through the EMA
and the fuller gp); whether the cap binds in production is a separate question.

## D. F_T -> dFc path (gauss / covariance channel)

Absolute gain for unit F covectors (N x 9 white field low-passed as above), plus two
structured seeds.

| seed | T=1, s=0.955 | T=1, s=0 | T=5, s=0.955 | T=5, s=0 | T=20, s=0.955 | T=20, s=0 | any T, s=1 | T=20 (assim) |
|---|---|---|---|---|---|---|---|---|
| F k=0 (white) | 0.0460 | 1.021 | 0.1031 | 2.369 | 0.2085 | 4.871 | 0 | 0.2084 |
| F k=8 (0.24 wu) | 0.0455 | 1.011 | 0.1013 | 2.284 | 0.202 | 4.691 | 0 | 0.2018 |
| F k=32 (0.47 wu) | 0.0446 | 0.990 | 0.0981 | 2.163 | 0.1931 | 4.858 | 0 | 0.1922 |
| F k=256 (1.06 wu) | 0.0419 | 0.932 | 0.0904 | 1.935 | 0.1757 | 5.958 | 0 | 0.1683 |
| identity on every particle | 0.0440 | 0.978 | 0.0917 | 1.729 | 0.1571 | 3.112 | 0 | 0.1552 |
| render-support seed (g_r tiled x3) | 0.0442 | 0.982 | 0.100 | 2.219 | 0.2037 | 4.701 | 0 | 0.2062 |

Relative to k=256, the F-path gain is 1.10-1.24 at k=0 and monotone-flat down to 1.0 (s=0.955)
- the finest scales pass *slightly better* than the smoothest, because the dominant term is
the per-particle direct injection `F_new = (I + dt C)(F + dFc)`, and the (1-s) s^(T-1-t)
temporal chain gives the T=1 value 0.046 = (1-s) exactly. s=0.955 vs s=0: 22x (T=1), 23x
(T=5), 23-34x (T=20). s=1.0: exactly zero (dFc never enters F; the x-path still works through
stress). A render-shaped sparse seed on F passes at the white-noise gain (0.204 vs 0.209).

## Caveats

1. The kNN-mean decomposition is volumetric: a field that is smooth along the rim but zero in
   the interior registers as sub-0.1-wu energy. "62% of the render covector below 0.1 wu"
   therefore overstates its roughness; the measured gain of that band (0.30 of reference vs
   0.14 for white noise) and the low-pass curve are the reliable statements.
2. Successive-difference bands are not orthogonal and their adjoint images are coherent (sum
   of per-band transmitted energies 0.24-0.39 of the direct value); `share` ranks bands, it
   does not partition energy.
3. Linearisation point: dFc = 0 on a warm state that is early-morph and violent (|v|max
   54-58 wu/s; free 20-step rollout takes d_vol from 244 to 2281) - the same rollout the
   optimizer's first iteration differentiates. The assimilated variant (F_e singular values
   0.69-1.14) reproduces every relative number within 10-20%, so the transfer function is a
   property of the discretisation, not of this state.
4. The translation leak at T=20 (0.3-1% of reference; 1e-6 at T<=5) was not chased; candidates
   are nodes dropped at m_g <= 1e-12 and float32 atomics on the fast trajectory.
5. Raw adjoint only: Adam's per-component normalisation, the line search, PCGrad, the surface
   mask and the balancer EMA all reshape what the *update* sees; none is measured here. Fp = I
   in the primary (dress_bench recipe); no gauss channel was instantiated.
6. Two primary runs of the final script differ by <= 10% in absolute gains (CUDA atomics +
   line-search path) and by <= 0.01 in every relative entry.

## Reproduction

```
scp -o BatchMode=yes -J chayo@hyde01.dabh.io scripts/probes/transfer_function.py \
    chayo@hyde06.dabh.io:/tmp/pm31/scripts/probes/transfer_function.py
ssh -o BatchMode=yes -J chayo@hyde01.dabh.io chayo@hyde06.dabh.io 'cd /tmp/pm31 && \
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 \
  /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python scripts/probes/transfer_function.py'
# variant: ... transfer_function.py --assim 1 --out /tmp/pm31_tf_results_assim.json
```
Defaults: `--n 20000 --commits 4 --T 1,5,20 --s 0.955,0.0,1.0 --knn 16 --seed 0`. The
script prints the tables above and writes the JSON progressively after each combo.
