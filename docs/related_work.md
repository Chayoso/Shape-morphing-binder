# Related work — what each design choice leans on (2023–2026 unless foundational)

## Render feedback → physical parameters (the core loop)

| paper | take |
|---|---|
| [NeuMA (NeurIPS 24)](https://proceedings.neurips.cc/paper_files/paper/2024/file/78b6beab44f92adc74ac1fdb212ac3a0-Paper-Conference.pdf) | image loss drives a LOW-DIM physical correction on a diff-MPM prior (our material field s) |
| [PAC-NeRF (ICLR 23)](https://arxiv.org/pdf/2303.05512) | Eulerian–Lagrangian split renderer↔simulator; the representation-conversion layer |
| [GIC (NeurIPS 24)](https://arxiv.org/pdf/2406.14927) | 2-D mask surrogates rendered from the simulated continuum (our D_render) |
| [PhysDreamer (ECCV 24)](https://arxiv.org/abs/2404.13026) | spatially-varying material optimised through image-space loss |
| [OmniPhysGS (ICLR 25)](https://arxiv.org/pdf/2501.18982) | per-Gaussian constitutive selection under render supervision |

## Gaussian–physics binding / surface feedback

[PhysGaussian (CVPR 24)](https://arxiv.org/pdf/2311.12198) (sim repr = render repr,
Σ=σ₀²FFᵀ rides F) · [Gaussian Surfels (SIGGRAPH 24)](https://arxiv.org/abs/2404.17774) ·
[GauSTAR (CVPR 25)](https://openaccess.thecvf.com/content/CVPR2025/papers/Zheng_GauSTAR_Gaussian_Surface_Tracking_and_Reconstruction_CVPR_2025_paper.pdf) ·
[GASP](https://arxiv.org/abs/2409.05819) · [GausSim (ICCV 25)](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Shao_GausSim_Foreseeing_Reality_by_Gaussian_Simulator_for_Elastic_Objects_ICCV_2025_paper.pdf)

## Optimisation stability of differentiable simulation

[SAPO/Rewarped (RSS 25)](https://arxiv.org/pdf/2412.12089) (analytic-gradient
stabilisation, batched Warp) · [Unrolled-training differentiability (TMLR 24)](https://arxiv.org/pdf/2402.12971)
(chaotic-horizon budget — why T stays short) · [Diff-MPM active damping control](https://arxiv.org/html/2512.13214)
(rest as an objective — our w_kin) · [FluidLab (ICLR 23)](https://arxiv.org/pdf/2303.02346)
(checkpointing + loss shaping) · subdivision-stabilised B-spline MPM (CMAME 23; cell-crossing
→ ejection at the discretisation level)

## Solvers — the v3 direction (grid-GS + VBD-MPM)

| paper | take |
|---|---|
| [VBD (SIGGRAPH 24)](https://arxiv.org/pdf/2403.06321) | per-vertex block descent, coloring, damped local solves — transplanted to GRID NODES here |
| [AVBD (2025)](https://graphics.cs.utah.edu/research/projects/avbd/) | augmented/hardened VBD; constraint handling if we ever need it |
| Gast et al. 2015 (TVCG) | implicit MPM as energy minimisation over grid DOFs — the variational ground our §7 stands on |
| [HOT (TOG 20)](https://arxiv.org/abs/1911.07913) | multigrid-quasi-Newton on the MPM grid — the "global" acceleration path beyond 2-color GS |
| [DiffPD (TOG 21)](https://arxiv.org/abs/2101.05917) | differentiating THROUGH a projective/block solver via the IFT adjoint — our probe replicates this |
| [Stable Neo-Hookean (Smith 18)](https://graphics.pixar.com/library/StableElasticity/paper.pdf) | the SVD-free energy the VBD arm minimises |
| [3DGS-LM (ICCV 25)](https://arxiv.org/abs/2409.12892) | second-order treatment of splat residuals (GN diagonal ideas for D_render) |
| [Repulsive Curves (TOG 21)](https://arxiv.org/pdf/2006.07859) / [Preconditioned Deformation Grids (PG 25)](https://arxiv.org/pdf/2509.18097) | Sobolev-metric gradients / grid-based gradient preconditioning — §6 verbatim lineage |
| Wang 2015 (Chebyshev PD) | semi-iterative acceleration once the sweep operator is stationary |

## Where to look when specific failures recur

- render term inert → norm balancing (C++ `get_control_layer_grad_norm`), NeuMA-style
  low-dim parameterisation; verify with grad_analysis probes.
- ejection / holes / ellipsoids → experiments.md forensics + method.md §5 (asym D_render,
  leash), §8 (assimilation), §9 (guards); CMAME 23 if it persists at the discretisation.
- oscillation → w_kin (active-damping lineage), assimilation, λ-free freeze; VBD arm
  removes the mechanism entirely.
- slow convergence → warm start, §6 Sobolev direction, HOT-style multigrid, Chebyshev.


## 2022–2026 survey for the render-controls-physics contract (2026-09-14)

Verified against arXiv/publisher pages on 2026-09-14 (PhysMorph-GS excluded by
instruction); mechanism digest in `docs/render_controls_physics.md` §1. Columns: what the
render/observation loss is, which PHYSICAL variable receives its gradient, how the two
are balanced, the gradient horizon, the stability trick.

| paper | observation loss | physics variable receiving the gradient | balancing | horizon / checkpointing | stability trick |
|---|---|---|---|---|---|
| PhysGaussian (CVPR 24, 2311.12198) | none (forward only) | none | – | – | anisotropy regulariser, opacity-ray interior filling; Σ = FΣ₀Fᵀ |
| PhysDreamer (ECCV 24, 2404.13026) | L1 + D-SSIM vs SVD video | triplane E(x) 8³; triplane v₀(x) | fixed λ, TV | gradient to previous frame only; checkpoint + recompute | v₀ first then E; ν fixed; 10–50× fewer driving particles than Gaussians |
| DreamPhysics (AAAI 25, 2406.01476) | motion distillation (SDS − static score) | KAN material field | – | frame boosting M=5 | log-scale material |
| Physics3D (2406.04338) | SDS | per-particle Lamé + viscosity corrections | – | 400 substeps/frame | viscoelastic split |
| OmniPhysGS (ICLR 25, 2501.18982) | SDS (text-to-video) | expert-selection nets (12 constitutive experts) | – | M≪N frame mini-batches, repeated passes | hardmax + STE |
| PhysFlow (CVPR 25, 2411.14423) | optical-flow L2 | E, ν, τ_Y, η, friction | – | per frame pair | MLLM init; render loss "restricted to subtle motions" |
| Resonance4D (2604.01994) | SSIM + temporal FFT | 6 log-domain params per part | additive | 16-frame windows | log-domain clipping, LHS init |
| PAC-NeRF (ICLR 23, 2303.05512) | photometric | material params, v₀ | – | staged | L-BFGS v₀ on frames 2–3 |
| GIC (NeurIPS 24, 2406.14927) | Chamfer + rendered-mask L1 | E, ν, τ_Y, friction, μ, κ, η | equal | staged | v₀ from 3 frames; coarse-to-fine fill |
| NeuMA (NeurIPS 24, 2410.08257) | L2 image | LoRA residual on a constitutive law | – | 1000 RAdam iters | Mahalanobis many-to-many Gaussian↔particle binding |
| Spring-Gaus (ECCV 24, 2403.09434) | L1 + D-SSIM (+mask) | per-anchor k, κ, v₀, friction | λ 0.05 | horizon grown as loss converges | 2,048 anchors + IDW |
| PhysTwin (ICCV 25, 2503.17973) | 3-D Chamfer + tracks (render only for appearance) | dense spring k, collision params | – | – | CMA-ES first, then gradients |
| MASIV (ICCV 25, 2508.01112) | trajectory L1 + silhouette | neural constitutive MLP | additive | 1000 steps | v₀ first |
| AS-DiffMPM (NeurIPS 25, 2511.06846) | photometric | μ, κ, τ_Y, η, friction | – | – | differentiable CPIC contact; v₀ fixed |
| ProJo4D (TMLR 26, 2506.05317) | render | geometry → appearance → physics | progressive unfreezing | – | staged (10× accuracy) |
| MonoPhysics (2605.30320) | opacity L1 + Sinkhorn silhouette + flow + kNN reg | E, ν, τ_Y, v₀, x_p, V_p, Σ_p | uniform | frames/iter 4 → 16 | distribution regulariser stops clustering |
| MOSIV / DiffWind (ICLR 26, 2603.06022 / 2603.09668) | video | per-object materials / grid wind field | – | – | "object-level, geometry-aligned objectives are critical" |
| PIDG (AAAI 26, 2511.06299) | flow + render | per-Gaussian time-varying constitutive params | – | – | Cauchy-momentum residual constraint |
| Xu et al. (SCA 23 / TVCG 25, 2409.15746) — the baseline | log-mass grid loss, no images | additive per-particle F̃ at control layers | line-searched Adam | 10-step chained graphs, multi-pass | γ-smoothing of F fwd+bwd, ζ=0.5 damping |
| Bolliger et al. (2512.13214) | kinetic-energy cost | boundary velocities, body forces | – | 8-step batches | RK4 |
| SHAC (ICLR 22, 2204.07137) | reward | policy | – | h=16–32 + critic (‖g‖>1e6 at 1000) | truncation |
| Metz et al. (2111.05803) / Suh et al. (ICML 22, 2202.00817) / Onoda et al. (ICLR 26, 2604.18161) | — | — | — | — | chaos-based explosion; zeroth/first-order mixing; variance control dominates |
| Schnell & Thuerey (2405.02041) | control loss | network | sign-agreement | full unroll | block the network-input feedback path |

Balancing / conflict: PCGrad 2001.06782, CAGrad 2110.14048, Nash-MTL 2202.01017, GradNorm
1711.02257; physics-domain evidence GradBlend 2607.25060 (anchor the step to the primary
gradient; surgery inflates the primary metric), Per-Loss Adapters 2605.10136 (surgery
fails on heterogeneous parameter spaces), Gradient–Update Mismatch 2609.01558 (Adam's
state re-introduces conflict; project the UPDATE).

Solvers: VBD (SIGGRAPH 24, 2403.06321; Chebyshev ρ=0.95, skipped on colliding vertices),
AVBD (SIGGRAPH 25), Chebyshev PD (Wang, TOG 2015), residual-driven Chebyshev XPBD
(Visual Computer 2026), JGS2 2506.06494, Coordinate Condensation 2510.12053,
Position-Based MPM (Lewin, SIGGRAPH Talks 2024 — the nearest "particle-wise
Jacobi MPM"), APS-MPM (Visual Computer 2025), HOT (TOG 20, 1911.07913),
i-PhysGaussian (2602.17117: Newton–GMRES implicit MPM on Gaussians, 20× dt).

MPM ringing / holes: CK-MPM 2412.10399 (C² kernel, cell-crossing), MPM Lite 2602.07853
(quadrature decoupled from particles), Continuum Foam (Yue 2015, resampling),
conservative split/merge (CMAME 2023), CPDI domain scaling (Homel 2016), APIC (Jiang
2015), GaussianFluent 2601.09265 (hollow surfaces / needle floaters at large deformation),
Sim Anything 2411.12789 (stiffness/curvature-adaptive driving-particle density).

Viewers: viser (`add_gaussian_splats`, ssh -L 8080), Rerun ≥0.36 (`GaussianSplats3D`,
`--serve-web`), gsplat.js / antimatter15 splat / GaussianSplats3D (client-side),
GS_Stream, dylanebert/gaussian-viewer (server-side render + WebRTC).
