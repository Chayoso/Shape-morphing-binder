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


## Thin-feature transport, numerical fracture and mass ejection (read 2026-09-15)

Read for `docs/thin_feature_transport.md`. Items marked * were verified from abstracts only.

- **Numerical fracture / separation.** Yue, Smith, Batty, Zheng, Grinspun, *Continuum Foam*, TOG 34(5) 2015 (10.1145/2751541): stencil-limited connectivity, SDF-gated Poisson-disk resampling, merging, sub-grid removal. Homel, Brannon, Guilkey, IJNME 107 2016 (10.1002/nme.5151)*: CPDI domain scaling controls the onset of "extension instabilities". de Vaucorbeil, Nguyen, Hutchinson, CMAME 360 2020 (Total-Lagrangian MPM: no fracture); Su et al. A-ULMPM, CGF 2022 (2108.00388); HLFEMP, Appl. Math. Model. 2024*. Liu, Wang, Li, *CK-MPM*, 2412.10399: narrower C² kernels fracture more, 27 ppc restores integrity. Feng et al., *MPM Lite*, 2602.07853: quadrature decoupled from particles; thin structures still degrade. Yao, Zhao, 2603.03860 (2026): conservative split + support-gated APIC (`ω_p = smoothstep((n_c/n_0 − r_lo)/(r_hi − r_lo))` on the affine term). Fei et al., *ASFLIP*, TOG 40(4) 2021 (10.1145/3450626.3459678): the positional trap; grid advection as cohesion. Jiang, Gast, Teran, TOG 36(4) 2017 (10.1145/3072959.3073623) and Han et al., PACMCGIT 2(2) 2019 (10.1145/3340258): Lagrangian-energy MPM (bond-graph forces scattered to the grid; meshless MPM fractures, manifold-only stress clumps). Bagherzadeh, Barani, Comput. Particle Mech. 2024 (10.1007/s40571-023-00608-8)*: MPM–peridynamics coupling. Truong et al., TVCG 2021 (2107.08093) merging/splitting; Gao et al., TOG 36(6) 2017 (10.1145/3130800.3130879)* adaptive GIMP merging; Lewin, *Position-Based MPM*, SIGGRAPH Talks 2024 (10.1145/3641233.3664323).
- **Mass ejection in controlled MPM.** Xu, Levin, SCA 2023 (10.1145/3606037.3606840)*: "the nodal mass loss … can lead to mass ejection", log loss introduced against it. Xu, Song, Levin, Hyde, TVCG 2025 (2409.15746): L = Σ ½(ln(m+1) − ln(m*+1))², ejected clumps as local minima ("quickest decrease"), F-smoothing γ 0.93–0.965, control every 10 steps, 32³, 7.6k–12k particles. Song, Hyde, *PhysMorph-GS*, 2511.16988 (for the record only — excluded from our design per instruction): min-mass retention term, thin features "remain difficult". Huang et al., PlasticineLab, ICLR 2021 (2104.03311); Hu et al., DiffTaichi, ICLR 2020 (1910.00935); Bolliger et al., 2512.13214 (kinetic cost, 8-step batches); Pahng et al., DiffeoMorph, 2512.17129 (Zernike-moment loss, blind to strays).
- **Coherence regularisers.** Luiten, Kopanas, Leibe, Ramanan, *Dynamic 3D Gaussians*, 3DV 2024 (2308.09713): L_rigid (local rigidity on frozen k=20 kNN, w_ij = exp(−λ_w‖Δμ₀‖²), λ_w = 2000), L_rot, long-term L_iso; "rigidity alone necessary and adequate". No 2024–26 differentiable-MPM paper adds ARAP/bond-stretch/neighbourhood-preserving terms (searched).
- **Connectivity-preserving morphing.** Alexa, Cohen-Or, Levin, ARAP interpolation, SIGGRAPH 2000 (10.1145/344779.344859); Eisenberger, Cremers, ECCV 2020 (2004.05199); Buonomo, Digne, Chaine, SGP/CGF 2025 (10.1111/cgf.70196); Bizzi et al., FLOWING, 2510.09537; Hsieh, Charon, varifold metamorphosis, 2112.04644 (transport alone cannot match differing thin features). Growth/target-metric control: Ortigosa-Martínez et al., Appl. Math. Optim. 2024 (10.1007/s00245-024-10117-6); Zhou, 2604.04984; van Rees, Vouga, Mahadevan, PNAS 114(44) 2017 (10.1073/pnas.1709025114); Cislo, Pavlopoulos, Shraiman, 2302.07839 (smoothest growth field); Solomon et al. 2015 (entropic OT blurs thin features).
- **Isolated-particle practice.** Houdini MPM Solver docs (affine-C clamping, CFL as a voxel fraction, voxel dilation); nialltl MLS-MPM notes; Stomakhin et al. 2013 (10.1145/2461912.2461948). Taichi Elements / ZENO: nothing documented.
- **Coherence metrics.** Venna, Kaski, JMLR 11 2010 (trustworthiness/continuity); Batschelet 1981 / Benhamou 2004 (straightness, tortuosity); Tsai, Leung, LTVE, 2401.09222; Wang, Gao, Duan, 2603.15683 (topological OT distortion).

## Surface from particles and the triangle-splatting line (2026-09-19)

See `docs/surface_gradient.md`: what the render channel resolves, why it does not flatten the surface, five mechanisms to give it that role, and a table of how SuGaR, 2DGS, Gaussian Surfels, Triangle Splatting (+, 2D) and 3D Gaussian Triangulation obtain their surfaces (regularisers with formulas, meshing, manifoldness, reported geometry).

## Smooth deformation DOF, particle shifting and resampling, oscillation damping (read 2026-09-23 night)

Read for the sub-cell question of docs/method.md §10.17a (the pair c300 / d300: the method has no
term that orders the quadrature below the cell; the outer loop reverses window to window while it
chases sub-cell residuals). Six papers, digest by the research agent (extracted texts in the
session scratchpad); Lind 2012 itself is paywalled and is taken from the authors' 2020 review.

- **Eisenberger, Lähner, Cremers — divergence-free shape interpolation (SGP 2019, arXiv
  1806.10417).** The deformation is a stationary Eulerian velocity v = ∇×Φ with Φ in the first
  K = 3000 Dirichlet-Laplacian eigenfunctions of the box (a sine basis, eq. 9–12); volume is
  preserved exactly (eq. 6) and high modes are damped a priori by a Karhunen–Loève prior
  λ_k = (π² Σ_d j_d²)^(−D/2) (eq. 13). Particles follow ẋ = v(x) by RK2 in T = 20 steps. Loss:
  soft correspondences (EM) + Huber, one Gauss-Newton / LM step per EM round; r₀ = 0.01, σ² = 0.01
  stated, not derived. Smoothness is by construction: there is no per-particle DOF.
- **Eisenberger, Cremers — Hamiltonian dynamics for shape interpolation (ECCV 2020).** The same
  basis with time-varying coefficients (K ≈ 1000); H = ½‖v‖² + W(p), W an anisotropic ARAP with
  learned per-vertex metrics (eq. 9); a variational implicit Euler with the divergence-free
  constraint (eq. 10–11), velocity extrapolation v̄ = 2v^{t+1} − v^t (Thm 5.1: O(τ²)); an outer
  shooting problem on the initial coefficients. No dissipation, nothing on oscillation. Repo:
  T = 16, 100 Adam iterations, lr 0.02.
- **Smooth Shells (CVPR 2020).** X′ = X + Φ_K α in the first K Laplace–Beltrami eigenfunctions
  (eq. 4) + ARAP; a shell operator with sigmoid weights (eq. 8) and Thm 1: the change between
  consecutive scales is bounded independently of K; K log-spaced 6 → 500 over 50 iterations,
  each warm-started. λ_feat, λ_arap, σ tuned ("the same set for all experiments").
- **NeuroMorph (CVPR 2021).** Per-vertex displacements from an EdgeConv net — no basis, no
  null-space removal; smoothness only through losses (ARAP between consecutive frames, geodesic
  preservation); T grown on a log scale (1 → 3 → 7) for "faster and more robust convergence";
  hyperparameters on a validation set.
- **Lind, Xu, Stansby, Rogers 2012 (JCP), via Lind, Rogers, Stansby 2020 (Proc. R. Soc. A §10).**
  Particle shifting: the disorder measure ∇C_i = Σ_j m_j ∇_i W_ij / ρ_j (10.1), the Fickian shift
  Δx_s = −D Δt ∇C_i (10.2) with D = λ h² / Δt, λ < 0.5 the explicit-diffusion stability limit
  (10.3–10.4) — the one constant is derived from the discretisation; masses are untouched, only
  positions move; the velocity correction is "often not necessary as the shifting distance was
  very small compared with the particle spacing"; shifting is restricted near the free surface.
  The velocity-scaled variant D = A h |u| Δt (A ∈ [1, 6], default 2; Skillen 2013) is tuned.
- **Yue et al. 2015 (continuum foam) §6; Gao et al. 2017 (adaptive GIMP) §7.2 — MPM resampling.**
  Yue: particles as spheres of radius h/2; Poisson-disk insertion where the SDF is below −2.2 r
  (empirical) outside spheres of radius ρ r with ρ = √3/2 + 1/100 chosen so an intact 8-ppc lattice
  triggers nothing; the new particle takes 1/(N+1) of each neighbour's mass and volume (exactly
  conserving); v, F by mass-weighted interpolation, F rescaled to the interpolated J; merge when a
  neighbour is closer than 0.03 r; every 50 steps. Gao: split into 4/8 children on a rotated cube
  of half-diagonal dx/4 (mass, volume divided; v, F copied), merge at the centroid (F by SVD
  averaging), gated by tuned distances to the surface and by per-cell counts.

What this settles for §10.17a:
(a) The only construction that removes the sub-cell null space of the control is a band-limited
    field for the control (the Eisenberger line: K Fourier / Laplace–Beltrami modes, high modes
    damped a priori, volume by curl). Our equivalent is the cell-scale control basis
    (`--control_grid`, method.md 10.x), whose 300k verdict (det F 0.005, run b300) was taken with
    the unit-mass bug and must be re-read with the mass contract. NeuroMorph's per-vertex DOF +
    losses is our current situation.
(b) The discretisation-derived, mass-preserving repair of sub-cell disorder is Fickian shifting
    with λ = ½ h² (the stability limit): positions only, one explicit diffusion step of the
    particle concentration per window, restricted to the tangent plane on the outer layer (the
    free-surface rule). Resampling (Yue, Gao) conserves mass exactly but carries empirical
    thresholds and breaks particle identity (our F conditioning and control are per particle).
(c) None of these optimises window by window; the shooting formulations damp by implicit
    variational integration, O(τ²) extrapolation, LM damping and warm starts across scales
    (Smooth Shells' bounded inter-scale change). Their lesson for our outer loop: the residual
    the control cannot resolve must not be in the objective it descends — (a) removes the DOF
    that chase it; the window-to-window reversal is the symptom.

Refinement of item 5 (the agent, 20:25): the anti-pairing factor entered shifting WITH Lind 2012 —
a citing paper (SPH-ASR, arXiv 2008.01326 §3) states that Lind et al. "modified the concentration
gradient by adding an artificial pressure-like function to prevent particle pairing instability";
the δ⁺-SPH form (arXiv 2109.09697 eq. 16) writes it as Σ_j [1 + R (W_ij / W(Δs))ⁿ] ∇W_ij V_j with
R = 0.2, n = 4 (Monaghan 2000) and Δs the initial spacing. Whether Lind 2012 used exactly R = 0.2,
n = 4 is unverified (paywalled). Kernel and h: the group's open ISPH paper (Xenakis et al.) uses a
quintic spline with h = 1.3 dx and shifting δr = −A h |u| Δt ∇C (A = 2, floor D₀ = 0.01 h²); the
quintic spline for Lind 2012 is corroborated by a snippet of Mokos 2017, h = 1.3 dx for Lind 2012
specifically is inferred, not read. Our (39) takes R = 0.2, n = 4 and the Gaussian at h = Δp (the
width of the spline at h = 1.3 Δp) on that basis, and marks them as literature constants.

## Competitors and baselines, 2024–2026 venues (scan 2026-09-23 night; verified from arXiv / OpenReview / venue pages unless marked)

The user asked what this is better than, against FEM-based morphing and the recent ICLR work.
The research agent's scan (17 papers; the closest first).

**The same family — differentiable MPM with a deformation-gradient control (the user's own prior
work; this project is its follow-up, closing what PhysMorph-GS lists as open):**
- *A Differentiable MPM Framework for Shape Morphing*, Xu, Song, Levin, Hyde — TVCG 2025 (SCA
  2024 best poster); predecessor *Deformation Gradient Control of Amorphous Solids*, Xu and Levin,
  SCA 2023. Per-particle F control, chained multi-pass, topology change (differing connected
  components), a log-based mass loss, a GEOMETRIC target (mesh / particles), no images. Code
  linked on the project page. This is the ancestry of our physics-only arm (the "C++ oracle" of
  the config); the morph by F control itself is prior work.
- *PhysMorph-GS*, Song, Hyde — arXiv 2511.16988 (venue unverified). MLS-MPM on a 32³ grid +
  3DGS render on a surface subset (~1k anchors → 10⁶ render particles), per-particle F control,
  Chamfer-guided plasticity, silhouette + depth + edge losses, a grid-mass loss at anchors only;
  topology partial (duck → quadruped fails); 21–30 min per episode at 18–25 GB on an RTX A4000;
  lists watertightness and thin features as open. The closest to "rendering controls physics".

**Implicit / neural morphs (public code, minutes per pair, watertight SDF surfaces):**
- *Implicit Neural Surface Deformation with Explicit Velocity Fields* — ICLR 2025 (Sang et al.,
  Cremers): neural SDF + velocity field from point-cloud pairs, topology change shown, a soft
  divergence-free term, sparse correspondences, ~20 min per pair.
- *4Deform* — CVPR 2025 (the same group): SDF + Euclidean velocity, deviatoric-stress and strain
  losses + a divergence regulariser, endpoint point clouds only, 8–10 min per pair.
- *Volume Preserving Neural Shape Morphing* — SGP/CGF 2025 (Buonomo, Digne, Chaine): an SDF
  advected by an adaptive-divergence velocity, volume preserved with a guarantee (equal volumes).
- *Spectral Meets Spatial* — CVPR 2024 (Cao et al.): mesh + functional maps + ARAP, no topology
  change; beats NeuroMorph, Hamiltonian (Eisenberger 2020), LIMP.
- *FLOWING* — NeurIPS 2025: an invertible INR flow, landmarks, no topology treatment.
- ICLR 2024: no 3D shape-interpolation paper found; ICLR 2026: none in this group.

**Image gradients through MPM / Gaussians (they fit materials or run forward; not morphs):**
PAC-NeRF (ICLR 2023), PhysGaussian (CVPR 2024), GIC (NeurIPS 2024 oral; mask L1 + Chamfer →
materials, at least 1.5 h per object), NeuMA (NeurIPS 2024), OmniPhysGS (ICLR 2025; SDS from a
video model, ~8k particles), AS-DiffMPM (NeurIPS 2025), Fracture-GS (ICLR 2026), PhysDreamer and
Spring-Gaus (ECCV 2024). **4D generation:** ShapeGen4D (ICLR 2026, feed-forward, no physics),
PhysGen3D (CVPR 2025), Phys4DGen (ACM MM 2025) — forward simulation from a generated asset.

**Synthesis (the agent, checked against our docs).**
(a) Runnable baselines on sphere → bunny and sphere → C with public code: the ICLR 2025 implicit
    velocity-field morph, 4Deform, the volume-preserving SDF morph (equal volumes needed), and
    the TVCG 2025 MPM morph (our own family, geometric target). The material-fitting works are
    baselines only for the render → physics gradient path, not for the task.
(b) Already covered by others: per-particle F control of differentiable MPM for topology-changing
    morphs (SCA 2023 / TVCG 2025); F control + splat silhouette/depth losses on MLS-MPM
    (PhysMorph-GS); image gradients through MPM (PAC-NeRF, GIC, NeuMA); volume-preserving neural
    morphs. Plausibly ours: the whole trajectory as optimal control with the Sinkhorn transport
    term coupled to the volumetric cell sum; exact mass conservation across the morph (against an
    anchor-only or a soft term); shading supervision (not silhouette/depth only); the λ = 0 twin
    isolating the render channel; the spacing-derived constant contract; the measured limits of
    the render channel (docs/oscillation.md Addendum 9; experiments 2026-09-23).
(c) Where we lose on their numbers: surface quality (watertight SDF surfaces, no sub-cell
    roughness; PhysMorph-GS names thin features open, as do we); speed (8–20 min per pair for the
    implicit morphs on old GPUs against our 16–44 min at 300k); memory (10⁶ render particles
    against their 8–100k). Unverified: several wall-clocks; PhysMorph-GS's venue.

## Geometry-side survey 2013–2026: smooth interpolants and smooth surfaces from particles (agent digest, 2026-09-24 00:05)

Read for the user's directive (docs/experiments.md 2026-09-23 night: until the oscillation is zero
and the surface smooth). Verified on ACM / Wiley / EG / GitHub pages unless marked; the full digest
with links is in the session scratchpad.

**Interpolation / morphing (smoothness, mass, topology):** Heeren et al., *Exploring the Geometry
of the Space of Shells* (SGP 2014) and *Splines in the Space of Shells* (SGP 2016; code GOAST) —
time-discrete geodesics and a covariant-acceleration functional per window (an oscillation
detector for a tracked mesh); Sassen, Schumacher, Rumpf, Crane, *Repulsive Shells* (SIGGRAPH 2024,
code) — intersection-free geodesics in shell space, a clean reference morph; Solomon et al.,
*Convolutional Wasserstein Distances* (SIGGRAPH 2015, code) — mass-preserving volume barycenters
at grid resolution; Lavenant et al., *Dynamical OT on Discrete Surfaces* (SIGGRAPH Asia 2018) —
the action ∫ρ|v|² as a tortuosity metric; Bonneel, Coeurjolly, *SPOT* (SIGGRAPH 2019, code) — a
W2-type distance between 10⁵–10⁶-point sets, cheap; Buonomo, Digne, Chaine, *Volume Preserving
Neural Shape Morphing* (SGP 2025, code) and *Explicit Flows for Implicit Surfaces* (SIGGRAPH
2026, code) — volume-preserving / invertible implicit morphs, the closest geometry-side
competitors; Eisenberger et al. (SGP 2019) already listed.

**Surface from a noisy particle set:** Kazhdan, Hoppe, *Screened Poisson* (TOG 2013) — our
reconstruction; Kazhdan et al., *PSR with Envelope Constraints* (SGP 2020, `--envelope` in
PoissonRecon) — Dirichlet constraints on a particle-derived hull so the fit cannot bridge gaps or
fatten one-cell sheets; Sellán, Jacobson, *Stochastic PSR* (SIGGRAPH Asia 2022, gpytoolbox) —
the PSR as a Gaussian process, its posterior variance = "is this bump supported by particles?";
Yu, Turk, *Anisotropic Kernels* (TOG 2013) — per-particle PCA covariances, smooth flats and
preserved sheets (our renderer's `--kernel pca` is this line); Löschner et al., *Weighted
Laplacian Smoothing for Particle Fluids* (VMV 2023, splashsurf) — post-smoothing damped near
splashes and isolated particles, no volume loss; Zhao, Shinar, Schroeder (CGF 2024) — CNN SDF
from splatted particles [venue detail unverified]; Wang et al., *Neural-Singular-Hessian*
(SIGGRAPH Asia 2023, code) — offline smooth implicit fit; Huang et al., *Edge-Aware Point Set
Resampling* (TOG 2013, CGAL) — feature-preserving point denoising; Sharp, Crane, *A Laplacian
for Nonmanifold Triangle Meshes* (SGP 2020, robust-laplacians) — a PSD Laplacian on the
outer-layer points, the tool for a spectral roughness / oscillation metric without meshing.

**Shortlist for us.** Surface: (1) envelope-constrained screened PSR with the envelope = the
union of particle spheres at the spacing rule (one cell at the ear); (2) anisotropic kernels +
feature-weighted Laplacian smoothing; (3) stochastic PSR variance as the test of which bumps
are particle-supported (smooth only below the uncertainty floor; report the retained area).
Trajectory: (4) a spectral band criterion on the point-cloud Laplacian — the per-window
displacement projected on the first K eigenvectors (K from the sub-cell length), roughness =
high-band energy, oscillation = consecutive-window correlation of the low band; (5) the path
action Σ W2(P_k, P_{k+1}) against W2(P_0, P_K) via SPOT — an excess-action oscillation budget,
mass-weighted by construction.

## MPM particles ↔ grid resolution ↔ surface, 2013–2026 (agent digest, 2026-09-24 00:20)

Read for the same directive. Verified on arXiv / ACM / Wiley / Springer pages unless marked. No
paper treats trust-region or Levenberg–Marquardt outer loops over MPM windows (ChainQueen,
DiffTaichi, PlasticineLab use plain Adam / GD) — a genuine gap our window-loop breathing sits in.

**Lagrangian mesh coupled to MPM:** Jiang, Gast, Teran (SIGGRAPH 2017) and Guo et al. (SIGGRAPH
2018) — a Lagrangian mesh carries in-manifold strain, the grid only inertia and contact (a
codimensional treatment of a 1.1-cell ear tip); Han et al. (SCA 2019) — tet mesh for internal
force, grid for self-collision; Cao et al., *Unstructured MLS-MPM* (Comput. Mech. 2025) — MLS
kernels on graded tets. **Adaptivity and error analysis:** Gao et al., *Adaptive GIMP* (SIGGRAPH
Asia 2017; third-party Taichi code) — octree SPGrid with a C¹ partition-of-unity basis; Sun et al.
(IJNME 2020), Luo, Li, Jiang (arXiv 2026) — local B-spline refinement / overlapping Schwarz
subdomains; Steffen, Kirby, Berzins (IJNME 2008) — the internal-force error is a QUADRATURE error
of the sub-cell arrangement; **Gritton, Berzins (Comput. Particle Mech. 2017), Tran, Sołowski
(IJNME 2019) — the null-space filter: per-cell SVD of the P2G operator, the particle components
invisible to the grid removed**; Sadeghirad, Brannon, Guilkey, *CPDI2* (IJNME 2013) — particle
domains x_p + F_p ξ tracked by corners. **Transfers that reduce sub-cell noise:** Hammerquist,
Nairn, *XPIC(m)* (CMAME 2017; NairnMPM) — removes the mapping's null space without PIC
dissipation, exact as m → ∞; Fu et al., *PolyPIC* (SIGGRAPH Asia 2017); Qu, Li, de Goes, Jiang,
*Power PIC* (SIGGRAPH 2022) — OT, volume-constrained particle domains, uniform distribution and
exact volume inside P2G/G2P; Lewin, *PB-MPM* (SIGGRAPH 2024 talk, code); de Vaucorbeil et al.,
*Total-Lagrangian MPM* (CMAME 2020); Ando, Thürey, Tsuruno (TVCG 2012) — anisotropic split/merge
in sheets. **Surface reconstruction / tracking:** Yu, Turk (TOG 2013); Bhattacharya, Gao, Bargteil
(TVCG 2015) — thin-plate energy between union-of-spheres shells; Löschner et al. (VMV 2023,
splashsurf); Yu, Wojtan, Turk, Yap (Eurographics 2012) and Dagenais et al. (CGF 2017) — a
persistent mesh advected and projected onto the implicit within a band; Baktash, Gillespie,
Crane, *Subgrid Marching Tetrahedra* (arXiv 2026). **Damping:** Gast et al. (TVCG 2015), Wang et
al., *HOT* (TOG 2020) — implicit integration removes simulation ringing, not optimiser reversal;
Koßler et al. (Comput. Particle Mech. 2025) — a random grid-origin shift per step removes
fixed-grid stress oscillation.

**Shortlist for us (the agent's, checked):** (1) F-domain surfels — CPDI2 corners x_p + F_p ξ
as sub-cell-resolved surfels for the objective and the render, anisotropy F_p F_pᵀ, constants
from V_p; no physics change. (2) **The null-space / XPIC projection at the window commit** — the
window's displacement projected onto the grid-representable subspace (G2P ∘ P2G with the
B-spline weights), applied to positions and the u channel; the direct test of whether the
breathing and the sub-cell disorder live in the grid-invisible subspace. (3) A grid-origin shift
per window (Koßler) as the diagnostic of grid-locked bias. (4) A persistent mesh surface projected
within a half-spacing band (Yu 2012 / Dagenais 2017). (5) Local refinement at thin features
(Sun 2020 / Luo 2026); Power PIC weights in place of the Fickian shift as the runner-up.

## Transport-driven droplets and filaments at protrusions; density-constrained and support-preserving transport (agent digest, 2026-09-24 01:30)

Question put to the survey: why does a transport-paced target grow a thin protrusion (the bunny
ear, ≈ 1.1 cells) as droplets and a filament, and which formulations keep the transported mass one
body. Statements marked *verified* were read from the paper's page by the agent; the others are its
reading of abstracts and secondary sources and are marked as such.

**A. Why a transport target splits mass**
- Bonneel, van de Panne, Paris, Heidrich, *Displacement interpolation using Lagrangian mass
  transport*, SIGGRAPH Asia 2011. An RBF decomposition, a Kantorovich pairing and partial transport
  of each blob; a blob may be split across destinations and each piece travels its own straight ray
  (wording unverified). Our per-window target is exactly such a blob-wise straight-ray advection.
- Solomon et al., *Convolutional Wasserstein distances*, SIGGRAPH 2015. *Verified*: "the method
  blurs the input distributions, and the interpolated distributions are typically of higher entropy
  than the endpoints", repaired by an entropy-bound projection. A 1.1-cell ear is what the blur erases.
  Feydy et al., *Sinkhorn divergences*, AISTATS 2019: the debiased divergence (our `ot_debias`).
- Chizat, Peyré, Schmitzer, Vialard, *Unbalanced optimal transport* (FoCM 2018) and *An interpolating
  distance between optimal transport and Fisher–Rao* (JFA 2018). *Verified*: "geodesics between
  mixtures of sufficiently close Diracs are made of translating mixtures of Diracs"; beyond a cut
  length mass is destroyed and created instead of transported (the cut length's value unverified).
  Growth in place next to the existing support instead of long rays.

**B. Formulations that keep the mass one body**
- Maury, Roudneff-Chupin, Santambrogio, *A macroscopic crowd motion model of gradient flow type*,
  M3AS 2010; Mészáros, Santambrogio, *Advection–diffusion equations with density constraints*, 2016.
  rho ≤ 1; the actual velocity is the projection of the desired one onto the admissible fields; the
  pressure lives only where rho = 1. The saturated zone moves as one body, no filaments. → §10.22.
- Perthame, Quirós, Vázquez, *The Hele-Shaw asymptotics for mechanical models of tumor growth*,
  ARMA 2014; Di Marino, Chizat, *A tumor growth model of Hele-Shaw type as a gradient flow*, ESAIM
  COCV 2020; Gallouët, Monsaingeon, *A JKO splitting scheme for Kantorovich–Fisher–Rao gradient
  flows*, SIMA 2017. Growth = pressure-driven expansion of a saturated region; protrusions grow as
  tongues from the boundary (tip growth: Campàs, Mahadevan 2009). → §10.22's free-surface pressure.
- Benamou, Carlier, Santambrogio, *Variational mean field games*, 2017; Cardaliaguet, Mészáros,
  Santambrogio, density constraints ("pressure equals price"), 2016; Papadakis, Peyré, Oudet,
  *Optimal transport with proximal splitting*, SIIMS 2014. Dynamic transport under a density cap on a
  staggered grid — the window target as the first frame of a congested geodesic on our own grid.
- Ferradans, Papadakis, Peyré, Aujol, *Regularized discrete optimal transport*, SIIMS 2014; Paty,
  d'Aspremont, Cuturi, *Regularity as regularization*, AISTATS 2020. A spatially regular plan /
  a smooth strongly convex potential ⇒ a Lipschitz map: neighbouring mass is not sent apart.
- Eisenberger, Lähner, Cremers, *Divergence-free shape correspondence by deformation*, SGP 2019, and
  *Hamiltonian dynamics for real-world shape interpolation*, ECCV 2020. A band-limited
  divergence-free velocity, exactly volume preserving, no self-intersection — it cannot form a
  sub-cell filament. Zhang, Smirnov, Solomon, *Wassersplines*, SCA 2022: one smooth neural velocity
  field carrying the whole density with a Sinkhorn divergence and PDE regularisers (their exact
  list unverified). Feydy, Charlier, Vialard, Peyré, *Optimal transport for diffeomorphic
  registration*, MICCAI 2017: transport as the objective, a diffeomorphic flow as the carrier.

**C. Thin features in particle and MPM methods**
- Ando, Thürey, Tsuruno, *Preserving fluid sheets with adaptively sampled anisotropic particles*,
  TVCG 2012: neighbourhood anisotropy splits particles in thin sheets and merges them in bulk, so
  sub-kernel features keep their particles per cell.
- Jiang, Gast, Teran, *Anisotropic elastoplasticity for cloth, knit and hair*, SIGGRAPH 2017; Guo et
  al., *A material point method for thin shells with frictional contact*, SIGGRAPH 2018; Fei, Guo,
  Wu, Huang, Gao, *Revisiting integration in the material point method*, SIGGRAPH 2021. A Lagrangian
  mesh carries the codimensional elasticity, the grid the contact; particles more than a cell apart
  lose their interaction — a 1.1-cell ear sits at the fracture threshold.

**D. Our own prior work, read for the same defect.** Xu, Song, Levin, Hyde, *A differentiable
material point method framework for shape morphing*, TVCG 2025: per-particle control F, a log
nodal-mass loss chosen against mass ejection, chained windows, the bunny's ears reproduced, the
smoothness weight hand-set per example, no thin-feature mechanism. *PhysMorph-GS* (arXiv
2511.16988): render gradients through F, the surface-focused subset, the largest gains on
thin-feature targets, and the stated limitation that particles cannot be created at thin
protrusions where the source lacks density.

**The shortlist as applied.** (1) the support-preserving paced target — the plan's step projected
onto divergence-free fields on the body with p = 0 on the free surface (Maury 2010, Perthame 2014,
Eisenberger 2019): built as §10.22, pre-registered as n300; (2) the unbalanced (WFR) window plan
(Chizat 2018): the fallback if (1) costs transport; (3) an entropy-bounded target (Solomon 2015):
not the mechanism here — the transit density is a property of displacement interpolation, not of
the blur; (4) the thin-feature carrier — Ando-split particles whose destination is thinner than two
cells, or a Lagrangian spine (Ando 2012, Jiang 2017): the fallback if (1) does not thicken the
stream.

*Verification of the three sources §10.22 leans on (02:40, from the arXiv abstracts):* Maury,
Roudneff-Chupin, Santambrogio 2010 — verbatim "the actual velocity is the projection of the desired
one onto the set of admissible velocities" under the incompressibility (density) constraint;
Eisenberger, Lähner, Cremers 2020 — verbatim "exactly volume preserving intermediate shapes" and
"avoids self-intersections" with a divergence-free deformation; Perthame, Quirós, Vázquez 2014 —
the abstract states the Hele-Shaw limit and that the free boundary's motion needs the cell-density
equation besides the pressure, but not where the pressure lives: that statement remains the agent's
reading of the paper's body.

## Temporal coherence of particle surfaces, surface vs material velocity, window-loop oscillation, the position null space (agent digest, 2026-09-24 09:30)

Question put to the survey: is the visible tail motion (a) the layer moving along its normal,
(b) the render objective pushing the surface back and forth, or (c) particles rearranging
tangentially under a stationary surface so that the per-frame Poisson fit re-samples a different
point set — and what the literature does about each. Marks: *verified* = read from the fetched
page; otherwise the agent's reading of abstracts.

**Temporally coherent reconstruction.** Yu, Turk, *Reconstructing surfaces of particle-based
fluids using anisotropic kernels*, TOG 2013 — anisotropic kernels from a weighted PCA plus
(*verified*) "a smoothing step that repositions the centers of these smoothing kernels": a
low-pass of the point set before the fit. Bhattacharya, Gao, Bargteil, *A level-set method for
skinning animated particle data*, SCA 2011 / TVCG 2015 — the smoothest (thin-plate) surface
constrained to lie between the union-of-spheres shells of the particles, "skins each frame
independently while preserving the temporal coherence of the underlying particle animation": a
point-set change smaller than the band leaves the optimum unchanged, where screened Poisson
interpolates the surfels and inherits their sub-band motion. Yu, Wojtan, Turk, Yap, *Explicit mesh
surfaces for particle-based fluids*, EG 2012 (*verified*: vertices "advected using nearby particle
velocities", "periodically project the mesh surface onto an implicit surface") and Dagenais,
Gagnon, Paquette, CGF 2017 (a detail-preserving SDF-band projection): a persistent mesh that
re-imports the implicit only outside a band. Bojsen-Hansen, Wojtan, *Liquid surface tracking with
error compensation*, SIGGRAPH 2013 (*verified*: an error function between the tracked surface
and the physically valid states) — the error itself is a measure of "how much the implicit moved
relative to a coherently advected surface". Akinci et al. CGF 2012 (narrow-band scalar field, no
temporal mechanism); van der Laan et al. I3D 2009 (screen-space curvature flow; a grid-locked
extraction is a documented flicker source); Löschner et al. VMV 2023 (splashsurf: weighted
Laplacian mesh smoothing, treats spatial roughness only); Zhao, Shinar, Schroeder, CGF 2024 / 2025
(CNN on a grid splat: the temporal coherence comes from averaging sub-cell jitter on the grid
before any fit); Chen, Zhou, Zhu, *Neural particle level set*, TOG 2025 (oriented particles as
trackers and seeders, an SDF evolved from the previous frame). Kazhdan's PoissonRecon README
(*verified*): `--samplesPerNode` 1.0–5.0 for clean samples, 15–20 for noisy ones; Open3D's
binding exposes depth, width, scale, linear_fit only. Every method reported coherent either
low-passes the point set, fits the smoothest surface inside a band, or keeps a persistent surface
with a band-limited projection; none blends level sets across frames.

**Surface velocity vs material velocity.** Stam, Schmidt, *On the velocity of an implicit
surface*, TOG 2011 (*verified*): "only the normal component of the velocity is unambiguously
defined" — the surface's normal velocity between consecutive implicits, compared with the
particles' normal displacement, is the discriminator between (a)/(b) and (c). Kelly, Optics
Letters 1979 (*verified*): the flicker sensitivity peak at 0.5 cpd / 7.6 Hz — a sign flip every
window at 20–30 fps is a low-spatial-frequency 10–15 Hz modulation inside that peak, invisible as
geometry (3 % of a spacing) and visible as flicker.

**Window-loop oscillation.** DiffTaichi (ICLR 2020), PlasticineLab (ICLR 2021), PhysMorph-GS
(arXiv 2511.16988) and the TVCG 2025 morph run tens of Adam/GD iterations and report no
window-to-window reversal measure or remedy (*verified* for the three reachable texts). Peng et
al., *Anderson acceleration for geometry optimization and physics simulation*, TOG 2018
(*verified*: window m = 5, the accelerated iterate taken only "if [it] decreases the energy").
Mao, Szmuk, Açıkmeşe, successive convexification (*verified*: a trust region updated by the ratio
of actual to predicted decrease). Metz et al. 2021, Suh et al. ICML 2022 (unrolled-gradient
chaos and bias). The gap: no differentiable-MPM paper tests a window's commit on the true merit.

**The position null space.** Jiang, Schroeder, Teran, *An angular momentum conserving APIC*, JCP
2017 (*verified*): "particle velocity modes persist, invisible to the dynamics on the grid only to
reappear after particle movement"; PIC/APIC filter velocity null modes — the text does not
discuss POSITION null modes, which no transfer filters. Gritton, Berzins 2017 (null-space
filter), Tran, Sołowski IJNME 2019 (temporal + null-space filter), Baumgarten, Kamrin IJNME 2023
(*verified*: MPM's errors are the ringing instability and the solution-dependent integration
errors; a δ-correction shifts positions), Fei et al. ASFLIP SIGGRAPH 2021 (a position correction
from the particle's own velocity), Sun et al. δ⁺-SPH CMAME 2019 and the corrected
transport-velocity line (the free-surface rule: kill the normal shift, keep the tangential — a
generator of (c) under a sub-cell-resolved fit).

**Shortlist as applied.** (1) the decomposition twins and the level-set normal velocity — the
measurement (no constant); (2) the reconstruction band-limited to the MPM cell (finest Poisson
leaf = the cell, from Kazhdan's samples-per-node rule: √15 spacings ≈ 0.9 cell) — expressible
with the existing `--poisson_cell` (cell / reference spacing = 2.27); (3) a tracked mesh advected
by the GRID velocity with a projection band of half a spacing; (4) a window commit accepted on a
true-merit ratio test (SCvx ρ ≥ 0) or a safeguarded Anderson iterate (m = 5) — the structural
cure for the merit reversal; (5) Yu–Turk repositioning of the surfels over a 2-spacing radius;
(6) the alternating (Nyquist-bin) power of the video's per-frame change as the perceptual
measure. Not to do: blending level sets across frames (no mechanism, hides (a)); tangential-only
shifting at the surface before (2) is in place.

## Thin-feature persistence in particle → surface pipelines (agent digest, 2026-09-24 09:50)

Question put to the survey: the tongue's sparse leading edge is reconstructed in one frame and
not the next — reconstruction or physics, and what keeps a thin feature present frame to frame.
Marks as above.

**Reconstruction of thin features from sparse points.** Yu, Turk TOG 2013 (additive anisotropic
kernels never cancel: a two-particle sheet still yields a slab of thickness ≥ h/k_r, an isolated
particle a sphere; per-frame). Ando, Thürey, Tsuruno TVCG 2012 (*verified*: "we preserve fluid
sheets by filling the breaking sheets with particle splitting in the thin regions … we compute the
anisotropy of the particle neighborhoods, and use this information as a resampling criterion").
Kazhdan, Hoppe, *Screened Poisson*, TOG 2013 (*verified* README: `--samplesPerNode` 1.5 adapts
the octree to the sampling density — sparser regions get COARSER nodes; `--pointWeight`, `--trim`)
— two failure modes at a thin tip: opposite-facing normals of a sheet thinner than the finest
node cancel in the splatted field, and the density adaptivity coarsens the node exactly there.
Kohlbrenner, Liu, Alexa, Kazhdan, *Symmetrized Poisson*, SGP 2025 (normals replaced by their
outer products: the two faces of a sheet no longer cancel). Lin et al., Parametric Gauss
Reconstruction, TOG 2022 (normal-free, thin-aware, ~1 min per 40k points). Huang et al., NKSR,
CVPR 2023 (*verified*: complete geometry "at the level of extreme sparsity", seconds per frame).
Bhattacharya, Gao, Bargteil SCA 2011 / TVCG 2015 (the surface must stay between the
union-of-spheres shells: a tip can never vanish). Löschner et al. VMV 2023 (splashsurf, every
particle contributes). Sandim et al. CGF 2016 / C&G 2019 (boundary detection by hidden-point
removal; boundary particle resampling in poorly sampled regions before the reconstruction).
Kong et al., Metric-Phase Fields, 2026 (*verified*: why signed implicits lose thin sheets — "a
stable sign becomes ill-conditioned" between nearby layers). Codimensional MPM (Jiang, Gast, Teran
2017; Guo et al. 2018; Han et al. 2019; Wang et al. 2020 MLS codimension flag): the rendered
surface is the carried mesh, never reconstructed.

**Temporal persistence.** Shen, Shah (Pixar) SIGGRAPH sketch 2007 / US 8,010,330 (*verified*: the
signed distance averaged with values at velocity-extrapolated positions in neighbouring frames);
Digital Domain US 8,199,148 (*verified*: particle kernels extended backward and forward in time;
holes and disconnected pieces removed by morphological operations, not by a volume cut); Yu,
Wojtan, Turk, Yap EG 2012 and Dagenais et al. CGF 2017 (a persistent mesh, projected only where
the implicit is trusted); Wojtan, Thürey, Gross, Turk SIGGRAPH 2010 (*verified*: topology changes
"in the presence of arbitrarily thin features like sheets and strands" by reusing the original
surface's points — thin sheets exempted from the topology grid's resolution); Brochu, Bridson
2009 (El Topo); Heiss-Synak et al. SIGGRAPH 2024; Adams et al. SIGGRAPH 2007 (a particle-carried
surface distance = temporal memory on the particles). Houdini's particle-fluid surfacing
(*verified* docs): dilate / smooth / erode in voxel units; no minimum-lifetime filter found.

**Physics side.** Ando 2012 splitting (a commit-time resampling pass: mass-conserving split, C
copied, no forward-model change); Marquez-Razon et al. SIGGRAPH 2022 (*verified*: surface
particles sampled from a union-of-spheres level set, mass m/(2|Π|+1), temporary); Ferstl et al.
narrow-band FLIP 2016 (reseeding in the band from the level set); Kim, Lee, Bhattarai PLOS ONE
2020 (*verified*: thin test σ₃ ≤ α σ₁, a pair-density threshold τ deciding "reconnect vs a real
break"); Levi 2024 (a per-cell particle-count bound as a linear programme); the codimensional
carriers (forward-model changes).

**Shortlist as applied.** The decisive census first: per frame in the ear mask, particles /
surfels / mesh vertices / dropped components and the finest Poisson node against the tip's PCA
thickness. Reconstruction side: R-1 replace the one-cell volume cut with a spacing-derived,
hysteretic keep rule (a native-spacing piece is 1/87 of a cell: every tongue fragment falls
under the cut; keep if ≥ the reference particles of one cell, or within two spacings of the body,
or overlapping a drawn component at t ± 1 by its own velocity); R-2 a finest node ≤ the tip's
half-thickness with no density coarsening (samples-per-node 1.0), Symmetrized Poisson if the
faces still cancel; R-3 anisotropic kernels / union of spheres for components the thin detector
flags (σ₃/σ₁ ≲ 1/4), Poisson elsewhere; R-4 in-plane surfel resampling (Sandim 2019) with Kim's τ
as the break rule; R-5 velocity-advected temporal averaging of the SDF over ±1–2 frames (Shen &
Shah); R-6 a tracked ear mesh. Physics side: P-1 sheet-aware splitting at the commit (Ando 2012;
α from the sheet thickness, gap 2 spacings); P-2 narrow-band reseeding of front cells below half
the reference count; P-3 a codimensional carrier (the principled answer, a forward-model change).

## Practitioner rules on resolution, thin features, pinned material and surfacing — graphics-side cookbooks, engine docs and repos (agent digest, 2026-09-25 16:40)
Question put to the agent (the user: "paper뿐 아니라 cookbook / 각종 해상도 및 관련 자료들 전부"): what the
course notes, engine documentation, tutorials and repos say about grid resolution vs particle count, thin
features, frozen material and surfacing. **[V]** = read on the source page/file; **[S]** = the agent's reading
of a summary. Where a page states no number the digest says so.

### 1. Grid resolution vs particle count (ppc, dx vs feature, dt)
- **8 particles per cell is the production default everywhere.** Houdini MPM Container, *Grid Scale*:
  "Multiplies the Particle Separation to define the voxel width dx of the background grid. The default of 2
  will pack 8 particles per voxel on average" [V]; the MPM Solver's deterministic P2G calls "tightly packed
  particles (8 per voxel default)" the fast path [V]. taichi_elements `mpm_solver.py`: `sample_density =
  2**self.dim` → 8 in 3-D [V]. Warp `example_apic_fluid.py`: `PARTICLES_PER_CELL_DIM = 2` (8 ppc), linear basis,
  free-slip projection `vel_adv -= max(v_n, 0) * sdf_gradient` [V]. ZIRAN 2019 `MpmInit3D.h`: every 3-D scene
  `particlesPerCell = 8`, `cfl = 0.6`, `max_dt = suggested_dt * 0.6`, FLIP/PIC blend 0.98–0.99, walls STICKY,
  ground SLIP μ = 0.15–0.4 [V]. GPU-MPM (Gao et al.): "eight particles per cell are usually required for
  stability considerations"; benchmarks sweep 4/8/12/16 [V]. Taichi `mpm99.py`: 4 ppc in 2-D, quadratic kernel
  [V]; Hu's 88-line MLS-MPM: the "higher quality" recipe drops dt 10× for 4× resolution and multiplies
  particles ×16 (ppc held) [V]. Genesis `MPMOptions`: `grid_density` (cells per metre) 64, `particle_size`
  proportional to the cell (≈ 4 ppc at the reference values, the agent's inference) [V]. PhysGaussian
  defaults `n_grid = 50`, `grid_lim = 2.0`, `substep_dt = 1e-4`; wolf (sand, E = 5e7) `n_grid = 200`, `dt =
  2e-5` [V]. PhysDreamer: `grid_size = 64`, 768 substeps per frame [V]. Houdini FLIP forum (Tamte): reseeding
  particles-per-voxel "should usually be set to gridscale^3" [V]; Lait: FLIP particles "are just markers" [V].
- **Why 8 and not 25.** The SIGGRAPH 2016 course notes (Jiang, Schroeder, Teran, Stomakhin, Selle): "the
  Eulerian grid is the essential computational mesh while particles act as quadrature points" [V]; the
  notes give no ppc number and no dt table [V, absence]. Engineering summaries: ~4 points per cell per
  direction minimises the quadrature error; the null-space (ringing) error is absent at 1 ppc without cell
  crossing [S]. More ppc buys quadrature accuracy up to a point; it adds no resolution and does not stop cell
  crossing — the kernel's job.
- **Kernel order.** Course notes: C1 continuity is required against the cell-crossing instability;
  quadratic B-splines are cheaper, cubic "provides wider coverage, therefore less sensitive to numerical errors
  such as numerical fracture when they are not desired artistic effects"; linear "theoretically unstable" [V].
  D_p = ¼ dx² I (quadratic), ⅓ dx² I (cubic) [V]. Our cubic choice is the conservative one for the ear.
- **dt / CFL.** Houdini MPM Solver: *CFL Condition* = "the fraction of voxel width that a particle is
  allowed to travel within a single timestep"; *Material Condition* = a multiplier on the stiffness-based
  maximum; substeps automatic up to 10 000 [V]. taichi_elements: `default_dt = 2e-2 * dx / size * dt_scale`,
  adaptive `cfl_dt = allowed_cfl * dx / max_grid_v` [V]. Blender Mantaflow: "CFL Number … in grid cells per
  time step" [V]. ZIRAN: cfl 0.6 and 0.6–0.7 × the stiffness dt [V]. FLIP Fluids: fast fluid disappears
  unless *Max Frame Substeps* is raised [V].
- **dx vs the smallest feature.** No MPM doc states an "N cells across a feature" rule; the FLIP / level-set
  world does: FLIP Fluids wiki — "the size of a voxel is the minimum amount of physics detail that can be
  resolved", an obstacle "should be at least 1 voxel thick" but "at least 2 voxels is best to be fully
  resolved in all cases", an inflow "at least one voxel" [V]; Houdini FLIP — "If the Particle Radius Scale /
  Grid Scale ≥ √3/2, then particles will never be under-resolved" [V]; OpenVDB `ParticlesToLevelSet` —
  minimum radius 1.5 voxels "corresponding to the Nyquist frequency" [V]; Houdini VDB from Particles — radius
  below 1.5 voxels "will likely cause aliasing artifacts" [V]; Houdini MPM Water Glass example — a thin glass
  needs *Voxel Size* 0.001 or "the top of the glass disappears" [V].

### 2. Thin features (sheets / strands) in practice
- Houdini FLIP keeps thin sheets by **reseeding**: *Surface Oversampling* within *Oversampling Bandwidth* of
  the surface (≥ 2 for thin sheets), with the warning that too much reseeding makes "fluid gaining volume
  over time" [V]; forum recipe against flicker: particles per voxel +1–2, surface oversampling +10–20,
  bandwidth 1.5–2 [V]. Droplet detection: "a fully droplet particle also does not contribute velocity back to
  the fluid simulation" and "can also break up tendrils on the leading edge of splashes" [V] — Houdini
  deliberately DECOUPLES sub-cell material from the grid rather than resolving it. *Narrow Band*: "very fast
  moving simulations might require a larger bandwidth to maintain stability" [V].
- Houdini MPM has no thin-shell / cloth mode (the H21 list: sleeping, resimulation, surface tension,
  deterministic P2G, phase IDs, debris, post-fracture) [V]; small-scale / sparse work goes to Vellum / Grains
  [V]. MPM Source *Surface* type, *Oversampling* "can make the material more resilient against fracturing",
  *Relax Points* against clumping [V]. Thin colliders: raise the collider's voxel resolution, *Particle-Level
  Collisions*, *Velocity-Based Move Outside Colliders*; "Materials stick together excessively → Reduce the
  Grid Scale … reduces material property bleeding between particles" [V].
- Genesis: `enable_CPIC` "to support coupling with thin objects" [V]; CPIC = coloured distance field +
  compatibility between particles and nodes (Hu et al. 2018) [S].
- splashsurf: radius "1.4 to 1.6 times larger than the original SPH particle radius"; cube size ≤ 1.0 ×
  radius, "start with 0.75 to 0.5"; smoothing length 2.0 r; threshold 0.6 [V]. Houdini Neural Point Surface
  *Liquid* model "good at reconstructing thin sheets of water by connecting nearby water droplets" [V] —
  the thin-feature problem solved at surfacing time. Blender: *Particle Radius* in cell units, *Upres Factor*
  [V]; FLIP Fluids: *Subdivisions*, particle scale 1.0–1.5 smooths, lower "risks mesh holes" [V].

### 3. Frozen / kinematic material and streams past it
- Course notes §12.1 (the rule every engine copies): collisions are applied "on the grid velocity vᵢ
  immediately after forces are applied", **relative to the collider**, `v_rel = v − v_co`; "If the bodies are
  separating (v_n ≥ 0), then no collision is applied"; sticking if |v_t| ≤ −μ v_n, else Coulomb; "Dirichlet
  boundary condition on grid nodes is equivalent to sticky collision"; a second pass on particles "will
  introduce inconsistency in the deformation gradients … should only be turned on if necessary" [V].
- taichi_elements: sticky → `grid_v = 0`, slip → `v − n(n·v)`, separate → `v − n·min(n·v, 0)` [V]. warp-mpm
  (PhysGaussian / PhysDreamer): sticky zeroes the node, slip removes the normal component, friction projects
  inward + Coulomb; `enforce_particle_velocity_*` overwrite PARTICLE velocity before P2G, `set_velocity_on_cuboid`
  overwrites GRID nodes [V] — the field has both idioms, and the particle idiom is ours: pinned particles keep
  entering the momentum average.
- Houdini MPM Collider *Sticky*: projection "even when moving away … prevents the material from detaching";
  *Detection Distance* = the particle radius; animated colliders rigid (VDB + transform) or deforming (surface +
  velocity VDB) [V]. Houdini FLIP *Stick on Collision* / *Slip on Collision* [V]. Houdini MPM Auto Sleep: three
  states — Passive, Active and "Boundary (red) … a hybrid state … needed for this to work under the hood" [V];
  *Pin Constraints* on the MPM Source [V]; the docs do not say whether passive particles leave the grid
  transfer [V, absence]. POP auto-sleep caveat: "Start Asleep" can produce "a drooping effect when sand is
  activated and settles" [V]. The documented dragging artefact: "Materials stick together excessively → reduce
  Grid Scale" (bleeding via shared nodes) [V]; CPIC exists because particles on either side of a sub-cell
  boundary share nodes [S].

### 4. Surface-reconstruction smoothness
- Houdini Particle Fluid Surface: *Voxel Scale*, *Influence Scale* (multiples of separation; "small increases
  can give smoother results but increase cooking time greatly"), *Droplet Scale* < influence; *Dilate / Smooth /
  Erode / Final Smooth* in voxels and iterations; flicker → *Limit Refinement*, adaptivity 0 [V]; forum:
  lower influence → bumpy, lower droplet scale → holes [V]. Houdini MPM Surface: *Masked Smooth* protects by
  *Min Stretch* (J_p) and *Min Curvature* [V]. VDB Smooth SDF: *Mean Curvature Flow* "flatten out bumps",
  *Median* "de-spiking" [V]; OpenVDB `LevelSetFilter`: gaussian ≈ 4 separable mean iterations [V].
- splashsurf: "25 iterations appeared to strike a good balance between an initially bumpy surface and
  potential over-smoothing"; `--mesh-smoothing-weights=on` (feature-preserving), normals smoothing 10,
  `--decimate-barnacles`; reference `-r 0.025 -l 2.0 -c 0.5 -t 0.6 --mesh-smoothing-iters 15` [V].
  PoissonRecon: `--samplesPerNode` 1.5 default, "[1.0–5.0]" for noise-free, "[15.0–20.0]" for noisy; `--depth`
  8 or `--width`; `--pointWeight` 2 × degree; `--envelope`; SurfaceTrimmer removes the low-density sheet [V].
- No numeric dihedral / bumpiness target anywhere; production tunes by eye [V, absence].

### Our values against the rules
| quantity | ours | the rule | source |
|---|---|---|---|
| ppc | ≈ 25 at 40k (dx 0.30 wu); ≈ 190 at 300k under `--disc_ref`, ≈ 25 native | 8 (2 per axis); 4–16 swept | Houdini MPM, taichi_elements, Warp, ZIRAN, GPU-MPM [V] |
| cells across the ear | ≈ 1 | ≥ 2 "fully resolved"; radius ≥ 1.5 voxels (Nyquist) | FLIP Fluids, OpenVDB, Houdini FLIP [V] |
| dt / CFL | window of 18 steps, fraction not reported | CFL 0.6 of a voxel; stiffness dt × 0.6 | ZIRAN, Houdini MPM [V] |
| reconstruction voxel vs separation | screened Poisson at depth; separation ≈ 0.10 wu | voxel 0.5–0.75 × radius, radius 1.4–1.6 × separation; samples-per-node 1–5 clean | splashsurf, OpenVDB, PoissonRecon [V] |

### Five changes, ranked by the agent (with our reading)
1. ≥ 2 cells across the ear AND 8 ppc, jointly: dx ≈ 0.12–0.15 wu — at 40k that is 1.6–3 ppc, so it means
   the 300k set at the native grid (ppc 25, ear ≈ 2 cells), not `--disc_ref`. Cost: nodes × 12, adjoint memory
   likewise unless the grid is sparse. (Our reading: the 300k native run c300 did not fill the ear and dp300
   compresses the dragon's spikes more — the resolution is necessary, the arrival's capacity (§14.0) is the
   missing half.)
2. **The pin as a grid-level collider with the relative-velocity separating rule**, pinned mass out of the
   momentum average. → method.md 10.27 addendum 5 (`--settle_pin_slip`), g41pw.
3. Reconstruct at radius ≥ 1.5 voxels with feature-weighted smoothing (splashsurf 15–25 iterations, OpenVDB
   mean-curvature pass, PoissonRecon `--width` / `--envelope`). → the R-2 item of the D2 phase.
4. Cap the per-step travel at ≤ 0.6 voxel with adaptive substeps; report the CFL fraction per run.
5. Oversample the thin-feature source region 2× and relax it (Houdini FLIP surface oversampling, MPM *Relax
   Points*), with an exact mass budget; keep the grid scaling with N^(1/3) at 300k.
Not reachable this session: the Disney snow PDF and the MLS-MPM/CPIC PDF (size), docs.blender.org (403; the
manual was read from its RST source).

## Slip at frozen and rigid bodies in (MLS-)MPM, 2000–2026 (agent digest, 2026-09-25 17:10)
Question put to the survey: a pinned body that deposits its mass with zero momentum is a no-slip wall with a
boundary layer one kernel support wide (2 dx, cubic B-spline) — how does the community let material slide
past a rigid or frozen body without the mass-averaged drag at shared nodes. *verified* = read from the fetched
paper or abstract page; otherwise the agent's reading.

**Grid-node collision objects: a velocity constraint, never a mass.** Stomakhin, Schroeder, Chai, Teran, Selle,
*A material point method for snow simulation*, SIGGRAPH 2013 (*verified* full text): collision objects are level
sets that deposit no mass; collisions are processed on the grid velocity after forces and again on particle
velocities before the position update; per node v_rel = v − v_co, nothing if v_n ≥ 0, else v_t = v_rel − n v_n
and v'_rel = v_t + μ v_n v_t/|v_t| (v'_rel = 0 when |v_t| ≤ −μ v_n); "sticky" = v'_rel = 0 unconditionally.
Klár et al., *Drucker-Prager elastoplasticity for sand animation*, SIGGRAPH 2016 (*verified*): "three types of
collisions: sticky, slipping, and separating", node constraints with Lagrange multipliers inside the implicit
solve; "processing collisions directly on particles produces poor results … it is necessary to process
collisions using the grid velocities". PlasticineLab, ICLR 2021 (*verified*): grid-based contact with Coulomb
friction after Stomakhin, rigid bodies as time-varying SDFs; for backpropagation a softened contact
s = min{exp(−αd), 1} blending pre/post-projection node velocity; "gradients will vanish if the tasks involve
detachment and reattachment". Newton (newton-physics, Warp) implicit MPM `rasterized_collisions.py`
(*verified* source): per node an SDF value, contact normal, collider velocity, friction and adhesion; the
relative velocity gets an isotropic Coulomb response; collider mass is never added to the node. GeoWarp (2025,
*verified*): Dirichlet nodal velocities and a penalty contact under Warp reverse-mode AD. Common structure: the
wall has its own reference velocity and no mass, so the kernel-wide layer is a slip layer.

**One grid, a discontinuity by node colouring.** Hu, Fang, Ge, Qu, Zhu, Pradhana, Jiang, *A moving least
squares MPM with displacement discontinuity and two-way rigid body coupling* (CPIC), SIGGRAPH 2018 (*verified*
full text): "rigid body" includes "a rigid collision boundary with scripted kinematics motion"; a colored
distance field (unsigned distance, per-surface affinity, side tag) is splatted on the grid and particles
inherit their colour; "a grid node i and a particle p are compatible if and only if for all surfaces shared by
the particle and the grid node, all tags are the same"; "near rigid surfaces, particles only transfer to
compatible grid nodes"; in G2P an incompatible node takes a ghost velocity, the particle's own velocity
projected against the body with boundary type (sticky / slip / separate) and friction, and the impulse goes
to the body. Limitation (*verified*): "it only resolves features at a scale of grid Δx … the compatibility
condition … is a binary decision and essentially grid-aligned". The drag vanishes because the two sides never
share a momentum average.

**Multi-velocity-field contact.** Bardenhagen, Brackbill, Sulsky, CMAME 2000 (own field per body, mass-gradient
normal) and CMES 2001 (*verified* abstract); Nairn, CMES 2013 (*verified* abstract): "extrapolates each material
to its own velocity field … by reconciling momenta at nodes interacting with two or more materials". Nairn,
Hammerquist, Smith, CMAME 2020 (*verified* full text): with the velocity criterion alone "contact is always
detected too early", and for a mid-cell interface "all grid-based methods falsely detect contact too soon" —
our phantom layer, named; the fix is a logistic-regression plane through the point cloud around each contact
node (normal + separation), contact only when approaching AND separation < 0. Graphics: Tampubolon et al.,
SIGGRAPH 2017 (*verified*): "a two-grid Material Point Method"; Gao et al., *Animating fluid sediment mixture in
particle-laden flows*, SIGGRAPH 2018 (*verified* abstract): "two-way coupled through a momentum exchange force …
two MPM background grids"; Han, Gast, Guo, Wang, Jiang, Teran, SCA 2019 (*verified* full text): "to prevent
numerical cohesion between phases common to MPM, we adopt two separate background MPM grids", contact via
collision particles sampled on the boundary "at a density proportional to the grid spacing", Coulomb
impulses — "removes the excessive numerical friction common to traditional MPM". Fang et al., IQ-MPM,
SIGGRAPH 2020 (*verified* full text): "the automatic MPM coupling is inherently restricted to sticky and no-slip
interactions. As a result, solid-solid interfaces experience infinite friction"; separate solid and fluid
domains, normals from the negative mass gradient, free slip weakly in one monolithic solve; open: "to allow
two solids to freely slide against one another". Ménager, Carpentier, arXiv 2602.02038 (2026, *verified*):
"classical node-based approaches enforce no-penetration and no-slip implicitly by sharing the same
interpolation space at a standard grid node"; one grid per object, contact by ADMM. Zong et al., arXiv
2503.05046 (2025, *verified*): single-valued grid, one contact point per particle inside rigid geometry.
SoftMAC (2023/24, *verified*): forecast G2P, per-particle SDF test, drop v_n, decay v_t by Coulomb,
differentiable. Qu et al., *The power particle-in-cell method*, SIGGRAPH 2022 (*verified* abstract): volume
clipping, not slip. Liu, Wang, Li, CK-MPM, 2025 (*verified*): a compact kernel (16 nodes per particle vs 27)
lets "a ball at 1.5Δx margin" fall freely where the quadratic kernel leaves it stuck — the stick layer is the
kernel support.

**Shortlist as applied.** (S-1) the pinned body as a Stomakhin/Newton collider, not a mass: its particles
rasterise a mass field, n = ∇m_pin/|∇m_pin|, v_co = 0, their mass and momentum leave the average; after the
grid update, nodes in the pinned support get the approaching normal component removed (μ = 0 slip, or
Coulomb μ) — one node pass, a piecewise-linear adjoint → **method.md 10.27 addendum 5, `--settle_pin_slip`**;
(S-2) CPIC colouring by the pinned/free interface (floor one cell, grid-aligned); (S-3) a second grid field for
pinned mass with Nairn's approach + separation test (×2 grid memory); (S-4) a narrower kernel (quadratic 1.5
dx; CK-MPM ~1 dx) shrinks whatever layer remains. Not to do: drag coupling (no-slip by construction); soft
contact at evaluation.

## Adaptive resolution and thin-feature carriers in MPM, 2006–2026 (agent digest, 2026-09-25 17:10)
Question put to the survey: resolve a one-cell-wide protrusion (the ear) without refining the whole grid,
under an adjoint that stores every step's grid (dx/2 = ×16 memory).

**Adaptive grids.** Gao, Tampubolon, Jiang, Sifakis, *An adaptive generalized interpolation material point
method for simulating elastoplastic materials*, SIGGRAPH Asia 2017 (*verified* abstract): "adaptive refining
and coarsening of different regions", "a C1 continuous adaptive basis function that satisfies the partition of
unity property and remains non-negative", SPGrid sparse multi-layered grids; levels graded, hanging nodes carry
no DOF (agent's reading). Bird, Coombs, Augarde, Pretti, O'Hare, *An implicit octree-based adaptive MPM*, 2026
(*verified*): refinement by proximity to the rigid surface; hanging nodes constrained to their parent (C0);
a particle splits into 8 when it "occupies elements smaller than β·l_p (β ≥ 2)", never merges back; 5.5× and
29.5× faster than the conforming mesh; CPU, no AD. He, Jin, Zhou, Yin, Chen, *A multi-resolution MPM based on
penalty formulation*, IJNAMG 2025 (*verified* abstract): separate MPM models per level, "bounded material
points that connect these levels", a penalty on their positional deviation, "it is only necessary to map the
penalty forces of bounded material points to the corresponding level background grids. No other modifications
are required"; a penalty factor independent of E and spacing; the fine region is prescribed. Ma, Lu,
Komanduri, CMES 2006 (structured GIMP refinement with transition cells) is the precursor. Li et al. 2026
(*verified* abstract): the multi-level blocks belong to the LBM fluid, the MPM stays single-resolution. Zhao et
al. 2026 (*verified* abstract): sparse but uniform; "adaptive refinement methods usually require significant
changes to the MPM formulation". HOT (Wang et al., TOG 2020): a solver hierarchy on a uniform grid.
Aanjaneya et al., SIGGRAPH 2017: octree liquids, power diagrams. Net: the graphics GPU line 2018–2026 stayed
uniform; every adaptive MPM picks the fine region by distance to a feature and pays a custom basis (C1 AGIMP,
C0 Bird) or a penalty overlap (He 2025).

**Narrower kernels at fixed dx.** CK-MPM (*verified*): compact kernel on dual grids offset ±¼Δx, 16 nodes per
particle vs 27 quadratic and 64 cubic; less diffusion; no AD discussion. MPM Lite (Feng et al., 2026, *verified*
abstract): linear kernels onto fixed quadrature points. A one-cell ear under the cubic kernel spreads over four
to five node columns per axis, three under the quadratic, two under CK.

**Lagrangian carriers.** Jiang, Gast, Teran, SIGGRAPH 2017 (*verified*): the mesh carries the in-manifold
deformation gradient, the grid the orthogonal components; "the apparent thickness of the surface/curve depends
on the grid resolution", "numerical separation … about half a grid cell width or numerical stickiness", "works
best if the size of the surface or curve elements is at most the size of the grid spacing". Guo et al.,
SIGGRAPH 2018 (*verified*): subdivision shells + MPM. Han et al., SCA 2019 (*verified*): elastic forces fully
Lagrangian; collision particles at grid density "remove the effect of grid resolution on collision
resolution". Fan, Chitalu, Komura, CGF 2025 (*verified* abstract): hybrid shell with phase-field tearing.
Hybrid Grains (Yue et al., SIGGRAPH Asia 2018, *verified* full text): an oracle from the packing fraction on a
grid, a reconciliation zone at a chosen distance from the threshold isocontour, mass-splitting with velocity
agreement enforced node-wise, enrichment / homogenisation when the zone moves — the closest published analogue
of the pin, and it re-homogenises when the zone moves.

**Particle-side thin features.** Ando, Thürey, Tsuruno, TVCG 2012: neighbourhood anisotropy as the criterion,
split thin, collapse deep. Ferstl, Ando, Wojtan, Westermann, Thuerey, *Narrow band FLIP*, EG 2016 (*verified*):
particles only within R = 3h–4h of the surface; the naive coupling "leads to a consistent over-estimation of
the fluid motion" at the band edge, fixed by blending. Yue et al., *Continuum foam*, TOG 2015: resampling plus
"an explicit tearing model to prevent regions from shearing into artificially thin threads". None of these
enlarges what the grid can represent; they repair sampling.

**Differentiability.** No adaptive MPM is differentiable; DiffTaichi / PlasticineLab keep static uniform
grids. A penalty-coupled fine box (He 2025) adds only a smooth force; a C0 hanging-node basis is piecewise
linear in the state; a Lagrangian carrier (Han 2019) is FEM, smooth.

**Shortlist as applied.** (R-1) a nested fine MPM box (dx/2) over the ear's bounding box, the coarse grid
untouched, overlap particles "bounded" by a penalty (He 2025) — 8 fine cells per coarse cell inside the box
only (a 2 % box costs +16 %, not ×16), differentiable; (R-2) the compact or quadratic kernel at the present dx
— zero memory, the ear's node footprint halves; (R-3) a Lagrangian carrier for the ear (Han 2019); (R-4) a C0
hanging-node octree inside the box; (R-5) band reseeding at the tip (Ferstl / Ando) for sampling only. Not to
do: global dx/2; ppc as a resolution substitute.

## Freezing converged variables and releasing them (agent digest, 2026-09-25 17:10)
Deep-learning freezing is monotone (FreezeOut 2017, AutoFreeze 2021, Egeria 2022/23, SmartFRZ 2023,
TimelyFreeze 2026, ULMFiT's gradual unfreezing — all *verified* abstracts): none re-opens a frozen block on
evidence. The principled release lives in optimisation: Bertsekas, *Projected Newton methods*, SIAM J. Control
Optim. 1982 (the ε-active set re-identified every iteration from the gradient); Facchinei, Fischer, Kanzow,
SIAM J. Optim. 1998 (*verified* abstract): an identification function of the KKT residual; Tibshirani et al.,
JRSS-B 2012 (*verified*): after solving on the screened set "rely on the Karush–Kuhn–Tucker conditions … add the
variables that fail a KKT check back"; LIBSVM shrinking (Chang & Lin 2011, *verified*): "if not, then we
reactivate all variables". Rigid-body engines release by contact: Box2D islands (*verified*): a velocity
threshold held for a time-to-sleep sleeps the whole island, "bodies wake other sleeping bodies through
constraints and this must propagate through all touching bodies immediately". Applied: the pin is an active
set; release it by (i) a KKT check at each window commit — the adjoint already yields the objective gradient
on pinned particles; un-pin those above the free-particle median — and (ii) contact wake — a pinned node
receiving free-stream momentum above the sleep threshold wakes its connected pinned component. Evidence rules,
not distance rules.

**What is genuinely new to us.** The 2 dx stick layer is a documented property of shared-node MPM (Nairn 2020,
Ménager 2026, CK-MPM), not our bug; every production collider deposits no mass — it is a per-node velocity
constraint with its own reference velocity, and our pin deposited mass, which is the drag; CPIC holds a
discontinuity on one grid at a one-cell floor; two-field contact needs a separation test or it fires a kernel
width early; solid–solid free slip in a monolithic solve is still open (IQ-MPM); no differentiable or GPU
adaptive MPM exists, He 2025's penalty-linked nested levels is the only scheme that changes nothing in the
transfers; codimensional carriers decouple a feature's resolution from dx at half a cell of contact
separation; a compact/quadratic kernel halves both the stick layer and the ear's smear at zero memory; DL
freezing never un-freezes on evidence — the principled release is a KKT check or contact wake; Hybrid Grains
is the closest analogue of the pin. **Top 3 to try:** (1) the mass-less slip collider (S-1, done as addendum
5); (2) the nested fine box by penalty over the ear (R-1: +10–20 % adjoint memory for a 1–2 % box); (3) the
evidence-based release of the pin (KKT on the adjoint gradient at the commit; contact wake). Runner-up at zero
memory: the quadratic or CK compact kernel.

## Growing a thin feature from a settled body: supply, sub-cell losses, capacities (agent digest, 2026-09-26 03:40)
Question put to the survey (the user: "300K 쪽, 귀 안 되면 계속 paper/cookbook 찾아줘"): with the settled body pinned, the
300k ear's tip stalls at 0.5–0.6 of its target density — the cell sum sees no gain, the paced target calls the material
arrived, nothing asks the body to send more. *verified* = read in full text / abstract / doc page. (Later the same night the
end-state ear turned out to be complete — the stall is a density difference, experiments.md 03:30 — and the growth's knob on
a neck is the real defect; the digest's mechanisms are kept for both.)

**1. Target-driven control: how mass reaches a thin feature.** McNamara, Treuille, Popović, Stam, SIGGRAPH 2004 (*verified*):
Gaussian wind controls; smoke bunny / armadillo on 50³ "faithfully reproduce fine scale detail such as the tail and horns";
liquids needed "sources" ("crucial … for matching complex shapes and for preserving the mass"); water "often could only reach
a certain level of detail"; "when the system has trouble matching keyframes, it is often because it is given excessive
control". Fattal & Lischinski, SIGGRAPH 2004 (*verified*): the driving force uses the normalised gradient of the blurred
target; "increasing σ makes it more difficult for the flow to cause ρ to form the finer features of ρ*"; "it is not possible
to match a given target density field solely by advection" when the target is sharper, hence the gathering term, which
diffuses the error; §3.3 splits the smoke into independent density fields with their own targets. (Our reading: gathering is
the local Wasserstein gradient flow of the cell-sum mismatch — it spreads a tip deficit back through the body over time, the
"whole body moves slightly" supply of ai300, exactly what the pin cuts.) Shi & Yu, SCA 2005 (*verified* abstract): a
divergence-free feedback force plus the gradient of a potential "defined by the shape and skeleton of the target". Thürey,
Keiser, Pauly, Rüde, SCA 2006 (*verified* abstract; slides): control particles sampled from the target, the attraction
"scaled down when the influence region of the control particle is already covered with fluid" (deficit-weighted), control on
the low-pass velocity only. Raveendran et al., SCA 2012; Nielsen & Bridson, SIGGRAPH 2011 (a thin shell around a coarse
guide); Pan et al. 2013; Schoentgen et al., SCA 2020 ("a variable proportion of temporary particles"); Chen, Levin, Langlois,
arXiv 2511.15189 (control on a floating coarse grid of 10–20 particle radii). Our own line: PhysMorph-GS v2 (*verified*,
April 2026) — a 64³ grid, the bunny at 534K particles with shell-biased sampling (correcting our 2026-09-23 note of 32³), and
"particles cannot be created or destroyed during simulation, limiting resolution at thin protrusions where the source has
insufficient density". PlasticineLab (*verified*): an SDF·mass term that is zero inside the target — it cannot feed a tip from
within. **Net:** nobody grows a one-cell protrusion from a settled body with exact mass; controllers source mass, prescribe the
transport, or act at the coarse scale; thin features finish last or fail in every source that discusses them.

**2. Losses that see a sub-cell deficit, and which reach the supply side.** GeomLoss (Feydy et al., AISTATS 2019; *verified*
docs): the blur is "the finest level of detail that should be handled"; kernel (MMD) losses are "blind to details which are
smaller than the blurring scale" and at small blur particles "may spread out"; the Sinkhorn divergence's gradient maps a point
to a barycenter and tends to the Monge map as the blur goes to 0 (200k × 202k at σ = 0.01 in 0.3–9 s). **Linearised OT is
Ḣ⁻¹:** Peyre, ESAIM COCV 2018 (*verified*): W₂ "is formally equivalent, for infinitesimally small perturbations, to some
weighted H⁻¹ homogeneous Sobolev norm"; Engquist, Ren, Yang 2020 (*verified*); the flux (Beckmann) form in Solomon et al.
2014 / 2015; Moser Flow, NeurIPS 2021. Our reading: an Ḣ⁻¹ misfit's force on a particle is the gradient of φ with
−Δφ = m − m* — non-local, and by flux conservation the flux through any cross-section of the ear's stem equals the deficit
beyond it: the only term whose supply-side gradient is proportional to what the tip lacks, with no blur. Its limit here: on
the loss grid it reads the same cubic splat as the cell sum — reach, not resolution. A 4ˡ-weighted pyramid approximates it
(reach about 2ᴸ cells). Sliced Wasserstein (Bonneel et al., JMIV 2015; *verified*) reaches the supply side, noisy and
global. DCD (Wu et al., NeurIPS 2021; *verified*): a 1/n weight for a target point shared by several particles (a soft
capacity). A one-sided target-to-particle Chamfer acts on the nearest particle only.

**3. Practitioners.** Houdini Suction Fluid (*verified*): "thin features in the target object will require a high resolution
fluid simulation to fully represent them"; "using a wide Outside Distance will target more particles but also tends to fill
up the region outside … Reducing Outside Distance once the target is full is a good approach" (a coarse-to-fine reach
schedule). POP Attract (a point per particle vs an average position), POP Steer Seek (the nearest goal via attribute transfer
and a goal id, arrival braking), SideFX "Fluid Transform I" (*verified*: "creating points that fill the letters, using an
attractor pop and using the ability of the flip fluid solver to turn off solving for a subset of particles" — the
practitioner's pin plus one goal per particle), tyFlow Set Target (*verified*: "Prevent duplicate assignments" — capacity 1),
FLIP "Reseed Particles", Houdini MPM Source pin constraints. Moana, SIGGRAPH 2017 talk (*verified*): FAB seeding for "the
perpetual birthing and sinking of particles"; Frozen 2 water horse, SIGGRAPH 2020 talk (*verified*): the mane and tail are
hair-solver curves used as particle sources. **Net:** production never grows thin features by long-range transport of bulk
material — it sources mass at the feature, gives the feature its own carrier, or fixes one goal per particle from the start
and freezes subsets.

**4. Capacity-exact vs entropic assignment.** Balzer et al., SIGGRAPH 2009; de Goes et al., SIGGRAPH Asia 2012 ("enforce the
capacity constraints exactly"); Mérigot, CGF 2011; Lévy, M2AN 2015 (semi-discrete, power diagrams); Plateau Holleville & Lévy
2026 (*verified* abstracts). Solomon et al., SIGGRAPH 2015 (*verified*): entropic plans grow "increasingly smooth"; the
barycentric map converges to the OT map as γ goes to 0; "numerics degrade if γ is too small". Pooladian, Cuturi, Niles-Weed
2022 (*verified*): at large ε a "bias towards the mean of the target". Does an exact plan send more mass to a thin tip? The
entropic plan's marginals are exact, but the paced target is built from BARYCENTRIC images, convex combinations of the row's
targets — at an extremal feature every row touching the tip also touches the interior, so the images sit about one blur
inside (method.md 10.8: ~0.9 spacings) and their splat under-fills the tip. An exact assignment's images are target points.

**What is new to us.** Supply is a flux, not a displacement: a body particle's share of the ear's supply falls as the ear's
cross-section over the local one, every link below the pace radius, so "arrived" and the plan-image release (50g) are blind to
it by construction. The paced target under-fills extremal features although the plan is balanced. Gathering is the local
form of our cell sum's Wasserstein flow; its non-local counterpart is the Ḣ⁻¹ misfit. Production avoids long-range supply.
**Ranked mechanisms (the agent's):** (1) the Ḣ⁻¹ Poisson misfit against the fixed target (two FFTs on the loss grid; in the
window objective and the merit; its flux field as the pin's release evidence); (2) exact-capacity images in the endgame (an
auction on a kNN graph, or full-N multiscale Sinkhorn at blur = spacing, debiased); (3) a label channel (Fattal §3.3: the
ear-bound material as its own density field); (4) sliced Wasserstein as a cross-check; (5) the 4ˡ pyramid as the no-FFT
fallback. Not to do: unbalanced OT (breaks exact mass), mass sourcing, entropic sharpening of the paced grid.
**Applied (03:50):** the growth's knob was answered first by the particle-scale KDE term already in the code (ak300 under the
pin: the ear grows as a tongue, knob index 1.1–1.3 against 2.3 without it); the Ḣ⁻¹ term stays the candidate for supply where
the particle-scale term cannot reach.
