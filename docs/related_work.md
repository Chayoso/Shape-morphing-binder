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
