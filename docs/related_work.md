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
