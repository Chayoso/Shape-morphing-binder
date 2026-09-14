# Engineering review — oscillation, render feedback, and surface Gaussians

Date: 2026-09-02. PhysMorph itself is intentionally excluded from the reading list.

## Executive finding

The safe architecture is:

```
surface Gaussian residual
    -> terminal surface covector (dL/dx_T, dL/dF_T)
    -> full dynamic MLS-MPM adjoint
    -> dL/ddFc[0:T]
    -> optional SPD control-space preconditioner
    -> Armijo/trust-gated MPM rollout
```

The surface restriction belongs on the terminal observation covector, not on mass,
momentum, grid transfer, or the post-adjoint control field. This lets the interior respond
through continuum mechanics while avoiding the false assumption that every volumetric MPM
sample is a visible Gaussian.

The current production path implements this architecture without a preconditioner. The
tested H1/kNN approximation was not demonstrably SPD, discarded the Gaussian `F` covector
in one path, and degraded every relevant ablation metric; it remains off.

## High-priority findings and fixes

1. `Vp0` must be a source/rest quantity. Re-estimating it from a deformed state changes
   density and stress between optimization windows.
2. Multiplicative plasticity requires the stress chain rule
   `Ptotal = Pe Fp^-T`; otherwise `Pe(F+dFc)^T` is not the intended symmetric Kirchhoff
   stress after assimilation.
3. Line search inside a horizon does not prevent cross-window zigzag. The optimizer now
   carries moments, anneals stale windows, and applies a fixed-scale outer merit/reversal
   rollback.
4. A volumetric particle cloud must not double as the render representation. A frozen
   material-surface subset now owns the Gaussian centers and covariance gradients.
5. Target distance is not a valid definition of a floating primitive. The export-only
   support test uses frozen source-material neighbours and current body isolation, never
   the target, and never modifies simulation state.
6. High sampling density must reduce **absolute** Gaussian size. The 20k operating point
   uses `sigma0=0.04032`, 56.5% below the earlier 5k diagnostic, with 512px CUDA QA.

## Is rendering feedback actually active?

Three independent checks are required:

- a selected-control finite difference agrees with the render-to-`dFc` adjoint;
- render on/off with a fixed `dFc` produces identical physics, proving that observation
  code has no hidden forward-state side effect;
- in a live optimization, raw render gradient, physics gradient, their cosine, accepted
  control step, `-g_render dot delta(x,F)`, and the matched render-off result are logged.

The first two are CPU regression tests. The 20k render-on and matched render-off runs are
the final causal comparison. `g_share` alone is not evidence because norm balancing makes
its nominal value largely algebraic.

## VBD / PD / Gauss-Seidel assessment

Vertex Block Descent minimizes the complete implicit-Euler variational energy by local
updates over all dynamic blocks. Updating only visible MPM surface particles after an
explicit rollout is not the same algorithm: it leaves `v`, APIC `C`, and interior state
inconsistent and can introduce unbalanced impulse/torque.

Two integrations are mechanically defensible:

1. Replace the explicit step with a full implicit MPM variational solve. All active grid
   nodes remain unknowns; the surface render term contributes only to the relevant right-
   hand side. Velocities are then derived from the accepted position solve, and the whole
   solve is differentiated implicitly or unrolled to convergence.
2. Keep the present simulator and use a symmetric material/grid operator only as a
   preconditioner on `dL/ddFc`. A candidate direction must satisfy `g dot p < 0` and pass
   the unchanged MPM rollout and line search. Symmetric GS or PCG is preferable to a
   directed kNN sweep.

The first is a research-scale integrator replacement. The second is an engineering
experiment, but current H1 results say there is no reason to enable it yet.

## Conservation caveat

Surface masking and support opacity do not alter MPM mass or momentum. The current default
simulator nevertheless contains global velocity drag (`drag=0.9`, applied as
`1-dt*drag`), so the production trajectory is dissipative and does **not** claim exact
linear-momentum conservation. A strict conservation experiment should use `drag=0`, no
floor/clamp, and objective symmetric strain-rate viscosity if settling is required; it
needs its own stability/quality ablation rather than a silent default change.

## Recommended papers

- [A Moving Least Squares Material Point Method with Displacement Discontinuity and
  Two-Way Rigid Body Coupling](https://yuanming.taichi.graphics/publication/2018-mlsmpm/)
  — the MLS-MPM weak-form and APIC-compatible discretisation underlying the solver.
- [An Angular Momentum Conserving Affine-Particle-In-Cell Method](https://arxiv.org/abs/1603.06188)
  — the correct reference for particle/grid transfer and momentum claims.
- [ChainQueen: A Real-Time Differentiable Physical Simulator for Soft Robotics](https://cdfg.mit.edu/assets/files/chain_queen_0.pdf)
  — differentiating an MLS-MPM trajectory for control/design.
- [Projective Dynamics: Fusing Constraint Projections for Fast Simulation](https://users.cs.utah.edu/~ladislav/bouaziz14projective/bouaziz14projective.html)
  — local/global implicit Euler and continuum-derived projective potentials.
- [Vertex Block Descent](https://graphics.cs.utah.edu/research/projects/vbd/vbd-siggraph2024.pdf)
  — energy-decreasing vertex-level Gauss-Seidel for a complete implicit variational solve.
- [DiffPD: Differentiable Projective Dynamics](https://diffpd.csail.mit.edu/)
  — efficient differentiation through a PD solve and the importance of converged implicit
  state for an unbiased adjoint.
- [SuGaR: Surface-Aligned Gaussian Splatting](https://openaccess.thecvf.com/content/CVPR2024/papers/Guedon_SuGaR_Surface-Aligned_Gaussian_Splatting_for_Efficient_3D_Mesh_Reconstruction_and_CVPR_2024_paper.pdf)
  — surface alignment and binding Gaussians to an editable surface.
- [2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://surfsplatting.github.io/)
  — oriented planar Gaussian disks; the strongest next step if isotropic rest Gaussians
  still look like blebs after the 20k/small-sigma change.

