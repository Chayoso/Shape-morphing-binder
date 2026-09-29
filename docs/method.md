# Method

This document describes the settled-transport method as implemented on this branch, the path its gradients take,
what it conserves, how it was verified, and the tools used to measure it. Results are in
[experiments.md](experiments.md).

## 1. Forward model

The body is `N` particles of equal mass (the dynamics mass per particle is `mass_ref_n / N`, so the body's mass does
not depend on `N`). One MLS-MPM step with quadratic B-splines on a grid of cell `dx` (set from the shape: `dx =`
source bounding-box diagonal / `cell_diag`, 26 by default) transfers mass and APIC momentum to the grid, adds the
stress term, damps momentum by `(1 - dt·drag)`, and transfers back. `dt = 1/240`.

There are two controls per window:

- `dFc`, a per-particle increment of the deformation gradient. The stress is the fixed-corotated first
  Piola–Kirchhoff stress of `Fe = (F + dFc) Fp⁻¹`; the control also enters the next `F`.
- `u`, a normal offset of each outer-layer particle, applied as a position update spread over the driven steps and
  gated to where the remaining transport is within one cell (`--layer_gate_ot`).

The outer layer is also relaxed toward the local plane of its neighbours every step (`--layer_relax`), and material
bonds keep detached fragments moving with their source neighbours (`--bonds`).

## 2. The settled window

A window rolls `2T` steps (`T = 20`): `T` driven steps with `dFc` and `u`, then `T` released steps with both set to
zero while the physics, the layer relaxation and the bonds keep running. Every loss is evaluated on the released
end state. A control that only looks good while it is being applied, and springs back once released, is scored on
where the body actually comes to rest.

## 3. Objective

- **Transport.** A debiased Sinkhorn divergence between the body's mass on the loss grid (CIC rasterisation) and the
  FIXED target's, with blur `ε = (max(1, ot_eps_cells) · loss cell)²`. The squared-Euclidean cost is separable by
  axis, so each Sinkhorn sweep is three one-dimensional log-sum-exp passes (a Warp kernel on CUDA); the solve anneals
  `ε` from the grid diameter down and must converge to the tolerance, otherwise the trial is inadmissible. Its value
  plus the residual drift `mean_p |T·dt·v_p|²` is the transport state energy `E`. `E` is scaled once, at the source,
  so its gradient norm equals that of the density loss.
- **Local support.** A kernel log-density at each particle over its 32 nearest neighbours, with a floor at half the
  target's median density; the penalty `B` is the mean squared shortfall below the floor. It enters as
  `E + E·wB / (E + wB)` (`w = support_weight`), so it can never exceed the remaining transport energy and vanishes
  as transport completes.
- **Rendering.** A multi-view silhouette loss (particles splatted with opacity `1 − e^{−k w}`) and a shading loss
  against a target rendered by the same forward operator (`pbr_target_mode = matched`).
- **Cleanup and regularisers.** Terms that pull stray particles back into the target band (re-evaluated on the
  current nearest target points for the delivery merit), the end kinetic energy,
  the velocity variance (driven phase: fluctuation around the mean; released phase: all motion), and control
  magnitude and smoothness penalties.

## 4. Gradient path

The physics loss `Lp` and the render loss `Lr` are differentiated separately through the same `2T`-step adjoint
(Warp tape, forward and adjoint captured as CUDA graphs), back to `dFc` and `u`. The Sinkhorn term is not
differentiated through its iterations: by the envelope theorem its gradient with respect to the grid mass is the
difference of the converged potentials, which then passes through the CIC weights to the particles. The rotation in
the stress uses a polar-factor adjoint that stays finite at repeated singular values.

The render gradient loses any component that opposes the physics gradient (one-sided PCGrad) and is weighted by
`λ`, set at the first iteration of the first window so that `λ·|g_r| = 0.5·|g_p|` and then held fixed for the loss
resolution. `--render_weight_scale s` multiplies `λ`; `s = 0` is the render-off twin. The update is
`g = g_p + λ·g_r + g_cleanup`; a backtracking line search accepts a step only if the full objective decreases and the
state stays valid.

## 5. Acceptance and delivery

Each window's candidate is scored by one merit: the full objective with the cleanup term re-evaluated on the current
geometry. A candidate that raises the merit by more than the brake margin is rejected and the state is kept. The run
ends at the best commit after `reject_stop` consecutive rejections (3), or when the merit has not improved by `tol`
over `patience` windows (5). A coarse-to-fine rebuild of the render targets starts a fresh calibration epoch.

## 6. What is conserved

Stress cannot change total momentum: in the transfer to the grid the stress enters as `G·(x_i − x_p)` and
`Σ_i w_ip (x_i − x_p) = 0` for the quadratic B-spline, so the forces sum to zero whatever `dFc` is. The Kirchhoff
stress `Pe·Feᵀ` is symmetric, so angular momentum is kept as well. The exceptions are known and bounded: drag decays
momentum (it never creates it), fragments take their neighbours' mean velocity, and the layer relaxation and `u` move
outer-layer positions without a velocity.

Measured on a 300k rollout: a clip-sized `dFc` step keeps `|P| / Σm|v|` at 3·10⁻⁶ to 2·10⁻⁴ and
`|L| / Σm|r||v|` at 3·10⁻⁴ to 8·10⁻³; interior particles move by velocity only (position edits ≤ 6·10⁻⁷); the outer
layer (4 % of the particles) moves 11–22 % by position edits, almost all of it the passive relaxation. Over a whole
morph the centre of mass moves ≤ 0.02 particle spacing (source and target are both centred).

## 7. Verification

- The branch's test suite passes (207 passed, 2 skipped on hyde06).
- The gradient path and the line-search path give the same objective values to seven digits at the same controls.
- Central finite differences through the line-search rollout agree with autograd within 3 % for the physics,
  render and cleanup terms along their descent directions, at windows 1 and 20 of a 300k run
  (`scripts/probes/settled/gradcheck_patch.py`).
- The released half of the control sequence is exactly zero.

## 8. Rendering and measurement

`scripts/render_splat_photoreal.py` renders the 4K deliverable: each particle is a disc-shaped Gaussian whose normal
comes from the smoothed density gradient and whose radius is the target spacing scaled by the local 8th-neighbour
distance (clamped to 1–4×), shaded per pixel as a uniform ceramic material in a studio setup. The covariance does not use `F`. Archives without
pins are rendered from their live state every frame.

Probes in `scripts/probes/settled/` (run on the server with `PYTHONPATH` set to the checkout):

| Probe | Measures |
|---|---|
| `end_probe.py` | End frame: pieces off the body, the ear tip's particle count, target under-fill. |
| `region_error.py` | Distance from the target surface to the nearest particle, by region; share beyond 1.5 spacings. |
| `hollow_frames.py` | Density of the ear region over the target's at fixed frames (thin, fast-moving material in transit). |
| `strip_probe.py` | Progress toward the end position by depth below the surface; late surface motion. |
| `ear_slab.py` | The ear's fill and thickness per height slab over time. |
| `window_reversal.py` | Window-to-window reversal of the outer layer's motion (surface oscillation). |
| `com_probe.py` | Centre-of-mass path over the whole morph. |
| `layer_probe.py` | Delivered motion by particle layer for replayed windows. |
| `video_flicker.py` | Frame-to-frame change of a rendered video. |
| `gate_table.py` | Per-target comparison of two arms against the adoption gates. |
| `gradcheck_patch.py`, `gc_read.py`, `gc_mom.py` | Finite-difference check, render share per control and momentum budget of one window. |

## 9. Open questions

- The head and body relief is fitted less closely than by the earlier line (7.8 % against 10.4–11.7 % of the target
  surface farther than 1.5 spacings); the transport blur is one loss cell.
- The ear tip holds 11–12 reference particles of the 13 the gate asks for.
- `support_weight = 8` and `ot_iters = 1600` are validation values, not derived from the discretisation.
- `u` is driven almost entirely by the render gradient and moves positions without momentum.
- After convergence a rejected candidate can repeat identically each window until the patience runs out.
