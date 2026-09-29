# Derivative boundary after P332

P332 selects a complete same-forward result before the ordinary runner handoff.
It does not differentiate that handoff, and no rejected P331 control is an
admissible starting candidate. The next bounded task is forward/adjoint parity
for the actual plastic-state update, before another candidate quality search.
P333 implements the stable assimilation primitive and frozen pin composition;
see `assimilation_adjoint_p333.md` for its distinct CPU/CUDA gates. The joint
trajectory bridge is now implemented in `post_assimilation_adjoint_p334.md`,
with CPU verification and pending actual CUDA gates. The complete successor
preparation derivative below remains pending.

This first scope excludes `w_grow > 0` and `assim_consensus`, and has no pin
follow, yield, KKT or release branch. Those policies have different handoff maps
and must not be inferred from the two operations below.

Let A denote the exact existing `assimilate_elastic`, F the terminal physical
deformation, and P0 the incoming Fp. The actual supported handoff composes:

    P1 = where(old_pins, P0, A(F, P0, eta=cfg.assim, iso=cfg.assim_iso))
    Pnext = where(new_pins, A(F, P1, eta=1, iso=False), P1)

The old-pin exception above applies under the production `settle_pin_assim`
policy; use the runner's actual branch/masks. The new-pin operation consumes P1,
not P0. Cumulative singular clamps prevent collapsing those two operations.
Positions and physical F pass unchanged; next-pin v/C are projected to zero.
Assimilation consumes physical F, not F+dFc.

Preserve the exact existing forward: elastic inverse/product, singular floor,
stretch power, optional isochoric increment normalization, multiplication by
incoming Fp, cumulative singular clamp, then the log-band/sum projection.
Both CPU and the actual CUDA branch for N>=20000 need forward parity. Existing
semantics should not silently change to make the derivative easier.

The current withdrawal bridge aliases coast Fp to head Fp; Trajectory.Fp is not
a differentiable leaf and the exposed backward returns controls only. A new
opt-in post-assimilation joint mode therefore needs independent coast Fp with
gradient storage and an explicit boundary pullback into the head. Preserve
direct x/v/C/F/Fg dependence as well. Detached C snapshots cannot carry the
head-C-to-coast path if the implementation splits at Torch. Keep the existing
pre-assimilation bridge unchanged.

Investigate a custom spectral VJP with analytic repeated-spectrum limits,
using the existing polar adjoint as a precedent. Generic differentiation of
SVD factors at identity is not an acceptable untested substitute. Differentiate
the fixed-active-set log-projection map, including the preceding clamp and
subsequent exponential, rather than the bisection iteration decisions. No
singular-value jitter or unclamped surrogate should alter the forward.

First gates: identity/rigid/isotropic and twofold spectra; noncommuting F/Fp
directional finite differences and adjoint-dot identity; active-band cases
with fixed margins; old/new/free pin composition; then a tiny controlled
head-to-assimilation-to-coast test with detached-Fp and detached-C negative
controls, repeated/missing seeds and aligned CUDA capture. For an exact-real
active-band example with eta=1, Fp=I and isochoric=True, diag(12,4,1/48)
projects to diag(2.5,2,.2) under [.2,5]. Its derivative must still include
the increment normalization even though the base determinant is one.

This is a conditional derivative: hold actual branch disposition, pin admission,
neighbor/fragment IDs and clamp active sets fixed, and verify perturbations
stay in those branches. Initially freezing the observed successor's layer
normals/weights/gates, bond rest and refreshed references gives only a partial
preparation derivative. Bond rest and layer geometry depend continuously on
positions; fixed neighbor IDs alone do not remove those dependencies.

Keep original complete head merit, lambda, reference and material cohorts
explicit. Pin-imposed zero velocity is not natural rest; newly pinned Fp can
still affect neighboring free material. Independent raw quality, real ordinary
continuation, gallery and 4K appearance gates remain required for adoption.
