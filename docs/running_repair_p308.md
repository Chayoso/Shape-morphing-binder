# P308 draft: motion reduction with data restoration in the displacement mode

P307 restored endpoint MSE but did not satisfy the movement and local-supply
gates. This is the next bounded diagnostic, not an implemented production
policy. Its code and exact active-set implementation still require review.

Use a fresh real W20 callback at the existing N300k,T20,dt1/240,
dx.3062907543956724wu,loss36^3,budget8 raw/no-PIC/no-shift recipe. Preserve
P307's failed archive/C-repeat records; do not admit that restart. Generate the
fresh callback's own terminal05. Freeze terminal coefficients, stress/u,
initial state, reference, lambda and both window-start cohorts throughout.

Let R be the mean squared position-derived velocity across all20 steps and
the fixed start-arrived-free cohort, including x0->x1. Optimize displacement
coefficients to reduce R. Do not substitute endpoint distance or stored v_T.
Measure stored/geometric terminal speeds separately, since fixed terminal
coefficients do not fix the nonlinear terminal response.

At the current displacement coefficients, compute gradients of R, prepared
volume and prepared render. With baseline loss ceilings from the three
original baseline repeats, use affine restoration constraints:

```
g_volume dot delta <= baseline_volume - current_volume
g_render dot delta <= baseline_render - current_render
```

The existing homogeneous halfspace projection is insufficient: terminal05
already violates volume. A descent direction that merely does not increase
that loss does not restore its deficit. Keep the original fixed reference and
lambda; no new weighted surrogate or kinetic coefficient is introduced.

Proposed bounded subproblem: minimize the linearized R over these two
halfspaces and a Euclidean trust ball whose RMS radius is the last ordinary
accepted displacement update. Enumerate the four possible active sets on GPU;
use a minimum-norm affine point and the remaining ball radius in its nullspace.
Check rank consistency and every inequality. For each of at most10 halvings,
solve again at the smaller radius: scaling an already feasible affine step can
destroy restoration. If no feasible linear step exists, record that result.

Apply the existing residual joint coefficient bound with terminal05 fixed.
Recheck the actual final projected displacement step against the affine
constraints; a pre-projection certificate is insufficient. Evaluate the full
MPM trajectory, then require valid raw state, exact pins, actual volume/render
within the baseline ceilings, and R reduction resolved beyond the existing
repeat/float32 floor. Record failed linear/projection/nonlinear attempts. Use
at most4 accepted updates, with no fallback that weakens a constraint.

All final candidates still face unchanged P306 gates: both cohorts' net/step/
path motion; resolved stored/geometric terminal braking; independent raw
silhouette/coverage/Chamfer/tip/local-density measurements. Density counters
remain validation metrics, not objectives. This subproblem does not guarantee
their discrete gates. A failure is confined to this search and subspace.

Retain full accepted X/V/F/C/coefficient evidence, actual render components,
source/input hashes, private/accepted X/V/F/data closure, and exact production
state/Adam isolation after the callback. No candidate commit. Only a candidate
passing all gates can justify original-total-merit reconstruction and actual
coupled continuation. That continuation must carry the full state and normal
assimilation; it remains necessary for any persistent-rest claim.
