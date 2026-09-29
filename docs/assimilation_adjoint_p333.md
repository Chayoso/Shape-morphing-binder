# P333: derivative of actual elastic assimilation

Status: opt-in primitive and frozen old/new-pin composition implemented; 46 CPU
checks pass. Initial CUDA run passes 18 parity/derivative/subset cases but fails
both graph-capture cases at the host-synchronizing inverse. Capture correction
uses the same inverse through inv_ex with a device status assertion, and uses
torch.where for the skipped-row increment. CUDA rerun pending.
No optimizer, runner or existing withdrawal default uses this derivative yet.

The next coast must respond to the plastic state the runner really commits.
`plasticity/assimilation_adjoint.py` differentiates the existing elastic
assimilation order: Fe=F inv(Fp), right stretch power, optional isochoric
increment normalization, multiplication by incoming Fp, cumulative singular
clamp, then optional fixed-target log-band projection. Physical F enters this
map; F+dFc does not. Growth, consensus, pin yield/release/follow/KKT and successor
layer/bond/reference preparation remain outside this primitive.

Forward float32 uses the existing Torch algebra. CPU production uses NumPy;
actual pin admission also subsets newly pinned particles before calling the
legacy function, whose SVD backend changes below 20000 rows. Thus primitive
large-CUDA equality and composed-subset numerical parity are separate gates.
Float64 exists for independent derivative checks, not a new production mode.

The increment pullback differentiates the symmetric right stretch using
pairwise singular sums and spectral divided differences. Isochoric diagonal
terms include all three singular values. The cumulative U diag(q) V-transpose
pullback retains both the symmetric divided differences and the skew factor
(q_i+q_j)/(s_i+s_j), including incoming Fp rotation. Repeated singular values
use their analytic smooth limits without perturbing the forward spectrum.

For isochoric cumulative projection, the pullback applies, in reverse order,
the exponential, the fixed-active-set projector diag(f)-f f^T/sum(f), the log,
and the FIRST singular clamp. The determinant target retains the existing
feasible-range clamp. Differentiating bisection decisions or omitting the first
clamp would give a different derivative. At an exact floor/clamp/health boundary,
no unique derivative is claimed; finite-difference tests stay within branches.
Only first derivatives are supported.

Invalid elastic determinant sets the increment to identity but still applies
the cumulative projection. Eta<=0 bypasses both, as in production. The pin
wrapper uses disjoint frozen masks. `settle_pin_assim=True` restores incoming
Fp on old pins and performs the second eta=1/noniso assimilation on new pins,
consuming the first call's result. False does neither pin-specific operation.
This wrapper maps Fp only; it does not implement v/C zeroing or pin admission.

Tests compare the existing forward, independent identity/isotropic/rigid and
twofold-spectrum derivatives, noncommuting F/Fp directional differences, active
bands, skipped increments, singular floor, old/new/free pin composition and
repeated backward calls. The CUDA gate additionally checks both subset backend
sizes, actual GPU graph replay and absence of explicit numerical host copies.

Next: connect a gradient-bearing coast Fp and v/C pin projection to the actual
controlled-head boundary, preserving direct x/v/C/F/Fg paths. Validate tiny
head-assimilation-coast finite differences with detached-Fp/C negative controls
before any production candidate search. A conditional derivative with frozen
successor preparation is not a complete derivative of preparation policies.

Rendering influence: unchanged; these are algebra/derivative tests without a
morph, render objective or candidate adoption. No new render influence number,
natural-rest, hole-removal or 4K appearance claim follows.
