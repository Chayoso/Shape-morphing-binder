# P326: differentiate a frozen pre-assimilation withdrawal

P324 found residual movement after the real handoff, including common-free IDs
that satisfied the coarse source arrival predicate. Some of that movement still
improved fitting. P326 supplies a derivative capability; it introduces no rest
penalty, new arrival rule or candidate adoption. Its review also found an inherited
fragment-activity lifetime bug; the correction below preserves fixed-control
forward physics but can change production optimizer gradients and learned paths.

`physmorph/mpm/withdrawal_adjoint.py` constructs two separate original-T
trajectories. The first retains its dFc, u and body controls; the second withdraws
future controls. Before recording the joint tape, the second trajectory's initial
x, v, APIC C, F and geometric Fg alias the first trajectory's terminal arrays.
Both keep the original source volumes, material parameters, original Fp and
prepared layer/bond/pin policies. Support gating keeps the resolved source count.
Head body pulses, u/T and layer relaxation are unchanged; this is not a single
2T trajectory with rescaled pulses. Stored F includes accumulated head control.

The derivative follows head controls through every shared full-state boundary
path to a future observable. Initial/material/configuration values are constants;
differentiable inputs there fail closed. Only dFc(T,N,3,3), optional u(N) and the
enabled body field(modes*N,3), all FP32, are accepted control leaves.

`WithdrawalAdjoint.apply` returns owned head x/F/v/Fg, head V/X sequences and
coast X/V/F/Fg sequences including coast time zero. F sequences are flattened
(T+1,N,9). Simultaneous seeds for head endpoint, last head sequence element and
coast time zero merge once. Every backward clears omitted seeds and all unique
gradient buffers. Reuse invalidates prior contexts before validation/input copy;
owned outputs survive later calls. CUDA requires `cuda_execution` and binds the
instance to its construction device/Torch stream. Torch, Warp and CuPy must
already share the same stream before construction and reuse, so an unsynchronized
switch after fixed-input creation is rejected. Aligned caller-selected streams
are supported; reuse on other streams is rejected.
Both uncaptured ordinary CUDA kernels and captured forward/backward graphs exist.

## Scope limit

This is **pre-assimilation**, with frozen Fp and prepared policies. It does not
differentiate actual postcommit assimilation, pin admission or re-preparation.
Calling two existing autograd bridges with detached boundary configuration would
lose those paths; the joint tape avoids that loss within the declared model only.
An actual postcommit adjoint still needs the assimilation VJP and explicit branch
policy. P324's similar cohort RMS does not validate a per-ID direction or gradient
proxy. No blanket withdrawal penalty is selected: suppressing useful fitting is
a measured risk. No hole, natural-rest or 4K quality gate is passed by this API.

## Verification contract and initial CPU finding

The nontrivial fixture uses N27,T20,dt.002,dx.5, a16^3 MPM grid, nonzero incoming
v/C/F/Fp, two body modes, layer relaxation/u, bonds, support gating and two pins.
It is a numerical test fixture, not the N300k production discretization.

Require existing-head forward/gradient parity; an independent zero-control coast
from the complete head state; a test that dropping C changes continuation; fixed
policy/owned-input checks; simultaneous/zero/missing/repeated seeds and stale
context rejection. Future-only directional finite differences exercise all three
control channels using a deterministic direction independent of the gradient.
The scalar is evaluated in FP64 over an FP32 rollout, with2%/5e-4 tolerance.

The first CPU body-control stencil1e-3/5e-4 failed with separating pin contact:
AD=-.01662637549; finite differences=-.01946646682/-.01762901248. Refinement to
1e-4/5e-5 gives-.01676401419/-.01660624267;2e-5 also gives-.01660732291.
With the separating collider disabled while retaining the same pins, the larger
stencil passes: AD=-.00769434972, FD=-.00773136388/-.00772106531.
This supports a local contact/nonlinearity explanation, not proof of the exact
active-set changes. Preserve the large-stencil no-slip test and the refined
contact test; neither the failure nor the tolerance is discarded. Stress/u keep
the original1e-3/5e-4 bracket. A separate boundary-seed test uses binary-exact
coefficients to avoid comparing two different decimal-rounding expressions.

Before the first server run, CUDA gates are fixed: ordinary/captured state parity
rtol/atol1e-5; gradient parity rtol2e-4,atol2e-5; the same two FD radii and
tolerances as above; no Torch/Warp/CuPy numerical array download; side-stream
graph execution, cross-stream rejection and seed/reuse/ownership checks. These
are derivative/operator gates, not performance or physical quality evidence.

## Correcting overwritten fragment activity

The original `Trajectory.frag_step` was one scratch array overwritten at every
step. Both P2G and the bond position update read it again during reverse. If
activity changed over time, earlier reverse steps could use the final mask.
The adjoint now retains one non-differentiable mask per step; forward-only
evaluation still uses one scratch buffer. `frag_step` remains the most recently
evaluated mask for diagnostics. No kernel, threshold or physical policy changes.
Memory overhead is(T-1)*N*4bytes per differentiable bonded trajectory:22.8MB
at N300k,T20, or45.6MB for two such segments. Existing optimizers also receive
the corrected gradients; old learned trajectories are not asserted equivalent.

An explicit N24,T8,dt.005,dx.5,16^3-grid fixture starts with one displaced
particle. Its dynamic fragment mask is1,1,1,0,0,0,0,0. Each perturbed trajectory
recomputes its own masks and stays on this temporal branch. For a deterministic
body-field direction, corrected CPU AD=-.003190334, FD=-.003187978/-.003187565
at1e-3/5e-4; reconstructing the old shared scratch gives AD=+.002087397.
All six forward outputs remain bit-identical in that CPU witness. A separate
no-grad reference verifies the same masks and endpoint with one scratch buffer.
This proves a gradient error in this fixture, not its share of the real morph's
holes, residual flow or oscillation. Actual CUDA and production comparisons are
required before claiming symptom improvement.

## Rendering influence

No rendering loss or optimizer calls the new withdrawal capability by default. Its synthetic
future-state loss is only a derivative test, so it has no measured effect on the
retained renders. The fragment correction can affect gradients from either
physics or rendering; it is not a renderer-weight change. P324's originating18-view64px
guidance remains the latest reported optimization influence; it does not supervise
the actual exported4K covariance. See `gaussian_footprint_followup.md` for the
separate F/stretch/footprint limitation.
