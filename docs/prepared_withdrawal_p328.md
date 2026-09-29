# P328: prepared body-control head and withdrawal

P326 provides the joint derivative for expanded controls. P327 does not establish
a rest or hole remedy: its production pair differs before either arm records a
time-varying fragment mask, and its raw quality/motion changes are mixed.
The next small implementation connects the joint derivative to the exact prepared
reduced body basis used by the live optimizer. This is a private capability, not
a new production objective, accepted replacement or deliverable.

## Scope

Construct a private `FrozenWithdrawalWindow` from a live `FrozenBodyWindow` in
the existing read-only checkpoint callback. Own the prepared spec, node indices,
weights, gate, fixed expanded stress and surface-u values. Vary the existing two
body modes through the same basis and original joint coefficient radius. Preserve
the original T normalization. Do not infer a body field from endpoint positions.
The adapter must expire when its source callback model closes.

Use P326's `WithdrawalAdjoint`: the head and zero-future-control coast retain all
x/v/C/F/Fg boundary dependencies. Return the ordinary head packet needed by
`evaluate_merit`, including candidate body energy, plus explicitly labeled coast
state. Own the head terminal C and complete head F/Fg sequence from that very
forward before any later replay. Check all head/coast state and pinned paths.
No x-only promotion, post-hoc velocity patch or candidate replay masquerading as
the selected state is allowed.

The coast is **pre-assimilation**, with the original Fp, pin set and prepared
layer/bond policy. It is not the actual next optimizer input. Closing the adapter
or replacing its forward must invalidate old gradient contexts. The original
optimizer controls, Adam state, accepted buffers and prepared loss caches remain
untouched.

## Capability gates before a proposal

- Exact CPU head/basis/energy parity against the existing private body rollout;
  nontrivial incoming v/C/F/Fp, two body modes, pins, layer and bonds.
- Owned state after another forward; source mutation isolation; callback lease
  expiration; whole head/coast health and exact pin constraints.
- Coast directional derivatives through both reduced modes, including directions
  that pass through the head boundary; no zero-gradient-vacuous check.
- Actual CUDA execution using aligned Torch/CuPy/Warp streams, captured/ordinary
  parity and no numerical array download. CPU success alone is insufficient.
- Independent refutation of the implementation and receipts.

These gates establish the prepared derivative and ownership, not useful control.
No new weights, stopping threshold or pin rule are authorized by their passage.

## Implementation and CPU verification

The private adapter is implemented in `pipeline/frozen_withdrawal_window.py`.
It binds the construction stream before any input clone and checks the same
Torch/CuPy/Warp execution context before evaluation reductions. Owner/self
expiration and successful or failed replacement forwards invalidate both joint
outputs and the separate Torch body-energy graph. Owned F/Fg/C sequences and
terminal C come from the selected forward. In both segments, pinned positions
must equal the original anchors and post-step pinned V/C must be exactly zero;
the incoming time-zero V/C is not asserted zero. Scalar health checks remain
host decisions; this is not a fully host-free implementation.

All26 CPU cases pass. The numerical fixture uses N27,T20,dt.002,dx.5,16^3,
nonzero incoming v/C/F/Fp, two body modes, nonuniform gates, layer/bonds and
pins. For the smooth no-slip contact branch, future-only directional derivatives
at radii.001/.0005 are displacement AD-.0029226254 versus FD
-.0029172273/-.0029218022, terminal AD-.0003424400 versus
-.0003381474/-.0003419275. The gate requires derivative magnitude>5e-5 with
2% relative/5e-6 absolute tolerance. Exact head/body-energy parity, an independent
full-state coast, owned snapshots, invalid coast/pinned-state negatives, stale
leases and actual callback original-merit closure also pass. This is a capability
test, not a production N300k result.

The registered CUDA suite has four cases: both reduced-mode finite differences
with ordinary/captured parity and no numerical array download; side-stream
autograd/lease behavior; early stream mismatch rejection before cloning or
finite checks. State parity is1e-5 relative/absolute, gradient parity2e-4/2e-5,
with unchanged CPU finite-difference gates.

Frozen4ab8510 passes allfour actual CUDA cases, zero skips, in2.17s on hyde06
GPU0 (`code_prepared_withdrawal1`, `p328_cuda1`). Displacement AD is
-.002922626595 versus FD-.002917099631/-.002921826004; terminal AD is
-.000342438531 versus FD-.000337917595/-.000341950670. Both original radii and
tolerances remain unchanged. The stream-preflight sentinel, no numerical array
download, captured/ordinary parity, worker-thread lifetime and owned-snapshot
checks pass. Full receipts are in `docs/evidence/p328`; no performance claim
for a production morph follows from this small numerical fixture.
Independent review matches allfour named XML cases, all source/receipt manifest
hashes and imported fixture bytes to the frozen server checkout, and recomputes
the four finite-difference comparisons. The receipt gate is closed; no additional
MPM or rendering run was used for that audit.

The complete returned FP32 state payload alone is about1.70GB at N300k,T20,
before trajectory/gradient storage and temporary stacks. No full-size runtime
or VRAM feasibility has yet been measured, and nothing is archived by default.

## Subsequent proposal and actual handoff gap

P324 shows that withdrawing control can improve fitting at one head and worsen
it at another. Coarse arrival alone therefore cannot select a blanket motion
penalty. A later bounded experiment must preserve one prepared head reference,
original merit, full raw shape/supply constraints and same material cohorts while
asking whether any feasible body-control direction reduces unwanted continuation.
It must keep useful fitting separate from rest; a lower coast speed is insufficient.

The existing callback has no adoption API. Candidate evaluation through the real
handoff still needs the runner's assimilation, old/new-pin handling, bond rebase
and layer preparation applied to the candidate's complete selected state. Frozen
pre-assimilation gradients may propose directions, but acceptance must not call
them exact post-commit derivatives. A differentiable actual handoff additionally
needs an Fp-map VJP, declared discrete branches and stable spectral derivatives.
Coupled subsequent optimization, ordinary rejection/rollback and full-horizon
quality gates remain mandatory before adopting a trajectory.

## Rendering influence

This adapter changes no rendering target, resolution, lambda or gradient policy.
The existing callback's `evaluate_merit` must evaluate the candidate's head render
loss at its frozen reference and lambda alongside every original physics term.
Report head render change, original merit, render-direction telemetry and raw
motion/shape separately. A passive coast invokes no rendering loss. Neither its
derivative nor the current 18-view 64px guidance certifies the exported 4K splat
footprint; `gaussian_footprint_followup.md` remains a separate unresolved item.
