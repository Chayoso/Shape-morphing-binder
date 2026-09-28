# P298: same-prepared-state variance and render-gradient audit

Status: implemented with CPU validation; bounded CUDA observation pending.
P297 reduced late motion but failed fit and thin-region supply gates;
geometric_variance remains off. This diagnostic does not promote it or relax
the failed CUDA primitive comparisons recorded in geometric_variance_p297.md.

The full P297 policies reach different states and different adaptive render
weights. Their lambda ratio cannot explain the cause. At the first inner
iteration of baseline window24, evaluate both variance gradients on ONE live
prepared autograd graph: the same entering state, materials, pins, layer
neighbors, targets, transport plan, controls, raw trajectory and promoted
endpoint. Only the variance input to phys_core is substituted. The observer
requests the owned full-position output but leaves the production objective,
line search and state updates unchanged. The default path has no observer cost.

Use the original bunny300k, T20, dt1/240, dx0.3062907543956724wu, loss36^3,
eight inner iterations, shared PIC, no subcell shift, committed-state outer
render merit and original positive render balancing. Rerun the baseline through
window24; record actual history and require that the inspected window is outer
accepted. This state is from the new baseline run, not claimed to be bitwise
identical to P297's separately executed window24. No full raw archive is needed.

The callback receives owned detached leaf gradients before PCGrad and lambda:
physical-variance physics p, geometric-variance physics q, repeated p/q adjoints,
the same post-smoothing render gradient r, and unchanged transport gradient t.
For the existing one-sided render projection P, clone the exact pre-update
balancer twice. Obtain lambda_p from (p,P_p(r)) and lambda_q from (q,P_q(r)).
Do not reuse cross-state weights from P297 or refit a gain. Compare:

1. p + lambda_p P_p(r) + t: current baseline direction.
2. q + lambda_p P_p(r) + t: only physics direction changed.
3. q + lambda_p P_q(r) + t: also update the PCGrad reference at fixed lambda.
4. q + lambda_q P_q(r) + t: also update adaptive lambda from the cloned state.

This sequential vector decomposition is order-dependent and its components
need not be orthogonal. Component norms do not add and are not causal energy
shares. Record vector closure and signed projections; lambda_q also depends on
the changed projected-render norm, so it is not a separate exogenous cause.

Respect the production surface-u render-only override if enabled. Report joint
and per-leaf norms, cosines, differences and repeat variability. The intermediate
directions are algebraic counterfactuals, not accepted Adam steps; actual Adam
moments, RPROP scaling, clipping and line search remain outside this comparison.
The shared graph establishes prepared-reference identity by construction, not
by comparing copied CLI arguments. Repeats measure this diagnostic's backward
variability; they cannot convert P297's failed strict gates into passes.

No rest, motion, hole, fit or renderer improvement can follow from gradients
alone. A subsequent single-window solve needs its own preserved preparation,
controls and balancing contract and independent raw geometry/trajectory checks.

CPU validation passed15 observer cases,12 driver/algebra cases and17 existing
geometric-variance integration regressions. Observer tests include active body,
surface-u and layer-relaxation paths, both unit systems and render/off projection.
Mutating every owned callback gradient and the copied balancer-state dictionary
does not alter the tested production continuation. Driver integration at a
noninitial window verifies that lambda_p matches the actual production lambda
exactly; this equality is also required before publishing a valid CUDA record.
These small CPU cases do not establish CUDA neutrality or physical quality.
