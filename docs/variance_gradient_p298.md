# P298: same-prepared-state variance and render-gradient audit

Status: bounded CUDA observation completed; no physical-policy promotion.
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

## Same-state CUDA result, September 28

Frozen source aa81ea5 ran on hyde06 GPU0 in420.624s. All24 attempted windows
were outer accepted and all state guards were zero. The observer ran only at
the first inner iteration of attempt24, with the original300k/T20/dt1/240,
dx0.3062907543956724wu/loss36^3 recipe. No alternative optimizer step or raw
trajectory archive was produced. Its baseline lambda reconstructs the actual
window lambda exactly:0.036338788168254914. The alternative weight below is
ONE cloned update from the same baseline EMA/cap state, not P297's geometric
policy history. Neither cap binds.

| Same prepared forward, first gradient of W24 | Physical variance | Geometric variance |
|---|---:|---:|
| Variance((wu/s)^2) |0.000374717143|0.035206709057|
| Weighted variance term |0.000002270782|0.000213352300|
| Joint physics leaf-gradient norm |0.000032113390|0.000165924842|
| Updated render lambda |0.036338788|0.072669826|

The common effective variance weight is0.00605998987556. The independently
expected versus measured scalar-core delta differs by5.08e-11. Changing the
observable multiplies this state's variance penalty by about94 and the physics
gradient norm by5.17; its direction rotates71.06 degrees. These are norms in
the optimizer's mixed leaf coordinates, not physical force or particle motion.
Physical-gradient changes occur in stress, body and surface-u leaves; their
L2 differences are3.53417e-5,1.26321e-4 and8.88707e-5 respectively.

| Ordered pre-Adam direction change | L2 norm |
|---|---:|
| Substitute the physics variance gradient |1.58442776e-4|
| Update PCGrad at fixed baseline lambda |3.34111276e-6|
| Update lambda from the cloned balancer |1.99999305e-5|
| Total direction difference |1.60797799e-4|

Both physics directions conflict with the same post-smoothing render gradient.
Its projected direction rotates9.61 degrees. Lambda doubles, but the direct
regularizer substitution is the largest local pre-Adam change. The ordered
signed projections onto the total difference are0.977069/0.002991/0.019940;
they are not independent causal shares of fit loss, displacement or energy.
Vector closure residual has L2 3.31e-12. The final composite rotates65.77 degrees
relative to the baseline composite before Adam, clipping and line search.

The repeated physical/geometric adjoints differ by L2 8.30e-10/4.78e-9. The
observable-change difference is about33117 times the larger observed repeat
difference. These two repeats are descriptive, not a confidence interval or a
new tolerance. They neither calibrate render-gradient variability nor clear
P297's strict failed primitive comparisons.

This result does not support blaming lambda alone for P297's fit regression or
selecting a render-weight correction. It establishes a large direct change to
the regularizing direction at this prepared state. No alternative state was
advanced, so it does not establish which change caused the full-policy fit or
coverage loss. An unchanged nominal weight does not preserve effective strength
when replacing momentum variance by positional-path variance. No new gain,
default, frozen-particle claim or rendered quality claim is selected here.

Evidence: local output/p298/gradient24/{protocol,gradient_audit}.json;
server /data/relcfd/chayo/physmorph_v2/work/p298/gradient24. Result SHA256:
e6cd713f7607f5ef909633d38525c4e210b32406c0b890041ef1c29a1654368c.
Protocol SHA256:b0bd3708f49dab53d815769954ada7e2be66b54ca0bfdd6c69fd906d1e8a9473.
