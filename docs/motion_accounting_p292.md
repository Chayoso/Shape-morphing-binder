# P292: where the saved particle motion comes from

The accepted-trajectory accounting identifies a large opposing PIC correction at
window commit. Late arrived, unpinned material moves during the rollout, then PIC
removes much of that displacement in one saved transition. Surface relaxation and
the direct surface channel also move positions, but their measured magnitudes are
smaller. This supports testing a complete PIC-off candidate; it does not establish
that PIC removal alone preserves final shape or solves every residual motion.

## Run and scope

`accounting60` runs the mixed body + stress + surface-u bunny recipe on hyde06:
300000 particles, T=20, dt=1/240, dx=0.3062907544 wu, loss grid 36 cubed, eight
inner iterations, animations=300 and cap=60. The render channel remains enabled,
body RPROP is off, and both commit PIC and subgrid shifting are on. Runtime is
650.59 s with 40 accepted commits in 41 attempts; all recorded safety guards are
zero. Attempt 40 was not accepted and is excluded from the summaries below.

Numerical source SHA-256:
`9373ed5f17a0b55ddf50deeb1d5a29726637cdf7cc818746943355ac2f863d4d`.
The diagnostic reads owned copies of final accepted rollout buffers; every used
record says `commit_source=accepted_buffer`. It does not replay an optimizer trial
or recover a rejected trial's state. CPU parity checks of the diagnostic are
recorded separately by the implementation review.

This is a diagnostic trajectory, not a matched endpoint-quality arm. It accepts
40 commits, whereas the earlier noninstrumented baseline accepts 32. GPU atomic
ordering/runtime effects and the diagnostic source revision confound an endpoint
comparison. The record alone does not identify why that difference arose. We use
the within-trajectory decomposition, not the difference in run length, as evidence.

The **late interval is accepted commits 31–40**, attempts 31–39 and 41. At each
window, `arrived_free` means unpinned at its start and arrived at **both** its start
and promoted endpoint against that window's frozen full plan. Cohort sizes are
51591, 50585, 49798, 49141, 48575, 46521, 44304, 40917, 38179 and 35765.
This is a changing per-window population, not the fixed 1374-ID sparse boundary
cohort in [the phase audit](quality_comparison_p292.md). It includes interior
material as well as boundary material and does not measure a shared normal basis.

## Exact accounting and interpretation

Each rollout displacement is decomposed as

`dx = dt*v + bond_residual + surface_u + layer_residual`.

`dt*v` uses stored MPM velocity after that actual step. It includes the effects of
the entire MPM force/control model; it does not separate body force from stress.
The residual after advection includes bond-position updates and arithmetic
roundoff. The residual after the known surface-u displacement includes layer
relaxation and arithmetic roundoff. The code deliberately does not label every
residual bit as a physical bond or relaxation force.

The full promoted window displacement adds `commit_other`, `commit_pic`, and
`commit_shift` to the vector sum of the rollout components. `commit_other` is zero
here. The operator called commit PIC is the current **XPIC(5) displacement filter**
with cubic stencils fixed at window-start positions; it changes committed positions,
not stored velocity, C or F. Across all 40 accepted windows, maximum vector reconstruction errors are
1.86e-9 wu per substep and 1.26e-7 wu per window. In the late interval they are
4.80e-10 and 3.41e-8 wu. All pinned-at-start component and net displacements are
exactly zero in these summaries.

For a window component vector `c_k` and actual promoted displacement `D`, its
signed coefficient is `sum_p dot(c_k[p], D[p]) / sum_p |D[p]|^2` over the cohort.
The step coefficient uses the analogous sums over both particles and substeps.
Negative coefficients mean opposition to the net displacement. Values above one
are possible because components cancel. They are not fractions of rendered motion,
causal influence percentages, energies or gradient shares. Component RMS values
also do not add to the net RMS.

All late-interval figures below are **medians of ten window summaries**, each on
its own arrived-free cohort. The median coefficients need not sum to one. Within
individual accepted windows, the sum of signed window coefficients differs from
one by at most 2.24e-7 in this run.

## Late arrived-free motion

For per-step RMS, first take the root mean square of the twenty reported per-phase
RMS values in each window, then the median across the ten windows.

| Rollout component, before commit | Per-step RMS, wu | Signed step coefficient |
| --- | ---: | ---: |
| MPM advection, dt*v | 0.0005892 | +0.84899 |
| Bond/arithmetic residual | 6.03e-8 | +8.66e-9 |
| Direct surface-u | 0.0001152 | +0.02278 |
| Layer/arithmetic residual | 0.0002519 | +0.12761 |
| Actual rollout step, before commit | 0.0006335 | reference |

MPM advection is the largest pre-commit term by these diagnostics. This rules out
attributing the entire visible particle motion to Gaussian attribute fitting or
the direct surface-u channel. Rendering loss remains part of how the controls were
optimized, so it does not rule out an indirect rendering-gradient influence.

| Full window component | RMS of its summed vector, wu | Signed window coefficient |
| --- | ---: | ---: |
| MPM advection | 0.010867 | +1.45467 |
| Bond/arithmetic residual | 2.70e-7 | +2.07e-9 |
| Direct surface-u | 0.002304 | +0.01172 |
| Layer/arithmetic residual | 0.004865 | +0.01565 |
| Other commit changes | 0 | 0 |
| Commit PIC | 0.009345 | -0.50024 |
| Commit subgrid shift | 0.001861 | +0.05623 |
| Actual promoted net displacement | 0.004421 | reference |

PIC is comparable in magnitude to the accumulated MPM advection and strongly
opposes the final net displacement. The layer term has appreciable RMS but a small
projection on the final net movement; its magnitude must not be mistaken for a
large share of successful transport. The same warning applies to surface-u.

At the final accepted commit alone, the arrived-free net RMS is 0.003424 wu.
Advection RMS is 0.009979, PIC RMS 0.008710, layer residual RMS 0.004749, surface-u
RMS 0.001862 and shift RMS 0.001823 wu. Their signed coefficients are respectively
+1.50983, -0.62844, +0.01552, +0.01081 and +0.09228, with negligible bond residual.
This is direct evidence of cancellation within one accepted window, not a
comparison of unrelated medians or separate runs.

## What the velocity measurement does and does not show

The instrumented terminal geometric velocity is the **last actual rollout
displacement divided by dt, before any commit correction**. On late arrived-free
cohorts, the median of window mean speeds is 0.11813 wu/s for stored MPM velocity
and 0.12064 wu/s for geometric velocity. Their median p95 speeds are
0.27023/0.28157 wu/s, and median RMS vector difference is 0.04129 wu/s.

At the final accepted commit, these cohort means are 0.09695/0.09949 wu/s.
Meanwhile the all-particle stored mean is only 0.01156 wu/s because most particles
are pinned. That whole-cloud mean is not evidence that remaining free material is
still.

The telemetry in **this executed accounting60 run** does not include the exact geometric velocity of the final
promoted archive transition after PIC and shifting. Component RMS summaries do
not retain the cross terms needed to reconstruct it. The earlier raw phase audit
measured that transition on its own trajectory and found much larger geometric
than stored speeds; those values cannot be substituted into this run. A later code
revision now reports promoted geometric mean/p95 speed and vector difference
directly from the owned terminal geometry plus `(promoted-rollout_end)/dt`. Those
new fields apply only to future runs; they were not retroactively calculated or
inserted into this JSON.

## Recommended next change

The immediate bounded candidate is a **full matched `commit_pic=False` run**, with
body force, stress, surface-u, layer relaxation, shifting and all other settings
unchanged. Its early eight-window screen reduced boundary normal return and
reversals, improved local density, and slightly worsened silhouette/target
coverage. Together with the stage accounting, that is enough to justify a full
follow-up, not enough to promote a default or claim final rest.

Compare full raw shape/coverage and tip retention, exact pin invariance and valid
deformation, then late motion on identical common-free material IDs at the same
accepted commits and fixed-progress crossings. The early 9314-ID cohort and late
1374-ID cohort must remain separate. Visual QA must cover every exported frame;
density/coverage alone cannot certify that no morphing hole appears.

If disabling PIC loses final shape or leaves substantial movement, the structural
fix should make the state optimized by the solver agree with the state it commits:

1. Factor the promoted endpoint map into one GPU implementation shared by data
   losses, line-search evaluation and runner commit. Differentiate through the
   fixed-within-window PIC stencil; preserve pin invariance. The current
   `grid_project` implementation is under `torch.no_grad()`, so inserting its
   detached result into an objective would not accomplish this.
2. Retain the physical terminal-velocity brake and expose actual geometric
   displacement, including the final commit transition, to the arrived-free rest
   objective/criterion. Endpoint-only RPROP misses substep out-and-back motion;
   stored `vT` omits position-only corrections. Keep the transit population's
   transport requirements separate when evaluating this rest change.
3. Validate the new gradient/state contract and repeat the same shape, supply,
   tip and raw-motion gates before promotion. This changes the optimization
   objective/gradient path and is not a guaranteed cure. A rendering interpolation
   or a held movie tail would not satisfy the physical-state requirement.

This recommendation follows the code distinction: `optimizer.losses_of` currently
uses rollout `xT`/`vT`; `runner` applies PIC and shifting afterward and recomputes
the fixed-target volumetric term. Outer rendering merit still uses its earlier
value unless the later opt-in `outer_render_committed` flag is enabled. Neither
the legacy partial recomputation nor a later committed-state scalar merit gives
the inner optimizer derivatives through those changes. A larger pin set or smaller
global lead is not justified by the component accounting alone.

## Evidence and review

Server artifacts: `/data/relcfd/chayo/physmorph_v2/work/p292/accounting60.json`
and `.log`; executed code snapshot: `work/p292/accounting`.
Local copies: `output/p292/accounting60.json`, `accounting60.log`; derived telemetry
summary: `output/p292/accounting_summary.json`. The input JSON SHA-256 is
`d51f21306bf945ed0f119c1d12556804ec1a734756b89074bf4ee6798f7af2ab`.
The summary reads already-collected scalar telemetry on the host; all original
trajectory/component calculations ran on CUDA. No new simulation or renderer was
used to prepare this report. Independent numeric/interpretation review is closed;
the reviewer recomputed the tables, closures, cohort scope and provenance from the
original JSON and found no remaining material blocker.
