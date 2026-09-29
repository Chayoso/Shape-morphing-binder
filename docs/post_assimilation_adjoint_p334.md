# P334: differentiate a controlled window through assimilation into a coast

`mpm/post_assimilation_adjoint.py` adds an opt-in `PostAssimilationAdjoint`.
It leaves the existing pre-assimilation bridge and production runner unchanged.
The controlled head carries its actual x/v/C/F/Fg to an independent coast,
updates plastic Fp with the P333 handoff, and zeroes v/C for the supplied next
pins. The coast withdraws stress, surface-u and body controls; an explicitly
supplied successor layer relaxation can remain active. This is not necessarily
an unforced physical rest experiment.

The caller supplies owned, fixed successor pins, layer, bonds and viscosity.
Next pins must include all binary old pins. Admission, release/follow/yield/KKT,
growth, consensus and the continuous derivative of successor preparation are
outside this API. Materials and initial state are fixed; only stress, surface-u
and body control derivatives are returned. Freezing actual prepared values
gives a conditional partial derivative, not the full ordinary runner derivative.

The coast has gradient-bearing independent Fp and independent step-zero state.
Its reverse pass produces x/v/C/F/Fg boundary covectors and an Fp covector. The
P333 assimilation pullback adds the latter to terminal head F. Pin projection
masks v/C only; a newly pinned Fp can still affect free neighbors. Direct head
loss seeds and all boundary contributions are combined BEFORE Warp's head
backward, whose seed assignment would otherwise overwrite a contribution.
Control and output ownership, generation guards and worker stream restoration
reuse the existing bridge. All gradient buffers, including coast Fp, are reset.

Captured execution orders four Warp graphs and two Torch assimilation graphs on
one aligned stream. The Torch forward retains the spectral backward context;
replay must update its saved buffers and use the current coast Fp covector.
No exact-boundary or higher-derivative claim is made.

## Preregistered verification

The joint fixture is N27, T20 per window, dt .002, dx .5, grid16^3, with two old
pins, one new pin, layer relaxation, bonds and spatial viscosity. No-slip pin
handling avoids adding collider branch crossings to this first derivative gate.
These are CPU/CUDA operator tests, not the N300k native production recipe.

- Compare the ordinary production assimilation (including its actual new-pin
  subset call) and a freshly constructed independent coast, across all coast
  x/v/F/Fg steps. Check independent storage and zeroed pinned v/C.
- Compare all three control directional derivatives with central differences
  at 1e-3 and 5e-4. Keep the existing 2 percent relative / 5e-6 absolute tolerance
  and require a nontrivial derivative. Verify captured/plain agreement, changed
  inputs, head/coast seed additivity, repeated/missing/zero seeds and ownership.
- Omit Fp or C ONLY in backward. A full derivative must pass and at least one
  omitted direction must fail the same independent finite-difference tolerance.
  The mixed coast scalar was too insensitive to Fp for this negative control;
  use weighted first coast-step acceleration for Fp, mixed coast loss for C.
  This changes the observation, not the physical forward or tolerance.
- Require the CUDA graph path itself, aligned alternate streams and explicit
  errors for unaligned/wrong streams, stale outputs and failed later calls.
  Forbid numerical array downloads during the apply/backward test scopes.

CPU: root's 10 new plus 32 existing bridge/FD cases pass (42 total, 7.52s).
Independent review passes 36 distinct formal cases, including both omissions,
and supplemental T1 and interleaved-instance checks. The Fp omission is detected
by surface-u; the C omission by body control. Actual CUDA results follow below.

Rendering influence is unchanged: these tests contain no render objective or
physical optimization. They cannot establish natural rest, hole removal or 4K
appearance. A better physical F need not improve visual covariance; the separate
active-renderer risk and rejected thickness-only trial remain documented in
`gaussian_footprint_followup.md` and `footprint_thickness_p323.md`.

## Actual CUDA verification

Frozen c4a7c82 runs as `work/p303/p334_cuda1` on hyde06, source directory
`code_post_assimilation1`, process770611. All9 cases pass, no skips, in5.754s,
exit0. The discretization is the N27/T20/dt.002/dx.5/grid16^3 fixture above.
Both ordinary and captured executions agree, all three mixed-observation control
directions pass both registered radii, and the independent production-subset
assimilation/fresh-coast forward agrees. No threshold was relaxed.

The captured negative controls use the SAME forward outputs, forbid eager
trajectory/boundary fallback, and intercept only the live boundary covector.
At epsilon5e-4, the Fp/surface-u witness has full AD -.0183089683 versus FD
-.0182145925; omitting Fp gives -.0104908700 (allowance .0003642919).
The C/body witness has full AD -.00711814382 versus FD -.00713741616;
omitting C gives -.00612606205 (allowance .00014274832). Thus these observations
actually detect the missing paths instead of merely reporting nonzero gradients.

Changed controls, alternate aligned streams, owned policies/outputs, stale and
failed-call guards, repeated/zero/missing seeds all pass. The one warning is a
test-only scalar conversion after replay, outside the captured numerical path.
Sampled process memory peaks at1300MiB; this is neither an exact high-water mark
nor an N300k memory estimate. Project usage is77414440492bytes, below100GB.
No cleanup or retained before/corrected change was needed.

`evidence/p334/source_receipt.json` binds Python source, wrapper and test artifacts.
The independent bounded audit verifies all92 source bindings (19 Git comparisons
differ only by CRLF), five result artifacts, nine XML cases and all12 logged
positive/negative finite-difference rows. It does not independently rerun CUDA
or hash installed shared libraries; see `evidence/p334/independent_audit.md`.
This closes the tiny fixed-policy joint derivative gate, not full successor
preparation, a production adapter or any physical/visual quality gate. Next,
bind this API to the actual prepared handoff and verify large-run forward/merit
parity before proposing another control candidate.
