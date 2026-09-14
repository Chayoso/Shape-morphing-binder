# Refutation gate: surface control and bunny development

The independent `refute_gradient_path` agent reviewed the protocol, response
solver changes, passive-surface path, input geometry, global checks and viewer.
These are code reviews and focused regression results, not final quality approval.

| Finding | Implementer response and evidence |
|---|---|
| Choosing held-out checkpoints or weakening the reference could manufacture a win | Fixed train/development/test camera splits, last accepted iterate, paired provenance and a 2x-iteration physics reference are required. Test evaluation remains disabled. |
| Subdividing only render triangles would not add physical surface degrees of freedom | Every refined vertex is an independent passive marker, advected every substep by the MPM grid; it has no direct control variable. |
| Last pre-advection density does not validate endpoint support | Rebuild final mass grid at final particle positions and sample it at final marker positions (`mpm/traj.py`). Endpoint regression compares independently recomputed support. |
| Restarted references silently relax cumulative bounds | Persist original marker positions and density; require them on restart. Regression rejects cumulative edge/support violations. |
| Inverting average rotations can be singular | Apply cumulative marker gradients to original tangents and reject degenerate expected normals (`pipeline/geometric.py`). |
| Local determinant and support checks do not prevent global intersections or exterior particles | Candidate/initial/final global intersection gate and all-state offline containment audit added. Online containment and completeness of the Open3D candidate set remain limitations. |
| Target random-sample COM differs from actual dense mass objective | One dense-COM translation is applied to mesh, quadrature and raw target sample. v9 dense-minus-source COM is approximately (9.6e-11, -7.6e-12, -2.1e-9), N=12000, dx=.25, dt=1/120. |
| Asset-space checks do not validate transformed float32 render geometry | Final source and target arrays are checked without repair before fixture save (`sampling/closed_mesh.py`). |
| Zero-control Open3D reports disjoint nearly coplanar triangles as intersections | Conservative float64 separating-axis certificates remove only provably disjoint candidates. Contact, uncertainty, degeneration and nonfinite inputs remain rejected; regression includes the measured pair and true coplanar/transverse intersections. |
| Global rejection was not regression-tested through actual optimizer acceptance | Injected candidate intersection rejection preserves zero coefficients and accepted state, with initial/final gate calls checked (`tests/test_response_control.py`). |
| Viewer stays waiting after an optimizer exit without callback | Final report/trajectory fallback and trajectory file monitoring added; regression covers that exit. |
| A separate viewer checkout can use a different rendering implementation | Check actual viewer render dependency files and loaded native binary against experiment provenance; regression rejects modified implementations. |

The final two viewer findings were independently re-reviewed and closed. The
reviewer found no additional required change in that scope. The source passed
190 CPU tests before the last two viewer regressions; all three focused viewer
tests then passed. Final compiled files and `git diff --check` passed. No CUDA
simulation ran locally.

No earlier v1-v5 failure is removed by these fixes. v6/v7 input problems and
v8 initial-gate rejection remain in their preserved server directories. v9 is
a new development comparison, with a different valid shared fixture. It cannot
retroactively establish the earlier render-guidance claims or the sealed 20% goal.

The subsequent surface-support response change was reviewed twice. The first
review pointed out that differentiating a time minimum can cancel opposing time
derivatives. The implementation now retains every time/marker as its own QP row.
An integrated CPU test injects support ratio .05 into a real rollout: diagnostic
probes build their model, but actual candidates are rejected and the prior
control remains unchanged. The focused 25 response/transport tests passed.
The reviewer found no additional required correction. The solver's existing
constraint-generation budget (16 rounds, up to 8 new rows each) can still reject
a problem before finding a feasible direction; this is a solver limit, not proof
of physical uncontrollability or absence of an improving image direction.
